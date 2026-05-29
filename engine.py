"""
Train / evaluation engine used by main.py.
  - get_linear_schedule_with_warmup
  - calculate_task_embeddings
  - cluster_prompts  (HAC + silhouette)
  - evaluate
  - train_and_evaluate
"""

import os
import sys
import json
from collections import deque
from itertools import cycle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

from models import SoftPromptModel
from utils.config import SoftPromptConfig
from utils.metrics import compute_metrics


# ------------------------------------------------------------------
# Scheduler
# ------------------------------------------------------------------

def get_linear_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1,
):
    """Linear warmup then constant LR schedule."""
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# ------------------------------------------------------------------
# Task Embedding
# ------------------------------------------------------------------

def calculate_task_embeddings(
    model: SoftPromptModel,
    dataloader: DataLoader,
    num_tasks: int,
    device: str,
):
    """
    Compute mean sentence embeddings per task.
    """
    task_embed_sums = torch.zeros(num_tasks, model.hidden_dim, device=device)
    task_counts     = torch.zeros(num_tasks, device=device)

    embed_layer = model.base_model.get_input_embeddings()
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Scanning Dataset"):
            input_ids      = batch['routing_input_ids'].to(device)
            attention_mask = batch['routing_attention_mask'].to(device)
            task_ids       = batch['task_ids'].to(device)

            embeds = embed_layer(input_ids)
            mask   = attention_mask.unsqueeze(-1)

            sentence_embeds = (
                (embeds * mask).sum(dim=1)
                / mask.sum(dim=1).clamp(min=1e-9)
            )

            for i in range(len(task_ids)):
                t_id = task_ids[i].item()
                task_embed_sums[t_id] += sentence_embeds[i]
                task_counts[t_id] += 1

    task_means = task_embed_sums / task_counts.clamp(min=1.0).unsqueeze(1)

    print("[Initialization] Task Embeddings Computed")

    return task_means


# ------------------------------------------------------------------
# Clustering
# ------------------------------------------------------------------

def cluster_prompts(
    prompt_mean: torch.Tensor,
    num_tasks: int,
):
    """
    Cluster task prompts via HAC + silhouette selection.
    """
    with torch.no_grad():
        prompt = prompt_mean.detach().cpu().numpy()

    prompts_norm = normalize(prompt, axis=1)

    Z = linkage(
        prompts_norm,
        method="average",
        metric="cosine",
    )

    best_k     = 2
    best_score = -1.0

    for k in range(2, num_tasks + 1):
        try:
            labels = fcluster(Z, t=k, criterion="maxclust")

            n_clusters = len(set(labels))

            if 2 <= n_clusters < num_tasks:
                score = silhouette_score(
                    prompts_norm,
                    labels,
                    metric="cosine",
                )

                if score > best_score:
                    best_score = score
                    best_k = k

        except Exception:
            continue

    labels = fcluster(Z, t=best_k, criterion="maxclust")

    return labels, best_k, best_score, Z


def compute_mean_ari(curr_labels, history_labels):
    """Compute mean ARI against historical cluster labels."""
    if len(history_labels) == 0:
        return None, []

    aris = [
        adjusted_rand_score(curr_labels, prev)
        for prev in history_labels
    ]

    return float(np.mean(aris)), aris


def save_dendrogram(
    Z,
    num_tasks: int,
    save_path: str,
):
    """Save HAC dendrogram."""
    plt.figure(figsize=(10, 5))

    dendrogram(
        Z,
        labels=[str(t) for t in range(num_tasks)],
    )

    plt.title("Task Hierarchical Clustering")
    plt.xlabel("Task ID")
    plt.ylabel("Cosine Distance")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: SoftPromptModel,
    dataloader: DataLoader,
    device: torch.device,
    config: SoftPromptConfig,
    dataset_instance,
):
    """
    Run evaluation and compute metrics.
    """
    model.eval()

    all_ids   = []
    all_preds = []
    all_gts   = []

    for batch in tqdm(
        dataloader,
        desc="Evaluating",
        leave=False,
    ):
        ids            = batch['ids']
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        task_ids       = batch['task_ids'].to(device)

        generated = model.generate(
            ids=ids,
            input_ids=input_ids,
            attention_mask=attention_mask,
            task_ids=task_ids,
            max_new_tokens=config.max_new_tokens,
            num_beams=config.num_beams,
            do_sample=config.do_sample,
        )

        preds = model.tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )

        all_ids.extend(ids)
        all_preds.extend(preds)
        all_gts.extend(batch['answers'])

    return compute_metrics(
        all_ids,
        all_preds,
        all_gts,
        dataset_instance,
    )


# ------------------------------------------------------------------
# Training Loop
# ------------------------------------------------------------------

def train_and_evaluate(
    model: SoftPromptModel,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    config: SoftPromptConfig,
    dataset_instance,
    output_dir: str,
):
    """
    Main training loop with automatic Phase 2 transition.
    """
    model.train()

    base_step = 0
    total_loss = 0.0
    best_score = 0.0

    phase_transition_done = False

    # ---- ARI stability tracking ----
    label_history = deque(
        maxlen=config.ari_history_size
    )

    stable_count = 0

    train_iterator = cycle(train_dataloader)

    num_tasks = len(dataset_instance.in_domain)

    task2group = {
        i: i for i in range(num_tasks)
    }

    pbar = tqdm(
        range(config.group_train_steps),
        desc="Training",
        dynamic_ncols=True,
        file=sys.stdout,
    )

    while base_step < config.group_train_steps:

        optimizer.zero_grad(set_to_none=True)

        # --------------------------------------------------------------
        # Gradient Accumulation
        # --------------------------------------------------------------
        for _ in range(config.gradient_accumulation_steps):

            try:
                batch = next(train_iterator)

            except StopIteration:
                train_iterator = cycle(train_dataloader)
                batch = next(train_iterator)

            batch = {
                k: v.to(device)
                if isinstance(v, torch.Tensor)
                else v
                for k, v in batch.items()
            }

            outputs, prompt_options = model(
                ids=batch['ids'],
                routing_input_ids=batch['routing_input_ids'],
                routing_attention_mask=batch['routing_attention_mask'],
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                task_ids=batch['task_ids'],
            )

            loss = outputs['loss']
            key_loss = prompt_options['key_loss']

            final_loss = (
                loss + config.key_coeff * key_loss
            ) / config.gradient_accumulation_steps

            final_loss.backward()

            total_loss += (
                final_loss.item()
                * config.gradient_accumulation_steps
            )

        torch.nn.utils.clip_grad_norm_(
            model.soft_prompt.parameters(),
            1.0,
        )

        optimizer.step()
        scheduler.step()

        base_step += 1

        pbar.update(1)

        pbar.set_postfix({
            "loss": (
                f"{total_loss / (base_step * config.gradient_accumulation_steps):.4f}"
            )
        })

        # --------------------------------------------------------------
        # Phase 2 Trigger
        # --------------------------------------------------------------
        if (
            not phase_transition_done
            and base_step % config.check_interval == 0
        ):
            with torch.no_grad():
                prompt_mean = (
                    model.soft_prompt.task_prompts.mean(dim=1)
                )

            curr_labels, best_k, best_silhouette, Z = cluster_prompts(
                prompt_mean,
                num_tasks,
            )

            history_ready = (
                len(label_history)
                == config.ari_history_size
            )

            mean_ari, _ = (
                compute_mean_ari(
                    curr_labels,
                    list(label_history),
                )
                if history_ready else (None, [])
            )

            label_history.append(curr_labels.copy())

            if base_step >= config.group_warmup_steps:

                if (
                    mean_ari is not None
                    and mean_ari >= config.ari_threshold
                ):
                    stable_count += 1

                else:
                    stable_count = 0

            else:
                stable_count = 0

            # ----------------------------------------------------------
            # Transition to Phase 2
            # ----------------------------------------------------------
            if (
                base_step >= config.group_warmup_steps
                and history_ready
                and stable_count >= config.stable_patience
            ):
                phase_transition_done = True

                print(
                    f"\n[Phase 2 Triggered] "
                    f"Step={base_step} | "
                    f"K={best_k} | "
                    f"ARI={mean_ari:.4f}"
                )

                # Finalize task-group mapping
                task2group = {
                    t: int(l - 1)
                    for t, l in enumerate(curr_labels)
                }

                save_dendrogram(
                    Z,
                    num_tasks,
                    os.path.join(
                        output_dir,
                        'prompt_dendrogram.png',
                    )
                )

                with open(
                    os.path.join(output_dir, 'task2group.json'),
                    'w'
                ) as f:
                    json.dump(task2group, f, indent=2)

                # Update model structure
                model.soft_prompt.update_phase2_structure(
                    task2group
                )

                optimizer.add_param_group({
                    'params': [
                        model.soft_prompt.group_prompts,
                        model.soft_prompt.task_keys,
                    ],
                    'lr': config.learning_rate,
                    'weight_decay': config.weight_decay,
                })

        # --------------------------------------------------------------
        # Evaluation
        # --------------------------------------------------------------
        if (
            (
                base_step % config.eval_steps == 0
                or base_step == config.group_train_steps
            )
            and phase_transition_done
        ):
            metrics = evaluate(
                model,
                val_dataloader,
                device,
                config,
                dataset_instance,
            )

            overall = metrics['overall']
            monitor_key = metrics['monitor_metric']

            task_metrics = metrics['task_metrics']
            task_outputs = metrics['task_outputs']

            metric_str = ", ".join([
                f"{k.upper()}: {overall[k]:.2f}"
                for k in overall
                if k != 'count'
            ])

            print(f"\n[Step {base_step}] {metric_str}")

            current_val_score = overall[monitor_key]

            if current_val_score > best_score:

                best_score = current_val_score

                torch.save({
                    'base_step': base_step,
                    'soft_prompt_state_dict': model.soft_prompt.state_dict(),
                    'task_embeddings': model.soft_prompt.task_embeddings.cpu(),
                    'config': vars(config),
                    'task_to_group': task2group,
                }, os.path.join(output_dir, 'best_model.pt'))

                print(
                    f"[Best Model Saved] "
                    f"{monitor_key.upper()}: {best_score:.2f}"
                )

                prefix = config.dataset_name.lower()

                with open(
                    os.path.join(
                        output_dir,
                        f"{prefix}_in_domain_task_outputs.json",
                    ),
                    "w"
                ) as f:
                    json.dump(
                        task_outputs,
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )

                with open(
                    os.path.join(
                        output_dir,
                        f"{prefix}_in_domain_metrics.json",
                    ),
                    "w"
                ) as f:
                    json.dump(
                        task_metrics,
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )

            model.train()

    pbar.close()
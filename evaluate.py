"""
Standalone evaluation script.
Loads a saved checkpoint and runs out-of-domain (test) evaluation.
"""

import os
import json
import argparse
from pathlib import Path
from functools import partial

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import SoftPromptModel
from dataset import DATASET_REGISTRY, collate_fn
from utils.metrics import compute_metrics


def load_model(checkpoint_path: str, device: str):
    """
    Restore model and group structure from a saved checkpoint.

    1. Load config and task_to_group from checkpoint
    2. Initialize SoftPromptModel
    3. Reconstruct Phase 2 group structure
    4. Load saved soft prompt weights
    """
    print(f"Loading checkpoint from {checkpoint_path}...")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    config_data   = checkpoint['config']
    task_to_group = checkpoint['task_to_group']

    # 1. Model initialization
    model = SoftPromptModel(
        model_name=config_data['model_name'],
        soft_prompt_length=config_data['soft_prompt_length'],
        low_rank_dim=config_data['low_rank_dim'],
        device=device,
    )

    # 2. Soft prompt initialization
    num_train_tasks = len(task_to_group)

    task_embeddings = checkpoint['task_embeddings'].to(device)

    soft_prompt_template = config_data['soft_prompt_template']

    model.init_soft_prompt(
        num_tasks=num_train_tasks,
        soft_prompt_template=soft_prompt_template,
        task_embeddings=task_embeddings,
    )

    # 3. Reconstruct Phase 2 group structure
    print("Reconstructing group structure...")

    task_to_group = {
        int(k): int(v)
        for k, v in task_to_group.items()
    }

    model.soft_prompt.update_phase2_structure(
        task_to_group
    )

    # 4. Load trained soft prompt
    model.soft_prompt.load_state_dict(
        checkpoint['soft_prompt_state_dict']
    )

    model.to(device)
    model.eval()

    print("Model loaded successfully.")
    return model, config_data


@torch.no_grad()
def evaluate_dataset(
    model: SoftPromptModel,
    dataloader: DataLoader,
    device: torch.device,
    config: dict,
    dataset_instance,
):
    """
    Run generation on the given dataloader and compute task-level metrics.

    Returns (metrics_dict, all_ids, all_predictions).
    """
    model.eval()
    all_ids, all_predictions, all_references = [], [], []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        task_ids       = batch['task_ids'].to(device)
        ids            = batch['ids']
        references     = batch['answers']

        outputs = model.generate(
            ids=ids,
            input_ids=input_ids,
            attention_mask=attention_mask,
            task_ids=task_ids,
            max_new_tokens=config.get('max_new_tokens', 100),
            num_beams=config.get('num_beams', 1),
            do_sample=config.get('do_sample', False),
        )

        predictions = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        all_ids.extend(ids)
        all_predictions.extend(predictions)
        all_references.extend(references)

    metrics = compute_metrics(all_ids, all_predictions, all_references, dataset_instance)
    return metrics, all_ids, all_predictions


def main(args):
    device = args.device if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    checkpoint_path = os.path.join(args.output_dir, 'best_model.pt')
    if not os.path.exists(checkpoint_path):
        print(f"Error: No checkpoint found at {checkpoint_path}")
        return

    model, config_dict = load_model(checkpoint_path, device)
    config_dict['dataset_name'] = args.dataset_name

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset_name = config_dict.get('dataset_name', 'MRQA')
    dataset_cls  = DATASET_REGISTRY.get(dataset_name)
    if not dataset_cls:
        raise ValueError(f"Unknown dataset name in config: {dataset_name}")

    cache_dir        = Path(config_dict.get('data_dir', './data')) / f"{dataset_name.lower()}_dataset"
    dataset_instance = dataset_cls(cache_dir=cache_dir, seed=config_dict.get('seed', 42))

    # ------------------------------------------------------------------
    # Test dataloader (out-of-domain)
    # ------------------------------------------------------------------
    print(f"Loading out-of-domain test set for {dataset_name}...")
    test_dataset = dataset_instance.load_dataset(split='test')

    test_loader = DataLoader(
        test_dataset,
        batch_size=config_dict.get('per_device_batch_size', 8) * 4,
        shuffle=False,
        collate_fn=partial(
            collate_fn,
            split='test',
            tokenizer=model.tokenizer,
            max_length=dataset_instance.max_seq_length,
        )
    )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    out_domain_result, _, _ = evaluate_dataset(model, test_loader, device, config_dict, dataset_instance)

    overall      = out_domain_result['overall']
    task_metrics = out_domain_result['task_metrics']
    task_outputs = out_domain_result['task_outputs']

    # Print results
    print("\n" + "=" * 50)
    print(f"   Out-of-Domain Results ({dataset_name})")
    print("=" * 50)

    overall_str = " | ".join([f"{k.upper()}: {overall[k]:.2f}" for k in overall if k != 'count'])
    print(f"[OVERALL] {overall_str} | Count: {overall['count']}")
    print("-" * 50)

    for task_name, m in task_metrics.items():
        task_str = " | ".join([f"{k.upper()}: {m[k]:.2f}" for k in m if k != 'count'])
        print(f"Task: {task_name:<25} | {task_str} | Count: {m['count']}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    prefix = dataset_name.lower()

    output_file_outputs = os.path.join(args.output_dir, f"{prefix}_out_domain_task_outputs.json")
    output_file_metrics = os.path.join(args.output_dir, f"{prefix}_out_domain_metrics.json")

    with open(output_file_outputs, "w", encoding="utf-8") as f:
        json.dump(task_outputs, f, indent=2, ensure_ascii=False)

    with open(output_file_metrics, "w", encoding="utf-8") as f:
        json.dump(task_metrics, f, indent=2, ensure_ascii=False)

    print(f"\n[Done] Metrics saved to: {output_file_metrics}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('HiVe Evaluation')

    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs'
    )

    parser.add_argument(
        '--dataset_name',
        type=str,
        required=True
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda'
    )

    args = parser.parse_args()

    main(args)
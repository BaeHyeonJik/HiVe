import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase


class HiVePrompt(nn.Module):
    """
    Hierarchical Residual Soft Prompt (HiVePrompt).

    Phase 1:
      - Prompt = base_prompt + task_prompt[task_id]
      - Operates in a low-rank space (low_rank_dim) and projects to hidden_dim via projector

    Phase 2 (triggered externally via update_phase2_structure):
      - Prompt = base_prompt + group_prompt[group_id] + task_prompt[task_id]  (train)
      - Prompt = router-selected level from {task, group, base}               (inference)
      - Router is trained with cross-entropy on task and group keys
    """

    def __init__(
        self,
        base_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        soft_prompt_template: list[str],
        task_embeddings: torch.Tensor,
        num_tasks: int,
        soft_prompt_length: int = 40,
        low_rank_dim: int = 36,
        hidden_dim: int = 2048,
        device: str = "cuda",
    ):
        super().__init__()

        self.task_embeddings   = task_embeddings.to(device)
        self.num_tasks         = num_tasks
        self.soft_prompt_length = soft_prompt_length
        self.low_rank_dim      = low_rank_dim
        self.hidden_dim        = hidden_dim
        self.device            = device

        # ------------------------------------------------------------------
        # Projector: low_rank_dim -> hidden_dim
        # ------------------------------------------------------------------
        self.projector = nn.Linear(low_rank_dim, hidden_dim, bias=False).to(device)

        # ------------------------------------------------------------------
        # SVD-based initialization on template embeddings
        # ------------------------------------------------------------------
        with torch.no_grad():
            tokenized = tokenizer(
                soft_prompt_template,
                padding="max_length",
                max_length=soft_prompt_length,
                truncation=True,
                return_tensors="pt",
            ).to(device)

            input_embeds        = base_model.get_input_embeddings()(tokenized["input_ids"])
            mean_template_embeds = input_embeds.mean(dim=0)  # [P, H]

            U, S, Vh = torch.linalg.svd(mean_template_embeds, full_matrices=False)
            r        = min(low_rank_dim, mean_template_embeds.shape[0], mean_template_embeds.shape[1])
            S_sqrt   = torch.diag(torch.sqrt(S[:r]))

            # Projector weight init: V * S^(1/2)
            V_init = torch.matmul(S_sqrt, Vh[:r, :]).T
            if low_rank_dim > r:
                V_init = torch.cat(
                    [V_init, torch.zeros(hidden_dim, low_rank_dim - r, device=device)], dim=1
                )
            self.projector.weight.copy_(V_init)

            # base prompt init: U * S^(1/2)
            base_init = torch.matmul(U[:, :r], S_sqrt)
            if low_rank_dim > r:
                base_init = torch.cat(
                    [base_init, torch.zeros(soft_prompt_length, low_rank_dim - r, device=device)], dim=1
                )

            self.base_prompt = nn.Parameter(base_init.unsqueeze(0))

        # ------------------------------------------------------------------
        # Group structure (populated in Phase 2)
        # ------------------------------------------------------------------
        self.group_prompts = None
        self.num_groups    = 0
        self.register_buffer("group_ids",           torch.zeros(num_tasks, dtype=torch.long, device=device))
        self.register_buffer("group_singleton_mask", torch.zeros(0, dtype=torch.bool, device=device))

        # ------------------------------------------------------------------
        # Task prompts: initialized relative to base
        # ------------------------------------------------------------------
        with torch.no_grad():
            task_prompts_low  = torch.matmul(self.task_embeddings, self.projector.weight)  # [T, d]
            task_prompts_init = task_prompts_low.unsqueeze(1).expand(-1, soft_prompt_length, -1).clone()

            task_mean = task_prompts_init.mean(dim=0, keepdim=True)
            self.base_prompt.add_(task_mean)
            task_prompts_init = task_prompts_init - task_mean

            noise             = torch.randn_like(task_prompts_init) * 1e-4
            task_prompts_init = task_prompts_init + noise

        self.task_prompts = nn.Parameter(task_prompts_init)

        # Prompt output scaling
        self.prompt_scale = nn.Parameter(torch.tensor(1.0, device=device))

        # Task keys (learnable; only initialized in Phase 2)
        self.task_keys      = None
        self.phase2_enabled = False

    def update_phase2_structure(self, task_to_group: dict):
        """
        Transition to Phase 2 by initializing group prompts and task keys.

        1. Map task -> group via task_to_group
        2. Allocate group_prompts as mean of member task prompts
        3. Subtract group mean from task prompts (residual decomposition)
        4. Initialize task_keys from task embeddings projected into low-rank space
        """
        new_group_ids = torch.tensor(
            [task_to_group[i] for i in range(self.num_tasks)],
            dtype=torch.long,
            device=self.device,
        )
        self.group_ids.copy_(new_group_ids)

        num_groups      = len(set(task_to_group.values()))
        self.num_groups = num_groups

        # 1. Allocate group prompt space
        self.group_prompts = nn.Parameter(
            torch.zeros(num_groups, self.soft_prompt_length, self.low_rank_dim, device=self.device)
        )

        group_counts             = torch.zeros(num_groups, device=self.device)
        for g_id in task_to_group.values():
            group_counts[g_id] += 1
        self.group_singleton_mask = (group_counts == 1).to(torch.bool)

        with torch.no_grad():
            for g in range(num_groups):
                members = (self.group_ids == g).nonzero(as_tuple=True)[0]
                if not self.group_singleton_mask[g]:
                    group_mean = self.task_prompts[members].mean(dim=0)
                    self.group_prompts[g].copy_(group_mean)
                    self.task_prompts[members] -= group_mean

            # 2. Initialize task keys from projected task embeddings
            W_norm                = F.normalize(self.projector.weight, p=2, dim=0)  # [H, d]
            task_embedding_norm   = F.normalize(self.task_embeddings, p=2, dim=-1)  # [T, H]
            task_key_init         = torch.matmul(task_embedding_norm, W_norm)       # [T, d]

        self.task_keys      = nn.Parameter(task_key_init.clone())
        self.phase2_enabled = True

    def _build_group_and_base_keys(self):
        """Derive group and base keys by averaging normalized task keys."""
        task_keys_unit = F.normalize(self.task_keys, p=2, dim=-1)

        if self.num_groups <= 0:
            group_keys = torch.zeros(0, self.low_rank_dim, device=self.device, dtype=self.task_keys.dtype)
        else:
            group_keys = torch.stack(
                [
                    task_keys_unit[(self.group_ids == g).nonzero(as_tuple=True)[0]].mean(dim=0)
                    for g in range(self.num_groups)
                ],
                dim=0,
            )

        base_key = task_keys_unit.mean(dim=0, keepdim=True)
        return group_keys, base_key

    def forward(
        self,
        ids: list[str],
        input_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        task_ids: torch.Tensor = None,
        debug: bool = False,
    ):
        """
        Compute the soft prompt to prepend for each sample in the batch.

        Phase 1:
          combined = base + task[task_id]

        Phase 2 (train):
          combined = base + group[group_id] + task[task_id]

        Phase 2 (inference):
          Router selects the best level (task / group / base) per sample
          via cosine similarity between the query and stored keys.
        """
        is_training_mode = self.training and (task_ids is not None)
        B = input_embeds.shape[0]
        T = self.num_tasks
        G = self.num_groups

        key_loss = torch.tensor(0.0, device=self.device)

        task_logits   = None
        group_logits  = None
        base_logits = None
        router_logits = None
        top1_logit    = None
        router_choice = None

        # ------------------------------------------------------------------
        # Phase 2: Router
        # ------------------------------------------------------------------
        if self.phase2_enabled:
            W_norm = F.normalize(self.projector.weight, p=2, dim=0)

            mask_expanded = attention_mask.unsqueeze(-1).float()
            sum_embeds    = (input_embeds * mask_expanded).sum(dim=1)
            sum_mask      = mask_expanded.sum(dim=1).clamp(min=1e-9)
            query         = sum_embeds / sum_mask

            query_norm = F.normalize(query, p=2, dim=-1)
            query_low  = torch.matmul(query_norm, W_norm)
            query_low  = F.normalize(query_low, p=2, dim=-1)

            task_key_norm  = F.normalize(self.task_keys, p=2, dim=-1)
            group_keys, base_key = self._build_group_and_base_keys()
            group_key_norm  = F.normalize(group_keys, p=2, dim=-1) if G > 0 else group_keys
            base_key_norm = F.normalize(base_key, p=2, dim=-1)

            task_logits   = torch.matmul(query_low, task_key_norm.t())   # [B, T]
            base_logits = torch.matmul(query_low, base_key_norm.t()) # [B, 1]

            if G > 0:
                group_logits = torch.matmul(query_low, group_key_norm.t())  # [B, G]
                singleton_mask = self.group_singleton_mask.unsqueeze(0).expand(B, -1)
                group_logits   = group_logits.masked_fill(singleton_mask, -1e9)
            else:
                group_logits = torch.zeros(B, 0, device=self.device)

            router_logits          = torch.cat([task_logits, group_logits, base_logits], dim=1)
            top1_logit, router_choice = torch.max(router_logits, dim=1)

            # Train task keys with CE loss on task and group levels
            if is_training_mode:
                task_loss     = F.cross_entropy(task_logits, task_ids)
                target_groups = self.group_ids[task_ids]
                valid_mask    = ~self.group_singleton_mask[target_groups]

                if valid_mask.any():
                    group_loss = F.cross_entropy(group_logits[valid_mask], target_groups[valid_mask])
                    key_loss   = task_loss + 1.0 * group_loss
                else:
                    key_loss = task_loss

        # ------------------------------------------------------------------
        # Prompt Construction
        # ------------------------------------------------------------------
        if not self.phase2_enabled:
            if task_ids is None:
                raise ValueError("Phase 1 requires task_ids.")
            combined  = self.base_prompt.expand(B, -1, -1).clone()
            combined  = combined + self.task_prompts[task_ids]

        elif is_training_mode:
            combined  = self.base_prompt.expand(B, -1, -1).clone()
            combined  = combined + self.task_prompts[task_ids]

            p_group          = self.group_prompts[self.group_ids[task_ids]]
            is_singleton     = self.group_singleton_mask[self.group_ids[task_ids]]
            is_singleton     = is_singleton.view(-1, 1, 1).expand_as(p_group)
            p_group          = torch.where(is_singleton, torch.zeros_like(p_group), p_group)
            combined         = combined + p_group

        else:
            # Inference: route each sample independently
            combined = self.base_prompt.expand(B, -1, -1).clone()

            for b in range(B):
                idx = router_choice[b].item()

                if idx < T:
                    t = idx
                    g = self.group_ids[t].item()
                    combined[b] = combined[b] + self.task_prompts[t]
                    if not self.group_singleton_mask[g]:
                        combined[b] = combined[b] + self.group_prompts[g]

                elif idx < T + G:
                    g = idx - T
                    combined[b] = combined[b] + self.group_prompts[g]

                else:
                    pass  # base only

        batched_prompt = (combined @ self.projector.weight.T) * self.prompt_scale

        # ------------------------------------------------------------------
        # Debug logging
        # ------------------------------------------------------------------
        if True:
            print(f"\n{'=' * 20} DEBUG {'=' * 20}")
            print(f"Phase2 Enabled: {self.phase2_enabled}")

            with torch.no_grad():
                base_norm = self.base_prompt.norm(dim=(1, 2)).mean().item()
                task_norm   = self.task_prompts.norm(dim=(1, 2)).mean().item()
                group_norm  = (
                    self.group_prompts.norm(dim=(1, 2)).mean().item()
                    if self.group_prompts is not None and self.group_prompts.numel() > 0
                    else 0.0
                )
                print(
                    f"Prompt Norm -> base: {base_norm:.6f}, "
                    f"group: {group_norm:.6f}, "
                    f"task: {task_norm:.6f}"
                )

                if self.phase2_enabled:
                    task_key_norms             = self.task_keys.norm(dim=-1)
                    group_keys_dbg, base_key_dbg = self._build_group_and_base_keys()
                    group_key_norms            = (
                        group_keys_dbg.norm(dim=-1)
                        if group_keys_dbg.numel() > 0
                        else torch.zeros(0, device=self.device)
                    )
                    print(
                        f"Key Norm -> "
                        f"task mean: {task_key_norms.mean().item():.3f}, "
                        f"task std: {task_key_norms.std().item():.3f}, "
                        f"group mean: {group_key_norms.mean().item() if group_key_norms.numel() > 0 else 0.0:.3f}, "
                        f"group std: {group_key_norms.std().item() if group_key_norms.numel() > 1 else 0.0:.3f}, "
                        f"base: {base_key_dbg.norm(dim=-1).mean().item():.3f}"
                    )

            if is_training_mode:
                print(f"Batch Loss -> Key: {key_loss.item():.4f}")

            for b in range(B):
                print(f"\n[Sample {b}] ID: {ids[b]}")
                if task_ids is not None:
                    print(f"Target Task : T{task_ids[b].item()}")

                if not self.phase2_enabled:
                    t = task_ids[b].item()
                    print("Prompt Route: task")
                    print(f"Prompt Used : T{t} + GL")

                else:
                    pred_idx = router_choice[b].item()

                    if is_training_mode:
                        t = task_ids[b].item()
                        g = self.group_ids[t].item()
                        prompt_str = f"T{t} + GL" if self.group_singleton_mask[g] else f"T{t} + G{g} + GL"
                        print("Prompt Route: task")
                        print(f"Prompt Used : {prompt_str}")
                    else:
                        if pred_idx < T:
                            t = pred_idx
                            g = self.group_ids[t].item()
                            prompt_str = f"T{t} + GL" if self.group_singleton_mask[g] else f"T{t} + G{g} + GL"
                            print("Prompt Route: task")
                        elif pred_idx < T + G:
                            g = pred_idx - T
                            prompt_str = f"G{g} + GL"
                            print("Prompt Route: group")
                        else:
                            prompt_str = "GL"
                            print("Prompt Route: base")
                        print(f"Prompt Used : {prompt_str}")

                    task_str   = " | ".join([f"T{t}:{task_logits[b, t]:.3f}" for t in range(T)])
                    group_str  = " | ".join([f"G{g}:{group_logits[b, g]:.3f}" for g in range(G)])
                    base_str = f"GL:{base_logits[b, 0]:.3f}"
                    print(f"Task Logits : {task_str}")
                    print(f"Group Logits: {group_str}")
                    print(f"base Logit: {base_str}")
                    print(f"Router Top1 : Logit={top1_logit[b].item():.3f}")

                print(f"{'-' * 70}")
            print(f"{'=' * 70}")

        return {
            "batched_prompt": batched_prompt,
            "key_loss": key_loss,
        }

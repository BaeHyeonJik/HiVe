import os
import sys
import json
import random
import argparse
from pathlib import Path
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from models import SoftPromptModel
from dataset import DATASET_REGISTRY, collate_fn
from utils.config import SoftPromptConfig
from engine import (
    get_linear_schedule_with_warmup,
    calculate_task_embeddings,
    train_and_evaluate,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_seed(seed: int):
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def main(args):
    config = SoftPromptConfig(
        output_dir=args.output_dir,
        dataset_name=args.dataset_name
    )

    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.data_dir, exist_ok=True)

    # Save config for reproducibility
    with open(os.path.join(config.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(config), f, indent=2)

    set_seed(config.seed)

    device = args.device if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Model Initialization
    # ------------------------------------------------------------------
    model = SoftPromptModel(
        model_name=config.model_name,
        soft_prompt_length=config.soft_prompt_length,
        low_rank_dim=config.low_rank_dim,
        device=device,
    )

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    dataset_cls = DATASET_REGISTRY.get(config.dataset_name)
    if not dataset_cls:
        raise ValueError(f"Unknown dataset name in config: {config.dataset_name}")

    cache_dir = Path(config.data_dir) / f"{config.dataset_name.lower()}_dataset"
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset_instance = dataset_cls(cache_dir=cache_dir, seed=config.seed)

    train_dataset = dataset_instance.load_dataset(split='train')
    val_dataset   = dataset_instance.load_dataset(split='validation')

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.per_device_batch_size,
        shuffle=True,
        collate_fn=partial(
            collate_fn,
            split='train',
            tokenizer=model.tokenizer,
            max_length=dataset_instance.max_seq_length
        )
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.per_device_batch_size * 4,
        shuffle=False,
        collate_fn=partial(
            collate_fn,
            split='validation',
            tokenizer=model.tokenizer,
            max_length=dataset_instance.max_seq_length
        )
    )

    # ------------------------------------------------------------------
    # Task Embeddings (cached)
    # ------------------------------------------------------------------
    num_tasks        = len(dataset_instance.in_domain)
    embed_cache_path = cache_dir / f"{config.dataset_name.lower()}_task_embeddings.pt"

    if os.path.exists(embed_cache_path):
        print(f"\n[Initialization] Loading cached task embeddings: {embed_cache_path}")
        task_embeddings = torch.load(embed_cache_path, map_location=device)
    else:
        print(f"\n[Initialization] Computing task embeddings for {config.dataset_name}...")
        task_embeddings = calculate_task_embeddings(
            model,
            train_dataloader,
            num_tasks=num_tasks,
            device=device
        )
        torch.save(task_embeddings, embed_cache_path)
        print(f"[Initialization] Task embeddings saved: {embed_cache_path}")

    set_seed(config.seed)

    # ------------------------------------------------------------------
    # Soft Prompt Initialization
    # ------------------------------------------------------------------
    model.init_soft_prompt(
        num_tasks=num_tasks,
        soft_prompt_template=config.soft_prompt_template,
        task_embeddings=task_embeddings
    )

    trainable_params = sum(p.numel() for p in model.soft_prompt.parameters() if p.requires_grad)
    print(f"Trainable Parameters (Initial): {trainable_params}")

    # ------------------------------------------------------------------
    # Optimizer & Scheduler
    # ------------------------------------------------------------------
    optimizer = AdamW(
        model.soft_prompt.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.group_warmup_steps
    )

    set_seed(config.seed)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    print(f"\n[Train] Starting training on {config.dataset_name} "
          f"(Max Seq Len: {dataset_instance.max_seq_length})...")

    train_and_evaluate(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        dataset_instance=dataset_instance,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser('HiVe Training')

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

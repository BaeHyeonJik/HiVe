import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class SoftPromptConfig:
    """Full configuration for HiVe training and evaluation."""

    # ---- Model & Directories ----
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    data_dir:   str = "./data"
    output_dir: str = "./outputs"

    # ---- Dataset Selection ----
    dataset_name: str = "MRQA"

    # ---- Reproducibility ----
    seed: int = 42

    # ---- Prompt Architecture ----
    soft_prompt_length: int = 40
    low_rank_dim:       int = 36

    soft_prompt_template: List[str] = field(default_factory=lambda: [
        "Process input and infer task.",
        "Select relevant information.",
        "Model relationships in the input.",
        "Maintain consistency with the input.",
        "Generate an appropriate response.",
    ])

    # ---- Training Hyperparameters ----
    learning_rate:                float = 3e-3
    weight_decay:                 float = 1e-4
    group_train_steps:            int   = 1000
    group_warmup_steps:           int   = 100
    gradient_accumulation_steps:  int   = 16
    per_device_batch_size:        int   = 16
    eval_steps:                   int   = 100
    key_coeff:                    float = 0.1

    # ---- Phase Transition (ARI Stability) ----
    check_interval:   int   = 10
    ari_history_size: int   = 5
    ari_threshold:    float = 0.90
    stable_patience:  int   = 2

    # ---- Generation ----
    max_new_tokens: int = None
    num_beams:      int = 1
    do_sample:      bool = False

    def __post_init__(self):

        if self.dataset_name == "MRQA":
            self.max_new_tokens = 100

        elif self.dataset_name == "summary":
            self.max_new_tokens = 128

        else:
            raise ValueError(
                f"Unsupported dataset: {self.dataset_name}"
            )
<<<<<<< HEAD
Anonymous repository for the submitted paper.
=======
# HiVe

Official implementation of **HiVe: Hierarchical Soft Prompt Tuning for Multi-Task and Out-of-Domain Generalization**.

---

## Requirements

```bash
pip install -r requirements.txt
```

**Environment**
- Python 3.10.18
- PyTorch 2.8+
- CUDA 12.6+

---

## Reproduction

### MRQA

```bash
bash scripts/run_mrqa.sh
```

### summary

```bash
bash scripts/run_summary.sh
```

Each script runs training followed by out-of-domain evaluation and saves all results under the output directory.

---

## Expected Results

| Dataset     | Metric  | In-Domain | Out-of-Domain |
|-------------|---------|-----------|---------------|
| MRQA        | F1      | -         | -             |
| summary | ROUGE-L | -         | -             |

---

## Project Structure

```
HiVe/
├── train.py              # Training entry point (task embedding, soft prompt init, optimizer setup)
├── evaluate.py           # Out-of-domain evaluation (checkpoint loading, test set inference)
├── engine.py             # Train loop (Phase 1/2 transition, HAC clustering, ARI stability check)
├── dataset.py            # MRQADataset, SummarizationDataset, collate_fn
├── scripts/
│   ├── run_mrqa.sh       # MRQA reproduction script (train + evaluate)
│   └── run_summary.sh    # Summary reproduction script (train + evaluate)
├── models/
│   ├── __init__.py
│   ├── model.py          # SoftPromptModel (frozen LLM wrapper, prompt prepending)
│   └── prompt.py         # HiVePrompt (hierarchical residual soft prompt, router)
└── utils/
    ├── __init__.py
    ├── config.py         # SoftPromptConfig (training/evaluation hyperparameters)
    └── metrics.py        # F1, EM (MRQA), ROUGE-L (Summary)
```

---
>>>>>>> 47005c5 (initial commit)

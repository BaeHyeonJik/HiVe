import re
import string
from collections import Counter, defaultdict

from rouge_score import rouge_scorer


# ------------------------------------------------------------------
# Normalization
# ------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Lower, strip punctuation, articles, and extra whitespace."""
    def remove_articles(text):  return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):  return ' '.join(text.split())
    def remove_punc(text):      return ''.join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text):            return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


# ------------------------------------------------------------------
# QA Metrics
# ------------------------------------------------------------------

def f1_score(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between prediction and ground truth."""
    pred_tokens  = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common   = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall    = num_same / len(truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Exact match after normalization."""
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


# ------------------------------------------------------------------
# ROUGE-L
# ------------------------------------------------------------------

ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rougeL_score(prediction: str, ground_truth: str) -> float:
    """ROUGE-L F1 between prediction and ground truth."""
    return ROUGE_SCORER.score(ground_truth, prediction)["rougeL"].fmeasure


# ------------------------------------------------------------------
# Metric Map
# ------------------------------------------------------------------

METRIC_MAP = {
    "f1":     f1_score,
    "em":     exact_match_score,
    "rougeL": rougeL_score,
}


# ------------------------------------------------------------------
# Main Compute
# ------------------------------------------------------------------

def compute_metrics(ids, predictions, ground_truths, dataset_instance):
    """
    Compute task-level and overall metrics.

    1. Group predictions by task (inferred from sample id prefix)
    2. Compute each metric in dataset_instance.metrics per task
    3. Macro-average across tasks for overall score

    Returns a dict with keys:
      task_metrics, overall, task_outputs, monitor_metric
    """
    target_metrics = getattr(dataset_instance, "metrics",        ["em"])
    monitor_metric = getattr(dataset_instance, "monitor_metric", "em")

    task_preds  = defaultdict(list)
    task_truths = defaultdict(list)
    task_ids    = defaultdict(list)

    for id_, pred, truths in zip(ids, predictions, ground_truths):
        task_name = id_.split('_', 1)[0] if '_' in id_ else "unknown"
        task_preds[task_name].append(pred)
        task_truths[task_name].append(truths)
        task_ids[task_name].append(id_)

    task_metrics = {}
    task_outputs = {}
    total_count  = 0

    for task_name, preds in task_preds.items():
        truths = task_truths[task_name]
        count  = len(preds)
        total_count += count

        res = {"count": count}

        for m_name in target_metrics:
            metric_fn = METRIC_MAP[m_name]
            scores    = []

            for p, gt_list in zip(preds, truths):
                if not isinstance(gt_list, list):
                    gt_list = [gt_list]
                gt_list = [str(t) for t in gt_list if str(t).strip()]
                if not gt_list:
                    gt_list = [""]
                scores.append(max(metric_fn(p, t) for t in gt_list))

            res[m_name] = (sum(scores) / count) * 100 if count > 0 else 0.0

        task_metrics[task_name] = res
        task_outputs[task_name] = [
            {"id": tid, "prediction": p.strip(), "ground_truths": gt}
            for tid, p, gt in zip(task_ids[task_name], preds, truths)
        ]

    # Macro-average across tasks
    overall = {"count": total_count}
    for m_name in target_metrics:
        vals = [m[m_name] for m in task_metrics.values() if m_name in m]
        overall[m_name] = sum(vals) / len(vals) if vals else 0.0

    return {
        "task_metrics":   task_metrics,
        "overall":        overall,
        "task_outputs":   task_outputs,
        "monitor_metric": monitor_metric,
    }

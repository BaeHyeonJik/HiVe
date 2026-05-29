from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from datasets import Dataset as HFDataset, load_dataset, load_from_disk, concatenate_datasets
from tqdm import tqdm


# ------------------------------------------------------------------
# Base
# ------------------------------------------------------------------

class BaseDataset(Dataset):
    """
    Abstract base class for all multi-task datasets.

    Handles cache-aware dataset loading:
      - Per-domain caching to disk (HuggingFace Arrow format)
      - Automatic creation of missing splits
    """

    def __init__(
        self,
        in_domain: List[str],
        out_domain: List[str],
        cache_dir: Path,
        seed: int,
        max_seq_length: int,
        metrics: List[str],
        monitor_metric: str,
    ):
        self.in_domain      = in_domain
        self.out_domain     = out_domain
        self.cache_dir      = cache_dir
        self.seed           = seed
        self.max_seq_length = max_seq_length
        self.metrics        = metrics
        self.monitor_metric = monitor_metric

    def _get_hf_split_and_domains(self, split: str):
        if split == "train":
            return "train", self.in_domain
        elif split == "validation":
            return "validation", self.in_domain
        elif split == "test":
            return "test", self.out_domain
        raise ValueError(f"Invalid split: {split}")

    def _format_example(self, sample, idx, split, domain=None):
        raise NotImplementedError

    def _create_dataset_by_domain(self, split, target_domains):
        raise NotImplementedError

    def load_dataset(self, split: str):
        """Load split from cache; create and cache any missing domains."""
        _, target_domains = self._get_hf_split_and_domains(split)

        if len(target_domains) == 0:
            return HFDataset.from_list([])

        cached_datasets = []
        missing_domains = []

        for domain in target_domains:
            cache_path = self.cache_dir / split / domain.lower()
            if cache_path.exists() and any(cache_path.iterdir()):
                try:
                    cached_datasets.append(load_from_disk(str(cache_path)))
                    print(f"[Info] Loaded {domain} {split} from cache.")
                except Exception:
                    missing_domains.append(domain)
            else:
                missing_domains.append(domain)

        if not missing_domains:
            return concatenate_datasets(cached_datasets)

        print(f"[Info] Creating missing {split}: {missing_domains}")
        new_datasets = self._create_dataset_by_domain(split, missing_domains)

        for domain, ds in new_datasets:
            cache_path = self.cache_dir / split / domain.lower()
            cache_path.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(cache_path))
            cached_datasets.append(ds)

        return concatenate_datasets(cached_datasets)


# ------------------------------------------------------------------
# MRQA
# ------------------------------------------------------------------

class MRQADataset(BaseDataset):
    """
    MRQA benchmark dataset.
      In-domain : SQuAD, NewsQA, TriviaQA-web, SearchQA, HotpotQA, NaturalQuestionsShort
      Out-domain: BioASQ, DROP, DuoRC.ParaphraseRC, RACE, RelationExtraction, TextbookQA
    Metrics: F1, EM  |  Monitor: F1
    """

    def __init__(self, cache_dir: Path, seed: int):
        super().__init__(
            in_domain=["SQuAD", "NewsQA", "TriviaQA-web", "SearchQA", "HotpotQA", "NaturalQuestionsShort"],
            out_domain=["BioASQ", "DROP", "DuoRC.ParaphraseRC", "RACE", "RelationExtraction", "TextbookQA"],
            cache_dir=cache_dir,
            seed=seed,
            max_seq_length=512,
            metrics=["f1", "em"],
            monitor_metric="f1",
        )

    def _create_dataset_by_domain(self, split, target_domains):
        raw_dataset = load_dataset("mrqa", "plain_text", split=split)

        grouped = {d: [] for d in target_domains}

        for idx, sample in enumerate(tqdm(raw_dataset)):
            parsed = self._format_example(sample, idx, split)
            if parsed and parsed["domain"] in target_domains:
                domain          = parsed["domain"]
                parsed["task_id"] = self.in_domain.index(domain) if domain in self.in_domain else -1
                grouped[domain].append(parsed)

        return [(d, HFDataset.from_list(v)) for d, v in grouped.items() if v]

    def _format_example(self, sample, idx, split, domain=None):
        domain   = sample["subset"]
        context  = sample.get("context", "")
        question = sample.get("question", "")
        answers  = sample.get("answers", [])

        if not context or not question or not answers:
            return None

        answers = (
            [str(a).strip() for a in answers if str(a).strip()]
            if isinstance(answers, list)
            else [str(answers).strip()]
        )

        if not answers:
            return None

        return {
            "id":          f"{domain}_{idx}",
            "domain":      domain,
            "prompt_head": f"<|start_header_id|>user<|end_header_id|>\nContext: {context}",
            "prompt_tail": f"\nQuestion: {question}\n<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\nAnswer:",
            "target_text": answers[0],
            "answers":     answers,
        }


# ------------------------------------------------------------------
# summary
# ------------------------------------------------------------------

class summarizationDataset(BaseDataset):
    """
    Diverse Summarization benchmark dataset.
      In-domain : WikiLingua, Reddit, XSum, CNN_DailyMail, Gigaword, MediaSum
      Out-domain: DialogSum, Scitldr, AESLC, WikiHow, MultiNews
    Metrics: ROUGE-L  |  Monitor: ROUGE-L
    """

    def __init__(self, cache_dir: Path, seed: int):
        super().__init__(
            in_domain=['WikiLingua', 'Reddit', 'XSum', 'CNN_DailyMail', 'Gigaword', 'MediaSum'],
            out_domain=["DialogSum", "Scitldr", "AESLC", "WikiHow", "MultiNews"],
            cache_dir=cache_dir,
            seed=seed,
            max_seq_length=768,
            metrics=["rougeL"],
            monitor_metric="rougeL",
        )

        self.CONFIG = {
            # In-domain
            "XSum":          ("xsum", None),
            "CNN_DailyMail": ("abisee/cnn_dailymail", "3.0.0"),
            "WikiLingua":    ("GEM/wiki_lingua", "en"),
            "MediaSum":      ("ccdv/mediasum", None),
            "Reddit":        ("reddit_tifu", "long"),
            "Gigaword":      ("SalmanFaroz/gigaword", None),
            # Out-of-domain
            "DialogSum":     ("knkarthick/dialogsum", None),
            "Scitldr":       ("allenai/scitldr", None),
            "AESLC":         ("Yale-LILY/aeslc", None),
            "WikiHow":       ("gursi26/wikihow-cleaned", None),
            "MultiNews":     ("multi_news", None),
        }

        self.seed      = 42
        self.MAX_TRAIN = 40000
        self.MAX_VAL   = 10000
        self.MAX_TEST  = 3000

    def _load(self, domain: str, split: str):
        if domain == "Reddit":
            raw        = load_dataset("reddit_tifu", "long", split="train")
            split_data = raw.train_test_split(test_size=0.1, seed=self.seed)
            return split_data["train"] if split == "train" else split_data["test"]

        path, config = self.CONFIG[domain]
        try:
            return load_dataset(path, config, split=split) if config else load_dataset(path, split=split)
        except Exception:
            raw        = load_dataset(path, config, split="train") if config else load_dataset(path, split="train")
            split_data = raw.train_test_split(test_size=0.1, seed=self.seed)
            return split_data["train"] if split == "train" else split_data["test"]

    def _truncate_by_split(self, ds, split: str):
        if split == "train"      and len(ds) > self.MAX_TRAIN: return ds.select(range(self.MAX_TRAIN))
        if split == "validation" and len(ds) > self.MAX_VAL:   return ds.select(range(self.MAX_VAL))
        if split == "test"       and len(ds) > self.MAX_TEST:  return ds.select(range(self.MAX_TEST))
        return ds

    def _create_dataset_by_domain(self, split, target_domains):
        result = []
        for domain in target_domains:
            print(f"[Info] Loading {domain}")
            try:
                ds    = self._load(domain, split)
                ds    = self._truncate_by_split(ds, split)
                items = []
                for i, sample in enumerate(tqdm(ds, leave=False)):
                    parsed = self._format_example(sample, i, split, domain)
                    if parsed:
                        parsed["task_id"] = self.in_domain.index(domain) if domain in self.in_domain else -1
                        items.append(parsed)
                if items:
                    result.append((domain, HFDataset.from_list(items)))
            except Exception as e:
                print(f"[Error] {domain}: {e}")
        return result

    def _format_example(self, sample, idx, split, domain=None):
        body, target = None, None

        # In-domain
        if domain == "XSum":
            body, target = sample.get("document", ""), sample.get("summary", "")
        elif domain == "CNN_DailyMail":
            body, target = sample.get("article", ""), sample.get("highlights", "")
        elif domain == "WikiLingua":
            body, target = sample.get("source", ""), sample.get("target", "")
        elif domain == "MediaSum":
            body, target = sample.get("document", ""), sample.get("summary", "")
        elif domain == "Reddit":
            body  = sample.get("documents", "") or sample.get("document", "")
            target = sample.get("tldr", "")
        elif domain == "Gigaword":
            body, target = sample.get("article", ""), sample.get("summary", "")
        # Out-of-domain
        elif domain == "DialogSum":
            body, target = sample.get("dialogue", ""), sample.get("summary", "")
        elif domain == "Scitldr":
            body, target = sample.get("source", ""), sample.get("target", "")
        elif domain == "AESLC":
            body, target = sample.get("email_body", ""), sample.get("subject_line", "")
        elif domain == "WikiHow":
            body, target = sample.get("text", ""), sample.get("summary", "")
        elif domain == "MultiNews":
            body, target = sample.get("document", ""), sample.get("summary", "")
        else:
            return None

        if isinstance(body,   list): body   = "\n".join([str(x) for x in body   if str(x).strip()])
        if isinstance(target, list): target = " ".join([str(x)  for x in target if str(x).strip()])

        body, target = str(body).strip(), str(target).strip()
        if not body or not target:
            return None

        return {
            "id":          f"{domain}_{idx}",
            "domain":      domain,
            "prompt_head": f"<|start_header_id|>user<|end_header_id|>\nDocument: {body}",
            "prompt_tail": "\n<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\nSummary:",
            "target_text": target,
            "answers":     [target],
        }


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

DATASET_REGISTRY = {
    "MRQA":    MRQADataset,
    "summary": summarizationDataset,
}


# ------------------------------------------------------------------
# Collate
# ------------------------------------------------------------------

def collate_fn(
    batch: list[dict],
    split: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
):
    """
    Collate a list of dataset items into a batched dict of tensors.

    Train split:
      - Concatenates prompt + answer tokens for teacher-forcing
      - Labels mask out prompt tokens with -100
      - Provides separate routing_input_ids for the prompt module

    Validation / Test split:
      - Left-pads prompt tokens for autoregressive generation
    """
    ids          = [item["id"]      for item in batch]
    task_ids     = [item["task_id"] for item in batch]
    answers_list = [item.get("answers", [item["target_text"]]) for item in batch]

    input_ids_list, attention_mask_list, labels_list = [], [], []
    routing_input_ids_list, routing_attention_mask_list = [], []

    for item in batch:
        prompt_head = item["prompt_head"]
        prompt_tail = item["prompt_tail"]
        target_text = " " + item["target_text"] + tokenizer.eos_token

        if split == "train":
            routing_token = tokenizer(
                prompt_head,
                prompt_tail,
                max_length=max_length,
                truncation="only_first",
                padding="max_length",
                return_tensors="pt",
                add_special_tokens=False,
            )

            min_prompt_len = 32

            answer_token = tokenizer(
                target_text,
                max_length=max_length - min_prompt_len,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )

            prompt_token = tokenizer(
                prompt_head,
                prompt_tail,
                max_length=max_length - len(answer_token["input_ids"][0]),
                truncation="only_first",
                return_tensors="pt",
                add_special_tokens=False,
            )

            input_ids      = prompt_token["input_ids"][0].tolist() + answer_token["input_ids"][0].tolist()
            attention_mask = prompt_token["attention_mask"][0].tolist() + [1] * len(answer_token["input_ids"][0])

            pad_len = max_length - len(input_ids)
            if pad_len > 0:
                input_ids      += [tokenizer.pad_token_id] * pad_len
                attention_mask += [0] * pad_len

            labels  = [-100] * len(prompt_token["input_ids"][0]) + answer_token["input_ids"][0].tolist()
            labels += [-100] * (max_length - len(labels))

            routing_input_ids_list.append(routing_token["input_ids"][0])
            routing_attention_mask_list.append(routing_token["attention_mask"][0])

        else:
            prompt_token = tokenizer(
                prompt_head,
                prompt_tail,
                max_length=max_length,
                truncation="only_first",
                return_tensors="pt",
                add_special_tokens=False,
            )
            input_ids      = prompt_token["input_ids"][0].tolist()
            attention_mask = prompt_token["attention_mask"][0].tolist()

            pad_len = max_length - len(input_ids)
            if pad_len > 0:
                input_ids      = [tokenizer.pad_token_id] * pad_len + input_ids
                attention_mask = [0] * pad_len + attention_mask

            labels = [-100] * max_length

        input_ids_list.append(torch.tensor(input_ids))
        attention_mask_list.append(torch.tensor(attention_mask))
        labels_list.append(torch.tensor(labels))

    batch_dict = {
        "ids":            ids,
        "task_ids":       torch.tensor(task_ids),
        "input_ids":      torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
        "labels":         torch.stack(labels_list),
        "answers":        answers_list,
    }

    if split == "train":
        batch_dict["routing_input_ids"]      = torch.stack(routing_input_ids_list)
        batch_dict["routing_attention_mask"] = torch.stack(routing_attention_mask_list)

    return batch_dict

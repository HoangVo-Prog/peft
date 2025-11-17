from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding
from typing import Optional, List

from ..utils.config import GLUE_SENTENCE_KEYS, is_regression_task, RunConfig


def load_glue_and_tokenizer(cfg: RunConfig) -> tuple[DatasetDict, AutoTokenizer, DataCollatorWithPadding, int, Optional[List[str]]]:
    """
    Load a GLUE dataset and prepare tokenizer, collator, and label metadata.

    Returns
    -------
    encoded : DatasetDict
        Tokenized GLUE dataset.
    tokenizer : AutoTokenizer
        Hugging Face tokenizer used for preprocessing.
    collator : DataCollatorWithPadding
        Data collator with dynamic padding.
    num_labels : int
        Number of labels (1 for regression tasks).
    label_list : list[str] or None
        List of label names for classification, or None for regression.
    """
    task = cfg.task_name.lower()
    if task not in GLUE_SENTENCE_KEYS:
        raise ValueError(f"Unknown GLUE task: {task}")

    # 1. Load raw dataset and tokenizer
    raw: DatasetDict = load_dataset("glue", task)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    # 2. Determine max sequence length (respect model_max_length)
    max_seq_length = min(cfg.max_seq_length, tokenizer.model_max_length)

    sent1_key, sent2_key = GLUE_SENTENCE_KEYS[task]

    # 3. Preprocessing function: tokenize and attach labels
    def preprocess(batch):
        if sent2_key is None:
            tokenized = tokenizer(
                batch[sent1_key],
                truncation=True,
                max_length=max_seq_length,
            )
        else:
            tokenized = tokenizer(
                batch[sent1_key],
                batch[sent2_key],
                truncation=True,
                max_length=max_seq_length,
            )

        if "label" in batch:
            tokenized["labels"] = batch["label"]
        return tokenized

    encoded = raw.map(
        preprocess,
        batched=True,
        remove_columns=raw["train"].column_names,
    )

    # 4. Collator: dynamic padding, pad_to_multiple_of=8 if using fp16 or bf16
    pad_mult = 8 if (cfg.fp16 or cfg.bf16) else None
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=pad_mult,
    )

    # 5. Label metadata
    if is_regression_task(task):
        num_labels = 1
        label_list: Optional[List[str]] = None
    else:
        label_list = raw["train"].features["label"].names
        num_labels = len(label_list)

    return encoded, tokenizer, collator, num_labels, label_list

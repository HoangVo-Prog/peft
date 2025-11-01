import json
import os

import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset

from ..utils.config import GLUE_SENTENCE_KEYS, is_regression_task
from .metrics import build_compute_metrics


def evaluate_checkpoint(checkpoint_dir: str, task: str, batch_size: int = 64):
    task = task.lower()

    raw = load_dataset("glue", task)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, use_fast=True)
    sent1_key, sent2_key = GLUE_SENTENCE_KEYS[task]

    def preprocess(batch):
        if sent2_key is None:
            tokenized = tokenizer(batch[sent1_key], truncation=True)
        else:
            tokenized = tokenizer(batch[sent1_key], batch[sent2_key], truncation=True)
        if "label" in batch:
            tokenized["labels"] = batch["label"]
        return tokenized

    encoded = raw.map(preprocess, batched=True, remove_columns=raw["train"].column_names)

    eval_ds = encoded["validation_matched"] if task == "mnli" else encoded["validation"]

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)

    compute_metrics = build_compute_metrics(task)
    collator = DataCollatorWithPadding(tokenizer)

    args = TrainingArguments(
        output_dir=os.path.join(checkpoint_dir, "eval_tmp"),
        per_device_eval_batch_size=batch_size,
        report_to=["none"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate()
    print("Reloaded checkpoint metrics:", metrics)
    with open(os.path.join(checkpoint_dir, "eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics
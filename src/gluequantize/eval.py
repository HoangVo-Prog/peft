
import json
import os

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer

from src.utils.metrics import build_compute_metrics
from src.utils.config import GLUE_SENTENCE_KEYS

# Prefer AutoPeft if available, fallback otherwise
try:
    from peft import AutoPeftModelForSequenceClassification as _AutoPeft
    _HAS_AP = True
except Exception:
    _HAS_AP = False
    from transformers import AutoModelForSequenceClassification


def evaluate_adapter(checkpoint_dir: str, task: str, batch_size: int = 64):
    task = task.lower()
    raw = load_dataset("glue", task)

    sent1, sent2 = GLUE_SENTENCE_KEYS[task]
    tok = AutoTokenizer.from_pretrained(checkpoint_dir)
    def preprocess(batch):
        if sent2 is None:
            return tok(batch[sent1], truncation=True)
        return tok(batch[sent1], batch[sent2], truncation=True)

    encoded = raw.map(preprocess, batched=True, remove_columns=raw["train"].column_names)
    collator = DataCollatorWithPadding(tokenizer=tok)

    eval_ds = encoded["validation_mismatched"] if "validation_mismatched" in encoded else encoded["validation"]

    if _HAS_AP:
        model = _AutoPeft.from_pretrained(checkpoint_dir, low_cpu_mem_usage=True)
    else:
        # Fallback: plain base model (assumes merged weights)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)

    args = TrainingArguments(
        output_dir=os.path.join(checkpoint_dir, "eval_out"),
        per_device_eval_batch_size=batch_size,
        report_to=["none"],
    )

    compute_metrics = build_compute_metrics(task)

    tr = Trainer(
        model=model,
        args=args,
        eval_dataset=eval_ds,
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    metrics = tr.evaluate()
    print("Eval (adapter) metrics:", metrics)
    with open(os.path.join(checkpoint_dir, "eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main():
    import argparse
    p = argparse.ArgumentParser(description="Evaluate LoRA/Adapter checkpoint on GLUE")
    p.add_argument("--checkpoint_dir", required=True, type=str)
    p.add_argument("--task", required=True, type=str)
    p.add_argument("--batch_size", default=64, type=int)
    args = p.parse_args()
    evaluate_adapter(args.checkpoint_dir, args.task, args.batch_size)


if __name__ == "__main__":
    main()

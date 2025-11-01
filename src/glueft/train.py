import json
import os
from datetime import datetime

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from ..utils.config import RunConfig, is_regression_task
from ..utils.data import load_glue_and_tokenizer
from ..utils.metrics import build_compute_metrics, get_best_metric_for_task
from ..utils.wandb_utils import setup_wandb


def _timestamp():
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def train(cfg: RunConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)

    task = cfg.task_name.lower()

    # Data
    encoded, tokenizer, collator, num_labels, _ = load_glue_and_tokenizer(task, cfg.model_name)

    # Model
    config = AutoConfig.from_pretrained(cfg.model_name, num_labels=num_labels)
    if is_regression_task(task):
        config.problem_type = "regression"
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, config=config)

    # Splits
    train_ds = encoded["train"]
    if task == "mnli":
        eval_ds = encoded["validation_matched"]
        eval_mm_ds = encoded["validation_mismatched"]
    else:
        eval_ds = encoded["validation"]
        eval_mm_ds = None

    # Metrics
    compute_metrics = build_compute_metrics(task)
    best_metric = get_best_metric_for_task(task)

    # W&B
    if cfg.wandb_enable:
        run_name = setup_wandb(
            task=task,
            model_name=cfg.model_name,
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            run_name=cfg.wandb_run_name,
            offline_fallback=cfg.wandb_offline_fallback,
        )
        report_targets = ["wandb"]
    else:
        run_name = f"{task}-{cfg.model_name}-{_timestamp()}"
        report_targets = ["none"]

    # Arguments
    args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        num_train_epochs=cfg.num_train_epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        eval_strategy=cfg.eval_strategy,
        save_strategy=cfg.save_strategy,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=best_metric,
        greater_is_better=True,
        fp16=bool(cfg.fp16 and torch.cuda.is_available()),
        bf16=bool(cfg.bf16 and torch.cuda.is_available()),
        logging_steps=50,
        report_to=report_targets,
        run_name=run_name,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()

    # Final marker for W&B
    try:
        if cfg.wandb_enable:
            import wandb  # type: ignore
            wandb.log({"final/global_step": trainer.state.global_step})
    except Exception:
        pass

    # Save best
    best_dir = os.path.join(cfg.output_dir, "best_model")
    os.makedirs(best_dir, exist_ok=True)
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)

    # Eval
    val_metrics = trainer.evaluate(eval_dataset=eval_ds)
    print("Validation:", val_metrics)

    if cfg.wandb_enable:
        try:
            import wandb  # type: ignore
            wandb.log({f"val/{k}": v for k, v in val_metrics.items()})
        except Exception:
            pass

    if eval_mm_ds is not None:
        mm_metrics = trainer.evaluate(eval_dataset=eval_mm_ds)
        print("Validation mismatched:", mm_metrics)
        if cfg.wandb_enable:
            try:
                import wandb  # type: ignore
                wandb.log({f"val_mm/{k}": v for k, v in mm_metrics.items()})
            except Exception:
                pass
    else:
        mm_metrics = None

    with open(os.path.join(cfg.output_dir, "val_metrics.json"), "w") as f:
        json.dump(val_metrics, f, indent=2)
    if mm_metrics is not None:
        with open(os.path.join(cfg.output_dir, "val_mm_metrics.json"), "w") as f:
            json.dump(mm_metrics, f, indent=2)

    # Dump logits for later analysis
    def dump_preds(ds, name):
        preds = trainer.predict(ds)
        np.save(os.path.join(cfg.output_dir, f"{name}_logits.npy"), preds.predictions)
        np.save(os.path.join(cfg.output_dir, f"{name}_labels.npy"), preds.label_ids)

    dump_preds(eval_ds, "val")
    if eval_mm_ds is not None:
        dump_preds(eval_mm_ds, "val_mismatched")

    if "test" in encoded:
        test_ds = encoded["test"]
    # Ensure no label columns exist to avoid CrossEntropy on invalid targets
    for col in ("label", "labels"):
        if col in test_ds.column_names:
            test_ds = test_ds.remove_columns(col)
    try:
        test_preds = trainer.predict(test_ds, metric_key_prefix="test").predictions
        np.save(os.path.join(cfg.output_dir, "test_logits.npy"), test_preds)
    except Exception as e:
        print("[WARN] Skipping test prediction due to:", e)
    
    print("Saved best model to:", best_dir)
    
    try:
        if cfg.wandb_enable:
            import wandb  # type: ignore
            wandb.finish()
    except Exception:
        pass


def main(cfg: RunConfig):
    train(cfg)
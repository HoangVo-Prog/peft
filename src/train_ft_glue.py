#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
import time
from typing import Optional

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)


from src.utils.config import RunConfig, is_regression_task, GLUE_TASKS  # type: ignore
from src.utils.data import load_glue_and_tokenizer  # type: ignore
from src.utils.metrics import build_compute_metrics, get_best_metric_for_task  # type: ignore
from src.utils.wandb_utils import setup_wandb  # type: ignore


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def train(cfg: RunConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    set_seed(cfg.seed)
    
    if cfg.fp16:
        precision = "fp16"
    elif cfg.bf16:
        precision = "bf16"
    else:
        precision = "fp32"


    task = cfg.task_name.lower()

    # Data
    encoded, tokenizer, collator, num_labels, _ = load_glue_and_tokenizer(task, cfg.model_name)

    # Model
    config = AutoConfig.from_pretrained(cfg.model_name, num_labels=num_labels)
    if is_regression_task(task):
        config.problem_type = "regression"
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, config=config)
    num_params = model.num_parameters() if hasattr(model, "num_parameters") else sum(p.numel() for p in model.parameters())

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

    # TrainingArguments
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
        logging_steps=cfg.logging_steps,
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
    start_time = time.perf_counter()
    trainer.train()
    train_time = time.perf_counter() - start_time

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
    def dump_preds(ds, name: str) -> None:
        preds = trainer.predict(ds)
        np.save(os.path.join(cfg.output_dir, f"{name}_logits.npy"), preds.predictions)
        np.save(os.path.join(cfg.output_dir, f"{name}_labels.npy"), preds.label_ids)

    dump_preds(eval_ds, "val")
    if eval_mm_ds is not None:
        dump_preds(eval_mm_ds, "val_mismatched")

    # Optional test set (some GLUE tasks hide test labels)
    test_ds = None
    if "test" in encoded:
        test_ds = encoded["test"]
        # Ensure no label columns exist to avoid CE on invalid targets
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
    
    run_summary = {
        "task": task,
        "model_name": cfg.model_name,
        "precision": precision,
        "num_parameters": int(num_params),
        "train_time_sec": float(train_time),
        "val_metrics": val_metrics,
        "val_mm_metrics": mm_metrics,
        "seed": cfg.seed,
        "output_dir": cfg.output_dir,
    }
    
    summary_path = os.path.join(
        cfg.output_dir,
        f"metrics_{task}_{precision}.json"
    )
    with open(summary_path, "w") as f:
        json.dump(run_summary, f, indent=2)
        
    return run_summary

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified GLUE fine-tuning (HF Trainer)")

    # Primary names
    p.add_argument("--all", "--all_task", dest="all", action="store_true", help="Run all GLUE tasks defined in GLUE_TASKS")
    p.add_argument("--task", "--task_name", dest="task_name", type=str, default="sst2")
    p.add_argument("--model", "--model_name", dest="model_name", type=str, default="bert-base-uncased")
    p.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default="./outputs")
    p.add_argument("--epochs", "--num_train_epochs", dest="num_train_epochs", type=float, default=3.0)
    p.add_argument("--train-bsz", "--per_device_train_batch_size", dest="per_device_train_batch_size", type=int, default=32)
    p.add_argument("--eval-bsz", "--per_device_eval_batch_size", dest="per_device_eval_batch_size", type=int, default=64)
    p.add_argument("--lr", "--learning_rate", dest="learning_rate", type=float, default=2e-5)
    p.add_argument("--wd", "--weight_decay", dest="weight_decay", type=float, default=0.01)
    p.add_argument("--warmup", "--warmup_ratio", dest="warmup_ratio", type=float, default=0.06)
    p.add_argument("--seed", dest="seed", type=int, default=42)

    p.add_argument("--save-strategy", "--save_strategy", dest="save_strategy", type=str, default="epoch")
    p.add_argument("--eval-strategy", "--eval_strategy", dest="eval_strategy", type=str, default="epoch")
    p.add_argument("--save-total", "--save_total_limit", dest="save_total_limit", type=int, default=2)

    p.add_argument("--fp16", dest="fp16", action="store_true")
    p.add_argument("--bf16", dest="bf16", action="store_true")

    # Logging
    p.add_argument("--logging-steps", dest="logging_steps", type=int, default=100)

    # W&B
    p.add_argument("--no-wandb", dest="wandb_enable", action="store_false")
    p.add_argument("--wandb-project", dest="wandb_project", type=str, default="glue-ft")
    p.add_argument("--wandb-entity", dest="wandb_entity", type=str, default=None)
    p.add_argument("--wandb-name", dest="wandb_run_name", type=str, default=None)
    p.add_argument("--wandb-offline", dest="wandb_offline_fallback", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RunConfig(**vars(args))
    if not cfg.all:
        train(cfg)
        
    run_cfg = cfg
    for task in GLUE_TASKS:
        run_cfg.task_name = task
        
        summaries = train(run_cfg)
        aggregated = {
            "model_name": args.model_name,
            "precision": summaries["precision"],   # compute once from args.fp16 / args.bf16
            "tasks": {summaries["task"]},
        }

        out_name = f"metrics_all_tasks_{summaries['precision']}.json"
        out_path = os.path.join(args.output_dir, out_name)
        with open(out_path, "w") as f:
            json.dump(aggregated, f, indent=2)

if __name__ == "__main__":
    main()

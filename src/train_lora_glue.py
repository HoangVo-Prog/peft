# train_lora_glue.py
from dataclasses import dataclass
import os
from typing import Optional, List
from datetime import datetime
import json
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)

from peft import LoraConfig, TaskType, get_peft_model

from src.utils.data import load_glue_and_tokenizer
from src.utils.metrics import build_compute_metrics, get_best_metric_for_task
from src.utils.config import RunConfig, is_regression_task, GLUE_TASKS
from src.utils.wandb_utils import setup_wandb  # type: ignore


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")

# ---------------- main training routine ----------------

@dataclass
class LoRAArgs:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"  # "none", "all", or "lora_only"
    target_modules: List[str] = ["key", "query", "value"]
    modules_to_save: List[str] = ["classifier"]
    seed: int = 42
    gradient_checkpointing: bool = False


def train(cfg: RunConfig, lora: LoRAArgs):
    
    os.makedirs(cfg.output_dir, exist_ok=True)
    out_dir = os.path.join(cfg.output_dir, cfg.task_name)
    os.makedirs(out_dir, exist_ok=True)
    
    set_seed(lora.seed)

    task = cfg.task_name.lower()

    # Load data and tokenizer
    encoded, tokenizer, collator, num_labels, _ = load_glue_and_tokenizer(cfg.task_name, cfg.model_name)

    # Avoid very large tokenizer.model_max_length warning
    try:
        if tokenizer.model_max_length is None or tokenizer.model_max_length > 4096:
            tokenizer.model_max_length = 512
    except Exception:
        pass

    # HF model config
    hf_cfg = AutoConfig.from_pretrained(cfg.model_name, num_labels=num_labels)
    if is_regression_task(task):
        hf_cfg.problem_type = "regression"

    base = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, config=hf_cfg)
    if lora.gradient_checkpointing:
        base.gradient_checkpointing_enable()

    lcfg = LoraConfig(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        bias=lora.bias,
        task_type=TaskType.SEQ_CLS,
        target_modules=lora.target_modules,    
        modules_to_save=lora.modules_to_save,  
    )

    model = get_peft_model(base, lcfg)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_ratio = f"{100 * trainable_params / total_params:.4f}%"
    
    if task == "mnli":
        eval_ds = encoded["validation_matched"]
        eval_mm_ds = encoded["validation_mismatched"]
    else:
        eval_ds = encoded["validation"]
        eval_mm_ds = None
        
    # Metrics
    compute_metrics = build_compute_metrics(task)
    best_metric = get_best_metric_for_task(task)
    
    try:
        model.print_trainable_parameters()
    except Exception:
        pass
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

    args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        num_train_epochs=cfg.num_train_epochs,
        weight_decay=cfg.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=best_metric,
        logging_steps=cfg.logging_steps,
        report_to=report_targets,
        run_name=run_name,
        seed=lora.seed,
        fp16=torch.cuda.is_available(),
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded["train"],
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    # Train
    start_time = time.perf_counter()
    trainer.train()
    train_time = time.perf_counter() - start_time
    
    # Save adapter and tokenizer
    trainer.model.save_pretrained(os.path.join(cfg.output_dir, task))
    tokenizer.save_pretrained(cfg.output_dir)

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

    with open(os.path.join(cfg.output_dir, f"val_metrics_{task}.json"), "w") as f:
        json.dump(val_metrics, f, indent=2)
    if mm_metrics is not None:
        with open(os.path.join(cfg.output_dir, f"val_mm_metrics_{task}.json"), "w") as f:
            json.dump(mm_metrics, f, indent=2)

    # Dump logits for later analysis
    def dump_preds(ds, name: str) -> None:
        preds = trainer.predict(ds)
        np.save(os.path.join(cfg.output_dir, f"{name}_logits_{task}.npy"), preds.predictions)
        np.save(os.path.join(cfg.output_dir, f"{name}_labels_{task}.npy"), preds.label_ids)

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
            np.save(os.path.join(cfg.output_dir, f"test_logits_{task}.npy"), test_preds)
        except Exception as e:
            print("[WARN] Skipping test prediction due to:", e)

    try:
        if cfg.wandb_enable:
            import wandb  # type: ignore
            wandb.finish()
    except Exception:
        pass

    run_summary = {
        "task": task,
        "num_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "trainable_ratio": trainable_ratio,
        "train_time_sec": float(train_time),
        "val_metrics": val_metrics,
        "val_mm_metrics": mm_metrics,
    }

    return run_summary

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GLUE LoRA finetune")
    p.add_argument("--all", "--all_task", dest="all", action="store_true", help="Run all GLUE tasks defined in GLUE_TASKS")
    p.add_argument("--task", "--task_name", dest="task_name", type=str, default="sst2")
    p.add_argument("--model_name", type=str, default="bert-base-uncased")
    p.add_argument("--output_dir", type=str, default="./outputs/lora")
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=32)
    p.add_argument("--per_device_eval_batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--logging_steps", type=int, default=50)

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_bias", type=str, default="none")
    p.add_argument("--lora_target_modules", type=str, nargs="+", default=["key", "query", "value"], help="List of target modules for LoRA") 
    p.add_argument("--modules_to_save", type=str, nargs="+", default=["classifier"], helper="Modules training no LoRA") 
    p.add_argument("--gradient_checkpointing", action="store_true")

    # W&B
    p.add_argument("--no-wandb", dest="wandb_enable", action="store_false")   
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_offline_fallback", action="store_true")

    return p.parse_args()

def main():
    args = parse_args()
    
    # LoRA config
    largs = LoRAArgs(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        bias=args.lora_bias,
        target_modules=args.lora_target_modules,
        seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    
    if not args.all:
        cfg = RunConfig(
            task_name=args.task_name,
            model_name=args.model_name,
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            logging_steps=args.logging_steps,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_name=args.wandb_run_name,
            wandb_offline_fallback=args.wandb_offline_fallback,
            wandb_enable=bool(args.wandb_enable and args.wandb_project),
        )
    
    for task in GLUE_TASKS:
        print(f"========================================= {task} =========================================")
        cfg = RunConfig(
            task_name=task,
            model_name=args.model_name,
            output_dir=args.output_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            logging_steps=args.logging_steps,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_name=args.wandb_run_name,
            wandb_offline_fallback=args.wandb_offline_fallback,
            wandb_enable=bool(args.wandb_enable and args.wandb_project),
        )

    

    metrics = train(cfg, largs)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
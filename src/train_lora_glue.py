# train_lora_glue.py
from dataclasses import dataclass
import os
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

import shutil
import glob

from peft import LoraConfig, TaskType, get_peft_model

from src.utils.data import load_glue_and_tokenizer
from src.utils.metrics import build_compute_metrics, get_best_metric_for_task
from src.utils.config import RunConfig, LoRAArgs, is_regression_task, GLUE_TASKS
from src.utils.wandb_utils import setup_wandb  # type: ignore


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")

# ---------------- main training routine ----------------
def train(cfg: RunConfig, lora: LoRAArgs):
    
    os.makedirs(cfg.output_dir, exist_ok=True)
    out_dir = os.path.join(cfg.output_dir, cfg.task_name)
    os.makedirs(out_dir, exist_ok=True)
    
    set_seed(lora.seed)
    
    if cfg.fp16:
        precision = "fp16"
    elif cfg.bf16:
        precision = "bf16"
    else:
        precision = "fp32"

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
        eval_strategy=cfg.eval_strategy, 
        save_strategy=cfg.save_strategy,
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=best_metric,
        logging_steps=cfg.logging_steps,
        report_to=report_targets,
        run_name=run_name,
        seed=lora.seed,
        fp16=bool(cfg.fp16 and torch.cuda.is_available()), # TODO:
        bf16=bool(cfg.bf16 and torch.cuda.is_available()),
        optim="adamw_torch", # default optim
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
    
    # # Save adapter and tokenizer
    # trainer.model.save_pretrained(os.path.join(cfg.output_dir, task))
    # tokenizer.save_pretrained(cfg.output_dir)

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
    
    # Delete all checkpoints to save storage
    for ckpt_dir in glob.glob(os.path.join(out_dir, "checkpoint-*")):
        shutil.rmtree(ckpt_dir)
    
    run_summary = {
        "task": task,
        "num_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "trainable_ratio": trainable_ratio,
        "train_time_sec": float(train_time),
        "val_metrics": val_metrics,
        "val_mm_metrics": mm_metrics,
    }
    
    summary_path = os.path.join(
        out_dir,
        f"metrics_{task}_{precision}.json"
    )
    with open(summary_path, "w") as f:
        json.dump(run_summary, f, indent=2)

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
    p.add_argument("--logging_steps", type=int, default=100)

    p.add_argument("--save-strategy", "--save_strategy", dest="save_strategy", type=str, default="epoch")
    p.add_argument("--eval-strategy", "--eval_strategy", dest="eval_strategy", type=str, default="epoch")
    p.add_argument("--save-total", "--save_total_limit", dest="save_total_limit", type=int, default=1)
    
    p.add_argument("--fp16", dest="fp16", action="store_true")
    p.add_argument("--bf16", dest="bf16", action="store_true")


    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_bias", type=str, default="none")
    p.add_argument("--lora_target_modules", dest="target_modules", type=str, nargs="+", default=["key", "query", "value"], help="List of target modules for LoRA") 
    p.add_argument("--modules_to_save", type=str, nargs="+", default=["classifier"], help="Modules training no LoRA") 
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
    
    # Run config
    cfg = RunConfig()  
    # LoRA config
    largs = LoRAArgs()

    for k, v in vars(args).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        if hasattr(largs, k):
            setattr(cfg, k, v)
    
    summaries = {
        "model_name": args.model_name,
        "task": []
    }
    
    if not args.all:
        summaries["task"].append(train(cfg, largs))
    else:
        for task in GLUE_TASKS:
            print(f"========================================= {task} =========================================")
            cfg.task_name = task
            summaries["task"].append(train(cfg, largs))

    
    model_name = str(args.model_name).replace("/", "_")
    out_name = f"{model_name}_all_tasks.json"
    out_path = os.path.join(args.output_dir, out_name)
    with open(out_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Saved summatries metrics to: {out_path}")


if __name__ == "__main__":
    main()
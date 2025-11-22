# qlora_glue_train.py
from dataclasses import dataclass
import os

import torch
import torch.nn as nn

from typing import Optional, List
from datetime import datetime
import json
import time
import glob
import shutil
import numpy as np

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
    BitsAndBytesConfig,
)

import argparse

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from src.utils.data import load_glue_and_tokenizer
from src.utils.metrics import build_compute_metrics, get_best_metric_for_task
from src.utils.config import RunConfig, is_regression_task, GLUE_TASKS, QLoRAArgs
from src.utils.wandb_utils import setup_wandb  


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


# ===================== QLoRA-specific helpers =====================

def keep_classifier_fp32(model: nn.Module):
    """
    Keep classifier head in fp32 even if inner layers are 4-bit.
    Handles heads shaped as .classifier.dense/.out_proj or simple Linear.
    """
    try:
        import bitsandbytes as bnb  # noqa
        Linear4bit = getattr(bnb.nn, "Linear4bit", tuple())
    except Exception:
        Linear4bit = tuple()

    def _upgrade(module_name: str):
        mod = model
        for part in module_name.split("."):
            if hasattr(mod, part):
                mod = getattr(mod, part)
            else:
                return False

        if isinstance(mod, Linear4bit):
            parent = model
            parts = module_name.split(".")
            for p in parts[:-1]:
                parent = getattr(parent, p)
            leaf_name = parts[-1]
            in_f, out_f = mod.in_features, mod.out_features
            new_lin = nn.Linear(in_f, out_f, bias=getattr(mod, "bias", None) is not None).to(torch.float32)
            try:
                with torch.no_grad():
                    new_lin.weight.copy_(mod.weight.float())
                    if getattr(mod, "bias", None) is not None:
                        new_lin.bias.copy_(mod.bias.float())
            except Exception:
                pass
            setattr(parent, leaf_name, new_lin)
            return True
        return False

    # try common locations
    _upgrade("classifier.dense")
    _upgrade("classifier.out_proj")
    # if classifier itself is Linear4bit
    if hasattr(model, "classifier") and isinstance(getattr(model, "classifier"), Linear4bit):
        cls = getattr(model, "classifier")
        in_f, out_f = cls.in_features, cls.out_features
        new_lin = nn.Linear(in_f, out_f, bias=getattr(cls, "bias", None) is not None).to(torch.float32)
        try:
            with torch.no_grad():
                new_lin.weight.copy_(cls.weight.float())
                if getattr(cls, "bias", None) is not None:
                    new_lin.bias.copy_(cls.bias.float())
        except Exception:
            pass
        model.classifier = new_lin

    if hasattr(model, "classifier"):
        model.classifier.to(torch.float32)
        for p in model.classifier.parameters():
            p.requires_grad = True

def align_classifier_device(model: nn.Module):
    dev = next(model.parameters()).device
    if hasattr(model, "classifier"):
        model.classifier.to(dev)

# ===================== Train =====================

def train(cfg: RunConfig, qlora: QLoRAArgs):
    
    os.makedirs(cfg.output_dir, exist_ok=True)
    out_dir = os.path.join(cfg.output_dir, cfg.task_name)
    os.makedirs(out_dir, exist_ok=True)
    
    set_seed(qlora.seed)

    if cfg.fp16:
        precision = "fp16"
        compute_type = torch.float16
    elif cfg.bf16:
        precision = "bf16"
        compute_type = torch.bfloat16
    else:
        precision = "fp32"
        compute_type = torch.float32
        
    task = cfg.task_name.lower()

    # Load data and tokenizer
    encoded, tokenizer, collator, num_labels, _ = load_glue_and_tokenizer(cfg)
    
    # Avoid very large tokenizer.model_max_length warning
    try:
        if tokenizer.model_max_length is None or tokenizer.model_max_length > 4096:
            tokenizer.model_max_length = 512
    except Exception:
        pass

    # HF config
    hf_cfg = AutoConfig.from_pretrained(cfg.model_name, num_labels=num_labels)
    if is_regression_task(cfg.task_name):
        hf_cfg.problem_type = "regression"

    # 4-bit quantization config
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=qlora.quant_type,
        bnb_4bit_use_double_quant=qlora.double_quantize,
        bnb_4bit_compute_dtype=compute_type,
    )

    # Load in 4-bit
    base = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        config=hf_cfg,
        quantization_config=bnb_cfg,
        torch_dtype=compute_type,
        device_map="auto",
    )

    # Keep head fp32 and on the right device
    keep_classifier_fp32(base)
    align_classifier_device(base)
    
    # # Verify that pooler.dense is not Linear4bit anymore
    # if hasattr(base, "pooler") and hasattr(base.pooler, "dense"):
    #     try:
    #         import bitsandbytes as bnb
    #         if isinstance(base.pooler.dense, bnb.nn.Linear4bit):
    #             raise RuntimeError("pooler.dense is still Linear4bit after conversion!")
    #     except ImportError:
    #         pass
  
    # prepare k-bit & gradient checkpointing
    base = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=cfg.gradient_enable,
    )
    
    # LoRA config
    lcfg = LoraConfig(
        r=qlora.r,
        lora_alpha=qlora.alpha,
        lora_dropout=qlora.dropout,
        bias=qlora.bias,
        task_type=TaskType.SEQ_CLS,
        target_modules=qlora.target_modules,    # may be list or "all-linear"
        modules_to_save=qlora.modules_to_save,  # keep head/pooler trainable
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
            task=cfg.task_name,
            model_name=cfg.model_name,
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            run_name=cfg.wandb_run_name,
            offline_fallback=cfg.wandb_offline_fallback,
        )
        report_targets = ["wandb"]
    else:
        run_name = f"{cfg.task_name}-{cfg.model_name}-{_timestamp()}"
        report_targets = ["none"]

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        num_train_epochs=cfg.num_train_epochs,
        weight_decay=cfg.weight_decay,
        save_total_limit=cfg.save_total_limit,
        eval_strategy=cfg.eval_strategy,     
        save_strategy=cfg.save_strategy,
        load_best_model_at_end=True,
        metric_for_best_model=best_metric,
        logging_steps=cfg.logging_steps,
        report_to=report_targets,
        run_name=run_name,
        seed=qlora.seed,
        fp16=bool(cfg.fp16 and torch.cuda.is_available()), # TODO:
        bf16=bool(cfg.bf16 and torch.cuda.is_available()),
        optim="paged_adamw_8bit", # defailt optim 
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
    
    # Reset thống kê VRAM trước khi train
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base_mem = torch.cuda.memory_allocated() / 1024**2  # MB

    # Train
    start_time = time.perf_counter()
    trainer.train()
    train_time = time.perf_counter() - start_time
    
    # Đọc peak VRAM sau khi train
    max_mem = None
    if torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
        print(f"[VRAM] base_mem={base_mem:.2f} MB, peak_train_mem={max_mem:.2f} MB")

    
    # Final marker for W&B
    try:
        if cfg.wandb_enable:
            import wandb  # type: ignore
            wandb.log({"final/global_step": trainer.state.global_step})
    except Exception:
        pass

    # # Save adapter + tokenizer
    # trainer.model.save_pretrained(cfg.output_dir)
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
        "target_modules": qlora.target_modules,
        "double_quantize": qlora.double_quantize,
        "quant_type": qlora.quant_type,
        "num_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "trainable_ratio": trainable_ratio,
        "train_time_sec": float(train_time),
        "val_metrics": val_metrics,
        "val_mm_metrics": mm_metrics,
    }
    
    if max_mem is not None:
        run_summary["vram_base_mb"] = float(base_mem)
        run_summary["vram_peak_mb"] = float(max_mem)
    
    summary_path = os.path.join(
        out_dir,
        f"metrics_{task}_{precision}.json"
    )
    with open(summary_path, "w") as f:
        json.dump(run_summary, f, indent=2)
        
    del trainer
    del model
    del train_ds
    del eval_ds
    try:
        del eval_mm_ds
    except NameError:
        pass
    try:
        del test_ds
    except NameError:
        pass

    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return run_summary

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GLUE QLoRA finetune")
    p.add_argument("--all", "--all_task", dest="all", action="store_true", help="Run all GLUE tasks defined in GLUE_TASKS")    
    p.add_argument("--tasks", "--task_names", dest="task_names", type=str, default="sst2")
    p.add_argument("--model_name", type=str, default="bert-base-uncased")
    p.add_argument("--output_dir", type=str, default="./outputs/qlora")
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=32)
    p.add_argument("--per_device_eval_batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--logging_steps", type=int, default=100)
    
    p.add_argument("--save-strategy", "--save_strategy", dest="save_strategy", type=str, default="epoch")
    p.add_argument("--eval-strategy", "--eval_strategy", dest="eval_strategy", type=str, default="epoch")
    p.add_argument("--save-total", "--save_total_limit", dest="save_total_limit", type=int, default=1)
    p.add_argument("--gradient-enable", dest="gradient_enable", action="store_true")
    
    p.add_argument("--fp16", dest="fp16", action="store_true")
    p.add_argument("--bf16", dest="bf16", action="store_true")

    # QLoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_bias", type=str, default="none")
    p.add_argument("--lora_target_modules", type=str, default="")  # comma-separated; empty => auto
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quant_type", type=str, default="nf4")
    p.add_argument("--double-quantize", type=bool, default=True)
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
    # QLoRA config
    qargs = QLoRAArgs()
    
    for k, v in vars(args).items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
        if hasattr(qargs, k):
            setattr(qargs, k, v)
            
    summaries = {
        "model_name": args.model_name,
        "task": []
    }

    model_name = str(args.model_name).replace("/", "_")    
    if not args.all:
        tasks = [t.strip() for t in args.task_names.split(" ")]
        out_name = f"{model_name}_qlora_" + "_".join(tasks) + ".json"
    else:
        tasks = GLUE_TASKS
        out_name = f"{model_name}_qlora_all_tasks.json"

    
    for task in tasks:
        print()
        print(f"========================================= {task} =========================================")
        print()
        cfg.task_name = task
        summaries["task"].append(train(cfg, qargs))

    os.makedirs(os.path.join(args.output_dir, "qlora"), exist_ok=True)
    out_path = os.path.join(args.output_dir, "qlora", out_name)
    with open(out_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Saved summatries metrics to: {out_path}")


if __name__ == "__main__":
    main()

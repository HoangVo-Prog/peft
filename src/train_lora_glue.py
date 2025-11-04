
from dataclasses import dataclass
import os
from typing import Optional, List
from datetime import datetime

import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)

from src.utils.data import load_glue_and_tokenizer
from src.utils.metrics import build_compute_metrics, get_best_metric_for_task
from src.utils.config import RunConfig, is_regression_task
from src.utils.wandb_utils import setup_wandb  # type: ignore


from peft import LoraConfig, TaskType, get_peft_model

def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")

def guess_lora_target_modules(model):
    names = [n for n, _ in model.named_modules()]
    # LLaMA/Mistral-like
    if any(n.endswith("q_proj") for n in names):
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    # BERT/RoBERTa/DeBERTa-like
    if any(n.endswith("query") for n in names):
        return ["query", "key", "value"]
    # GPT-2
    if any(n.endswith("c_attn") for n in names):
        return ["c_attn"]
    # Fallback
    return ["query", "key", "value"]


@dataclass
class LoRAArgs:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"  # "none" or "all" or "lora_only"
    target_modules: Optional[List[str]] = None
    seed: int = 42
    gradient_checkpointing: bool = False


def train(cfg: RunConfig, lora: LoRAArgs):
    set_seed(lora.seed)

    # Load data/tokenizer and infer labels
    encoded, tokenizer, collator, num_labels, _label_list = load_glue_and_tokenizer(
        cfg.task_name, cfg.model_name
    )

    # Configure model
    hf_cfg = AutoConfig.from_pretrained(cfg.model_name, num_labels=num_labels)
    if is_regression_task(cfg.task_name):
        hf_cfg.problem_type = "regression"

    base = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, config=hf_cfg)
    if lora.gradient_checkpointing:
        base.gradient_checkpointing_enable()

    # Guess targets if not provided
    target_modules = lora.target_modules or guess_lora_target_modules(base)

    lcfg = LoraConfig(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        bias=lora.bias,
        task_type=TaskType.SEQ_CLS,
        target_modules=target_modules,
    )
    model = get_peft_model(base, lcfg)

    metric_for_best = get_best_metric_for_task(cfg.task_name)

    # W&B if configured
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
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best,
        logging_steps=cfg.logging_steps,
        report_to=report_targets,
        run_name=run_name,
        seed=lora.seed,
        fp16=torch.cuda.is_available(),
        optim="adamw_torch",
    )

    compute_metrics = build_compute_metrics(cfg.task_name)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation_mismatched"] if "validation_mismatched" in encoded else encoded["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # Save adapter and tokenizer
    trainer.model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    # Final eval
    metrics = trainer.evaluate()
    with open(os.path.join(cfg.output_dir, "eval_metrics.json"), "w") as f:
        import json
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    import argparse, json
    p = argparse.ArgumentParser(description="GLUE LoRA finetune")
    p.add_argument("--task_name", type=str, default="sst2")
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
    p.add_argument("--lora_target_modules", type=str, default="")  # comma-separated, empty to auto
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient_checkpointing", action="store_true")

    # W&B
    p.add_argument("--no-wandb", dest="wandb_enable", action="store_false")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_offline_fallback", action="store_true")

    args = p.parse_args()

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
        wandb_enable=bool(args.wandb_project),
    )

    tmods = [s.strip() for s in args.lora_target_modules.split(",") if s.strip()] or None
    largs = LoRAArgs(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        bias=args.lora_bias,
        target_modules=tmods,
        seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    metrics = train(cfg, largs)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

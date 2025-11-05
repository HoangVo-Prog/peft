# train_lora_glue.py
from dataclasses import dataclass
import os
from typing import Optional, List
from datetime import datetime
import json
import re

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

# Your project utilities
from src.utils.data import load_glue_and_tokenizer
from src.utils.metrics import build_compute_metrics, get_best_metric_for_task
from src.utils.config import RunConfig, is_regression_task
from src.utils.wandb_utils import setup_wandb  # type: ignore


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


# -------- LoRA helpers: model-agnostic target detection --------

FAMILY_PATTERNS = {
    # LLaMA and friends
    "llama_like": ["q_proj", "k_proj", "v_proj", "o_proj"],
    # BERT, RoBERTa, XLM-R
    "bert_like": ["query", "key", "value", "dense"],
    # DeBERTa v2 and v3
    "deberta_v2_v3": ["query_proj", "key_proj", "value_proj", "dense"],
    # GPT-2
    "gpt2_like": ["c_attn"],
    # GPT-NeoX, Falcon
    "neox_falcon": ["query_key_value", "dense"],
    # BLOOM
    "bloom": ["query_key_value", "dense"],
    # MPT
    "mpt": ["Wqkv", "out_proj"],
}


def _all_module_names(model: nn.Module) -> List[str]:
    return [n for n, _ in model.named_modules()]


def _find_hits(names: List[str], tokens: List[str]) -> List[str]:
    found = []
    for t in tokens:
        if any(t in n for n in names):
            found.append(t)
    return found


def guess_lora_target_modules(model: nn.Module):
    """
    Try common families first. If nothing matches, scan generically.
    If still nothing, fall back to all-linear.
    Returns a list of substrings or the string 'all-linear'.
    """
    names = _all_module_names(model)

    order = [
        "deberta_v2_v3",
        "llama_like",
        "bert_like",
        "neox_falcon",
        "bloom",
        "mpt",
        "gpt2_like",
    ]
    for fam in order:
        hits = _find_hits(names, FAMILY_PATTERNS[fam])
        if hits:
            # ensure dense is included for bert-like families if present anywhere
            if fam in ("deberta_v2_v3", "bert_like"):
                if "dense" not in hits and any(n.endswith("dense") or n.endswith(".dense") or ".output.dense" in n for n in names):
                    hits = sorted(set(hits + ["dense"]))
            return sorted(set(hits))

    # generic regex scan for attention projections and dense
    generic = set()
    for n in names:
        if re.search(r"(q(uery)?|k(ey)?|v(alue)?)_?proj$", n):
            if "q_proj" in n: generic.add("q_proj")
            if "k_proj" in n: generic.add("k_proj")
            if "v_proj" in n: generic.add("v_proj")
            if n.endswith("query_proj"): generic.add("query_proj")
            if n.endswith("key_proj"): generic.add("key_proj")
            if n.endswith("value_proj"): generic.add("value_proj")
        if n.endswith("query"): generic.add("query")
        if n.endswith("key"):   generic.add("key")
        if n.endswith("value"): generic.add("value")
        if n.endswith("dense") or ".output.dense" in n: generic.add("dense")
    if generic:
        return sorted(generic)

    # final fallback
    return "all-linear"


def pick_modules_to_save(model: nn.Module) -> List[str]:
    """
    Keep heads or poolers trainable if they exist. This is safe across families.
    """
    candidates = ["classifier", "score", "lm_head", "pooler.dense"]
    names = _all_module_names(model)
    found = []
    for c in candidates:
        if any(n.endswith(c) or n.endswith(f".{c}") for n in names):
            found.append(c)
    return found or ["classifier"]


# ---------------- main training routine ----------------

@dataclass
class LoRAArgs:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"  # "none", "all", or "lora_only"
    target_modules: Optional[List[str]] = None
    seed: int = 42
    gradient_checkpointing: bool = False


def train(cfg: RunConfig, lora: LoRAArgs):
    set_seed(lora.seed)

    # Load data and tokenizer
    encoded, tokenizer, collator, num_labels, _label_list = load_glue_and_tokenizer(
        cfg.task_name, cfg.model_name
    )

    # Avoid very large tokenizer.model_max_length warning
    try:
        if tokenizer.model_max_length is None or tokenizer.model_max_length > 4096:
            tokenizer.model_max_length = 512
    except Exception:
        pass

    # HF model config
    hf_cfg = AutoConfig.from_pretrained(cfg.model_name, num_labels=num_labels)
    if is_regression_task(cfg.task_name):
        hf_cfg.problem_type = "regression"

    base = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, config=hf_cfg)
    if lora.gradient_checkpointing:
        base.gradient_checkpointing_enable()

    # Guess LoRA target modules if not provided
    target_modules = lora.target_modules or guess_lora_target_modules(base)
    modules_to_save = pick_modules_to_save(base)

    lcfg = LoraConfig(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        bias=lora.bias,
        task_type=TaskType.SEQ_CLS,
        target_modules=target_modules,     # list or 'all-linear'
        modules_to_save=modules_to_save,   # keep head or pooler
    )

    # Optional sanity check if target_modules is a list
    if isinstance(target_modules, list):
        hit_names = [
            n for n, m in base.named_modules()
            if isinstance(m, nn.Linear) and any(t in n for t in target_modules)
        ]
        if len(hit_names) == 0:
            example_names = [n for n, _ in list(base.named_modules())[:30]]
            raise RuntimeError(
                f"No matching modules for {target_modules}. "
                f"Example module names: {example_names}"
            )

    model = get_peft_model(base, lcfg)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    metric_for_best = get_best_metric_for_task(cfg.task_name)

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
        evaluation_strategy="epoch",
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

    eval_split = (
        encoded["validation_mismatched"]
        if "validation_mismatched" in encoded
        else encoded["validation"]
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded["train"],
        eval_dataset=eval_split,
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
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    import argparse
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
    p.add_argument("--lora_target_modules", type=str, default="")  # comma separated, empty to auto
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient_checkpointing", action="store_true")

    # W&B
    p.add_argument("--wandb_enable", action="store_true")  # explicit flag
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
        wandb_enable=bool(args.wandb_enable and args.wandb_project),
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

# qlora_glue_train.py
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
    BitsAndBytesConfig,
)

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

# ---- your project utils (unchanged) ----
from src.utils.data import load_glue_and_tokenizer
from src.utils.metrics import build_compute_metrics, get_best_metric_for_task
from src.utils.config import RunConfig, is_regression_task
from src.utils.wandb_utils import setup_wandb  # type: ignore


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


# ===================== LoRA helpers (model-agnostic) =====================

FAMILY_PATTERNS = {
    # LLaMA and friends
    "llama_like": ["q_proj", "k_proj", "v_proj", "o_proj"],
    # BERT / RoBERTa / XLM-R
    "bert_like": ["query", "key", "value", "dense"],
    # DeBERTa v2/v3
    "deberta_v2_v3": ["query_proj", "key_proj", "value_proj", "dense"],
    # GPT-2
    "gpt2_like": ["c_attn"],
    # GPT-NeoX / Falcon
    "neox_falcon": ["query_key_value", "dense"],
    # BLOOM
    "bloom": ["query_key_value", "dense"],
    # MPT
    "mpt": ["Wqkv", "out_proj"],
}

def _all_module_names(model: nn.Module) -> List[str]:
    return [n for n, _ in model.named_modules()]

def _find_hits(names: List[str], tokens: List[str]) -> List[str]:
    return [t for t in tokens if any(t in n for n in names)]

def guess_lora_target_modules(model: nn.Module):
    """
    Detect attention proj names across families.
    Return a list of substrings or the string 'all-linear' as a last resort.
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
            # For BERT/DeBERTa families, ensure 'dense' is included if present anywhere
            if fam in ("deberta_v2_v3", "bert_like"):
                if "dense" not in hits and any(
                    n.endswith("dense") or n.endswith(".dense") or ".output.dense" in n
                    for n in names
                ):
                    hits = sorted(set(hits + ["dense"]))
            return sorted(set(hits))

    # Generic regex pass
    generic = set()
    for n in names:
        if re.search(r"(q(uery)?|k(ey)?|v(alue)?)_?proj", n):
            if "q_proj" in n: generic.add("q_proj")
            if "k_proj" in n: generic.add("k_proj")
            if "v_proj" in n: generic.add("v_proj")
            if "query_proj" in n: generic.add("query_proj")
            if "key_proj" in n: generic.add("key_proj")
            if "value_proj" in n: generic.add("value_proj")
        if n.endswith("query"): generic.add("query")
        if n.endswith("key"):   generic.add("key")
        if n.endswith("value"): generic.add("value")
        if n.endswith("dense") or ".output.dense" in n: generic.add("dense")
    if generic:
        return sorted(generic)

    # Final fallback: all Linear layers (PEFT supports this)
    return "all-linear"

def pick_modules_to_save(model: nn.Module) -> List[str]:
    """
    Keep heads/poolers trainable and saved when present.
    Works across many architectures.
    """
    candidates = ["classifier", "score", "lm_head", "pooler.dense"]
    names = _all_module_names(model)
    found = []
    for c in candidates:
        if any(n.endswith(c) or n.endswith(f".{c}") for n in names):
            found.append(c)
    return found or ["classifier"]


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


# ===================== Args =====================

@dataclass
class QLoRAArgs:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    target_modules: Optional[List[str]] = None
    seed: int = 42
    gradient_checkpointing: bool = True
    quant_type: str = "nf4"  # "nf4" or "fp4"


# ===================== Train =====================

def train(cfg: RunConfig, qlora: QLoRAArgs):
    set_seed(qlora.seed)

    # Data + tokenizer
    encoded, tokenizer, collator, num_labels, _ = load_glue_and_tokenizer(
        cfg.task_name, cfg.model_name
    )
    # clamp huge model_max_length to something sane
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
    bf16_ok = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    compute_dtype = torch.bfloat16 if bf16_ok else torch.float16
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=qlora.quant_type,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # Load in 4-bit
    # device_map=None + .to("cuda") keeps everything on a single GPU (Kaggle-style env)
    base = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        config=hf_cfg,
        quantization_config=bnb_cfg,
        torch_dtype=compute_dtype,
        device_map=None,
    )
    if torch.cuda.is_available():
        base.to("cuda")

    # Keep head fp32 and on the right device
    keep_classifier_fp32(base)
    align_classifier_device(base)

    # Gradient checkpointing
    if qlora.gradient_checkpointing:
        try:
            base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            base.gradient_checkpointing_enable()

    # prepare k-bit
    try:
        base = prepare_model_for_kbit_training(
            base,
            use_gradient_checkpointing=qlora.gradient_checkpointing,
            output_embedding_layer_name="classifier",
        )
    except TypeError:
        base = prepare_model_for_kbit_training(
            base,
            use_gradient_checkpointing=qlora.gradient_checkpointing,
        )

    # after prepare, ensure head dtype/device are correct
    keep_classifier_fp32(base)
    align_classifier_device(base)

    # Target modules + modules_to_save
    target_modules = qlora.target_modules or guess_lora_target_modules(base)
    modules_to_save = pick_modules_to_save(base)

    lcfg = LoraConfig(
        r=qlora.r,
        lora_alpha=qlora.alpha,
        lora_dropout=qlora.dropout,
        bias=qlora.bias,
        task_type=TaskType.SEQ_CLS,
        target_modules=target_modules,    # may be list or "all-linear"
        modules_to_save=modules_to_save,  # keep head/pooler trainable
    )

    # Optional sanity check when target_modules is a list
    if isinstance(target_modules, list):
        hit_names = [
            n for n, m in base.named_modules()
            if isinstance(m, nn.Linear) and any(t in n for t in target_modules)
        ]
        if len(hit_names) == 0:
            example_names = [n for n, _ in list(base.named_modules())[:40]]
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
        evaluation_strategy="epoch",     # fix: use evaluation_strategy (not eval_strategy)
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best,
        logging_steps=cfg.logging_steps,
        report_to=report_targets,
        run_name=run_name,
        seed=qlora.seed,
        bf16=(compute_dtype == torch.bfloat16),
        optim="paged_adamw_8bit",
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

    # Save adapter + tokenizer
    trainer.model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    metrics = trainer.evaluate()
    with open(os.path.join(cfg.output_dir, "eval_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    import argparse
    p = argparse.ArgumentParser(description="GLUE QLoRA finetune")
    p.add_argument("--task_name", type=str, default="sst2")
    p.add_argument("--model_name", type=str, default="bert-base-uncased")
    p.add_argument("--output_dir", type=str, default="./outputs/qlora")
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--per_device_train_batch_size", type=int, default=32)
    p.add_argument("--per_device_eval_batch_size", type=int, default=64)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--logging_steps", type=int, default=50)

    # QLoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_bias", type=str, default="none")
    p.add_argument("--lora_target_modules", type=str, default="")  # comma-separated; empty => auto
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--quant_type", type=str, default="nf4")

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
        wandb_enable=bool(args.wandb_enable and args.wandb_project),
    )

    tmods = [s.strip() for s in args.lora_target_modules.split(",") if s.strip()] or None
    qargs = QLoRAArgs(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        bias=args.lora_bias,
        target_modules=tmods,
        seed=args.seed,
        gradient_checkpointing=args.gradient_checkpointing,
        quant_type=args.quant_type,
    )

    metrics = train(cfg, qargs)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

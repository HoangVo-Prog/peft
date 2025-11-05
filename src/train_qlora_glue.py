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
    BitsAndBytesConfig,
)

from src.utils.data import load_glue_and_tokenizer
from src.utils.metrics import build_compute_metrics, get_best_metric_for_task
from src.utils.config import RunConfig, is_regression_task
from src.utils.wandb_utils import setup_wandb  # type: ignore

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training


def _timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def guess_lora_target_modules(model):
    names = [n for n, _ in model.named_modules()]
    # LLaMA or Mistral
    if any(n.endswith("q_proj") for n in names):
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    # BERT or RoBERTa or DeBERTa
    if any(n.endswith("query") for n in names):
        return ["query", "key", "value"]
    # GPT2
    if any(n.endswith("c_attn") for n in names):
        return ["c_attn"]
    # Fallback
    return ["query", "key", "value"]


def keep_classifier_fp32(model):
    """
    Giữ nguyên classifier ở FP32.
    Nếu đã bị bitsandbytes chuyển thành Linear4bit thì thay lại bằng Linear thường.
    Hỗ trợ cấu trúc .dense và .out_proj của RoBERTa.
    """
    try:
        import bitsandbytes as bnb
        Linear4bit = getattr(bnb.nn, "Linear4bit", tuple())
    except Exception:
        Linear4bit = tuple()

    if hasattr(model, "classifier") and hasattr(model.classifier, "dense"):
        if isinstance(model.classifier.dense, Linear4bit):
            in_f = model.classifier.dense.in_features
            out_f = model.classifier.dense.out_features
            new_dense = torch.nn.Linear(in_f, out_f, bias=True).to(torch.float32)
            try:
                with torch.no_grad():
                    new_dense.weight.copy_(model.classifier.dense.weight.float())
                    if model.classifier.dense.bias is not None:
                        new_dense.bias.copy_(model.classifier.dense.bias.float())
            except Exception:
                pass
            model.classifier.dense = new_dense

    if hasattr(model, "classifier") and hasattr(model.classifier, "out_proj"):
        if isinstance(model.classifier.out_proj, Linear4bit):
            in_f = model.classifier.out_proj.in_features
            out_f = model.classifier.out_proj.out_features
            new_proj = torch.nn.Linear(in_f, out_f, bias=True).to(torch.float32)
            try:
                with torch.no_grad():
                    new_proj.weight.copy_(model.classifier.out_proj.weight.float())
                    if model.classifier.out_proj.bias is not None:
                        new_proj.bias.copy_(model.classifier.out_proj.bias.float())
            except Exception:
                pass
            model.classifier.out_proj = new_proj

    if hasattr(model, "classifier"):
        model.classifier.to(torch.float32)
        for p in model.classifier.parameters():
            p.requires_grad = True


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


def train(cfg: RunConfig, qlora: QLoRAArgs):
    set_seed(qlora.seed)

    encoded, tokenizer, collator, num_labels, _ = load_glue_and_tokenizer(
        cfg.task_name, cfg.model_name
    )

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

    base = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        config=hf_cfg,
        quantization_config=bnb_cfg,
        device_map="auto",
    )

    # Giữ nguyên head ở FP32, tránh 4-bit
    keep_classifier_fp32(base)

    # Bật gradient checkpointing nếu cần, và thêm kwargs để sạch cảnh báo
    if qlora.gradient_checkpointing:
        try:
            base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            base.gradient_checkpointing_enable()

    # Chuẩn bị k-bit, báo cho PEFT biết output head để giữ nguyên
    base = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=qlora.gradient_checkpointing,
        output_embedding_layer_name="classifier",
    )

    target_modules = qlora.target_modules or guess_lora_target_modules(base)

    lcfg = LoraConfig(
        r=qlora.r,
        lora_alpha=qlora.alpha,
        lora_dropout=qlora.dropout,
        bias=qlora.bias,
        task_type=TaskType.SEQ_CLS,
        target_modules=target_modules,     # không chứa "classifier"
        modules_to_save=["classifier"],    # lưu và train head thường
    )
    model = get_peft_model(base, lcfg)

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
        eval_strategy="epoch",
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

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation_mismatched"] if "validation_mismatched" in encoded else encoded["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        label_names=["labels"],
    )

    trainer.train()

    # Save adapter and tokenizer
    trainer.model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    metrics = trainer.evaluate()
    with open(os.path.join(cfg.output_dir, "eval_metrics.json"), "w") as f:
        import json
        json.dump(metrics, f, indent=2)

    return metrics


def main():
    import argparse, json
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
    p.add_argument("--lora_target_modules", type=str, default="")  # comma separated, empty to auto
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
        wandb_enable=bool(args.wandb_project),
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

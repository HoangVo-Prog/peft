#!/usr/bin/env python3
import argparse

from src.utils.config import RunConfig
from src.glueft.train import main


def parse_args():
    p = argparse.ArgumentParser(description="Train GLUE with HF Trainer")
    p.add_argument("--task", dest="task_name", type=str, default="sst2")
    p.add_argument("--model", dest="model_name", type=str, default="bert-base-uncased")
    p.add_argument("--output-dir", dest="output_dir", type=str, default="./outputs")
    p.add_argument("--epochs", dest="num_train_epochs", type=float, default=3.0)
    p.add_argument("--train-bsz", dest="per_device_train_batch_size", type=int, default=32)
    p.add_argument("--eval-bsz", dest="per_device_eval_batch_size", type=int, default=64)
    p.add_argument("--lr", dest="learning_rate", type=float, default=2e-5)
    p.add_argument("--wd", dest="weight_decay", type=float, default=0.01)
    p.add_argument("--warmup", dest="warmup_ratio", type=float, default=0.06)
    p.add_argument("--seed", dest="seed", type=int, default=42)
    p.add_argument("--save-strategy", dest="save_strategy", type=str, default="epoch")
    p.add_argument("--eval-strategy", dest="eval_strategy", type=str, default="epoch")
    p.add_argument("--save-total", dest="save_total_limit", type=int, default=2)
    p.add_argument("--fp16", dest="fp16", action="store_true")
    p.add_argument("--bf16", dest="bf16", action="store_true")
    p.add_argument("--no-wandb", dest="wandb_enable", action="store_false")
    p.add_argument("--wandb-project", dest="wandb_project", type=str, default="glue-ft")
    p.add_argument("--wandb-entity", dest="wandb_entity", type=str, default=None)
    p.add_argument("--wandb-name", dest="wandb_run_name", type=str, default=None)
    p.add_argument("--wandb-offline", dest="wandb_offline_fallback", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = RunConfig(**vars(args))
    main(cfg)
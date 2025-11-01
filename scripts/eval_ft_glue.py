#!/usr/bin/env python3
import argparse
import pathlib
import sys

# Allow running without installing the package
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from glueft.eval import evaluate_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a saved GLUE checkpoint")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--task", type=str, required=True)
    p.add_argument("--eval-bsz", type=int, default=64)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_checkpoint(args.checkpoint, args.task, batch_size=args.eval_bsz)
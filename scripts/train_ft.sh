#!/bin/bash

TASK=sst2
MODEL=roberta-large # roberta-base roberta-large microsoft/deberta-v3-base
EPOCHS=3
LR=2e-5
TRAIN_BSZ=32
EVAL_BSZ=64

# Tạo timestamp động
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Tạo thư mục output động
OUTPUT_DIR="./outputs/${TASK}-${MODEL}-${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

python -m src.glueft.train_ft_glue.py \
  --task "$TASK" \
  --model "$MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --train-bsz "$TRAIN_BSZ" \
  --eval-bsz "$EVAL_BSZ" \
  --fp16 \
  --no-wandb

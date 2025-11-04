#!/bin/bash

TASK=sst2
MODEL=bert-base-uncased
EPOCHS=3
LR=2e-5
TRAIN_BSZ=32
EVAL_BSZ=64
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
QUANT_TYPE=nf4

# Tạo timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Tạo thư mục output động
OUTPUT_DIR="./outputs/qlora-${TASK}-${MODEL}-${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

python -m src.train_qlora_glue \
  --task_name "$TASK" \
  --model_name "$MODEL" \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs "$EPOCHS" \
  --per_device_train_batch_size "$TRAIN_BSZ" \
  --per_device_eval_batch_size "$EVAL_BSZ" \
  --learning_rate "$LR" \
  --lora_r "$LORA_R" \
  --lora_alpha "$LORA_ALPHA" \
  --lora_dropout "$LORA_DROPOUT" \
  --quant_type "$QUANT_TYPE" \
  --gradient_checkpointing 
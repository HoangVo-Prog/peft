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

# Tên file log
LOG_FILE="${OUTPUT_DIR}/train_${TASK}_${MODEL}_${TIMESTAMP}.log"

# Chạy bằng nohup và lưu log
nohup python src/glueft/train_ft_glue.py \
  --task "$TASK" \
  --model "$MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --train-bsz "$TRAIN_BSZ" \
  --eval-bsz "$EVAL_BSZ" \
  --fp16 > "$LOG_FILE" 2>&1 &

# In thông báo ra terminal
echo "Running fine-tuning task: $TASK with model: $MODEL"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "Process running in background (PID: $!)"

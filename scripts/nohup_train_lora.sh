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
LORA_TARGET_MODULES="query,key,value"

# Tạo timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Tạo thư mục output động
OUTPUT_DIR="./outputs/lora-${TASK}-${MODEL}-${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# Tạo file log động
LOG_FILE="${OUTPUT_DIR}/train_lora_${TASK}_${MODEL}_${TIMESTAMP}.log"

# Chạy bằng nohup
nohup python -m src.gluelora.train_lora_glue \
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
  --lora_target_modules "$LORA_TARGET_MODULES" > "$LOG_FILE" 2>&1 &

# Hiển thị thông tin tiến trình
echo "Running LoRA fine-tuning for task: $TASK"
echo "Model: $MODEL"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "Process running in background (PID: $!)"

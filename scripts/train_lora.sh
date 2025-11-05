#!/bin/bash

#!/bin/bash

# Lấy TASK từ tham số dòng lệnh
TASK=$1

# Nếu không truyền TASK thì báo lỗi và dừng
if [ -z "$TASK" ]; then
  echo "Vui lòng truyền tên TASK khi chạy script."
  echo "Cách dùng: bash run_ft_models.sh <task_name>"
  echo "Ví dụ: bash run_ft_models.sh sst2"
  exit 1
fi

EPOCHS=3
LR=2e-5
TRAIN_BSZ=32
EVAL_BSZ=64
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="query,key,value"

# Danh sách model
MODELS=(
  "roberta-base"
  "roberta-large"
  "microsoft/deberta-v3-base"
)

# Loop qua từng model
for MODEL in "${MODELS[@]}"; do
  TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  OUTPUT_DIR="./outputs/lora-${TASK}-${MODEL}-${TIMESTAMP}"
  mkdir -p "$OUTPUT_DIR"

  echo "===== Training LoRA on $MODEL for task $TASK ====="
  python -m src.train_lora_glue \
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
    --lora_target_modules "$LORA_TARGET_MODULES" \
    --no-wandb
done

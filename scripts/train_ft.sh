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

# Danh sách model
MODELS=(
  "roberta-base"
  "roberta-large"
  "microsoft/deberta-v3-base"
)

# Loop qua từng model
for MODEL in "${MODELS[@]}"; do
  TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  OUTPUT_DIR="./outputs/${TASK}-${MODEL}-${TIMESTAMP}"
  mkdir -p "$OUTPUT_DIR"

  echo "===== Training $MODEL on $TASK ====="
  python -m src.train_ft_glue \
    --task "$TASK" \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --train-bsz "$TRAIN_BSZ" \
    --eval-bsz "$EVAL_BSZ" \
    --no-wandb
done

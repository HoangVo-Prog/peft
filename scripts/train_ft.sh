#!/bin/bash

# ==========================================
# run_ft_models.sh
# Fine-tune GLUE tasks with one or more models
# Usage:
#   bash run_ft_models.sh <task_name> [model_name]
# Example:
#   bash run_ft_models.sh sst2
#   bash run_ft_models.sh sst2 roberta-base
# ==========================================

# Nhận TASK và MODEL từ dòng lệnh
TASK=$1
MODEL_INPUT=$2

# Nếu không truyền TASK thì báo lỗi và dừng
if [ -z "$TASK" ]; then
  echo "Vui lòng truyền tên TASK khi chạy script."
  echo "Cách dùng: bash run_ft_models.sh <task_name> [model_name]"
  echo "Ví dụ: bash run_ft_models.sh sst2"
  echo "Hoặc: bash run_ft_models.sh sst2 roberta-base"
  exit 1
fi

# Siêu tham số
EPOCHS=3
LR=2e-5
TRAIN_BSZ=32
EVAL_BSZ=64

# Nếu người dùng không truyền model, chạy 3 model mặc định
if [ -z "$MODEL_INPUT" ]; then
  MODELS=(
    "roberta-base"
    "roberta-large"
    "microsoft/deberta-v3-base"
  )
else
  MODELS=("$MODEL_INPUT")
fi

# Lặp qua từng model
for MODEL in "${MODELS[@]}"; do
  TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  OUTPUT_DIR="./outputs/${TASK}-${MODEL//\//_}-${TIMESTAMP}"
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

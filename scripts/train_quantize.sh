#!/bin/bash

# ==========================================
# train_quantize.sh
# Usage:
#   bash train_quantize.sh <task_name> [model_name] [--nohup]
#   ALLOW_GPU_RESET=1 GPU_ID=0 bash train_quantize.sh sst2
# ==========================================

set -euo pipefail

TASK=${1:-}

if [ -z "$TASK" ]; then
  echo "Vui lòng truyền tên TASK khi chạy script."
  echo "Cách dùng: bash train_quantize.sh <task_name> [model_name] [--nohup]"
  echo "Ví dụ: bash train_quantize.sh sst2"
  echo "Hoặc:  bash train_quantize.sh sst2 roberta-base"
  exit 1
fi

# Xử lý MODEL_INPUT: chỉ nhận arg thứ 2 nếu không phải là flag
MODEL_INPUT=""
if [ $# -ge 2 ]; then
  if [[ "$2" != --* ]]; then
    MODEL_INPUT="$2"
  fi
fi

USE_NOHUP=0

# Parse flags (từ toàn bộ CLI args, để bắt được --nohup)
for arg in "$@"; do
  if [ "$arg" = "--nohup" ]; then
    USE_NOHUP=1
  fi
done

# Timestamp cho log tổng
GLOBAL_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Hyperparams
EPOCHS=3
LR=2e-5
TRAIN_BSZ=4
EVAL_BSZ=8
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
QUANT_TYPE=${QUANT_TYPE:-nf4}   # nf4 hoặc fp4

# Giảm phân mảnh bộ nhớ giữa các runs
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.6,expandable_segments:False"

# Chọn GPU
GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES="$GPU_ID"

OUTPUT_DIR="./outputs"
LOG_DIR = "./logs"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Bọc toàn bộ script bằng nohup một lần nếu có --nohup
if [ "$USE_NOHUP" -eq 1 ] && [ "${NOHUP_WRAPPED:-0}" -ne 1 ]; then
  MASTER_LOG="$OUTPUT_DIR/train_qlora_${TASK}_all_${GLOBAL_TIMESTAMP}.log"
  echo "[Info] Re exec script dưới nohup, log tổng: $MASTER_LOG"
  export NOHUP_WRAPPED=1
  nohup "$0" "$@" >"$MASTER_LOG" 2>&1 &
  echo "[Info] Đã gửi script QLoRA cho task $TASK chạy background với PID $!"
  exit 0
fi

# Danh sách model mặc định hoặc chỉ model được truyền
if [ -z "$MODEL_INPUT" ]; then
  MODELS=(
    "roberta-base"
    "roberta-large"
    "microsoft/deberta-v3-base"
  )
else
  MODELS=("$MODEL_INPUT")
fi

# ---------- Hàm dọn GPU giữa các model ----------

gpu_soft_clear() {
  python - <<'PY' || true
import sys
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        try:
            import cupy as cp
            cp.cuda.runtime.deviceReset()
        except Exception:
            pass
        try:
            import ctypes
            libcudart = ctypes.CDLL('libcudart.so')
            libcudart.cudaDeviceReset()
        except Exception:
            pass
        print("GPU soft clear done")
except Exception as e:
    print("GPU soft clear note:", e, file=sys.stderr)
PY
}

gpu_hard_reset_if_allowed() {
  if [ "${ALLOW_GPU_RESET:-0}" = "1" ]; then
    echo "[Info] Trying nvidia-smi --gpu-reset on GPU $GPU_ID"
    nvidia-smi -i "$GPU_ID" -pm 0 >/dev/null 2>&1 || true
    nvidia-smi -i "$GPU_ID" --gpu-reset || echo "[Warn] gpu-reset không khả dụng hoặc GPU còn tiến trình khác."
  fi
}

cleanup_between_models() {
  echo "[Cleanup] Đang dọn GPU và cache..."
  sync || true
  sleep 1
  gpu_soft_clear
  gpu_hard_reset_if_allowed
  echo "[Cleanup] Hoàn tất."
}

# ---------- Vòng lặp train QLoRA ----------

for MODEL in "${MODELS[@]}"; do
  TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  SAFE_MODEL="${MODEL//\//_}"
  OUTPUT_DIR="./outputs/qlora-${TASK}-${SAFE_MODEL}-${TIMESTAMP}"
  mkdir -p "$OUTPUT_DIR"

  echo "===== Training QLoRA on $MODEL for task $TASK ====="

  CMD=(
    python -m src.train_qlora_glue
    --task_name "$TASK"
    --model_name "$MODEL"
    --output_dir "$OUTPUT_DIR"
    --num_train_epochs "$EPOCHS"
    --per_device_train_batch_size "$TRAIN_BSZ"
    --per_device_eval_batch_size "$EVAL_BSZ"
    --learning_rate "$LR"
    --lora_r "$LORA_R"
    --lora_alpha "$LORA_ALPHA"
    --lora_dropout "$LORA_DROPOUT"
    --quant_type "$QUANT_TYPE"
    --gradient_checkpointing
    --no-wandb
  )

  # Log riêng cho từng model
  LOG_FILE="$LOG_DIR/train_qlora_${TASK}_${SAFE_MODEL}_${TIMESTAMP}.log"
  echo "[Info] Log riêng cho $MODEL: $LOG_FILE"

  # Chạy và vừa log ra file, vừa in ra stdout
  "${CMD[@]}" 2>&1 | tee "$LOG_FILE"

  # Cleanup giữa hai model
  cleanup_between_models
done

echo "Tất cả model QLoRA đã train xong."

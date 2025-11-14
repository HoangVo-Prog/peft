#!/bin/bash

# ==========================================
# train_lora.sh
# Usage:
#   bash train_lora.sh [model_name] "query,key,value,dense"
#   ALLOW_GPU_RESET=1 GPU_ID=0 bash train_lora.sh "query,key,value,dense"
# ==========================================

set -euo pipefail

MODEL_INPUT=${1:-}

# Hyperparams
EPOCHS=3
LR=2e-5
TRAIN_BSZ=32
EVAL_BSZ=64
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-query,key,value}"

# Nếu user không truyền env → dùng default list
if [ -z "${LORA_TARGET_MODULES:-}" ]; then
  LORA_TARGET_MODULES=("query" "key" "value")
else
  # Nếu user truyền env dạng string → convert thành array
  # Ví dụ: LORA_TARGET_MODULES="query key value dense"
  read -r -a LORA_TARGET_MODULES <<< "$LORA_TARGET_MODULES"
fi

if [ -z "${MODULES_TO_SAVE:-}" ]; then
  MODULES_TO_SAVE=("classifier")
else
  read -r -a MODULES_TO_SAVE <<< "$MODULES_TO_SAVE"
fi

# Giảm phân mảnh bộ nhớ giữa các runs
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.6,expandable_segments:False"

# Chọn GPU
GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES="$GPU_ID"

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

# ---------- Vòng lặp train LoRA ----------

for MODEL in "${MODELS[@]}"; do
  TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  SAFE_MODEL="${MODEL//\//_}"
  OUTPUT_DIR="./outputs"
  mkdir -p "$OUTPUT_DIR"

  echo "===== Training LoRA on $MODEL ====="
  python -m src.train_lora_glue \
    --all \
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

  cleanup_between_models
done

echo "Tất cả model LoRA đã chạy xong."

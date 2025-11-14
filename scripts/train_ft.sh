#!/bin/bash

# ==========================================
# train_ft.sh
# Usage:
#   bash run_ft_models.sh [model_name]
#   ALLOW_GPU_RESET=1 GPU_ID=0 bash run_ft_models.sh 
# ==========================================

set -euo pipefail

MODEL_INPUT=${1:-}


# Hyperparams
EPOCHS=3
LR=2e-5
TRAIN_BSZ=128
EVAL_BSZ=256

# Tinh chỉnh bộ nhớ allocator của PyTorch để giảm phân mảnh giữa các runs
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.6,expandable_segments:False"

# Chọn GPU nếu cần
GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# Danh sách model mặc định
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
  # Thu gom bộ nhớ đã giải phóng, hữu ích khi driver vẫn giữ context
  python - <<'PY' || true
import sys
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        # thử reset thiết bị qua CuPy nếu có
        try:
            import cupy as cp
            cp.cuda.runtime.deviceReset()
        except Exception:
            pass
        # thử gọi trực tiếp cudaDeviceReset qua libcudart
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
  # Chỉ dùng khi bạn biết chắc GPU rảnh và ALLOW_GPU_RESET=1
  if [ "${ALLOW_GPU_RESET:-0}" = "1" ]; then
    echo "[Info] Trying nvidia-smi --gpu-reset on GPU $GPU_ID"
    # Tắt persistence nếu cần, bỏ qua lỗi
    nvidia-smi -i "$GPU_ID" -pm 0 >/dev/null 2>&1 || true
    # Reset GPU, nếu driver và quyền cho phép
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

# ---------- Vòng lặp train ----------

for MODEL in "${MODELS[@]}"; do
  TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  SAFE_MODEL="${MODEL//\//_}"
  OUTPUT_DIR="./outputs"
  mkdir -p "$OUTPUT_DIR"

  echo "===== Training $MODEL====="
  python -m src.train_ft_glue \
    --all \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --train-bsz "$TRAIN_BSZ" \
    --eval-bsz "$EVAL_BSZ" \
    --no-wandb \

  # Dọn dẹp trước khi sang model tiếp theo
  cleanup_between_models
done

echo "Tất cả model đã chạy xong."

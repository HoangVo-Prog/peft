#!/bin/bash

set -euo pipefail

MODEL_INPUT=${1:-}

FP16_FLAG=""
BF16_FLAG=""

# Detect precision flags from CLI
for arg in "$@"; do
  if [ "$arg" = "--fp16" ]; then
    FP16_FLAG="--fp16"
  fi
  if [ "$arg" = "--bf16" ]; then   # note: train_ft_glue.py expects --bf16
    BF16_FLAG="--bf16"
  fi
done

# Hyperparams
EPOCHS=3
LR=2e-5

# Important:
# This is per-device batch size. On 2 GPUs, effective batch = TRAIN_BSZ * 2.
# If you want to keep global batch ~32 like your single GPU script, use 16 here.
TRAIN_BSZ=16
EVAL_BSZ=64

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.6,expandable_segments:False"

# torchrun settings
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
MASTER_PORT=${MASTER_PORT:-29500}

# We do NOT set CUDA_VISIBLE_DEVICES here so that torchrun can see all GPUs.
# If you want a custom set, export CUDA_VISIBLE_DEVICES before calling this script.

# If no model input or first arg is a flag, use default list
if [ -z "$MODEL_INPUT" ] || [[ "$MODEL_INPUT" == --* ]]; then
  MODELS=(
    "roberta-base"
    "roberta-large"
    "microsoft/deberta-v3-base"
  )
else
  MODELS=("$MODEL_INPUT")
fi

gpu_soft_clear() {
  python - <<'PY' || true
import sys
try:
    import torch
    if torch.cuda.is_available():
        try:
            n = torch.cuda.device_count()
        except Exception:
            n = 1
        for i in range(n):
            try:
                torch.cuda.set_device(i)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass
        try:
            import cupy as cp
            for i in range(n):
                try:
                    with cp.cuda.Device(i):
                        cp.cuda.runtime.deviceReset()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            import ctypes
            libcudart = ctypes.CDLL('libcudart.so')
            # cudaDeviceReset resets current device, we already looped above
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
    echo "[Info] Trying nvidia-smi --gpu-reset on all visible GPUs"
    # This usually is not allowed on Kaggle, but keep behavior optional
    nvidia-smi --gpu-reset || echo "[Warn] gpu-reset not supported or GPUs busy."
  fi
}

cleanup_between_models() {
  echo "[Cleanup] Clearing GPU and cache..."
  sync || true
  sleep 1
  gpu_soft_clear
  gpu_hard_reset_if_allowed
  echo "[Cleanup] Done."
}

for MODEL in "${MODELS[@]}"; do
  TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  SAFE_MODEL="${MODEL//\//_}"
  OUTPUT_DIR="./outputs"
  mkdir -p "$OUTPUT_DIR"

  echo "===== Training $MODEL on $NPROC_PER_NODE GPUs ====="

  torchrun \
    --nproc_per_node="$NPROC_PER_NODE" \
    --master_port="$MASTER_PORT" \
    -m src.train_ft_glue \
    --all \
    --model "$MODEL" \
    --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --train-bsz "$TRAIN_BSZ" \
    --eval-bsz "$EVAL_BSZ" \
    --no-wandb \
    --ddp \
    $FP16_FLAG \
    $BF16_FLAG

  cleanup_between_models
done

echo "All models finished."

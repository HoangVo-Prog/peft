#!/bin/bash

#----------------------------------------------------------------------------------------
# bash scripts/train_ft.sh [MODEL_NAME] [--fp16] [--bf16] [--nohup] [--tasks "TASK1 TASK2 ..."]
# tasks:
# - run 1: "cola sst2 mrpc qqp stsb"
# - run 2: "mnli qnli rte wnli"
#----------------------------------------------------------------------------------------

set -euo pipefail

MODEL_INPUT=${1:-}

FP16_FLAG=""
bf16_FLAG=""
USE_NOHUP=0
TASKS_ARG=""
EXPECT_TASKS_ARG=0

# Parse flags
for arg in "$@"; do
  # Nếu đang chờ value cho --tasks
  if [ "$EXPECT_TASKS_ARG" -eq 1 ]; then
    TASKS_ARG="$arg"
    EXPECT_TASKS_ARG=0
    continue
  fi

  if [ "$arg" = "--fp16" ]; then
    FP16_FLAG="--fp16"
  elif [ "$arg" = "--bf16" ]; then
    bf16_FLAG="--bf16"
  elif [ "$arg" = "--nohup" ]; then
    USE_NOHUP=1
  elif [ "$arg" = "--tasks" ]; then
    EXPECT_TASKS_ARG=1
  fi
done

# Timestamp cho log tổng
GLOBAL_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Hyperparams
EPOCHS=3
LR=2e-5
TRAIN_BSZ=4
EVAL_BSZ=8

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.6,expandable_segments:False"

GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES="$GPU_ID"

OUTPUT_DIR="./outputs"
LOG_DIR="./logs"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Bọc toàn bộ script bằng nohup một lần
if [ "$USE_NOHUP" -eq 1 ] && [ "${NOHUP_WRAPPED:-0}" -ne 1 ]; then
  MASTER_LOG="$LOG_DIR/train_ft_all_${GLOBAL_TIMESTAMP}.log"
  echo "[Info] Re exec script duoi nohup, log tong: $MASTER_LOG"
  export NOHUP_WRAPPED=1
  nohup "$0" "$@" >"$MASTER_LOG" 2>&1 &
  echo "[Info] Da gui script train_ft chay background voi PID $!"
  exit 0
fi

# Nếu model input rỗng hoặc là flag → default list
if [ -z "$MODEL_INPUT" ] || [[ "$MODEL_INPUT" == --* ]]; then
  MODELS=(
    "roberta-base"
    "roberta-large"
    "microsoft/deberta-v3-base"
  )
else
  MODELS=("$MODEL_INPUT")
fi

# ---------------- GPU cleanup utilities ----------------

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
    nvidia-smi -i "$GPU_ID" --gpu-reset || echo "[Warn] gpu-reset khong kha dung hoac GPU con tien trinh khac."
  fi
}

cleanup_between_models() {
  echo "[Cleanup] Dang don GPU va cache..."
  sync || true
  sleep 1
  gpu_soft_clear
  gpu_hard_reset_if_allowed
  echo "[Cleanup] Hoan tat."
}

# ---------------- Train loop ----------------

for MODEL in "${MODELS[@]}"; do
  TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
  SAFE_MODEL="${MODEL//\//_}"

  echo "===== Training $MODEL ====="

  CMD=(
    python -m src.train_ft_glue
    --model "$MODEL"
    --output-dir "$OUTPUT_DIR"
    --epochs "$EPOCHS"
    --lr "$LR"
    --train-bsz "$TRAIN_BSZ"
    --eval-bsz "$EVAL_BSZ"
    --no-wandb
  )

  # Nếu có TASKS_ARG thì dùng --tasks, không thì dùng --all
  if [ -n "$TASKS_ARG" ]; then
    CMD+=(--tasks "$TASKS_ARG")
  else
    CMD+=(--all)
  fi

  # Append flags nếu có
  if [ -n "$FP16_FLAG" ]; then
    CMD+=("$FP16_FLAG")
  fi
  if [ -n "$bf16_FLAG" ]; then
    CMD+=("$bf16_FLAG")
  fi

  # Log riêng cho từng model
  LOG_FILE="$LOG_DIR/train_ft_${SAFE_MODEL}_${TIMESTAMP}.log"

  # Vừa ghi log file riêng, vừa in ra stdout
  "${CMD[@]}" 2>&1 | tee "$LOG_FILE"

  # Dọn GPU giữa hai model
  cleanup_between_models
done

echo "Completed"

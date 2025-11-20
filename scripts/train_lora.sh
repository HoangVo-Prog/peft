#!/bin/bash

# ==========================================
# train_lora.sh
# Usage:
#   bash train_lora.sh [model_name]  --nohup --all
#   ALLOW_GPU_RESET=1 GPU_ID=0 bash train_lora.sh [model_name] --tasks "cola sst2 mrpc"
#   bash train_lora.sh "roberta-base" --tasks "sst2 mrpc"
# ==========================================

set -euo pipefail

# Lưu lại toàn bộ args gốc để dùng cho nohup
ORIG_ARGS=("$@")

MODEL_INPUT=""
FP16_FLAG=""
bf16_FLAG=""
USE_NOHUP=0

# Task control
USE_ALL_TASKS=1         # mặc định: dùng --all
TASKS_STR=""
TASKS=()

# Parse args
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fp16)
      FP16_FLAG="--fp16"
      shift
      ;;
    --bf16)
      bf16_FLAG="--bf16"
      shift
      ;;
    --nohup)
      USE_NOHUP=1
      shift
      ;;
    --tasks)
      if [[ $# -lt 2 ]]; then
        echo "[Error] --tasks cần một chuỗi task, ví dụ: --tasks \"cola sst2 mrpc\"" >&2
        exit 1
      fi
      USE_ALL_TASKS=0
      TASKS_STR="$2"
      shift 2
      ;;
    --all)
      USE_ALL_TASKS=1
      TASKS_STR=""
      shift
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

# Lấy model đầu tiên trong phần positional nếu có
if [ "${#POSITIONAL[@]}" -gt 0 ]; then
  MODEL_INPUT="${POSITIONAL[0]}"
fi

# Nếu dùng --tasks thì tách chuỗi thành mảng
if [ "$USE_ALL_TASKS" -eq 0 ]; then
  # TASKS_STR là chuỗi với task cách nhau bởi khoảng trắng
  # ví dụ: "cola sst2 mrpc"
  read -r -a TASKS <<< "$TASKS_STR"
fi

# Timestamp để dùng cho log tổng khi wrap nohup
GLOBAL_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# ====================================================
# Hyperparams
# ====================================================
EPOCHS=3
LR=2e-5
TRAIN_BSZ=4
EVAL_BSZ=8
LORA_R=16
LORA_ALPHA=32
LORA_DROPOUT=0.05

# target modules
if [ -z "${LORA_TARGET_MODULES:-}" ]; then
  LORA_TARGET_MODULES=("query" "key" "value")
else
  read -r -a LORA_TARGET_MODULES <<< "$LORA_TARGET_MODULES"
fi

if [ -z "${MODULES_TO_SAVE:-}" ]; then
  MODULES_TO_SAVE=("classifier")
else
  read -r -a MODULES_TO_SAVE <<< "$MODULES_TO_SAVE"
fi

export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.6,expandable_segments:False"

GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES="$GPU_ID"

OUTPUT_DIR="./outputs"
LOG_DIR="./logs"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# =========================
# Wrap toàn bộ script bằng nohup một lần
# =========================
if [ "$USE_NOHUP" -eq 1 ] && [ "${NOHUP_WRAPPED:-0}" -ne 1 ]; then
  MASTER_LOG="$LOG_DIR/train_lora_all_${GLOBAL_TIMESTAMP}.log"
  echo "[Info] Re exec script dưới nohup, log tổng: $MASTER_LOG"
  export NOHUP_WRAPPED=1
  # Chạy lại chính script, giữ nguyên toàn bộ args gốc
  nohup "$0" "${ORIG_ARGS[@]}" >"$MASTER_LOG" 2>&1 &
  echo "[Info] Đã gửi script train_lora.sh chạy background với PID $!"
  exit 0
fi

# Nếu arg đầu tiên là flag hoặc không có gì thì dùng default model list
if [ -z "$MODEL_INPUT" ] || [[ "$MODEL_INPUT" == --* ]]; then
  MODELS=(
    "roberta-base"
    "roberta-large"
    "microsoft/deberta-v3-base"
  )
else
  MODELS=("$MODEL_INPUT")
fi

# ---------- GPU cleanup utilities ----------

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

# ---------- Train Loop ----------

for MODEL in "${MODELS[@]}"; do
  echo "===== Training LoRA trên $MODEL với target ${LORA_TARGET_MODULES[*]} ====="

  # Base command chung cho mọi task
  BASE_CMD=(
    python -m src.train_lora_glue
    --model_name "$MODEL"
    --output_dir "$OUTPUT_DIR"
    --num_train_epochs "$EPOCHS"
    --per_device_train_batch_size "$TRAIN_BSZ"
    --per_device_eval_batch_size "$EVAL_BSZ"
    --learning_rate "$LR"
    --lora_r "$LORA_R"
    --lora_alpha "$LORA_ALPHA"
    --lora_dropout "$LORA_DROPOUT"
    --lora_target_modules "${LORA_TARGET_MODULES[@]}"
    --modules_to_save "${MODULES_TO_SAVE[@]}"
    --no-wandb
  )

  # Thêm flag fp16 / bf16 nếu có
  if [ -n "$FP16_FLAG" ]; then
    BASE_CMD+=("$FP16_FLAG")
  fi
  if [ -n "$bf16_FLAG" ]; then
    BASE_CMD+=("$bf16_FLAG")
  fi

  if [ "$USE_ALL_TASKS" -eq 1 ]; then
    # Chạy một lần với --all
    RUN_TS=$(date +"%Y%m%d_%H%M%S")
    CMD=("${BASE_CMD[@]}" --all)
    LOG_FILE="$LOG_DIR/train_lora_${MODEL//\//_}_all_${RUN_TS}.log"
    echo "[Info] Log cho model $MODEL (all tasks): $LOG_FILE"
    "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
  else
    # Loop qua từng task, thêm --task <task>
    for TASK in "${TASKS[@]}"; do
      RUN_TS=$(date +"%Y%m%d_%H%M%S")
      CMD=("${BASE_CMD[@]}" --task "$TASK")
      LOG_FILE="$LOG_DIR/train_lora_${MODEL//\//_}_${TASK}_${RUN_TS}.log"
      echo "[Info] Log cho model $MODEL task $TASK: $LOG_FILE"
      "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
    done
  fi

  # Dọn GPU giữa hai model
  cleanup_between_models
done

echo "Tất cả model LoRA đã train xong."

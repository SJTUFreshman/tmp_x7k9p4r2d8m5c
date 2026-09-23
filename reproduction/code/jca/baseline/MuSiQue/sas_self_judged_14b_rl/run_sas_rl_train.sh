#!/usr/bin/env bash
# Train the one-shot SAS policy from Qwen3-14B base with fresh LoRA.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/wangyuheng/jca}"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/qwen35/bin/python}"
PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-/data/wangyuheng}"
ACCELERATE_MODULE="${ACCELERATE_MODULE:-accelerate.commands.launch}"

ROLLOUT="${ROLLOUT:-$PROJECT_ROOT/baseline/MuSiQue/sas_self_judged_14b_rl/rl_data/sas_train.jsonl}"
MODEL="${MODEL:-/data/wangyuheng/models/Qwen3-14B}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/rl_runs/sas_self_judged_14b}"
NUM_GPUS="${NUM_GPUS:-8}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

KL_COEF="${KL_COEF:-0.2}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
LR="${LR:-1e-5}"
SEED="${SEED:-42}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
MAX_SEQ="${MAX_SEQ:-4096}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
REPORT_TO="${REPORT_TO:-tensorboard}"

cd "$PROJECT_ROOT"
[[ -f "$ROLLOUT" ]] || { echo "[fatal] rollout file not found: $ROLLOUT" >&2; exit 1; }
[[ -d "$MODEL" ]] || { echo "[fatal] model directory not found: $MODEL" >&2; exit 1; }
mkdir -p "$OUT_DIR"

LAUNCH_ARGS=(--num_processes "$NUM_GPUS" --mixed_precision "$MIXED_PRECISION")
if [[ "$NUM_GPUS" != "1" ]]; then
  LAUNCH_ARGS+=(--multi_gpu)
fi

PYTHONPATH="$PYTHONPATH_ROOT" "$PYTHON_BIN" -m "$ACCELERATE_MODULE" "${LAUNCH_ARGS[@]}" \
  scripts/rl_train.py \
  --agent SAS \
  --rollout "$ROLLOUT" \
  --out-dir "$OUT_DIR" \
  --model-name-or-path "$MODEL" \
  --kl-coef "$KL_COEF" \
  --num-epochs "$NUM_EPOCHS" \
  --learning-rate "$LR" \
  --seed "$SEED" \
  --per-device-batch-size "$PER_DEVICE_BATCH" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --max-seq-length "$MAX_SEQ" \
  --lora-rank "$LORA_RANK" \
  --lora-alpha "$LORA_ALPHA" \
  --lora-dropout "$LORA_DROPOUT" \
  --warmup-ratio "$WARMUP_RATIO" \
  --logging-steps "$LOGGING_STEPS" \
  --save-steps "$SAVE_STEPS" \
  --save-total-limit "$SAVE_TOTAL_LIMIT" \
  --report-to "$REPORT_TO" \
  --gradient-checkpointing \
  --bf16

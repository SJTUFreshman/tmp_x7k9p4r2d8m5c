#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/wangyuheng/jca}"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/qwen35/bin/python}"
PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-/data/wangyuheng}"
MODEL="${MODEL:-/data/wangyuheng/models/Qwen3-14B}"
RUN_ID="${RUN_ID:-gsm_sas14b_self_rl_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$PROJECT_ROOT/baseline/GSM-Hard/sas_self_judged_14b_rl/runs/$RUN_ID}"
ROLLOUT="${ROLLOUT:-$RUN_DIR/sas_train_judged.jsonl}"
TRAIN_OUT="${TRAIN_OUT:-$PROJECT_ROOT/rl_runs/gsm_sas_self_rl/$RUN_ID}"

NUM_GPUS="${NUM_GPUS:-8}"
KL_COEF="${KL_COEF:-0.1}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LR="${LR:-3e-6}"
SEED="${SEED:-42}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
MAX_SEQ="${MAX_SEQ:-8192}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
REPORT_TO="${REPORT_TO:-tensorboard}"

cd "$PROJECT_ROOT"
[[ -d "$MODEL" ]] || { echo "[fatal] missing model: $MODEL" >&2; exit 1; }
mkdir -p "$RUN_DIR" "$TRAIN_OUT"

echo "RUN_ID=$RUN_ID"
echo "ROLLOUT=$ROLLOUT"
echo "TRAIN_OUT=$TRAIN_OUT"
echo "epochs=$NUM_EPOCHS lr=$LR reference_coef=$KL_COEF max_seq=$MAX_SEQ"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[dry-run] training paths and configuration validated"
  exit 0
fi

[[ -s "$ROLLOUT" ]] || { echo "[fatal] missing judged rollout: $ROLLOUT" >&2; exit 1; }

launch_args=(--num_processes "$NUM_GPUS" --mixed_precision bf16)
if [[ "$NUM_GPUS" != "1" ]]; then
  launch_args+=(--multi_gpu)
fi

PYTHONPATH="$PYTHONPATH_ROOT" "$PYTHON_BIN" -m accelerate.commands.launch "${launch_args[@]}" \
  scripts/rl_train.py \
  --agent SAS \
  --rollout "$ROLLOUT" \
  --out-dir "$TRAIN_OUT" \
  --model-name-or-path "$MODEL" \
  --kl-coef "$KL_COEF" \
  --num-epochs "$NUM_EPOCHS" \
  --learning-rate "$LR" \
  --seed "$SEED" \
  --per-device-batch-size "$PER_DEVICE_BATCH" \
  --gradient-accumulation-steps "$GRAD_ACCUM" \
  --max-seq-length "$MAX_SEQ" \
  --warmup-ratio "$WARMUP_RATIO" \
  --lora-rank "$LORA_RANK" \
  --lora-alpha "$LORA_ALPHA" \
  --lora-dropout "$LORA_DROPOUT" \
  --logging-steps 10 \
  --save-steps 100 \
  --save-total-limit 3 \
  --report-to "$REPORT_TO" \
  --gradient-checkpointing \
  --bf16 \
  2>&1 | tee "$RUN_DIR/train.log"

[[ -s "$TRAIN_OUT/final/adapter_model.safetensors" ]] || {
  echo "[fatal] training finished without final adapter" >&2
  exit 1
}
echo "[done] adapter=$TRAIN_OUT/final"

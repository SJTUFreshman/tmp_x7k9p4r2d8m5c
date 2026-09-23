#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"
validate_common

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] ${NUM_GPUS}-GPU signed-RWR: rollout=$JUDGED_ROLLOUT out=$TRAIN_OUT grad_accum=$GRAD_ACCUM"
  exit 0
fi
[[ -s "$JUDGED_ROLLOUT" ]] || fatal "missing judged rollout: $JUDGED_ROLLOUT"
if [[ "$RESUME" == "1" && -s "$TRAIN_OUT/final/adapter_model.safetensors" ]]; then
  echo "[resume] training already complete: $TRAIN_OUT/final"
  exit 0
fi
if [[ "$RESUME" != "1" && -e "$TRAIN_OUT" ]]; then
  fatal "training output exists; set RESUME=1 or choose a new RUN_ID"
fi

mkdir -p "$TRAIN_OUT" "$RUN_DIR/03_train"
if [[ -n "$TRAIN_RESUME_CHECKPOINT" ]]; then
  [[ -s "$TRAIN_RESUME_CHECKPOINT/training_state.json" && \
     -s "$TRAIN_RESUME_CHECKPOINT/optimizer.bin" && \
     -s "$TRAIN_RESUME_CHECKPOINT/custom_checkpoint_0.pkl" && \
     -s "$TRAIN_RESUME_CHECKPOINT/policy_adapter/adapter_model.safetensors" ]] || \
    fatal "explicit resume checkpoint is incomplete: $TRAIN_RESUME_CHECKPOINT"
fi
if [[ "$RESUME" == "1" ]] && compgen -G "$TRAIN_OUT/checkpoint-*" >/dev/null; then
  complete_checkpoint=0
  for candidate in "$TRAIN_OUT"/checkpoint-*; do
    [[ -d "$candidate" && -s "$candidate/training_state.json" && \
       -s "$candidate/optimizer.bin" && -s "$candidate/custom_checkpoint_0.pkl" && \
       -s "$candidate/policy_adapter/adapter_model.safetensors" ]] || continue
    complete_checkpoint=1
    break
  done
  (( complete_checkpoint == 1 )) || fatal "checkpoint directories exist but none is complete: $TRAIN_OUT"
fi
wait_idle_gpus
train_args=(
  --agent SAS --rollout "$JUDGED_ROLLOUT" --reward-field reward
  --out-dir "$TRAIN_OUT" --model-name-or-path "$MODEL"
  --kl-coef "$KL_COEF" --num-epochs "$NUM_EPOCHS" --learning-rate "$LR"
  --seed "$GENERATION_SEED" --per-device-batch-size "$PER_DEVICE_BATCH"
  --gradient-accumulation-steps "$GRAD_ACCUM" --max-seq-length "$TRAIN_MAX_SEQ"
  --warmup-ratio "$WARMUP_RATIO" --lora-rank "$LORA_RANK"
  --lora-alpha "$LORA_ALPHA" --lora-dropout "$LORA_DROPOUT"
  --logging-steps 10 --save-steps "$TRAIN_SAVE_STEPS"
  --save-total-limit "$TRAIN_SAVE_TOTAL_LIMIT" --report-to tensorboard
  --gradient-checkpointing --bf16
)
if [[ "$RESUME" == "1" ]] && compgen -G "$TRAIN_OUT/checkpoint-*" >/dev/null; then
  train_args+=(--resume-from-checkpoint latest)
  [[ "$ALLOW_WORLD_SIZE_CHANGE_RESUME" == "1" ]] && train_args+=(--allow-world-size-change)
elif [[ "$RESUME" == "1" && -n "$TRAIN_RESUME_CHECKPOINT" ]]; then
  train_args+=(--resume-from-checkpoint "$TRAIN_RESUME_CHECKPOINT")
  [[ "$ALLOW_WORLD_SIZE_CHANGE_RESUME" == "1" ]] && train_args+=(--allow-world-size-change)
fi
if [[ "$MAX_TRAIN_STEPS" != "0" ]]; then
  train_args+=(--max-steps "$MAX_TRAIN_STEPS")
fi

launch_args=(--num_processes "$NUM_GPUS" --mixed_precision bf16)
[[ "$NUM_GPUS" == "1" ]] || launch_args+=(--multi_gpu)
env PYTHONPATH="$PYTHONPATH_ROOT" "$ACCELERATE" launch "${launch_args[@]}" \
  "$PROJECT_ROOT/scripts/rl_train.py" "${train_args[@]}" \
  2>&1 | tee -a "$RUN_DIR/03_train/run.log"

if [[ "$MAX_TRAIN_STEPS" != "0" ]]; then
  echo "[smoke] reached max training steps=$MAX_TRAIN_STEPS"
  exit 0
fi
[[ -s "$TRAIN_OUT/final/adapter_model.safetensors" ]] || fatal "training produced no final adapter"
echo "[done] adapter=$TRAIN_OUT/final"

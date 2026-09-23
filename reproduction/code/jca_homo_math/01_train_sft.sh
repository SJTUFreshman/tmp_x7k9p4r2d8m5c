#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=common.sh
source "${EXPERIMENT_ROOT}/common.sh"
validate_static_config

TRAINER="${EXPERIMENT_ROOT}/sft_train_resume.py"
require_file "$TRAINER"
if [[ "$DRY_RUN" != "1" ]]; then
  for agent in A1 A2 A3; do
    require_file "$SFT_DATA_DIR/$agent.jsonl"
  done
fi

latest_checkpoint() {
  local root="$1"
  find "$root" -maxdepth 1 -type d -name 'checkpoint-[0-9]*' -printf '%f\n' 2>/dev/null |
    awk -F- '$2 ~ /^[0-9]+$/ {print $2, $0}' |
    sort -n |
    tail -n 1 |
    awk -v root="$root" '{print root "/" $2}'
}

train_agent() {
  local agent="$1" model="$2"
  local data="$SFT_DATA_DIR/$agent.jsonl"
  local output="$SFT_RUN_ROOT/$agent"
  if adapter_complete "$output/final" && [[ -s "$output/final/train_summary.json" ]]; then
    log "[resume] SFT $agent already complete"
    return 0
  fi

  local resume_checkpoint=""
  if [[ -d "$output" ]]; then
    if [[ "$RESUME" != "1" ]]; then
      fatal "SFT output exists and RESUME=0: $output"
    fi
    resume_checkpoint="$(latest_checkpoint "$output")"
    if [[ -n "$resume_checkpoint" && -s "$resume_checkpoint/trainer_state.json" ]]; then
      log "resuming SFT $agent from $resume_checkpoint"
    elif [[ "$DRY_RUN" == "1" ]]; then
      log "[dry-run] would archive incomplete SFT directory: $output"
      resume_checkpoint=""
    else
      archive_partial_dir "$output"
      resume_checkpoint=""
    fi
  fi

  local precision_args=(--bf16)
  if [[ "$MIXED_PRECISION" == "fp16" ]]; then
    precision_args=(--no-bf16 --fp16)
  elif [[ "$MIXED_PRECISION" == "no" ]]; then
    precision_args=(--no-bf16)
  elif [[ "$MIXED_PRECISION" != "bf16" ]]; then
    fatal "MIXED_PRECISION must be bf16, fp16, or no"
  fi
  local launch_args=(--num_processes "$NUM_GPUS" --mixed_precision "$MIXED_PRECISION")
  [[ "$NUM_GPUS" == "1" ]] || launch_args+=(--multi_gpu)
  local command=(
    env "PYTHONPATH=$PYTHONPATH_ROOT" "$ACCELERATE" launch
    "${launch_args[@]}"
    "$TRAINER"
    --agent "$agent" --data "$data" --out-dir "$output"
    --model-name-or-path "$model"
    --eval-fraction "$SFT_EVAL_FRACTION" --split-group-by problem_id
    --seed "$SFT_SEED" --num-epochs "$SFT_NUM_EPOCHS"
    --learning-rate "$SFT_LR"
    --lora-rank "$SFT_LORA_RANK" --lora-alpha "$SFT_LORA_ALPHA"
    --lora-dropout "$SFT_LORA_DROPOUT"
    --per-device-train-batch-size "$SFT_PER_DEVICE_BATCH"
    --per-device-eval-batch-size "$SFT_PER_DEVICE_BATCH"
    --gradient-accumulation-steps "$SFT_GRAD_ACCUM"
    --max-seq-length "$SFT_MAX_SEQ"
    --max-tokenization-drop-ratio "$SFT_MAX_TOKENIZATION_DROP_RATIO"
    --logging-steps 10 --eval-steps "$SFT_EVAL_STEPS"
    --save-steps "$SFT_SAVE_STEPS" --save-total-limit "$SFT_SAVE_TOTAL_LIMIT"
    --early-stopping-patience 3 --no-load-best-model-at-end
    --num-eval-samples-to-generate 2 --dataloader-num-workers 0
    --report-to "$SFT_REPORT_TO" --gradient-checkpointing
    "${precision_args[@]}"
  )
  [[ -z "$resume_checkpoint" ]] || command+=(--resume-from-checkpoint "$resume_checkpoint")
  echo "SFT $agent"
  echo "  base: $model"
  echo "  data: $data"
  echo "  out:  $output"
  print_command "${command[@]}"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  wait_for_idle_gpus "MATH-specific SFT $agent"
  "${command[@]}"
  require_adapter "$output/final"
  require_file "$output/final/train_summary.json"
}

echo "MATH-specific role SFT training"
echo "  initialization: fresh base-model LoRA for every role (no GSM adapter)"
echo "  adapters:       $SFT_RUN_ROOT/{A1,A2,A3}/final"
echo "  resume:         latest complete Trainer checkpoint, otherwise clean restart"

acquire_gpu_lock
train_agent A1 "$MODEL_A1"
train_agent A2 "$MODEL_A2"
train_agent A3 "$MODEL_A3"
echo "[done] all MATH-specific SFT adapters are complete"

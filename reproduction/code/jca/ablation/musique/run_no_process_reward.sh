#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.env
source "${SCRIPT_DIR}/common.env"

MODE="${MODE:-all}"
DRY_RUN="${DRY_RUN:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_no_process_reward}"
RUN_ROOT="${RUN_ROOT:-${ABLATION_RUNS_ROOT}/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ABLATION_LOGS_ROOT}/${RUN_ID}}"
STAGE1_ROOT="${RUN_ROOT}/stage1"
STAGE2_ROOT="${RUN_ROOT}/stage2"
COMMAND_LOG="${LOG_ROOT}/commands.sh"
CONFIG_LOG="${LOG_ROOT}/config.env"

mkdir -p "$LOG_ROOT"
cd "$PROJECT_ROOT"

case "$MODE" in
  train|eval|all) ;;
  *) echo "ERROR: MODE must be train, eval, or all" >&2; exit 2 ;;
esac

require_file() { [[ -f "$2" ]] || { echo "ERROR: $1 not found: $2" >&2; exit 1; }; }
require_dir() { [[ -d "$2" ]] || { echo "ERROR: $1 not found: $2" >&2; exit 1; }; }

require_file "main rollout" "$MAIN_ROLLOUT"
require_file "A3 rollout" "$A3_ROLLOUT"
require_file "RWR launcher" "${PROJECT_ROOT}/scripts/rl_train.sh"
require_dir "A1 model" "$MODEL_A1"
require_dir "A2 model" "$MODEL_A2"
require_dir "A3 model" "$MODEL_A3"
require_dir "A1 SFT adapter" "$SFT_A1"
require_dir "A2 SFT adapter" "$SFT_A2"
require_dir "A3 SFT adapter" "$SFT_A3"
require_file "A1 SFT weights" "${SFT_A1}/adapter_model.safetensors"
require_file "A2 SFT weights" "${SFT_A2}/adapter_model.safetensors"
require_file "A3 SFT weights" "${SFT_A3}/adapter_model.safetensors"

cat >"$CONFIG_LOG" <<EOF
ABLATION=no_process_reward
BASELINE_REFERENCE=logs/mas_eval_concurrent/rl_sft_mas_0717_1130
RUN_ID=$RUN_ID
RUN_ROOT=$RUN_ROOT
MAIN_ROLLOUT=$MAIN_ROLLOUT
A3_ROLLOUT=$A3_ROLLOUT
REWARD_FIELD=task_reward
KL_COEF=$KL_COEF
NUM_EPOCHS=$NUM_EPOCHS
LR=$LR
SEED=$SEED
PER_DEVICE_BATCH=$PER_DEVICE_BATCH
GRAD_ACCUM=$GRAD_ACCUM
MAX_SEQ=$MAX_SEQ
WARMUP_RATIO=$WARMUP_RATIO
LORA_RANK=$LORA_RANK
LORA_ALPHA=$LORA_ALPHA
LORA_DROPOUT=$LORA_DROPOUT
NUM_GPUS=$NUM_GPUS
EVAL_LIMIT=$EVAL_LIMIT
EVAL_T_MAX=$EVAL_T_MAX
EVAL_MAX_NEW_TOKENS=$EVAL_MAX_NEW_TOKENS
EVAL_MAX_CONCURRENCY=$EVAL_MAX_CONCURRENCY
EOF
: >"$COMMAND_LOG"

run_command() {
  printf '%q ' "$@" | tee -a "$COMMAND_LOG"
  printf '\n' | tee -a "$COMMAND_LOG"
  [[ "$DRY_RUN" == "1" ]] || "$@"
}

train_stage() {
  local agents="$1" rollout="$2" output_root="$3" sft_a3="$4"
  run_command env \
    "PROJECT_ROOT=$PROJECT_ROOT" \
    "PYTHON_BIN=$PYTHON_BIN" \
    "PYTHONPATH_ROOT=$PYTHONPATH_ROOT" \
    "ACCELERATE=$ACCELERATE" \
    "ROLLOUT=$rollout" \
    "REWARD_FIELD=task_reward" \
    "AGENTS=$agents" \
    "SFT_A1=$SFT_A1" "SFT_A2=$SFT_A2" "SFT_A3=$sft_a3" \
    "MODEL_A1=$MODEL_A1" "MODEL_A2=$MODEL_A2" "MODEL_A3=$MODEL_A3" \
    "RL_RUNS_DIR=$output_root" \
    "KL_COEF=$KL_COEF" "NUM_EPOCHS=$NUM_EPOCHS" "LR=$LR" \
    "SEED=$SEED" "PER_DEVICE_BATCH=$PER_DEVICE_BATCH" \
    "GRAD_ACCUM=$GRAD_ACCUM" "MAX_SEQ=$MAX_SEQ" \
    "WARMUP_RATIO=$WARMUP_RATIO" \
    "LORA_RANK=$LORA_RANK" "LORA_ALPHA=$LORA_ALPHA" \
    "LORA_DROPOUT=$LORA_DROPOUT" "NUM_GPUS=$NUM_GPUS" \
    "MIXED_PRECISION=$MIXED_PRECISION" \
    "LOGGING_STEPS=$LOGGING_STEPS" "SAVE_STEPS=$SAVE_STEPS" \
    "SAVE_TOTAL_LIMIT=$SAVE_TOTAL_LIMIT" "REPORT_TO=$REPORT_TO" \
    bash scripts/rl_train.sh
}

run_eval() {
  local adapter_a1="${STAGE1_ROOT}/A1/final"
  local adapter_a2="${STAGE1_ROOT}/A2/final"
  local adapter_a3="${STAGE2_ROOT}/A3/final"
  if [[ "$DRY_RUN" != "1" ]]; then
    require_file "A1 final adapter" "${adapter_a1}/adapter_model.safetensors"
    require_file "A2 final adapter" "${adapter_a2}/adapter_model.safetensors"
    require_file "A3 final adapter" "${adapter_a3}/adapter_model.safetensors"
  fi
  run_command env \
    "PROJECT_ROOT=$PROJECT_ROOT" \
    "PYTHON_BIN=$PYTHON_BIN" \
    "PYTHONPATH_ROOT=$PYTHONPATH_ROOT" \
    "MODEL_A1=$MODEL_A1" "MODEL_A2=$MODEL_A2" "MODEL_A3=$MODEL_A3" \
    "ADAPTER_A1=$adapter_a1" "ADAPTER_A2=$adapter_a2" "ADAPTER_A3=$adapter_a3" \
    "RUN_ID=${RUN_ID}_eval" \
    "LOG_DIR=$LOG_ROOT/eval" \
    "SPLIT=$EVAL_SPLIT" "START=$EVAL_START" "LIMIT=$EVAL_LIMIT" \
    "T_MAX=$EVAL_T_MAX" "START_AGENT=$EVAL_START_AGENT" \
    "EVAL_PROTOCOL=old_sft" \
    "MAX_NEW_TOKENS=$EVAL_MAX_NEW_TOKENS" \
    "TEMPERATURE=$EVAL_TEMPERATURE" "TOP_P=$EVAL_TOP_P" \
    "MAX_MODEL_LEN=$EVAL_MAX_MODEL_LEN" \
    "LOG_RAW_CHARS=0" "ENABLE_THINKING=0" \
    "EXTRA_ARGS=--max-concurrency $EVAL_MAX_CONCURRENCY" \
    bash scripts/run_sft_lora_vllm_8gpu.sh
}

if [[ "$MODE" == "train" || "$MODE" == "all" ]]; then
  [[ "$DRY_RUN" == "1" || ! -e "$RUN_ROOT" ]] || {
    echo "ERROR: run directory already exists: $RUN_ROOT" >&2
    echo "Use a new RUN_ID, or MODE=eval to evaluate an existing run." >&2
    exit 1
  }
  train_stage "A1 A2 A3" "$MAIN_ROLLOUT" "$STAGE1_ROOT" "$SFT_A3"
  train_stage "A3" "$A3_ROLLOUT" "$STAGE2_ROOT" "${STAGE1_ROOT}/A3/final"
fi

if [[ "$MODE" == "eval" || "$MODE" == "all" ]]; then
  run_eval
fi

echo "No-process-reward ablation complete: $RUN_ID"

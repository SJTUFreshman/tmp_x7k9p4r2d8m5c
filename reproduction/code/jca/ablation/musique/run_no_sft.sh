#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.env
source "${SCRIPT_DIR}/common.env"

MODE="${MODE:-all}"
DRY_RUN="${DRY_RUN:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_no_sft}"
RUN_ROOT="${RUN_ROOT:-${ABLATION_RUNS_ROOT}/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ABLATION_LOGS_ROOT}/${RUN_ID}}"
DATA_ROOT="${DATA_ROOT:-${ABLATION_DATA_ROOT}/${RUN_ID}}"
STAGE1_ROOT="${RUN_ROOT}/stage1"
COMMAND_LOG="${LOG_ROOT}/commands.sh"
CONFIG_LOG="${LOG_ROOT}/config.env"

RAW_ROLLOUT="${DATA_ROOT}/base_rollout_raw.jsonl"
MAIN_SCORED_ROLLOUT="${DATA_ROOT}/base_rollout_qwen14b.jsonl"

mkdir -p "$LOG_ROOT" "$DATA_ROOT"
cd "$PROJECT_ROOT"

case "$MODE" in
  rollout|score|train|eval|all) ;;
  *) echo "ERROR: MODE must be rollout, score, train, eval, or all" >&2; exit 2 ;;
esac

require_file() { [[ -f "$2" ]] || { echo "ERROR: $1 not found: $2" >&2; exit 1; }; }
require_dir() { [[ -d "$2" ]] || { echo "ERROR: $1 not found: $2" >&2; exit 1; }; }

require_dir "A1 base model" "$MODEL_A1"
require_dir "A2 base model" "$MODEL_A2"
require_dir "A3 base model" "$MODEL_A3"
require_file "base rollout launcher" "${SCRIPT_DIR}/run_base_rollout_8gpu.sh"
require_file "Qwen14B rescore launcher" "${SCRIPT_DIR}/run_qwen14b_rescore.sh"
require_file "base RWR trainer" "${SCRIPT_DIR}/train_base_rwr.py"

cat >"$CONFIG_LOG" <<EOF
ABLATION=no_sft_end_to_end
BASELINE_REFERENCE=logs/mas_eval_concurrent/rl_sft_mas_0717_1130
RUN_ID=$RUN_ID
RUN_ROOT=$RUN_ROOT
DATA_ROOT=$DATA_ROOT
ROLLOUT_SPLIT=$ROLLOUT_SPLIT
ROLLOUT_START=$ROLLOUT_START
ROLLOUT_LIMIT=$ROLLOUT_LIMIT
ROLLOUT_T_MAX=$ROLLOUT_T_MAX
ROLLOUT_NUM_SAMPLES=$ROLLOUT_NUM_SAMPLES
ROLLOUT_MAX_NEW_TOKENS=$ROLLOUT_MAX_NEW_TOKENS
ROLLOUT_TEMPERATURE=$ROLLOUT_TEMPERATURE
ROLLOUT_TOP_P=$ROLLOUT_TOP_P
ROLLOUT_CONCURRENCY=$ROLLOUT_CONCURRENCY
REWARD_ALPHA=$REWARD_ALPHA
QWEN_JUDGE_MODEL=$QWEN_JUDGE_MODEL
QWEN_JUDGE_CONCURRENCY=$QWEN_JUDGE_CONCURRENCY
QWEN_JUDGE_MAX_TOKENS=$QWEN_JUDGE_MAX_TOKENS
RAW_ROLLOUT=$RAW_ROLLOUT
MAIN_SCORED_ROLLOUT=$MAIN_SCORED_ROLLOUT
KL_COEF=$KL_COEF
NUM_EPOCHS=$NUM_EPOCHS
LR=$LR
SEED=$SEED
PER_DEVICE_BATCH=$PER_DEVICE_BATCH
GRAD_ACCUM=$GRAD_ACCUM
MAX_SEQ=$MAX_SEQ
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

run_rollout() {
  run_command env \
    "PROJECT_ROOT=$PROJECT_ROOT" \
    "PYTHON_BIN=$PYTHON_BIN" \
    "PYTHONPATH_ROOT=$PYTHONPATH_ROOT" \
    "VLLM_PYTHON_BIN=$VLLM_PYTHON_BIN" \
    "VLLM_LD_LIBRARY_PATH=$VLLM_LD_LIBRARY_PATH" \
    "MODEL_A1=$MODEL_A1" "MODEL_A2=$MODEL_A2" "MODEL_A3=$MODEL_A3" \
    "DATA_DIR=$DATA_DIR" \
    "OUTPUT_PATH=$RAW_ROLLOUT" \
    "RUN_DIR=$LOG_ROOT/rollout" \
    "DRY_RUN=$DRY_RUN" \
    "ROLLOUT_SPLIT=$ROLLOUT_SPLIT" \
    "ROLLOUT_START=$ROLLOUT_START" "ROLLOUT_LIMIT=$ROLLOUT_LIMIT" \
    "ROLLOUT_T_MAX=$ROLLOUT_T_MAX" \
    "ROLLOUT_START_AGENT=$ROLLOUT_START_AGENT" \
    "ROLLOUT_NUM_SAMPLES=$ROLLOUT_NUM_SAMPLES" \
    "ROLLOUT_MAX_NEW_TOKENS=$ROLLOUT_MAX_NEW_TOKENS" \
    "ROLLOUT_TEMPERATURE=$ROLLOUT_TEMPERATURE" \
    "ROLLOUT_TOP_P=$ROLLOUT_TOP_P" \
    "ROLLOUT_CONCURRENCY=$ROLLOUT_CONCURRENCY" \
    "REWARD_ALPHA=$REWARD_ALPHA" \
    bash "${SCRIPT_DIR}/run_base_rollout_8gpu.sh"
}

run_score() {
  if [[ "$DRY_RUN" != "1" ]]; then
    require_file "raw base rollout" "$RAW_ROLLOUT"
  fi
  run_command env \
    "PROJECT_ROOT=$PROJECT_ROOT" \
    "PYTHON_BIN=$PYTHON_BIN" \
    "PYTHONPATH_ROOT=$PYTHONPATH_ROOT" \
    "VLLM_PYTHON_BIN=$VLLM_PYTHON_BIN" \
    "VLLM_LD_LIBRARY_PATH=$VLLM_LD_LIBRARY_PATH" \
    "DATA_DIR=$DATA_DIR" \
    "QWEN_JUDGE_MODEL=$QWEN_JUDGE_MODEL" \
    "QWEN_JUDGE_NAME=$QWEN_JUDGE_NAME" \
    "QWEN_JUDGE_GPUS=$QWEN_JUDGE_GPUS" \
    "QWEN_JUDGE_TP=$QWEN_JUDGE_TP" \
    "QWEN_JUDGE_PORT=$QWEN_JUDGE_PORT" \
    "QWEN_JUDGE_CONCURRENCY=$QWEN_JUDGE_CONCURRENCY" \
    "QWEN_JUDGE_MAX_TOKENS=$QWEN_JUDGE_MAX_TOKENS" \
    "QWEN_JUDGE_TEMPERATURE=$QWEN_JUDGE_TEMPERATURE" \
    "REWARD_ALPHA=$REWARD_ALPHA" \
    "INPUT_PATH=$RAW_ROLLOUT" \
    "OUTPUT_PATH=$MAIN_SCORED_ROLLOUT" \
    "RUN_DIR=$LOG_ROOT/rescore" \
    "DRY_RUN=$DRY_RUN" \
    bash "${SCRIPT_DIR}/run_qwen14b_rescore.sh"
}

train_agent() {
  local agent="$1" model="$2" rollout="$3" out_dir="$4" init_adapter="${5:-}"
  [[ "$DRY_RUN" == "1" || ! -e "$out_dir/final" ]] || {
    echo "ERROR: final adapter already exists: $out_dir/final" >&2
    exit 1
  }
  local launch_args=(--num_processes "$NUM_GPUS" --mixed_precision "$MIXED_PRECISION")
  [[ "$NUM_GPUS" == "1" ]] || launch_args+=(--multi_gpu)
  local command=(
    env "PYTHONPATH=${PYTHONPATH_ROOT}"
    "$ACCELERATE" launch
    "${launch_args[@]}"
    "${SCRIPT_DIR}/train_base_rwr.py"
    --agent "$agent"
    --rollout "$rollout"
    --reward-field reward
    --out-dir "$out_dir"
    --model-name-or-path "$model"
    --kl-coef "$KL_COEF"
    --num-epochs "$NUM_EPOCHS"
    --learning-rate "$LR"
    --weight-decay "$WEIGHT_DECAY"
    --warmup-ratio "$WARMUP_RATIO"
    --seed "$SEED"
    --per-device-batch-size "$PER_DEVICE_BATCH"
    --gradient-accumulation-steps "$GRAD_ACCUM"
    --max-seq-length "$MAX_SEQ"
    --lora-rank "$LORA_RANK"
    --lora-alpha "$LORA_ALPHA"
    --lora-dropout "$LORA_DROPOUT"
    --logging-steps "$LOGGING_STEPS"
    --save-steps "$SAVE_STEPS"
    --save-total-limit "$SAVE_TOTAL_LIMIT"
    --report-to "$REPORT_TO"
    --gradient-checkpointing
    --bf16
  )
  [[ -n "$init_adapter" ]] && command+=(--init-adapter "$init_adapter")
  run_command "${command[@]}"
}

run_train() {
  if [[ "$DRY_RUN" != "1" ]]; then
    require_file "main Qwen14B-scored rollout" "$MAIN_SCORED_ROLLOUT"
  fi
  train_agent A1 "$MODEL_A1" "$MAIN_SCORED_ROLLOUT" "${STAGE1_ROOT}/A1"
  train_agent A2 "$MODEL_A2" "$MAIN_SCORED_ROLLOUT" "${STAGE1_ROOT}/A2"
  train_agent A3 "$MODEL_A3" "$MAIN_SCORED_ROLLOUT" "${STAGE1_ROOT}/A3"
}

run_eval() {
  local adapter_a1="${STAGE1_ROOT}/A1/final"
  local adapter_a2="${STAGE1_ROOT}/A2/final"
  local adapter_a3="${STAGE1_ROOT}/A3/final"
  if [[ "$DRY_RUN" != "1" ]]; then
    require_file "A1 final adapter" "${adapter_a1}/adapter_model.safetensors"
    require_file "A2 final adapter" "${adapter_a2}/adapter_model.safetensors"
    require_file "A3 final adapter" "${adapter_a3}/adapter_model.safetensors"
  fi
  run_command env \
    "PROJECT_ROOT=$PROJECT_ROOT" "PYTHON_BIN=$PYTHON_BIN" \
    "PYTHONPATH_ROOT=$PYTHONPATH_ROOT" \
    "MODEL_A1=$MODEL_A1" "MODEL_A2=$MODEL_A2" "MODEL_A3=$MODEL_A3" \
    "ADAPTER_A1=$adapter_a1" "ADAPTER_A2=$adapter_a2" "ADAPTER_A3=$adapter_a3" \
    "RUN_ID=${RUN_ID}_eval" "LOG_DIR=$LOG_ROOT/eval" \
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

[[ "$MODE" == "rollout" || "$MODE" == "all" ]] && run_rollout
[[ "$MODE" == "score" || "$MODE" == "all" ]] && run_score
[[ "$MODE" == "train" || "$MODE" == "all" ]] && run_train
[[ "$MODE" == "eval" || "$MODE" == "all" ]] && run_eval

echo "End-to-end no-SFT ablation complete: $RUN_ID"

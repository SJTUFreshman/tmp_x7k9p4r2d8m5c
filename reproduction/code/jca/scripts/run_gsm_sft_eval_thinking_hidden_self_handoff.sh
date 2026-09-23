#!/usr/bin/env bash
# Evaluate GSM adapters with configurable seeded starts and role-batched loading.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/wangyuheng/jca}"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/qwen35/bin/python}"
PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-/data/wangyuheng}"
EVAL_LAUNCHER="${EVAL_LAUNCHER:-${PROJECT_ROOT}/gsm/scripts/run_gsm_vllm_role_batched_8gpu_thinking_hidden_self_handoff.sh}"

SFT_TAG="${SFT_TAG:-gsm_corr30_balanced_20260723_v2}"
RUN_ID="${RUN_ID:-${SFT_TAG}_fixed_a1_role_batched_dev132}"
DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/Math/data/GSM-HARD/splits/gsmhardv2_dev.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/outputs/gsm_eval/${RUN_ID}.jsonl}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/gsm_eval}"
DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-0}"
START="${START:-0}"
LIMIT="${LIMIT:-132}"
T_MAX="${T_MAX:-8}"
START_AGENT="${START_AGENT:-A1}"
START_AGENT_SEED="${START_AGENT_SEED:-42}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.95}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
API_TIMEOUT="${API_TIMEOUT:-120}"
GENERATION_SEED="${GENERATION_SEED:-}"
JSON_TRANSPORT="${JSON_TRANSPORT:-json_schema}"

MODEL_A1="${MODEL_A1:-/data/wangyuheng/models/Qwen3-1.7B}"
MODEL_A2="${MODEL_A2:-/data/wangyuheng/models/Qwen3-4B}"
MODEL_A3="${MODEL_A3:-/data/wangyuheng/models/Qwen3-8B}"
ADAPTER_A1="${ADAPTER_A1:-${PROJECT_ROOT}/sft_runs/${SFT_TAG}/A1/final}"
ADAPTER_A2="${ADAPTER_A2:-${PROJECT_ROOT}/sft_runs/${SFT_TAG}/A2/final}"
ADAPTER_A3="${ADAPTER_A3:-${PROJECT_ROOT}/sft_runs/${SFT_TAG}/A3/final}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-8}"
WAIT_FOR_IDLE_GPUS="${WAIT_FOR_IDLE_GPUS:-1}"

fatal() {
  echo "[fatal] $*" >&2
  exit 1
}

[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || fatal "RUN_ID contains unsupported characters"
[[ "$DRY_RUN" == "0" || "$DRY_RUN" == "1" ]] || fatal "DRY_RUN must be 0 or 1"
[[ "$RESUME" == "0" || "$RESUME" == "1" ]] || fatal "RESUME must be 0 or 1"
[[ "$START" =~ ^[0-9]+$ ]] || fatal "START must be non-negative"
[[ "$START_AGENT" == "A1" || "$START_AGENT" == "A2" || "$START_AGENT" == "A3" || "$START_AGENT" == "balanced" || "$START_AGENT" == "random" ]] || \
  fatal "START_AGENT must be A1, A2, A3, balanced, or random"
[[ "$START_AGENT_SEED" =~ ^[0-9]+$ ]] || fatal "START_AGENT_SEED must be non-negative"
for value_name in LIMIT T_MAX MAX_CONCURRENCY MAX_NEW_TOKENS MAX_MODEL_LEN API_TIMEOUT TENSOR_PARALLEL_SIZE DATA_PARALLEL_SIZE; do
  [[ "${!value_name}" =~ ^[1-9][0-9]*$ ]] || fatal "$value_name must be positive"
done
[[ "$ENABLE_THINKING" == "0" || "$ENABLE_THINKING" == "1" ]] || fatal "ENABLE_THINKING must be 0 or 1"
if [[ -n "$GENERATION_SEED" ]]; then
  [[ "$GENERATION_SEED" =~ ^[0-9]+$ ]] || fatal "GENERATION_SEED must be non-negative"
fi
[[ "$JSON_TRANSPORT" == "json_schema" || "$JSON_TRANSPORT" == "json_object" || "$JSON_TRANSPORT" == "none" ]] || \
  fatal "JSON_TRANSPORT must be json_schema, json_object, or none"
for path in "$PYTHON_BIN" "$EVAL_LAUNCHER" "$DATA_PATH" \
  "$MODEL_A1" "$MODEL_A2" "$MODEL_A3" \
  "$ADAPTER_A1" "$ADAPTER_A2" "$ADAPTER_A3"; do
  [[ -e "$path" ]] || fatal "required path not found: $path"
done
for adapter in "$ADAPTER_A1" "$ADAPTER_A2" "$ADAPTER_A3"; do
  [[ -s "$adapter/adapter_model.safetensors" ]] || fatal "missing adapter weights: $adapter"
done
[[ ! -e "$OUTPUT_PATH" ]] || fatal "evaluation output already exists: $OUTPUT_PATH"

EVAL_CMD=(
  env
  "PROJECT_ROOT=$PROJECT_ROOT" "PYTHONPATH_ROOT=$PYTHONPATH_ROOT"
  "PYTHON_BIN=$PYTHON_BIN" "DATA_PATH=$DATA_PATH"
  "START=$START" "LIMIT=$LIMIT" "T_MAX=$T_MAX"
  "START_AGENT=$START_AGENT" "START_AGENT_SEED=$START_AGENT_SEED"
  "MIN_AGENTS_BEFORE_STOP=1"
  "ENFORCE_COLLABORATION_POLICY=0"
  "MAX_CONCURRENCY=$MAX_CONCURRENCY" "MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
  "TEMPERATURE=$TEMPERATURE" "TOP_P=$TOP_P" "ENABLE_THINKING=$ENABLE_THINKING"
  "MAX_MODEL_LEN=$MAX_MODEL_LEN" "API_TIMEOUT=$API_TIMEOUT" "LOG_RAW_CHARS=0"
  "SFT_WARMUP_PROMPT=0"
  "MODEL_A1=$MODEL_A1" "MODEL_A2=$MODEL_A2" "MODEL_A3=$MODEL_A3"
  "ADAPTER_A1=$ADAPTER_A1" "ADAPTER_A2=$ADAPTER_A2" "ADAPTER_A3=$ADAPTER_A3"
  "GPU_IDS=$GPU_IDS" "TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE"
  "DATA_PARALLEL_SIZE=$DATA_PARALLEL_SIZE"
  "WAIT_FOR_IDLE_GPUS=$WAIT_FOR_IDLE_GPUS"
  "GENERATION_SEED=$GENERATION_SEED"
  "JSON_TRANSPORT=$JSON_TRANSPORT"
  "OUTPUT_PATH=$OUTPUT_PATH" "RUN_ID=$RUN_ID" "LOG_DIR=$LOG_DIR"
  "RESUME=$RESUME"
  "DRY_RUN=0"
  bash "$EVAL_LAUNCHER"
)

echo "GSM seeded-start role-batched evaluation"
echo "  sft_tag:     $SFT_TAG"
echo "  start_agent: $START_AGENT"
echo "  start_seed:  $START_AGENT_SEED"
echo "  role loop:   A1 -> A2 -> A3 on TP=1/DP=$DATA_PARALLEL_SIZE"
echo "  collaboration enforcement: off (model actions are evaluated as generated)"
echo "  temperature: $TEMPERATURE"
echo "  max_new_tokens: $MAX_NEW_TOKENS"
echo "  enable_thinking: $ENABLE_THINKING"
echo "  max_model_len: $MAX_MODEL_LEN"
echo "  api_timeout: $API_TIMEOUT"
echo "  JSON:       $JSON_TRANSPORT"
echo "  seed:       ${GENERATION_SEED:-none}"
echo "  concurrency: $MAX_CONCURRENCY"
echo "  output:      $OUTPUT_PATH"
printf '  command: '
printf '%q ' "${EVAL_CMD[@]}"
printf '\n'

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] evaluation was not started"
  exit 0
fi

exec "${EVAL_CMD[@]}"

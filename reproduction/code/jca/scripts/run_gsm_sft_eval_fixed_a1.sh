#!/usr/bin/env bash
# Evaluate existing GSM SFT adapters with the fixed-A1 experiment protocol.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/wangyuheng/jca}"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/qwen35/bin/python}"
PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-/data/wangyuheng}"
EVAL_LAUNCHER="${EVAL_LAUNCHER:-${PROJECT_ROOT}/gsm/scripts/run_gsm_vllm_8gpu.sh}"

SFT_TAG="${SFT_TAG:-gsm_corr30_balanced_20260723_v2}"
RUN_ID="${RUN_ID:-${SFT_TAG}_fixed_a1_dev132}"
DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/Math/data/GSM-HARD/splits/gsmhardv2_dev.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/outputs/gsm_eval/${RUN_ID}.jsonl}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/gsm_eval}"
DRY_RUN="${DRY_RUN:-0}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-64}"

MODEL_A1="${MODEL_A1:-/data/wangyuheng/models/Qwen3-1.7B}"
MODEL_A2="${MODEL_A2:-/data/wangyuheng/models/Qwen3-4B}"
MODEL_A3="${MODEL_A3:-/data/wangyuheng/models/Qwen3-8B}"
ADAPTER_A1="${ADAPTER_A1:-${PROJECT_ROOT}/sft_runs/${SFT_TAG}/A1/final}"
ADAPTER_A2="${ADAPTER_A2:-${PROJECT_ROOT}/sft_runs/${SFT_TAG}/A2/final}"
ADAPTER_A3="${ADAPTER_A3:-${PROJECT_ROOT}/sft_runs/${SFT_TAG}/A3/final}"

fatal() {
  echo "[fatal] $*" >&2
  exit 1
}

[[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || fatal "RUN_ID contains unsupported characters"
[[ "$DRY_RUN" == "0" || "$DRY_RUN" == "1" ]] || fatal "DRY_RUN must be 0 or 1"
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
  "PYTHON_BIN=$PYTHON_BIN" "MODE=sft_lora_mas"
  "DATA_PATH=$DATA_PATH" "START=0" "LIMIT=132" "T_MAX=8"
  "START_AGENT=A1" "MIN_AGENTS_BEFORE_STOP=1"
  "ENFORCE_COLLABORATION_POLICY=0"
  "MAX_CONCURRENCY=$MAX_CONCURRENCY" "MAX_NEW_TOKENS=1024"
  "TEMPERATURE=0.0" "TOP_P=0.95" "LOG_RAW_CHARS=0"
  "SFT_WARMUP_PROMPT=0"
  "MODEL_A1=$MODEL_A1" "MODEL_A2=$MODEL_A2" "MODEL_A3=$MODEL_A3"
  "ADAPTER_A1=$ADAPTER_A1" "ADAPTER_A2=$ADAPTER_A2" "ADAPTER_A3=$ADAPTER_A3"
  "OUTPUT_PATH=$OUTPUT_PATH" "RUN_ID=$RUN_ID" "LOG_DIR=$LOG_DIR"
  "KEEP_SERVERS=0"
  bash "$EVAL_LAUNCHER"
)

echo "GSM SFT fixed-A1 dev132 evaluation"
echo "  sft_tag:     $SFT_TAG"
echo "  start_agent: A1 (fixed)"
echo "  collaboration enforcement: off (model actions are evaluated as generated)"
echo "  temperature: 0.0"
echo "  output:      $OUTPUT_PATH"
printf '  command: '
printf '%q ' "${EVAL_CMD[@]}"
printf '\n'

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] evaluation was not started"
  exit 0
fi

exec "${EVAL_CMD[@]}"

#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/wangyuheng/jca}"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/qwen35/bin/python}"
PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-/data/wangyuheng}"
DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/Math/data/GSM-HARD/splits/gsmhardv2_dev.jsonl}"

TAG="${TAG:-gsm_judge_rl_v13_ablation_without_sft_warmup}"
TRAIN_RUNNER="${TRAIN_RUNNER:-${PROJECT_ROOT}/ablation/GSM-Hard/v13_without_sft_warmup/run.sh}"
EVAL_RUNNER="${EVAL_RUNNER:-${PROJECT_ROOT}/scripts/run_gsm_sft_eval_thinking_hidden_self_handoff.sh}"
TRAIN_RUN_DIR="${TRAIN_RUN_DIR:-${PROJECT_ROOT}/rl_runs/gsm_judge_rl/${TAG}}"

QUEUE_ID="${QUEUE_ID:-gsm_without_sft_warmup_$(date +%Y%m%d_%H%M%S)}"
QUEUE_ROOT="${QUEUE_ROOT:-${PROJECT_ROOT}/logs/gsm_hard_ablation_queue/${QUEUE_ID}}"
LOCK_ROOT="${LOCK_ROOT:-${PROJECT_ROOT}/logs/gsm_hard_all}"
LOCK_PATH="${LOCK_PATH:-${LOCK_ROOT}/.lock}"

EVAL_RUN_ID="${EVAL_RUN_ID:-${QUEUE_ID}_eval}"
EVAL_ROOT="${EVAL_ROOT:-${QUEUE_ROOT}/eval}"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-${EVAL_ROOT}/logs}"
EVAL_OUTPUT="${EVAL_OUTPUT:-${EVAL_ROOT}/results.jsonl}"
EVAL_STATE="${EVAL_STATE:-${EVAL_LOG_DIR}/${EVAL_RUN_ID}/trajectory_state.json}"

ADAPTER_A1="${TRAIN_RUN_DIR}/A1/final"
ADAPTER_A2="${TRAIN_RUN_DIR}/A2/final"
ADAPTER_A3="${TRAIN_RUN_DIR}/A3/final"

fatal() {
  echo "[fatal] $*" >&2
  exit 1
}

validate_adapters() {
  local adapter
  for adapter in "$ADAPTER_A1" "$ADAPTER_A2" "$ADAPTER_A3"; do
    [[ -s "$adapter/adapter_model.safetensors" ]] || fatal "missing adapter weights: $adapter"
    [[ -s "$adapter/adapter_config.json" ]] || fatal "missing adapter config: $adapter"
  done
  echo "[validate] A1/A2/A3 adapters complete"
}

validate_eval() {
  "$PYTHON_BIN" - "$EVAL_OUTPUT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"missing evaluation output: {path}")
records = []
for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip():
        continue
    try:
        records.append(json.loads(line))
    except Exception as exc:
        raise SystemExit(f"invalid JSON at line {line_number}: {exc}")
if len(records) != 132:
    raise SystemExit(f"expected 132 evaluation rows, got {len(records)}")
em = sum(float(record.get("em", 0.0) or 0.0) for record in records) / 132
f1 = sum(float(record.get("f1", 0.0) or 0.0) for record in records) / 132
print(f"[validate] rows=132 em={em:.4f} f1={f1:.4f}")
PY
}

[[ "$QUEUE_ID" =~ ^[A-Za-z0-9._-]+$ ]] || fatal "QUEUE_ID contains unsupported characters"
for required in "$PYTHON_BIN" "$DATA_PATH" "$TRAIN_RUNNER" "$EVAL_RUNNER"; do
  [[ -e "$required" ]] || fatal "required path not found: $required"
done

mkdir -p "$QUEUE_ROOT" "$LOCK_ROOT" "$EVAL_ROOT"
exec > >(tee -a "$QUEUE_ROOT/queue.log") 2>&1

echo "GSM-Hard without-SFT-warmup queued run"
echo "  queue_id:       $QUEUE_ID"
echo "  lock:           $LOCK_PATH"
echo "  train_output:   $TRAIN_RUN_DIR/{A1,A2,A3}/final"
echo "  eval_output:    $EVAL_OUTPUT"
echo "  without_sft:    enabled"

exec 9>"$LOCK_PATH"
echo "[$(date '+%F %T %Z')] waiting for the active suite to release the GPU lock"
flock 9
echo "[$(date '+%F %T %Z')] GPU lock acquired; starting without-SFT-warmup training"

if [[ -s "$ADAPTER_A1/adapter_model.safetensors" \
  && -s "$ADAPTER_A1/adapter_config.json" \
  && -s "$ADAPTER_A2/adapter_model.safetensors" \
  && -s "$ADAPTER_A2/adapter_config.json" \
  && -s "$ADAPTER_A3/adapter_model.safetensors" \
  && -s "$ADAPTER_A3/adapter_config.json" ]]; then
  echo "[resume] training adapters already complete"
else
  env \
    PROJECT_ROOT="$PROJECT_ROOT" \
    PYTHON_BIN="$PYTHON_BIN" \
    PYTHONPATH_ROOT="$PYTHONPATH_ROOT" \
    TAG="$TAG" \
    RUN_DIR="$TRAIN_RUN_DIR" \
    RESUME=1 \
    WAIT_FOR_GPUS=1 \
    bash "$TRAIN_RUNNER"
fi
validate_adapters

if [[ -s "$EVAL_OUTPUT" ]]; then
  if validate_eval; then
    echo "[resume] evaluation already complete"
    exit 0
  fi
fi

EVAL_RESUME=0
if [[ -s "$EVAL_STATE" ]]; then
  EVAL_RESUME=1
fi

echo "[$(date '+%F %T %Z')] starting without-SFT-warmup evaluation"
env \
  PROJECT_ROOT="$PROJECT_ROOT" \
  PYTHON_BIN="$PYTHON_BIN" \
  PYTHONPATH_ROOT="$PYTHONPATH_ROOT" \
  SFT_TAG="$TAG" \
  RUN_ID="$EVAL_RUN_ID" \
  LOG_DIR="$EVAL_LOG_DIR" \
  OUTPUT_PATH="$EVAL_OUTPUT" \
  STATE_PATH="$EVAL_STATE" \
  DATA_PATH="$DATA_PATH" \
  START=0 \
  LIMIT=132 \
  T_MAX=8 \
  START_AGENT=A1 \
  START_AGENT_SEED=42 \
  MAX_CONCURRENCY=64 \
  MAX_NEW_TOKENS=8192 \
  MAX_MODEL_LEN=40960 \
  TEMPERATURE=0.0 \
  TOP_P=0.95 \
  ENABLE_THINKING=1 \
  GENERATION_SEED=42 \
  JSON_TRANSPORT=none \
  API_TIMEOUT=900 \
  ADAPTER_A1="$ADAPTER_A1" \
  ADAPTER_A2="$ADAPTER_A2" \
  ADAPTER_A3="$ADAPTER_A3" \
  WAIT_FOR_IDLE_GPUS=1 \
  RESUME="$EVAL_RESUME" \
  PYTHONHASHSEED=0 \
  bash "$EVAL_RUNNER"

validate_eval
echo "Without-SFT-warmup training and evaluation complete: $QUEUE_ROOT"

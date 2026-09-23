#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/wangyuheng/jca}"
MODE="${MODE:-search}"
ENABLE_THINKING="${ENABLE_THINKING:-1}"
# Match the MuSiQue AFlow allocation:
#   A1/A2/A3 heterogeneous executor pool, plus a 14B search optimizer.
MODEL_A1="${MODEL_A1:-/data/wangyuheng/models/Qwen3-1.7B}"
MODEL_A2="${MODEL_A2:-/data/wangyuheng/models/Qwen3-4B}"
MODEL_A3="${MODEL_A3:-/data/wangyuheng/models/Qwen3-8B}"
MODEL_OPT="${MODEL_OPT:-/data/wangyuheng/models/Qwen3-14B}"
SERVED_A1="${SERVED_A1:-A1_base}"
SERVED_A2="${SERVED_A2:-A2_base}"
SERVED_A3="${SERVED_A3:-A3_base}"
SERVED_OPT="${SERVED_OPT:-Optimizer_14B}"
SPLIT="${SPLIT:-}"
if [[ -z "$SPLIT" ]]; then
  if [[ "$MODE" == "search" ]]; then
    SPLIT="train"
  else
    SPLIT="dev"
  fi
fi
if [[ -z "${DATA_PATH:-}" ]]; then
  if [[ "$SPLIT" == "train" ]]; then
    DATA_PATH="$PROJECT_ROOT/Math/data/GSM-HARD/splits/gsmhardv2_train.jsonl"
  else
    DATA_PATH="$PROJECT_ROOT/Math/data/GSM-HARD/splits/gsmhardv2_dev.jsonl"
  fi
fi

SEARCH_SPLIT="${SEARCH_SPLIT:-train}"
EVAL_SPLIT="${EVAL_SPLIT:-dev}"
SEARCH_DATA_PATH="${SEARCH_DATA_PATH:-${PROJECT_ROOT}/Math/data/GSM-HARD/splits/gsmhardv2_train.jsonl}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-${PROJECT_ROOT}/Math/data/GSM-HARD/splits/gsmhardv2_dev.jsonl}"

[[ "$ENABLE_THINKING" == 1 ]] || {
  echo "[fatal] GSM-Hard AFlow must run with ENABLE_THINKING=1" >&2
  exit 1
}
export PROJECT_ROOT SPLIT MODE ENABLE_THINKING
export MODEL_A1 MODEL_A2 MODEL_A3 MODEL_OPT
export SERVED_A1 SERVED_A2 SERVED_A3 SERVED_OPT
# Match the JCA evaluation budget. Qwen3 thinking can consume smaller executor
# budgets before emitting the JSON payload required by AFlow operators. The
# optimizer remains separate because it is used only by the one-time search.
export MAX_NEW_TOKENS_EXEC="${MAX_NEW_TOKENS_EXEC:-8192}"
export MAX_NEW_TOKENS_OPT="${MAX_NEW_TOKENS_OPT:-2048}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
export DATA_DIR="$DATA_PATH"
export LIMIT="${LIMIT:-132}"
export DEV_SIZE="${DEV_SIZE:-20}"
export RUNNER_SEARCH="${RUNNER_SEARCH:-baseline/GSM-Hard/AFlow/run_search_gsmhard.py}"
export RUNNER_EVAL="${RUNNER_EVAL:-baseline/GSM-Hard/AFlow/run_eval_gsmhard.py}"
export INITIAL_WORKFLOW="${INITIAL_WORKFLOW:-baseline/GSM-Hard/AFlow/workflows/round_00_initial.py}"
export WORKFLOW_FILE="${WORKFLOW_FILE:-baseline/GSM-Hard/AFlow/workflows/round_00_initial.py}"
export LOG_DIR="${LOG_DIR:-baseline/GSM-Hard/AFlow/logs}"
export EVAL_OUTPUT_PATH="${EVAL_OUTPUT_PATH:-}"

AFLOW_RELEASE_TIMEOUT="${AFLOW_RELEASE_TIMEOUT:-180}"
AFLOW_RELEASE_POLL_SECONDS="${AFLOW_RELEASE_POLL_SECONDS:-2}"

BASE_LAUNCHER="$PROJECT_ROOT/baseline/MuSiQue/AFlow/run_vllm_8gpu.sh"

case "$MODE" in
  search|eval|both) ;;
  *) echo "[fatal] MODE must be search, eval, or both" >&2; exit 1 ;;
esac
[[ "$AFLOW_RELEASE_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || {
  echo "[fatal] AFLOW_RELEASE_TIMEOUT must be positive" >&2
  exit 1
}
[[ "$AFLOW_RELEASE_POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || {
  echo "[fatal] AFLOW_RELEASE_POLL_SECONDS must be positive" >&2
  exit 1
}

wait_for_search_cleanup() {
  local deadline=$((SECONDS + AFLOW_RELEASE_TIMEOUT))
  local gpu_processes listeners
  echo "Waiting for AFlow search servers to release GPUs and ports..."
  while true; do
    gpu_processes=""
    listeners=""
    if command -v nvidia-smi >/dev/null 2>&1; then
      gpu_processes="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
    fi
    if command -v ss >/dev/null 2>&1; then
      listeners="$(
        ss -H -ltn 2>/dev/null \
          | awk -v exec_port="${EXEC_PORT:-8203}" -v opt_port="${OPT_PORT:-8204}" \
              '$4 ~ (":" exec_port "$") || $4 ~ (":" opt_port "$")'
      )"
    fi
    if [[ -z "${gpu_processes//[[:space:]]/}" && -z "${listeners//[[:space:]]/}" ]]; then
      echo "AFlow search resources released."
      return 0
    fi
    if (( SECONDS >= deadline )); then
      echo "[fatal] timed out waiting for AFlow search resource cleanup" >&2
      return 1
    fi
    sleep "$AFLOW_RELEASE_POLL_SECONDS"
  done
}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1: no vLLM server or runner will be started"
  echo "  A1: model=$MODEL_A1 served=$SERVED_A1"
  echo "  A2: model=$MODEL_A2 served=$SERVED_A2"
  echo "  A3: model=$MODEL_A3 served=$SERVED_A3"
  echo "  judge_agent=${JUDGE_AGENT:-A3} judge_seed=${JUDGE_AGENT_SEED:-42} (eval only)"
  echo "  OPT: model=$MODEL_OPT served=$SERVED_OPT (search only)"
  echo "  executor_max_new_tokens=$MAX_NEW_TOKENS_EXEC max_model_len=$MAX_MODEL_LEN"
  echo "  optimizer_max_new_tokens=$MAX_NEW_TOKENS_OPT (search only)"
  if [[ "$MODE" == "both" ]]; then
    echo "  search: split=$SEARCH_SPLIT data=$SEARCH_DATA_PATH size=$DEV_SIZE"
    echo "  eval:   split=$EVAL_SPLIT data=$EVAL_DATA_PATH limit=$LIMIT thinking=$ENABLE_THINKING"
  else
    echo "  mode=$MODE split=$SPLIT data=$DATA_DIR limit=$LIMIT thinking=$ENABLE_THINKING"
  fi
  exit 0
fi

if [[ "$MODE" == "both" ]]; then
  BASE_RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_gsmhard_aflow_both}"
  SEARCH_RUN_DIR="${SEARCH_RUN_DIR:-baseline/GSM-Hard/AFlow/search_runs/${BASE_RUN_ID}}"
  PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/qwen35/bin/python}"

  MODE=search SPLIT="$SEARCH_SPLIT" DATA_DIR="$SEARCH_DATA_PATH" \
    RUN_ID="${BASE_RUN_ID}_search" SEARCH_RUN_DIR="$SEARCH_RUN_DIR" \
    KEEP_SERVERS=0 bash "$BASE_LAUNCHER"

  wait_for_search_cleanup

  BEST_WORKFLOW="$($PYTHON_BIN - "$SEARCH_RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
nodes = json.loads((root / "state.json").read_text(encoding="utf-8"))
valid = [node for node in nodes if node.get("parse_ok")]
if not valid:
    raise SystemExit("No valid AFlow workflow was produced.")
best = max(valid, key=lambda node: (node["dev_em"], node["dev_f1"]))
print(best["source_file"])
PY
)"

  MODE=eval SPLIT="$EVAL_SPLIT" DATA_DIR="$EVAL_DATA_PATH" \
    RUN_ID="${BASE_RUN_ID}_eval" WORKFLOW_FILE="$BEST_WORKFLOW" \
    SEARCH_RUN_DIR="$SEARCH_RUN_DIR" EVAL_OUTPUT_PATH="$EVAL_OUTPUT_PATH" \
    bash "$BASE_LAUNCHER"
  exit 0
fi

exec bash "$BASE_LAUNCHER"

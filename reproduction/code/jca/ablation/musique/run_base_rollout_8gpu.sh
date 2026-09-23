#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.env
source "${SCRIPT_DIR}/common.env"

OUTPUT_PATH="${OUTPUT_PATH:?OUTPUT_PATH is required}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
DRY_RUN="${DRY_RUN:-0}"
HOST="${HOST:-127.0.0.1}"
A1_PORT="${A1_PORT:-8201}"
A2_PORT="${A2_PORT:-8202}"
A3_PORT="${A3_PORT:-8203}"
A1_GPUS="${A1_GPUS:-0,1}"
A2_GPUS="${A2_GPUS:-2,3}"
A3_GPUS="${A3_GPUS:-4,5,6,7}"
A1_TP="${A1_TP:-2}"
A2_TP="${A2_TP:-2}"
A3_TP="${A3_TP:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-900}"

mkdir -p "$RUN_DIR" "$(dirname "$OUTPUT_PATH")"
cd "$PROJECT_ROOT"

SERVER_PIDS=()
cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  for pid in "${SERVER_PIDS[@]:-}"; do
    [[ -n "$pid" ]] || continue
    kill -TERM "-$pid" >/dev/null 2>&1 || kill -TERM "$pid" >/dev/null 2>&1 || true
  done
  if [[ "${#SERVER_PIDS[@]}" -gt 0 ]]; then
    sleep 5
    for pid in "${SERVER_PIDS[@]}"; do
      kill -KILL "-$pid" >/dev/null 2>&1 || kill -KILL "$pid" >/dev/null 2>&1 || true
    done
  fi
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

require_dir() { [[ -d "$2" ]] || { echo "ERROR: $1 not found: $2" >&2; exit 1; }; }
count_gpus() { awk -F',' '{print NF}' <<<"$1"; }

server_ready() {
  "$PYTHON_BIN" - "$1" <<'PY' >/dev/null 2>&1
import json, sys, urllib.request
with urllib.request.urlopen(sys.argv[1], timeout=3) as response:
    payload = json.loads(response.read().decode("utf-8"))
if "data" not in payload:
    raise SystemExit(1)
PY
}

wait_for_model() {
  local url="$1" model="$2" timeout="$3" pid="$4"
  local start now
  start="$(date +%s)"
  while true; do
    kill -0 "$pid" >/dev/null 2>&1 || {
      echo "ERROR: server for $model exited before readiness" >&2
      return 1
    }
    if "$PYTHON_BIN" - "$url" "$model" <<'PY' >/dev/null 2>&1
import json, sys, urllib.request
with urllib.request.urlopen(sys.argv[1], timeout=5) as response:
    models = [item.get("id") for item in json.loads(response.read())["data"]]
raise SystemExit(0 if sys.argv[2] in models else 1)
PY
    then
      echo "$model ready at $url"
      return 0
    fi
    now="$(date +%s)"
    (( now - start < timeout )) || { echo "ERROR: timeout waiting for $model" >&2; return 1; }
    sleep 5
  done
}

start_server() {
  local agent="$1" model="$2" port="$3" gpus="$4" tp="$5" log="$6"
  [[ "$(count_gpus "$gpus")" == "$tp" ]] || {
    echo "ERROR: $agent GPU count does not match TP=$tp" >&2
    exit 1
  }
  local command=(
    "$VLLM_PYTHON_BIN" -m vllm.entrypoints.openai.api_server
    --host "$HOST" --port "$port"
    --model "$model" --served-model-name "$agent"
    --tensor-parallel-size "$tp"
    --dtype bfloat16
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$MAX_MODEL_LEN"
    --trust-remote-code
    --enforce-eager
  )
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpus"
  printf '%q ' "${command[@]}"
  printf '\n'
  if [[ "$DRY_RUN" != "1" ]]; then
    setsid env \
      CUDA_VISIBLE_DEVICES="$gpus" \
      LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
      "${command[@]}" >"$log" 2>&1 &
    SERVER_PIDS+=("$!")
  fi
}

require_dir "A1 base model" "$MODEL_A1"
require_dir "A2 base model" "$MODEL_A2"
require_dir "A3 base model" "$MODEL_A3"
[[ -f "${SCRIPT_DIR}/rollout_without_judge.py" ]] || {
  echo "ERROR: judge-free rollout wrapper missing" >&2
  exit 1
}

if [[ "$DRY_RUN" != "1" ]]; then
  for port in "$A1_PORT" "$A2_PORT" "$A3_PORT"; do
    if server_ready "http://${HOST}:${port}/v1/models"; then
      echo "ERROR: port $port already has an OpenAI-compatible server" >&2
      exit 1
    fi
  done
fi

start_server A1 "$MODEL_A1" "$A1_PORT" "$A1_GPUS" "$A1_TP" "$RUN_DIR/A1_server.log"
start_server A2 "$MODEL_A2" "$A2_PORT" "$A2_GPUS" "$A2_TP" "$RUN_DIR/A2_server.log"
start_server A3 "$MODEL_A3" "$A3_PORT" "$A3_GPUS" "$A3_TP" "$RUN_DIR/A3_server.log"

if [[ "$DRY_RUN" != "1" ]]; then
  wait_for_model "http://${HOST}:${A1_PORT}/v1/models" A1 "$SERVER_WAIT_TIMEOUT" "${SERVER_PIDS[0]}"
  wait_for_model "http://${HOST}:${A2_PORT}/v1/models" A2 "$SERVER_WAIT_TIMEOUT" "${SERVER_PIDS[1]}"
  wait_for_model "http://${HOST}:${A3_PORT}/v1/models" A3 "$SERVER_WAIT_TIMEOUT" "${SERVER_PIDS[2]}"
fi

ROLLOUT_COMMAND=(
  "$PYTHON_BIN" -u "${SCRIPT_DIR}/rollout_without_judge.py"
  --split "$ROLLOUT_SPLIT"
  --start "$ROLLOUT_START"
  --limit "$ROLLOUT_LIMIT"
  --data-dir "$DATA_DIR"
  --t-max "$ROLLOUT_T_MAX"
  --start-agent "$ROLLOUT_START_AGENT"
  --api-base-a1 "http://${HOST}:${A1_PORT}/v1"
  --api-base-a2 "http://${HOST}:${A2_PORT}/v1"
  --api-base-a3 "http://${HOST}:${A3_PORT}/v1"
  --api-model-a1 A1 --api-model-a2 A2 --api-model-a3 A3
  --api-timeout 900
  --max-new-tokens "$ROLLOUT_MAX_NEW_TOKENS"
  --temperature "$ROLLOUT_TEMPERATURE"
  --top-p "$ROLLOUT_TOP_P"
  --alpha "$REWARD_ALPHA"
  --num-rollouts "$ROLLOUT_NUM_SAMPLES"
  --rollout-concurrency "$ROLLOUT_CONCURRENCY"
  --output "$OUTPUT_PATH"
  --resume
)
printf 'PYTHONPATH=%q ' "$PYTHONPATH_ROOT"
printf '%q ' "${ROLLOUT_COMMAND[@]}"
printf '\n'
if [[ "$DRY_RUN" != "1" ]]; then
  PYTHONPATH="$PYTHONPATH_ROOT" "${ROLLOUT_COMMAND[@]}"
fi

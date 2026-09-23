#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.env
source "${SCRIPT_DIR}/common.env"

INPUT_PATH="${INPUT_PATH:?INPUT_PATH is required}"
OUTPUT_PATH="${OUTPUT_PATH:?OUTPUT_PATH is required}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
DRY_RUN="${DRY_RUN:-0}"
HOST="${HOST:-127.0.0.1}"

mkdir -p "$RUN_DIR" "$(dirname "$OUTPUT_PATH")"
cd "$PROJECT_ROOT"

SERVER_PID=""
cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  if [[ -n "$SERVER_PID" ]]; then
    kill -TERM "-$SERVER_PID" >/dev/null 2>&1 || kill -TERM "$SERVER_PID" >/dev/null 2>&1 || true
    sleep 5
    kill -KILL "-$SERVER_PID" >/dev/null 2>&1 || kill -KILL "$SERVER_PID" >/dev/null 2>&1 || true
  fi
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

if [[ "$DRY_RUN" != "1" ]]; then
  [[ -f "$INPUT_PATH" ]] || { echo "ERROR: rollout missing: $INPUT_PATH" >&2; exit 1; }
  [[ -d "$QWEN_JUDGE_MODEL" ]] || { echo "ERROR: judge model missing: $QWEN_JUDGE_MODEL" >&2; exit 1; }
fi

SERVER_COMMAND=(
  "$VLLM_PYTHON_BIN" -m vllm.entrypoints.openai.api_server
  --host "$HOST" --port "$QWEN_JUDGE_PORT"
  --model "$QWEN_JUDGE_MODEL"
  --served-model-name "$QWEN_JUDGE_NAME"
  --tensor-parallel-size "$QWEN_JUDGE_TP"
  --dtype bfloat16
  --gpu-memory-utilization 0.80
  --max-model-len 8192
  --trust-remote-code
  --enforce-eager
)
printf 'CUDA_VISIBLE_DEVICES=%q ' "$QWEN_JUDGE_GPUS"
printf '%q ' "${SERVER_COMMAND[@]}"
printf '\n'

if [[ "$DRY_RUN" != "1" ]]; then
  setsid env \
    CUDA_VISIBLE_DEVICES="$QWEN_JUDGE_GPUS" \
    LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "${SERVER_COMMAND[@]}" >"$RUN_DIR/qwen14b_server.log" 2>&1 &
  SERVER_PID="$!"

  "$PYTHON_BIN" - "http://${HOST}:${QWEN_JUDGE_PORT}/v1/models" "$QWEN_JUDGE_NAME" <<'PY'
import json, sys, time, urllib.request
url, model = sys.argv[1], sys.argv[2]
deadline = time.time() + 900
last_error = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            models = [item.get("id") for item in json.loads(response.read())["data"]]
        if model in models:
            print(f"{model} ready")
            raise SystemExit(0)
        last_error = f"{model!r} not listed by {url}"
    except Exception as exc:
        last_error = repr(exc)
    time.sleep(5)
raise SystemExit(f"timed out waiting for judge server: {last_error}")
PY
fi

RESCORE_COMMAND=(
  env "PYTHONPATH=$PYTHONPATH_ROOT"
  "$PYTHON_BIN" -u "${SCRIPT_DIR}/rescore_with_prompt.py"
  --judge-prompt original
  --input "$INPUT_PATH"
  --output "$OUTPUT_PATH"
  --split train
  --data-dir "$DATA_DIR"
  --judge-api-base "http://${HOST}:${QWEN_JUDGE_PORT}/v1"
  --judge-model "$QWEN_JUDGE_NAME"
  --alpha "$REWARD_ALPHA"
  --concurrency "$QWEN_JUDGE_CONCURRENCY"
  --temperature "$QWEN_JUDGE_TEMPERATURE"
  --max-tokens "$QWEN_JUDGE_MAX_TOKENS"
  --timeout 300
  --retries 2
  --resume
)
printf '%q ' "${RESCORE_COMMAND[@]}"
printf '\n'
if [[ "$DRY_RUN" != "1" ]]; then
  "${RESCORE_COMMAND[@]}"
fi

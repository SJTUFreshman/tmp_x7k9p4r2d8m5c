#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/wangyuheng/jca}"
PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-/data/wangyuheng}"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/qwen35/bin/python}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-/data/conda_envs/deep_research/bin/python}"
VLLM_LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH:-/data/conda_envs/deep_research/lib}"
PYTHON_COMPAT_DIR="${PYTHON_COMPAT_DIR:-${PROJECT_ROOT}/scripts/gsm_judge_rl/python_compat}"
COLLECTOR="${COLLECTOR:-}"

DATA_ROOT="${DATA_ROOT:-/data/wangyuheng/jca/Math/data/MATH}"
START="${START:-0}"
LIMIT="${LIMIT:-3160}"
MAX_MISSING="${MAX_MISSING:-0}"
PLAN_OFFSET="${PLAN_OFFSET:-0}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/outputs/math_correction_sft_rollout.jsonl}"
RUN_ID="${RUN_ID:-math_correction_sft_$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/math_correction_sft_rollout}"
RUN_DIR="${RUN_DIR:-${LOG_DIR}/${RUN_ID}}"

MODEL_8B="${MODEL_8B:-/data/wangyuheng/models/Qwen3-8B}"
HOST="${HOST:-127.0.0.1}"
A1_PORT="${A1_PORT:-8401}"
A2_PORT="${A2_PORT:-8402}"
A3_PORT="${A3_PORT:-8403}"
A1_GPUS="${A1_GPUS:-0,1}"
A2_GPUS="${A2_GPUS:-2,3}"
A3_GPUS="${A3_GPUS:-4,5,6,7}"
A1_TP="${A1_TP:-2}"
A2_TP="${A2_TP:-2}"
A3_TP="${A3_TP:-4}"

MAX_CONCURRENCY="${MAX_CONCURRENCY:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
TEMPERATURE="${TEMPERATURE:-0.2}"
TOP_P="${TOP_P:-0.95}"
MAX_STEP_RETRIES="${MAX_STEP_RETRIES:-2}"
MAX_TRAJECTORY_ATTEMPTS="${MAX_TRAJECTORY_ATTEMPTS:-20}"
MAX_VERIFIER_SIMILARITY="${MAX_VERIFIER_SIMILARITY:-0.85}"
GENERATION_SEED="${GENERATION_SEED:-42}"
API_TIMEOUT="${API_TIMEOUT:-600}"

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-48}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-900}"
RESUME="${RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"

fatal() {
  echo "[fatal] $*" >&2
  exit 1
}

count_gpus() {
  awk -F, '{print NF}' <<<"$1"
}

[[ -n "$COLLECTOR" && -f "$COLLECTOR" ]] || fatal "COLLECTOR must name the bundled collector"
[[ -d "$DATA_ROOT" ]] || fatal "MATH data root missing: $DATA_ROOT"
[[ -d "$MODEL_8B" ]] || fatal "8B rollout model missing: $MODEL_8B"
[[ -x "$PYTHON_BIN" && -x "$VLLM_PYTHON_BIN" ]] || fatal "Python executable missing"
[[ "$RESUME" == "0" || "$RESUME" == "1" ]] || fatal "RESUME must be 0 or 1"
[[ "$DRY_RUN" == "0" || "$DRY_RUN" == "1" ]] || fatal "DRY_RUN must be 0 or 1"
[[ "$MAX_MISSING" =~ ^[0-9]+$ && "$MAX_MISSING" -lt "$LIMIT" ]] || \
  fatal "MAX_MISSING must be a non-negative integer smaller than LIMIT"
[[ "$(count_gpus "$A1_GPUS")" == "$A1_TP" ]] || fatal "A1 GPU/TP mismatch"
[[ "$(count_gpus "$A2_GPUS")" == "$A2_TP" ]] || fatal "A2 GPU/TP mismatch"
[[ "$(count_gpus "$A3_GPUS")" == "$A3_TP" ]] || fatal "A3 GPU/TP mismatch"

mkdir -p "$RUN_DIR" "$(dirname "$OUTPUT_PATH")"
LOG_PATH="${RUN_DIR}/run.log"
A1_LOG="${RUN_DIR}/A1_server.log"
A2_LOG="${RUN_DIR}/A2_server.log"
A3_LOG="${RUN_DIR}/A3_server.log"
SERVER_PIDS=()

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if ((${#SERVER_PIDS[@]})); then
    for pid in "${SERVER_PIDS[@]}"; do
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    done
    for _ in {1..30}; do
      local alive=0
      for pid in "${SERVER_PIDS[@]}"; do
        kill -0 "$pid" 2>/dev/null && alive=1
      done
      (( alive == 0 )) && break
      sleep 1
    done
    for pid in "${SERVER_PIDS[@]}"; do
      kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    done
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

wait_for_server() {
  local name="$1" port="$2"
  "$PYTHON_BIN" - "$name" "http://${HOST}:${port}/v1/models" "$SERVER_WAIT_TIMEOUT" <<'PY'
import json
import sys
import time
import urllib.request

name, url, timeout_text = sys.argv[1:]
deadline = time.time() + float(timeout_text)
last_error = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            models = [item["id"] for item in json.loads(response.read())["data"]]
        print(f"{name} ready: {models}")
        raise SystemExit(0)
    except SystemExit:
        raise
    except Exception as exc:
        last_error = exc
    time.sleep(5)
raise SystemExit(f"{name} server timeout: {last_error}")
PY
}

start_server() {
  local agent="$1" port="$2" gpus="$3" tp="$4" log_path="$5"
  local command=(
    "$VLLM_PYTHON_BIN" -m vllm.entrypoints.openai.api_server
    --host "$HOST" --port "$port"
    --model "$MODEL_8B" --served-model-name "$agent"
    --tensor-parallel-size "$tp" --dtype bfloat16
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --enable-prefix-caching --enable-chunked-prefill
    --trust-remote-code --no-enable-log-requests
  )
  echo "starting $agent on GPUs=$gpus TP=$tp port=$port"
  setsid env \
    CUDA_VISIBLE_DEVICES="$gpus" \
    LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    PYTHONPATH="${PYTHON_COMPAT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${command[@]}" >"$log_path" 2>&1 &
  SERVER_PIDS+=("$!")
}

COLLECT_CMD=(
  env "PYTHONPATH=$PYTHONPATH_ROOT" "$PYTHON_BIN" -u "$COLLECTOR"
  --data-root "$DATA_ROOT" --start "$START" --limit "$LIMIT"
  --max-missing "$MAX_MISSING"
  --plan-offset "$PLAN_OFFSET"
  --api-base-a1 "http://${HOST}:${A1_PORT}/v1" --api-model-a1 A1
  --api-base-a2 "http://${HOST}:${A2_PORT}/v1" --api-model-a2 A2
  --api-base-a3 "http://${HOST}:${A3_PORT}/v1" --api-model-a3 A3
  --api-key EMPTY --api-timeout "$API_TIMEOUT"
  --max-new-tokens "$MAX_NEW_TOKENS" --temperature "$TEMPERATURE"
  --top-p "$TOP_P" --max-concurrency "$MAX_CONCURRENCY"
  --max-step-retries "$MAX_STEP_RETRIES"
  --max-trajectory-attempts "$MAX_TRAJECTORY_ATTEMPTS"
  --max-verifier-similarity "$MAX_VERIFIER_SIMILARITY"
  --seed "$GENERATION_SEED" --output "$OUTPUT_PATH"
)
[[ "$RESUME" == "1" ]] && COLLECT_CMD+=(--resume) || COLLECT_CMD+=(--no-resume)

echo "MATH fixed-A1 correction SFT rollout"
echo "  source:       MATH train"
echo "  teachers:     Qwen3-8B x 3, base weights"
echo "  plans:        $LIMIT from offset $PLAN_OFFSET"
echo "  output:       $OUTPUT_PATH"
printf '  command:      '
printf '%q ' "${COLLECT_CMD[@]}"
printf '\n'

if [[ "$DRY_RUN" == "1" ]]; then
  "${COLLECT_CMD[@]}" --plan-only
  exit 0
fi

exec > >(tee -a "$LOG_PATH") 2>&1
start_server A1 "$A1_PORT" "$A1_GPUS" "$A1_TP" "$A1_LOG"
start_server A2 "$A2_PORT" "$A2_GPUS" "$A2_TP" "$A2_LOG"
start_server A3 "$A3_PORT" "$A3_GPUS" "$A3_TP" "$A3_LOG"
wait_for_server A1 "$A1_PORT"
wait_for_server A2 "$A2_PORT"
wait_for_server A3 "$A3_PORT"
"${COLLECT_CMD[@]}"
"$PYTHON_BIN" "$COLLECTOR" --data-root "$DATA_ROOT" --limit "$LIMIT" \
  --plan-offset "$PLAN_OFFSET" --max-missing "$MAX_MISSING" \
  --output "$OUTPUT_PATH" --validate-existing

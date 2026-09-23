#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/data/wangyuheng/jca}"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/qwen35/bin/python}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-/data/conda_envs/deep_research/bin/python}"
VLLM_LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH:-/data/conda_envs/deep_research/lib}"
PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-/data/wangyuheng}"
PYTHON_COMPAT_DIR="${PYTHON_COMPAT_DIR:-${PROJECT_ROOT}/scripts/gsm_judge_rl/python_compat}"

MODE="${MODE:-both}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/Math/data/MATH}"
INITIAL_WORKFLOW="${INITIAL_WORKFLOW:-${SCRIPT_DIR}/workflows/round_00_initial.py}"
WORKFLOW_FILE="${WORKFLOW_FILE:-$INITIAL_WORKFLOW}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20}"
SEARCH_SIZE="${SEARCH_SIZE:-20}"
SEARCH_SEED="${SEARCH_SEED:-20260810}"
RNG_SEED="${RNG_SEED:-20260810}"
GENERATION_SEED="${GENERATION_SEED:-42}"
EVAL_START="${EVAL_START:-0}"
EVAL_LIMIT="${EVAL_LIMIT:-5000}"
PROBLEM_IDS_FILE="${PROBLEM_IDS_FILE:-}"

MODEL_A1="${MODEL_A1:-/data/wangyuheng/models/Qwen3-1.7B}"
MODEL_A2="${MODEL_A2:-/data/wangyuheng/models/Qwen3-4B}"
MODEL_A3="${MODEL_A3:-/data/wangyuheng/models/Qwen3-8B}"
MODEL_OPT="${MODEL_OPT:-/data/wangyuheng/models/Qwen3-14B}"
SERVED_A1="${SERVED_A1:-A1_base}"
SERVED_A2="${SERVED_A2:-A2_base}"
SERVED_A3="${SERVED_A3:-A3_base}"
SERVED_OPT="${SERVED_OPT:-Optimizer_14B}"

MAX_NEW_TOKENS_EXEC="${MAX_NEW_TOKENS_EXEC:-8192}"
MAX_NEW_TOKENS_OPT="${MAX_NEW_TOKENS_OPT:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
TEMPERATURE_EXEC="${TEMPERATURE_EXEC:-0.7}"
TEMPERATURE_OPT="${TEMPERATURE_OPT:-0.8}"
TOP_P="${TOP_P:-0.95}"
REQUEST_RETRIES="${REQUEST_RETRIES:-4}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
REQUIRE_THINKING="${REQUIRE_THINKING:-0}"
API_TIMEOUT="${API_TIMEOUT:-900}"
SEARCH_CONCURRENCY="${SEARCH_CONCURRENCY:-20}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-64}"

HOST="${HOST:-127.0.0.1}"
A1_PORT="${A1_PORT:-8201}"
A2_PORT="${A2_PORT:-8202}"
A3_PORT="${A3_PORT:-8203}"
OPT_PORT="${OPT_PORT:-8204}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-900}"
SERVER_STOP_TIMEOUT="${SERVER_STOP_TIMEOUT:-45}"

RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"
WAIT_FOR_IDLE_GPUS="${WAIT_FOR_IDLE_GPUS:-1}"
GPU_IDLE_TIMEOUT_SECONDS="${GPU_IDLE_TIMEOUT_SECONDS:-86400}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-10}"
GPU_IDLE_CHECKS="${GPU_IDLE_CHECKS:-2}"
GPU_IDLE_MAX_MEMORY_MIB="${GPU_IDLE_MAX_MEMORY_MIB:-2048}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_math_aflow}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/math_baselines/aflow}"
RUN_DIR="${RUN_DIR:-${LOG_ROOT}/${RUN_ID}}"
SEARCH_RUN_DIR="${SEARCH_RUN_DIR:-${RUN_DIR}/search}"
EVAL_OUTPUT="${EVAL_OUTPUT:-${RUN_DIR}/trajectories.jsonl}"
EVAL_SUMMARY="${EVAL_SUMMARY:-${RUN_DIR}/summary.json}"
EVAL_PROGRESS="${EVAL_PROGRESS:-${RUN_DIR}/eval_progress.jsonl}"
LOG_PATH="${LOG_PATH:-${RUN_DIR}/run.log}"
RUNTIME_ROOT="${RUNTIME_ROOT:-$(mktemp -d /data/tmp/jca_math_aflow.XXXXXX)}"

source "$PROJECT_ROOT/scripts/math_gpu_guard.sh"

SEARCH_RUNNER="${SCRIPT_DIR}/run_search.py"
EVAL_RUNNER="${SCRIPT_DIR}/run_eval.py"

fatal() { echo "[fatal] $*" >&2; exit 1; }
runner() { env PYTHONPATH="$PYTHONPATH_ROOT" "$PYTHON_BIN" -u "$@"; }

validate() {
  case "$MODE" in search|eval|both) ;; *) fatal "MODE must be search, eval, or both";; esac
  [[ "$MAX_ITERATIONS" == 20 && "$SEARCH_SIZE" == 20 ]] || fatal "canonical search requires 20 proposals x 20 tasks"
  [[ "$MAX_NEW_TOKENS_EXEC" == 8192 && "$MAX_NEW_TOKENS_OPT" == 2048 && "$MAX_MODEL_LEN" == 40960 ]] || fatal "token budgets must be exec=8192 opt=2048 context=40960"
  [[ "$TEMPERATURE_EXEC" == 0.7 && "$TEMPERATURE_OPT" == 0.8 ]] || fatal "temperatures must be exec=0.7 opt=0.8"
  [[ "$ENABLE_THINKING" =~ ^[01]$ && "$REQUIRE_THINKING" =~ ^[01]$ ]] || fatal "thinking flags must be 0 or 1"
  [[ "$REQUIRE_THINKING" == 0 || "$ENABLE_THINKING" == 1 ]] || fatal "REQUIRE_THINKING=1 requires ENABLE_THINKING=1"
  [[ -x "$PYTHON_BIN" && -x "$VLLM_PYTHON_BIN" ]] || fatal "python executable missing"
  [[ -d "$DATA_ROOT" && -f "$SEARCH_RUNNER" && -f "$EVAL_RUNNER" && -f "$INITIAL_WORKFLOW" ]] || fatal "data, runner, or initial workflow missing"
  if [[ -n "$PROBLEM_IDS_FILE" ]]; then
    [[ -s "$PROBLEM_IDS_FILE" ]] || fatal "problem IDs file missing: $PROBLEM_IDS_FILE"
    [[ "$EVAL_START" == 0 ]] || fatal "EVAL_START must be 0 with PROBLEM_IDS_FILE"
  fi
  local model
  for model in "$MODEL_A1" "$MODEL_A2" "$MODEL_A3" "$MODEL_OPT"; do
    [[ -s "$model/config.json" ]] || fatal "invalid base model: $model"
    [[ ! -e "$model/adapter_config.json" ]] || fatal "LoRA is forbidden: $model"
  done
}

wait_idle() {
  [[ "$WAIT_FOR_IDLE_GPUS" == 1 ]] || return 0
  gpu_guard_wait_idle "AFlow" "$GPU_IDS" "$GPU_IDLE_MAX_MEMORY_MIB" \
    "$GPU_IDLE_CHECKS" "$GPU_POLL_SECONDS" "$GPU_IDLE_TIMEOUT_SECONDS" ||
    fatal "GPU readiness check failed"
}

SERVER_PIDS=()
stop_servers() {
  local pid end=$((SECONDS + SERVER_STOP_TIMEOUT))
  for pid in "${SERVER_PIDS[@]}"; do
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  done
  for pid in "${SERVER_PIDS[@]}"; do
    while kill -0 -- "-$pid" 2>/dev/null; do
      (( SECONDS < end )) || { kill -KILL -- "-$pid" 2>/dev/null || true; break; }
      sleep 1
    done
    wait "$pid" 2>/dev/null || true
  done
  SERVER_PIDS=()
}
cleanup() { local code=$?; trap - EXIT INT TERM; stop_servers; exit "$code"; }
trap cleanup EXIT INT TERM

wait_server() {
  local expected="$1" port="$2"
  "$PYTHON_BIN" - "$expected" "http://${HOST}:${port}/v1/models" "$SERVER_WAIT_TIMEOUT" <<'PY'
import json, sys, time, urllib.request
expected, url, timeout = sys.argv[1], sys.argv[2], float(sys.argv[3])
deadline, last = time.time() + timeout, None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            names = [str(item["id"]) for item in json.loads(response.read()).get("data", [])]
        if expected in names:
            print(f"server ready: {expected}"); raise SystemExit(0)
        last = names
    except Exception as exc: last = exc
    time.sleep(5)
raise SystemExit(f"server timeout {expected}: {last}")
PY
}

start_server() {
  local label="$1" model="$2" served="$3" port="$4" gpus="$5" tp="$6" stage="$7"
  local runtime="${RUNTIME_ROOT}/${stage}_${label}"
  mkdir -p "$runtime/ray" "$runtime/torchinductor"
  local -a command=(
    "$VLLM_PYTHON_BIN" -m vllm.entrypoints.openai.api_server
    --host "$HOST" --port "$port" --model "$model" --served-model-name "$served"
    --tensor-parallel-size "$tp" --dtype "$TORCH_DTYPE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS" --trust-remote-code
  )
  [[ "$ENFORCE_EAGER" == 1 ]] && command+=(--enforce-eager)
  [[ "$ENABLE_PREFIX_CACHING" == 1 ]] && command+=(--enable-prefix-caching)
  [[ "$ENABLE_CHUNKED_PREFILL" == 1 ]] && command+=(--enable-chunked-prefill)
  [[ -n "$EXTRA_VLLM_ARGS" ]] && { local -a extra=($EXTRA_VLLM_ARGS); command+=("${extra[@]}"); }
  gpu_guard_wait_port_free "$stage/$label" "$HOST" "$port" "$SERVER_WAIT_TIMEOUT" "$GPU_POLL_SECONDS" ||
    fatal "port $HOST:$port is not free before $stage/$label"
  setsid env CUDA_VISIBLE_DEVICES="$gpus" \
    LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    TMPDIR="$runtime" RAY_TMPDIR="$runtime/ray" TORCHINDUCTOR_CACHE_DIR="$runtime/torchinductor" \
    PYTHONPATH="${PYTHON_COMPAT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${command[@]}" >"${RUN_DIR}/${stage}_${label}_server.log" 2>&1 &
  SERVER_PIDS+=("$!")
  echo "[server] $label model=$model GPUs=$gpus TP=$tp"
}

start_search_servers() {
  start_server A1 "$MODEL_A1" "$SERVED_A1" "$A1_PORT" 0,1 2 search
  start_server A2 "$MODEL_A2" "$SERVED_A2" "$A2_PORT" 2,3 2 search
  start_server A3 "$MODEL_A3" "$SERVED_A3" "$A3_PORT" 4,5 2 search
  start_server OPT "$MODEL_OPT" "$SERVED_OPT" "$OPT_PORT" 6,7 2 search
  wait_server "$SERVED_A1" "$A1_PORT"; wait_server "$SERVED_A2" "$A2_PORT"
  wait_server "$SERVED_A3" "$A3_PORT"; wait_server "$SERVED_OPT" "$OPT_PORT"
}

start_eval_servers() {
  start_server A1 "$MODEL_A1" "$SERVED_A1" "$A1_PORT" 0,1 2 eval
  start_server A2 "$MODEL_A2" "$SERVED_A2" "$A2_PORT" 2,3 2 eval
  start_server A3 "$MODEL_A3" "$SERVED_A3" "$A3_PORT" 4,5,6,7 4 eval
  wait_server "$SERVED_A1" "$A1_PORT"; wait_server "$SERVED_A2" "$A2_PORT"; wait_server "$SERVED_A3" "$A3_PORT"
}

search_command() {
  local -a thinking_args=(--no-enable-thinking --no-require-thinking)
  [[ "$ENABLE_THINKING" == 1 ]] && thinking_args[0]=--enable-thinking
  [[ "$REQUIRE_THINKING" == 1 ]] && thinking_args[1]=--require-thinking
  runner "$SEARCH_RUNNER" --data-root "$DATA_ROOT" --run-dir "$SEARCH_RUN_DIR" \
    --initial-workflow "$INITIAL_WORKFLOW" --search-size "$SEARCH_SIZE" \
    --search-seed "$SEARCH_SEED" --rng-seed "$RNG_SEED" --max-iterations "$MAX_ITERATIONS" \
    --generation-seed "$GENERATION_SEED" --max-new-tokens-executor "$MAX_NEW_TOKENS_EXEC" \
    --max-new-tokens-optimizer "$MAX_NEW_TOKENS_OPT" --temperature-executor "$TEMPERATURE_EXEC" \
    --temperature-optimizer "$TEMPERATURE_OPT" --top-p "$TOP_P" --request-retries "$REQUEST_RETRIES" \
    --max-concurrency "$SEARCH_CONCURRENCY" --api-timeout "$API_TIMEOUT" \
    --api-base-a1 "http://${HOST}:${A1_PORT}/v1" --api-model-a1 "$SERVED_A1" --model-path-a1 "$MODEL_A1" \
    --api-base-a2 "http://${HOST}:${A2_PORT}/v1" --api-model-a2 "$SERVED_A2" --model-path-a2 "$MODEL_A2" \
    --api-base-a3 "http://${HOST}:${A3_PORT}/v1" --api-model-a3 "$SERVED_A3" --model-path-a3 "$MODEL_A3" \
    --api-base-optimizer "http://${HOST}:${OPT_PORT}/v1" --api-model-optimizer "$SERVED_OPT" \
    --model-path-optimizer "$MODEL_OPT" "${thinking_args[@]}" "$@"
}

find_best_workflow() {
  "$PYTHON_BIN" - "$SEARCH_RUN_DIR/state.json" <<'PY'
import json, sys
state = json.load(open(sys.argv[1], encoding="utf-8"))
valid = [node for node in state["nodes"] if node["parse_ok"]]
best = max(valid, key=lambda node: (node["dev_em"], -node["round_id"]))
print(best["source_file"])
PY
}

eval_command() {
  local -a thinking_args=(--no-enable-thinking --no-require-thinking)
  [[ "$ENABLE_THINKING" == 1 ]] && thinking_args[0]=--enable-thinking
  [[ "$REQUIRE_THINKING" == 1 ]] && thinking_args[1]=--require-thinking
  local -a command=(
    "$EVAL_RUNNER" --data-root "$DATA_ROOT" --split test --start "$EVAL_START" --limit "$EVAL_LIMIT"
    --workflow-file "$WORKFLOW_FILE" --output "$EVAL_OUTPUT" --summary "$EVAL_SUMMARY" --progress "$EVAL_PROGRESS"
    --generation-seed "$GENERATION_SEED" --max-new-tokens "$MAX_NEW_TOKENS_EXEC"
    --temperature "$TEMPERATURE_EXEC" --top-p "$TOP_P" --request-retries "$REQUEST_RETRIES"
    --max-concurrency "$EVAL_CONCURRENCY" --api-timeout "$API_TIMEOUT"
    --api-base-a1 "http://${HOST}:${A1_PORT}/v1" --api-model-a1 "$SERVED_A1" --model-path-a1 "$MODEL_A1"
    --api-base-a2 "http://${HOST}:${A2_PORT}/v1" --api-model-a2 "$SERVED_A2" --model-path-a2 "$MODEL_A2"
    --api-base-a3 "http://${HOST}:${A3_PORT}/v1" --api-model-a3 "$SERVED_A3" --model-path-a3 "$MODEL_A3"
    "${thinking_args[@]}"
  )
  [[ -z "$PROBLEM_IDS_FILE" ]] || command+=(--problem-ids-file "$PROBLEM_IDS_FILE")
  [[ "$RESUME" == 1 ]] && command+=(--resume)
  runner "${command[@]}" "$@"
}

validate
if [[ "$DRY_RUN" == 1 ]]; then
  echo "MATH AFlow dry-run mode=$MODE search=train20x20 eval=test${EVAL_LIMIT} ids=${PROBLEM_IDS_FILE:-contiguous}"
  echo "thinking=$ENABLE_THINKING/$REQUIRE_THINKING retries=$REQUEST_RETRIES exec_tokens=$MAX_NEW_TOKENS_EXEC opt_tokens=$MAX_NEW_TOKENS_OPT context=$MAX_MODEL_LEN"
  echo "search_gpus=2+2+2+2 eval_gpus=2+2+4"
  if [[ "$MODE" == search || "$MODE" == both ]]; then search_command --dry-run; fi
  if [[ "$MODE" == eval ]]; then eval_command --dry-run; fi
  exit 0
fi

mkdir -p "$RUN_DIR" "$SEARCH_RUN_DIR"
exec > >(tee "$LOG_PATH") 2>&1
wait_idle
cat >"${RUN_DIR}/config.env" <<EOF
RUN_ID=$RUN_ID
MODE=$MODE
TRAINING_FREE=1
ENABLE_THINKING=$ENABLE_THINKING
REQUIRE_THINKING=$REQUIRE_THINKING
SEARCH_SIZE=$SEARCH_SIZE
MAX_ITERATIONS=$MAX_ITERATIONS
SEARCH_SEED=$SEARCH_SEED
RNG_SEED=$RNG_SEED
MODEL_A1=$MODEL_A1
MODEL_A2=$MODEL_A2
MODEL_A3=$MODEL_A3
MODEL_OPT=$MODEL_OPT
MAX_NEW_TOKENS_EXEC=$MAX_NEW_TOKENS_EXEC
MAX_NEW_TOKENS_OPT=$MAX_NEW_TOKENS_OPT
MAX_MODEL_LEN=$MAX_MODEL_LEN
TEMPERATURE_EXEC=$TEMPERATURE_EXEC
TEMPERATURE_OPT=$TEMPERATURE_OPT
REQUEST_RETRIES=$REQUEST_RETRIES
SEARCH_GPU_LAYOUT=2+2+2+2
EVAL_GPU_LAYOUT=2+2+4
PROBLEM_IDS_FILE=$PROBLEM_IDS_FILE
EOF

if [[ "$MODE" == search || "$MODE" == both ]]; then
  start_search_servers
  search_command
  stop_servers
  WORKFLOW_FILE="$(find_best_workflow)"
  echo "Frozen best workflow: $WORKFLOW_FILE"
fi
if [[ "$MODE" == eval || "$MODE" == both ]]; then
  [[ -f "$WORKFLOW_FILE" ]] || fatal "workflow missing: $WORKFLOW_FILE"
  if [[ "$MODE" == both ]]; then
    wait_idle
  fi
  start_eval_servers
  eval_command
  stop_servers
fi
echo "MATH AFlow complete mode=$MODE run_dir=$RUN_DIR"

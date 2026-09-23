#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/data/wangyuheng/jca}"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/qwen35/bin/python}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-/data/conda_envs/deep_research/bin/python}"
VLLM_LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH:-/data/conda_envs/deep_research/lib}"
PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-/data/wangyuheng}"
PYTHON_COMPAT_DIR="${PYTHON_COMPAT_DIR:-${PROJECT_ROOT}/scripts/gsm_judge_rl/python_compat}"
RUNNER="${RUNNER:-${SCRIPT_DIR}/math_mad_role_batched.py}"
PROMPT_DIR="${PROMPT_DIR:-${SCRIPT_DIR}/prompts}"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/Math/data/MATH}"
SPLIT="${SPLIT:-test}"
START="${START:-0}"
LIMIT="${LIMIT:-5000}"
PROBLEM_IDS_FILE="${PROBLEM_IDS_FILE:-}"
N_ROUNDS="${N_ROUNDS:-3}"
MODEL_A1="${MODEL_A1:-/data/wangyuheng/models/Qwen3-1.7B}"
MODEL_A2="${MODEL_A2:-/data/wangyuheng/models/Qwen3-4B}"
MODEL_A3="${MODEL_A3:-/data/wangyuheng/models/Qwen3-8B}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
TEMPERATURE_ROUND0="${TEMPERATURE_ROUND0:-0.9}"
TEMPERATURE_DEBATE="${TEMPERATURE_DEBATE:-0.3}"
TOP_P="${TOP_P:-0.95}"
GENERATION_SEED="${GENERATION_SEED:-42}"
TURN_RETRIES="${TURN_RETRIES:-4}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
REQUIRE_THINKING="${REQUIRE_THINKING:-0}"
API_TIMEOUT="${API_TIMEOUT:-900}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-128}"
JOURNAL_FSYNC_EVERY="${JOURNAL_FSYNC_EVERY:-25}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8201}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
TP="${TP:-1}"
DP="${DP:-8}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-900}"
SERVER_STOP_TIMEOUT="${SERVER_STOP_TIMEOUT:-45}"
START_RETRIES="${START_RETRIES:-3}"

RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"
WAIT_FOR_IDLE_GPUS="${WAIT_FOR_IDLE_GPUS:-1}"
GPU_IDLE_TIMEOUT_SECONDS="${GPU_IDLE_TIMEOUT_SECONDS:-86400}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-10}"
GPU_IDLE_CHECKS="${GPU_IDLE_CHECKS:-2}"
GPU_IDLE_MAX_MEMORY_MIB="${GPU_IDLE_MAX_MEMORY_MIB:-2048}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_math_mad_r3}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/math_baselines/mad}"
RUN_DIR="${RUN_DIR:-${LOG_ROOT}/${RUN_ID}}"
STATE_PATH="${STATE_PATH:-${RUN_DIR}/trajectory_state.json}"
OUTPUT_PATH="${OUTPUT_PATH:-${RUN_DIR}/trajectories.jsonl}"
SUMMARY_PATH="${SUMMARY_PATH:-${RUN_DIR}/summary.json}"
LOG_PATH="${LOG_PATH:-${RUN_DIR}/run.log}"
RUNTIME_ROOT="${RUNTIME_ROOT:-$(mktemp -d /data/tmp/jca_math_mad.XXXXXX)}"

source "$PROJECT_ROOT/scripts/math_gpu_guard.sh"

fatal() { echo "[fatal] $*" >&2; exit 1; }
runner() { env PYTHONPATH="$PYTHONPATH_ROOT" "$PYTHON_BIN" -u "$RUNNER" "$@"; }
pending() { runner pending --state "$STATE_PATH" --round "$1" --agent "$2"; }

validate() {
  [[ "$SPLIT" == test ]] || fatal "formal MATH MAD evaluation requires SPLIT=test"
  [[ "$N_ROUNDS" == 3 ]] || fatal "N_ROUNDS must be 3"
  [[ "$GPU_IDS" == 0,1,2,3,4,5,6,7 && "$TP" == 1 && "$DP" == 8 ]] || \
    fatal "role batching requires GPU_IDS=0,1,2,3,4,5,6,7 TP=1 DP=8"
  [[ "$MAX_NEW_TOKENS" == 8192 && "$MAX_MODEL_LEN" == 40960 ]] || \
    fatal "token budgets must match JCA: 8192/40960"
  [[ "$TEMPERATURE_ROUND0" == 0.9 && "$TEMPERATURE_DEBATE" == 0.3 ]] || \
    fatal "MAD temperatures must be round0=0.9 debate=0.3"
  [[ "$ENABLE_THINKING" =~ ^[01]$ && "$REQUIRE_THINKING" =~ ^[01]$ ]] || \
    fatal "thinking flags must be 0 or 1"
  [[ "$REQUIRE_THINKING" == 0 || "$ENABLE_THINKING" == 1 ]] || \
    fatal "REQUIRE_THINKING=1 requires ENABLE_THINKING=1"
  [[ -x "$PYTHON_BIN" && -x "$VLLM_PYTHON_BIN" ]] || fatal "python executable missing"
  [[ -f "$RUNNER" && -d "$PROMPT_DIR" && -d "$DATA_ROOT" ]] || fatal "runner, prompts, or data missing"
  if [[ -n "$PROBLEM_IDS_FILE" ]]; then
    [[ -s "$PROBLEM_IDS_FILE" ]] || fatal "problem IDs file missing: $PROBLEM_IDS_FILE"
    [[ "$START" == 0 ]] || fatal "START must be 0 with PROBLEM_IDS_FILE"
  fi
  local model
  for model in "$MODEL_A1" "$MODEL_A2" "$MODEL_A3"; do
    [[ -s "$model/config.json" ]] || fatal "invalid base model: $model"
    [[ ! -e "$model/adapter_config.json" ]] || fatal "LoRA is forbidden: $model"
  done
}

wait_idle() {
  [[ "$WAIT_FOR_IDLE_GPUS" == 1 ]] || return 0
  gpu_guard_wait_idle "MAD" "$GPU_IDS" "$GPU_IDLE_MAX_MEMORY_MIB" \
    "$GPU_IDLE_CHECKS" "$GPU_POLL_SECONDS" "$GPU_IDLE_TIMEOUT_SECONDS" ||
    fatal "GPU readiness check failed"
}

SERVER_PID=""
stop_server() {
  [[ -n "$SERVER_PID" ]] || return 0
  kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
  local end=$((SECONDS + SERVER_STOP_TIMEOUT))
  while kill -0 -- "-$SERVER_PID" 2>/dev/null; do
    (( SECONDS < end )) || { kill -KILL -- "-$SERVER_PID" 2>/dev/null || true; break; }
    sleep 1
  done
  wait "$SERVER_PID" 2>/dev/null || true
  SERVER_PID=""
}
cleanup() { local code=$?; trap - EXIT INT TERM; stop_server; exit "$code"; }
trap cleanup EXIT INT TERM

wait_server() {
  local expected="$1"
  "$PYTHON_BIN" - "$expected" "http://${HOST}:${PORT}/v1/models" "$SERVER_WAIT_TIMEOUT" <<'PY'
import json, os, sys, time, urllib.request
expected, url, timeout = sys.argv[1], sys.argv[2], float(sys.argv[3])
deadline, last = time.time() + timeout, None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            names = [str(item["id"]) for item in json.loads(response.read()).get("data", [])]
        if expected in names:
            print(f"server ready: {names}"); raise SystemExit(0)
        last = names
    except Exception as exc:
        last = exc
    pid = os.environ.get("VLLM_SERVER_PID")
    if pid:
        try: os.kill(int(pid), 0)
        except ProcessLookupError: raise SystemExit(f"server {pid} exited: {last}")
    time.sleep(5)
raise SystemExit(f"server timeout: {last}")
PY
}

start_server() {
  local agent="$1" model="$2" stage="$3" attempt runtime
  local -a command=(
    "$VLLM_PYTHON_BIN" -m vllm.entrypoints.openai.api_server
    --host "$HOST" --port "$PORT" --model "$model" --served-model-name "${agent}_base"
    --tensor-parallel-size "$TP" --data-parallel-size "$DP"
    --dtype "$TORCH_DTYPE" --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" --trust-remote-code
  )
  [[ "$ENABLE_PREFIX_CACHING" == 1 ]] && command+=(--enable-prefix-caching)
  [[ "$ENABLE_CHUNKED_PREFILL" == 1 ]] && command+=(--enable-chunked-prefill)
  [[ "$ENFORCE_EAGER" == 1 ]] && command+=(--enforce-eager)
  [[ -n "$EXTRA_VLLM_ARGS" ]] && { local -a extra=($EXTRA_VLLM_ARGS); command+=("${extra[@]}"); }
  for ((attempt=1; attempt<=START_RETRIES; attempt++)); do
    wait_idle
    gpu_guard_wait_port_free "$stage/$agent" "$HOST" "$PORT" "$SERVER_WAIT_TIMEOUT" "$GPU_POLL_SECONDS" ||
      fatal "port $HOST:$PORT is not free before $stage/$agent"
    runtime="${RUNTIME_ROOT}/${stage}_${agent}_a${attempt}"
    mkdir -p "$runtime/ray" "$runtime/torchinductor"
    echo "[server] stage=$stage agent=$agent attempt=$attempt/$START_RETRIES model=$model"
    setsid env CUDA_VISIBLE_DEVICES="$GPU_IDS" \
      LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
      TMPDIR="$runtime" RAY_TMPDIR="$runtime/ray" TORCHINDUCTOR_CACHE_DIR="$runtime/torchinductor" \
      PYTHONPATH="${PYTHON_COMPAT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${command[@]}" >"${RUN_DIR}/${stage}_server.log" 2>&1 &
    SERVER_PID="$!"
    if VLLM_SERVER_PID="$SERVER_PID" wait_server "${agent}_base"; then return 0; fi
    stop_server
    (( attempt < START_RETRIES )) && sleep 5
  done
  fatal "failed to start $agent"
}

run_stage() {
  local round="$1" agent="$2" model="$3"
  local stage="round${round}_${agent}" queued
  queued="$(pending "$round" "$agent")"
  (( queued > 0 )) || { echo "[resume] stage=$stage already complete or blocked"; return 0; }
  start_server "$agent" "$model" "$stage"
  runner run-turns --state "$STATE_PATH" --round "$round" --agent "$agent" \
    --prompt-dir "$PROMPT_DIR" --api-base "http://${HOST}:${PORT}/v1" \
    --api-model "${agent}_base" --api-key EMPTY --api-timeout "$API_TIMEOUT" \
    --max-concurrency "$MAX_CONCURRENCY" --journal-fsync-every "$JOURNAL_FSYNC_EVERY" \
    2>&1 | tee "${RUN_DIR}/${stage}_phase.log"
  stop_server
}

validate
thinking_args=(--no-enable-thinking --no-require-thinking)
[[ "$ENABLE_THINKING" == 1 ]] && thinking_args[0]=--enable-thinking
[[ "$REQUIRE_THINKING" == 1 ]] && thinking_args[1]=--require-thinking
problem_ids_args=()
[[ -z "$PROBLEM_IDS_FILE" ]] || problem_ids_args=(--problem-ids-file "$PROBLEM_IDS_FILE")
if [[ "$DRY_RUN" == 1 ]]; then
  echo "MATH MAD dry-run: split=$SPLIT start=$START limit=$LIMIT ids=${PROBLEM_IDS_FILE:-contiguous} rounds=$N_ROUNDS"
  echo "thinking=$ENABLE_THINKING/$REQUIRE_THINKING retries=$TURN_RETRIES max_tokens=$MAX_NEW_TOKENS max_model_len=$MAX_MODEL_LEN"
  echo "temperatures=$TEMPERATURE_ROUND0/$TEMPERATURE_DEBATE GPUs=$GPU_IDS TP=$TP DP=$DP"
  echo "aggregation=normalize_math_answer majority tie_break=A3,A2,A1"
  exit 0
fi

mkdir -p "$RUN_DIR"
exec > >(tee "$LOG_PATH") 2>&1
wait_idle
if [[ "$RESUME" == 0 ]]; then
  [[ ! -e "$STATE_PATH" && ! -e "$OUTPUT_PATH" ]] || fatal "run exists; set RESUME=1"
  runner init --state "$STATE_PATH" --data-root "$DATA_ROOT" --split "$SPLIT" \
    --start "$START" --limit "$LIMIT" --n-rounds "$N_ROUNDS" \
    "${problem_ids_args[@]}" \
    --max-new-tokens "$MAX_NEW_TOKENS" --temperature-round0 "$TEMPERATURE_ROUND0" \
    --temperature-debate "$TEMPERATURE_DEBATE" --top-p "$TOP_P" \
    --generation-seed "$GENERATION_SEED" --turn-retries "$TURN_RETRIES" \
    "${thinking_args[@]}" \
    --model-a1 "$MODEL_A1" --model-a2 "$MODEL_A2" --model-a3 "$MODEL_A3"
else
  [[ -s "$STATE_PATH" ]] || fatal "RESUME=1 requires state: $STATE_PATH"
fi

cat >"${RUN_DIR}/config.env" <<EOF
RUN_ID=$RUN_ID
TRAINING_FREE=1
ENABLE_THINKING=$ENABLE_THINKING
REQUIRE_THINKING=$REQUIRE_THINKING
MODEL_A1=$MODEL_A1
MODEL_A2=$MODEL_A2
MODEL_A3=$MODEL_A3
N_ROUNDS=$N_ROUNDS
MAX_NEW_TOKENS=$MAX_NEW_TOKENS
MAX_MODEL_LEN=$MAX_MODEL_LEN
TEMPERATURE_ROUND0=$TEMPERATURE_ROUND0
TEMPERATURE_DEBATE=$TEMPERATURE_DEBATE
TOP_P=$TOP_P
GENERATION_SEED=$GENERATION_SEED
TURN_RETRIES=$TURN_RETRIES
PROBLEM_IDS_FILE=$PROBLEM_IDS_FILE
GPU_IDS=$GPU_IDS
TP=$TP
DP=$DP
AGGREGATION=normalize_math_answer_majority_A3_A2_A1
EOF

for round in 0 1 2; do
  run_stage "$round" A1 "$MODEL_A1"
  run_stage "$round" A2 "$MODEL_A2"
  run_stage "$round" A3 "$MODEL_A3"
done

runner status --state "$STATE_PATH"
finalize=(finalize --state "$STATE_PATH" --output "$OUTPUT_PATH" --summary "$SUMMARY_PATH")
[[ "$RESUME" == 1 ]] && finalize+=(--resume)
runner "${finalize[@]}"
echo "MATH MAD complete: $OUTPUT_PATH"

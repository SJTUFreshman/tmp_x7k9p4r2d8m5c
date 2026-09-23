#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/wangyuheng/jca}"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/qwen35/bin/python}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-/data/conda_envs/deep_research/bin/python}"
VLLM_LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH:-/data/conda_envs/deep_research/lib}"
PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-/data/wangyuheng}"
PYTHON_COMPAT_DIR="${PYTHON_COMPAT_DIR:-${PROJECT_ROOT}/scripts/gsm_judge_rl/python_compat}"
RUNNER="${RUNNER:-${PROJECT_ROOT}/experiments/math_rl_mas_thinking/math_role_batched.py}"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/Math/data/MATH}"
SPLIT="${SPLIT:-train}"
SUBJECTS="${SUBJECTS:-}"
START="${START:-0}"
LIMIT="${LIMIT:-7500}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-8}"
T_MAX="${T_MAX:-8}"
START_AGENT="${START_AGENT:-A1}"
START_AGENT_SEED="${START_AGENT_SEED:-42}"
BOOTSTRAP_AGENT="${BOOTSTRAP_AGENT:-}"
MIN_AGENTS_BEFORE_STOP="${MIN_AGENTS_BEFORE_STOP:-1}"
ALLOW_FIRST_TURN_STOP="${ALLOW_FIRST_TURN_STOP:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_P="${TOP_P:-0.95}"
GENERATION_SEED="${GENERATION_SEED:-42}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
REQUIRE_THINKING="${REQUIRE_THINKING:-0}"
API_TIMEOUT="${API_TIMEOUT:-900}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-128}"
GROUP_RETRIES="${GROUP_RETRIES:-5}"
RETRY_FAILED_GROUPS="${RETRY_FAILED_GROUPS:-0}"
STEP_RETRIES="${STEP_RETRIES:-4}"
JSON_TRANSPORT="${JSON_TRANSPORT:-json_object}"
OUTPUT_MODE="${OUTPUT_MODE:-turns}"
LOG_RAW_CHARS="${LOG_RAW_CHARS:-0}"
JOURNAL_FSYNC_EVERY="${JOURNAL_FSYNC_EVERY:-50}"

MODEL_A1="${MODEL_A1:-/data/wangyuheng/models/Qwen3-1.7B}"
MODEL_A2="${MODEL_A2:-/data/wangyuheng/models/Qwen3-4B}"
MODEL_A3="${MODEL_A3:-/data/wangyuheng/models/Qwen3-8B}"
ADAPTER_A1="${ADAPTER_A1:-}"
ADAPTER_A2="${ADAPTER_A2:-}"
ADAPTER_A3="${ADAPTER_A3:-}"
USE_LORA="${USE_LORA:-1}"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"
MAX_LORAS="${MAX_LORAS:-1}"
MAX_CPU_LORAS="${MAX_CPU_LORAS:-1}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8201}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
TP="${TP:-1}"
DP="${DP:-8}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-900}"
SERVER_STOP_TIMEOUT="${SERVER_STOP_TIMEOUT:-45}"
START_RETRIES="${START_RETRIES:-3}"
WAIT_FOR_IDLE_GPUS="${WAIT_FOR_IDLE_GPUS:-1}"
GPU_IDLE_TIMEOUT_SECONDS="${GPU_IDLE_TIMEOUT_SECONDS:-86400}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-10}"
GPU_IDLE_CHECKS="${GPU_IDLE_CHECKS:-2}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_math_role_batched_${SPLIT}}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/math_role_batched}"
RUN_DIR="${RUN_DIR:-${LOG_DIR}/${RUN_ID}}"
STATE_PATH="${STATE_PATH:-${RUN_DIR}/trajectory_state.json}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/math_role_batched}/${RUN_ID}.jsonl}"
LOG_PATH="${LOG_PATH:-${RUN_DIR}/run.log}"
CONFIG_PATH="${CONFIG_PATH:-${RUN_DIR}/config.env}"
RUNTIME_ROOT="${RUNTIME_ROOT:-$(mktemp -d /data/tmp/jca_math_rb.XXXXXX)}"

fatal() { echo "[fatal] $*" >&2; exit 1; }
runner() { env PYTHONPATH="$PYTHONPATH_ROOT" "$PYTHON_BIN" -u "$RUNNER" "$@"; }
pending() {
  if [[ -n "${2:-}" ]]; then
    runner pending --state "$STATE_PATH" --agent "${1:-all}" --turn "$2"
  else
    runner pending --state "$STATE_PATH" --agent "${1:-all}"
  fi
}
status_count() { runner status --state "$STATE_PATH" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin)[sys.argv[1]])' "$1"; }

STATE_CONFIG_DIGEST=""
config_contract() {
  {
    printf 'contract_version=%q\n' 'math_role_batched_launcher_v1'
    printf 'state_config_sha256=%q\n' "$STATE_CONFIG_DIGEST"
    printf 'PROJECT_ROOT=%q\n' "$PROJECT_ROOT"
    printf 'PYTHON_BIN=%q\n' "$PYTHON_BIN"
    printf 'VLLM_PYTHON_BIN=%q\n' "$VLLM_PYTHON_BIN"
    printf 'VLLM_LD_LIBRARY_PATH=%q\n' "$VLLM_LD_LIBRARY_PATH"
    printf 'PYTHONPATH_ROOT=%q\n' "$PYTHONPATH_ROOT"
    printf 'PYTHON_COMPAT_DIR=%q\n' "$PYTHON_COMPAT_DIR"
    printf 'RUNNER=%q\n' "$RUNNER"
    printf 'RUN_ID=%q\n' "$RUN_ID"
    printf 'RUN_DIR=%q\n' "$RUN_DIR"
    printf 'STATE_PATH=%q\n' "$STATE_PATH"
    printf 'OUTPUT_PATH=%q\n' "$OUTPUT_PATH"
    printf 'LOG_PATH=%q\n' "$LOG_PATH"
    printf 'CONFIG_PATH=%q\n' "$CONFIG_PATH"
    printf 'DATA_ROOT=%q\n' "$DATA_ROOT"
    printf 'SPLIT=%q\n' "$SPLIT"
    printf 'SUBJECTS=%q\n' "$SUBJECTS"
    printf 'START=%q\n' "$START"
    printf 'LIMIT=%q\n' "$LIMIT"
    printf 'NUM_ROLLOUTS=%q\n' "$NUM_ROLLOUTS"
    printf 'T_MAX=%q\n' "$T_MAX"
    printf 'START_AGENT=%q\n' "$START_AGENT"
    printf 'START_AGENT_SEED=%q\n' "$START_AGENT_SEED"
    printf 'BOOTSTRAP_AGENT=%q\n' "$BOOTSTRAP_AGENT"
    printf 'MIN_AGENTS_BEFORE_STOP=%q\n' "$MIN_AGENTS_BEFORE_STOP"
    printf 'ALLOW_FIRST_TURN_STOP=%q\n' "$ALLOW_FIRST_TURN_STOP"
    printf 'GENERATION_SEED=%q\n' "$GENERATION_SEED"
    printf 'MAX_NEW_TOKENS=%q\n' "$MAX_NEW_TOKENS"
    printf 'TEMPERATURE=%q\n' "$TEMPERATURE"
    printf 'TOP_P=%q\n' "$TOP_P"
    printf 'ENABLE_THINKING=%q\n' "$ENABLE_THINKING"
    printf 'REQUIRE_THINKING=%q\n' "$REQUIRE_THINKING"
    printf 'API_TIMEOUT=%q\n' "$API_TIMEOUT"
    printf 'GROUP_RETRIES=%q\n' "$GROUP_RETRIES"
    printf 'RETRY_FAILED_GROUPS=%q\n' "$RETRY_FAILED_GROUPS"
    printf 'STEP_RETRIES=%q\n' "$STEP_RETRIES"
    printf 'JSON_TRANSPORT=%q\n' "$JSON_TRANSPORT"
    printf 'OUTPUT_MODE=%q\n' "$OUTPUT_MODE"
    printf 'MODEL_A1=%q\n' "$MODEL_A1"
    printf 'MODEL_A2=%q\n' "$MODEL_A2"
    printf 'MODEL_A3=%q\n' "$MODEL_A3"
    printf 'ADAPTER_A1=%q\n' "$ADAPTER_A1"
    printf 'ADAPTER_A2=%q\n' "$ADAPTER_A2"
    printf 'ADAPTER_A3=%q\n' "$ADAPTER_A3"
    printf 'USE_LORA=%q\n' "$USE_LORA"
    printf 'MAX_LORA_RANK=%q\n' "$MAX_LORA_RANK"
    printf 'MAX_LORAS=%q\n' "$MAX_LORAS"
    printf 'MAX_CPU_LORAS=%q\n' "$MAX_CPU_LORAS"
    printf 'HOST=%q\n' "$HOST"
    printf 'PORT=%q\n' "$PORT"
    printf 'GPU_IDS=%q\n' "$GPU_IDS"
    printf 'TP=%q\n' "$TP"
    printf 'DP=%q\n' "$DP"
    printf 'TORCH_DTYPE=%q\n' "$TORCH_DTYPE"
    printf 'GPU_MEMORY_UTILIZATION=%q\n' "$GPU_MEMORY_UTILIZATION"
    printf 'MAX_MODEL_LEN=%q\n' "$MAX_MODEL_LEN"
    printf 'MAX_NUM_SEQS=%q\n' "$MAX_NUM_SEQS"
    printf 'MAX_NUM_BATCHED_TOKENS=%q\n' "$MAX_NUM_BATCHED_TOKENS"
    printf 'ENFORCE_EAGER=%q\n' "$ENFORCE_EAGER"
    printf 'ENABLE_PREFIX_CACHING=%q\n' "$ENABLE_PREFIX_CACHING"
    printf 'ENABLE_CHUNKED_PREFILL=%q\n' "$ENABLE_CHUNKED_PREFILL"
    printf 'EXTRA_VLLM_ARGS=%q\n' "$EXTRA_VLLM_ARGS"
    printf 'MAX_CONCURRENCY=%q\n' "$MAX_CONCURRENCY"
    printf 'LOG_RAW_CHARS=%q\n' "$LOG_RAW_CHARS"
    printf 'JOURNAL_FSYNC_EVERY=%q\n' "$JOURNAL_FSYNC_EVERY"
    printf 'SERVER_WAIT_TIMEOUT=%q\n' "$SERVER_WAIT_TIMEOUT"
    printf 'SERVER_STOP_TIMEOUT=%q\n' "$SERVER_STOP_TIMEOUT"
    printf 'START_RETRIES=%q\n' "$START_RETRIES"
    printf 'WAIT_FOR_IDLE_GPUS=%q\n' "$WAIT_FOR_IDLE_GPUS"
    printf 'GPU_IDLE_TIMEOUT_SECONDS=%q\n' "$GPU_IDLE_TIMEOUT_SECONDS"
    printf 'GPU_POLL_SECONDS=%q\n' "$GPU_POLL_SECONDS"
    printf 'GPU_IDLE_CHECKS=%q\n' "$GPU_IDLE_CHECKS"
  }
}

state_config_digest() {
  "$PYTHON_BIN" - "$STATE_PATH" <<'PY'
import hashlib
import json
import sys

path = sys.argv[1]
try:
    with open(path, encoding="utf-8") as handle:
        state = json.load(handle)
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"cannot read state config for resume contract: {exc}")
config = state.get("config") if isinstance(state, dict) else None
if not isinstance(config, dict):
    raise SystemExit("state has no object config for resume contract")
canonical = json.dumps(
    config, ensure_ascii=True, sort_keys=True, separators=(",", ":")
).encode("utf-8")
print(hashlib.sha256(canonical).hexdigest())
PY
}

write_config_snapshot() {
  local temporary="${CONFIG_PATH}.tmp.$$"
  mkdir -p -- "$(dirname -- "$CONFIG_PATH")"
  if ! (
    umask 077
    config_contract >"$temporary"
  ); then
    rm -f -- "$temporary"
    fatal "could not write launcher config snapshot: $CONFIG_PATH"
  fi
  mv -f -- "$temporary" "$CONFIG_PATH"
}

validate_resume_config() {
  [[ -s "$CONFIG_PATH" ]] || fatal "RESUME=1 requires config snapshot: $CONFIG_PATH"
  local current_contract="${CONFIG_PATH}.check.$$"
  if ! config_contract >"$current_contract"; then
    rm -f -- "$current_contract"
    fatal "could not build current resume contract"
  fi
  if ! cmp -s "$CONFIG_PATH" "$current_contract"; then
    echo "[fatal] resume configuration mismatch; refusing to mix runs" >&2
    diff -u -- "$CONFIG_PATH" "$current_contract" | sed -n '1,120p' >&2 || true
    rm -f -- "$current_contract"
    fatal "resume contract mismatch"
  fi
  rm -f -- "$current_contract"
}

wait_idle() {
  [[ "$WAIT_FOR_IDLE_GPUS" == "1" ]] || return 0
  local end=$((SECONDS + GPU_IDLE_TIMEOUT_SECONDS)) checks=0 pids
  while (( checks < GPU_IDLE_CHECKS )); do
    pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ -z "${pids//[[:space:]]/}" ]]; then
      checks=$((checks + 1)); echo "[gpu] idle check ${checks}/${GPU_IDLE_CHECKS}"
    else
      checks=0; echo "[gpu] GPUs occupied: ${pids//$'\n'/, }"
    fi
    (( checks >= GPU_IDLE_CHECKS )) || sleep "$GPU_POLL_SECONDS"
    (( SECONDS < end )) || fatal "GPUs did not become idle in time"
  done
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

agent_settings() {
  case "$1" in
    A1) ACTIVE_MODEL="$MODEL_A1"; ACTIVE_ADAPTER="$ADAPTER_A1";;
    A2) ACTIVE_MODEL="$MODEL_A2"; ACTIVE_ADAPTER="$ADAPTER_A2";;
    A3) ACTIVE_MODEL="$MODEL_A3"; ACTIVE_ADAPTER="$ADAPTER_A3";;
    *) fatal "unknown agent $1";;
  esac
}

wait_server() {
  local expected="$1"
  "$PYTHON_BIN" - "$expected" "http://${HOST}:${PORT}/v1/models" "$SERVER_WAIT_TIMEOUT" <<'PY'
import json, os, sys, time, urllib.request
expected, url, timeout = sys.argv[1], sys.argv[2], float(sys.argv[3])
deadline = time.time() + timeout
last = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            names = [str(x["id"]) for x in json.loads(response.read())["data"]]
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
  local agent="$1" stage="$2"; agent_settings "$agent"
  local log="${RUN_DIR}/${stage}_server.log"
  local served_model_name="$agent"
  if [[ "$USE_LORA" == "1" ]]; then
    served_model_name="${agent}_base"
  fi
  local -a cmd=(
    "$VLLM_PYTHON_BIN" -m vllm.entrypoints.openai.api_server
    --host "$HOST" --port "$PORT" --model "$ACTIVE_MODEL"
    --served-model-name "$served_model_name"
    --tensor-parallel-size "$TP" --data-parallel-size "$DP"
    --dtype "$TORCH_DTYPE" --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$MAX_MODEL_LEN" --trust-remote-code
  )
  if [[ "$USE_LORA" == "1" ]]; then
    [[ -n "$ACTIVE_ADAPTER" ]] || fatal "$agent adapter is empty"
    cmd+=(--enable-lora --max-lora-rank "$MAX_LORA_RANK" --max-loras "$MAX_LORAS"
      --max-cpu-loras "$MAX_CPU_LORAS" --lora-modules "${agent}=${ACTIVE_ADAPTER}")
  fi
  [[ "$ENFORCE_EAGER" == "1" ]] && cmd+=(--enforce-eager)
  [[ "$ENABLE_PREFIX_CACHING" == "1" ]] && cmd+=(--enable-prefix-caching)
  [[ "$ENABLE_CHUNKED_PREFILL" == "1" ]] && cmd+=(--enable-chunked-prefill)
  [[ -n "$MAX_NUM_SEQS" ]] && cmd+=(--max-num-seqs "$MAX_NUM_SEQS")
  [[ -n "$MAX_NUM_BATCHED_TOKENS" ]] && cmd+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
  [[ -n "$EXTRA_VLLM_ARGS" ]] && { local -a extra=($EXTRA_VLLM_ARGS); cmd+=("${extra[@]}"); }
  local attempt runtime
  for ((attempt=1; attempt<=START_RETRIES; attempt++)); do
    runtime="${RUNTIME_ROOT}/s${stage}_${agent}_a${attempt}"
    mkdir -p "$runtime/ray" "$runtime/torchinductor"
    echo "[server] start ${agent}, stage=${stage}, attempt=${attempt}/${START_RETRIES}"
    setsid env CUDA_VISIBLE_DEVICES="$GPU_IDS" \
      LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
      TMPDIR="$runtime" RAY_TMPDIR="$runtime/ray" TORCHINDUCTOR_CACHE_DIR="$runtime/torchinductor" \
      PYTHONPATH="${PYTHON_COMPAT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${cmd[@]}" >"$log" 2>&1 &
    SERVER_PID="$!"
    if VLLM_SERVER_PID="$SERVER_PID" wait_server "$agent"; then return 0; fi
    stop_server
    (( attempt < START_RETRIES )) && sleep 5
  done
  fatal "failed to start $agent after $START_RETRIES attempts"
}

validate() {
  [[ "$SPLIT" == "train" || "$SPLIT" == "test" ]] || fatal "invalid SPLIT"
  [[ "$JSON_TRANSPORT" == "json_object" ]] || fatal "MATH role batching requires JSON_TRANSPORT=json_object"
  [[ "$USE_LORA" == "0" || "$USE_LORA" == "1" ]] || fatal "USE_LORA must be 0 or 1"
  [[ "$RETRY_FAILED_GROUPS" == "0" || "$RETRY_FAILED_GROUPS" == "1" ]] || fatal "RETRY_FAILED_GROUPS must be 0 or 1"
  [[ "$ALLOW_FIRST_TURN_STOP" == "0" || "$ALLOW_FIRST_TURN_STOP" == "1" ]] || fatal "ALLOW_FIRST_TURN_STOP must be 0 or 1"
  [[ -z "$BOOTSTRAP_AGENT" || "$BOOTSTRAP_AGENT" == "A1" || "$BOOTSTRAP_AGENT" == "A2" || "$BOOTSTRAP_AGENT" == "A3" ]] || fatal "BOOTSTRAP_AGENT must be empty or A1/A2/A3"
  [[ -x "$PYTHON_BIN" && -x "$VLLM_PYTHON_BIN" ]] || fatal "python executable missing"
  [[ -f "$RUNNER" && -d "$DATA_ROOT" ]] || fatal "runner or data root missing"
  [[ "$GPU_IDS" == "0,1,2,3,4,5,6,7" && "$TP" == "1" && "$DP" == "8" ]] || fatal "MATH role batching requires all 8 GPUs with TP=1 DP=8"
  if [[ "$USE_LORA" == "1" ]]; then
    for adapter in "$ADAPTER_A1" "$ADAPTER_A2" "$ADAPTER_A3"; do
      [[ -s "$adapter/adapter_config.json" && -s "$adapter/adapter_model.safetensors" ]] || fatal "invalid adapter: $adapter"
    done
  fi
}

validate
if [[ "$DRY_RUN" == "1" ]]; then
  echo "MATH role-batched dry-run: split=$SPLIT limit=$LIMIT rollouts=$NUM_ROLLOUTS JSON_TRANSPORT=$JSON_TRANSPORT"
  exit 0
fi
mkdir -p "$RUN_DIR" "$(dirname "$OUTPUT_PATH")"
exec > >(tee "$LOG_PATH") 2>&1

if [[ "$RESUME" == "1" ]]; then
  [[ -s "$STATE_PATH" ]] || fatal "RESUME=1 requires state: $STATE_PATH"
  if ! STATE_CONFIG_DIGEST="$(state_config_digest)"; then
    fatal "could not read state config for resume contract: $STATE_PATH"
  fi
  validate_resume_config
else
  [[ ! -e "$STATE_PATH" ]] || fatal "new run refuses existing state: $STATE_PATH"
  [[ ! -e "$CONFIG_PATH" ]] || fatal "new run refuses existing config snapshot: $CONFIG_PATH"
fi

wait_idle

if [[ "$RESUME" == "0" ]]; then
  init_cmd=(--state "$STATE_PATH" --data-root "$DATA_ROOT" --split "$SPLIT"
    --start "$START" --limit "$LIMIT" --num-rollouts "$NUM_ROLLOUTS" --t-max "$T_MAX"
    --start-agent "$START_AGENT" --start-agent-seed "$START_AGENT_SEED"
    --min-agents-before-stop "$MIN_AGENTS_BEFORE_STOP" --generation-seed "$GENERATION_SEED"
    --max-new-tokens "$MAX_NEW_TOKENS" --temperature "$TEMPERATURE" --top-p "$TOP_P"
    --api-timeout "$API_TIMEOUT" --group-retries "$GROUP_RETRIES" --step-retries "$STEP_RETRIES"
    --json-transport "$JSON_TRANSPORT" --output-mode "$OUTPUT_MODE")
  if [[ "$RETRY_FAILED_GROUPS" == "1" ]]; then
    init_cmd+=(--retry-failed-groups)
  else
    init_cmd+=(--no-retry-failed-groups)
  fi
  if [[ "$ALLOW_FIRST_TURN_STOP" == "1" ]]; then
    init_cmd+=(--allow-first-turn-stop)
  else
    init_cmd+=(--no-allow-first-turn-stop)
  fi
  [[ -n "$SUBJECTS" ]] && init_cmd+=(--subjects "$SUBJECTS")
  [[ -n "$BOOTSTRAP_AGENT" ]] && init_cmd+=(--bootstrap-agent "$BOOTSTRAP_AGENT")
  if [[ "$ENABLE_THINKING" == "1" ]]; then init_cmd+=(--enable-thinking); else init_cmd+=(--no-enable-thinking); fi
  if [[ "$REQUIRE_THINKING" == "1" ]]; then init_cmd+=(--require-thinking); else init_cmd+=(--no-require-thinking); fi
  runner init "${init_cmd[@]}"
  if ! STATE_CONFIG_DIGEST="$(state_config_digest)"; then
    fatal "could not read initialized state config: $STATE_PATH"
  fi
  write_config_snapshot
else
  :
fi

bootstrap_pending="$(runner bootstrap-pending --state "$STATE_PATH" --agent all)"
if (( bootstrap_pending > 0 )); then
  [[ -n "$BOOTSTRAP_AGENT" ]] || fatal "state has pending bootstrap work but BOOTSTRAP_AGENT is empty"
  echo "================ direct-solver bootstrap: agent=${BOOTSTRAP_AGENT} pending=${bootstrap_pending} ================"
  start_server "$BOOTSTRAP_AGENT" "bootstrap_${BOOTSTRAP_AGENT}"
  runner run-bootstrap --state "$STATE_PATH" --agent "$BOOTSTRAP_AGENT" \
    --api-base "http://${HOST}:${PORT}/v1" --api-model "$BOOTSTRAP_AGENT" --api-key EMPTY \
    --max-concurrency "$MAX_CONCURRENCY" \
    --journal-fsync-every "$JOURNAL_FSYNC_EVERY" \
    2>&1 | tee "${RUN_DIR}/bootstrap_${BOOTSTRAP_AGENT}_phase.log"
  stop_server
  bootstrap_pending="$(runner bootstrap-pending --state "$STATE_PATH" --agent all)"
  (( bootstrap_pending == 0 )) || fatal "bootstrap phase left $bootstrap_pending pending trajectories"
fi

for ((pass=0; ; pass++)); do
  failed="$(status_count failed)"
  if (( failed > 0 )); then
    failed_zero_step="$(status_count failed_zero_step_failures)"
    if (( failed != failed_zero_step )); then
      fatal "$failed trajectories exhausted retries (at least one has usable turns)"
    fi
    echo "[quarantine] $failed zero-step trajectories exhausted retries; they will be recorded in the failure sidecar"
  fi
  pending_total="$(pending all)"; retry_total="$(status_count retry)"
  (( pending_total > 0 || retry_total > 0 )) || break
  echo "================ global-turn pass ${pass}: pending=${pending_total} retry=${retry_total} ================"
  for ((turn=0; turn<T_MAX; turn++)); do
    for agent in A1 A2 A3; do
      queued="$(pending "$agent" "$turn")"
      (( queued > 0 )) || continue
      stage="p$(printf '%02d' "$pass")_t$(printf '%02d' "$turn")_${agent}"
      start_server "$agent" "$stage"
      runner run-agent --state "$STATE_PATH" --agent "$agent" --turn "$turn" \
        --api-base "http://${HOST}:${PORT}/v1" --api-model "$agent" --api-key EMPTY \
        --max-concurrency "$MAX_CONCURRENCY" --log-raw-chars "$LOG_RAW_CHARS" \
        --journal-fsync-every "$JOURNAL_FSYNC_EVERY" \
        2>&1 | tee "${RUN_DIR}/${stage}_phase.log"
      stop_server
    done
  done
  pending_total="$(pending all)"
  (( pending_total == 0 )) || fatal "global pass left $pending_total pending trajectories"
  retry_total="$(status_count retry)"
  if (( retry_total > 0 )); then
    runner reset-retries --state "$STATE_PATH"
  else
    break
  fi
done

runner status --state "$STATE_PATH"
finalize_cmd=(finalize --state "$STATE_PATH" --output "$OUTPUT_PATH")
[[ "$RESUME" == "1" ]] && finalize_cmd+=(--resume)
if [[ -n "${MATH_FAILURE_LEDGER:-}" ]]; then
  [[ "$MATH_FAILURE_LEDGER" == "${OUTPUT_PATH}.failures.jsonl" ]] || fatal "failure ledger path differs from native sidecar"
  [[ "${MATH_SAMPLE_STATE:-}" == "$STATE_PATH" ]] || fatal "failure ledger state differs from sampling state"
  finalize_cmd+=(--allow-zero-step-failures)
fi
runner "${finalize_cmd[@]}"
echo "MATH role-batched run complete: $OUTPUT_PATH"

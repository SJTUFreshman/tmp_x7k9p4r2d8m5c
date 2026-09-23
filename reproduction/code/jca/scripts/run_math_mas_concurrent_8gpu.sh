#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/wangyuheng/jca}"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/qwen35/bin/python}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-/data/conda_envs/deep_research/bin/python}"
VLLM_LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH:-/data/conda_envs/deep_research/lib}"
PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-/data/wangyuheng}"
PYTHON_COMPAT_DIR="${PYTHON_COMPAT_DIR:-${PROJECT_ROOT}/scripts/gsm_judge_rl/python_compat}"
RUNNER="${RUNNER:-${PROJECT_ROOT}/experiments/math_rl_mas_thinking/math_role_batched.py}"

DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/Math/data/MATH}"
SPLIT="${SPLIT:-test}"
SUBJECTS="${SUBJECTS:-}"
START="${START:-0}"
LIMIT="${LIMIT:-500}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-1}"
T_MAX="${T_MAX:-8}"
START_AGENT="${START_AGENT:-A3}"
START_AGENT_SEED="${START_AGENT_SEED:-43}"
BOOTSTRAP_AGENT="${BOOTSTRAP_AGENT:-A3}"
BOOTSTRAP_HANDOFF_TARGET="${BOOTSTRAP_HANDOFF_TARGET:-}"
PRESERVE_REASONABLE_INCUMBENT="${PRESERVE_REASONABLE_INCUMBENT:-0}"
LOCK_UPSTREAM_ANSWER_AGENTS="${LOCK_UPSTREAM_ANSWER_AGENTS:-}"
MIN_AGENTS_BEFORE_STOP="${MIN_AGENTS_BEFORE_STOP:-1}"
ALLOW_FIRST_TURN_STOP="${ALLOW_FIRST_TURN_STOP:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
PROTOCOL_THINKING_MAX_TOKENS="${PROTOCOL_THINKING_MAX_TOKENS:-2048}"
PROTOCOL_MAX_TOKENS="${PROTOCOL_MAX_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.95}"
GENERATION_SEED="${GENERATION_SEED:-42}"
ENABLE_THINKING="${ENABLE_THINKING:-1}"
REQUIRE_THINKING="${REQUIRE_THINKING:-1}"
API_TIMEOUT="${API_TIMEOUT:-1800}"
GROUP_RETRIES="${GROUP_RETRIES:-0}"
RETRY_FAILED_GROUPS="${RETRY_FAILED_GROUPS:-0}"
STEP_RETRIES="${STEP_RETRIES:-2}"
JSON_TRANSPORT="${JSON_TRANSPORT:-json_schema}"
OUTPUT_MODE="${OUTPUT_MODE:-trajectories}"
JOURNAL_FSYNC_EVERY="${JOURNAL_FSYNC_EVERY:-25}"
PROGRESS_EVERY="${PROGRESS_EVERY:-25}"
SCHEDULER_MODE="${SCHEDULER_MODE:-shared_all_gpu}"
if [[ -z "${ADAPTIVE_LAYOUT+x}" ]]; then
  if [[ "$SCHEDULER_MODE" == "shared_all_gpu" ]]; then
    ADAPTIVE_LAYOUT=0
  else
    ADAPTIVE_LAYOUT=1
  fi
fi
REBALANCE_AFTER_ADVANCED="${REBALANCE_AFTER_ADVANCED:-64}"
MAX_ADVANCED_PER_PASS="${MAX_ADVANCED_PER_PASS:-0}"
MAX_INFLIGHT="${MAX_INFLIGHT:-0}"
A1_WORK_WEIGHT="${A1_WORK_WEIGHT:-2.0}"
A2_WORK_WEIGHT="${A2_WORK_WEIGHT:-1.0}"
A3_WORK_WEIGHT="${A3_WORK_WEIGHT:-1.0}"
ADAPTIVE_ALLOW_ZERO="${ADAPTIVE_ALLOW_ZERO:-0}"
ADAPTIVE_LOOKAHEAD="${ADAPTIVE_LOOKAHEAD:-0.00}"
ADAPTIVE_LOOKAHEAD_MIN_PENDING="${ADAPTIVE_LOOKAHEAD_MIN_PENDING:-64}"

MODEL_A1="${MODEL_A1:-/data/wangyuheng/models/Qwen3-1.7B}"
MODEL_A2="${MODEL_A2:-/data/wangyuheng/models/Qwen3-4B}"
MODEL_A3="${MODEL_A3:-/data/wangyuheng/models/Qwen3-8B}"
ADAPTER_A1="${ADAPTER_A1:-}"
ADAPTER_A2="${ADAPTER_A2:-}"
ADAPTER_A3="${ADAPTER_A3:-}"

HOST="${HOST:-127.0.0.1}"
A1_PORT="${A1_PORT:-8201}"
A2_PORT="${A2_PORT:-8202}"
A3_PORT="${A3_PORT:-8203}"
ENDPOINT_PORT_STRIDE="${ENDPOINT_PORT_STRIDE:-10}"

TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
A1_GPU_MEMORY_UTILIZATION="${A1_GPU_MEMORY_UTILIZATION:-0.15}"
A2_GPU_MEMORY_UTILIZATION="${A2_GPU_MEMORY_UTILIZATION:-0.24}"
A3_GPU_MEMORY_UTILIZATION="${A3_GPU_MEMORY_UTILIZATION:-0.45}"
DEDUP_GPU_MEMORY_UTILIZATION="${DEDUP_GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-48}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"
DEDUPLICATE_IDENTICAL_STACK="${DEDUPLICATE_IDENTICAL_STACK:-1}"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"
MAX_LORAS="${MAX_LORAS:-1}"
MAX_CPU_LORAS="${MAX_CPU_LORAS:-1}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
STRUCTURED_OUTPUTS_CONFIG="${STRUCTURED_OUTPUTS_CONFIG:-}"
if [[ -z "$STRUCTURED_OUTPUTS_CONFIG" ]]; then
  STRUCTURED_OUTPUTS_CONFIG='{"backend":"xgrammar","disable_any_whitespace":true}'
fi

SHARED_GPU_IDS="${SHARED_GPU_IDS:-0,1,2,3,4,5,6,7}"
SHARED_DP="${SHARED_DP:-8}"
if [[ "$SCHEDULER_MODE" == "shared_all_gpu" ]]; then
  A1_GPU_IDS="${A1_GPU_IDS:-$SHARED_GPU_IDS}"
  A2_GPU_IDS="${A2_GPU_IDS:-$SHARED_GPU_IDS}"
  A3_GPU_IDS="${A3_GPU_IDS:-$SHARED_GPU_IDS}"
  A1_DP="${A1_DP:-$SHARED_DP}"
  A2_DP="${A2_DP:-$SHARED_DP}"
  A3_DP="${A3_DP:-$SHARED_DP}"
else
  A1_GPU_IDS="${A1_GPU_IDS:-0,1}"
  A2_GPU_IDS="${A2_GPU_IDS:-2,3}"
  A3_GPU_IDS="${A3_GPU_IDS:-4,5,6,7}"
  A1_DP="${A1_DP:-2}"
  A2_DP="${A2_DP:-2}"
  A3_DP="${A3_DP:-4}"
fi
A1_MAX_CONCURRENCY="${A1_MAX_CONCURRENCY:-$((A1_DP * MAX_NUM_SEQS))}"
A2_MAX_CONCURRENCY="${A2_MAX_CONCURRENCY:-$((A2_DP * MAX_NUM_SEQS))}"
A3_MAX_CONCURRENCY="${A3_MAX_CONCURRENCY:-$((A3_DP * MAX_NUM_SEQS))}"

SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-900}"
SERVER_STOP_TIMEOUT="${SERVER_STOP_TIMEOUT:-45}"
SERVER_START_RETRIES="${SERVER_START_RETRIES:-3}"
SERVER_START_RETRY_DELAY="${SERVER_START_RETRY_DELAY:-5}"
WAIT_FOR_IDLE_GPUS="${WAIT_FOR_IDLE_GPUS:-1}"
GPU_IDLE_TIMEOUT_SECONDS="${GPU_IDLE_TIMEOUT_SECONDS:-86400}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-10}"
GPU_IDLE_CHECKS="${GPU_IDLE_CHECKS:-2}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_math_concurrent_${SPLIT}}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/math_role_concurrent}"
RUN_DIR="${RUN_DIR:-${LOG_DIR}/${RUN_ID}}"
STATE_PATH="${STATE_PATH:-${RUN_DIR}/trajectory_state.json}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/outputs/math_role_concurrent/${RUN_ID}.jsonl}"
LOG_PATH="${LOG_PATH:-${RUN_DIR}/run.log}"
CONFIG_PATH="${CONFIG_PATH:-${RUN_DIR}/concurrent_config.env}"
RUNTIME_ROOT="${RUNTIME_ROOT:-$(mktemp -d /data/tmp/jca_math_concurrent.XXXXXX)}"
EMPTY_STDIN="${RUNTIME_ROOT}/empty.stdin"
SUPPRESSED_LOG="${RUNTIME_ROOT}/suppressed.log"
: >"$EMPTY_STDIN"
: >"$SUPPRESSED_LOG"

fatal() { echo "[fatal] $*" >&2; exit 1; }
runner() { env PYTHONPATH="$PYTHONPATH_ROOT" "$PYTHON_BIN" -u "$RUNNER" "$@"; }
status_field() {
  runner status --state "$STATE_PATH" | "$PYTHON_BIN" -c \
    'import json,sys; print(json.load(sys.stdin)[sys.argv[1]])' "$1"
}

identical_stack() {
  [[ "$SCHEDULER_MODE" == "shared_all_gpu" \
    && "$DEDUPLICATE_IDENTICAL_STACK" == "1" \
    && "$MODEL_A1" == "$MODEL_A2" && "$MODEL_A2" == "$MODEL_A3" \
    && "$ADAPTER_A1" == "$ADAPTER_A2" && "$ADAPTER_A2" == "$ADAPTER_A3" ]]
}

gpu_range() {
  local start="$1" count="$2" end
  (( count > 0 )) || return 0
  end=$((start + count - 1))
  seq -s, "$start" "$end"
}

layout_key() {
  if identical_stack; then
    printf 'deduplicated-%s' "$A3_DP"
    return
  fi
  printf '%s/%s/%s' "$A1_DP" "$A2_DP" "$A3_DP"
}

compute_adaptive_layout() {
  local status allocation
  status="$(runner status --state "$STATE_PATH")"
  allocation="$(printf '%s\n' "$status" | "$PYTHON_BIN" -c '
import json, sys
status = json.load(sys.stdin)
weights = [float(value) for value in sys.argv[1:4]]
pending = [
    float(status["pending_by_agent"]["A1"]),
    float(status["pending_by_agent"]["A2"]),
    float(status["pending_by_agent"]["A3"]) + float(status["bootstrap_pending"]),
]
lookahead = max(0.0, min(1.0, float(sys.argv[5])))
minimum_pending = max(1.0, float(sys.argv[6]))
scale = min(1.0, sum(pending) / minimum_pending) if sum(pending) else 0.0
lookahead *= scale
# The protocol normally advances A1 -> A2 -> A3 -> A1.  Forecasting one and
# two hops keeps a downstream endpoint warm without reserving GPUs forever.
pressure = [
    (pending[0] + lookahead * pending[2] + lookahead * lookahead * pending[1]) * weights[0],
    (pending[1] + lookahead * pending[0] + lookahead * lookahead * pending[2]) * weights[1],
    (pending[2] + lookahead * pending[1] + lookahead * lookahead * pending[0]) * weights[2],
]
total_gpus = 8
allow_zero = bool(int(sys.argv[4]))
if allow_zero:
    # Do not reserve a process/GPU for an agent with no ready work.  A
    # bootstrap queue always keeps A3 resident; newly-created downstream work
    # causes the runner to drain and the shell to relaunch that role.
    allocation = [1 if value > 0 else 0 for value in pressure]
    if not any(allocation):
        allocation = [1, 1, 1]
else:
    allocation = [1, 1, 1]
while sum(allocation) < total_gpus:
    target = max(
        range(3),
        key=lambda index: (
            pressure[index] / allocation[index] if allocation[index] else pressure[index],
            pressure[index],
            -index,
        ),
    )
    allocation[target] += 1
if sum(allocation) > total_gpus:
    # This can only happen when all three queues are non-empty and the
    # configured minimum exceeds the available device count.
    allocation = [1, 1, 1]
print(*allocation)
' "$A1_WORK_WEIGHT" "$A2_WORK_WEIGHT" "$A3_WORK_WEIGHT" \
    "$ADAPTIVE_ALLOW_ZERO" "$ADAPTIVE_LOOKAHEAD" "$ADAPTIVE_LOOKAHEAD_MIN_PENDING")"
  read -r A1_DP A2_DP A3_DP <<<"$allocation"
  A1_GPU_IDS="$(gpu_range 0 "$A1_DP")"
  A2_GPU_IDS="$(gpu_range "$A1_DP" "$A2_DP")"
  A3_GPU_IDS="$(gpu_range "$((A1_DP + A2_DP))" "$A3_DP")"
  A1_MAX_CONCURRENCY=$((A1_DP * MAX_NUM_SEQS))
  A2_MAX_CONCURRENCY=$((A2_DP * MAX_NUM_SEQS))
  A3_MAX_CONCURRENCY=$((A3_DP * MAX_NUM_SEQS))
  validate_layout
}

validate_layout() {
  "$PYTHON_BIN" - "$SCHEDULER_MODE" "$SHARED_GPU_IDS" \
    "$A1_GPU_IDS" "$A2_GPU_IDS" "$A3_GPU_IDS" "$A1_DP" "$A2_DP" "$A3_DP" <<'PY'
import sys
mode = sys.argv[1]
shared = sys.argv[2].split(",")
groups = [value.split(",") for value in sys.argv[3:6]]
dp = [int(value) for value in sys.argv[6:9]]
for index, (group, size) in enumerate(zip(groups, dp), 1):
    group = [gpu for gpu in group if gpu]
    if len(group) != size or len(set(group)) != len(group):
        raise SystemExit(f"A{index} GPU count must equal DP size")
for gpu in shared:
    if not gpu.isdigit() or not 0 <= int(gpu) <= 7:
        raise SystemExit(f"invalid shared GPU id: {gpu!r}")
if not shared or len(set(shared)) != len(shared):
    raise SystemExit("shared GPU ids must be non-empty and unique")
if mode == "shared_all_gpu":
    if any(group != shared for group in groups):
        raise SystemExit("shared_all_gpu requires every agent on SHARED_GPU_IDS")
elif mode == "adaptive_partition":
    flat = [gpu for group in groups for gpu in group if gpu]
    if sorted(flat) != [str(index) for index in range(8)] or len(set(flat)) != 8:
        raise SystemExit("adaptive groups must partition GPUs 0..7 exactly")
else:
    raise SystemExit(f"unsupported scheduler mode: {mode}")
PY
}

validate() {
  [[ "$BOOTSTRAP_AGENT" == "A3" ]] || fatal "concurrent evaluator requires BOOTSTRAP_AGENT=A3"
  [[ "$START_AGENT" == "$BOOTSTRAP_AGENT" ]] || fatal "first protocol agent must match BOOTSTRAP_AGENT"
  [[ -z "$BOOTSTRAP_HANDOFF_TARGET" || "$BOOTSTRAP_HANDOFF_TARGET" == "A1" || "$BOOTSTRAP_HANDOFF_TARGET" == "A2" || "$BOOTSTRAP_HANDOFF_TARGET" == "A3" ]] || \
    fatal "BOOTSTRAP_HANDOFF_TARGET must be empty, A1, A2, or A3"
  [[ "$PRESERVE_REASONABLE_INCUMBENT" == "0" || "$PRESERVE_REASONABLE_INCUMBENT" == "1" ]] || \
    fatal "PRESERVE_REASONABLE_INCUMBENT must be 0 or 1"
  [[ "$LOCK_UPSTREAM_ANSWER_AGENTS" =~ ^(A[123](,A[123])*)?$ ]] || \
    fatal "LOCK_UPSTREAM_ANSWER_AGENTS must be a comma-separated list of A1/A2/A3"
  [[ "$ALLOW_FIRST_TURN_STOP" == "0" ]] || fatal "first protocol turn must hand off"
  [[ "$JSON_TRANSPORT" == "json_schema" ]] || fatal "JSON_TRANSPORT must be json_schema"
  [[ "$REASONING_PARSER" == "qwen3" ]] || fatal "REASONING_PARSER must be qwen3"
  [[ "$OUTPUT_MODE" == "trajectories" ]] || fatal "concurrent evaluator requires trajectory output"
  [[ "$RESUME" == "0" || "$RESUME" == "1" ]] || fatal "RESUME must be 0 or 1"
  [[ "$DRY_RUN" == "0" || "$DRY_RUN" == "1" ]] || fatal "DRY_RUN must be 0 or 1"
  [[ "$SCHEDULER_MODE" == "shared_all_gpu" || "$SCHEDULER_MODE" == "adaptive_partition" ]] || \
    fatal "SCHEDULER_MODE must be shared_all_gpu or adaptive_partition"
  [[ "$ADAPTIVE_LAYOUT" == "0" || "$ADAPTIVE_LAYOUT" == "1" ]] || fatal "ADAPTIVE_LAYOUT must be 0 or 1"
  [[ "$ADAPTIVE_ALLOW_ZERO" == "0" || "$ADAPTIVE_ALLOW_ZERO" == "1" ]] || \
    fatal "ADAPTIVE_ALLOW_ZERO must be 0 or 1"
  "$PYTHON_BIN" - "$ADAPTIVE_LOOKAHEAD" "$ADAPTIVE_LOOKAHEAD_MIN_PENDING" <<'PY'
import sys
lookahead = float(sys.argv[1])
minimum_pending = int(sys.argv[2])
if not 0.0 <= lookahead <= 1.0:
    raise SystemExit("ADAPTIVE_LOOKAHEAD must be in [0, 1]")
if minimum_pending <= 0:
    raise SystemExit("ADAPTIVE_LOOKAHEAD_MIN_PENDING must be positive")
PY
  [[ "$SCHEDULER_MODE" != "shared_all_gpu" || "$ADAPTIVE_LAYOUT" == "0" ]] || \
    fatal "shared_all_gpu must keep ADAPTIVE_LAYOUT=0"
  [[ "$ENFORCE_EAGER" == "0" || "$ENFORCE_EAGER" == "1" ]] || fatal "ENFORCE_EAGER must be 0 or 1"
  [[ "$DEDUPLICATE_IDENTICAL_STACK" == "0" || "$DEDUPLICATE_IDENTICAL_STACK" == "1" ]] || \
    fatal "DEDUPLICATE_IDENTICAL_STACK must be 0 or 1"
  [[ "$REBALANCE_AFTER_ADVANCED" =~ ^[1-9][0-9]*$ ]] || fatal "REBALANCE_AFTER_ADVANCED must be positive"
  [[ "$MAX_ADVANCED_PER_PASS" =~ ^[0-9]+$ ]] || fatal "MAX_ADVANCED_PER_PASS must be non-negative"
  [[ "$MAX_INFLIGHT" =~ ^[0-9]+$ ]] || fatal "MAX_INFLIGHT must be non-negative"
  [[ "$SERVER_START_RETRIES" =~ ^[1-9][0-9]*$ ]] || fatal "SERVER_START_RETRIES must be positive"
  [[ "$SERVER_START_RETRY_DELAY" =~ ^[0-9]+$ ]] || fatal "SERVER_START_RETRY_DELAY must be non-negative"
  [[ "$ENDPOINT_PORT_STRIDE" =~ ^[1-9][0-9]*$ ]] || fatal "ENDPOINT_PORT_STRIDE must be positive"
  for value in "$MAX_NUM_SEQS" "$MAX_NUM_BATCHED_TOKENS" "$PROTOCOL_THINKING_MAX_TOKENS" \
    "$PROTOCOL_MAX_TOKENS"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || fatal "token/sequence limits must be positive integers"
  done
  for value in "$A1_MAX_CONCURRENCY" "$A2_MAX_CONCURRENCY" "$A3_MAX_CONCURRENCY"; do
    [[ "$value" =~ ^[0-9]+$ ]] || fatal "concurrency values must be non-negative integers"
  done
  if [[ "$SCHEDULER_MODE" == "shared_all_gpu" ]]; then
    for value in "$A1_MAX_CONCURRENCY" "$A2_MAX_CONCURRENCY" "$A3_MAX_CONCURRENCY"; do
      [[ "$value" =~ ^[1-9][0-9]*$ ]] || fatal "shared scheduler concurrency must be positive"
    done
  else
    (( A1_MAX_CONCURRENCY + A2_MAX_CONCURRENCY + A3_MAX_CONCURRENCY > 0 )) || \
      fatal "adaptive scheduler needs at least one active endpoint"
  fi
  "$PYTHON_BIN" - "$SCHEDULER_MODE" "$GPU_MEMORY_UTILIZATION" \
    "$A1_GPU_MEMORY_UTILIZATION" "$A2_GPU_MEMORY_UTILIZATION" \
    "$A3_GPU_MEMORY_UTILIZATION" "$DEDUP_GPU_MEMORY_UTILIZATION" <<'PY'
import sys
mode = sys.argv[1]
values = [float(value) for value in sys.argv[2:]]
if any(not 0.0 < value < 1.0 for value in values):
    raise SystemExit("GPU memory utilizations must be in (0, 1)")
if mode == "shared_all_gpu" and sum(values[1:4]) >= 0.95:
    raise SystemExit("shared agent GPU memory utilizations must sum to less than 0.95")
PY
  [[ -x "$PYTHON_BIN" && -x "$VLLM_PYTHON_BIN" && -f "$RUNNER" && -d "$DATA_ROOT" ]] || \
    fatal "runner, Python, or data root is missing"
  for path in "$MODEL_A1" "$MODEL_A2" "$MODEL_A3"; do [[ -d "$path" ]] || fatal "model missing: $path"; done
  for path in "$ADAPTER_A1" "$ADAPTER_A2" "$ADAPTER_A3"; do
    [[ -s "$path/adapter_config.json" && -s "$path/adapter_model.safetensors" ]] || \
      fatal "adapter missing or incomplete: $path"
  done
  validate_layout
}

wait_idle() {
  [[ "$WAIT_FOR_IDLE_GPUS" == "1" ]] || return 0
  local deadline=$((SECONDS + GPU_IDLE_TIMEOUT_SECONDS)) checks=0 pids
  while (( checks < GPU_IDLE_CHECKS )); do
    pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>>"$SUPPRESSED_LOG" || true)"
    if [[ -z "${pids//[[:space:]]/}" ]]; then
      checks=$((checks + 1)); echo "[gpu] idle check ${checks}/${GPU_IDLE_CHECKS}"
    else
      checks=0; echo "[gpu] occupied: ${pids//$'\n'/, }"
    fi
    (( checks >= GPU_IDLE_CHECKS )) || sleep "$GPU_POLL_SECONDS"
    (( SECONDS < deadline )) || fatal "GPUs did not become idle in time"
  done
}

declare -A SERVER_PIDS=()
stop_servers() {
  local key pid deadline
  for key in "${!SERVER_PIDS[@]}"; do
    pid="${SERVER_PIDS[$key]}"
    kill -TERM -- "-$pid" 2>>"$SUPPRESSED_LOG" || kill -TERM "$pid" 2>>"$SUPPRESSED_LOG" || true
  done
  deadline=$((SECONDS + SERVER_STOP_TIMEOUT))
  while (( SECONDS < deadline )); do
    local alive=0
    for key in "${!SERVER_PIDS[@]}"; do
      pid="${SERVER_PIDS[$key]}"
      if kill -0 -- "-$pid" 2>>"$SUPPRESSED_LOG"; then
        alive=1
      fi
    done
    (( alive == 0 )) && break
    sleep 1
  done
  for key in "${!SERVER_PIDS[@]}"; do
    pid="${SERVER_PIDS[$key]}"
    kill -KILL -- "-$pid" 2>>"$SUPPRESSED_LOG" || true
    wait "$pid" 2>>"$SUPPRESSED_LOG" || true
  done
  SERVER_PIDS=()
}
cleanup() { local code=$?; trap - EXIT INT TERM; stop_servers; exit "$code"; }
trap cleanup EXIT INT TERM

wait_server() {
  local agent="$1" port="$2" pid="$3" label="$4"
  "$PYTHON_BIN" - "$agent" "$label" "http://${HOST}:${port}/v1/models" "$SERVER_WAIT_TIMEOUT" "$pid" <<'PY'
import json, os, sys, time, urllib.request
agent, label, url, timeout, pid = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5])
deadline = time.time() + timeout
last = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            names = [str(item["id"]) for item in json.loads(response.read())["data"]]
        if agent in names:
            print(f"[server] {label} ready: {names}")
            raise SystemExit(0)
        last = names
    except Exception as exc:
        last = exc
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        raise SystemExit(f"{label} server exited during startup: {last}")
    time.sleep(5)
raise SystemExit(f"{label} server timeout: {last}")
PY
}

start_server() {
  local agent="$1" model="$2" adapter="$3" gpu="$4" port="$5" memory_utilization="$6"
  local key="${agent}_gpu${gpu}"
  local log="${RUN_DIR}/persistent_${agent}_gpu${gpu}_server.log"
  local runtime="${RUNTIME_ROOT}/${key}"
  mkdir -p "$runtime/ray" "$runtime/torchinductor"
  local -a command=(
    "$VLLM_PYTHON_BIN" -m vllm.entrypoints.openai.api_server
    --host "$HOST" --port "$port" --model "$model"
    --served-model-name "${agent}_base" --tensor-parallel-size 1 --data-parallel-size 1
    --dtype "$TORCH_DTYPE" --gpu-memory-utilization "$memory_utilization"
    --max-model-len "$MAX_MODEL_LEN" --trust-remote-code
    --enable-lora --max-lora-rank "$MAX_LORA_RANK" --max-loras "$MAX_LORAS"
    --max-cpu-loras "$MAX_CPU_LORAS" --lora-modules "${agent}=${adapter}"
    --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --reasoning-parser "$REASONING_PARSER"
    --structured-outputs-config "$STRUCTURED_OUTPUTS_CONFIG"
  )
  [[ "$ENFORCE_EAGER" == "1" ]] && command+=(--enforce-eager)
  [[ "$ENABLE_PREFIX_CACHING" == "1" ]] && command+=(--enable-prefix-caching)
  [[ "$ENABLE_CHUNKED_PREFILL" == "1" ]] && command+=(--enable-chunked-prefill)
  [[ -n "$EXTRA_VLLM_ARGS" ]] && { local -a extra=($EXTRA_VLLM_ARGS); command+=("${extra[@]}"); }
  echo "[server] starting $agent GPU=$gpu port=$port DP=1 memory=$memory_utilization concurrency=$MAX_NUM_SEQS"
  setsid env CUDA_VISIBLE_DEVICES="$gpu" \
    LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    TMPDIR="$runtime" RAY_TMPDIR="$runtime/ray" TORCHINDUCTOR_CACHE_DIR="$runtime/torchinductor" \
    PYTHONPATH="${PYTHON_COMPAT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${command[@]}" <"$EMPTY_STDIN" >>"$log" 2>&1 &
  SERVER_PIDS[$key]="$!"
}

start_agent_pool() {
  local agent="$1" model="$2" adapter="$3" gpus="$4" base_port="$5" memory_utilization="$6"
  [[ -n "$gpus" ]] || return 0
  local -a gpu_list=()
  local index gpu port
  IFS=',' read -r -a gpu_list <<<"$gpus"
  for index in "${!gpu_list[@]}"; do
    gpu="${gpu_list[$index]}"
    port=$((base_port + index * ENDPOINT_PORT_STRIDE))
    start_server "$agent" "$model" "$adapter" "$gpu" "$port" "$memory_utilization"
  done
}

wait_agent_pool() {
  local agent="$1" gpus="$2" base_port="$3"
  [[ -n "$gpus" ]] || return 0
  local -a gpu_list=()
  local index gpu port key
  IFS=',' read -r -a gpu_list <<<"$gpus"
  for index in "${!gpu_list[@]}"; do
    gpu="${gpu_list[$index]}"
    port=$((base_port + index * ENDPOINT_PORT_STRIDE))
    key="${agent}_gpu${gpu}"
    wait_server "$agent" "$port" "${SERVER_PIDS[$key]}" "${agent}/gpu${gpu}" || return 1
  done
}

pool_api_bases() {
  local gpus="$1" base_port="$2"
  [[ -n "$gpus" ]] || return 0
  local -a gpu_list=()
  local index port result=""
  IFS=',' read -r -a gpu_list <<<"$gpus"
  for index in "${!gpu_list[@]}"; do
    port=$((base_port + index * ENDPOINT_PORT_STRIDE))
    result="${result:+${result},}http://${HOST}:${port}/v1"
  done
  printf '%s' "$result"
}

write_runtime_config() {
  local temporary="${CONFIG_PATH}.tmp.$$"
  mkdir -p "$(dirname "$CONFIG_PATH")"
  {
    printf 'scheduler=%q\nSCHEDULER_MODE=%q\n' 'persistent_explicit_endpoint_pool_v4' "$SCHEDULER_MODE"
    printf 'ENDPOINT_PORT_STRIDE=%q\n' "$ENDPOINT_PORT_STRIDE"
    printf 'ADAPTIVE_LAYOUT=%q\nREBALANCE_AFTER_ADVANCED=%q\n' \
      "$ADAPTIVE_LAYOUT" "$REBALANCE_AFTER_ADVANCED"
    printf 'MAX_ADVANCED_PER_PASS=%q\nMAX_INFLIGHT=%q\n' \
      "$MAX_ADVANCED_PER_PASS" "$MAX_INFLIGHT"
    printf 'ADAPTIVE_ALLOW_ZERO=%q\n' "$ADAPTIVE_ALLOW_ZERO"
    printf 'ADAPTIVE_LOOKAHEAD=%q\nADAPTIVE_LOOKAHEAD_MIN_PENDING=%q\n' \
      "$ADAPTIVE_LOOKAHEAD" "$ADAPTIVE_LOOKAHEAD_MIN_PENDING"
    printf 'A1_WORK_WEIGHT=%q\nA2_WORK_WEIGHT=%q\nA3_WORK_WEIGHT=%q\n' \
      "$A1_WORK_WEIGHT" "$A2_WORK_WEIGHT" "$A3_WORK_WEIGHT"
    printf 'A1_GPU_IDS=%q\nA2_GPU_IDS=%q\nA3_GPU_IDS=%q\n' "$A1_GPU_IDS" "$A2_GPU_IDS" "$A3_GPU_IDS"
    printf 'A1_DP=%q\nA2_DP=%q\nA3_DP=%q\n' "$A1_DP" "$A2_DP" "$A3_DP"
    printf 'A1_MAX_CONCURRENCY=%q\nA2_MAX_CONCURRENCY=%q\nA3_MAX_CONCURRENCY=%q\n' \
      "$A1_MAX_CONCURRENCY" "$A2_MAX_CONCURRENCY" "$A3_MAX_CONCURRENCY"
    printf 'A1_GPU_MEMORY_UTILIZATION=%q\nA2_GPU_MEMORY_UTILIZATION=%q\nA3_GPU_MEMORY_UTILIZATION=%q\n' \
      "$A1_GPU_MEMORY_UTILIZATION" "$A2_GPU_MEMORY_UTILIZATION" "$A3_GPU_MEMORY_UTILIZATION"
    printf 'DEDUPLICATE_IDENTICAL_STACK=%q\nDEDUP_GPU_MEMORY_UTILIZATION=%q\n' \
      "$DEDUPLICATE_IDENTICAL_STACK" "$DEDUP_GPU_MEMORY_UTILIZATION"
    printf 'MAX_NUM_SEQS=%q\nMAX_NUM_BATCHED_TOKENS=%q\n' "$MAX_NUM_SEQS" "$MAX_NUM_BATCHED_TOKENS"
    printf 'ENFORCE_EAGER=%q\n' "$ENFORCE_EAGER"
    printf 'PROTOCOL_THINKING_MAX_TOKENS=%q\nPROTOCOL_MAX_TOKENS=%q\nREASONING_PARSER=%q\n' \
      "$PROTOCOL_THINKING_MAX_TOKENS" "$PROTOCOL_MAX_TOKENS" "$REASONING_PARSER"
    printf 'BOOTSTRAP_HANDOFF_TARGET=%q\nPRESERVE_REASONABLE_INCUMBENT=%q\n' \
      "$BOOTSTRAP_HANDOFF_TARGET" "$PRESERVE_REASONABLE_INCUMBENT"
    printf 'LOCK_UPSTREAM_ANSWER_AGENTS=%q\n' "$LOCK_UPSTREAM_ANSWER_AGENTS"
    printf 'STRUCTURED_OUTPUTS_CONFIG=%q\n' "$STRUCTURED_OUTPUTS_CONFIG"
  } >"$temporary"
  mv -f "$temporary" "$CONFIG_PATH"
}

validate
if [[ "$DRY_RUN" == "1" ]]; then
  echo "MATH persistent-concurrent dry-run: mode=$SCHEDULER_MODE GPUs=A1[$A1_GPU_IDS] A2[$A2_GPU_IDS] A3[$A3_GPU_IDS]"
  exit 0
fi
mkdir -p "$RUN_DIR" "$(dirname "$OUTPUT_PATH")"
exec > >(tee -a "$LOG_PATH") 2>&1

if [[ "$RESUME" == "1" ]]; then
  [[ -s "$STATE_PATH" ]] || fatal "RESUME=1 requires state: $STATE_PATH"
else
  [[ ! -e "$STATE_PATH" ]] || fatal "new run refuses existing state: $STATE_PATH"
  init_cmd=(init --state "$STATE_PATH" --data-root "$DATA_ROOT" --split "$SPLIT"
    --start "$START" --limit "$LIMIT" --num-rollouts "$NUM_ROLLOUTS" --t-max "$T_MAX"
    --start-agent "$START_AGENT" --start-agent-seed "$START_AGENT_SEED"
    --bootstrap-agent "$BOOTSTRAP_AGENT" --min-agents-before-stop "$MIN_AGENTS_BEFORE_STOP"
    --generation-seed "$GENERATION_SEED" --max-new-tokens "$MAX_NEW_TOKENS"
    --protocol-thinking-max-tokens "$PROTOCOL_THINKING_MAX_TOKENS" \
    --protocol-max-tokens "$PROTOCOL_MAX_TOKENS" \
    --temperature "$TEMPERATURE" --top-p "$TOP_P" --api-timeout "$API_TIMEOUT"
    --group-retries "$GROUP_RETRIES" --step-retries "$STEP_RETRIES"
    --json-transport "$JSON_TRANSPORT" --output-mode "$OUTPUT_MODE")
  [[ -n "$SUBJECTS" ]] && init_cmd+=(--subjects "$SUBJECTS")
  [[ -n "$BOOTSTRAP_HANDOFF_TARGET" ]] && \
    init_cmd+=(--bootstrap-handoff-target "$BOOTSTRAP_HANDOFF_TARGET")
  [[ -n "$LOCK_UPSTREAM_ANSWER_AGENTS" ]] && \
    init_cmd+=(--lock-upstream-answer-agents "$LOCK_UPSTREAM_ANSWER_AGENTS")
  [[ "$PRESERVE_REASONABLE_INCUMBENT" == "1" ]] && \
    init_cmd+=(--preserve-reasonable-incumbent) || \
    init_cmd+=(--no-preserve-reasonable-incumbent)
  [[ "$ALLOW_FIRST_TURN_STOP" == "1" ]] && init_cmd+=(--allow-first-turn-stop) || init_cmd+=(--no-allow-first-turn-stop)
  [[ "$ENABLE_THINKING" == "1" ]] && init_cmd+=(--enable-thinking) || init_cmd+=(--no-enable-thinking)
  [[ "$REQUIRE_THINKING" == "1" ]] && init_cmd+=(--require-thinking) || init_cmd+=(--no-require-thinking)
  [[ "$RETRY_FAILED_GROUPS" == "1" ]] && init_cmd+=(--retry-failed-groups) || init_cmd+=(--no-retry-failed-groups)
  runner "${init_cmd[@]}"
fi

start_layout_once() {
  if identical_stack; then
    echo "[server] identical A1/A2/A3 model+adapter detected; using one shared A3 endpoint pool"
    start_agent_pool A3 "$MODEL_A3" "$ADAPTER_A3" "$A3_GPU_IDS" "$A3_PORT" \
      "$DEDUP_GPU_MEMORY_UTILIZATION"
    wait_agent_pool A3 "$A3_GPU_IDS" "$A3_PORT"
  elif [[ "$SCHEDULER_MODE" == "shared_all_gpu" ]]; then
    start_agent_pool A3 "$MODEL_A3" "$ADAPTER_A3" "$A3_GPU_IDS" "$A3_PORT" \
      "$A3_GPU_MEMORY_UTILIZATION"
    wait_agent_pool A3 "$A3_GPU_IDS" "$A3_PORT"
    start_agent_pool A2 "$MODEL_A2" "$ADAPTER_A2" "$A2_GPU_IDS" "$A2_PORT" \
      "$A2_GPU_MEMORY_UTILIZATION"
    wait_agent_pool A2 "$A2_GPU_IDS" "$A2_PORT"
    start_agent_pool A1 "$MODEL_A1" "$ADAPTER_A1" "$A1_GPU_IDS" "$A1_PORT" \
      "$A1_GPU_MEMORY_UTILIZATION"
    wait_agent_pool A1 "$A1_GPU_IDS" "$A1_PORT"
  else
    start_agent_pool A1 "$MODEL_A1" "$ADAPTER_A1" "$A1_GPU_IDS" "$A1_PORT" \
      "$GPU_MEMORY_UTILIZATION"
    start_agent_pool A2 "$MODEL_A2" "$ADAPTER_A2" "$A2_GPU_IDS" "$A2_PORT" \
      "$GPU_MEMORY_UTILIZATION"
    start_agent_pool A3 "$MODEL_A3" "$ADAPTER_A3" "$A3_GPU_IDS" "$A3_PORT" \
      "$GPU_MEMORY_UTILIZATION"
    wait_agent_pool A1 "$A1_GPU_IDS" "$A1_PORT"
    wait_agent_pool A2 "$A2_GPU_IDS" "$A2_PORT"
    wait_agent_pool A3 "$A3_GPU_IDS" "$A3_PORT"
  fi
}

start_layout() {
  local attempt
  write_runtime_config
  for ((attempt=1; attempt<=SERVER_START_RETRIES; attempt++)); do
    if (( attempt > 1 )); then
      echo "[server] retrying layout startup attempt ${attempt}/${SERVER_START_RETRIES}"
    fi
    if start_layout_once; then
      return 0
    fi
    echo "[server] layout startup failed on attempt ${attempt}/${SERVER_START_RETRIES}; cleaning up"
    stop_servers
    if [[ "$attempt" == "$SERVER_START_RETRIES" ]]; then
      fatal "layout startup failed after ${SERVER_START_RETRIES} attempts"
    fi
    sleep "$SERVER_START_RETRY_DELAY"
    wait_idle
  done
}

refresh_api_endpoints() {
  API_BASE_A1="$(pool_api_bases "$A1_GPU_IDS" "$A1_PORT")"
  API_BASE_A2="$(pool_api_bases "$A2_GPU_IDS" "$A2_PORT")"
  API_BASE_A3="$(pool_api_bases "$A3_GPU_IDS" "$A3_PORT")"
  API_MODEL_A1=A1
  API_MODEL_A2=A2
  API_MODEL_A3=A3
  if identical_stack; then
    API_BASE_A1="$API_BASE_A3"
    API_BASE_A2="$API_BASE_A3"
    API_MODEL_A1=A3
    API_MODEL_A2=A3
  fi
  echo "[layout] endpoint pools ready: A1=$API_BASE_A1 A2=$API_BASE_A2 A3=$API_BASE_A3"
}

if [[ "$ADAPTIVE_LAYOUT" == "1" ]]; then
  compute_adaptive_layout
fi
echo "[layout] mode=$SCHEDULER_MODE selected A1/A2/A3=$(layout_key) GPUs: A1[$A1_GPU_IDS] A2[$A2_GPU_IDS] A3[$A3_GPU_IDS]"
wait_idle
start_layout
refresh_api_endpoints
REBALANCE_LIMIT=0
if [[ "$ADAPTIVE_LAYOUT" == "1" ]]; then
  REBALANCE_LIMIT="$REBALANCE_AFTER_ADVANCED"
fi

for ((pass=0; ; pass++)); do
  echo "================ adaptive concurrent pass $pass layout=$(layout_key) ================"
  runner run-concurrent --state "$STATE_PATH" \
    --api-base-a1 "$API_BASE_A1" --api-model-a1 "$API_MODEL_A1" \
    --api-base-a2 "$API_BASE_A2" --api-model-a2 "$API_MODEL_A2" \
    --api-base-a3 "$API_BASE_A3" --api-model-a3 "$API_MODEL_A3" \
    --api-key EMPTY --max-concurrency-a1 "$A1_MAX_CONCURRENCY" \
    --max-concurrency-a2 "$A2_MAX_CONCURRENCY" --max-concurrency-a3 "$A3_MAX_CONCURRENCY" \
    --progress-every "$PROGRESS_EVERY" --journal-fsync-every "$JOURNAL_FSYNC_EVERY" \
    --rebalance-after-advanced "$REBALANCE_LIMIT" \
    --max-advanced-per-pass "$MAX_ADVANCED_PER_PASS" --max-inflight "$MAX_INFLIGHT"
  retry="$(status_field retry)"
  if (( retry > 0 )); then
    runner reset-retries --state "$STATE_PATH"
  fi
  bootstrap_pending="$(status_field bootstrap_pending)"
  pending="$(status_field pending_total)"
  (( bootstrap_pending > 0 || pending > 0 )) || break
  if [[ "$ADAPTIVE_LAYOUT" == "1" ]]; then
    previous_layout="$(layout_key)"
    compute_adaptive_layout
    next_layout="$(layout_key)"
    if [[ "$next_layout" != "$previous_layout" ]]; then
      echo "[layout] rebalancing A1/A2/A3 ${previous_layout} -> ${next_layout}"
      stop_servers
      wait_idle
      start_layout
      refresh_api_endpoints
    else
      echo "[layout] keeping A1/A2/A3=${next_layout}"
    fi
  fi
done

bootstrap_pending="$(status_field bootstrap_pending)"
pending="$(status_field pending_total)"
(( bootstrap_pending == 0 && pending == 0 )) || \
  fatal "pipeline stopped with bootstrap_pending=$bootstrap_pending pending=$pending"
runner status --state "$STATE_PATH"
finalize_cmd=(finalize --state "$STATE_PATH" --output "$OUTPUT_PATH")
[[ "$RESUME" == "1" ]] && finalize_cmd+=(--resume)
runner "${finalize_cmd[@]}"
echo "MATH persistent-concurrent run complete: $OUTPUT_PATH"

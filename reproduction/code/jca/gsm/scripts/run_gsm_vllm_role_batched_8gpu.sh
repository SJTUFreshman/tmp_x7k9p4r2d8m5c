#!/usr/bin/env bash
# Cycle one LoRA-backed GSM agent at a time across all GPUs (TP=1, DP=N).

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/wangyuheng/jca}"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/qwen35/bin/python}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-/data/conda_envs/deep_research/bin/python}"
VLLM_LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH:-/data/conda_envs/deep_research/lib}"
PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-/data/wangyuheng}"
RUNNER="${RUNNER:-${PROJECT_ROOT}/gsm/scripts/run_mas_role_batched.py}"
PYTHON_COMPAT_DIR="${PYTHON_COMPAT_DIR:-${PROJECT_ROOT}/scripts/gsm_judge_rl/python_compat}"

DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/Math/data/GSM-HARD/splits/gsmhardv2_dev.jsonl}"
START="${START:-0}"
LIMIT="${LIMIT:-100}"
T_MAX="${T_MAX:-8}"
START_AGENT="${START_AGENT:-A1}"
START_AGENT_SEED="${START_AGENT_SEED:-42}"
MIN_AGENTS_BEFORE_STOP="${MIN_AGENTS_BEFORE_STOP:-3}"
ENFORCE_COLLABORATION_POLICY="${ENFORCE_COLLABORATION_POLICY:-1}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
SFT_WARMUP_PROMPT="${SFT_WARMUP_PROMPT:-0}"
SFT_CONTROLLED_GENERATION="${SFT_CONTROLLED_GENERATION:-0}"
SFT_MAX_STEP_RETRIES="${SFT_MAX_STEP_RETRIES:-2}"
SFT_MAX_VERIFIER_SIMILARITY="${SFT_MAX_VERIFIER_SIMILARITY:-0.85}"
SFT_EXTENDED_FRACTION="${SFT_EXTENDED_FRACTION:-0.4}"
SFT_PROTOCOL_SEED="${SFT_PROTOCOL_SEED:-42}"
SFT_STANDARD_MIN_HANDOFFS="${SFT_STANDARD_MIN_HANDOFFS:-2}"
SFT_EXTENDED_MIN_HANDOFFS="${SFT_EXTENDED_MIN_HANDOFFS:-3 4}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.95}"
LOG_RAW_CHARS="${LOG_RAW_CHARS:-0}"
API_TIMEOUT="${API_TIMEOUT:-120}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-64}"
GENERATION_SEED="${GENERATION_SEED:-}"
JSON_TRANSPORT="${JSON_TRANSPORT:-json_schema}"
GLOBAL_TURN_BATCHING="${GLOBAL_TURN_BATCHING:-0}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-1}"
GROUP_RETRIES="${GROUP_RETRIES:-5}"
STEP_RETRIES="${STEP_RETRIES:-4}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-512}"

MODEL_A1="${MODEL_A1:-/data/wangyuheng/models/Qwen3-1.7B}"
MODEL_A2="${MODEL_A2:-/data/wangyuheng/models/Qwen3-4B}"
MODEL_A3="${MODEL_A3:-/data/wangyuheng/models/Qwen3-8B}"
ADAPTER_A1="${ADAPTER_A1:-}"
ADAPTER_A2="${ADAPTER_A2:-}"
ADAPTER_A3="${ADAPTER_A3:-}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8201}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-8}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-0}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
MAX_LORA_RANK="${MAX_LORA_RANK:-64}"
MAX_LORAS="${MAX_LORAS:-1}"
MAX_CPU_LORAS="${MAX_CPU_LORAS:-1}"
SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-900}"
SERVER_STOP_TIMEOUT="${SERVER_STOP_TIMEOUT:-30}"
# vLLM DP workers use short-lived TCP rendezvous ports during startup.  A
# transient EADDRINUSE can therefore make an otherwise healthy server fail
# before the API endpoint is exposed; retry the whole server process group.
START_RETRIES="${START_RETRIES:-4}"

WAIT_FOR_IDLE_GPUS="${WAIT_FOR_IDLE_GPUS:-1}"
GPU_IDLE_TIMEOUT_SECONDS="${GPU_IDLE_TIMEOUT_SECONDS:-86400}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-10}"
GPU_IDLE_CHECKS="${GPU_IDLE_CHECKS:-2}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/gsm_eval_role_batched}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/gsm_eval_role_batched}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_gsm_role_batched_start${START}_n${LIMIT}}"
RUN_DIR="${RUN_DIR:-${LOG_DIR}/${RUN_ID}}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/${RUN_ID}.jsonl}"
STATE_PATH="${STATE_PATH:-${RUN_DIR}/trajectory_state.json}"
LOG_PATH="${RUN_DIR}/run.log"
CONFIG_PATH="${RUN_DIR}/config.env"

fatal() {
  echo "[fatal] $*" >&2
  exit 1
}

print_command() {
  printf '  command: '
  printf '%q ' "$@"
  printf '\n'
}

count_gpus() {
  awk -F',' '{print NF}' <<<"$1"
}

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

validate_inputs() {
  [[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || fatal "RUN_ID contains unsupported characters"
  [[ "$START_AGENT" == "A1" || "$START_AGENT" == "A2" || "$START_AGENT" == "A3" || "$START_AGENT" == "balanced" || "$START_AGENT" == "random" ]] || \
    fatal "START_AGENT must be A1, A2, A3, balanced, or random"
  [[ "$JSON_TRANSPORT" == "json_schema" || "$JSON_TRANSPORT" == "json_object" || "$JSON_TRANSPORT" == "none" ]] || \
    fatal "JSON_TRANSPORT must be json_schema, json_object, or none"
  [[ "$START" =~ ^[0-9]+$ ]] || fatal "START must be non-negative"
  [[ "$START_AGENT_SEED" =~ ^[0-9]+$ ]] || fatal "START_AGENT_SEED must be non-negative"
  for value_name in LIMIT T_MAX MIN_AGENTS_BEFORE_STOP MAX_NEW_TOKENS API_TIMEOUT \
    MAX_CONCURRENCY PORT TENSOR_PARALLEL_SIZE DATA_PARALLEL_SIZE MAX_LORA_RANK \
    MAX_LORAS MAX_CPU_LORAS SERVER_WAIT_TIMEOUT SERVER_STOP_TIMEOUT \
    START_RETRIES GPU_IDLE_TIMEOUT_SECONDS GPU_POLL_SECONDS GPU_IDLE_CHECKS NUM_ROLLOUTS \
    CHECKPOINT_EVERY; do
    is_positive_integer "${!value_name}" || fatal "$value_name must be a positive integer"
  done
  for value_name in ENFORCE_COLLABORATION_POLICY SFT_WARMUP_PROMPT \
    SFT_CONTROLLED_GENERATION ENFORCE_EAGER ENABLE_PREFIX_CACHING \
    ENABLE_CHUNKED_PREFILL WAIT_FOR_IDLE_GPUS RESUME DRY_RUN \
    GLOBAL_TURN_BATCHING ENABLE_THINKING; do
    [[ "${!value_name}" == "0" || "${!value_name}" == "1" ]] || \
      fatal "$value_name must be 0 or 1"
  done
  [[ "$GROUP_RETRIES" =~ ^[0-9]+$ ]] || fatal "GROUP_RETRIES must be non-negative"
  [[ "$STEP_RETRIES" =~ ^[0-9]+$ ]] || fatal "STEP_RETRIES must be non-negative"
  if [[ "$GLOBAL_TURN_BATCHING" == "1" ]]; then
    [[ "$START_AGENT" == "A1" || "$START_AGENT" == "A2" || "$START_AGENT" == "A3" || "$START_AGENT" == "balanced" ]] || \
      fatal "global-turn batching requires A1, A2, A3, or balanced start"
    [[ "$SFT_WARMUP_PROMPT" == "0" && "$SFT_CONTROLLED_GENERATION" == "0" ]] || \
      fatal "global-turn judge-RL batching does not support SFT generation controls"
  fi
  [[ "$MIN_AGENTS_BEFORE_STOP" -le 3 ]] || fatal "MIN_AGENTS_BEFORE_STOP cannot exceed 3"
  [[ -x "$PYTHON_BIN" ]] || fatal "PYTHON_BIN is not executable: $PYTHON_BIN"
  [[ -x "$VLLM_PYTHON_BIN" ]] || fatal "VLLM_PYTHON_BIN is not executable: $VLLM_PYTHON_BIN"
  local path
  for path in "$RUNNER" "$DATA_PATH" "$PYTHON_COMPAT_DIR" \
    "$MODEL_A1" "$MODEL_A2" "$MODEL_A3" \
    "$ADAPTER_A1" "$ADAPTER_A2" "$ADAPTER_A3"; do
    [[ -e "$path" ]] || fatal "required path not found: $path"
  done
  for path in "$ADAPTER_A1" "$ADAPTER_A2" "$ADAPTER_A3"; do
    [[ -s "$path/adapter_model.safetensors" ]] || fatal "missing adapter weights: $path"
  done
  local gpu_count
  gpu_count="$(count_gpus "$GPU_IDS")"
  [[ "$gpu_count" -eq $((TENSOR_PARALLEL_SIZE * DATA_PARALLEL_SIZE)) ]] || \
    fatal "GPU count $gpu_count != TP*DP ($TENSOR_PARALLEL_SIZE*$DATA_PARALLEL_SIZE)"
  if [[ "$RESUME" == "0" ]]; then
    [[ ! -e "$STATE_PATH" ]] || fatal "state file already exists: $STATE_PATH"
  else
    [[ -s "$STATE_PATH" ]] || fatal "RESUME=1 requires an existing state: $STATE_PATH"
  fi
  [[ ! -e "$OUTPUT_PATH" ]] || fatal "output already exists: $OUTPUT_PATH"
}

runner_command() {
  env PYTHONPATH="$PYTHONPATH_ROOT" "$PYTHON_BIN" -u "$RUNNER" "$@"
}

pending_count() {
  local agent="$1"
  runner_command pending --state "$STATE_PATH" --agent "$agent"
}

pending_turn_count() {
  local agent="$1" turn="$2"
  runner_command pending --state "$STATE_PATH" --agent "$agent" --turn "$turn"
}

item_status_count() {
  local status="$1"
  runner_command count-status --state "$STATE_PATH" --status "$status"
}

completed_steps() {
  runner_command steps --state "$STATE_PATH"
}

wait_for_idle_gpus() {
  if [[ "$WAIT_FOR_IDLE_GPUS" == "0" ]]; then
    echo "[gpu] initial idle wait disabled"
    return
  fi
  local deadline=$((SECONDS + GPU_IDLE_TIMEOUT_SECONDS))
  local idle_checks=0 gpu_processes
  echo "[gpu] waiting for $GPU_IDLE_CHECKS stable all-GPU idle checks"
  while (( idle_checks < GPU_IDLE_CHECKS )); do
    gpu_processes="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" || \
      fatal "nvidia-smi failed"
    if [[ -z "${gpu_processes//[[:space:]]/}" ]]; then
      idle_checks=$((idle_checks + 1))
      echo "[$(date '+%F %T')] all GPUs idle ($idle_checks/$GPU_IDLE_CHECKS)"
    else
      idle_checks=0
      echo "[$(date '+%F %T')] GPUs occupied; active PIDs: ${gpu_processes//$'\n'/, }"
    fi
    if (( SECONDS >= deadline )); then
      fatal "GPUs did not become idle within ${GPU_IDLE_TIMEOUT_SECONDS}s"
    fi
    if (( idle_checks < GPU_IDLE_CHECKS )); then
      sleep "$GPU_POLL_SECONDS"
    fi
  done
}

validate_vllm_data_parallel() {
  local help_text
  help_text="$(
    env \
      CUDA_VISIBLE_DEVICES="$GPU_IDS" \
      LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
      "$VLLM_PYTHON_BIN" -m vllm.entrypoints.openai.api_server --help 2>&1 || true
  )"
  [[ "$help_text" == *"--data-parallel-size"* ]] || \
    fatal "selected vLLM API server does not support --data-parallel-size"
}

ensure_endpoint_free() {
  if "$PYTHON_BIN" - "http://${HOST}:${PORT}/v1/models" <<'PY' >/dev/null 2>&1
import sys
import urllib.request

with urllib.request.urlopen(sys.argv[1], timeout=2):
    pass
PY
  then
    fatal "an API server is already listening at http://${HOST}:${PORT}/v1"
  fi
}

wait_for_server() {
  local expected_model="$1"
  "$PYTHON_BIN" - "$expected_model" "http://${HOST}:${PORT}/v1/models" "$SERVER_WAIT_TIMEOUT" <<'PY'
import json
import os
import sys
import time
import urllib.request

expected, url, timeout = sys.argv[1], sys.argv[2], float(sys.argv[3])
deadline = time.time() + timeout
last_error = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            models = [str(item["id"]) for item in json.loads(response.read())["data"]]
        if expected in models:
            print(f"{expected} ready: {models}")
            raise SystemExit(0)
        last_error = RuntimeError(f"expected {expected}, served {models}")
    except Exception as exc:
        last_error = exc
    # If the vLLM parent has already exited, waiting for the full startup
    # timeout only delays recovery from transient worker/rendezvous failures.
    # The shell wrapper exposes its process-group leader as SERVER_PID.
    server_pid = os.environ.get("VLLM_SERVER_PID")
    if server_pid:
        try:
            os.kill(int(server_pid), 0)
        except ProcessLookupError:
            print(f"server process {server_pid} exited before {expected} became ready", file=sys.stderr)
            raise SystemExit(1)
        except PermissionError:
            pass
    time.sleep(5)
print(f"server timed out waiting for {expected}: {last_error}", file=sys.stderr)
raise SystemExit(1)
PY
}

SERVER_PID=""
RUNTIME_ROOT=""
RUNTIME_ROOT_OWNED=0

stop_server() {
  local pid="${SERVER_PID:-}"
  [[ -n "$pid" ]] || return
  echo "Stopping vLLM server process group: $pid"
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  local deadline=$((SECONDS + SERVER_STOP_TIMEOUT))
  while kill -0 -- "-$pid" 2>/dev/null; do
    if (( SECONDS >= deadline )); then
      echo "Server group $pid did not stop after ${SERVER_STOP_TIMEOUT}s; sending KILL"
      kill -KILL -- "-$pid" 2>/dev/null || true
      break
    fi
    sleep 1
  done
  wait "$pid" 2>/dev/null || true
  SERVER_PID=""
}

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  stop_server
  if [[ "$RUNTIME_ROOT_OWNED" == "1" && "$RUNTIME_ROOT" == /data/tmp/jca_gsm_rb.* ]]; then
    rm -rf -- "$RUNTIME_ROOT"
  fi
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

agent_settings() {
  local agent="$1"
  case "$agent" in
    A1) ACTIVE_MODEL="$MODEL_A1"; ACTIVE_ADAPTER="$ADAPTER_A1" ;;
    A2) ACTIVE_MODEL="$MODEL_A2"; ACTIVE_ADAPTER="$ADAPTER_A2" ;;
    A3) ACTIVE_MODEL="$MODEL_A3"; ACTIVE_ADAPTER="$ADAPTER_A3" ;;
    *) fatal "unknown agent: $agent" ;;
  esac
}

start_server() {
  local agent="$1" stage_label="$2" server_log="$3"
  agent_settings "$agent"
  local -a command=(
    "$VLLM_PYTHON_BIN" -m vllm.entrypoints.openai.api_server
    --host "$HOST" --port "$PORT"
    --model "$ACTIVE_MODEL"
    --served-model-name "${agent}_base"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --data-parallel-size "$DATA_PARALLEL_SIZE"
    --dtype "$TORCH_DTYPE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$MAX_MODEL_LEN"
    --trust-remote-code
    --enable-lora
    --max-lora-rank "$MAX_LORA_RANK"
    --max-loras "$MAX_LORAS"
    --max-cpu-loras "$MAX_CPU_LORAS"
    --lora-modules "${agent}=${ACTIVE_ADAPTER}"
  )
  [[ "$ENFORCE_EAGER" == "1" ]] && command+=(--enforce-eager)
  [[ "$ENABLE_PREFIX_CACHING" == "1" ]] && command+=(--enable-prefix-caching)
  [[ "$ENABLE_CHUNKED_PREFILL" == "1" ]] && command+=(--enable-chunked-prefill)
  [[ -n "$MAX_NUM_SEQS" ]] && command+=(--max-num-seqs "$MAX_NUM_SEQS")
  [[ -n "$MAX_NUM_BATCHED_TOKENS" ]] && command+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
  if [[ -n "$EXTRA_VLLM_ARGS" ]]; then
    # shellcheck disable=SC2206
    local extra_args=($EXTRA_VLLM_ARGS)
    command+=("${extra_args[@]}")
  fi

  local attempt attempt_log stage_runtime
  for (( attempt = 1; attempt <= START_RETRIES; attempt++ )); do
    # vLLM appends a UUID to this path for its Unix socket. Keep the runtime
    # component short so global-turn labels cannot exceed sockaddr_un.sun_path.
    stage_runtime="${RUNTIME_ROOT}/s${stage}_${agent}_a${attempt}"
    mkdir -p "$stage_runtime/ray" "$stage_runtime/torchinductor"
    if (( attempt == 1 )); then
      attempt_log="$server_log"
    else
      attempt_log="${server_log%.log}.retry${attempt}.log"
    fi
    echo "Starting $agent LoRA: GPUs=$GPU_IDS TP=$TENSOR_PARALLEL_SIZE DP=$DATA_PARALLEL_SIZE (attempt $attempt/$START_RETRIES)"
    echo "  model:   $ACTIVE_MODEL"
    echo "  adapter: $ACTIVE_ADAPTER"
    echo "  log:     $attempt_log"
    printf '%q ' "${command[@]}" >"${attempt_log%.log}.command.txt"
    printf '\n' >>"${attempt_log%.log}.command.txt"
    setsid env \
      CUDA_VISIBLE_DEVICES="$GPU_IDS" \
      LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
      TMPDIR="$stage_runtime" \
      RAY_TMPDIR="$stage_runtime/ray" \
      TORCHINDUCTOR_CACHE_DIR="$stage_runtime/torchinductor" \
      PYTHONPATH="${PYTHON_COMPAT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${command[@]}" >"$attempt_log" 2>&1 &
    SERVER_PID="$!"
    if VLLM_SERVER_PID="$SERVER_PID" wait_for_server "$agent"; then
      return 0
    fi
    echo "[server] $agent startup attempt $attempt/$START_RETRIES failed; cleaning process group and retrying" >&2
    stop_server
    if (( attempt < START_RETRIES )); then
      sleep 5
    fi
  done
  fatal "$agent vLLM server failed to start after $START_RETRIES attempts"
}

validate_inputs

INIT_COMMAND=(
  env "PYTHONPATH=$PYTHONPATH_ROOT" "$PYTHON_BIN" -u "$RUNNER" init
  --state "$STATE_PATH"
  --data-path "$DATA_PATH"
  --start "$START" --limit "$LIMIT" --t-max "$T_MAX"
  --start-agent "$START_AGENT"
  --start-agent-seed "$START_AGENT_SEED"
  --min-agents-before-stop "$MIN_AGENTS_BEFORE_STOP"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE" --top-p "$TOP_P"
  --api-timeout "$API_TIMEOUT"
  --json-transport "$JSON_TRANSPORT"
)
[[ "$ENABLE_THINKING" == "1" ]] && INIT_COMMAND+=(--enable-thinking)
if [[ "$ENFORCE_COLLABORATION_POLICY" == "0" ]]; then
  INIT_COMMAND+=(--no-enforce-collaboration-policy)
else
  INIT_COMMAND+=(--enforce-collaboration-policy)
fi
[[ -n "$GENERATION_SEED" ]] && INIT_COMMAND+=(--generation-seed "$GENERATION_SEED")
if [[ "$GLOBAL_TURN_BATCHING" == "1" ]]; then
  INIT_COMMAND+=(
    --num-rollouts "$NUM_ROLLOUTS"
    --group-retries "$GROUP_RETRIES"
    --step-retries "$STEP_RETRIES"
  )
fi
if [[ "$SFT_WARMUP_PROMPT" == "1" ]]; then
  read -r -a SFT_EXTENDED_MIN_HANDOFFS_ARRAY <<<"$SFT_EXTENDED_MIN_HANDOFFS"
  INIT_COMMAND+=(
    --sft-warmup-prompt
    --sft-extended-fraction "$SFT_EXTENDED_FRACTION"
    --sft-protocol-seed "$SFT_PROTOCOL_SEED"
    --sft-standard-min-handoffs "$SFT_STANDARD_MIN_HANDOFFS"
    --sft-extended-min-handoffs "${SFT_EXTENDED_MIN_HANDOFFS_ARRAY[@]}"
  )
  if [[ "$SFT_CONTROLLED_GENERATION" == "1" ]]; then
    INIT_COMMAND+=(
      --sft-controlled-generation
      --sft-max-step-retries "$SFT_MAX_STEP_RETRIES"
      --sft-max-verifier-similarity "$SFT_MAX_VERIFIER_SIMILARITY"
    )
  fi
fi

echo "================ GSM Role-Batched Run ================="
echo "run_id:          $RUN_ID"
echo "data:            $DATA_PATH"
echo "start/limit:     $START/$LIMIT"
echo "t_max:           $T_MAX"
echo "start_agent:     $START_AGENT"
echo "start_seed:      $START_AGENT_SEED"
echo "max_concurrency: $MAX_CONCURRENCY"
echo "json_transport:  $JSON_TRANSPORT"
echo "enable_thinking:  $ENABLE_THINKING"
echo "generation_seed: ${GENERATION_SEED:-none}"
echo "global_turns:    $GLOBAL_TURN_BATCHING"
if [[ "$GLOBAL_TURN_BATCHING" == "1" ]]; then
  echo "num_rollouts:    $NUM_ROLLOUTS"
  echo "group_retries:   $GROUP_RETRIES"
  echo "step_retries:    $STEP_RETRIES"
  echo "checkpoint_every: $CHECKPOINT_EVERY"
fi
echo "GPU layout:      $GPU_IDS (TP=$TENSOR_PARALLEL_SIZE, DP=$DATA_PARALLEL_SIZE)"
if [[ "$GLOBAL_TURN_BATCHING" == "1" ]]; then
  echo "role order:      each global turn: A1 -> A2 -> A3"
else
  echo "role order:      A1 -> A2 -> A3, repeated until no pending trajectories"
fi
echo "output:          $OUTPUT_PATH"
echo "run_dir:         $RUN_DIR"
print_command "${INIT_COMMAND[@]}"
echo "========================================================="

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] no state, logs, servers, or output were created"
  exit 0
fi

mkdir -p "$RUN_DIR" "$(dirname "$OUTPUT_PATH")"
if [[ -n "${RUNTIME_ROOT_OVERRIDE:-}" ]]; then
  RUNTIME_ROOT="$RUNTIME_ROOT_OVERRIDE"
  [[ "$RUNTIME_ROOT" != "/" ]] || fatal "RUNTIME_ROOT_OVERRIDE cannot be /"
  mkdir -p "$RUNTIME_ROOT"
else
  RUNTIME_ROOT="$(mktemp -d /data/tmp/jca_gsm_rb.XXXXXX)"
  RUNTIME_ROOT_OWNED=1
fi

exec > >(tee "$LOG_PATH") 2>&1

{
  printf 'RUN_ID=%q\n' "$RUN_ID"
  printf 'DATA_PATH=%q\nSTART=%q\nLIMIT=%q\nT_MAX=%q\n' "$DATA_PATH" "$START" "$LIMIT" "$T_MAX"
  printf 'START_AGENT=%q\nSTART_AGENT_SEED=%q\nMIN_AGENTS_BEFORE_STOP=%q\n' "$START_AGENT" "$START_AGENT_SEED" "$MIN_AGENTS_BEFORE_STOP"
  printf 'MAX_CONCURRENCY=%q\nMAX_NEW_TOKENS=%q\n' "$MAX_CONCURRENCY" "$MAX_NEW_TOKENS"
  printf 'TEMPERATURE=%q\nTOP_P=%q\nGENERATION_SEED=%q\nJSON_TRANSPORT=%q\n' "$TEMPERATURE" "$TOP_P" "$GENERATION_SEED" "$JSON_TRANSPORT"
  printf 'ENABLE_THINKING=%q\n' "$ENABLE_THINKING"
  printf 'GLOBAL_TURN_BATCHING=%q\nNUM_ROLLOUTS=%q\nGROUP_RETRIES=%q\nSTEP_RETRIES=%q\nCHECKPOINT_EVERY=%q\n' \
    "$GLOBAL_TURN_BATCHING" "$NUM_ROLLOUTS" "$GROUP_RETRIES" "$STEP_RETRIES" "$CHECKPOINT_EVERY"
  printf 'GPU_IDS=%q\nTENSOR_PARALLEL_SIZE=%q\nDATA_PARALLEL_SIZE=%q\n' "$GPU_IDS" "$TENSOR_PARALLEL_SIZE" "$DATA_PARALLEL_SIZE"
  printf 'MODEL_A1=%q\nMODEL_A2=%q\nMODEL_A3=%q\n' "$MODEL_A1" "$MODEL_A2" "$MODEL_A3"
  printf 'ADAPTER_A1=%q\nADAPTER_A2=%q\nADAPTER_A3=%q\n' "$ADAPTER_A1" "$ADAPTER_A2" "$ADAPTER_A3"
  printf 'OUTPUT_PATH=%q\nSTATE_PATH=%q\nRUNTIME_ROOT=%q\n' "$OUTPUT_PATH" "$STATE_PATH" "$RUNTIME_ROOT"
} >"$CONFIG_PATH"

wait_for_idle_gpus
validate_vllm_data_parallel
ensure_endpoint_free

if [[ "$RESUME" == "0" ]]; then
  echo "[init] creating role queues"
  "${INIT_COMMAND[@]}"
else
  echo "[resume] using existing role queues: $STATE_PATH"
  if [[ "$GLOBAL_TURN_BATCHING" == "1" ]]; then
    runner_command extend-retries --state "$STATE_PATH" \
      --group-retries "$GROUP_RETRIES"
  fi
fi

stage=0
if [[ "$GLOBAL_TURN_BATCHING" == "1" ]]; then
  invocation_id="$(date '+%Y%m%d_%H%M%S')_pid$$"
  batch_pass=0
  while true; do
    failed_total="$(item_status_count failed)"
    (( failed_total == 0 )) || fatal "$failed_total trajectories exhausted group retries"
    pending_total="$(pending_count all)"
    retry_total="$(item_status_count retry)"
    if (( pending_total == 0 && retry_total == 0 )); then
      break
    fi

    echo
    echo "================ Global-turn pass $batch_pass ================"
    echo "pending=$pending_total retry_waiting=$retry_total"
    for (( turn=0; turn<T_MAX; turn++ )); do
      for agent in A1 A2 A3; do
        queued="$(pending_turn_count "$agent" "$turn")"
        if (( queued == 0 )); then
          echo "[skip] pass=$batch_pass turn=$turn agent=$agent pending=0"
          continue
        fi
        stage=$((stage + 1))
        stage_label="rb_${invocation_id}_stage_$(printf '%03d' "$stage")_pass_$(printf '%02d' "$batch_pass")_turn_$(printf '%02d' "$turn")_${agent}"
        server_log="${RUN_DIR}/${stage_label}_server.log"
        phase_log="${RUN_DIR}/${stage_label}_phase.log"
        echo
        echo "[phase] pass=$batch_pass turn=$turn stage=$stage agent=$agent queued=$queued"
        start_server "$agent" "$stage_label" "$server_log"
        PHASE_COMMAND=(
          env "PYTHONPATH=$PYTHONPATH_ROOT" "$PYTHON_BIN" -u "$RUNNER" run-agent
          --state "$STATE_PATH"
          --agent "$agent" --turn "$turn"
          --api-base "http://${HOST}:${PORT}/v1"
          --api-model "$agent"
          --api-key EMPTY
          --json-transport "$JSON_TRANSPORT"
          --max-concurrency "$MAX_CONCURRENCY"
          --checkpoint-every "$CHECKPOINT_EVERY"
          --log-raw-chars "$LOG_RAW_CHARS"
        )
        print_command "${PHASE_COMMAND[@]}"
        "${PHASE_COMMAND[@]}" 2>&1 | tee "$phase_log"
        stop_server
      done
    done

    pending_total="$(pending_count all)"
    (( pending_total == 0 )) || \
      fatal "global-turn pass left $pending_total trajectories pending"
    retry_total="$(item_status_count retry)"
    if (( retry_total > 0 )); then
      echo "[retry] restarting $retry_total incomplete trajectories from turn 0"
      runner_command reset-retries --state "$STATE_PATH"
      batch_pass=$((batch_pass + 1))
      continue
    fi
    break
  done
else
  cycle=0
  pending_total="$(pending_count all)"
  while (( pending_total > 0 )); do
    cycle=$((cycle + 1))
    (( cycle <= T_MAX )) || fatal "role loop exceeded T_MAX=$T_MAX cycles"
    echo
    echo "================ Cycle $cycle (pending=$pending_total) ================"
    steps_before_cycle="$(completed_steps)"

    for agent in A1 A2 A3; do
      queued="$(pending_count "$agent")"
      if (( queued == 0 )); then
        echo "[skip] cycle=$cycle agent=$agent pending=0"
        continue
      fi
      stage=$((stage + 1))
      stage_label="stage_$(printf '%03d' "$stage")_cycle_$(printf '%02d' "$cycle")_${agent}"
      server_log="${RUN_DIR}/${stage_label}_server.log"
      phase_log="${RUN_DIR}/${stage_label}_phase.log"
      echo
      echo "[phase] cycle=$cycle stage=$stage agent=$agent queued=$queued"
      start_server "$agent" "$stage_label" "$server_log"
      PHASE_COMMAND=(
        env "PYTHONPATH=$PYTHONPATH_ROOT" "$PYTHON_BIN" -u "$RUNNER" run-agent
        --state "$STATE_PATH"
        --agent "$agent"
        --api-base "http://${HOST}:${PORT}/v1"
        --api-model "$agent"
        --api-key EMPTY
        --json-transport "$JSON_TRANSPORT"
        --max-concurrency "$MAX_CONCURRENCY"
        --log-raw-chars "$LOG_RAW_CHARS"
      )
      print_command "${PHASE_COMMAND[@]}"
      "${PHASE_COMMAND[@]}" 2>&1 | tee "$phase_log"
      stop_server
    done

    steps_after_cycle="$(completed_steps)"
    pending_total="$(pending_count all)"
    if (( pending_total > 0 && steps_after_cycle <= steps_before_cycle )); then
      fatal "cycle $cycle made no trajectory progress"
    fi
  done
fi

echo
echo "================ Final Status ================"
runner_command status --state "$STATE_PATH"
FINALIZE_COMMAND=(
  env "PYTHONPATH=$PYTHONPATH_ROOT" "$PYTHON_BIN" -u "$RUNNER" finalize
  --state "$STATE_PATH" --output "$OUTPUT_PATH"
)
print_command "${FINALIZE_COMMAND[@]}"
"${FINALIZE_COMMAND[@]}"

echo "================================================"
echo "Done:   $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "output: $OUTPUT_PATH"
echo "state:  $STATE_PATH"
echo "log:    $LOG_PATH"
echo "================================================"

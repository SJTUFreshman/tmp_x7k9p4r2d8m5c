#!/usr/bin/env bash
# 8-GPU AgentVerse baseline launcher: start 4 vLLM OpenAI servers with
# pure base models (no LoRA), then run AgentVerse pipeline and report
# EM/F1.
#
# GPU layout:
#   A1 Qwen3-1.7B  -> GPUs 0,1   TP=2   port 8201  served=A1_base
#   A2 Qwen3-4B    -> GPUs 2,3   TP=2   port 8202  served=A2_base
#   A3 Qwen3-8B    -> GPUs 4,5   TP=2   port 8203  served=A3_base   (A3 reduced from TP=4 to TP=2 vs MAD)
#   M  Qwen3-8B    -> GPUs 6,7   TP=2   port 8204  served=Meta_8B    (Recruiter + Evaluator)
#
# Smoke test (10 problems):
#   LIMIT=10 bash baseline/MuSiQue/AgentVerse/run_vllm_8gpu.sh
#
# Full dev run:
#   LIMIT=2417 bash baseline/MuSiQue/AgentVerse/run_vllm_8gpu.sh
#
# Adjust iterations / threshold:
#   MAX_ITERATIONS=2 SCORE_THRESHOLD=7 bash baseline/MuSiQue/AgentVerse/run_vllm_8gpu.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ----------------------------- Configuration -----------------------------

PROJECT_ROOT="${PROJECT_ROOT:-/data/wangyuheng/jca}"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/qwen35/bin/python}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-/data/conda_envs/deep_research/bin/python}"
VLLM_LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH:-/data/conda_envs/deep_research/lib}"
VLLM_PYTHON_COMPAT_DIR="${VLLM_PYTHON_COMPAT_DIR:-${PROJECT_ROOT}/baseline/MuSiQue/vllm_compat}"
PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-/data/wangyuheng}"

SPLIT="${SPLIT:-dev}"
START="${START:-0}"
LIMIT="${LIMIT:-100}"
MAX_ITERATIONS="${MAX_ITERATIONS:-3}"
SCORE_THRESHOLD="${SCORE_THRESHOLD:-8}"

TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TEMPERATURE_AGENT="${TEMPERATURE_AGENT:-0.7}"
TEMPERATURE_META="${TEMPERATURE_META:-0.0}"
META_AGENT="${META_AGENT:-meta}"
META_AGENT_SEED="${META_AGENT_SEED:-42}"
TOP_P="${TOP_P:-0.95}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
LOG_RAW_CHARS="${LOG_RAW_CHARS:-0}"
API_TIMEOUT="${API_TIMEOUT:-900}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"

HOST="${HOST:-127.0.0.1}"
A1_PORT="${A1_PORT:-8201}"
A2_PORT="${A2_PORT:-8202}"
A3_PORT="${A3_PORT:-8203}"
META_PORT="${META_PORT:-8204}"

A1_GPUS="${A1_GPUS:-0,1}"
A2_GPUS="${A2_GPUS:-2,3}"
A3_GPUS="${A3_GPUS:-4,5}"
META_GPUS="${META_GPUS:-6,7}"

A1_TP="${A1_TP:-2}"
A2_TP="${A2_TP:-2}"
A3_TP="${A3_TP:-2}"
META_TP="${META_TP:-2}"

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
DISABLE_PREFIX_CACHING="${DISABLE_PREFIX_CACHING:-1}"
DISABLE_CHUNKED_PREFILL="${DISABLE_CHUNKED_PREFILL:-1}"

MODEL_A1="${MODEL_A1:-/data/wangyuheng/models/Qwen3-1.7B}"
MODEL_A2="${MODEL_A2:-/data/wangyuheng/models/Qwen3-4B}"
MODEL_A3="${MODEL_A3:-/data/wangyuheng/models/Qwen3-8B}"
MODEL_META="${MODEL_META:-/data/wangyuheng/models/Qwen3-8B}"

SERVED_A1="${SERVED_A1:-A1_base}"
SERVED_A2="${SERVED_A2:-A2_base}"
SERVED_A3="${SERVED_A3:-A3_base}"
SERVED_META="${SERVED_META:-Meta_8B}"

DATA_DIR="${DATA_DIR:-musique_data}"
OUTPUT_DIR="${OUTPUT_DIR:-baseline/MuSiQue/AgentVerse/outputs}"
LOG_DIR="${LOG_DIR:-baseline/MuSiQue/AgentVerse/logs}"

RUNNER_PATH="${RUNNER_PATH:-baseline/MuSiQue/AgentVerse/run_agentverse.py}"
CORE_MODULE_PATH="${CORE_MODULE_PATH:-baseline/MuSiQue/AgentVerse/agentverse.py}"
PROMPT_RECRUITER_PATH="${PROMPT_RECRUITER_PATH:-baseline/MuSiQue/AgentVerse/prompts/recruiter.md}"
PROMPT_AGENT_PATH="${PROMPT_AGENT_PATH:-baseline/MuSiQue/AgentVerse/prompts/agent.md}"
PROMPT_EVALUATOR_PATH="${PROMPT_EVALUATOR_PATH:-baseline/MuSiQue/AgentVerse/prompts/evaluator.md}"

SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-900}"
USE_EXISTING_SERVERS="${USE_EXISTING_SERVERS:-0}"
KEEP_SERVERS="${KEEP_SERVERS:-0}"
DRY_RUN="${DRY_RUN:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"

# ------------------------------- Utilities -------------------------------

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_av_${SPLIT}_start${START}_n${LIMIT}_iters${MAX_ITERATIONS}}"
RUN_DIR="${RUN_DIR:-${LOG_DIR}/${RUN_ID}}"
mkdir -p "$RUN_DIR"

OUTPUT_PATH="${OUTPUT_PATH:-${RUN_DIR}/results.jsonl}"
LOG_PATH="${LOG_PATH:-${RUN_DIR}/run.log}"
CONFIG_PATH="${CONFIG_PATH:-${RUN_DIR}/config.env}"
COMMAND_PATH="${COMMAND_PATH:-${RUN_DIR}/command.txt}"
SUMMARY_PATH="${SUMMARY_PATH:-${RUN_DIR}/summary.txt}"
A1_SERVER_LOG="${RUN_DIR}/A1_server.log"
A2_SERVER_LOG="${RUN_DIR}/A2_server.log"
A3_SERVER_LOG="${RUN_DIR}/A3_server.log"
META_SERVER_LOG="${RUN_DIR}/META_server.log"

SERVER_PIDS=()

exec > >(tee "$LOG_PATH") 2>&1

write_config_snapshot() {
  cat >"$CONFIG_PATH" <<EOF
RUN_ID=$RUN_ID
RUN_DIR=$RUN_DIR
OUTPUT_PATH=$OUTPUT_PATH
LOG_PATH=$LOG_PATH
PROJECT_ROOT=$PROJECT_ROOT
PYTHON_BIN=$PYTHON_BIN
VLLM_PYTHON_BIN=$VLLM_PYTHON_BIN
VLLM_LD_LIBRARY_PATH=$VLLM_LD_LIBRARY_PATH
VLLM_PYTHON_COMPAT_DIR=$VLLM_PYTHON_COMPAT_DIR
PYTHONPATH_ROOT=$PYTHONPATH_ROOT
SPLIT=$SPLIT
START=$START
LIMIT=$LIMIT
MAX_ITERATIONS=$MAX_ITERATIONS
SCORE_THRESHOLD=$SCORE_THRESHOLD
TORCH_DTYPE=$TORCH_DTYPE
MAX_NEW_TOKENS=$MAX_NEW_TOKENS
TEMPERATURE_AGENT=$TEMPERATURE_AGENT
TEMPERATURE_META=$TEMPERATURE_META
META_AGENT=$META_AGENT
META_AGENT_SEED=$META_AGENT_SEED
TOP_P=$TOP_P
ENABLE_THINKING=$ENABLE_THINKING
LOG_RAW_CHARS=$LOG_RAW_CHARS
API_TIMEOUT=$API_TIMEOUT
MAX_CONCURRENCY=$MAX_CONCURRENCY
HOST=$HOST
A1_PORT=$A1_PORT A2_PORT=$A2_PORT A3_PORT=$A3_PORT META_PORT=$META_PORT
A1_GPUS=$A1_GPUS A2_GPUS=$A2_GPUS A3_GPUS=$A3_GPUS META_GPUS=$META_GPUS
A1_TP=$A1_TP A2_TP=$A2_TP A3_TP=$A3_TP META_TP=$META_TP
GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION
MAX_MODEL_LEN=$MAX_MODEL_LEN
ENFORCE_EAGER=$ENFORCE_EAGER
DISABLE_PREFIX_CACHING=$DISABLE_PREFIX_CACHING
DISABLE_CHUNKED_PREFILL=$DISABLE_CHUNKED_PREFILL
MODEL_A1=$MODEL_A1
MODEL_A2=$MODEL_A2
MODEL_A3=$MODEL_A3
MODEL_META=$MODEL_META
SERVED_A1=$SERVED_A1
SERVED_A2=$SERVED_A2
SERVED_A3=$SERVED_A3
SERVED_META=$SERVED_META
DATA_DIR=$DATA_DIR
SERVER_WAIT_TIMEOUT=$SERVER_WAIT_TIMEOUT
KEEP_SERVERS=$KEEP_SERVERS
USE_EXISTING_SERVERS=$USE_EXISTING_SERVERS
DRY_RUN=$DRY_RUN
EXTRA_ARGS=$EXTRA_ARGS
EXTRA_VLLM_ARGS=$EXTRA_VLLM_ARGS
EOF
}

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM

  if [[ "$KEEP_SERVERS" == "1" ]]; then
    echo "KEEP_SERVERS=1, leaving vLLM servers running: ${SERVER_PIDS[*]:-}"
    exit "$exit_code"
  fi

  if [[ "${#SERVER_PIDS[@]}" -gt 0 ]]; then
    echo "Stopping vLLM server process groups: ${SERVER_PIDS[*]}"
    for pid in "${SERVER_PIDS[@]}"; do
      if kill -0 "-$pid" >/dev/null 2>&1; then
        kill -TERM "-$pid" >/dev/null 2>&1 || true
      elif kill -0 "$pid" >/dev/null 2>&1; then
        kill -TERM "$pid" >/dev/null 2>&1 || true
      fi
    done
    sleep 5
    for pid in "${SERVER_PIDS[@]}"; do
      if kill -0 "-$pid" >/dev/null 2>&1; then
        kill -KILL "-$pid" >/dev/null 2>&1 || true
      elif kill -0 "$pid" >/dev/null 2>&1; then
        kill -KILL "$pid" >/dev/null 2>&1 || true
      fi
    done
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    echo
    echo "---------------- nvidia-smi after cleanup ----------------"
    nvidia-smi || true
  fi
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

count_gpus() {
  local csv="$1"
  if [[ -z "$csv" ]]; then
    echo 0
  else
    awk -F',' '{print NF}' <<<"$csv"
  fi
}

require_dir() {
  local label="$1"
  local path="$2"
  if [[ ! -d "$path" ]]; then
    echo "ERROR: $label directory does not exist: $path" >&2
    exit 1
  fi
}

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: $label file does not exist: $path" >&2
    exit 1
  fi
}

wait_for_model() {
  local name="$1"
  local url="$2"
  local model_name="$3"
  local timeout="$4"
  "$PYTHON_BIN" - "$name" "$url" "$model_name" "$timeout" <<'PY'
import json
import sys
import time
import urllib.request

name, url, model_name, timeout = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
deadline = time.time() + timeout
last_error = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            body = response.read().decode("utf-8")
        payload = json.loads(body)
        models = [m.get("id") for m in payload.get("data", [])]
        if model_name in models:
            print(f"{name} ready: {models}")
            sys.exit(0)
        last_error = f"{model_name!r} not in model list {models!r}"
    except Exception as exc:
        last_error = exc
    time.sleep(5)
print(f"{name} did not become ready before timeout. Last error: {last_error}", file=sys.stderr)
sys.exit(1)
PY
}

server_has_model() {
  local url="$1"
  local model_name="$2"
  "$PYTHON_BIN" - "$url" "$model_name" <<'PY' >/dev/null 2>&1
import json
import sys
import urllib.request

url, model_name = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(url, timeout=3) as response:
    payload = json.loads(response.read().decode("utf-8"))
models = [m.get("id") for m in payload.get("data", [])]
if model_name not in models:
    raise SystemExit(1)
PY
}

server_ready() {
  local url="$1"
  "$PYTHON_BIN" - "$url" <<'PY' >/dev/null 2>&1
import json
import sys
import urllib.request

url = sys.argv[1]
with urllib.request.urlopen(url, timeout=3) as response:
    payload = json.loads(response.read().decode("utf-8"))
if "data" not in payload:
    raise SystemExit(1)
PY
}

check_existing_server() {
  local served_name="$1"
  local port="$2"
  local url="http://${HOST}:${port}/v1/models"

  if server_ready "$url"; then
    if [[ "$USE_EXISTING_SERVERS" == "1" ]]; then
      if server_has_model "$url" "$served_name"; then
        echo "$served_name existing server detected at $url; reusing it."
        return 0
      fi
      echo "ERROR: USE_EXISTING_SERVERS=1 but $url does not expose model $served_name." >&2
      exit 1
    fi
    echo "ERROR: port $port already has a vLLM-compatible server." >&2
    echo "Stop the old server, change ports, or set USE_EXISTING_SERVERS=1." >&2
    exit 1
  fi

  return 1
}

start_base_server() {
  local agent="$1"
  local model_path="$2"
  local served_name="$3"
  local port="$4"
  local gpus="$5"
  local tp="$6"
  local log_path="$7"

  local visible_count
  visible_count="$(count_gpus "$gpus")"
  if [[ "$visible_count" != "$tp" ]]; then
    echo "ERROR: $agent GPU count ($visible_count from $gpus) must equal TP size ($tp)." >&2
    exit 1
  fi

  local cmd=(
    "$VLLM_PYTHON_BIN" -m vllm.entrypoints.openai.api_server
    --host "$HOST"
    --port "$port"
    --model "$model_path"
    --served-model-name "$served_name"
    --tensor-parallel-size "$tp"
    --dtype "$TORCH_DTYPE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$MAX_MODEL_LEN"
    --trust-remote-code
  )
  if [[ "$ENFORCE_EAGER" == "1" ]]; then
    cmd+=(--enforce-eager)
  fi
  if [[ "$DISABLE_PREFIX_CACHING" == "1" ]]; then
    cmd+=(--no-enable-prefix-caching)
  fi
  if [[ "$DISABLE_CHUNKED_PREFILL" == "1" ]]; then
    cmd+=(--no-enable-chunked-prefill)
  fi
  if [[ -n "$EXTRA_VLLM_ARGS" ]]; then
    # shellcheck disable=SC2206
    local extra_vllm_args_array=($EXTRA_VLLM_ARGS)
    cmd+=("${extra_vllm_args_array[@]}")
  fi

  echo "Starting $agent base server on GPUs $gpus, TP=$tp, port=$port"
  echo "  model:  $model_path"
  echo "  served: $served_name"
  echo "  log:    $log_path"
  setsid env \
    CUDA_VISIBLE_DEVICES="$gpus" \
    LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    PYTHONPATH="${VLLM_PYTHON_COMPAT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${cmd[@]}" >"$log_path" 2>&1 &
  local pid=$!
  SERVER_PIDS+=("$pid")
  echo "  process_group: $pid"
}

build_runner_command() {
  RUN_CMD=(
    "$PYTHON_BIN" "$RUNNER_PATH"
    --split "$SPLIT"
    --data-dir "$DATA_DIR"
    --start "$START"
    --limit "$LIMIT"
    --max-iterations "$MAX_ITERATIONS"
    --score-threshold "$SCORE_THRESHOLD"
    --max-new-tokens "$MAX_NEW_TOKENS"
    --temperature-agent "$TEMPERATURE_AGENT"
    --temperature-meta "$TEMPERATURE_META"
    --meta-agent "$META_AGENT"
    --meta-agent-seed "$META_AGENT_SEED"
    --top-p "$TOP_P"
    --api-base-a1 "http://${HOST}:${A1_PORT}/v1"
    --api-base-a2 "http://${HOST}:${A2_PORT}/v1"
    --api-base-a3 "http://${HOST}:${A3_PORT}/v1"
    --api-base-meta "http://${HOST}:${META_PORT}/v1"
    --api-model-a1 "$SERVED_A1"
    --api-model-a2 "$SERVED_A2"
    --api-model-a3 "$SERVED_A3"
    --api-model-meta "$SERVED_META"
    --api-timeout "$API_TIMEOUT"
    --max-concurrency "$MAX_CONCURRENCY"
    --output "$OUTPUT_PATH"
    --log-raw-chars "$LOG_RAW_CHARS"
  )
  if [[ "$ENABLE_THINKING" == "1" ]]; then
    RUN_CMD+=(--enable-thinking)
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    RUN_CMD+=(--dry-run)
  fi
  if [[ -n "$EXTRA_ARGS" ]]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS_ARRAY=($EXTRA_ARGS)
    RUN_CMD+=("${EXTRA_ARGS_ARRAY[@]}")
  fi
}

# ------------------------------- Execution -------------------------------

echo "================ AgentVerse (base + 8B meta) 8-GPU vLLM Run ================"
echo "time: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "host: $(hostname)"
echo "project_root: $PROJECT_ROOT"
echo "python: $PYTHON_BIN"
echo "vllm_python: $VLLM_PYTHON_BIN"
echo "run_id: $RUN_ID"
echo "run_dir: $RUN_DIR"
echo "output_path: $OUTPUT_PATH"
echo "log_path: $LOG_PATH"
echo "config_path: $CONFIG_PATH"
echo "command_path: $COMMAND_PATH"
echo "summary_path: $SUMMARY_PATH"
echo "server_logs:"
echo "  A1:   $A1_SERVER_LOG"
echo "  A2:   $A2_SERVER_LOG"
echo "  A3:   $A3_SERVER_LOG"
echo "  META: $META_SERVER_LOG"
echo

echo "---------------- Configuration ----------------"
write_config_snapshot
echo "Config snapshot written to $CONFIG_PATH"
echo "SPLIT=$SPLIT START=$START LIMIT=$LIMIT MAX_ITERATIONS=$MAX_ITERATIONS SCORE_THRESHOLD=$SCORE_THRESHOLD MAX_CONCURRENCY=$MAX_CONCURRENCY"
echo "TEMPERATURE_AGENT=$TEMPERATURE_AGENT TEMPERATURE_META=$TEMPERATURE_META"
echo "TORCH_DTYPE=$TORCH_DTYPE MAX_NEW_TOKENS=$MAX_NEW_TOKENS TOP_P=$TOP_P"
echo "ENABLE_THINKING=$ENABLE_THINKING LOG_RAW_CHARS=$LOG_RAW_CHARS API_TIMEOUT=$API_TIMEOUT"
echo "HOST=$HOST"
echo "A1:   model=$MODEL_A1  served=$SERVED_A1  port=$A1_PORT   gpus=$A1_GPUS   tp=$A1_TP"
echo "A2:   model=$MODEL_A2  served=$SERVED_A2  port=$A2_PORT   gpus=$A2_GPUS   tp=$A2_TP"
echo "A3:   model=$MODEL_A3  served=$SERVED_A3  port=$A3_PORT   gpus=$A3_GPUS   tp=$A3_TP"
echo "META: model=$MODEL_META served=$SERVED_META port=$META_PORT gpus=$META_GPUS tp=$META_TP"
echo "GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION MAX_MODEL_LEN=$MAX_MODEL_LEN ENFORCE_EAGER=$ENFORCE_EAGER"
echo "DISABLE_PREFIX_CACHING=$DISABLE_PREFIX_CACHING DISABLE_CHUNKED_PREFILL=$DISABLE_CHUNKED_PREFILL"
echo "SERVER_WAIT_TIMEOUT=$SERVER_WAIT_TIMEOUT USE_EXISTING_SERVERS=$USE_EXISTING_SERVERS KEEP_SERVERS=$KEEP_SERVERS DRY_RUN=$DRY_RUN"
echo "EXTRA_ARGS=$EXTRA_ARGS"
echo "EXTRA_VLLM_ARGS=$EXTRA_VLLM_ARGS"
echo

echo "---------------- Preflight ----------------"
require_dir "A1 model" "$MODEL_A1"
require_dir "A2 model" "$MODEL_A2"
require_dir "A3 model" "$MODEL_A3"
require_dir "META (8B) model" "$MODEL_META"
require_file "AgentVerse runner" "$RUNNER_PATH"
require_file "AgentVerse module" "$CORE_MODULE_PATH"
require_file "recruiter prompt" "$PROMPT_RECRUITER_PATH"
require_file "agent prompt" "$PROMPT_AGENT_PATH"
require_file "evaluator prompt" "$PROMPT_EVALUATOR_PATH"
require_file "vLLM tokenizer compatibility shim" "$VLLM_PYTHON_COMPAT_DIR/sitecustomize.py"

"$PYTHON_BIN" --version
env \
  LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  PYTHONPATH="${VLLM_PYTHON_COMPAT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  "$VLLM_PYTHON_BIN" - <<'PY'
import importlib.util
import importlib.metadata

for name in ["torch", "transformers", "vllm"]:
    spec = importlib.util.find_spec(name)
    print(f"{name}: {'OK' if spec else 'MISSING'}")
    if spec:
        distribution = "vllm" if name == "vllm" else name
        try:
            print(f"{name}_version: {importlib.metadata.version(distribution)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{name}_version: unknown")
PY
if command -v nvidia-smi >/dev/null 2>&1; then
  echo
  echo "---------------- nvidia-smi before ----------------"
  nvidia-smi || true
else
  echo "nvidia-smi: not found"
fi

build_runner_command
echo
echo "---------------- Runner Command ----------------"
printf '%q ' "${RUN_CMD[@]}"
echo
{
  printf 'PYTHONPATH=%q ' "$PYTHONPATH_ROOT"
  printf '%q ' "${RUN_CMD[@]}"
  echo
} >"$COMMAND_PATH"
echo "Runner command written to $COMMAND_PATH"

if [[ "$DRY_RUN" == "1" ]]; then
  echo
  echo "---------------- Dry Run Output ----------------"
  PYTHONPATH="$PYTHONPATH_ROOT" "${RUN_CMD[@]}"
  exit 0
fi

echo
echo "---------------- Starting vLLM Base Servers ----------------"
if [[ "$USE_EXISTING_SERVERS" == "1" ]]; then
  check_existing_server "$SERVED_A1"   "$A1_PORT"   || { echo "ERROR: A1 not ready on $A1_PORT" >&2; exit 1; }
  check_existing_server "$SERVED_A2"   "$A2_PORT"   || { echo "ERROR: A2 not ready on $A2_PORT" >&2; exit 1; }
  check_existing_server "$SERVED_A3"   "$A3_PORT"   || { echo "ERROR: A3 not ready on $A3_PORT" >&2; exit 1; }
  check_existing_server "$SERVED_META" "$META_PORT" || { echo "ERROR: META not ready on $META_PORT" >&2; exit 1; }
else
  check_existing_server "$SERVED_A1"   "$A1_PORT"   || true
  check_existing_server "$SERVED_A2"   "$A2_PORT"   || true
  check_existing_server "$SERVED_A3"   "$A3_PORT"   || true
  check_existing_server "$SERVED_META" "$META_PORT" || true
  start_base_server "A1"   "$MODEL_A1"   "$SERVED_A1"   "$A1_PORT"   "$A1_GPUS"   "$A1_TP"   "$A1_SERVER_LOG"
  start_base_server "A2"   "$MODEL_A2"   "$SERVED_A2"   "$A2_PORT"   "$A2_GPUS"   "$A2_TP"   "$A2_SERVER_LOG"
  start_base_server "A3"   "$MODEL_A3"   "$SERVED_A3"   "$A3_PORT"   "$A3_GPUS"   "$A3_TP"   "$A3_SERVER_LOG"
  start_base_server "META" "$MODEL_META" "$SERVED_META" "$META_PORT" "$META_GPUS" "$META_TP" "$META_SERVER_LOG"
fi

echo
echo "---------------- Waiting for Base Models ----------------"
wait_for_model "A1"   "http://${HOST}:${A1_PORT}/v1/models"   "$SERVED_A1"   "$SERVER_WAIT_TIMEOUT"
wait_for_model "A2"   "http://${HOST}:${A2_PORT}/v1/models"   "$SERVED_A2"   "$SERVER_WAIT_TIMEOUT"
wait_for_model "A3"   "http://${HOST}:${A3_PORT}/v1/models"   "$SERVED_A3"   "$SERVER_WAIT_TIMEOUT"
wait_for_model "META" "http://${HOST}:${META_PORT}/v1/models" "$SERVED_META" "$SERVER_WAIT_TIMEOUT"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo
  echo "---------------- nvidia-smi after server start ----------------"
  nvidia-smi || true
fi

echo
echo "---------------- Run Output ----------------"
PYTHONPATH="$PYTHONPATH_ROOT" "${RUN_CMD[@]}"

echo
echo "---------------- Completed ----------------"
echo "time: $(date '+%Y-%m-%d %H:%M:%S %Z')"
{
  echo "RUN_ID=$RUN_ID"
  echo "RUN_DIR=$RUN_DIR"
  echo "OUTPUT_PATH=$OUTPUT_PATH"
  echo "LOG_PATH=$LOG_PATH"
  echo "A1_SERVER_LOG=$A1_SERVER_LOG"
  echo "A2_SERVER_LOG=$A2_SERVER_LOG"
  echo "A3_SERVER_LOG=$A3_SERVER_LOG"
  echo "META_SERVER_LOG=$META_SERVER_LOG"
  echo "CONFIG_PATH=$CONFIG_PATH"
  echo "COMMAND_PATH=$COMMAND_PATH"
  echo "COMPLETED_AT=$(date '+%Y-%m-%d %H:%M:%S %Z')"
} >"$SUMMARY_PATH"
echo "Summary written to $SUMMARY_PATH"

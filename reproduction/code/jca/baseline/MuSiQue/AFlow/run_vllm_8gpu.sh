#!/usr/bin/env bash
# 8-GPU AFlow launcher: heterogeneous solver pool plus a 14B optimizer.
#
# GPU layout:
#   A1 Qwen3-1.7B -> GPUs 0,1 TP=2 port 8201
#   A2 Qwen3-4B   -> GPUs 2,3 TP=2 port 8202
#   A3 Qwen3-8B   -> GPUs 4,5 TP=2 port 8203 (also the judge)
#   Optimizer 14B -> GPUs 6,7 TP=2 port 8204
#
# Modes (choose via MODE env var):
#   MODE=search   run MCTS search (default)
#   MODE=eval     evaluate one workflow file on a split
#   MODE=both     search first, then eval best workflow on --limit problems
#
# Examples:
#   # 20-iter search on 20 dev problems:
#   MODE=search MAX_ITERATIONS=20 DEV_SIZE=20 bash baseline/MuSiQue/AFlow/run_vllm_8gpu.sh
#
#   # Eval a specific workflow on full dev:
#   MODE=eval WORKFLOW_FILE=baseline/MuSiQue/AFlow/workflows/round_00_initial.py \
#       LIMIT=2417 bash baseline/MuSiQue/AFlow/run_vllm_8gpu.sh
#
#   # Full pipeline: search then eval best on full dev
#   MODE=both LIMIT=2417 bash baseline/MuSiQue/AFlow/run_vllm_8gpu.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ----------------------------- Configuration -----------------------------

PROJECT_ROOT="${PROJECT_ROOT:-/data/wangyuheng/jca}"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/qwen35/bin/python}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-/data/conda_envs/deep_research/bin/python}"
VLLM_LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH:-/data/conda_envs/deep_research/lib}"
VLLM_PYTHON_COMPAT_DIR="${VLLM_PYTHON_COMPAT_DIR:-${PROJECT_ROOT}/baseline/MuSiQue/vllm_compat}"
PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-/data/wangyuheng}"

MODE="${MODE:-search}"

# Common
SPLIT="${SPLIT:-dev}"
DATA_DIR="${DATA_DIR:-musique_data}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-32}"

# Search-only
MAX_ITERATIONS="${MAX_ITERATIONS:-20}"
DEV_SIZE="${DEV_SIZE:-20}"
DEV_SHUFFLE_SEED="${DEV_SHUFFLE_SEED:-20260810}"
RNG_SEED="${RNG_SEED:-20260810}"
INITIAL_WORKFLOW="${INITIAL_WORKFLOW:-baseline/MuSiQue/AFlow/workflows/round_00_initial.py}"
SEARCH_RUN_DIR="${SEARCH_RUN_DIR:-}"   # optional; else auto-timestamped inside run_search.py

# Eval-only
WORKFLOW_FILE="${WORKFLOW_FILE:-baseline/MuSiQue/AFlow/workflows/round_00_initial.py}"
START="${START:-0}"
LIMIT="${LIMIT:-2417}"
EVAL_OUTPUT_PATH="${EVAL_OUTPUT_PATH:-}"

# Model
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
MAX_NEW_TOKENS_EXEC="${MAX_NEW_TOKENS_EXEC:-1024}"
MAX_NEW_TOKENS_OPT="${MAX_NEW_TOKENS_OPT:-2048}"
TEMPERATURE_EXEC="${TEMPERATURE_EXEC:-0.7}"
TEMPERATURE_OPT="${TEMPERATURE_OPT:-0.8}"
JUDGE_AGENT="${JUDGE_AGENT:-A3}"
JUDGE_AGENT_SEED="${JUDGE_AGENT_SEED:-42}"
TOP_P="${TOP_P:-0.95}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
API_TIMEOUT="${API_TIMEOUT:-900}"

HOST="${HOST:-127.0.0.1}"
 A1_PORT="${A1_PORT:-8201}"
 A2_PORT="${A2_PORT:-8202}"
 A3_PORT="${A3_PORT:-8203}"
OPT_PORT="${OPT_PORT:-8204}"

A1_GPUS="${A1_GPUS:-0,1}"
A2_GPUS="${A2_GPUS:-2,3}"
A3_GPUS="${A3_GPUS:-4,5}"
OPT_GPUS="${OPT_GPUS:-6,7}"
A1_TP="${A1_TP:-2}"
A2_TP="${A2_TP:-2}"
A3_TP="${A3_TP:-2}"
OPT_TP="${OPT_TP:-2}"

GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
DISABLE_PREFIX_CACHING="${DISABLE_PREFIX_CACHING:-1}"
DISABLE_CHUNKED_PREFILL="${DISABLE_CHUNKED_PREFILL:-1}"

MODEL_A1="${MODEL_A1:-/data/wangyuheng/models/Qwen3-1.7B}"
MODEL_A2="${MODEL_A2:-/data/wangyuheng/models/Qwen3-4B}"
MODEL_A3="${MODEL_A3:-/data/wangyuheng/models/Qwen3-8B}"
MODEL_OPT="${MODEL_OPT:-/data/wangyuheng/models/Qwen3-14B}"

SERVED_A1="${SERVED_A1:-A1_base}"
SERVED_A2="${SERVED_A2:-A2_base}"
SERVED_A3="${SERVED_A3:-A3_base}"
SERVED_OPT="${SERVED_OPT:-Optimizer_14B}"

RUNNER_SEARCH="${RUNNER_SEARCH:-baseline/MuSiQue/AFlow/run_search.py}"
RUNNER_EVAL="${RUNNER_EVAL:-baseline/MuSiQue/AFlow/run_eval.py}"

LOG_DIR="${LOG_DIR:-baseline/MuSiQue/AFlow/logs}"
SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-900}"
USE_EXISTING_SERVERS="${USE_EXISTING_SERVERS:-0}"
KEEP_SERVERS="${KEEP_SERVERS:-0}"
DRY_RUN="${DRY_RUN:-0}"

# ------------------------------- Utilities -------------------------------

cd "$PROJECT_ROOT"
mkdir -p "$LOG_DIR"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$_aflow_${MODE}}"
RUN_DIR="${RUN_DIR:-${LOG_DIR}/${RUN_ID}}"
mkdir -p "$RUN_DIR"

if [[ -z "$EVAL_OUTPUT_PATH" && ( "$MODE" == "eval" || "$MODE" == "both" ) ]]; then
  EVAL_OUTPUT_PATH="baseline/MuSiQue/AFlow/outputs/${RUN_ID}.jsonl"
fi

LOG_PATH="${LOG_PATH:-${RUN_DIR}/run.log}"
CONFIG_PATH="${CONFIG_PATH:-${RUN_DIR}/config.env}"
COMMAND_PATH="${COMMAND_PATH:-${RUN_DIR}/command.txt}"
SUMMARY_PATH="${SUMMARY_PATH:-${RUN_DIR}/summary.txt}"
A1_SERVER_LOG="${RUN_DIR}/A1_server.log"
A2_SERVER_LOG="${RUN_DIR}/A2_server.log"
A3_SERVER_LOG="${RUN_DIR}/A3_server.log"
OPT_SERVER_LOG="${RUN_DIR}/OPT_server.log"

SERVER_PIDS=()

exec > >(tee "$LOG_PATH") 2>&1

write_config_snapshot() {
  cat >"$CONFIG_PATH" <<EOF
RUN_ID=$RUN_ID
RUN_DIR=$RUN_DIR
MODE=$MODE
PROJECT_ROOT=$PROJECT_ROOT
PYTHON_BIN=$PYTHON_BIN
VLLM_PYTHON_BIN=$VLLM_PYTHON_BIN
VLLM_LD_LIBRARY_PATH=$VLLM_LD_LIBRARY_PATH
VLLM_PYTHON_COMPAT_DIR=$VLLM_PYTHON_COMPAT_DIR
PYTHONPATH_ROOT=$PYTHONPATH_ROOT
SPLIT=$SPLIT
DATA_DIR=$DATA_DIR
MAX_CONCURRENCY=$MAX_CONCURRENCY
MAX_ITERATIONS=$MAX_ITERATIONS
DEV_SIZE=$DEV_SIZE
DEV_SHUFFLE_SEED=$DEV_SHUFFLE_SEED
RNG_SEED=$RNG_SEED
INITIAL_WORKFLOW=$INITIAL_WORKFLOW
SEARCH_RUN_DIR=$SEARCH_RUN_DIR
WORKFLOW_FILE=$WORKFLOW_FILE
START=$START
LIMIT=$LIMIT
EVAL_OUTPUT_PATH=$EVAL_OUTPUT_PATH
TORCH_DTYPE=$TORCH_DTYPE
MAX_NEW_TOKENS_EXEC=$MAX_NEW_TOKENS_EXEC
MAX_NEW_TOKENS_OPT=$MAX_NEW_TOKENS_OPT
TEMPERATURE_EXEC=$TEMPERATURE_EXEC
TEMPERATURE_OPT=$TEMPERATURE_OPT
JUDGE_AGENT=$JUDGE_AGENT
JUDGE_AGENT_SEED=$JUDGE_AGENT_SEED
TOP_P=$TOP_P
ENABLE_THINKING=$ENABLE_THINKING
API_TIMEOUT=$API_TIMEOUT
HOST=$HOST
A1_PORT=$A1_PORT A2_PORT=$A2_PORT A3_PORT=$A3_PORT OPT_PORT=$OPT_PORT
A1_GPUS=$A1_GPUS A2_GPUS=$A2_GPUS A3_GPUS=$A3_GPUS OPT_GPUS=$OPT_GPUS
A1_TP=$A1_TP A2_TP=$A2_TP A3_TP=$A3_TP OPT_TP=$OPT_TP
GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION
MAX_MODEL_LEN=$MAX_MODEL_LEN
ENFORCE_EAGER=$ENFORCE_EAGER
DISABLE_PREFIX_CACHING=$DISABLE_PREFIX_CACHING
DISABLE_CHUNKED_PREFILL=$DISABLE_CHUNKED_PREFILL
MODEL_A1=$MODEL_A1
MODEL_A2=$MODEL_A2
MODEL_A3=$MODEL_A3
MODEL_OPT=$MODEL_OPT
SERVED_A1=$SERVED_A1
SERVED_A2=$SERVED_A2
SERVED_A3=$SERVED_A3
SERVED_OPT=$SERVED_OPT
SERVER_WAIT_TIMEOUT=$SERVER_WAIT_TIMEOUT
KEEP_SERVERS=$KEEP_SERVERS
USE_EXISTING_SERVERS=$USE_EXISTING_SERVERS
DRY_RUN=$DRY_RUN
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
        echo "$served_name existing server at $url; reusing it."
        return 0
      fi
      echo "ERROR: USE_EXISTING_SERVERS=1 but $url does not expose model $served_name." >&2
      exit 1
    fi
    echo "ERROR: port $port already has a vLLM-compatible server." >&2
    echo "Stop it, change ports, or set USE_EXISTING_SERVERS=1." >&2
    exit 1
  fi

  return 1
}

start_base_server() {
  local tag="$1"
  local model_path="$2"
  local served_name="$3"
  local port="$4"
  local gpus="$5"
  local tp="$6"
  local log_path="$7"

  local visible_count
  visible_count="$(count_gpus "$gpus")"
  if [[ "$visible_count" != "$tp" ]]; then
    echo "ERROR: $tag GPU count ($visible_count from $gpus) must equal TP size ($tp)." >&2
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

  echo "Starting $tag server on GPUs $gpus, TP=$tp, port=$port"
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

# ------------------------------- Execution -------------------------------

echo "================ AFlow (A1/A2/A3 solvers + 14B optimizer) 8-GPU vLLM Run ================"
echo "time: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "mode: $MODE"
echo "host: $(hostname)"
echo "project_root: $PROJECT_ROOT"
echo "run_id: $RUN_ID"
echo "run_dir: $RUN_DIR"
echo

echo "---------------- Configuration ----------------"
write_config_snapshot
echo "Config snapshot written to $CONFIG_PATH"
[[ "$ENABLE_THINKING" == 0 || "$ENABLE_THINKING" == 1 ]] || {
  echo "ERROR: ENABLE_THINKING must be 0 or 1" >&2
  exit 1
}
case "$MODE" in
  search) echo "MAX_ITERATIONS=$MAX_ITERATIONS DEV_SIZE=$DEV_SIZE" ;;
  eval)   echo "WORKFLOW_FILE=$WORKFLOW_FILE START=$START LIMIT=$LIMIT" ;;
  both)   echo "MAX_ITERATIONS=$MAX_ITERATIONS DEV_SIZE=$DEV_SIZE  then  LIMIT=$LIMIT" ;;
  *) echo "ERROR: unknown MODE=$MODE (expected search|eval|both)" >&2; exit 1 ;;
esac
echo "MAX_CONCURRENCY=$MAX_CONCURRENCY  TORCH_DTYPE=$TORCH_DTYPE"
echo "ENABLE_THINKING=$ENABLE_THINKING"
echo "JUDGE_AGENT=$JUDGE_AGENT JUDGE_AGENT_SEED=$JUDGE_AGENT_SEED"
echo "A1:    model=$MODEL_A1 served=$SERVED_A1 port=$A1_PORT gpus=$A1_GPUS tp=$A1_TP"
echo "A2:    model=$MODEL_A2 served=$SERVED_A2 port=$A2_PORT gpus=$A2_GPUS tp=$A2_TP"
echo "A3:    model=$MODEL_A3 served=$SERVED_A3 port=$A3_PORT gpus=$A3_GPUS tp=$A3_TP"
echo "OPT:   model=$MODEL_OPT  served=$SERVED_OPT  port=$OPT_PORT  gpus=$OPT_GPUS  tp=$OPT_TP"
echo "GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION  MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo

echo "---------------- Preflight ----------------"
require_dir "A1 model" "$MODEL_A1"
require_dir "A2 model" "$MODEL_A2"
require_dir "A3 model" "$MODEL_A3"
require_dir "optimizer model" "$MODEL_OPT"
require_file "search runner" "$RUNNER_SEARCH"
require_file "eval runner"   "$RUNNER_EVAL"
require_file "initial workflow" "$INITIAL_WORKFLOW"
require_file "vLLM tokenizer compatibility shim" "$VLLM_PYTHON_COMPAT_DIR/sitecustomize.py"
if [[ "$MODE" == "eval" ]]; then
  require_file "workflow to eval" "$WORKFLOW_FILE"
fi

"$PYTHON_BIN" --version
env \
  LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
  PYTHONPATH="${VLLM_PYTHON_COMPAT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
  "$VLLM_PYTHON_BIN" - <<'PY'
import importlib.util
for name in ["torch", "transformers", "vllm"]:
    spec = importlib.util.find_spec(name)
    print(f"{name}: {'OK' if spec else 'MISSING'}")
if importlib.util.find_spec("torch"):
    import torch
    print(f"torch_version: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"cuda_device_count: {torch.cuda.device_count()}")
if importlib.util.find_spec("vllm"):
    import vllm
    print(f"vllm_version: {getattr(vllm, '__version__', 'unknown')}")
PY
if command -v nvidia-smi >/dev/null 2>&1; then
  echo
  echo "---------------- nvidia-smi before ----------------"
  nvidia-smi || true
else
  echo "nvidia-smi: not found"
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo
  echo "DRY_RUN=1, preflight passed; skipping vLLM startup and runners."
  exit 0
fi

# Optimizer server only started when we need it (search or both).
NEED_OPT_SERVER=0
if [[ "$MODE" == "search" || "$MODE" == "both" ]]; then
  NEED_OPT_SERVER=1
fi

echo
echo "---------------- Starting vLLM Base Servers ----------------"
if [[ "$USE_EXISTING_SERVERS" == "1" ]]; then
  check_existing_server "$SERVED_A1" "$A1_PORT" || { echo "ERROR: A1 not ready on $A1_PORT" >&2; exit 1; }
  check_existing_server "$SERVED_A2" "$A2_PORT" || { echo "ERROR: A2 not ready on $A2_PORT" >&2; exit 1; }
  check_existing_server "$SERVED_A3" "$A3_PORT" || { echo "ERROR: A3 not ready on $A3_PORT" >&2; exit 1; }
  if [[ "$NEED_OPT_SERVER" == "1" ]]; then
    check_existing_server "$SERVED_OPT" "$OPT_PORT" || { echo "ERROR: opt not ready on $OPT_PORT" >&2; exit 1; }
  fi
else
  check_existing_server "$SERVED_A1" "$A1_PORT" || true
  start_base_server "A1" "$MODEL_A1" "$SERVED_A1" "$A1_PORT" "$A1_GPUS" "$A1_TP" "$A1_SERVER_LOG"
  check_existing_server "$SERVED_A2" "$A2_PORT" || true
  start_base_server "A2" "$MODEL_A2" "$SERVED_A2" "$A2_PORT" "$A2_GPUS" "$A2_TP" "$A2_SERVER_LOG"
  check_existing_server "$SERVED_A3" "$A3_PORT" || true
  start_base_server "A3" "$MODEL_A3" "$SERVED_A3" "$A3_PORT" "$A3_GPUS" "$A3_TP" "$A3_SERVER_LOG"
  if [[ "$NEED_OPT_SERVER" == "1" ]]; then
    check_existing_server "$SERVED_OPT" "$OPT_PORT" || true
    start_base_server "OPT" "$MODEL_OPT" "$SERVED_OPT" "$OPT_PORT" "$OPT_GPUS" "$OPT_TP" "$OPT_SERVER_LOG"
  fi
fi

echo
echo "---------------- Waiting for Base Models ----------------"
wait_for_model "A1" "http://${HOST}:${A1_PORT}/v1/models" "$SERVED_A1" "$SERVER_WAIT_TIMEOUT"
wait_for_model "A2" "http://${HOST}:${A2_PORT}/v1/models" "$SERVED_A2" "$SERVER_WAIT_TIMEOUT"
wait_for_model "A3" "http://${HOST}:${A3_PORT}/v1/models" "$SERVED_A3" "$SERVER_WAIT_TIMEOUT"
if [[ "$NEED_OPT_SERVER" == "1" ]]; then
  wait_for_model "OPT"  "http://${HOST}:${OPT_PORT}/v1/models"  "$SERVED_OPT"  "$SERVER_WAIT_TIMEOUT"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo
  echo "---------------- nvidia-smi after server start ----------------"
  nvidia-smi || true
fi

# ------------------------------- Dispatch --------------------------------

run_search() {
  local extra_run_dir_args=()
  if [[ -n "$SEARCH_RUN_DIR" ]]; then
    extra_run_dir_args=(--run-dir "$SEARCH_RUN_DIR")
  fi
  local cmd=(
    "$PYTHON_BIN" "$RUNNER_SEARCH"
    --data-dir "$DATA_DIR"
    --dev-split "$SPLIT"
    --dev-size "$DEV_SIZE"
    --dev-shuffle-seed "$DEV_SHUFFLE_SEED"
    --max-iterations "$MAX_ITERATIONS"
    --rng-seed "$RNG_SEED"
    --initial-workflow "$INITIAL_WORKFLOW"
    --max-new-tokens-executor "$MAX_NEW_TOKENS_EXEC"
    --max-new-tokens-optimizer "$MAX_NEW_TOKENS_OPT"
    --temperature-executor "$TEMPERATURE_EXEC"
    --temperature-optimizer "$TEMPERATURE_OPT"
    --top-p "$TOP_P"
    --api-base-a1 "http://${HOST}:${A1_PORT}/v1"
    --api-base-a2 "http://${HOST}:${A2_PORT}/v1"
    --api-base-a3 "http://${HOST}:${A3_PORT}/v1"
    --api-base-optimizer "http://${HOST}:${OPT_PORT}/v1"
    --api-model-a1 "$SERVED_A1"
    --api-model-a2 "$SERVED_A2"
    --api-model-a3 "$SERVED_A3"
    --api-model-optimizer "$SERVED_OPT"
    --api-timeout "$API_TIMEOUT"
    --max-concurrency "$MAX_CONCURRENCY"
    "${extra_run_dir_args[@]}"
  )
  if [[ "$ENABLE_THINKING" == 1 ]]; then
    cmd+=(--enable-thinking)
  fi
  echo
  echo "---------------- Search Command ----------------"
  printf '%q ' "${cmd[@]}"
  echo
  {
    printf 'PYTHONPATH=%q ' "$PYTHONPATH_ROOT"
    printf '%q ' "${cmd[@]}"
    echo
  } >>"$COMMAND_PATH"
  PYTHONPATH="$PYTHONPATH_ROOT" "${cmd[@]}"
}

run_eval() {
  local wf_file="$1"
  local cmd=(
    "$PYTHON_BIN" "$RUNNER_EVAL"
    --workflow-file "$wf_file"
    --split "$SPLIT"
    --data-dir "$DATA_DIR"
    --start "$START"
    --limit "$LIMIT"
    --max-new-tokens "$MAX_NEW_TOKENS_EXEC"
    --temperature "$TEMPERATURE_EXEC"
    --top-p "$TOP_P"
    --api-base-a1 "http://${HOST}:${A1_PORT}/v1"
    --api-base-a2 "http://${HOST}:${A2_PORT}/v1"
    --api-base-a3 "http://${HOST}:${A3_PORT}/v1"
    --api-model-a1 "$SERVED_A1"
    --api-model-a2 "$SERVED_A2"
    --api-model-a3 "$SERVED_A3"
    --api-timeout "$API_TIMEOUT"
    --judge-agent "$JUDGE_AGENT"
    --judge-agent-seed "$JUDGE_AGENT_SEED"
    --max-concurrency "$MAX_CONCURRENCY"
  )
  if [[ "$ENABLE_THINKING" == 1 ]]; then
    cmd+=(--enable-thinking)
  fi
  if [[ -n "$EVAL_OUTPUT_PATH" ]]; then
    cmd+=(--output "$EVAL_OUTPUT_PATH")
  fi
  echo
  echo "---------------- Eval Command (workflow=$wf_file) ----------------"
  printf '%q ' "${cmd[@]}"
  echo
  {
    printf 'PYTHONPATH=%q ' "$PYTHONPATH_ROOT"
    printf '%q ' "${cmd[@]}"
    echo
  } >>"$COMMAND_PATH"
  PYTHONPATH="$PYTHONPATH_ROOT" "${cmd[@]}"
}

find_best_workflow() {
  local search_root="$1"
  "$PYTHON_BIN" - "$search_root" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
state = json.loads((root / "state.json").read_text(encoding="utf-8"))
valid = [n for n in state if n.get("parse_ok")]
if not valid:
    sys.exit("No parse_ok nodes.")
best = max(valid, key=lambda n: (n["dev_em"], n["dev_f1"]))
print(best["source_file"])
PY
}

find_latest_search_run() {
  local latest=""
  local candidate
  for candidate in baseline/MuSiQue/AFlow/search_runs/*; do
    if [[ ! -f "$candidate/state.json" ]]; then
      continue
    fi
    if [[ -z "$latest" || "$candidate" -nt "$latest" ]]; then
      latest="$candidate"
    fi
  done
  printf '%s\n' "$latest"
}

case "$MODE" in
  search)
    run_search
    ;;
  eval)
    run_eval "$WORKFLOW_FILE"
    ;;
  both)
    run_search
    if [[ -n "$SEARCH_RUN_DIR" ]]; then
      LATEST_RUN_DIR="${SEARCH_RUN_DIR%/}"
    else
      LATEST_RUN_DIR="$(find_latest_search_run)"
    fi
    if [[ -z "$LATEST_RUN_DIR" ]]; then
      echo "ERROR: no search_runs directory found after search." >&2
      exit 1
    fi
    BEST_WF="$(find_best_workflow "$LATEST_RUN_DIR")"
    echo "Best workflow from search: $BEST_WF"
    run_eval "$BEST_WF"
    ;;
esac

echo
echo "---------------- Completed ----------------"
echo "time: $(date '+%Y-%m-%d %H:%M:%S %Z')"
{
  echo "RUN_ID=$RUN_ID"
  echo "RUN_DIR=$RUN_DIR"
  echo "MODE=$MODE"
  echo "LOG_PATH=$LOG_PATH"
  echo "A1_SERVER_LOG=$A1_SERVER_LOG"
  echo "A2_SERVER_LOG=$A2_SERVER_LOG"
  echo "A3_SERVER_LOG=$A3_SERVER_LOG"
  echo "OPT_SERVER_LOG=$OPT_SERVER_LOG"
  echo "CONFIG_PATH=$CONFIG_PATH"
  echo "COMMAND_PATH=$COMMAND_PATH"
  echo "COMPLETED_AT=$(date '+%Y-%m-%d %H:%M:%S %Z')"
} >"$SUMMARY_PATH"
echo "Summary written to $SUMMARY_PATH"

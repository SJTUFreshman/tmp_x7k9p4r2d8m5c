#!/usr/bin/env bash
# 8-GPU SFT LoRA launcher: start 3 vLLM OpenAI servers with adapters, then run
# JCA inference and report EM/F1.
#
# Default GPU layout:
#   A1 Qwen3-1.7B + A1 LoRA -> GPUs 0,1     TP=2
#   A2 Qwen3-4B   + A2 LoRA -> GPUs 2,3     TP=2
#   A3 Qwen3-8B   + A3 LoRA -> GPUs 4,5,6,7 TP=4
#
# Quick effect run:
#   bash scripts/run_sft_lora_vllm_8gpu.sh
#
# Full dev run with the old SFT prompt/schema:
#   LIMIT=2417 bash scripts/run_sft_lora_vllm_8gpu.sh
#
# Current JCA prompt/schema run:
#   EVAL_PROTOCOL=current LIMIT=2417 bash scripts/run_sft_lora_vllm_8gpu.sh
#
# Smoke test:
#   LIMIT=5 T_MAX=4 MAX_NEW_TOKENS=512 bash scripts/run_sft_lora_vllm_8gpu.sh
#
# The defaults prefer stability over raw throughput because this local vLLM
# build can crash with static LoRA + CUDA graphs/prefix caching.

set -euo pipefail

# ----------------------------- Configuration -----------------------------

PROJECT_ROOT="${PROJECT_ROOT:-/data/wangyuheng/jca}"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/qwen35/bin/python}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-/data/conda_envs/deep_research/bin/python}"
VLLM_LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH:-/data/conda_envs/deep_research/lib}"
PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-/data/wangyuheng}"

SPLIT="${SPLIT:-dev}"
START="${START:-0}"
LIMIT="${LIMIT:-2417}"
T_MAX="${T_MAX:-8}"
START_AGENT="${START_AGENT:-A1}"
SINGLE_AGENT="${SINGLE_AGENT:-0}"
EVAL_PROTOCOL="${EVAL_PROTOCOL:-old_sft}"

TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.95}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
LOG_RAW_CHARS="${LOG_RAW_CHARS:-2000}"
API_TIMEOUT="${API_TIMEOUT:-900}"

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
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
DISABLE_PREFIX_CACHING="${DISABLE_PREFIX_CACHING:-1}"
DISABLE_CHUNKED_PREFILL="${DISABLE_CHUNKED_PREFILL:-0}"

MAX_LORA_RANK="${MAX_LORA_RANK:-64}"
MAX_LORAS="${MAX_LORAS:-1}"
MAX_CPU_LORAS="${MAX_CPU_LORAS:-1}"
LORA_DTYPE="${LORA_DTYPE:-auto}"

MODEL_A1="${MODEL_A1:-/data/wangyuheng/models/Qwen3-1.7B}"
MODEL_A2="${MODEL_A2:-/data/wangyuheng/models/Qwen3-4B}"
MODEL_A3="${MODEL_A3:-/data/wangyuheng/models/Qwen3-8B}"

ADAPTER_A1="${ADAPTER_A1:-/data/wangyuheng/jca/rl_runs/rl_14B_0716_v2/A1/final}"
ADAPTER_A2="${ADAPTER_A2:-/data/wangyuheng/jca/rl_runs/rl_14B_0716_v2/A2/final}"
ADAPTER_A3="${ADAPTER_A3:-/data/wangyuheng/jca/rl_runs/rl_14B_0716_v3/A3/final}"

DATA_DIR="${DATA_DIR:-musique_data}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/sft_lora}"
LOG_DIR="${LOG_DIR:-logs/sft_lora}"

SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-900}"
USE_EXISTING_SERVERS="${USE_EXISTING_SERVERS:-0}"
KEEP_SERVERS="${KEEP_SERVERS:-0}"
DRY_RUN="${DRY_RUN:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"

# ------------------------------- Utilities -------------------------------

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_sft_lora_${EVAL_PROTOCOL}_${SPLIT}_start${START}_n${LIMIT}}"
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
PYTHONPATH_ROOT=$PYTHONPATH_ROOT
SPLIT=$SPLIT
START=$START
LIMIT=$LIMIT
T_MAX=$T_MAX
START_AGENT=$START_AGENT
SINGLE_AGENT=$SINGLE_AGENT
EVAL_PROTOCOL=$EVAL_PROTOCOL
TORCH_DTYPE=$TORCH_DTYPE
MAX_NEW_TOKENS=$MAX_NEW_TOKENS
TEMPERATURE=$TEMPERATURE
TOP_P=$TOP_P
ENABLE_THINKING=$ENABLE_THINKING
LOG_RAW_CHARS=$LOG_RAW_CHARS
API_TIMEOUT=$API_TIMEOUT
HOST=$HOST
A1_PORT=$A1_PORT
A2_PORT=$A2_PORT
A3_PORT=$A3_PORT
A1_GPUS=$A1_GPUS
A2_GPUS=$A2_GPUS
A3_GPUS=$A3_GPUS
A1_TP=$A1_TP
A2_TP=$A2_TP
A3_TP=$A3_TP
GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION
MAX_MODEL_LEN=$MAX_MODEL_LEN
ENFORCE_EAGER=$ENFORCE_EAGER
DISABLE_PREFIX_CACHING=$DISABLE_PREFIX_CACHING
DISABLE_CHUNKED_PREFILL=$DISABLE_CHUNKED_PREFILL
MAX_LORA_RANK=$MAX_LORA_RANK
MAX_LORAS=$MAX_LORAS
MAX_CPU_LORAS=$MAX_CPU_LORAS
LORA_DTYPE=$LORA_DTYPE
MODEL_A1=$MODEL_A1
MODEL_A2=$MODEL_A2
MODEL_A3=$MODEL_A3
ADAPTER_A1=$ADAPTER_A1
ADAPTER_A2=$ADAPTER_A2
ADAPTER_A3=$ADAPTER_A3
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

vllm_supports_option() {
  local option="$1"
  local gpus="$2"
  local help_text
  help_text="$(
    env \
      CUDA_VISIBLE_DEVICES="$gpus" \
      LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
      "$VLLM_PYTHON_BIN" -m vllm.entrypoints.openai.api_server --help 2>&1 || true
  )"
  grep -Fq -- "$option" <<<"$help_text"
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
  local agent="$1"
  local port="$2"
  local url="http://${HOST}:${port}/v1/models"

  if server_ready "$url"; then
    if [[ "$USE_EXISTING_SERVERS" == "1" ]]; then
      if server_has_model "$url" "$agent"; then
        echo "$agent existing server detected at $url and exposes LoRA model $agent; reusing it."
        return 0
      fi
      echo "ERROR: USE_EXISTING_SERVERS=1 but $url does not expose model $agent." >&2
      exit 1
    fi
    echo "ERROR: $agent port $port already has a vLLM-compatible server." >&2
    echo "Stop the old server, change ports, or set USE_EXISTING_SERVERS=1." >&2
    exit 1
  fi

  return 1
}

start_server() {
  local agent="$1"
  local model_path="$2"
  local adapter_path="$3"
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
    --served-model-name "${agent}_base"
    --tensor-parallel-size "$tp"
    --dtype "$TORCH_DTYPE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$MAX_MODEL_LEN"
    --trust-remote-code
    --enable-lora
    --max-lora-rank "$MAX_LORA_RANK"
    --max-loras "$MAX_LORAS"
    --max-cpu-loras "$MAX_CPU_LORAS"
    --lora-dtype "$LORA_DTYPE"
    --lora-modules "${agent}=${adapter_path}"
  )
  if [[ "$ENFORCE_EAGER" == "1" ]]; then
    cmd+=(--enforce-eager)
  fi
  if [[ "$DISABLE_PREFIX_CACHING" == "1" ]]; then
    if vllm_supports_option "--no-enable-prefix-caching" "$gpus"; then
      cmd+=(--no-enable-prefix-caching)
    else
      echo "  note: this vLLM does not support --no-enable-prefix-caching; leaving prefix caching at server default"
    fi
  fi
  if [[ "$DISABLE_CHUNKED_PREFILL" == "1" ]]; then
    if vllm_supports_option "--no-enable-chunked-prefill" "$gpus"; then
      cmd+=(--no-enable-chunked-prefill)
    else
      echo "  note: this vLLM does not support --no-enable-chunked-prefill; leaving chunked prefill at server default"
    fi
  fi
  if [[ -n "$EXTRA_VLLM_ARGS" ]]; then
    # shellcheck disable=SC2206
    local extra_vllm_args_array=($EXTRA_VLLM_ARGS)
    cmd+=("${extra_vllm_args_array[@]}")
  fi

  echo "Starting $agent LoRA server on GPUs $gpus, TP=$tp, port=$port"
  echo "  base_model: $model_path"
  echo "  adapter:    $adapter_path"
  echo "  served base name: ${agent}_base"
  echo "  served lora name: $agent"
  echo "  server log: $log_path"
  setsid env \
    CUDA_VISIBLE_DEVICES="$gpus" \
    LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "${cmd[@]}" >"$log_path" 2>&1 &
  local pid=$!
  SERVER_PIDS+=("$pid")
  echo "  process_group: $pid"
}

build_runner_command() {
  case "$EVAL_PROTOCOL" in
    old_sft|old|sft)
      if [[ "$SINGLE_AGENT" == "1" ]]; then
        echo "ERROR: SINGLE_AGENT=1 is not supported with EVAL_PROTOCOL=$EVAL_PROTOCOL." >&2
        echo "Use EVAL_PROTOCOL=current for single-agent zero-shot runs." >&2
        exit 1
      fi
      RUN_CMD=(
        "$PYTHON_BIN" scripts/run_sft_old_protocol.py
        --split "$SPLIT"
        --data-dir "$DATA_DIR"
        --start "$START"
        --limit "$LIMIT"
        --t-max "$T_MAX"
        --start-agent "$START_AGENT"
        --max-new-tokens "$MAX_NEW_TOKENS"
        --temperature "$TEMPERATURE"
        --top-p "$TOP_P"
        --api-base-a1 "http://${HOST}:${A1_PORT}/v1"
        --api-base-a2 "http://${HOST}:${A2_PORT}/v1"
        --api-base-a3 "http://${HOST}:${A3_PORT}/v1"
        --api-model-a1 A1
        --api-model-a2 A2
        --api-model-a3 A3
        --api-timeout "$API_TIMEOUT"
        --output "$OUTPUT_PATH"
        --log-raw-chars "$LOG_RAW_CHARS"
      )
      ;;
    current|zero_shot|new)
      RUN_CMD=(
        "$PYTHON_BIN" scripts/run_zero_shot.py
        --split "$SPLIT"
        --data-dir "$DATA_DIR"
        --start "$START"
        --limit "$LIMIT"
        --t-max "$T_MAX"
        --start-agent "$START_AGENT"
        --backend openai
        --max-new-tokens "$MAX_NEW_TOKENS"
        --temperature "$TEMPERATURE"
        --top-p "$TOP_P"
        --model-a1 "$MODEL_A1"
        --model-a2 "$MODEL_A2"
        --model-a3 "$MODEL_A3"
        --api-base-a1 "http://${HOST}:${A1_PORT}/v1"
        --api-base-a2 "http://${HOST}:${A2_PORT}/v1"
        --api-base-a3 "http://${HOST}:${A3_PORT}/v1"
        --api-model-a1 A1
        --api-model-a2 A2
        --api-model-a3 A3
        --output "$OUTPUT_PATH"
        --log-raw-chars "$LOG_RAW_CHARS"
      )
      if [[ "$SINGLE_AGENT" == "1" ]]; then
        RUN_CMD+=(--single-agent)
      fi
      ;;
    *)
      echo "ERROR: unsupported EVAL_PROTOCOL=$EVAL_PROTOCOL" >&2
      echo "Supported values: old_sft, current" >&2
      exit 1
      ;;
  esac
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

echo "================ SFT LoRA JCA 8-GPU vLLM Run ================"
echo "time: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "host: $(hostname)"
echo "project_root: $PROJECT_ROOT"
echo "runner_python: $PYTHON_BIN"
echo "vllm_python: $VLLM_PYTHON_BIN"
echo "run_id: $RUN_ID"
echo "run_dir: $RUN_DIR"
echo "output_path: $OUTPUT_PATH"
echo "log_path: $LOG_PATH"
echo "config_path: $CONFIG_PATH"
echo "command_path: $COMMAND_PATH"
echo "summary_path: $SUMMARY_PATH"
echo "server_logs:"
echo "  A1: $A1_SERVER_LOG"
echo "  A2: $A2_SERVER_LOG"
echo "  A3: $A3_SERVER_LOG"
echo

echo "---------------- Configuration ----------------"
write_config_snapshot
echo "Config snapshot written to $CONFIG_PATH"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "VLLM_PYTHON_BIN=$VLLM_PYTHON_BIN"
echo "VLLM_LD_LIBRARY_PATH=$VLLM_LD_LIBRARY_PATH"
echo "SPLIT=$SPLIT START=$START LIMIT=$LIMIT T_MAX=$T_MAX START_AGENT=$START_AGENT SINGLE_AGENT=$SINGLE_AGENT"
echo "EVAL_PROTOCOL=$EVAL_PROTOCOL"
echo "TORCH_DTYPE=$TORCH_DTYPE MAX_NEW_TOKENS=$MAX_NEW_TOKENS TEMPERATURE=$TEMPERATURE TOP_P=$TOP_P"
echo "ENABLE_THINKING=$ENABLE_THINKING LOG_RAW_CHARS=$LOG_RAW_CHARS API_TIMEOUT=$API_TIMEOUT"
echo "HOST=$HOST"
echo "A1_PORT=$A1_PORT A1_GPUS=$A1_GPUS A1_TP=$A1_TP"
echo "A2_PORT=$A2_PORT A2_GPUS=$A2_GPUS A2_TP=$A2_TP"
echo "A3_PORT=$A3_PORT A3_GPUS=$A3_GPUS A3_TP=$A3_TP"
echo "MODEL_A1=$MODEL_A1"
echo "MODEL_A2=$MODEL_A2"
echo "MODEL_A3=$MODEL_A3"
echo "ADAPTER_A1=$ADAPTER_A1"
echo "ADAPTER_A2=$ADAPTER_A2"
echo "ADAPTER_A3=$ADAPTER_A3"
echo "MAX_LORA_RANK=$MAX_LORA_RANK MAX_LORAS=$MAX_LORAS MAX_CPU_LORAS=$MAX_CPU_LORAS LORA_DTYPE=$LORA_DTYPE"
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
require_dir "A1 adapter" "$ADAPTER_A1"
require_dir "A2 adapter" "$ADAPTER_A2"
require_dir "A3 adapter" "$ADAPTER_A3"
require_file "A1 adapter weights" "$ADAPTER_A1/adapter_model.safetensors"
require_file "A2 adapter weights" "$ADAPTER_A2/adapter_model.safetensors"
require_file "A3 adapter weights" "$ADAPTER_A3/adapter_model.safetensors"
require_file "A1 adapter config" "$ADAPTER_A1/adapter_config.json"
require_file "A2 adapter config" "$ADAPTER_A2/adapter_config.json"
require_file "A3 adapter config" "$ADAPTER_A3/adapter_config.json"
case "$EVAL_PROTOCOL" in
  old_sft|old|sft)
    require_file "old SFT protocol runner" "scripts/run_sft_old_protocol.py"
    ;;
  current|zero_shot|new)
    require_file "current protocol runner" "scripts/run_zero_shot.py"
    ;;
  *)
    echo "ERROR: unsupported EVAL_PROTOCOL=$EVAL_PROTOCOL" >&2
    echo "Supported values: old_sft, current" >&2
    exit 1
    ;;
esac
"$PYTHON_BIN" --version
"$PYTHON_BIN" - <<'PY'
import importlib.util
for name in ["torch", "transformers", "peft"]:
    spec = importlib.util.find_spec(name)
    print(f"{name}: {'OK' if spec else 'MISSING'}")
if importlib.util.find_spec("torch"):
    import torch
    print(f"torch_version: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"cuda_device_count: {torch.cuda.device_count()}")
PY
echo "vLLM server Python:"
env LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" "$VLLM_PYTHON_BIN" --version
env LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" "$VLLM_PYTHON_BIN" - <<'PY'
import importlib.util
for name in ["torch", "transformers", "vllm"]:
    spec = importlib.util.find_spec(name)
    print(f"server_{name}: {'OK' if spec else 'MISSING'}")
if importlib.util.find_spec("torch"):
    import torch
    print(f"server_torch_version: {torch.__version__}")
if importlib.util.find_spec("transformers"):
    import transformers
    print(f"server_transformers_version: {transformers.__version__}")
if importlib.util.find_spec("vllm"):
    import vllm
    print(f"server_vllm_version: {getattr(vllm, '__version__', 'unknown')}")
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
echo "---------------- Starting vLLM LoRA Servers ----------------"
if [[ "$USE_EXISTING_SERVERS" == "1" ]]; then
  check_existing_server "A1" "$A1_PORT" || {
    echo "ERROR: USE_EXISTING_SERVERS=1 but A1 is not ready on port $A1_PORT." >&2
    exit 1
  }
  check_existing_server "A2" "$A2_PORT" || {
    echo "ERROR: USE_EXISTING_SERVERS=1 but A2 is not ready on port $A2_PORT." >&2
    exit 1
  }
  check_existing_server "A3" "$A3_PORT" || {
    echo "ERROR: USE_EXISTING_SERVERS=1 but A3 is not ready on port $A3_PORT." >&2
    exit 1
  }
else
  check_existing_server "A1" "$A1_PORT" || true
  check_existing_server "A2" "$A2_PORT" || true
  check_existing_server "A3" "$A3_PORT" || true
  start_server "A1" "$MODEL_A1" "$ADAPTER_A1" "$A1_PORT" "$A1_GPUS" "$A1_TP" "$A1_SERVER_LOG"
  start_server "A2" "$MODEL_A2" "$ADAPTER_A2" "$A2_PORT" "$A2_GPUS" "$A2_TP" "$A2_SERVER_LOG"
  start_server "A3" "$MODEL_A3" "$ADAPTER_A3" "$A3_PORT" "$A3_GPUS" "$A3_TP" "$A3_SERVER_LOG"
fi

echo
echo "---------------- Waiting for LoRA Models ----------------"
wait_for_model "A1" "http://${HOST}:${A1_PORT}/v1/models" "A1" "$SERVER_WAIT_TIMEOUT"
wait_for_model "A2" "http://${HOST}:${A2_PORT}/v1/models" "A2" "$SERVER_WAIT_TIMEOUT"
wait_for_model "A3" "http://${HOST}:${A3_PORT}/v1/models" "A3" "$SERVER_WAIT_TIMEOUT"

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
  echo "EVAL_PROTOCOL=$EVAL_PROTOCOL"
  echo "OUTPUT_PATH=$OUTPUT_PATH"
  echo "LOG_PATH=$LOG_PATH"
  echo "A1_SERVER_LOG=$A1_SERVER_LOG"
  echo "A2_SERVER_LOG=$A2_SERVER_LOG"
  echo "A3_SERVER_LOG=$A3_SERVER_LOG"
  echo "CONFIG_PATH=$CONFIG_PATH"
  echo "COMMAND_PATH=$COMMAND_PATH"
  echo "COMPLETED_AT=$(date '+%Y-%m-%d %H:%M:%S %Z')"
} >"$SUMMARY_PATH"
echo "Summary written to $SUMMARY_PATH"

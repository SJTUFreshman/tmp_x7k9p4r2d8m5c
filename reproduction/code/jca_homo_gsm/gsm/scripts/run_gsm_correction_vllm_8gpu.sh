#!/usr/bin/env bash
# GSM correction-SFT data collection launcher - 8 GPU vLLM.
#
# Supports three modes via MODE variable:
#   zero_shot_mas    : Zero-shot MAS (A1+A2+A3, no LoRA)            [default]
#   single_a3        : Single agent A3 only (8B, no LoRA)
#   sft_lora_mas     : SFT LoRA MAS (A1+A2+A3 with adapters)
#
# GPU layout:
#   A1 Qwen3-1.7B -> GPUs 0,1     TP=2
#   A2 Qwen3-4B   -> GPUs 2,3     TP=2
#   A3 Qwen3-8B   -> GPUs 4,5,6,7 TP=4
#
# Quick examples:
#   bash jca/gsm/scripts/run_gsm_vllm_8gpu.sh                    # zero-shot MAS, 100 problems
#   MODE=single_a3 LIMIT=200 bash jca/gsm/scripts/run_gsm_vllm_8gpu.sh
#   MODE=sft_lora_mas ADAPTER_A1=... ADAPTER_A2=... ADAPTER_A3=... bash ...

set -euo pipefail

# ============================================================
# Paths
# ============================================================
PROJECT_ROOT="${PROJECT_ROOT:-/data/wangyuheng/jca}"
MAS_SCRIPT="${MAS_SCRIPT:-gsm/scripts/run_mas_correction_sft.py}"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/qwen35/bin/python}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-/data/conda_envs/deep_research/bin/python}"
VLLM_LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH:-/data/conda_envs/deep_research/lib}"
PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-/data/wangyuheng}"

# ============================================================
# Mode
# ============================================================
MODE="${MODE:-zero_shot_mas}"   # zero_shot_mas | single_a3 | sft_lora_mas

# ============================================================
# Data
# ============================================================
DATA_PATH="${DATA_PATH:-/data/wangyuheng/jca/Math/data/GSM-HARD/splits/gsmhardv2_dev.jsonl}"
START="${START:-0}"
LIMIT="${LIMIT:-100}"
T_MAX="${T_MAX:-8}"
START_AGENT="${START_AGENT:-A1}"
MIN_AGENTS_BEFORE_STOP="${MIN_AGENTS_BEFORE_STOP:-2}"
SFT_WARMUP_PROMPT="${SFT_WARMUP_PROMPT:-0}"
SFT_CONTROLLED_GENERATION="${SFT_CONTROLLED_GENERATION:-0}"
SFT_MAX_STEP_RETRIES="${SFT_MAX_STEP_RETRIES:-2}"
SFT_MAX_TRAJECTORY_ATTEMPTS="${SFT_MAX_TRAJECTORY_ATTEMPTS:-0}"
SFT_MAX_VERIFIER_SIMILARITY="${SFT_MAX_VERIFIER_SIMILARITY:-0.85}"
SFT_EXTENDED_FRACTION="${SFT_EXTENDED_FRACTION:-0.4}"
SFT_PROTOCOL_SEED="${SFT_PROTOCOL_SEED:-42}"
SFT_STANDARD_MIN_HANDOFFS="${SFT_STANDARD_MIN_HANDOFFS:-2}"
SFT_EXTENDED_MIN_HANDOFFS="${SFT_EXTENDED_MIN_HANDOFFS:-3 4}"
CORRECTION_FRACTION="${CORRECTION_FRACTION:-0.30}"
SFT_PLAN_OFFSET="${SFT_PLAN_OFFSET:--1}"
CYCLE_DATA="${CYCLE_DATA:-1}"

# ============================================================
# Generation
# ============================================================
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.95}"
LOG_RAW_CHARS="${LOG_RAW_CHARS:-500}"
API_TIMEOUT="${API_TIMEOUT:-120}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"

# ============================================================
# Models
# ============================================================
MODEL_A1="${MODEL_A1:-/data/wangyuheng/models/Qwen3-1.7B}"
MODEL_A2="${MODEL_A2:-/data/wangyuheng/models/Qwen3-4B}"
MODEL_A3="${MODEL_A3:-/data/wangyuheng/models/Qwen3-8B}"

# LoRA adapters (only used in sft_lora_mas mode)
ADAPTER_A1="${ADAPTER_A1:-}"
ADAPTER_A2="${ADAPTER_A2:-}"
ADAPTER_A3="${ADAPTER_A3:-}"

# ============================================================
# vLLM server settings
# ============================================================
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
USE_EXISTING_SERVERS="${USE_EXISTING_SERVERS:-0}"
KEEP_SERVERS="${KEEP_SERVERS:-0}"

# ============================================================
# Output
# ============================================================
LOG_DIR="${LOG_DIR:-logs/gsm_eval}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/gsm_eval}"

# ============================================================
# Setup
# ============================================================
cd "$PROJECT_ROOT"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_gsm_${MODE}_start${START}_n${LIMIT}}"
RUN_DIR="${RUN_DIR:-${LOG_DIR}/${RUN_ID}}"
mkdir -p "$RUN_DIR"

OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/${RUN_ID}.jsonl}"
LOG_PATH="${RUN_DIR}/run.log"
A1_SERVER_LOG="${RUN_DIR}/A1_server.log"
A2_SERVER_LOG="${RUN_DIR}/A2_server.log"
A3_SERVER_LOG="${RUN_DIR}/A3_server.log"
COMMAND_PATH="${RUN_DIR}/command.txt"
PYTHON_COMPAT_DIR="${RUN_DIR}/python_compat"
mkdir -p "$PYTHON_COMPAT_DIR"
cat > "${PYTHON_COMPAT_DIR}/sitecustomize.py" <<'PY'
from transformers import PreTrainedTokenizerBase

if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    PreTrainedTokenizerBase.all_special_tokens_extended = property(
        lambda self: list(self.all_special_tokens)
    )
PY

SERVER_PIDS=()
exec > >(tee "$LOG_PATH") 2>&1

# ============================================================
# Utilities
# ============================================================

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  if [[ "$KEEP_SERVERS" == "1" ]]; then
    echo "KEEP_SERVERS=1 — leaving servers running"
    exit "$exit_code"
  fi
  if [[ "${#SERVER_PIDS[@]}" -gt 0 ]]; then
    echo "Stopping vLLM servers: ${SERVER_PIDS[*]}"
    for pid in "${SERVER_PIDS[@]}"; do
      kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 5
    for pid in "${SERVER_PIDS[@]}"; do
      kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    done
  fi
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

count_gpus() { awk -F',' '{print NF}' <<<"$1"; }

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

wait_for_server() {
  local name="$1" url="$2" timeout="$3"
  "$PYTHON_BIN" - "$name" "$url" "$timeout" <<'PY'
import json, sys, time, urllib.request
name, url, timeout = sys.argv[1], sys.argv[2], float(sys.argv[3])
deadline = time.time() + timeout
last_err = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            ids = [m["id"] for m in json.loads(r.read())["data"]]
        print(f"{name} ready: {ids}"); sys.exit(0)
    except Exception as e:
        last_err = e
    time.sleep(5)
print(f"{name} timed out. Last: {last_err}", file=sys.stderr); sys.exit(1)
PY
}

start_server_base() {
  local agent="$1" model="$2" port="$3" gpus="$4" tp="$5" log="$6"
  local gpu_count; gpu_count="$(count_gpus "$gpus")"
  [[ "$gpu_count" == "$tp" ]] || { echo "ERROR: $agent GPU count $gpu_count != TP $tp"; exit 1; }
  local cmd=(
    "$VLLM_PYTHON_BIN" -m vllm.entrypoints.openai.api_server
    --host "$HOST" --port "$port"
    --model "$model"
    --served-model-name "$agent"
    --tensor-parallel-size "$tp"
    --dtype "$TORCH_DTYPE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$MAX_MODEL_LEN"
    --trust-remote-code
  )
  [[ "$ENFORCE_EAGER"          == "1" ]] && cmd+=(--enforce-eager)
  if [[ -n "$MAX_NUM_SEQS" ]] && vllm_supports_option "--max-num-seqs" "$gpus"; then
    cmd+=(--max-num-seqs "$MAX_NUM_SEQS")
  fi
  if [[ -n "$MAX_NUM_BATCHED_TOKENS" ]] && vllm_supports_option "--max-num-batched-tokens" "$gpus"; then
    cmd+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
  fi
  if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
    if vllm_supports_option "--enable-prefix-caching" "$gpus"; then
      cmd+=(--enable-prefix-caching)
    fi
  elif vllm_supports_option "--no-enable-prefix-caching" "$gpus"; then
    cmd+=(--no-enable-prefix-caching)
  fi
  if [[ "$ENABLE_CHUNKED_PREFILL" == "1" ]]; then
    if vllm_supports_option "--enable-chunked-prefill" "$gpus"; then
      cmd+=(--enable-chunked-prefill)
    fi
  elif vllm_supports_option "--no-enable-chunked-prefill" "$gpus"; then
    cmd+=(--no-enable-chunked-prefill)
  fi
  if [[ -n "$EXTRA_VLLM_ARGS" ]]; then
    # shellcheck disable=SC2206
    local extra_vllm_args_array=($EXTRA_VLLM_ARGS)
    cmd+=("${extra_vllm_args_array[@]}")
  fi
  echo "Starting $agent (base) on GPUs=$gpus port=$port"
  setsid env \
    CUDA_VISIBLE_DEVICES="$gpus" \
    LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    PYTHONPATH="${PYTHON_COMPAT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${cmd[@]}" >"$log" 2>&1 &
  SERVER_PIDS+=("$!")
}

start_server_lora() {
  local agent="$1" model="$2" adapter="$3" port="$4" gpus="$5" tp="$6" log="$7"
  local gpu_count; gpu_count="$(count_gpus "$gpus")"
  [[ "$gpu_count" == "$tp" ]] || { echo "ERROR: $agent GPU count $gpu_count != TP $tp"; exit 1; }
  local cmd=(
    "$VLLM_PYTHON_BIN" -m vllm.entrypoints.openai.api_server
    --host "$HOST" --port "$port"
    --model "$model"
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
    --lora-modules "${agent}=${adapter}"
  )
  [[ "$ENFORCE_EAGER"          == "1" ]] && cmd+=(--enforce-eager)
  if [[ -n "$MAX_NUM_SEQS" ]] && vllm_supports_option "--max-num-seqs" "$gpus"; then
    cmd+=(--max-num-seqs "$MAX_NUM_SEQS")
  fi
  if [[ -n "$MAX_NUM_BATCHED_TOKENS" ]] && vllm_supports_option "--max-num-batched-tokens" "$gpus"; then
    cmd+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
  fi
  if [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
    if vllm_supports_option "--enable-prefix-caching" "$gpus"; then
      cmd+=(--enable-prefix-caching)
    fi
  elif vllm_supports_option "--no-enable-prefix-caching" "$gpus"; then
    cmd+=(--no-enable-prefix-caching)
  fi
  if [[ "$ENABLE_CHUNKED_PREFILL" == "1" ]]; then
    if vllm_supports_option "--enable-chunked-prefill" "$gpus"; then
      cmd+=(--enable-chunked-prefill)
    fi
  elif vllm_supports_option "--no-enable-chunked-prefill" "$gpus"; then
    cmd+=(--no-enable-chunked-prefill)
  fi
  if [[ -n "$EXTRA_VLLM_ARGS" ]]; then
    # shellcheck disable=SC2206
    local extra_vllm_args_array=($EXTRA_VLLM_ARGS)
    cmd+=("${extra_vllm_args_array[@]}")
  fi
  echo "Starting $agent (LoRA) on GPUs=$gpus port=$port  adapter=$adapter"
  setsid env \
    CUDA_VISIBLE_DEVICES="$gpus" \
    LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    PYTHONPATH="${PYTHON_COMPAT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${cmd[@]}" >"$log" 2>&1 &
  SERVER_PIDS+=("$!")
}

# ============================================================
# Print config
# ============================================================
echo "================ GSM-HARD Eval ================"
echo "time:    $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "host:    $(hostname)"
echo "run_id:  $RUN_ID"
echo "mode:    $MODE"
echo "data:    $DATA_PATH"
echo "start=$START  limit=$LIMIT  t_max=$T_MAX"
echo "max_concurrency=$MAX_CONCURRENCY"
if [[ "$MODE" != "single_a3" ]]; then
  echo "start_agent=$START_AGENT"
  echo "min_agents_before_stop=$MIN_AGENTS_BEFORE_STOP"
  echo "sft_warmup_prompt=$SFT_WARMUP_PROMPT"
  if [[ "$SFT_WARMUP_PROMPT" == "1" ]]; then
    echo "sft_extended_fraction=$SFT_EXTENDED_FRACTION"
    echo "sft_controlled_generation=$SFT_CONTROLLED_GENERATION"
    echo "sft_max_step_retries=$SFT_MAX_STEP_RETRIES"
    echo "sft_standard_min_handoffs=$SFT_STANDARD_MIN_HANDOFFS"
    echo "sft_extended_min_handoffs=$SFT_EXTENDED_MIN_HANDOFFS"
  fi
fi
echo "output:  $OUTPUT_PATH"
echo "================================================"

# ============================================================
# Start servers based on MODE
# ============================================================
if [[ "$USE_EXISTING_SERVERS" != "1" ]]; then
  case "$MODE" in
    zero_shot_mas)
      start_server_base "A1" "$MODEL_A1" "$A1_PORT" "$A1_GPUS" "$A1_TP" "$A1_SERVER_LOG"
      start_server_base "A2" "$MODEL_A2" "$A2_PORT" "$A2_GPUS" "$A2_TP" "$A2_SERVER_LOG"
      start_server_base "A3" "$MODEL_A3" "$A3_PORT" "$A3_GPUS" "$A3_TP" "$A3_SERVER_LOG"
      ;;
    single_a3)
      # Only start A3 on all 8 GPUs for max throughput
      start_server_base "A3" "$MODEL_A3" "$A3_PORT" "0,1,2,3,4,5,6,7" "8" "$A3_SERVER_LOG"
      ;;
    sft_lora_mas)
      [[ -d "$ADAPTER_A1" ]] || { echo "ERROR: ADAPTER_A1 not found: $ADAPTER_A1"; exit 1; }
      [[ -d "$ADAPTER_A2" ]] || { echo "ERROR: ADAPTER_A2 not found: $ADAPTER_A2"; exit 1; }
      [[ -d "$ADAPTER_A3" ]] || { echo "ERROR: ADAPTER_A3 not found: $ADAPTER_A3"; exit 1; }
      start_server_lora "A1" "$MODEL_A1" "$ADAPTER_A1" "$A1_PORT" "$A1_GPUS" "$A1_TP" "$A1_SERVER_LOG"
      start_server_lora "A2" "$MODEL_A2" "$ADAPTER_A2" "$A2_PORT" "$A2_GPUS" "$A2_TP" "$A2_SERVER_LOG"
      start_server_lora "A3" "$MODEL_A3" "$ADAPTER_A3" "$A3_PORT" "$A3_GPUS" "$A3_TP" "$A3_SERVER_LOG"
      ;;
    *)
      echo "ERROR: Unknown MODE=$MODE. Use: zero_shot_mas | single_a3 | sft_lora_mas"; exit 1
      ;;
  esac
fi

# ============================================================
# Wait for servers
# ============================================================
echo ""
echo "-------- Waiting for servers --------"
case "$MODE" in
  single_a3)
    wait_for_server "A3" "http://${HOST}:${A3_PORT}/v1/models" "$SERVER_WAIT_TIMEOUT"
    ;;
  *)
    wait_for_server "A1" "http://${HOST}:${A1_PORT}/v1/models" "$SERVER_WAIT_TIMEOUT"
    wait_for_server "A2" "http://${HOST}:${A2_PORT}/v1/models" "$SERVER_WAIT_TIMEOUT"
    wait_for_server "A3" "http://${HOST}:${A3_PORT}/v1/models" "$SERVER_WAIT_TIMEOUT"
    ;;
esac

# ============================================================
# Build run command
# ============================================================
case "$MODE" in
  single_a3)
    # Single agent: only A3, no handoff
  RUN_CMD=(
      "$PYTHON_BIN" -u gsm/scripts/run_single_agent.py
      --data-path "$DATA_PATH"
      --start "$START" --limit "$LIMIT"
      --t-max "$T_MAX"
      --api-base "http://${HOST}:${A3_PORT}/v1"
      --api-model A3
      --api-key EMPTY
      --api-timeout "$API_TIMEOUT"
      --max-new-tokens "$MAX_NEW_TOKENS"
      --temperature "$TEMPERATURE"
      --top-p "$TOP_P"
      --output "$OUTPUT_PATH"
      --log-raw-chars "$LOG_RAW_CHARS"
    )
    ;;
  zero_shot_mas|sft_lora_mas)
    A1_MODEL_NAME="A1"
    A2_MODEL_NAME="A2"
    A3_MODEL_NAME="A3"
    [[ "$MODE" == "sft_lora_mas" ]] && {
      A1_MODEL_NAME="A1"; A2_MODEL_NAME="A2"; A3_MODEL_NAME="A3"
    }
    RUN_CMD=(
      "$PYTHON_BIN" -u "$MAS_SCRIPT"
      --data-path "$DATA_PATH"
      --start "$START" --limit "$LIMIT"
      --t-max "$T_MAX"
      --start-agent "$START_AGENT"
      --min-agents-before-stop "$MIN_AGENTS_BEFORE_STOP"
      --api-base-a1 "http://${HOST}:${A1_PORT}/v1"
      --api-base-a2 "http://${HOST}:${A2_PORT}/v1"
      --api-base-a3 "http://${HOST}:${A3_PORT}/v1"
      --api-model-a1 "$A1_MODEL_NAME"
      --api-model-a2 "$A2_MODEL_NAME"
      --api-model-a3 "$A3_MODEL_NAME"
      --api-key EMPTY
      --api-timeout "$API_TIMEOUT"
      --max-new-tokens "$MAX_NEW_TOKENS"
      --temperature "$TEMPERATURE"
      --top-p "$TOP_P"
      --max-concurrency "$MAX_CONCURRENCY"
      --output "$OUTPUT_PATH"
      --log-raw-chars "$LOG_RAW_CHARS"
      --correction-fraction "$CORRECTION_FRACTION"
      --sft-plan-offset "$SFT_PLAN_OFFSET"
    )
    if [[ "$CYCLE_DATA" == "1" ]]; then
      RUN_CMD+=(--cycle-data)
    else
      RUN_CMD+=(--no-cycle-data)
    fi
    if [[ "$SFT_WARMUP_PROMPT" == "1" ]]; then
      read -r -a SFT_EXTENDED_MIN_HANDOFFS_ARRAY <<< "$SFT_EXTENDED_MIN_HANDOFFS"
      RUN_CMD+=(
        --sft-warmup-prompt
        --sft-extended-fraction "$SFT_EXTENDED_FRACTION"
        --sft-protocol-seed "$SFT_PROTOCOL_SEED"
        --sft-standard-min-handoffs "$SFT_STANDARD_MIN_HANDOFFS"
        --sft-extended-min-handoffs "${SFT_EXTENDED_MIN_HANDOFFS_ARRAY[@]}"
      )
      if [[ "$SFT_CONTROLLED_GENERATION" == "1" ]]; then
        RUN_CMD+=(
          --sft-controlled-generation
          --sft-max-step-retries "$SFT_MAX_STEP_RETRIES"
          --sft-max-trajectory-attempts "$SFT_MAX_TRAJECTORY_ATTEMPTS"
          --sft-max-verifier-similarity "$SFT_MAX_VERIFIER_SIMILARITY"
        )
      fi
    fi
    ;;
esac

# Save command
printf 'PYTHONPATH=%q ' "$PYTHONPATH_ROOT" > "$COMMAND_PATH"
printf '%q ' "${RUN_CMD[@]}" >> "$COMMAND_PATH"
echo >> "$COMMAND_PATH"

echo ""
echo "-------- Running --------"
PYTHONPATH="$PYTHONPATH_ROOT" "${RUN_CMD[@]}"

echo ""
echo "================================================"
echo "Done: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "output: $OUTPUT_PATH"
echo "log:    $LOG_PATH"
echo "================================================"

#!/usr/bin/env bash
# Re-score the immutable GSM fixed-A1 trajectories with a local Qwen3-14B
# judge.  The output is a GSM collaboration-v3 rejudge file and can be fed
# directly to run_pipeline.sh via SOURCE_INPUT.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/wangyuheng/jca}"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/qwen35/bin/python}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-$PYTHON_BIN}"
VLLM_LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH:-$(dirname "$(dirname "$VLLM_PYTHON_BIN")")/lib}"
PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-/data/wangyuheng}"
PYTHON_COMPAT_DIR="${PYTHON_COMPAT_DIR:-${PROJECT_ROOT}/scripts/gsm_judge_rl/python_compat}"
REJUDGE_SCRIPT="${REJUDGE_SCRIPT:-${PROJECT_ROOT}/gsm/scripts/rejudge_gsm_collaboration_v3.py}"

MODEL_PATH="${MODEL_PATH:-/data/wangyuheng/models/Qwen3-14B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen14b_gsm_judge}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8400}"
JUDGE_GPUS="${JUDGE_GPUS:-0,1,2,3,4,5,6,7}"
JUDGE_TP="${JUDGE_TP:-8}"
JUDGE_DP="${JUDGE_DP:-1}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
ENABLE_THINKING="${ENABLE_THINKING:-0}"
JCA_JUDGE_CONFIDENT_EXTREMES="${JCA_JUDGE_CONFIDENT_EXTREMES:-0}"
REASONING_PARSER="${REASONING_PARSER:-deepseek_r1}"
if [[ "$ENABLE_THINKING" == "1" ]]; then
  DEFAULT_CHAT_TEMPLATE_KWARGS='{"enable_thinking":true}'
else
  DEFAULT_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'
fi
CHAT_TEMPLATE_KWARGS="${CHAT_TEMPLATE_KWARGS:-$DEFAULT_CHAT_TEMPLATE_KWARGS}"
EXTRA_VLLM_ARGS="${EXTRA_VLLM_ARGS:-}"
# vLLM 0.8.2's V1 worker imports FlashInfer unconditionally.  The installed
# FlashInfer extension is built against an incompatible Torch ABI, while the
# legacy V0 engine supports this model and avoids that import path.
VLLM_USE_V1="${VLLM_USE_V1:-0}"
VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
# V0 otherwise forks GPU workers after the API process has initialized CUDA.
# Spawn gives each worker a fresh CUDA process context.
VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
PATCH_VLLM_TRANSFORMERS_LM_HEAD="${PATCH_VLLM_TRANSFORMERS_LM_HEAD:-1}"

INPUT_PATH="${INPUT_PATH:-${PROJECT_ROOT}/rl_data/gsm/gsm_fixed_a1_corr30_20260723_v3_judge_rl_full1187_v4_rejudged.jsonl}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/rl_data/gsm/judge_rl/qwen14b/source_rejudged.jsonl}"
STATS_OUTPUT="${STATS_OUTPUT:-${PROJECT_ROOT}/rl_data/gsm/judge_rl/qwen14b/source_rejudge_stats.json}"
DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/Math/data/GSM-HARD/splits/gsmhardv2_train.jsonl}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-16}"
GROUP_RETRIES="${GROUP_RETRIES:-2}"
# The server enforces prompt_tokens + max_tokens <= MAX_MODEL_LEN.  Judge
# outputs are short JSON arrays, so 4096 leaves safe prompt headroom.
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-4096}"
JUDGE_TEMPERATURE="${JUDGE_TEMPERATURE:-0.0}"
JUDGE_TOP_P="${JUDGE_TOP_P:-0.95}"
JUDGE_REASONING_EFFORT="${JUDGE_REASONING_EFFORT:-low}"
JUDGE_PARSE_RETRIES="${JUDGE_PARSE_RETRIES:-2}"
LIMIT_GROUPS="${LIMIT_GROUPS:-0}"
PROJECT_CONFLICTING_POSITIVE_SCORES="${PROJECT_CONFLICTING_POSITIVE_SCORES:-0}"
DROP_ALL_FAILED_PROBLEMS="${DROP_ALL_FAILED_PROBLEMS:-0}"
EXPECTED_ROLLOUTS_PER_PROBLEM="${EXPECTED_ROLLOUTS_PER_PROBLEM:-8}"
ALLOW_PARTIAL_ROLLOUTS="${ALLOW_PARTIAL_ROLLOUTS:-0}"
RESUME="${RESUME:-1}"
AUTO_RESUME_PASSES="${AUTO_RESUME_PASSES:-1}"
DRY_RUN="${DRY_RUN:-0}"
WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"
GPU_IDLE_CHECKS="${GPU_IDLE_CHECKS:-2}"
USE_EXISTING_SERVER="${USE_EXISTING_SERVER:-0}"
KEEP_SERVER="${KEEP_SERVER:-0}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-900}"
SERVER_STOP_TIMEOUT="${SERVER_STOP_TIMEOUT:-30}"
# DP workers rendezvous over short-lived TCP ports. Retry the complete server
# process group when a transient port collision aborts startup.
START_RETRIES="${START_RETRIES:-4}"
RUNTIME_ROOT_OVERRIDE="${RUNTIME_ROOT_OVERRIDE:-}"

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/gsm_judge_rl/qwen14b_rejudge}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_qwen14b_gsm_rejudge}"
RUN_DIR="${RUN_DIR:-${LOG_DIR}/${RUN_ID}}"
RUN_LOG="${RUN_DIR}/run.log"
SERVER_LOG_BASE="${RUN_DIR}/server.log"
SERVER_LOG="$SERVER_LOG_BASE"

fatal() {
  echo "[fatal] $*" >&2
  exit 1
}

count_gpus() { awk -F',' '{print NF}' <<<"$1"; }

wait_for_idle_gpus() {
  [[ "$WAIT_FOR_GPUS" == "1" ]] || return 0
  local idle_checks=0 processes
  while (( idle_checks < GPU_IDLE_CHECKS )); do
    processes="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits || true)"
    if [[ -z "${processes//[[:space:]]/}" ]]; then
      idle_checks=$((idle_checks + 1))
      echo "[gpu] idle check ${idle_checks}/${GPU_IDLE_CHECKS}"
    else
      idle_checks=0
      echo "[gpu] GPUs occupied; waiting"
    fi
    if (( idle_checks < GPU_IDLE_CHECKS )); then
      sleep "$GPU_POLL_SECONDS"
    fi
  done
}

server_ready() {
  "$PYTHON_BIN" - "http://${HOST}:${PORT}/v1/models" <<'PY' >/dev/null 2>&1
import json, sys, urllib.request
with urllib.request.urlopen(sys.argv[1], timeout=3) as response:
    payload = json.loads(response.read().decode("utf-8"))
if "data" not in payload:
    raise SystemExit(1)
PY
}

server_has_model() {
  "$PYTHON_BIN" - "http://${HOST}:${PORT}/v1/models" "$SERVED_MODEL_NAME" <<'PY' >/dev/null 2>&1
import json, sys, urllib.request
with urllib.request.urlopen(sys.argv[1], timeout=3) as response:
    payload = json.loads(response.read().decode("utf-8"))
if sys.argv[2] not in [item.get("id") for item in payload.get("data", [])]:
    raise SystemExit(1)
PY
}

wait_for_model() {
  local timeout="$1" start now
  start="$(date +%s)"
  while true; do
    if [[ -n "${SERVER_PID:-}" ]] && ! kill -0 "$SERVER_PID" >/dev/null 2>&1; then
      echo "[server] exited before becoming ready" >&2
      tail -n 100 "$SERVER_LOG" >&2 || true
      return 1
    fi
    if server_has_model; then
      echo "[server] ready: ${SERVED_MODEL_NAME}"
      return 0
    fi
    now="$(date +%s)"
    if (( now - start >= timeout )); then
      echo "[server] timed out after ${timeout}s" >&2
      tail -n 100 "$SERVER_LOG" >&2 || true
      return 1
    fi
    sleep 5
  done
}

VLLM_HELP_LOADED=0
VLLM_HELP_TEXT=""
vllm_supports_option() {
  local option="$1"
  if [[ "$VLLM_HELP_LOADED" == "0" ]]; then
    if ! VLLM_HELP_TEXT="$(env \
      CUDA_VISIBLE_DEVICES="$JUDGE_GPUS" \
      LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
      VLLM_USE_V1="$VLLM_USE_V1" \
      PYTHONPATH="${PYTHON_COMPAT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
      "$VLLM_PYTHON_BIN" -m vllm.entrypoints.openai.api_server --help 2>&1)"; then
      echo "[server] unable to inspect vLLM options" >&2
      return 1
    fi
    VLLM_HELP_LOADED=1
  fi
  [[ "$VLLM_HELP_TEXT" == *"$option"* ]]
}

mkdir -p "$RUN_DIR" "$(dirname "$OUTPUT_PATH")" "$(dirname "$STATS_OUTPUT")"
exec > >(tee "$RUN_LOG") 2>&1

echo "GSM Qwen3-14B judge re-score"
echo "  input:       $INPUT_PATH"
echo "  output:      $OUTPUT_PATH"
echo "  stats:       $STATS_OUTPUT"
echo "  model:       $MODEL_PATH"
echo "  endpoint:    http://${HOST}:${PORT}/v1"
echo "  GPUs:        $JUDGE_GPUS (TP=$JUDGE_TP, DP=$JUDGE_DP)"
echo "  concurrency: $JUDGE_CONCURRENCY"
echo "  thinking:    $ENABLE_THINKING"
echo "  confident extremes: $JCA_JUDGE_CONFIDENT_EXTREMES"
echo "  project conflicting positive scores: $PROJECT_CONFLICTING_POSITIVE_SCORES"
echo "  drop all-failed problems: $DROP_ALL_FAILED_PROBLEMS"

[[ -x "$PYTHON_BIN" ]] || fatal "python not found: $PYTHON_BIN"
[[ -d "$MODEL_PATH" ]] || fatal "14B model not found: $MODEL_PATH"
[[ -f "$PYTHON_COMPAT_DIR/sitecustomize.py" ]] || \
  fatal "transformers compatibility shim not found: $PYTHON_COMPAT_DIR/sitecustomize.py"
[[ -f "$REJUDGE_SCRIPT" ]] || fatal "rejudge script not found: $REJUDGE_SCRIPT"
[[ -f "$INPUT_PATH" ]] || fatal "input not found: $INPUT_PATH"
[[ -e "$DATA_PATH" ]] || fatal "evaluation data not found: $DATA_PATH"
[[ "$RESUME" == "0" || "$RESUME" == "1" ]] || fatal "RESUME must be 0 or 1"
[[ "$AUTO_RESUME_PASSES" =~ ^[1-9][0-9]*$ ]] || \
  fatal "AUTO_RESUME_PASSES must be positive"
if (( AUTO_RESUME_PASSES > 1 )) && [[ "$RESUME" != "1" ]]; then
  fatal "AUTO_RESUME_PASSES > 1 requires RESUME=1"
fi
[[ "$DRY_RUN" == "0" || "$DRY_RUN" == "1" ]] || fatal "DRY_RUN must be 0 or 1"
[[ "$PREFLIGHT_ONLY" == "0" || "$PREFLIGHT_ONLY" == "1" ]] || \
  fatal "PREFLIGHT_ONLY must be 0 or 1"
[[ "$ENABLE_THINKING" == "0" || "$ENABLE_THINKING" == "1" ]] || \
  fatal "ENABLE_THINKING must be 0 or 1"
[[ "$JCA_JUDGE_CONFIDENT_EXTREMES" == "0" || "$JCA_JUDGE_CONFIDENT_EXTREMES" == "1" ]] || \
  fatal "JCA_JUDGE_CONFIDENT_EXTREMES must be 0 or 1"
[[ "$DROP_ALL_FAILED_PROBLEMS" == "0" || "$DROP_ALL_FAILED_PROBLEMS" == "1" ]] || \
  fatal "DROP_ALL_FAILED_PROBLEMS must be 0 or 1"
[[ "$PROJECT_CONFLICTING_POSITIVE_SCORES" == "0" || "$PROJECT_CONFLICTING_POSITIVE_SCORES" == "1" ]] || \
  fatal "PROJECT_CONFLICTING_POSITIVE_SCORES must be 0 or 1"
[[ "$EXPECTED_ROLLOUTS_PER_PROBLEM" =~ ^[1-9][0-9]*$ ]] || \
  fatal "EXPECTED_ROLLOUTS_PER_PROBLEM must be positive"
[[ "$ALLOW_PARTIAL_ROLLOUTS" == "0" || "$ALLOW_PARTIAL_ROLLOUTS" == "1" ]] || \
  fatal "ALLOW_PARTIAL_ROLLOUTS must be 0 or 1"
[[ "$JUDGE_TP" =~ ^[1-9][0-9]*$ ]] || fatal "JUDGE_TP must be positive"
[[ "$JUDGE_DP" =~ ^[1-9][0-9]*$ ]] || fatal "JUDGE_DP must be positive"
[[ "$START_RETRIES" =~ ^[1-9][0-9]*$ ]] || fatal "START_RETRIES must be positive"
[[ "$SERVER_WAIT_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || fatal "SERVER_WAIT_TIMEOUT must be positive"
[[ "$SERVER_STOP_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || fatal "SERVER_STOP_TIMEOUT must be positive"
[[ "$(count_gpus "$JUDGE_GPUS")" -eq $((JUDGE_TP * JUDGE_DP)) ]] || \
  fatal "JUDGE_GPUS count does not match JUDGE_TP*JUDGE_DP"
[[ "$LIMIT_GROUPS" =~ ^[0-9]+$ ]] || fatal "LIMIT_GROUPS must be non-negative"
[[ "$JUDGE_MAX_TOKENS" =~ ^[1-9][0-9]*$ ]] || fatal "JUDGE_MAX_TOKENS must be positive"
[[ "$MAX_MODEL_LEN" =~ ^[1-9][0-9]*$ ]] || fatal "MAX_MODEL_LEN must be positive"
(( JUDGE_MAX_TOKENS < MAX_MODEL_LEN )) || \
  fatal "JUDGE_MAX_TOKENS must be smaller than MAX_MODEL_LEN to leave prompt headroom"

SERVER_PID=""
RUNTIME_ROOT=""
RUNTIME_ROOT_OWNED=0

stop_server() {
  local pid="${SERVER_PID:-}"
  [[ -n "$pid" ]] || return
  echo "[server] stopping judge process group $pid"
  kill -TERM -- "-$pid" >/dev/null 2>&1 || kill -TERM "$pid" >/dev/null 2>&1 || true
  local deadline=$((SECONDS + SERVER_STOP_TIMEOUT))
  while kill -0 -- "-$pid" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      echo "[server] process group $pid did not stop after ${SERVER_STOP_TIMEOUT}s; sending KILL"
      kill -KILL -- "-$pid" >/dev/null 2>&1 || kill -KILL "$pid" >/dev/null 2>&1 || true
      break
    fi
    sleep 1
  done
  wait "$pid" >/dev/null 2>&1 || true
  SERVER_PID=""
}

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  if [[ "$KEEP_SERVER" != "1" ]]; then
    stop_server
    if [[ "$RUNTIME_ROOT_OWNED" == "1" && "$RUNTIME_ROOT" == /data/tmp/jca_gsm_judge.* ]]; then
      rm -rf -- "$RUNTIME_ROOT"
    fi
  fi
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] would start Qwen3-14B and run:"
  printf '  PYTHONPATH=%q JCA_JUDGE_API_BASE=%q JCA_JUDGE_DISABLE_REASONING_EFFORT=1 ' \
    "$PYTHONPATH_ROOT" "http://${HOST}:${PORT}/v1"
  if [[ "$ENABLE_THINKING" == "0" ]]; then
    printf 'JCA_JUDGE_DISABLE_THINKING=1 JCA_JUDGE_ENABLE_THINKING=0 '
  else
    printf 'JCA_JUDGE_DISABLE_THINKING=0 JCA_JUDGE_ENABLE_THINKING=1 '
  fi
  printf 'JCA_JUDGE_CONFIDENT_EXTREMES=%q ' "$JCA_JUDGE_CONFIDENT_EXTREMES"
  printf '%q ' "$PYTHON_BIN" -u "$REJUDGE_SCRIPT" \
    --input "$INPUT_PATH" --data-path "$DATA_PATH" --output "$OUTPUT_PATH" \
    --stats-output "$STATS_OUTPUT" --max-concurrency "$JUDGE_CONCURRENCY" \
    --group-retries "$GROUP_RETRIES" --judge-model "$SERVED_MODEL_NAME" \
    --judge-temperature "$JUDGE_TEMPERATURE" --judge-top-p "$JUDGE_TOP_P" \
    --judge-max-tokens "$JUDGE_MAX_TOKENS" \
    --judge-reasoning-effort "$JUDGE_REASONING_EFFORT" \
    --judge-parse-retries "$JUDGE_PARSE_RETRIES" \
    --expected-rollouts-per-problem "$EXPECTED_ROLLOUTS_PER_PROBLEM"
  if (( LIMIT_GROUPS > 0 )); then
    printf '%q ' --limit-groups "$LIMIT_GROUPS"
  fi
  if [[ "$DROP_ALL_FAILED_PROBLEMS" == "1" ]]; then
    printf '%q ' --drop-all-failed-problems
  else
    printf '%q ' --no-drop-all-failed-problems
  fi
  if [[ "$PROJECT_CONFLICTING_POSITIVE_SCORES" == "1" ]]; then
    printf '%q ' --project-conflicting-positive-scores
  fi
  if [[ "$RESUME" == "1" ]]; then
    printf '%q ' --resume
  else
    printf '%q ' --no-resume
  fi
  printf '\n'
  exit 0
fi

wait_for_idle_gpus
if server_ready; then
  [[ "$USE_EXISTING_SERVER" == "1" ]] || \
    fatal "port ${PORT} already has an OpenAI-compatible server"
  server_has_model || fatal "existing server does not expose ${SERVED_MODEL_NAME}"
  echo "[server] reusing existing server"
else
  VLLM_CMD=(
    "$VLLM_PYTHON_BIN" -m vllm.entrypoints.openai.api_server
    --host "$HOST" --port "$PORT" --model "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --tensor-parallel-size "$JUDGE_TP" --dtype "$TORCH_DTYPE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --max-model-len "$MAX_MODEL_LEN" --trust-remote-code
  )
  if vllm_supports_option "--data-parallel-size"; then
    VLLM_CMD+=(--data-parallel-size "$JUDGE_DP")
  elif (( JUDGE_DP > 1 )); then
    fatal "selected vLLM API server does not support --data-parallel-size"
  fi
  [[ "$ENFORCE_EAGER" == "1" ]] && VLLM_CMD+=(--enforce-eager)
  # Some vLLM versions expose a server-wide default.  Older versions accept
  # only the request-level field; the rejudge client sets that explicitly.
  if vllm_supports_option "--default-chat-template-kwargs"; then
    VLLM_CMD+=(--default-chat-template-kwargs "$CHAT_TEMPLATE_KWARGS")
  elif vllm_supports_option "--chat-template-kwargs"; then
    VLLM_CMD+=(--chat-template-kwargs "$CHAT_TEMPLATE_KWARGS")
  else
    echo "[server] no server-level chat-template kwargs option; client will set thinking per request"
  fi
  if [[ "$ENABLE_THINKING" == "1" ]]; then
    if vllm_supports_option "--enable-reasoning" && vllm_supports_option "--reasoning-parser"; then
      VLLM_CMD+=(--enable-reasoning --reasoning-parser "$REASONING_PARSER")
    else
      echo "[server] no reasoning parser CLI; compact judge will use its JSON fallback parser"
    fi
  fi
  [[ -n "$EXTRA_VLLM_ARGS" ]] && read -ra EXTRA_ARGS <<<"$EXTRA_VLLM_ARGS" && VLLM_CMD+=("${EXTRA_ARGS[@]}")
  if [[ -n "$RUNTIME_ROOT_OVERRIDE" ]]; then
    RUNTIME_ROOT="$RUNTIME_ROOT_OVERRIDE"
    [[ "$RUNTIME_ROOT" != "/" ]] || fatal "RUNTIME_ROOT_OVERRIDE cannot be /"
    mkdir -p "$RUNTIME_ROOT"
  else
    RUNTIME_ROOT="$(mktemp -d /data/tmp/jca_gsm_judge.XXXXXX)"
    RUNTIME_ROOT_OWNED=1
  fi

  server_started=0
  for (( attempt = 1; attempt <= START_RETRIES; attempt++ )); do
    attempt_runtime="${RUNTIME_ROOT}/attempt${attempt}"
    mkdir -p "$attempt_runtime/ray" "$attempt_runtime/torchinductor"
    if (( attempt == 1 )); then
      SERVER_LOG="$SERVER_LOG_BASE"
    else
      SERVER_LOG="${SERVER_LOG_BASE%.log}.retry${attempt}.log"
    fi
    echo "[server] starting Qwen3-14B (attempt ${attempt}/${START_RETRIES})"
    setsid env CUDA_VISIBLE_DEVICES="$JUDGE_GPUS" \
      LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
      TMPDIR="$attempt_runtime" \
      RAY_TMPDIR="$attempt_runtime/ray" \
      TORCHINDUCTOR_CACHE_DIR="$attempt_runtime/torchinductor" \
      VLLM_USE_V1="$VLLM_USE_V1" \
      VLLM_USE_FLASHINFER_SAMPLER="$VLLM_USE_FLASHINFER_SAMPLER" \
      VLLM_WORKER_MULTIPROC_METHOD="$VLLM_WORKER_MULTIPROC_METHOD" \
      JCA_PATCH_VLLM_TRANSFORMERS_LM_HEAD="$PATCH_VLLM_TRANSFORMERS_LM_HEAD" \
      PYTHONPATH="${PYTHON_COMPAT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${VLLM_CMD[@]}" >"$SERVER_LOG" 2>&1 &
    SERVER_PID="$!"
    echo "[server] pid=$SERVER_PID log=$SERVER_LOG"
    if wait_for_model "$SERVER_WAIT_TIMEOUT"; then
      server_started=1
      break
    fi
    echo "[server] startup attempt ${attempt}/${START_RETRIES} failed; cleaning process group" >&2
    stop_server
    if (( attempt < START_RETRIES )); then
      sleep 5
    fi
  done
  (( server_started == 1 )) || fatal "Qwen3-14B server failed to start after ${START_RETRIES} attempts"
fi

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  echo "[done] Qwen3-14B server preflight passed"
  exit 0
fi

REJUDGE_CMD=(
  "$PYTHON_BIN" -u "$REJUDGE_SCRIPT"
  --input "$INPUT_PATH" --data-path "$DATA_PATH" --output "$OUTPUT_PATH"
  --stats-output "$STATS_OUTPUT" --max-concurrency "$JUDGE_CONCURRENCY"
  --group-retries "$GROUP_RETRIES" --judge-model "$SERVED_MODEL_NAME"
  --judge-temperature "$JUDGE_TEMPERATURE" --judge-top-p "$JUDGE_TOP_P"
  --judge-max-tokens "$JUDGE_MAX_TOKENS"
  --judge-reasoning-effort "$JUDGE_REASONING_EFFORT"
  --judge-parse-retries "$JUDGE_PARSE_RETRIES"
  --expected-rollouts-per-problem "$EXPECTED_ROLLOUTS_PER_PROBLEM"
)
[[ "$ALLOW_PARTIAL_ROLLOUTS" == "1" ]] && REJUDGE_CMD+=(--allow-partial-rollouts)
if (( LIMIT_GROUPS > 0 )); then
  REJUDGE_CMD+=(--limit-groups "$LIMIT_GROUPS")
fi
if [[ "$DROP_ALL_FAILED_PROBLEMS" == "1" ]]; then
  REJUDGE_CMD+=(--drop-all-failed-problems)
else
  REJUDGE_CMD+=(--no-drop-all-failed-problems)
fi
if [[ "$PROJECT_CONFLICTING_POSITIVE_SCORES" == "1" ]]; then
  REJUDGE_CMD+=(--project-conflicting-positive-scores)
fi
if [[ "$RESUME" == "1" ]]; then
  REJUDGE_CMD+=(--resume)
else
  REJUDGE_CMD+=(--no-resume)
fi
echo "[rejudge] ${REJUDGE_CMD[*]}"
echo "[rejudge] automatic resume passes: $AUTO_RESUME_PASSES"
REJUDGE_ENV=(
  "PYTHONPATH=$PYTHONPATH_ROOT"
  "JCA_JUDGE_API_BASE=http://${HOST}:${PORT}/v1"
  "JCA_JUDGE_API_KEY=EMPTY"
  "JCA_JUDGE_DISABLE_REASONING_EFFORT=1"
  "JCA_JUDGE_CONFIDENT_EXTREMES=$JCA_JUDGE_CONFIDENT_EXTREMES"
)
if [[ "$ENABLE_THINKING" == "0" ]]; then
  REJUDGE_ENV+=("JCA_JUDGE_DISABLE_THINKING=1")
  REJUDGE_ENV+=("JCA_JUDGE_ENABLE_THINKING=0")
else
  # Explicitly override a parent-shell setting so the client cannot silently
  # send enable_thinking=false or rely on a server default.
  REJUDGE_ENV+=("JCA_JUDGE_DISABLE_THINKING=0")
  REJUDGE_ENV+=("JCA_JUDGE_ENABLE_THINKING=1")
fi
rejudge_pass=1
while true; do
  echo "[rejudge] pass ${rejudge_pass}/${AUTO_RESUME_PASSES}"
  if env "${REJUDGE_ENV[@]}" "${REJUDGE_CMD[@]}"; then
    break
  fi
  if (( rejudge_pass >= AUTO_RESUME_PASSES )); then
    fatal "rejudge still incomplete after ${AUTO_RESUME_PASSES} automatic resume passes"
  fi
  rejudge_pass=$((rejudge_pass + 1))
  echo "[rejudge] retrying only incomplete groups with --resume"
done

[[ -s "$OUTPUT_PATH" ]] || fatal "rejudge output is empty: $OUTPUT_PATH"
echo "[done] Qwen3-14B rejudged source: $OUTPUT_PATH"

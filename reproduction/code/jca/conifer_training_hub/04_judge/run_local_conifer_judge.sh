#!/usr/bin/env bash
set -euo pipefail

# Score completed Conifer trajectories with a local Qwen3-14B vLLM judge.
# The three rollout servers must already be stopped so this stage can use all
# eight GPUs, matching the local-judge pattern used by the MATH/GSM pipelines.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/qwen35/bin/python}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-/data/conda_envs/deep_research/bin/python}"
VLLM_LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH:-/data/conda_envs/deep_research/lib}"
INPUT="${INPUT:-${ROOT}/10_outputs/default/train_trajectories.jsonl}"
SCORED_OUTPUT="${SCORED_OUTPUT:-${ROOT}/10_outputs/default/train_scored.jsonl}"
RL_OUTPUT="${RL_OUTPUT:-${ROOT}/13_rl_data/default.jsonl}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_conifer_local_judge}"
RUN_DIR="${RUN_DIR:-${ROOT}/09_logs/judge/${RUN_ID}}"

MODEL_PATH="${JUDGE_MODEL_PATH:-/data/wangyuheng/models/Qwen3-14B}"
SERVED_MODEL_NAME="${JUDGE_MODEL:-conifer_qwen3_14b_judge}"
HOST="${JUDGE_HOST:-127.0.0.1}"
PORT="${JUDGE_PORT:-8314}"
GPUS="${JUDGE_GPUS:-0,1,2,3,4,5,6,7}"
TP="${JUDGE_TP:-1}"
DP="${JUDGE_DP:-8}"
GPU_MEMORY_UTILIZATION="${JUDGE_GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-40960}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-512}"
AUTO_TUNE_CONCURRENCY="${AUTO_TUNE_CONCURRENCY:-1}"
JUDGE_TIMEOUT="${JUDGE_TIMEOUT:-600}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-1536}"
JUDGE_LENGTH_RETRIES="${JUDGE_LENGTH_RETRIES:-1}"
JUDGE_CACHE_SAVE_EVERY="${JUDGE_CACHE_SAVE_EVERY:-256}"
MAX_JUDGE_FAILURE_RATE="${MAX_JUDGE_FAILURE_RATE:-0.05}"
JUDGE_MAX_ROWS="${JUDGE_MAX_ROWS:-0}"
ALPHA="${ALPHA:-0.5}"
SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-900}"
KEEP_SERVER="${KEEP_SERVER:-0}"
SKIP_COMPLETE="${SKIP_COMPLETE:-1}"
VLLM_EXTRA_ARGS="${JUDGE_VLLM_EXTRA_ARGS:-}"
JUDGE_VLLM_MAX_NUM_SEQS="${JUDGE_VLLM_MAX_NUM_SEQS:-96}"
JUDGE_VLLM_MAX_NUM_BATCHED_TOKENS="${JUDGE_VLLM_MAX_NUM_BATCHED_TOKENS:-65536}"
JUDGE_VLLM_ENFORCE_EAGER="${JUDGE_VLLM_ENFORCE_EAGER:-0}"
JUDGE_VLLM_ENABLE_PREFIX_CACHING="${JUDGE_VLLM_ENABLE_PREFIX_CACHING:-1}"
JUDGE_VLLM_ENABLE_CHUNKED_PREFILL="${JUDGE_VLLM_ENABLE_CHUNKED_PREFILL:-1}"
JUDGE_VLLM_PERFORMANCE_MODE="${JUDGE_VLLM_PERFORMANCE_MODE:-throughput}"
JUDGE_VLLM_GENERATION_CONFIG="${JUDGE_VLLM_GENERATION_CONFIG:-auto}"
JUDGE_VLLM_DISABLE_LOG_STATS="${JUDGE_VLLM_DISABLE_LOG_STATS:-0}"
JUDGE_VLLM_ASYNC_SCHEDULING="${JUDGE_VLLM_ASYNC_SCHEDULING:-1}"
JUDGE_VLLM_DISABLE_UVICORN_ACCESS_LOG="${JUDGE_VLLM_DISABLE_UVICORN_ACCESS_LOG:-1}"
JUDGE_VLLM_EXECUTOR_BACKEND="${JUDGE_VLLM_EXECUTOR_BACKEND:-uni}"
JUDGE_VLLM_BACKEND="${JUDGE_VLLM_BACKEND:-replicas}"
JUDGE_REPLICA_PORT_STRIDE="${JUDGE_REPLICA_PORT_STRIDE:-1}"
JUDGE_VLLM_COORD_PORT_BASE="${JUDGE_VLLM_COORD_PORT_BASE:-52800}"
JUDGE_VLLM_DP_MASTER_PORT="${JUDGE_VLLM_DP_MASTER_PORT:-52900}"
JUDGE_VLLM_DP_RPC_PORT="${JUDGE_VLLM_DP_RPC_PORT:-52950}"
JUDGE_VLLM_INTERNAL_PORT_BASE="${JUDGE_VLLM_INTERNAL_PORT_BASE:-53000}"
JUDGE_VLLM_INTERNAL_PORT_STRIDE="${JUDGE_VLLM_INTERNAL_PORT_STRIDE:-32}"
JUDGE_VLLM_START_RETRIES="${JUDGE_VLLM_START_RETRIES:-2}"
JUDGE_VLLM_OMP_NUM_THREADS="${JUDGE_VLLM_OMP_NUM_THREADS:-4}"
JCA_HTTP_KEEPALIVE="${JCA_HTTP_KEEPALIVE:-1}"
JCA_HTTP_POOL_MAXSIZE="${JCA_HTTP_POOL_MAXSIZE:-128}"
JCA_ENDPOINT_RETRIES="${JCA_ENDPOINT_RETRIES:-2}"
PORT_WAIT_TIMEOUT="${PORT_WAIT_TIMEOUT:-180}"
PORT_WAIT_MAX_SLEEP="${PORT_WAIT_MAX_SLEEP:-8}"
export JCA_HTTP_KEEPALIVE JCA_HTTP_POOL_MAXSIZE JCA_ENDPOINT_RETRIES
DRY_RUN="${DRY_RUN:-0}"

[[ "${DRY_RUN}" == 1 || -s "${INPUT}" ]] || { echo "[fatal] judge input not found: ${INPUT}" >&2; exit 1; }
[[ "${DRY_RUN}" == 1 || -d "${MODEL_PATH}" ]] || { echo "[fatal] judge model not found: ${MODEL_PATH}" >&2; exit 1; }
if [[ "${SKIP_COMPLETE}" == 1 && "${JUDGE_MAX_ROWS}" == 0 && -s "${SCORED_OUTPUT}" && -s "${RL_OUTPUT}" && -s "${SCORED_OUTPUT%.jsonl}_stats.json" ]]; then
  echo "[resume] judge outputs are complete; skipping local judge server"
  exit 0
fi
[[ "${TP}" =~ ^[1-9][0-9]*$ && "${DP}" =~ ^[1-9][0-9]*$ ]] || { echo "[fatal] JUDGE_TP/JUDGE_DP must be positive integers" >&2; exit 1; }
case "${AUTO_TUNE_CONCURRENCY}" in 0|1) ;; *) echo "[fatal] AUTO_TUNE_CONCURRENCY must be 0 or 1" >&2; exit 1;; esac
for numeric_name in JUDGE_CONCURRENCY JUDGE_MAX_TOKENS JUDGE_LENGTH_RETRIES JUDGE_VLLM_MAX_NUM_SEQS JUDGE_VLLM_MAX_NUM_BATCHED_TOKENS JUDGE_REPLICA_PORT_STRIDE JUDGE_VLLM_COORD_PORT_BASE JUDGE_VLLM_DP_MASTER_PORT JUDGE_VLLM_DP_RPC_PORT JUDGE_VLLM_INTERNAL_PORT_BASE JUDGE_VLLM_INTERNAL_PORT_STRIDE JUDGE_VLLM_START_RETRIES JUDGE_VLLM_OMP_NUM_THREADS PORT_WAIT_TIMEOUT PORT_WAIT_MAX_SLEEP; do
  numeric_value="${!numeric_name}"
  if [[ "${numeric_name}" == JUDGE_LENGTH_RETRIES ]]; then
    [[ "${numeric_value}" =~ ^[0-9]+$ ]] || { echo "[fatal] ${numeric_name} must be a non-negative integer" >&2; exit 1; }
  else
    [[ "${numeric_value}" =~ ^[1-9][0-9]*$ ]] || { echo "[fatal] ${numeric_name} must be a positive integer" >&2; exit 1; }
  fi
done
[[ "${JCA_ENDPOINT_RETRIES}" =~ ^[0-9]+$ ]] || { echo "[fatal] JCA_ENDPOINT_RETRIES must be a non-negative integer" >&2; exit 1; }
case "${JUDGE_VLLM_EXECUTOR_BACKEND}" in uni|mp) ;; *) echo "[fatal] JUDGE_VLLM_EXECUTOR_BACKEND must be uni or mp" >&2; exit 1;; esac
case "${JUDGE_VLLM_BACKEND}" in replicas|dp) ;; *) echo "[fatal] JUDGE_VLLM_BACKEND must be replicas or dp" >&2; exit 1;; esac
for boolean_name in JUDGE_VLLM_ENFORCE_EAGER JUDGE_VLLM_ENABLE_PREFIX_CACHING JUDGE_VLLM_ENABLE_CHUNKED_PREFILL JUDGE_VLLM_DISABLE_LOG_STATS JUDGE_VLLM_ASYNC_SCHEDULING JUDGE_VLLM_DISABLE_UVICORN_ACCESS_LOG; do
  boolean_value="${!boolean_name}"
  [[ "${boolean_value}" =~ ^[01]$ ]] || { echo "[fatal] ${boolean_name} must be 0 or 1" >&2; exit 1; }
done
gpu_count="$(awk -F',' '{print NF}' <<<"${GPUS}")"
[[ "${gpu_count}" == "$((TP * DP))" ]] || { echo "[fatal] JUDGE_GPUS count must equal JUDGE_TP*JUDGE_DP" >&2; exit 1; }

check_ports_available() {
  "${PYTHON_BIN}" - "${HOST}" "$@" <<'PY'
import socket
import sys

host = sys.argv[1]
busy = []
for raw_port in sys.argv[2:]:
    port = int(raw_port)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError as exc:
            busy.append(f"{port} ({exc})")
if busy:
    print("unavailable judge port(s): " + ", ".join(busy), file=sys.stderr)
    raise SystemExit(1)
PY
}

wait_ports_available() {
  local timeout="$1"
  shift
  local deadline=$((SECONDS + timeout))
  local sleep_for=0.25
  while :; do
    if check_ports_available "$@" >/dev/null 2>&1; then
      return 0
    fi
    if ((SECONDS >= deadline)); then
      check_ports_available "$@"
      return 1
    fi
    sleep "${sleep_for}"
    if (( $(awk -v value="${sleep_for}" 'BEGIN { print (value < 1) }') )); then
      sleep_for=1
    elif (( $(awk -v value="${sleep_for}" -v max="${PORT_WAIT_MAX_SLEEP}" 'BEGIN { print (value * 2 < max) }') )); then
      sleep_for=$(awk -v value="${sleep_for}" 'BEGIN { printf "%.2f", value * 2 }')
    else
      sleep_for="${PORT_WAIT_MAX_SLEEP}"
    fi
  done
}

startup_ports=()
if [[ "${JUDGE_VLLM_BACKEND}" == replicas ]]; then
  startup_gpu_list=()
  IFS=',' read -r -a startup_gpu_list <<<"${GPUS}"
  for startup_rank in "${!startup_gpu_list[@]}"; do
    startup_ports+=("$((PORT + startup_rank * JUDGE_REPLICA_PORT_STRIDE))")
  done
else
  startup_ports+=("${PORT}")
fi
if [[ "${DRY_RUN}" != 1 ]]; then
  wait_ports_available "${PORT_WAIT_TIMEOUT}" "${startup_ports[@]}" || { echo "[fatal] judge API port pool is unavailable" >&2; exit 1; }
fi

mkdir -p "${RUN_DIR}" "$(dirname "${SCORED_OUTPUT}")" "$(dirname "${RL_OUTPUT}")"
exec > >(tee -a "${RUN_DIR}/run.log") 2>&1

server_pid=""
server_pids=()
judge_endpoint_pool=""
stop_process_group() {
  local pid="$1"
  [[ -n "${pid}" ]] || return 0
  local session_id
  session_id="$(ps -o sid= -p "${pid}" 2>/dev/null | tr -d '[:space:]' || true)"
  [[ -n "${session_id}" ]] || session_id="${pid}"
  kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  pkill -TERM -s "${session_id}" 2>/dev/null || true
  local deadline=$((SECONDS + 60))
  while kill -0 -- "-${pid}" 2>/dev/null || kill -0 "${pid}" 2>/dev/null || ps -eo sid= | awk -v sid="${session_id}" '$1 == sid {found=1; exit} END {exit !found}'; do
    if ((SECONDS >= deadline)); then
      kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
      pkill -KILL -s "${session_id}" 2>/dev/null || true
      break
    fi
    sleep 1
  done
  wait "${pid}" 2>/dev/null || true
}

stop_process_groups_parallel() {
  local -a wait_pids=()
  local pid
  for pid in "$@"; do
    [[ -n "${pid}" ]] || continue
    stop_process_group "${pid}" &
    wait_pids+=("$!")
  done
  local status=0 wait_pid
  for wait_pid in "${wait_pids[@]}"; do
    wait "${wait_pid}" || status=1
  done
  return "${status}"
}

cleanup() {
  local code=$?
  trap - EXIT INT TERM
  if [[ "${KEEP_SERVER}" != 1 ]]; then
    if [[ "${JUDGE_VLLM_BACKEND}" == replicas ]]; then
      echo "[server] stopping local judge replica process groups"
      stop_process_groups_parallel "${server_pids[@]}" || true
    elif [[ -n "${server_pid}" ]]; then
      echo "[server] stopping local judge process group ${server_pid}"
      stop_process_group "${server_pid}"
    fi
  fi
  exit "${code}"
}
trap cleanup EXIT INT TERM

wait_ready() {
  local url="$1" pid="$2" timeout="${3:-${SERVER_WAIT_TIMEOUT}}" expected="${4:-${SERVED_MODEL_NAME}}"
  "${PYTHON_BIN}" - "${url}" "${timeout}" "${expected}" "${pid}" <<'PY'
import json
import os
import sys
import time
import urllib.request

url, timeout, expected, server_pid = sys.argv[1], float(sys.argv[2]), sys.argv[3], int(sys.argv[4])
deadline = time.time() + timeout
last = None
while time.time() < deadline:
    try:
        os.kill(server_pid, 0)
    except OSError:
        print(f"server exited before ready (pid={server_pid})", file=sys.stderr)
        raise SystemExit(1)
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = json.loads(response.read().decode())
        names = {str(item.get("id")) for item in payload.get("data") or []}
        if expected in names:
            print("ready", url, expected)
            raise SystemExit(0)
    except Exception as exc:
        last = exc
    time.sleep(3)
print("server timeout", url, last, file=sys.stderr)
raise SystemExit(1)
PY
}

vllm_cmd=(
  "${VLLM_PYTHON_BIN}" "${ROOT}/06_evaluation/serve_vllm_deterministic.py" api
  --host "${HOST}" --port "${PORT}" --model "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --tensor-parallel-size "${TP}" --data-parallel-size "${DP}"
  --distributed-executor-backend "${JUDGE_VLLM_EXECUTOR_BACKEND}"
  --data-parallel-rpc-port "${JUDGE_VLLM_DP_RPC_PORT}"
  --dtype bfloat16 --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}" --trust-remote-code
  --max-num-seqs "${JUDGE_VLLM_MAX_NUM_SEQS}"
  --max-num-batched-tokens "${JUDGE_VLLM_MAX_NUM_BATCHED_TOKENS}"
  --default-chat-template-kwargs '{"enable_thinking":false}'
)
if [[ "${JUDGE_VLLM_ENABLE_PREFIX_CACHING}" == 1 ]]; then vllm_cmd+=(--enable-prefix-caching); fi
if [[ "${JUDGE_VLLM_ENABLE_CHUNKED_PREFILL}" == 1 ]]; then vllm_cmd+=(--enable-chunked-prefill); fi
if [[ "${JUDGE_VLLM_ENFORCE_EAGER}" == 1 ]]; then vllm_cmd+=(--enforce-eager); fi
if [[ -n "${JUDGE_VLLM_PERFORMANCE_MODE}" ]]; then vllm_cmd+=(--performance-mode "${JUDGE_VLLM_PERFORMANCE_MODE}"); fi
if [[ "${JUDGE_VLLM_GENERATION_CONFIG}" != auto ]]; then vllm_cmd+=(--generation-config "${JUDGE_VLLM_GENERATION_CONFIG}"); fi
if [[ "${JUDGE_VLLM_DISABLE_LOG_STATS}" == 1 ]]; then vllm_cmd+=(--disable-log-stats); fi
if [[ "${JUDGE_VLLM_ASYNC_SCHEDULING}" == 1 ]]; then vllm_cmd+=(--async-scheduling); fi
if [[ "${JUDGE_VLLM_DISABLE_UVICORN_ACCESS_LOG}" == 1 ]]; then vllm_cmd+=(--disable-uvicorn-access-log); fi
if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then
  read -r -a extra_args <<<"${VLLM_EXTRA_ARGS}"
  vllm_cmd+=("${extra_args[@]}")
fi

if [[ "${JUDGE_VLLM_BACKEND}" == replicas ]]; then
  judge_endpoint_pool=""
  declare -a judge_gpu_list=()
  IFS=',' read -r -a judge_gpu_list <<<"${GPUS}"
  for judge_rank in "${!judge_gpu_list[@]}"; do
    judge_port=$((PORT + judge_rank * JUDGE_REPLICA_PORT_STRIDE))
    [[ -z "${judge_endpoint_pool}" ]] || judge_endpoint_pool+=","
    judge_endpoint_pool+="http://${HOST}:${judge_port}/v1"
  done
else
  judge_endpoint_pool="http://${HOST}:${PORT}/v1"
fi

if [[ "${AUTO_TUNE_CONCURRENCY}" == 1 ]]; then
  if [[ "${JUDGE_VLLM_BACKEND}" == replicas ]]; then
    judge_gpu_count="$(awk -F',' '{print NF}' <<<"${GPUS}")"
  else
    judge_gpu_count="${DP}"
  fi
  JUDGE_CONCURRENCY=$((judge_gpu_count * JUDGE_VLLM_MAX_NUM_SEQS))
  if (( JCA_HTTP_POOL_MAXSIZE < JUDGE_VLLM_MAX_NUM_SEQS )); then
    JCA_HTTP_POOL_MAXSIZE="${JUDGE_VLLM_MAX_NUM_SEQS}"
    export JCA_HTTP_POOL_MAXSIZE
  fi
  echo "[throughput] auto judge concurrency=${JUDGE_CONCURRENCY} (${judge_gpu_count} endpoint(s) x ${JUDGE_VLLM_MAX_NUM_SEQS} sequence(s))"
fi

score_cmd=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/score_conifer_rollouts.py"
  --input "${INPUT}" --scored-output "${SCORED_OUTPUT}" --rl-output "${RL_OUTPUT}"
  --judge-mode llm --judge-model "${SERVED_MODEL_NAME}"
  --judge-api-base "${judge_endpoint_pool}" --judge-api-key EMPTY
  --judge-timeout "${JUDGE_TIMEOUT}" --judge-concurrency "${JUDGE_CONCURRENCY}"
  --judge-max-tokens "${JUDGE_MAX_TOKENS}" --judge-length-retries "${JUDGE_LENGTH_RETRIES}"
  --judge-cache-save-every "${JUDGE_CACHE_SAVE_EVERY}"
  --require-complete-judge --allow-judge-failure
  --max-judge-failure-rate "${MAX_JUDGE_FAILURE_RATE}" --alpha "${ALPHA}"
)
if [[ "${JUDGE_MAX_ROWS}" != 0 ]]; then
  score_cmd+=(--max-rows "${JUDGE_MAX_ROWS}")
fi

printf '[server-command] '; printf '%q ' "${vllm_cmd[@]}"; printf '\n'
printf '[score-command] '; printf '%q ' "${score_cmd[@]}"; printf '\n'
if [[ "${DRY_RUN}" == 1 ]]; then
  exit 0
fi

start_dp_server() {
  setsid env -u VLLM_MAX_NUM_SEQS -u VLLM_MAX_NUM_BATCHED_TOKENS \
    -u VLLM_ENFORCE_EAGER -u VLLM_ENABLE_PREFIX_CACHING \
    -u VLLM_ENABLE_CHUNKED_PREFILL -u VLLM_PERFORMANCE_MODE \
    -u VLLM_GENERATION_CONFIG -u VLLM_DISABLE_LOG_STATS \
    -u VLLM_ASYNC_SCHEDULING -u VLLM_DISABLE_UVICORN_ACCESS_LOG \
    CUDA_VISIBLE_DEVICES="${GPUS}" \
    VLLM_PORT="${JUDGE_VLLM_COORD_PORT_BASE}" \
    VLLM_DP_MASTER_IP="${HOST}" VLLM_DP_MASTER_PORT="${JUDGE_VLLM_DP_MASTER_PORT}" \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    CONIFER_DP_PORT_BASE="${JUDGE_VLLM_INTERNAL_PORT_BASE}" \
    CONIFER_DP_PORT_STRIDE="${JUDGE_VLLM_INTERNAL_PORT_STRIDE}" \
    OMP_NUM_THREADS="${JUDGE_VLLM_OMP_NUM_THREADS}" \
    LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "${vllm_cmd[@]}" >"${RUN_DIR}/server.log" 2>&1 &
  server_pid="$!"
  echo "[server] pid=${server_pid} log=${RUN_DIR}/server.log"
}

start_replica_servers() {
  local -a gpu_list=()
  IFS=',' read -r -a gpu_list <<<"${GPUS}"
  [[ "${TP}" == 1 ]] || { echo "[fatal] JUDGE_VLLM_BACKEND=replicas requires JUDGE_TP=1" >&2; exit 1; }
  server_pids=()
  local rank gpu port
  for rank in "${!gpu_list[@]}"; do
    gpu="${gpu_list[rank]}"
    gpu="${gpu//[[:space:]]/}"
    [[ -n "${gpu}" ]] || { echo "[fatal] empty GPU entry in JUDGE_GPUS" >&2; exit 1; }
    port=$((PORT + rank * JUDGE_REPLICA_PORT_STRIDE))
    local -a command=(
      "${VLLM_PYTHON_BIN}" -m vllm.entrypoints.openai.api_server
      --host "${HOST}" --port "${port}" --model "${MODEL_PATH}"
      --served-model-name "${SERVED_MODEL_NAME}" --tensor-parallel-size 1
      --dtype bfloat16 --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
      --max-model-len "${MAX_MODEL_LEN}" --max-num-seqs "${JUDGE_VLLM_MAX_NUM_SEQS}"
      --max-num-batched-tokens "${JUDGE_VLLM_MAX_NUM_BATCHED_TOKENS}" --trust-remote-code
      --default-chat-template-kwargs '{"enable_thinking":false}'
    )
    if [[ "${JUDGE_VLLM_ENABLE_PREFIX_CACHING}" == 1 ]]; then command+=(--enable-prefix-caching); fi
    if [[ "${JUDGE_VLLM_ENABLE_CHUNKED_PREFILL}" == 1 ]]; then command+=(--enable-chunked-prefill); fi
    if [[ "${JUDGE_VLLM_ENFORCE_EAGER}" == 1 ]]; then command+=(--enforce-eager); fi
    if [[ -n "${JUDGE_VLLM_PERFORMANCE_MODE}" ]]; then command+=(--performance-mode "${JUDGE_VLLM_PERFORMANCE_MODE}"); fi
    if [[ "${JUDGE_VLLM_GENERATION_CONFIG}" != auto ]]; then command+=(--generation-config "${JUDGE_VLLM_GENERATION_CONFIG}"); fi
    if [[ "${JUDGE_VLLM_DISABLE_LOG_STATS}" == 1 ]]; then command+=(--disable-log-stats); fi
    if [[ "${JUDGE_VLLM_ASYNC_SCHEDULING}" == 1 ]]; then command+=(--async-scheduling); fi
    if [[ "${JUDGE_VLLM_DISABLE_UVICORN_ACCESS_LOG}" == 1 ]]; then command+=(--disable-uvicorn-access-log); fi
    if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then read -r -a extra_args <<<"${VLLM_EXTRA_ARGS}"; command+=("${extra_args[@]}"); fi
    echo "[server] judge replica=${rank} model=${MODEL_PATH} GPU=${gpu} port=${port}"
    setsid env -u VLLM_MAX_NUM_SEQS -u VLLM_MAX_NUM_BATCHED_TOKENS \
      -u VLLM_ENFORCE_EAGER -u VLLM_ENABLE_PREFIX_CACHING \
      -u VLLM_ENABLE_CHUNKED_PREFILL -u VLLM_PERFORMANCE_MODE \
      -u VLLM_GENERATION_CONFIG -u VLLM_DISABLE_LOG_STATS \
      -u VLLM_ASYNC_SCHEDULING -u VLLM_DISABLE_UVICORN_ACCESS_LOG \
      -u VLLM_PORT -u VLLM_DP_MASTER_IP -u VLLM_DP_MASTER_PORT \
      CUDA_VISIBLE_DEVICES="${gpu}" VLLM_ENABLE_V1_MULTIPROCESSING=0 \
      OMP_NUM_THREADS="${JUDGE_VLLM_OMP_NUM_THREADS}" \
      LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
      "${command[@]}" >"${RUN_DIR}/replica_${rank}_server.log" 2>&1 &
    server_pids+=("$!")
  done
}

wait_replica_servers() {
  local -a gpu_list=()
  IFS=',' read -r -a gpu_list <<<"${GPUS}"
  local rank port
  local -a wait_pids=()
  for rank in "${!gpu_list[@]}"; do
    port=$((PORT + rank * JUDGE_REPLICA_PORT_STRIDE))
    wait_ready "http://${HOST}:${port}/v1/models" "${server_pids[rank]}" 900 "${SERVED_MODEL_NAME}" &
    wait_pids+=("$!")
  done
  local status=0 wait_pid
  for wait_pid in "${wait_pids[@]}"; do
    if ! wait "${wait_pid}"; then
      status=1
    fi
  done
  if ((status != 0)); then
    return 1
  fi
  judge_endpoint_pool=""
  for rank in "${!gpu_list[@]}"; do
    port=$((PORT + rank * JUDGE_REPLICA_PORT_STRIDE))
    [[ -z "${judge_endpoint_pool}" ]] || judge_endpoint_pool+=","
    judge_endpoint_pool+="http://${HOST}:${port}/v1"
  done
}

server_attempts=$((JUDGE_VLLM_START_RETRIES + 1))
server_attempt=0
while ((server_attempt < server_attempts)); do
  server_attempt=$((server_attempt + 1))
  if [[ "${JUDGE_VLLM_BACKEND}" == replicas ]]; then
    start_replica_servers
    if wait_replica_servers; then
      break
    fi
    echo "[server] judge replica startup attempt ${server_attempt}/${server_attempts} failed; recycling process groups" >&2
    stop_process_groups_parallel "${server_pids[@]}" || true
    server_pids=()
  else
    start_dp_server
    if wait_ready "http://${HOST}:${PORT}/v1/models" "${server_pid}" 900 "${SERVED_MODEL_NAME}"; then
      judge_endpoint_pool="http://${HOST}:${PORT}/v1"
      break
    fi
    echo "[server] judge startup attempt ${server_attempt}/${server_attempts} failed; recycling process group ${server_pid}" >&2
    stop_process_group "${server_pid}"
    server_pid=""
  fi
  if ((server_attempt < server_attempts)); then
    wait_ports_available "${PORT_WAIT_TIMEOUT}" "${startup_ports[@]}" || true
    sleep 1
  fi
done
if [[ "${JUDGE_VLLM_BACKEND}" == replicas ]]; then
  [[ "${#server_pids[@]}" -gt 0 ]] || { echo "[fatal] local judge replicas failed to become ready" >&2; exit 1; }
else
  [[ -n "${server_pid}" ]] || { echo "[fatal] local judge failed to become ready" >&2; exit 1; }
fi
"${score_cmd[@]}"
echo "[ok] local Conifer judging complete: ${SCORED_OUTPUT}"

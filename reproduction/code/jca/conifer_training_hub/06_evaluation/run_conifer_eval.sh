#!/usr/bin/env bash
set -euo pipefail

# End-to-end Conifer holdout evaluation.  The generated trajectory file is
# subsequently scored deterministically and, optionally, by an LLM judge.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${ROOT}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/qwen35/bin/python}"
vllm_python_usable() {
  local candidate="$1" help_output required_option
  [[ -x "${candidate}" ]] || return 1
  help_output="$("${candidate}" -m vllm.entrypoints.openai.api_server --help 2>&1)" || return 1
  for required_option in --enable-lora --performance-mode --async-scheduling --disable-uvicorn-access-log; do
    [[ "${help_output}" == *"${required_option}"* ]] || return 1
  done
}
if [[ -z "${VLLM_PYTHON_BIN:-}" ]]; then
  for _candidate in \
    "/data/conda_envs/deep_research/bin/python" \
    "/data/conda_envs/drb_py311_clean/bin/python" \
    "/data/conda_envs/qwen35/bin/python" \
    "$(command -v python3 2>/dev/null || true)"; do
    if vllm_python_usable "${_candidate}"; then
      VLLM_PYTHON_BIN="${_candidate}"
      break
    fi
  done
fi
if [[ -z "${VLLM_PYTHON_BIN:-}" || ! -x "${VLLM_PYTHON_BIN}" ]]; then
  echo "[fatal] could not find a usable Python interpreter with vLLM; set VLLM_PYTHON_BIN explicitly" >&2
  exit 1
fi
if [[ -z "${VLLM_LD_LIBRARY_PATH:-}" ]]; then
  VLLM_LD_LIBRARY_PATH="$(cd "$(dirname "${VLLM_PYTHON_BIN}")/.." && pwd)/lib"
fi
MODE="${MODE:-zero_shot_mas}" # zero_shot_mas | teacher_rollout_mas | sft_lora_mas | baseline_* | single_*
REQUESTED_MODE="${MODE}"
BASELINE_STRATEGY="${BASELINE_STRATEGY:-none}"
BASELINE_MAD_TEMPERATURE_ROUND0="${BASELINE_MAD_TEMPERATURE_ROUND0:-0.9}"
BASELINE_MAD_TEMPERATURE_DEBATE="${BASELINE_MAD_TEMPERATURE_DEBATE:-0.3}"
BASELINE_AGENT_TEMPERATURE="${BASELINE_AGENT_TEMPERATURE:-0.7}"
BASELINE_META_TEMPERATURE="${BASELINE_META_TEMPERATURE:-0.0}"
BASELINE_SCORE_THRESHOLD="${BASELINE_SCORE_THRESHOLD:-8}"
BASELINE_GPTSWARM_TEMPERATURE="${BASELINE_GPTSWARM_TEMPERATURE:-0.7}"
BASELINE_AFLOW_TEMPERATURE="${BASELINE_AFLOW_TEMPERATURE:-0.7}"
# Empty keeps the hardcoded 5-op graph (previous behaviour).  Set it to a
# searched workflow produced by 03_rollout/aflow/run_search_conifer.py to run
# the real AFlow method.
BASELINE_AFLOW_WORKFLOW_FILE="${BASELINE_AFLOW_WORKFLOW_FILE:-}"
BASELINE_ROUNDS="${BASELINE_ROUNDS:-3}"
BASELINE_ITERATIONS="${BASELINE_ITERATIONS:-3}"
USE_ADAPTERS=0
SINGLE_MODE=0
case "${MODE}" in
  mad|baseline_mad) MODE=baseline_mad; BASELINE_STRATEGY=mad ;;
  agentverse|baseline_agentverse) MODE=baseline_agentverse; BASELINE_STRATEGY=agentverse ;;
  gptswarm|baseline_gptswarm) MODE=baseline_gptswarm; BASELINE_STRATEGY=gptswarm ;;
  aflow|baseline_aflow) MODE=baseline_aflow; BASELINE_STRATEGY=aflow ;;
  wo_sft|no_sft|wo_process_reward|no_process_reward) MODE=sft_lora_mas; USE_ADAPTERS=1 ;;
  sft_lora_mas) USE_ADAPTERS=1 ;;
  single_a3) SINGLE_MODE=1; USE_ADAPTERS=1 ;;
  single_14b) SINGLE_MODE=1 ;;
esac
DATA_PATH="${DATA_PATH:-${ROOT}/01_dataset/processed/test.jsonl}"
START="${START:-0}"
LIMIT="${LIMIT:-0}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-1}"
T_MAX="${T_MAX:-6}"
START_AGENT="${START_AGENT:-balanced}"
START_AGENT_SEED="${START_AGENT_SEED:-42}"
SEED="${SEED:-42}"
MIN_AGENTS_BEFORE_STOP="${MIN_AGENTS_BEFORE_STOP:-3}"
MIN_HANDOFFS_BEFORE_STOP="${MIN_HANDOFFS_BEFORE_STOP:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}"
PROTOCOL_MAX_NEW_TOKENS="${PROTOCOL_MAX_NEW_TOKENS:-1024}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.95}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-768}"
AUTO_TUNE_CONCURRENCY="${AUTO_TUNE_CONCURRENCY:-1}"
CONTEXT_TURNS="${CONTEXT_TURNS:-2}"
JSON_TRANSPORT="${JSON_TRANSPORT:-json_schema}"
ROUTING="${ROUTING:-dynamic}"
FORCE_HANDOFF_UNTIL_FINAL="${FORCE_HANDOFF_UNTIL_FINAL:-0}"
JUDGE_MODE="${JUDGE_MODE:-deterministic}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5}"
ALPHA="${ALPHA:-0.5}"
SCORE_OUTPUTS="${SCORE_OUTPUTS:-1}"
ROLLOUT_STAGE="${ROLLOUT_STAGE:-unknown}"
DRY_RUN="${DRY_RUN:-0}"
KEEP_SERVERS="${KEEP_SERVERS:-0}"
MOCK="${MOCK:-0}"

SINGLE_MODEL_PATH="${SINGLE_MODEL_PATH:-/data/wangyuheng/models/Qwen3-14B}"
SINGLE_MODEL_NAME="${SINGLE_MODEL_NAME:-ConiferSingle14B}"
SINGLE_MODEL_PORT="${SINGLE_MODEL_PORT:-8304}"
SINGLE_MODEL_GPUS="${SINGLE_MODEL_GPUS:-0,1,2,3,4,5,6,7}"
SINGLE_MODEL_TP="${SINGLE_MODEL_TP:-8}"
SINGLE_REPLICA_MODE="${SINGLE_REPLICA_MODE:-0}"

MODEL_A1="${MODEL_A1:-/data/wangyuheng/models/Qwen3-1.7B}"
MODEL_A2="${MODEL_A2:-/data/wangyuheng/models/Qwen3-4B}"
MODEL_A3="${MODEL_A3:-/data/wangyuheng/models/Qwen3-8B}"
ROLLOUT_MODEL_A1="${ROLLOUT_MODEL_A1:-/data/wangyuheng/models/Qwen3-8B}"
ROLLOUT_MODEL_A2="${ROLLOUT_MODEL_A2:-/data/wangyuheng/models/Qwen3-8B}"
ROLLOUT_MODEL_A3="${ROLLOUT_MODEL_A3:-/data/wangyuheng/models/Qwen3-8B}"
ROLLOUT_IS_TEACHER="${ROLLOUT_IS_TEACHER:-0}"
ROLLOUT_SOURCE_POLICY="${ROLLOUT_SOURCE_POLICY:-unknown}"
ADAPTER_A1="${ADAPTER_A1:-}"
ADAPTER_A2="${ADAPTER_A2:-}"
ADAPTER_A3="${ADAPTER_A3:-}"

SERVE_MODEL_A1="${MODEL_A1}"
SERVE_MODEL_A2="${MODEL_A2}"
SERVE_MODEL_A3="${MODEL_A3}"
if [[ "${MODE}" == teacher_rollout_mas || "${ROLLOUT_IS_TEACHER}" == 1 ]]; then
  SERVE_MODEL_A1="${ROLLOUT_MODEL_A1}"
  SERVE_MODEL_A2="${ROLLOUT_MODEL_A2}"
  SERVE_MODEL_A3="${ROLLOUT_MODEL_A3}"
fi

HOST="${HOST:-127.0.0.1}"
A1_PORT="${A1_PORT:-8301}"; A2_PORT="${A2_PORT:-8302}"; A3_PORT="${A3_PORT:-8303}"
A1_GPUS="${A1_GPUS:-0,1}"; A2_GPUS="${A2_GPUS:-2,3}"; A3_GPUS="${A3_GPUS:-4,5,6,7}"
A1_TP="${A1_TP:-2}"; A2_TP="${A2_TP:-2}"; A3_TP="${A3_TP:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-96}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-98304}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-1}"
VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-1}"
VLLM_PERFORMANCE_MODE="${VLLM_PERFORMANCE_MODE:-throughput}"
VLLM_GENERATION_CONFIG="${VLLM_GENERATION_CONFIG:-auto}"
VLLM_DISABLE_LOG_STATS="${VLLM_DISABLE_LOG_STATS:-0}"
VLLM_ASYNC_SCHEDULING="${VLLM_ASYNC_SCHEDULING:-1}"
VLLM_DISABLE_UVICORN_ACCESS_LOG="${VLLM_DISABLE_UVICORN_ACCESS_LOG:-1}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

# A homogeneous teacher rollout benefits from one replica per GPU.  ``auto``
# enables this only when all three served models are identical; set
# SHARED_VLLM=0 to retain three independent role endpoints.
SHARED_VLLM="${SHARED_VLLM:-auto}"
SHARED_VLLM_BACKEND="${SHARED_VLLM_BACKEND:-replicas}"
SHARED_VLLM_PORT="${SHARED_VLLM_PORT:-8301}"
SHARED_VLLM_REPLICA_PORT_STRIDE="${SHARED_VLLM_REPLICA_PORT_STRIDE:-1}"
SHARED_VLLM_GPUS="${SHARED_VLLM_GPUS:-0,1,2,3,4,5,6,7}"
SHARED_VLLM_TP="${SHARED_VLLM_TP:-1}"
SHARED_VLLM_DP="${SHARED_VLLM_DP:-8}"
SHARED_VLLM_MAX_NUM_SEQS="${SHARED_VLLM_MAX_NUM_SEQS:-96}"
SHARED_VLLM_MAX_NUM_BATCHED_TOKENS="${SHARED_VLLM_MAX_NUM_BATCHED_TOKENS:-98304}"
SHARED_VLLM_MODEL_NAME="${SHARED_VLLM_MODEL_NAME:-ConiferShared8B}"
SHARED_VLLM_EXECUTOR_BACKEND="${SHARED_VLLM_EXECUTOR_BACKEND:-uni}"
SHARED_VLLM_COORD_PORT_BASE="${SHARED_VLLM_COORD_PORT_BASE:-51800}"
SHARED_VLLM_DP_MASTER_PORT="${SHARED_VLLM_DP_MASTER_PORT:-51900}"
SHARED_VLLM_DP_RPC_PORT="${SHARED_VLLM_DP_RPC_PORT:-51950}"
SHARED_VLLM_INTERNAL_PORT_BASE="${SHARED_VLLM_INTERNAL_PORT_BASE:-52000}"
SHARED_VLLM_INTERNAL_PORT_STRIDE="${SHARED_VLLM_INTERNAL_PORT_STRIDE:-32}"
SHARED_VLLM_START_RETRIES="${SHARED_VLLM_START_RETRIES:-2}"
SHARED_VLLM_OMP_NUM_THREADS="${SHARED_VLLM_OMP_NUM_THREADS:-4}"
ROLE_REPLICA_MODE="${ROLE_REPLICA_MODE:-auto}"
ROLE_REPLICA_PORT_BASE="${ROLE_REPLICA_PORT_BASE:-8401}"
ROLE_REPLICA_ROLE_PORT_STRIDE="${ROLE_REPLICA_ROLE_PORT_STRIDE:-16}"
ROLE_REPLICA_PORT_STRIDE="${ROLE_REPLICA_PORT_STRIDE:-1}"
ROLE_REPLICA_PORT_AUTO_SHIFT="${ROLE_REPLICA_PORT_AUTO_SHIFT:-1}"
ROLE_REPLICA_PORT_SCAN_BLOCKS="${ROLE_REPLICA_PORT_SCAN_BLOCKS:-64}"
ROLE_REPLICA_MAX_NUM_SEQS="${ROLE_REPLICA_MAX_NUM_SEQS:-${SHARED_VLLM_MAX_NUM_SEQS}}"
ROLE_REPLICA_MAX_NUM_BATCHED_TOKENS="${ROLE_REPLICA_MAX_NUM_BATCHED_TOKENS:-${SHARED_VLLM_MAX_NUM_BATCHED_TOKENS}}"
ROLE_REPLICA_START_RETRIES="${ROLE_REPLICA_START_RETRIES:-4}"
ROLE_REPLICA_OMP_NUM_THREADS="${ROLE_REPLICA_OMP_NUM_THREADS:-${SHARED_VLLM_OMP_NUM_THREADS}}"
PORT_WAIT_TIMEOUT="${PORT_WAIT_TIMEOUT:-180}"
PORT_WAIT_MAX_SLEEP="${PORT_WAIT_MAX_SLEEP:-8}"
SERVER_HEALTH_INTERVAL="${SERVER_HEALTH_INTERVAL:-2}"
JCA_HTTP_KEEPALIVE="${JCA_HTTP_KEEPALIVE:-1}"
JCA_HTTP_POOL_MAXSIZE="${JCA_HTTP_POOL_MAXSIZE:-128}"
JCA_ENDPOINT_RETRIES="${JCA_ENDPOINT_RETRIES:-2}"
export JCA_HTTP_KEEPALIVE JCA_HTTP_POOL_MAXSIZE JCA_ENDPOINT_RETRIES

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_conifer_${MODE}}"
RUN_DIR="${RUN_DIR:-${ROOT}/09_logs/eval/${RUN_ID}}"
OUTPUT="${OUTPUT:-${ROOT}/10_outputs/${RUN_ID}/trajectories.jsonl}"
SCORED_OUTPUT="${SCORED_OUTPUT:-${ROOT}/10_outputs/${RUN_ID}/scored.jsonl}"
RL_OUTPUT="${RL_OUTPUT:-${ROOT}/13_rl_data/${RUN_ID}.jsonl}"
CONFIG_PATH="${CONFIG_PATH:-${RUN_DIR}/config.env}"

[[ "${DRY_RUN}" == 1 || -s "${DATA_PATH}" ]] || { echo "[fatal] data not found: ${DATA_PATH}" >&2; exit 1; }
case "${MODE}" in
  zero_shot_mas|teacher_rollout_mas|sft_lora_mas|single_a3|single_14b|baseline_mad|baseline_agentverse|baseline_gptswarm|baseline_aflow) ;;
  *) echo "[fatal] invalid MODE=${REQUESTED_MODE}" >&2; exit 1 ;;
esac
case "${BASELINE_STRATEGY}" in none|mad|agentverse|gptswarm|aflow) ;; *) echo "[fatal] invalid BASELINE_STRATEGY=${BASELINE_STRATEGY}" >&2; exit 1;; esac
[[ "${BASELINE_ROUNDS}" =~ ^[1-9][0-9]*$ ]] || { echo "[fatal] BASELINE_ROUNDS must be positive" >&2; exit 1; }
[[ "${BASELINE_ITERATIONS}" =~ ^[1-9][0-9]*$ ]] || { echo "[fatal] BASELINE_ITERATIONS must be positive" >&2; exit 1; }
[[ "${BASELINE_SCORE_THRESHOLD}" =~ ^([0-9]|10)$ ]] || { echo "[fatal] BASELINE_SCORE_THRESHOLD must be in [0,10]" >&2; exit 1; }
case "${ROLLOUT_IS_TEACHER}" in 0|1) ;; *) echo "[fatal] ROLLOUT_IS_TEACHER must be 0 or 1" >&2; exit 1;; esac
case "${SCORE_OUTPUTS}" in 0|1) ;; *) echo "[fatal] SCORE_OUTPUTS must be 0 or 1" >&2; exit 1;; esac
case "${AUTO_TUNE_CONCURRENCY}" in 0|1) ;; *) echo "[fatal] AUTO_TUNE_CONCURRENCY must be 0 or 1" >&2; exit 1;; esac
case "${SHARED_VLLM}" in auto|0|1) ;; *) echo "[fatal] SHARED_VLLM must be auto, 0, or 1" >&2; exit 1;; esac
case "${SHARED_VLLM_BACKEND}" in replicas|dp) ;; *) echo "[fatal] SHARED_VLLM_BACKEND must be replicas or dp" >&2; exit 1;; esac
case "${ROLE_REPLICA_MODE}" in auto|0|1) ;; *) echo "[fatal] ROLE_REPLICA_MODE must be auto, 0, or 1" >&2; exit 1;; esac
case "${SINGLE_REPLICA_MODE}" in 0|1) ;; *) echo "[fatal] SINGLE_REPLICA_MODE must be 0 or 1" >&2; exit 1;; esac
for numeric_name in MAX_CONCURRENCY MAX_NEW_TOKENS PROTOCOL_MAX_NEW_TOKENS VLLM_MAX_NUM_SEQS VLLM_MAX_NUM_BATCHED_TOKENS SHARED_VLLM_PORT SHARED_VLLM_REPLICA_PORT_STRIDE SHARED_VLLM_TP SHARED_VLLM_DP SHARED_VLLM_MAX_NUM_SEQS SHARED_VLLM_MAX_NUM_BATCHED_TOKENS SHARED_VLLM_COORD_PORT_BASE SHARED_VLLM_DP_MASTER_PORT SHARED_VLLM_DP_RPC_PORT SHARED_VLLM_INTERNAL_PORT_BASE SHARED_VLLM_INTERNAL_PORT_STRIDE SHARED_VLLM_START_RETRIES SHARED_VLLM_OMP_NUM_THREADS ROLE_REPLICA_PORT_BASE ROLE_REPLICA_ROLE_PORT_STRIDE ROLE_REPLICA_PORT_STRIDE ROLE_REPLICA_PORT_SCAN_BLOCKS ROLE_REPLICA_MAX_NUM_SEQS ROLE_REPLICA_MAX_NUM_BATCHED_TOKENS ROLE_REPLICA_START_RETRIES ROLE_REPLICA_OMP_NUM_THREADS JCA_HTTP_POOL_MAXSIZE PORT_WAIT_TIMEOUT PORT_WAIT_MAX_SLEEP SERVER_HEALTH_INTERVAL; do
  numeric_value="${!numeric_name}"
  [[ "${numeric_value}" =~ ^[1-9][0-9]*$ ]] || { echo "[fatal] ${numeric_name} must be a positive integer" >&2; exit 1; }
done
for boolean_name in ROLE_REPLICA_PORT_AUTO_SHIFT; do
  boolean_value="${!boolean_name}"
  [[ "${boolean_value}" =~ ^[01]$ ]] || { echo "[fatal] ${boolean_name} must be 0 or 1" >&2; exit 1; }
done
[[ "${JCA_ENDPOINT_RETRIES}" =~ ^[0-9]+$ ]] || { echo "[fatal] JCA_ENDPOINT_RETRIES must be a non-negative integer" >&2; exit 1; }
case "${SHARED_VLLM_EXECUTOR_BACKEND}" in uni|mp) ;; *) echo "[fatal] SHARED_VLLM_EXECUTOR_BACKEND must be uni or mp" >&2; exit 1;; esac
for boolean_name in VLLM_ENFORCE_EAGER VLLM_ENABLE_PREFIX_CACHING VLLM_ENABLE_CHUNKED_PREFILL VLLM_DISABLE_LOG_STATS VLLM_ASYNC_SCHEDULING VLLM_DISABLE_UVICORN_ACCESS_LOG; do
  boolean_value="${!boolean_name}"
  [[ "${boolean_value}" =~ ^[01]$ ]] || { echo "[fatal] ${boolean_name} must be 0 or 1" >&2; exit 1; }
done
mkdir -p "${RUN_DIR}" "$(dirname "${OUTPUT}")" "$(dirname "${SCORED_OUTPUT}")" "$(dirname "${RL_OUTPUT}")"
exec > >(tee -a "${RUN_DIR}/run.log") 2>&1

write_config_snapshot() {
  mkdir -p "$(dirname "${CONFIG_PATH}")"
  cat >"${CONFIG_PATH}" <<EOF
RUN_ID=${RUN_ID}
MODE=${MODE}
REQUESTED_MODE=${REQUESTED_MODE}
PYTHON_BIN=${PYTHON_BIN}
VLLM_PYTHON_BIN=${VLLM_PYTHON_BIN}
VLLM_LD_LIBRARY_PATH=${VLLM_LD_LIBRARY_PATH}
BASELINE_STRATEGY=${BASELINE_STRATEGY}
BASELINE_MAD_TEMPERATURE_ROUND0=${BASELINE_MAD_TEMPERATURE_ROUND0}
BASELINE_MAD_TEMPERATURE_DEBATE=${BASELINE_MAD_TEMPERATURE_DEBATE}
BASELINE_AGENT_TEMPERATURE=${BASELINE_AGENT_TEMPERATURE}
BASELINE_META_TEMPERATURE=${BASELINE_META_TEMPERATURE}
BASELINE_SCORE_THRESHOLD=${BASELINE_SCORE_THRESHOLD}
BASELINE_GPTSWARM_TEMPERATURE=${BASELINE_GPTSWARM_TEMPERATURE}
BASELINE_AFLOW_TEMPERATURE=${BASELINE_AFLOW_TEMPERATURE}
BASELINE_AFLOW_WORKFLOW_FILE=${BASELINE_AFLOW_WORKFLOW_FILE}
BASELINE_ROUNDS=${BASELINE_ROUNDS}
BASELINE_ITERATIONS=${BASELINE_ITERATIONS}
DATA_PATH=${DATA_PATH}
OUTPUT=${OUTPUT}
SCORED_OUTPUT=${SCORED_OUTPUT}
RL_OUTPUT=${RL_OUTPUT}
ROLLOUT_IS_TEACHER=${ROLLOUT_IS_TEACHER}
ROLLOUT_SOURCE_POLICY=${ROLLOUT_SOURCE_POLICY}
ROLLOUT_STAGE=${ROLLOUT_STAGE}
SCORE_OUTPUTS=${SCORE_OUTPUTS}
MODEL_A1=${MODEL_A1}
MODEL_A2=${MODEL_A2}
MODEL_A3=${MODEL_A3}
ROLLOUT_MODEL_A1=${ROLLOUT_MODEL_A1}
ROLLOUT_MODEL_A2=${ROLLOUT_MODEL_A2}
ROLLOUT_MODEL_A3=${ROLLOUT_MODEL_A3}
SERVE_MODEL_A1=${SERVE_MODEL_A1}
SERVE_MODEL_A2=${SERVE_MODEL_A2}
SERVE_MODEL_A3=${SERVE_MODEL_A3}
ADAPTER_A1=${ADAPTER_A1}
ADAPTER_A2=${ADAPTER_A2}
ADAPTER_A3=${ADAPTER_A3}
SINGLE_MODEL_PATH=${SINGLE_MODEL_PATH}
SINGLE_MODEL_NAME=${SINGLE_MODEL_NAME}
SINGLE_MODEL_PORT=${SINGLE_MODEL_PORT}
SINGLE_MODEL_GPUS=${SINGLE_MODEL_GPUS}
SINGLE_MODEL_TP=${SINGLE_MODEL_TP}
SINGLE_REPLICA_MODE=${SINGLE_REPLICA_MODE}
START=${START}
LIMIT=${LIMIT}
NUM_ROLLOUTS=${NUM_ROLLOUTS}
T_MAX=${T_MAX}
START_AGENT=${START_AGENT}
START_AGENT_SEED=${START_AGENT_SEED}
SEED=${SEED}
MIN_AGENTS_BEFORE_STOP=${MIN_AGENTS_BEFORE_STOP}
MIN_HANDOFFS_BEFORE_STOP=${MIN_HANDOFFS_BEFORE_STOP}
TEMPERATURE=${TEMPERATURE}
TOP_P=${TOP_P}
MAX_CONCURRENCY=${MAX_CONCURRENCY}
AUTO_TUNE_CONCURRENCY=${AUTO_TUNE_CONCURRENCY}
JCA_HTTP_KEEPALIVE=${JCA_HTTP_KEEPALIVE}
JCA_HTTP_POOL_MAXSIZE=${JCA_HTTP_POOL_MAXSIZE}
JCA_ENDPOINT_RETRIES=${JCA_ENDPOINT_RETRIES}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS}
PROTOCOL_MAX_NEW_TOKENS=${PROTOCOL_MAX_NEW_TOKENS}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}
MAX_MODEL_LEN=${MAX_MODEL_LEN}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS}
VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS}
VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER}
VLLM_ENABLE_PREFIX_CACHING=${VLLM_ENABLE_PREFIX_CACHING}
VLLM_ENABLE_CHUNKED_PREFILL=${VLLM_ENABLE_CHUNKED_PREFILL}
VLLM_PERFORMANCE_MODE=${VLLM_PERFORMANCE_MODE}
VLLM_GENERATION_CONFIG=${VLLM_GENERATION_CONFIG}
VLLM_DISABLE_LOG_STATS=${VLLM_DISABLE_LOG_STATS}
VLLM_ASYNC_SCHEDULING=${VLLM_ASYNC_SCHEDULING}
VLLM_DISABLE_UVICORN_ACCESS_LOG=${VLLM_DISABLE_UVICORN_ACCESS_LOG}
ROLE_REPLICA_MODE=${ROLE_REPLICA_MODE}
ROLE_REPLICA_PORT_BASE=${ROLE_REPLICA_PORT_BASE}
ROLE_REPLICA_ROLE_PORT_STRIDE=${ROLE_REPLICA_ROLE_PORT_STRIDE}
ROLE_REPLICA_PORT_STRIDE=${ROLE_REPLICA_PORT_STRIDE}
ROLE_REPLICA_PORT_AUTO_SHIFT=${ROLE_REPLICA_PORT_AUTO_SHIFT}
ROLE_REPLICA_PORT_SCAN_BLOCKS=${ROLE_REPLICA_PORT_SCAN_BLOCKS}
ROLE_REPLICA_MAX_NUM_SEQS=${ROLE_REPLICA_MAX_NUM_SEQS}
ROLE_REPLICA_MAX_NUM_BATCHED_TOKENS=${ROLE_REPLICA_MAX_NUM_BATCHED_TOKENS}
ROLE_REPLICA_START_RETRIES=${ROLE_REPLICA_START_RETRIES}
ROLE_REPLICA_OMP_NUM_THREADS=${ROLE_REPLICA_OMP_NUM_THREADS}
PORT_WAIT_TIMEOUT=${PORT_WAIT_TIMEOUT}
PORT_WAIT_MAX_SLEEP=${PORT_WAIT_MAX_SLEEP}
SERVER_HEALTH_INTERVAL=${SERVER_HEALTH_INTERVAL}
SHARED_VLLM=${SHARED_VLLM}
SHARED_VLLM_BACKEND=${SHARED_VLLM_BACKEND}
SHARED_VLLM_PORT=${SHARED_VLLM_PORT}
SHARED_VLLM_REPLICA_PORT_STRIDE=${SHARED_VLLM_REPLICA_PORT_STRIDE}
SHARED_VLLM_GPUS=${SHARED_VLLM_GPUS}
SHARED_VLLM_TP=${SHARED_VLLM_TP}
SHARED_VLLM_DP=${SHARED_VLLM_DP}
SHARED_VLLM_MAX_NUM_SEQS=${SHARED_VLLM_MAX_NUM_SEQS}
SHARED_VLLM_MAX_NUM_BATCHED_TOKENS=${SHARED_VLLM_MAX_NUM_BATCHED_TOKENS}
SHARED_VLLM_MODEL_NAME=${SHARED_VLLM_MODEL_NAME}
SHARED_VLLM_EXECUTOR_BACKEND=${SHARED_VLLM_EXECUTOR_BACKEND}
SHARED_VLLM_COORD_PORT_BASE=${SHARED_VLLM_COORD_PORT_BASE}
SHARED_VLLM_DP_MASTER_PORT=${SHARED_VLLM_DP_MASTER_PORT}
SHARED_VLLM_DP_RPC_PORT=${SHARED_VLLM_DP_RPC_PORT}
SHARED_VLLM_INTERNAL_PORT_BASE=${SHARED_VLLM_INTERNAL_PORT_BASE}
SHARED_VLLM_INTERNAL_PORT_STRIDE=${SHARED_VLLM_INTERNAL_PORT_STRIDE}
SHARED_VLLM_START_RETRIES=${SHARED_VLLM_START_RETRIES}
SHARED_VLLM_OMP_NUM_THREADS=${SHARED_VLLM_OMP_NUM_THREADS}
ROUTING=${ROUTING}
FORCE_HANDOFF_UNTIL_FINAL=${FORCE_HANDOFF_UNTIL_FINAL}
EOF
}
write_config_snapshot

PIDS=()
SHARED_SERVER_PIDS=()
ROLE_REPLICA_PIDS_A1=()
ROLE_REPLICA_PIDS_A2=()
ROLE_REPLICA_PIDS_A3=()
ROLE_REPLICA_PIDS=()
ROLE_REPLICA_ENDPOINTS_A1=""
ROLE_REPLICA_ENDPOINTS_A2=""
ROLE_REPLICA_ENDPOINTS_A3=""
ROLE_REPLICA_PORT_BASE_RUNTIME="${ROLE_REPLICA_PORT_BASE}"
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
cleanup() {
  local code=$?
  trap - EXIT INT TERM
  if [[ "${KEEP_SERVERS}" != 1 ]]; then
    stop_process_groups_parallel "${PIDS[@]}" || true
  fi
  exit "${code}"
}
trap cleanup EXIT INT TERM

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

wait_ready() {
  local url="$1" pid="$2"; local timeout="${3:-600}" expected="${4:-}"
  "${PYTHON_BIN}" - "${url}" "${timeout}" "${pid}" "${expected}" <<'PY'
import json, os, sys, time, urllib.request
url, timeout, pid, expected = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
deadline = time.time() + timeout; last = None
while time.time() < deadline:
    try:
        os.kill(pid, 0)
    except OSError:
        print(f"server exited before ready (pid={pid})", file=sys.stderr); raise SystemExit(1)
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = json.loads(response.read().decode())
        names = {str(item.get("id")) for item in payload.get("data") or []}
        if names and (not expected or expected in names):
            print("ready", url); raise SystemExit(0)
    except Exception as exc:
        last = exc
    time.sleep(3)
print("server timeout", url, last, file=sys.stderr); raise SystemExit(1)
PY
}

wait_ready_parallel() {
  local timeout="$1"
  shift
  local -a wait_pids=()
  local url pid expected
  while (( $# > 0 )); do
    [[ $# -ge 3 ]] || { echo "[fatal] wait_ready_parallel requires URL/PID/model triples" >&2; return 1; }
    url="$1"; pid="$2"; expected="$3"; shift 3
    wait_ready "${url}" "${pid}" "${timeout}" "${expected}" &
    wait_pids+=("$!")
  done
  local status=0 wait_pid
  for wait_pid in "${wait_pids[@]}"; do
    if ! wait "${wait_pid}"; then
      status=1
    fi
  done
  return "${status}"
}

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
    print("unavailable port(s): " + ", ".join(busy), file=sys.stderr)
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

check_port_available() {
  check_ports_available "$1"
}

count_gpus() { awk -F',' '{print NF}' <<<"$1"; }

role_index() {
  case "$1" in
    A1) echo 0 ;;
    A2) echo 1 ;;
    A3) echo 2 ;;
    *) return 1 ;;
  esac
}

role_replica_base_port() {
  local index
  index="$(role_index "$1")" || return 1
  echo $((ROLE_REPLICA_PORT_BASE_RUNTIME + index * ROLE_REPLICA_ROLE_PORT_STRIDE))
}

select_role_replica_port_block() {
  if [[ "${ROLE_REPLICA_PORT_AUTO_SHIFT}" != 1 ]]; then
    ROLE_REPLICA_PORT_BASE_RUNTIME="${ROLE_REPLICA_PORT_BASE}"
    return 0
  fi
  local block candidate role rank base_port role_gpus_var
  local -a role_ports=() role_gpu_list=()
  for ((block = 0; block < ROLE_REPLICA_PORT_SCAN_BLOCKS; block++)); do
    candidate=$((ROLE_REPLICA_PORT_BASE + block * ROLE_REPLICA_ROLE_PORT_STRIDE))
    role_ports=()
    for role in A1 A2 A3; do
      base_port=$((candidate + $(role_index "${role}") * ROLE_REPLICA_ROLE_PORT_STRIDE))
      role_gpus_var="${role}_GPUS"
      IFS=',' read -r -a role_gpu_list <<<"${!role_gpus_var}"
      for rank in "${!role_gpu_list[@]}"; do
        role_ports+=("$((base_port + rank * ROLE_REPLICA_PORT_STRIDE))")
      done
    done
    if check_ports_available "${role_ports[@]}" >/dev/null 2>&1; then
      if [[ "${ROLE_REPLICA_PORT_BASE_RUNTIME}" != "${candidate}" ]]; then
        echo "[server] role replica port block ${ROLE_REPLICA_PORT_BASE} unavailable; using ${candidate}"
      fi
      ROLE_REPLICA_PORT_BASE_RUNTIME="${candidate}"
      return 0
    fi
  done
  echo "[server] no free role-replica port block found after ${ROLE_REPLICA_PORT_SCAN_BLOCKS} candidates" >&2
  return 1
}

validate_role_gpu_layout() {
  declare -A seen_gpu=()
  local agent gpus_var tp_var gpus tp gpu
  local -a gpu_list=()
  for agent in A1 A2 A3; do
    gpus_var="${agent}_GPUS"; tp_var="${agent}_TP"
    gpus="${!gpus_var}"; tp="${!tp_var}"
    [[ "$(count_gpus "${gpus}")" == "${tp}" ]] || {
      echo "[role-replica] ${agent}: GPU count/TP mismatch (${gpus} vs TP=${tp})" >&2
      return 1
    }
    IFS=',' read -r -a gpu_list <<<"${gpus}"
    [[ "${#gpu_list[@]}" -gt 0 ]] || { echo "[role-replica] ${agent}: empty GPU list" >&2; return 1; }
    for gpu in "${gpu_list[@]}"; do
      gpu="${gpu//[[:space:]]/}"
      [[ "${gpu}" =~ ^[0-9]+$ ]] || { echo "[role-replica] invalid GPU id: ${gpu}" >&2; return 1; }
      [[ -z "${seen_gpu[${gpu}]:-}" ]] || { echo "[role-replica] GPU ${gpu} is assigned to multiple roles" >&2; return 1; }
      seen_gpu[${gpu}]=1
    done
  done
}

role_replica_possible() {
  validate_role_gpu_layout
}

append_vllm_tuning_args() {
  local -n command_ref="$1"
  command_ref+=(--max-num-seqs "${VLLM_MAX_NUM_SEQS}" --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}")
  if [[ "${VLLM_ENABLE_PREFIX_CACHING}" == 1 ]]; then command_ref+=(--enable-prefix-caching); fi
  if [[ "${VLLM_ENABLE_CHUNKED_PREFILL}" == 1 ]]; then command_ref+=(--enable-chunked-prefill); fi
  if [[ "${VLLM_ENFORCE_EAGER}" == 1 ]]; then command_ref+=(--enforce-eager); fi
  if [[ -n "${VLLM_PERFORMANCE_MODE}" ]]; then command_ref+=(--performance-mode "${VLLM_PERFORMANCE_MODE}"); fi
  if [[ "${VLLM_GENERATION_CONFIG}" != auto ]]; then command_ref+=(--generation-config "${VLLM_GENERATION_CONFIG}"); fi
  if [[ "${VLLM_DISABLE_LOG_STATS}" == 1 ]]; then command_ref+=(--disable-log-stats); fi
  if [[ "${VLLM_ASYNC_SCHEDULING}" == 1 ]]; then command_ref+=(--async-scheduling); fi
  if [[ "${VLLM_DISABLE_UVICORN_ACCESS_LOG}" == 1 ]]; then command_ref+=(--disable-uvicorn-access-log); fi
}

start_server() {
  local agent="$1" model="$2" adapter="$3" port="$4" gpus="$5" tp="$6" served_name_override="${7:-}"
  wait_ports_available "${PORT_WAIT_TIMEOUT}" "${port}" || { echo "[fatal] ${agent} API port is unavailable: ${port}" >&2; exit 1; }
  local served_name="${served_name_override:-${agent}}"
  if [[ "${USE_ADAPTERS}" == 1 && -n "${adapter}" && -z "${served_name_override}" ]]; then
    served_name="${agent}_base"
  fi
  [[ "$(count_gpus "${gpus}")" == "${tp}" ]] || { echo "[fatal] ${agent}: GPU count/TP mismatch" >&2; exit 1; }
  local cmd=("${VLLM_PYTHON_BIN}" -m vllm.entrypoints.openai.api_server
    --host "${HOST}" --port "${port}" --model "${model}" --served-model-name "${served_name}"
    --tensor-parallel-size "${tp}" --dtype bfloat16
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" --max-model-len "${MAX_MODEL_LEN}"
    --trust-remote-code)
  append_vllm_tuning_args cmd
  if [[ "${USE_ADAPTERS}" == 1 && -n "${adapter}" ]]; then
    cmd+=(--enable-lora --max-lora-rank 64 --max-loras 1 --max-cpu-loras 1 --lora-modules "${agent}=${adapter}")
  fi
  if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then read -r -a extra <<<"${VLLM_EXTRA_ARGS}"; cmd+=("${extra[@]}"); fi
  echo "[server] ${agent} model=${model} adapter=${adapter:-none} GPUs=${gpus} port=${port}"
  setsid env CUDA_VISIBLE_DEVICES="${gpus}" \
    LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "${cmd[@]}" >"${RUN_DIR}/${agent}_server.log" 2>&1 &
  PIDS+=("$!")
}

start_role_replicas() {
  local agent model adapter gpus base_port
  local model_var adapter_var gpus_var
  local -a gpu_list=()
  validate_role_gpu_layout || return 1
  for agent in A1 A2 A3; do
    model_var="SERVE_MODEL_${agent}"; adapter_var="ADAPTER_${agent}"; gpus_var="${agent}_GPUS"
    model="${!model_var}"; adapter="${!adapter_var}"; gpus="${!gpus_var}"
    [[ -d "${model}" ]] || { echo "[fatal] missing model ${model}" >&2; return 1; }
    if [[ "${USE_ADAPTERS}" == 1 ]]; then
      [[ -d "${adapter}" ]] || { echo "[fatal] missing adapter ${adapter}" >&2; return 1; }
    fi
    base_port="$(role_replica_base_port "${agent}")"
    IFS=',' read -r -a gpu_list <<<"${gpus}"
    local rank gpu port served_name pid
    local -n agent_pids="ROLE_REPLICA_PIDS_${agent}"
    local -n agent_endpoints="ROLE_REPLICA_ENDPOINTS_${agent}"
    agent_pids=()
    agent_endpoints=""
    for rank in "${!gpu_list[@]}"; do
      gpu="${gpu_list[rank]}"; gpu="${gpu//[[:space:]]/}"
      port=$((base_port + rank * ROLE_REPLICA_PORT_STRIDE))
      served_name="${agent}"
      if [[ "${USE_ADAPTERS}" == 1 && -n "${adapter}" ]]; then
        served_name="${agent}_base"
      fi
      local -a cmd=("${VLLM_PYTHON_BIN}" -m vllm.entrypoints.openai.api_server
        --host "${HOST}" --port "${port}" --model "${model}"
        --served-model-name "${served_name}" --tensor-parallel-size 1 --dtype bfloat16
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" --max-model-len "${MAX_MODEL_LEN}"
        --max-num-seqs "${ROLE_REPLICA_MAX_NUM_SEQS}"
        --max-num-batched-tokens "${ROLE_REPLICA_MAX_NUM_BATCHED_TOKENS}" --trust-remote-code)
      if [[ "${VLLM_ENABLE_PREFIX_CACHING}" == 1 ]]; then cmd+=(--enable-prefix-caching); fi
      if [[ "${VLLM_ENABLE_CHUNKED_PREFILL}" == 1 ]]; then cmd+=(--enable-chunked-prefill); fi
      if [[ "${VLLM_ENFORCE_EAGER}" == 1 ]]; then cmd+=(--enforce-eager); fi
      if [[ -n "${VLLM_PERFORMANCE_MODE}" ]]; then cmd+=(--performance-mode "${VLLM_PERFORMANCE_MODE}"); fi
      if [[ "${VLLM_GENERATION_CONFIG}" != auto ]]; then cmd+=(--generation-config "${VLLM_GENERATION_CONFIG}"); fi
      if [[ "${VLLM_DISABLE_LOG_STATS}" == 1 ]]; then cmd+=(--disable-log-stats); fi
      if [[ "${VLLM_ASYNC_SCHEDULING}" == 1 ]]; then cmd+=(--async-scheduling); fi
      if [[ "${VLLM_DISABLE_UVICORN_ACCESS_LOG}" == 1 ]]; then cmd+=(--disable-uvicorn-access-log); fi
      if [[ "${USE_ADAPTERS}" == 1 && -n "${adapter}" ]]; then
        cmd+=(--enable-lora --max-lora-rank 64 --max-loras 1 --max-cpu-loras 1 --lora-modules "${agent}=${adapter}")
      fi
      if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then read -r -a extra <<<"${VLLM_EXTRA_ARGS}"; cmd+=("${extra[@]}"); fi
      echo "[server] ${agent} replica=${rank} model=${model} adapter=${adapter:-none} GPU=${gpu} port=${port}"
      setsid env -u VLLM_MAX_NUM_SEQS -u VLLM_MAX_NUM_BATCHED_TOKENS \
        -u VLLM_ENFORCE_EAGER -u VLLM_ENABLE_PREFIX_CACHING \
        -u VLLM_ENABLE_CHUNKED_PREFILL -u VLLM_PERFORMANCE_MODE \
        -u VLLM_GENERATION_CONFIG -u VLLM_DISABLE_LOG_STATS \
        -u VLLM_ASYNC_SCHEDULING -u VLLM_DISABLE_UVICORN_ACCESS_LOG \
        -u VLLM_PORT -u VLLM_DP_MASTER_IP -u VLLM_DP_MASTER_PORT \
        CUDA_VISIBLE_DEVICES="${gpu}" VLLM_ENABLE_V1_MULTIPROCESSING=0 \
        OMP_NUM_THREADS="${ROLE_REPLICA_OMP_NUM_THREADS}" \
        LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
        "${cmd[@]}" >"${RUN_DIR}/${agent}_replica_${rank}_server.log" 2>&1 &
      pid="$!"
      agent_pids+=("${pid}")
      ROLE_REPLICA_PIDS+=("${pid}")
      PIDS+=("${pid}")
      [[ -z "${agent_endpoints}" ]] || agent_endpoints+=","
      agent_endpoints+="http://${HOST}:${port}/v1"
    done
  done
}

wait_role_replicas() {
  local agent pid_index pid endpoint model_name
  local -a wait_specs=()
  for agent in A1 A2 A3; do
    model_name="${agent}"
    local -n agent_pids="ROLE_REPLICA_PIDS_${agent}"
    local -n agent_endpoints="ROLE_REPLICA_ENDPOINTS_${agent}"
    IFS=',' read -r -a endpoint_list <<<"${agent_endpoints}"
    for pid_index in "${!agent_pids[@]}"; do
      pid="${agent_pids[pid_index]}"; endpoint="${endpoint_list[pid_index]}"
      wait_specs+=("${endpoint}/models" "${pid}" "${model_name}")
    done
  done
  if ! wait_ready_parallel 900 "${wait_specs[@]}"; then
    return 1
  fi
}

stop_role_replicas() {
  stop_process_groups_parallel "${ROLE_REPLICA_PIDS[@]}" || true
  ROLE_REPLICA_PIDS=()
  ROLE_REPLICA_PIDS_A1=(); ROLE_REPLICA_PIDS_A2=(); ROLE_REPLICA_PIDS_A3=()
  ROLE_REPLICA_ENDPOINTS_A1=""; ROLE_REPLICA_ENDPOINTS_A2=""; ROLE_REPLICA_ENDPOINTS_A3=""
  PIDS=()
}

start_role_replicas_with_retry() {
  local attempts=$((ROLE_REPLICA_START_RETRIES + 1))
  local attempt
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    ROLE_REPLICA_PIDS=()
    if ! select_role_replica_port_block; then
      echo "[server] role replica port allocation failed before attempt ${attempt}/${attempts}" >&2
      if ((attempt < attempts)); then sleep 1; fi
      continue
    fi
    local -a role_ports=()
    local role rank base_port role_gpus_var
    local -a role_gpu_list=()
    for role in A1 A2 A3; do
      base_port="$(role_replica_base_port "${role}")"
      role_gpus_var="${role}_GPUS"
      IFS=',' read -r -a role_gpu_list <<<"${!role_gpus_var}"
      for rank in "${!role_gpu_list[@]}"; do
        role_ports+=("$((base_port + rank * ROLE_REPLICA_PORT_STRIDE))")
      done
    done
    if ! wait_ports_available "${PORT_WAIT_TIMEOUT}" "${role_ports[@]}"; then
      echo "[server] role replica ports remain busy before attempt ${attempt}/${attempts}" >&2
      if ((attempt < attempts)); then sleep 1; fi
      continue
    fi
    if start_role_replicas && wait_role_replicas; then
      return 0
    fi
    echo "[server] role replica startup attempt ${attempt}/${attempts} failed; recycling process groups" >&2
    stop_role_replicas
    if ((attempt < attempts)); then
      wait_ports_available "${PORT_WAIT_TIMEOUT}" "${role_ports[@]}" || true
      sleep 1
    fi
  done
  return 1
}

start_shared_server() {
  local model="${SERVE_MODEL_A1}"
  wait_ports_available "${PORT_WAIT_TIMEOUT}" "${SHARED_VLLM_PORT}" || { echo "[fatal] shared API port is unavailable: ${SHARED_VLLM_PORT}" >&2; exit 1; }
  [[ "$(count_gpus "${SHARED_VLLM_GPUS}")" == "$((SHARED_VLLM_TP * SHARED_VLLM_DP))" ]] || {
    echo "[fatal] shared GPU count must equal SHARED_VLLM_TP*SHARED_VLLM_DP" >&2; exit 1;
  }
  local cmd=("${VLLM_PYTHON_BIN}" "${ROOT}/06_evaluation/serve_vllm_deterministic.py" cli serve "${model}"
    --host "${HOST}" --port "${SHARED_VLLM_PORT}" --served-model-name "${SHARED_VLLM_MODEL_NAME}"
    --tensor-parallel-size "${SHARED_VLLM_TP}" --data-parallel-size "${SHARED_VLLM_DP}"
    --distributed-executor-backend "${SHARED_VLLM_EXECUTOR_BACKEND}"
    --data-parallel-rpc-port "${SHARED_VLLM_DP_RPC_PORT}"
    --dtype bfloat16 --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-seqs "${SHARED_VLLM_MAX_NUM_SEQS}"
    --max-num-batched-tokens "${SHARED_VLLM_MAX_NUM_BATCHED_TOKENS}"
    --trust-remote-code)
  if [[ "${VLLM_ENABLE_PREFIX_CACHING}" == 1 ]]; then cmd+=(--enable-prefix-caching); fi
  if [[ "${VLLM_ENABLE_CHUNKED_PREFILL}" == 1 ]]; then cmd+=(--enable-chunked-prefill); fi
  if [[ "${VLLM_ENFORCE_EAGER}" == 1 ]]; then cmd+=(--enforce-eager); fi
  if [[ -n "${VLLM_PERFORMANCE_MODE}" ]]; then cmd+=(--performance-mode "${VLLM_PERFORMANCE_MODE}"); fi
  if [[ "${VLLM_GENERATION_CONFIG}" != auto ]]; then cmd+=(--generation-config "${VLLM_GENERATION_CONFIG}"); fi
  if [[ "${VLLM_DISABLE_LOG_STATS}" == 1 ]]; then cmd+=(--disable-log-stats); fi
  if [[ "${VLLM_ASYNC_SCHEDULING}" == 1 ]]; then cmd+=(--async-scheduling); fi
  if [[ "${VLLM_DISABLE_UVICORN_ACCESS_LOG}" == 1 ]]; then cmd+=(--disable-uvicorn-access-log); fi
  if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then read -r -a extra <<<"${VLLM_EXTRA_ARGS}"; cmd+=("${extra[@]}"); fi
  echo "[server] shared model=${model} GPUs=${SHARED_VLLM_GPUS} TP=${SHARED_VLLM_TP} DP=${SHARED_VLLM_DP} port=${SHARED_VLLM_PORT}"
  setsid env -u VLLM_MAX_NUM_SEQS -u VLLM_MAX_NUM_BATCHED_TOKENS \
    -u VLLM_ENFORCE_EAGER -u VLLM_ENABLE_PREFIX_CACHING \
    -u VLLM_ENABLE_CHUNKED_PREFILL -u VLLM_PERFORMANCE_MODE \
    -u VLLM_GENERATION_CONFIG -u VLLM_DISABLE_LOG_STATS \
    -u VLLM_ASYNC_SCHEDULING -u VLLM_DISABLE_UVICORN_ACCESS_LOG \
    CUDA_VISIBLE_DEVICES="${SHARED_VLLM_GPUS}" \
    VLLM_PORT="${SHARED_VLLM_COORD_PORT_BASE}" \
    VLLM_DP_MASTER_IP="${HOST}" VLLM_DP_MASTER_PORT="${SHARED_VLLM_DP_MASTER_PORT}" \
    VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    CONIFER_DP_PORT_BASE="${SHARED_VLLM_INTERNAL_PORT_BASE}" \
    CONIFER_DP_PORT_STRIDE="${SHARED_VLLM_INTERNAL_PORT_STRIDE}" \
    OMP_NUM_THREADS="${SHARED_VLLM_OMP_NUM_THREADS}" \
    LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "${cmd[@]}" >"${RUN_DIR}/shared_server.log" 2>&1 &
  SHARED_SERVER_PID="$!"
  PIDS+=("${SHARED_SERVER_PID}")
}

start_shared_replicas() {
  local model="${SERVE_MODEL_A1}"
  [[ "${SHARED_VLLM_TP}" == 1 ]] || {
    echo "[fatal] SHARED_VLLM_BACKEND=replicas requires SHARED_VLLM_TP=1" >&2; exit 1;
  }
  local -a gpu_list=()
  IFS=',' read -r -a gpu_list <<<"${SHARED_VLLM_GPUS}"
  [[ "${#gpu_list[@]}" -gt 0 ]] || { echo "[fatal] SHARED_VLLM_GPUS is empty" >&2; exit 1; }
  SHARED_SERVER_PIDS=()
  local rank gpu port
  local -a shared_ports=()
  for rank in "${!gpu_list[@]}"; do
    shared_ports+=("$((SHARED_VLLM_PORT + rank * SHARED_VLLM_REPLICA_PORT_STRIDE))")
  done
  wait_ports_available "${PORT_WAIT_TIMEOUT}" "${shared_ports[@]}" || { echo "[fatal] shared replica API port pool is unavailable" >&2; exit 1; }
  for rank in "${!gpu_list[@]}"; do
    gpu="${gpu_list[rank]}"
    gpu="${gpu//[[:space:]]/}"
    [[ -n "${gpu}" ]] || { echo "[fatal] empty GPU entry in SHARED_VLLM_GPUS" >&2; exit 1; }
    port=$((SHARED_VLLM_PORT + rank * SHARED_VLLM_REPLICA_PORT_STRIDE))
    local cmd=("${VLLM_PYTHON_BIN}" -m vllm.entrypoints.openai.api_server
      --host "${HOST}" --port "${port}" --model "${model}"
      --served-model-name "${SHARED_VLLM_MODEL_NAME}" --tensor-parallel-size 1
      --dtype bfloat16 --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
      --max-model-len "${MAX_MODEL_LEN}" --max-num-seqs "${SHARED_VLLM_MAX_NUM_SEQS}"
      --max-num-batched-tokens "${SHARED_VLLM_MAX_NUM_BATCHED_TOKENS}" --trust-remote-code)
    if [[ "${VLLM_ENABLE_PREFIX_CACHING}" == 1 ]]; then cmd+=(--enable-prefix-caching); fi
    if [[ "${VLLM_ENABLE_CHUNKED_PREFILL}" == 1 ]]; then cmd+=(--enable-chunked-prefill); fi
    if [[ "${VLLM_ENFORCE_EAGER}" == 1 ]]; then cmd+=(--enforce-eager); fi
    if [[ -n "${VLLM_PERFORMANCE_MODE}" ]]; then cmd+=(--performance-mode "${VLLM_PERFORMANCE_MODE}"); fi
    if [[ "${VLLM_GENERATION_CONFIG}" != auto ]]; then cmd+=(--generation-config "${VLLM_GENERATION_CONFIG}"); fi
    if [[ "${VLLM_DISABLE_LOG_STATS}" == 1 ]]; then cmd+=(--disable-log-stats); fi
    if [[ "${VLLM_ASYNC_SCHEDULING}" == 1 ]]; then cmd+=(--async-scheduling); fi
    if [[ "${VLLM_DISABLE_UVICORN_ACCESS_LOG}" == 1 ]]; then cmd+=(--disable-uvicorn-access-log); fi
    if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then read -r -a extra <<<"${VLLM_EXTRA_ARGS}"; cmd+=("${extra[@]}"); fi
    echo "[server] shared replica=${rank} model=${model} GPU=${gpu} port=${port}"
    setsid env -u VLLM_MAX_NUM_SEQS -u VLLM_MAX_NUM_BATCHED_TOKENS \
      -u VLLM_ENFORCE_EAGER -u VLLM_ENABLE_PREFIX_CACHING \
      -u VLLM_ENABLE_CHUNKED_PREFILL -u VLLM_PERFORMANCE_MODE \
      -u VLLM_GENERATION_CONFIG -u VLLM_DISABLE_LOG_STATS \
      -u VLLM_ASYNC_SCHEDULING -u VLLM_DISABLE_UVICORN_ACCESS_LOG \
      -u VLLM_PORT -u VLLM_DP_MASTER_IP -u VLLM_DP_MASTER_PORT \
      CUDA_VISIBLE_DEVICES="${gpu}" VLLM_ENABLE_V1_MULTIPROCESSING=0 \
      OMP_NUM_THREADS="${SHARED_VLLM_OMP_NUM_THREADS}" \
      LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
      "${cmd[@]}" >"${RUN_DIR}/shared_replica_${rank}_server.log" 2>&1 &
    SHARED_SERVER_PIDS+=("$!")
    PIDS+=("$!")
  done
}

wait_shared_replicas() {
  local -a gpu_list=()
  IFS=',' read -r -a gpu_list <<<"${SHARED_VLLM_GPUS}"
  local rank port
  local -a wait_specs=()
  for rank in "${!gpu_list[@]}"; do
    port=$((SHARED_VLLM_PORT + rank * SHARED_VLLM_REPLICA_PORT_STRIDE))
    wait_specs+=("http://${HOST}:${port}/v1/models" "${SHARED_SERVER_PIDS[rank]}" "${SHARED_VLLM_MODEL_NAME}")
  done
  if ! wait_ready_parallel 900 "${wait_specs[@]}"; then
    return 1
  fi
  SHARED_BASE_POOL=""
  for rank in "${!gpu_list[@]}"; do
    port=$((SHARED_VLLM_PORT + rank * SHARED_VLLM_REPLICA_PORT_STRIDE))
    [[ -z "${SHARED_BASE_POOL}" ]] || SHARED_BASE_POOL+=","
    SHARED_BASE_POOL+="http://${HOST}:${port}/v1"
  done
}

stop_shared_replicas() {
  stop_process_groups_parallel "${SHARED_SERVER_PIDS[@]}" || true
  SHARED_SERVER_PIDS=()
  PIDS=()
}

start_shared_server_with_retry() {
  local attempts=$((SHARED_VLLM_START_RETRIES + 1))
  local attempt
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    if [[ "${SHARED_VLLM_BACKEND}" == replicas ]]; then
      start_shared_replicas
      if wait_shared_replicas; then
        return 0
      fi
      echo "[server] shared replica startup attempt ${attempt}/${attempts} failed; recycling process group" >&2
      stop_shared_replicas
      if ((attempt < attempts)); then sleep 3; fi
      continue
    fi
    start_shared_server
    if wait_ready "http://${HOST}:${SHARED_VLLM_PORT}/v1/models" "${SHARED_SERVER_PID}" 900 "${SHARED_VLLM_MODEL_NAME}"; then
      SHARED_BASE_POOL="http://${HOST}:${SHARED_VLLM_PORT}/v1"
      return 0
    fi
    echo "[server] shared startup attempt ${attempt}/${attempts} failed; recycling process group ${SHARED_SERVER_PID}" >&2
    stop_process_group "${SHARED_SERVER_PID}"
    unset "PIDS[$((${#PIDS[@]} - 1))]"
    if ((attempt < attempts)); then sleep 3; fi
  done
  echo "[fatal] shared vLLM failed to become ready after ${attempts} attempts" >&2
  return 1
}

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "[dry-run] MODE=${MODE} DATA=${DATA_PATH} OUTPUT=${OUTPUT}"
  echo "[dry-run] max_concurrency=${MAX_CONCURRENCY} shared_vllm=${SHARED_VLLM} backend=${SHARED_VLLM_EXECUTOR_BACKEND} eager=${VLLM_ENFORCE_EAGER} max_seqs=${VLLM_MAX_NUM_SEQS} max_batch_tokens=${VLLM_MAX_NUM_BATCHED_TOKENS}"
  echo "[dry-run] runner: ${PYTHON_BIN} ${ROOT}/03_rollout/run_conifer_mas.py ..."
  echo "[dry-run] scorer: ${PYTHON_BIN} ${ROOT}/04_judge/score_conifer_rollouts.py ..."
  exit 0
fi

if [[ "${MOCK}" == 1 ]]; then
  BASE_A1=""; BASE_A2=""; BASE_A3=""
  if [[ "${SINGLE_MODE}" == 1 ]]; then
    API_MODEL_A1="${SINGLE_MODEL_NAME}"; API_MODEL_A2="${SINGLE_MODEL_NAME}"; API_MODEL_A3="${SINGLE_MODEL_NAME}"
  else
    API_MODEL_A1=A1; API_MODEL_A2=A2; API_MODEL_A3=A3
  fi
elif [[ "${SINGLE_MODE}" == 1 ]]; then
  single_model="${MODEL_A3}"
  single_name="A3"
  single_port="${A3_PORT}"
  single_gpus="0,1,2,3,4,5,6,7"
  single_tp=8
  if [[ "${MODE}" == single_14b ]]; then
    single_model="${SINGLE_MODEL_PATH}"
    single_name="${SINGLE_MODEL_NAME}"
    single_port="${SINGLE_MODEL_PORT}"
    single_gpus="${SINGLE_MODEL_GPUS}"
    single_tp="${SINGLE_MODEL_TP}"
  fi
  [[ -d "${single_model}" ]] || { echo "[fatal] missing ${single_model}" >&2; exit 1; }
  if [[ "${MODE}" == single_a3 ]]; then
    [[ -z "${ADAPTER_A3}" || -d "${ADAPTER_A3}" ]] || { echo "[fatal] missing ${ADAPTER_A3}" >&2; exit 1; }
  fi
  single_replica_active=0
  if [[ "${SINGLE_REPLICA_MODE}" == 1 ]]; then
    [[ -z "${ADAPTER_A3}" ]] || { echo "[fatal] SINGLE_REPLICA_MODE does not support an adapter" >&2; exit 1; }
    SERVE_MODEL_A1="${single_model}"
    SHARED_VLLM_MODEL_NAME="${single_name}"
    SHARED_VLLM_PORT="${single_port}"
    SHARED_VLLM_GPUS="${single_gpus}"
    SHARED_VLLM_TP=1
    SHARED_VLLM_BACKEND=replicas
    start_shared_server_with_retry
    BASE_A1="${SHARED_BASE_POOL}"; BASE_A2="${BASE_A1}"; BASE_A3="${BASE_A1}"
    single_replica_active=1
    echo "[server] single-model replica pool active: ${BASE_A1}"
  else
    start_server A3 "${single_model}" "${ADAPTER_A3}" "${single_port}" "${single_gpus}" "${single_tp}" "${single_name}"
    wait_ready "http://${HOST}:${single_port}/v1/models" "${PIDS[0]}" 600 "${single_name}"
    BASE_A1="http://${HOST}:${single_port}/v1"; BASE_A2="${BASE_A1}"; BASE_A3="${BASE_A1}"
  fi
  API_MODEL_A1="${single_name}"; API_MODEL_A2="${single_name}"; API_MODEL_A3="${single_name}"
else
  use_shared=0
  use_role_replicas=0
  if [[ "${SHARED_VLLM}" == 1 || ( "${SHARED_VLLM}" == auto && "${USE_ADAPTERS}" == 0 && ( "${MODE}" == teacher_rollout_mas || "${ROLLOUT_IS_TEACHER}" == 1 ) && "${SERVE_MODEL_A1}" == "${SERVE_MODEL_A2}" && "${SERVE_MODEL_A1}" == "${SERVE_MODEL_A3}" && -z "${ADAPTER_A1}" && -z "${ADAPTER_A2}" && -z "${ADAPTER_A3}" ) ]]; then
    use_shared=1
  fi
  if [[ "${use_shared}" == 1 ]]; then
    [[ "${SERVE_MODEL_A1}" == "${SERVE_MODEL_A2}" && "${SERVE_MODEL_A1}" == "${SERVE_MODEL_A3}" ]] || {
      echo "[fatal] SHARED_VLLM requires identical A1/A2/A3 served models" >&2; exit 1;
    }
    [[ -z "${ADAPTER_A1}" && -z "${ADAPTER_A2}" && -z "${ADAPTER_A3}" ]] || {
      echo "[fatal] SHARED_VLLM is incompatible with per-agent adapters" >&2; exit 1;
    }
    [[ -d "${SERVE_MODEL_A1}" ]] || { echo "[fatal] missing model ${SERVE_MODEL_A1}" >&2; exit 1; }
    start_shared_server_with_retry
    BASE_SHARED="${SHARED_BASE_POOL}"
    BASE_A1="${BASE_SHARED}"; BASE_A2="${BASE_SHARED}"; BASE_A3="${BASE_SHARED}"
    API_MODEL_A1="${SHARED_VLLM_MODEL_NAME}"; API_MODEL_A2="${SHARED_VLLM_MODEL_NAME}"; API_MODEL_A3="${SHARED_VLLM_MODEL_NAME}"
    echo "[server] shared endpoint active for A1/A2/A3"
  else
    if [[ "${ROLE_REPLICA_MODE}" == 1 ]]; then
      use_role_replicas=1
    elif [[ "${ROLE_REPLICA_MODE}" == auto ]] && role_replica_possible; then
      use_role_replicas=1
    fi
    if [[ "${use_role_replicas}" == 1 ]]; then
      if ! start_role_replicas_with_retry; then
        if [[ "${ROLE_REPLICA_MODE}" == auto ]]; then
          echo "[server] role-replica startup failed; falling back to tensor-parallel role servers" >&2
          stop_role_replicas
          use_role_replicas=0
        else
          echo "[fatal] role-replica startup failed" >&2
          exit 1
        fi
      fi
    fi
    if [[ "${use_role_replicas}" == 1 ]]; then
      BASE_A1="${ROLE_REPLICA_ENDPOINTS_A1}"; BASE_A2="${ROLE_REPLICA_ENDPOINTS_A2}"; BASE_A3="${ROLE_REPLICA_ENDPOINTS_A3}"
      API_MODEL_A1=A1; API_MODEL_A2=A2; API_MODEL_A3=A3
      echo "[server] role-replica pools active: A1=${BASE_A1} A2=${BASE_A2} A3=${BASE_A3}"
    else
      for agent in A1 A2 A3; do
        model_var="SERVE_MODEL_${agent}"; adapter_var="ADAPTER_${agent}"; port_var="${agent}_PORT"; gpus_var="${agent}_GPUS"; tp_var="${agent}_TP"
        model="${!model_var}"; adapter="${!adapter_var}"; port="${!port_var}"; gpus="${!gpus_var}"; tp="${!tp_var}"
        [[ -d "${model}" ]] || { echo "[fatal] missing model ${model}" >&2; exit 1; }
        if [[ "${USE_ADAPTERS}" == 1 ]]; then [[ -d "${adapter}" ]] || { echo "[fatal] missing adapter ${adapter}" >&2; exit 1; }; fi
        start_server "${agent}" "${model}" "${adapter}" "${port}" "${gpus}" "${tp}"
      done
      wait_ready_parallel 600 \
        "http://${HOST}:${A1_PORT}/v1/models" "${PIDS[0]}" A1 \
        "http://${HOST}:${A2_PORT}/v1/models" "${PIDS[1]}" A2 \
        "http://${HOST}:${A3_PORT}/v1/models" "${PIDS[2]}" A3
      BASE_A1="http://${HOST}:${A1_PORT}/v1"; BASE_A2="http://${HOST}:${A2_PORT}/v1"; BASE_A3="http://${HOST}:${A3_PORT}/v1"
      API_MODEL_A1=A1; API_MODEL_A2=A2; API_MODEL_A3=A3
    fi
  fi
fi

if [[ "${AUTO_TUNE_CONCURRENCY}" == 1 && "${MOCK}" != 1 ]]; then
  if [[ "${SINGLE_MODE}" == 1 ]]; then
    if [[ "${single_replica_active:-0}" == 1 ]]; then
      endpoint_count="$(count_gpus "${SINGLE_MODEL_GPUS}")"
      sequence_limit="${SHARED_VLLM_MAX_NUM_SEQS}"
    else
      endpoint_count=1
      sequence_limit="${VLLM_MAX_NUM_SEQS}"
    fi
  elif [[ "${use_shared:-0}" == 1 ]]; then
    if [[ "${SHARED_VLLM_BACKEND}" == replicas ]]; then
      IFS=',' read -r -a shared_gpu_list <<<"${SHARED_VLLM_GPUS}"
      endpoint_count="${#shared_gpu_list[@]}"
    else
      endpoint_count="${SHARED_VLLM_DP}"
    fi
    sequence_limit="${SHARED_VLLM_MAX_NUM_SEQS}"
  elif [[ "${use_role_replicas:-0}" == 1 ]]; then
    endpoint_count=$(( $(count_gpus "${A1_GPUS}") + $(count_gpus "${A2_GPUS}") + $(count_gpus "${A3_GPUS}") ))
    sequence_limit="${ROLE_REPLICA_MAX_NUM_SEQS}"
  else
    endpoint_count=3
    sequence_limit="${VLLM_MAX_NUM_SEQS}"
  fi
  MAX_CONCURRENCY=$((endpoint_count * sequence_limit))
  if (( JCA_HTTP_POOL_MAXSIZE < sequence_limit )); then
    JCA_HTTP_POOL_MAXSIZE="${sequence_limit}"
    export JCA_HTTP_POOL_MAXSIZE
  fi
  echo "[throughput] auto concurrency=${MAX_CONCURRENCY} (${endpoint_count} endpoint(s) x ${sequence_limit} sequence(s))"
fi

{
  if [[ "${MOCK}" == 1 ]]; then
    echo "ACTIVE_ENDPOINT_MODE=mock"
  elif [[ "${use_shared:-0}" == 1 ]]; then
    echo "ACTIVE_ENDPOINT_MODE=shared_${SHARED_VLLM_BACKEND}"
  elif [[ "${use_role_replicas:-0}" == 1 ]]; then
    echo "ACTIVE_ENDPOINT_MODE=role_replicas"
  else
    echo "ACTIVE_ENDPOINT_MODE=role_tensor_parallel"
  fi
  echo "ACTIVE_ROLE_REPLICA_PORT_BASE=${ROLE_REPLICA_PORT_BASE_RUNTIME}"
  echo "ACTIVE_MAX_CONCURRENCY=${MAX_CONCURRENCY}"
  echo "ACTIVE_ENDPOINT_COUNT=${endpoint_count:-0}"
  echo "ACTIVE_SEQUENCE_LIMIT=${sequence_limit:-0}"
} >>"${CONFIG_PATH}"

runner_args=("${ROOT}/03_rollout/run_conifer_mas.py" --data-path "${DATA_PATH}" --output "${OUTPUT}" \
  --start "${START}" --limit "${LIMIT}" --num-rollouts "${NUM_ROLLOUTS}" --t-max "${T_MAX}" \
  --start-agent "${START_AGENT}" --start-agent-seed "${START_AGENT_SEED}" \
  --seed "${SEED}" \
  --min-agents-before-stop "${MIN_AGENTS_BEFORE_STOP}" \
  --min-handoffs-before-stop "${MIN_HANDOFFS_BEFORE_STOP}" --max-new-tokens "${MAX_NEW_TOKENS}" \
  --protocol-max-new-tokens "${PROTOCOL_MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" --top-p "${TOP_P}" --max-concurrency "${MAX_CONCURRENCY}" \
  --routing "${ROUTING}" --context-turns "${CONTEXT_TURNS}" \
  --json-transport "${JSON_TRANSPORT}" --no-include-reference \
  --api-base-a1 "${BASE_A1}" --api-base-a2 "${BASE_A2}" --api-base-a3 "${BASE_A3}" \
  --model-a1 "${API_MODEL_A1}" --model-a2 "${API_MODEL_A2}" --model-a3 "${API_MODEL_A3}" \
  --model-path-a1 "${SERVE_MODEL_A1}" --model-path-a2 "${SERVE_MODEL_A2}" --model-path-a3 "${SERVE_MODEL_A3}" \
  --adapter-path-a1 "${ADAPTER_A1}" --adapter-path-a2 "${ADAPTER_A2}" --adapter-path-a3 "${ADAPTER_A3}" \
  --student-model-path-a1 "${MODEL_A1}" --student-model-path-a2 "${MODEL_A2}" --student-model-path-a3 "${MODEL_A3}" \
  --source-policy "${ROLLOUT_SOURCE_POLICY}" --rollout-stage "${ROLLOUT_STAGE}" \
  --baseline-strategy "${BASELINE_STRATEGY}" \
  --baseline-rounds "${BASELINE_ROUNDS}" --baseline-iterations "${BASELINE_ITERATIONS}" \
  --baseline-mad-temperature-round0 "${BASELINE_MAD_TEMPERATURE_ROUND0}" \
  --baseline-mad-temperature-debate "${BASELINE_MAD_TEMPERATURE_DEBATE}" \
  --baseline-agent-temperature "${BASELINE_AGENT_TEMPERATURE}" \
  --baseline-meta-temperature "${BASELINE_META_TEMPERATURE}" \
  --baseline-score-threshold "${BASELINE_SCORE_THRESHOLD}" \
  --baseline-gptswarm-temperature "${BASELINE_GPTSWARM_TEMPERATURE}" \
  --baseline-aflow-temperature "${BASELINE_AFLOW_TEMPERATURE}")
if [[ -n "${BASELINE_AFLOW_WORKFLOW_FILE}" ]]; then
  runner_args+=(--baseline-aflow-workflow-file "${BASELINE_AFLOW_WORKFLOW_FILE}")
fi
if [[ "${MOCK}" == 1 ]]; then runner_args+=(--mock); fi
if [[ "${MODE}" == teacher_rollout_mas || "${ROLLOUT_IS_TEACHER}" == 1 ]]; then runner_args+=(--teacher-rollout); fi
if [[ "${SINGLE_MODE}" == 1 ]]; then
  runner_args+=(--start-agent A3 --min-agents-before-stop 1 --min-handoffs-before-stop 0 --t-max 1 --no-enforce-runtime-policy --force-handoff-until-final)
elif [[ "${FORCE_HANDOFF_UNTIL_FINAL}" == 1 ]]; then
  runner_args+=(--force-handoff-until-final)
else
  runner_args+=(--no-force-handoff-until-final)
fi
setsid "${PYTHON_BIN}" "${runner_args[@]}" &
runner_pid="$!"
failed_server_pid=""
while kill -0 "${runner_pid}" 2>/dev/null; do
  if [[ "${MOCK}" != 1 ]]; then
    for server_pid in "${PIDS[@]}"; do
      if ! kill -0 "${server_pid}" 2>/dev/null; then
        failed_server_pid="${server_pid}"
        break
      fi
    done
  fi
  [[ -z "${failed_server_pid}" ]] || break
  sleep "${SERVER_HEALTH_INTERVAL}"
done
if [[ -n "${failed_server_pid}" ]]; then
  echo "[fatal] vLLM server pid ${failed_server_pid} exited during rollout; stopping runner for resumable retry" >&2
  kill -TERM -- "-${runner_pid}" 2>/dev/null || kill -TERM "${runner_pid}" 2>/dev/null || true
  wait "${runner_pid}" 2>/dev/null || true
  exit 1
fi
runner_status=0
wait "${runner_pid}" || runner_status="$?"
if [[ "${runner_status}" != 0 ]]; then
  echo "[fatal] rollout runner exited with status ${runner_status}" >&2
  exit "${runner_status}"
fi

if [[ "${SCORE_OUTPUTS}" == 1 ]]; then
  scorer_args=("${ROOT}/04_judge/score_conifer_rollouts.py" --input "${OUTPUT}" \
    --scored-output "${SCORED_OUTPUT}" --rl-output "${RL_OUTPUT}" --judge-mode "${JUDGE_MODE}" \
    --judge-model "${JUDGE_MODEL}" --alpha "${ALPHA}")
  "${PYTHON_BIN}" "${scorer_args[@]}"
  echo "[ok] evaluation complete: ${SCORED_OUTPUT}"
else
  echo "[ok] rollout complete; scoring deferred: ${OUTPUT}"
fi

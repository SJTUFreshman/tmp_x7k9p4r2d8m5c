#!/usr/bin/env bash
set -euo pipefail

# Reproducible phase controller.  No phase silently overwrites an existing
# artifact; use a new TAG or explicitly select a later MODE to resume.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SCRIPT_DIR}"
PROJECT_ROOT="$(cd "${ROOT}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/qwen35/bin/python}"
MODE="${MODE:-prepare}" # prepare|rollout|score|build_sft|sft|rl_rollout|rl_score|rl|eval|ablations|train_ablations|official|all
TAG="${TAG:-conifer_v1}"
TRAIN_DATA="${TRAIN_DATA:-${ROOT}/01_dataset/processed/train.jsonl}"
DEV_DATA="${DEV_DATA:-${ROOT}/01_dataset/processed/dev.jsonl}"
TEST_DATA="${TEST_DATA:-${ROOT}/01_dataset/processed/test.jsonl}"
TRAIN_LIMIT="${TRAIN_LIMIT:-1000}"
SFT_TRAIN_DATA="${SFT_TRAIN_DATA:-${TRAIN_DATA}}"
SFT_TRAIN_LIMIT="${SFT_TRAIN_LIMIT:-${TRAIN_LIMIT}}"
RL_TRAIN_DATA="${RL_TRAIN_DATA:-${TRAIN_DATA}}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-4}"
JUDGE_MODE="${JUDGE_MODE:-llm}"
JUDGE_BACKEND="${JUDGE_BACKEND:-local_vllm}" # local_vllm|external
SKIP_PREPARE="${SKIP_PREPARE:-0}"
DRY_RUN="${DRY_RUN:-0}"

STUDENT_MODEL_A1="${STUDENT_MODEL_A1:-${MODEL_A1:-/data/wangyuheng/models/Qwen3-1.7B}}"
STUDENT_MODEL_A2="${STUDENT_MODEL_A2:-${MODEL_A2:-/data/wangyuheng/models/Qwen3-4B}}"
STUDENT_MODEL_A3="${STUDENT_MODEL_A3:-${MODEL_A3:-/data/wangyuheng/models/Qwen3-8B}}"
ROLLOUT_MODEL_A1="${ROLLOUT_MODEL_A1:-/data/wangyuheng/models/Qwen3-8B}"
ROLLOUT_MODEL_A2="${ROLLOUT_MODEL_A2:-/data/wangyuheng/models/Qwen3-8B}"
ROLLOUT_MODEL_A3="${ROLLOUT_MODEL_A3:-/data/wangyuheng/models/Qwen3-8B}"
ROLLOUT_MODE="${ROLLOUT_MODE:-teacher_rollout_mas}"
ROLLOUT_SOURCE_POLICY="${ROLLOUT_SOURCE_POLICY:-3xQwen3-8B_balanced_start_teacher_rollout_v2}"
ROLLOUT_START_AGENT="${ROLLOUT_START_AGENT:-balanced}"
ROLLOUT_START_AGENT_SEED="${ROLLOUT_START_AGENT_SEED:-42}"
ROLLOUT_SEED="${ROLLOUT_SEED:-42}"
ROLLOUT_MIN_AGENTS_BEFORE_STOP="${ROLLOUT_MIN_AGENTS_BEFORE_STOP:-3}"
ROLLOUT_MIN_HANDOFFS_BEFORE_STOP="${ROLLOUT_MIN_HANDOFFS_BEFORE_STOP:-2}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.2}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.95}"
ROLLOUT_FORCE_HANDOFF_UNTIL_FINAL="${ROLLOUT_FORCE_HANDOFF_UNTIL_FINAL:-0}"
REQUIRE_TEACHER_ROLLOUT="${REQUIRE_TEACHER_ROLLOUT:-1}"

# RL is deliberately a second sampling boundary.  These trajectories are
# generated only after SFT and are never reused as SFT demonstrations.
RL_ROLLOUT_MODE="${RL_ROLLOUT_MODE:-sft_lora_mas}"
RL_ROLLOUT_SOURCE_POLICY="${RL_ROLLOUT_SOURCE_POLICY:-sft_policy_balanced_rl_rollout_v1}"
RL_ROLLOUT_START_AGENT="${RL_ROLLOUT_START_AGENT:-balanced}"
RL_ROLLOUT_START_AGENT_SEED="${RL_ROLLOUT_START_AGENT_SEED:-1042}"
RL_ROLLOUT_MIN_AGENTS_BEFORE_STOP="${RL_ROLLOUT_MIN_AGENTS_BEFORE_STOP:-${ROLLOUT_MIN_AGENTS_BEFORE_STOP}}"
RL_ROLLOUT_MIN_HANDOFFS_BEFORE_STOP="${RL_ROLLOUT_MIN_HANDOFFS_BEFORE_STOP:-${ROLLOUT_MIN_HANDOFFS_BEFORE_STOP}}"
RL_ROLLOUT_TEMPERATURE="${RL_ROLLOUT_TEMPERATURE:-0.8}"
RL_ROLLOUT_TOP_P="${RL_ROLLOUT_TOP_P:-0.95}"
RL_ROLLOUT_FORCE_HANDOFF_UNTIL_FINAL="${RL_ROLLOUT_FORCE_HANDOFF_UNTIL_FINAL:-0}"
RL_ROLLOUT_NUM_ROLLOUTS="${RL_ROLLOUT_NUM_ROLLOUTS:-${NUM_ROLLOUTS}}"
RL_ROLLOUT_START="${RL_ROLLOUT_START:-0}"
RL_ROLLOUT_LIMIT="${RL_ROLLOUT_LIMIT:-${TRAIN_LIMIT}}"
RL_ROLLOUT_SEED="${RL_ROLLOUT_SEED:-1042}"

# Throughput controls shared by rollout, holdout evaluation, and ablations.
MAX_CONCURRENCY="${MAX_CONCURRENCY:-768}"
EVAL_MAX_CONCURRENCY="${EVAL_MAX_CONCURRENCY:-768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}"
PROTOCOL_MAX_NEW_TOKENS="${PROTOCOL_MAX_NEW_TOKENS:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
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
ROLE_REPLICA_MAX_NUM_SEQS="${ROLE_REPLICA_MAX_NUM_SEQS:-96}"
ROLE_REPLICA_MAX_NUM_BATCHED_TOKENS="${ROLE_REPLICA_MAX_NUM_BATCHED_TOKENS:-98304}"
ROLE_REPLICA_START_RETRIES="${ROLE_REPLICA_START_RETRIES:-4}"
ROLE_REPLICA_OMP_NUM_THREADS="${ROLE_REPLICA_OMP_NUM_THREADS:-4}"
PORT_WAIT_TIMEOUT="${PORT_WAIT_TIMEOUT:-180}"
PORT_WAIT_MAX_SLEEP="${PORT_WAIT_MAX_SLEEP:-8}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-768}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-1536}"
JUDGE_LENGTH_RETRIES="${JUDGE_LENGTH_RETRIES:-1}"
JUDGE_GPU_MEMORY_UTILIZATION="${JUDGE_GPU_MEMORY_UTILIZATION:-0.90}"
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
RL_PER_DEVICE_BATCH="${RL_PER_DEVICE_BATCH:-1}"
RL_GRAD_ACCUM="${RL_GRAD_ACCUM:-16}"
RL_NUM_EPOCHS="${RL_NUM_EPOCHS:-3}"
SFT_NUM_EPOCHS="${SFT_NUM_EPOCHS:-3}"
RL_MAX_SEQ="${RL_MAX_SEQ:-6144}"
RL_SAVE_STEPS="${RL_SAVE_STEPS:-10}"
RL_RESUME_FROM_CHECKPOINT="${RL_RESUME_FROM_CHECKPOINT:-latest}"
RL_ROLLOUT_PHASE_RETRIES="${RL_ROLLOUT_PHASE_RETRIES:-3}"
REQUIRE_SFT_INIT="${REQUIRE_SFT_INIT:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
JCA_HTTP_KEEPALIVE="${JCA_HTTP_KEEPALIVE:-1}"
JCA_HTTP_POOL_MAXSIZE="${JCA_HTTP_POOL_MAXSIZE:-128}"
JCA_ENDPOINT_RETRIES="${JCA_ENDPOINT_RETRIES:-2}"
SERVER_HEALTH_INTERVAL="${SERVER_HEALTH_INTERVAL:-2}"
export JCA_HTTP_KEEPALIVE JCA_HTTP_POOL_MAXSIZE JCA_ENDPOINT_RETRIES
JUDGE_VLLM_EXTRA_ARGS="${JUDGE_VLLM_EXTRA_ARGS:-}"

ROLL_OUT="${ROLL_OUT:-${ROOT}/10_outputs/${TAG}/train_trajectories.jsonl}"
SCORED_OUT="${SCORED_OUT:-${ROOT}/10_outputs/${TAG}/train_scored.jsonl}"
# Keep the historical teacher path stable; RL_OUT is now exclusively the
# independently sampled, SFT-policy rollout consumed by RWR.
TEACHER_RL_OUT="${TEACHER_RL_OUT:-${ROOT}/13_rl_data/${TAG}.jsonl}"
RL_ROLLOUT_DIR="${RL_ROLLOUT_DIR:-${ROOT}/10_outputs/${TAG}/rl_rollout}"
RL_ROLLOUT_OUT="${RL_ROLLOUT_OUT:-${RL_ROLLOUT_DIR}/trajectories.jsonl}"
RL_SCORED_OUT="${RL_SCORED_OUT:-${RL_ROLLOUT_DIR}/scored.jsonl}"
RL_OUT="${RL_OUT:-${ROOT}/13_rl_data/${TAG}_rl_from_sft.jsonl}"
SFT_DATA_DIR="${SFT_DATA_DIR:-${ROOT}/12_sft_data/${TAG}}"
SFT_RUN_DIR="${SFT_RUN_DIR:-${ROOT}/11_runs/sft/${TAG}}"
RL_RUN_DIR="${RL_RUN_DIR:-${ROOT}/11_runs/rl/${TAG}}"
EVAL_DIR="${EVAL_DIR:-${ROOT}/10_outputs/${TAG}/eval}"

echo "[config] SFT data=${SFT_TRAIN_DATA} limit=${SFT_TRAIN_LIMIT} epochs=${SFT_NUM_EPOCHS}; RL data=${RL_TRAIN_DATA} limit=${RL_ROLLOUT_LIMIT} epochs=${RL_NUM_EPOCHS}"

ensure() {
  if [[ "${DRY_RUN}" == 1 ]]; then return 0; fi
  [[ -e "$1" ]] || { echo "[fatal] missing $2: $1" >&2; exit 1; }
}
run() { printf '[run] '; printf '%q ' "$@"; printf '\n'; [[ "${DRY_RUN}" == 1 ]] || "$@"; }

score_complete_for() {
  local scored="$1" rl="$2"
  [[ -s "${scored}" && -s "${rl}" && -s "${scored%.jsonl}_stats.json" ]]
}

sft_data_complete() {
  [[ -s "${SFT_DATA_DIR}/A1.jsonl" && -s "${SFT_DATA_DIR}/A2.jsonl" && -s "${SFT_DATA_DIR}/A3.jsonl" && -s "${SFT_DATA_DIR}/stats.json" ]]
}

rollout_complete_for() {
  local data_path="$1" rollout_path="$2" start="$3" limit="$4" num_rollouts="$5"
  local start_policy="$6" start_seed="$7" expected_teacher="$8" expected_policy="$9" expected_stage="${10:-}"
  [[ -s "${rollout_path}" ]] || return 1
  "${PYTHON_BIN}" - "${data_path}" "${rollout_path}" "${start}" "${limit}" "${num_rollouts}" \
    "${start_policy}" "${start_seed}" "${expected_teacher}" "${expected_policy}" "${expected_stage}" <<'PY'
import json
import sys
from pathlib import Path

train_path, rollout_path = map(Path, sys.argv[1:3])
start = int(sys.argv[3])
limit = int(sys.argv[4])
num_rollouts = int(sys.argv[5])
start_policy = sys.argv[6]
start_seed = int(sys.argv[7])
expected_teacher = sys.argv[8] == "1"
expected_policy = sys.argv[9]
expected_stage = sys.argv[10]
agents = ("A1", "A2", "A3")
with train_path.open(encoding="utf-8") as handle:
    train_rows = [json.loads(line) for line in handle if line.strip()]
selected = train_rows[start : start + limit if limit else None]
expected = {}
for row_index, row in enumerate(selected, start=start):
    problem_id = str(row["problem_id"])
    for rollout_index in range(num_rollouts):
        if start_policy == "balanced":
            start_agent = agents[(row_index + rollout_index + start_seed) % len(agents)]
        elif start_policy in agents:
            start_agent = start_policy
        else:
            raise SystemExit(2)
        expected[(problem_id, rollout_index)] = start_agent
seen = {}
try:
    with rollout_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            key = (str(item.get("problem_id")), int(item.get("rollout_idx", 0)))
            if key not in expected or key in seen:
                raise ValueError("unexpected or duplicate rollout key")
            actual_start = str(item.get("start_agent") or (item.get("trajectory") or {}).get("start_agent") or "")
            if actual_start != expected[key]:
                raise ValueError("rollout start-agent policy mismatch")
            sampling = item.get("sampling") if isinstance(item.get("sampling"), dict) else {}
            actual_teacher = bool(item.get("teacher_rollout", sampling.get("teacher_rollout", False)))
            if actual_teacher != expected_teacher:
                raise ValueError("rollout teacher-policy mismatch")
            actual_policy = str(item.get("source_policy") or sampling.get("source_policy") or "")
            if expected_policy and actual_policy != expected_policy:
                raise ValueError("rollout source-policy mismatch")
            if expected_stage and str(sampling.get("rollout_stage") or "") != expected_stage:
                raise ValueError("rollout stage mismatch")
            trajectory = item.get("trajectory") if isinstance(item.get("trajectory"), dict) else {}
            if str(trajectory.get("terminated_by") or "") == "exception":
                raise ValueError("rollout contains infrastructure exception")
            seen[key] = actual_start
except (OSError, ValueError, TypeError, json.JSONDecodeError, KeyError):
    raise SystemExit(1)
raise SystemExit(0 if seen.keys() == expected.keys() else 1)
PY
}

case "${REQUIRE_TEACHER_ROLLOUT}" in 0|1) ;; *) echo "[fatal] REQUIRE_TEACHER_ROLLOUT must be 0 or 1" >&2; exit 2 ;; esac
[[ "${RL_ROLLOUT_PHASE_RETRIES}" =~ ^[0-9]+$ ]] || { echo "[fatal] RL_ROLLOUT_PHASE_RETRIES must be a non-negative integer" >&2; exit 2; }

phase_prepare() {
  run bash "${ROOT}/00_setup/download_conifer.sh"
  run "${PYTHON_BIN}" "${ROOT}/01_dataset/prepare_conifer.py" --input "${ROOT}/01_dataset/raw/Conifer/data/train_sft-00000-of-00001.parquet" --out-dir "${ROOT}/01_dataset/processed"
}

phase_rollout() {
  ensure "${SFT_TRAIN_DATA}" sft-train-data
  if rollout_complete_for "${SFT_TRAIN_DATA}" "${ROLL_OUT}" 0 "${SFT_TRAIN_LIMIT}" "${NUM_ROLLOUTS}" \
      "${ROLLOUT_START_AGENT}" "${ROLLOUT_START_AGENT_SEED}" 1 "${ROLLOUT_SOURCE_POLICY}"; then
    echo "[resume] rollout artifact is complete; skipping teacher generation"
    return 0
  fi
  run env DATA_PATH="${SFT_TRAIN_DATA}" LIMIT="${SFT_TRAIN_LIMIT}" NUM_ROLLOUTS="${NUM_ROLLOUTS}" \
    MODE="${ROLLOUT_MODE}" ROLLOUT_IS_TEACHER=1 ROLLOUT_SOURCE_POLICY="${ROLLOUT_SOURCE_POLICY}" \
    ROLLOUT_MODEL_A1="${ROLLOUT_MODEL_A1}" ROLLOUT_MODEL_A2="${ROLLOUT_MODEL_A2}" ROLLOUT_MODEL_A3="${ROLLOUT_MODEL_A3}" \
    RUN_ID="${TAG}_rollout" START_AGENT="${ROLLOUT_START_AGENT}" START_AGENT_SEED="${ROLLOUT_START_AGENT_SEED}" \
    SEED="${ROLLOUT_SEED}" \
    MIN_AGENTS_BEFORE_STOP="${ROLLOUT_MIN_AGENTS_BEFORE_STOP}" \
    MIN_HANDOFFS_BEFORE_STOP="${ROLLOUT_MIN_HANDOFFS_BEFORE_STOP}" \
    TEMPERATURE="${ROLLOUT_TEMPERATURE}" TOP_P="${ROLLOUT_TOP_P}" \
    FORCE_HANDOFF_UNTIL_FINAL="${ROLLOUT_FORCE_HANDOFF_UNTIL_FINAL}" \
    MAX_CONCURRENCY="${MAX_CONCURRENCY}" MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" PROTOCOL_MAX_NEW_TOKENS="${PROTOCOL_MAX_NEW_TOKENS}" \
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
    VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS}" VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS}" \
    VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER}" VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING}" \
    VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL}" VLLM_PERFORMANCE_MODE="${VLLM_PERFORMANCE_MODE}" \
    VLLM_GENERATION_CONFIG="${VLLM_GENERATION_CONFIG}" VLLM_DISABLE_LOG_STATS="${VLLM_DISABLE_LOG_STATS}" \
    VLLM_ASYNC_SCHEDULING="${VLLM_ASYNC_SCHEDULING}" VLLM_DISABLE_UVICORN_ACCESS_LOG="${VLLM_DISABLE_UVICORN_ACCESS_LOG}" \
    VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS}" SHARED_VLLM="${SHARED_VLLM}" SHARED_VLLM_BACKEND="${SHARED_VLLM_BACKEND}" \
    SHARED_VLLM_PORT="${SHARED_VLLM_PORT}" SHARED_VLLM_REPLICA_PORT_STRIDE="${SHARED_VLLM_REPLICA_PORT_STRIDE}" \
    SHARED_VLLM_GPUS="${SHARED_VLLM_GPUS}" SHARED_VLLM_TP="${SHARED_VLLM_TP}" SHARED_VLLM_DP="${SHARED_VLLM_DP}" \
    SHARED_VLLM_MAX_NUM_SEQS="${SHARED_VLLM_MAX_NUM_SEQS}" \
    SHARED_VLLM_MAX_NUM_BATCHED_TOKENS="${SHARED_VLLM_MAX_NUM_BATCHED_TOKENS}" \
    SHARED_VLLM_MODEL_NAME="${SHARED_VLLM_MODEL_NAME}" \
    SHARED_VLLM_EXECUTOR_BACKEND="${SHARED_VLLM_EXECUTOR_BACKEND}" \
    SHARED_VLLM_COORD_PORT_BASE="${SHARED_VLLM_COORD_PORT_BASE}" \
    SHARED_VLLM_DP_MASTER_PORT="${SHARED_VLLM_DP_MASTER_PORT}" \
    SHARED_VLLM_DP_RPC_PORT="${SHARED_VLLM_DP_RPC_PORT}" \
    SHARED_VLLM_INTERNAL_PORT_BASE="${SHARED_VLLM_INTERNAL_PORT_BASE}" \
    SHARED_VLLM_INTERNAL_PORT_STRIDE="${SHARED_VLLM_INTERNAL_PORT_STRIDE}" \
    SHARED_VLLM_START_RETRIES="${SHARED_VLLM_START_RETRIES}" \
    SHARED_VLLM_OMP_NUM_THREADS="${SHARED_VLLM_OMP_NUM_THREADS}" \
    ROLE_REPLICA_MODE="${ROLE_REPLICA_MODE}" ROLE_REPLICA_PORT_BASE="${ROLE_REPLICA_PORT_BASE}" \
    ROLE_REPLICA_ROLE_PORT_STRIDE="${ROLE_REPLICA_ROLE_PORT_STRIDE}" ROLE_REPLICA_PORT_STRIDE="${ROLE_REPLICA_PORT_STRIDE}" \
    ROLE_REPLICA_PORT_AUTO_SHIFT="${ROLE_REPLICA_PORT_AUTO_SHIFT}" ROLE_REPLICA_PORT_SCAN_BLOCKS="${ROLE_REPLICA_PORT_SCAN_BLOCKS}" \
    ROLE_REPLICA_MAX_NUM_SEQS="${ROLE_REPLICA_MAX_NUM_SEQS}" \
    ROLE_REPLICA_MAX_NUM_BATCHED_TOKENS="${ROLE_REPLICA_MAX_NUM_BATCHED_TOKENS}" \
    ROLE_REPLICA_START_RETRIES="${ROLE_REPLICA_START_RETRIES}" ROLE_REPLICA_OMP_NUM_THREADS="${ROLE_REPLICA_OMP_NUM_THREADS}" \
    JCA_ENDPOINT_RETRIES="${JCA_ENDPOINT_RETRIES}" PORT_WAIT_TIMEOUT="${PORT_WAIT_TIMEOUT}" PORT_WAIT_MAX_SLEEP="${PORT_WAIT_MAX_SLEEP}" \
    OUTPUT="${ROLL_OUT}" SCORED_OUTPUT="${SCORED_OUT}" RL_OUTPUT="${TEACHER_RL_OUT}" \
    ROLLOUT_STAGE="teacher_sft_source" SCORE_OUTPUTS=0 JUDGE_MODE=deterministic \
    bash "${ROOT}/06_evaluation/run_conifer_eval.sh"
}

score_paths() {
  local input="$1" scored="$2" rl="$3" run_id="$4"
  ensure "${input}" rollout
  if score_complete_for "${scored}" "${rl}"; then
    echo "[resume] score artifacts are complete; skipping judge (${run_id})"
    return 0
  fi
  if [[ "${JUDGE_MODE}" == llm && "${JUDGE_BACKEND}" == local_vllm ]]; then
    run env INPUT="${input}" SCORED_OUTPUT="${scored}" RL_OUTPUT="${rl}" \
      RUN_ID="${run_id}" ALPHA="${ALPHA:-0.5}" JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY}" \
      JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS}" JUDGE_LENGTH_RETRIES="${JUDGE_LENGTH_RETRIES}" \
      JUDGE_GPU_MEMORY_UTILIZATION="${JUDGE_GPU_MEMORY_UTILIZATION}" \
      JUDGE_VLLM_MAX_NUM_SEQS="${JUDGE_VLLM_MAX_NUM_SEQS}" \
      JUDGE_VLLM_MAX_NUM_BATCHED_TOKENS="${JUDGE_VLLM_MAX_NUM_BATCHED_TOKENS}" \
      JUDGE_VLLM_ENFORCE_EAGER="${JUDGE_VLLM_ENFORCE_EAGER}" \
      JUDGE_VLLM_ENABLE_PREFIX_CACHING="${JUDGE_VLLM_ENABLE_PREFIX_CACHING}" \
      JUDGE_VLLM_ENABLE_CHUNKED_PREFILL="${JUDGE_VLLM_ENABLE_CHUNKED_PREFILL}" \
      JUDGE_VLLM_PERFORMANCE_MODE="${JUDGE_VLLM_PERFORMANCE_MODE}" \
      JUDGE_VLLM_GENERATION_CONFIG="${JUDGE_VLLM_GENERATION_CONFIG}" \
      JUDGE_VLLM_DISABLE_LOG_STATS="${JUDGE_VLLM_DISABLE_LOG_STATS}" \
      JUDGE_VLLM_ASYNC_SCHEDULING="${JUDGE_VLLM_ASYNC_SCHEDULING}" \
      JUDGE_VLLM_DISABLE_UVICORN_ACCESS_LOG="${JUDGE_VLLM_DISABLE_UVICORN_ACCESS_LOG}" \
      JUDGE_VLLM_EXTRA_ARGS="${JUDGE_VLLM_EXTRA_ARGS}" \
      JUDGE_VLLM_EXECUTOR_BACKEND="${JUDGE_VLLM_EXECUTOR_BACKEND}" JUDGE_VLLM_BACKEND="${JUDGE_VLLM_BACKEND}" \
      JUDGE_REPLICA_PORT_STRIDE="${JUDGE_REPLICA_PORT_STRIDE}" \
      JUDGE_VLLM_COORD_PORT_BASE="${JUDGE_VLLM_COORD_PORT_BASE}" \
      JUDGE_VLLM_DP_MASTER_PORT="${JUDGE_VLLM_DP_MASTER_PORT}" \
      JUDGE_VLLM_DP_RPC_PORT="${JUDGE_VLLM_DP_RPC_PORT}" \
      JUDGE_VLLM_INTERNAL_PORT_BASE="${JUDGE_VLLM_INTERNAL_PORT_BASE}" \
      JUDGE_VLLM_INTERNAL_PORT_STRIDE="${JUDGE_VLLM_INTERNAL_PORT_STRIDE}" \
      JUDGE_VLLM_START_RETRIES="${JUDGE_VLLM_START_RETRIES}" \
      JUDGE_VLLM_OMP_NUM_THREADS="${JUDGE_VLLM_OMP_NUM_THREADS}" \
      PORT_WAIT_TIMEOUT="${PORT_WAIT_TIMEOUT}" PORT_WAIT_MAX_SLEEP="${PORT_WAIT_MAX_SLEEP}" \
      SKIP_COMPLETE="${SKIP_COMPLETED}" \
      bash "${ROOT}/04_judge/run_local_conifer_judge.sh"
  else
    [[ "${JUDGE_BACKEND}" == external || "${JUDGE_MODE}" == deterministic ]] || { echo "[fatal] invalid JUDGE_BACKEND=${JUDGE_BACKEND}" >&2; exit 2; }
    run "${PYTHON_BIN}" "${ROOT}/04_judge/score_conifer_rollouts.py" --input "${input}" \
      --scored-output "${scored}" --rl-output "${rl}" --judge-mode "${JUDGE_MODE}"
  fi
}

phase_score() {
  score_paths "${ROLL_OUT}" "${SCORED_OUT}" "${TEACHER_RL_OUT}" "${TAG}_teacher_judge"
}

phase_build_sft() {
  ensure "${TEACHER_RL_OUT}" scored-turn-data
  if sft_data_complete; then
    echo "[resume] SFT data artifacts are complete; skipping builder"
    return 0
  fi
  local builder_args=("${PYTHON_BIN}" "${ROOT}/05_training/build_conifer_sft.py" --input "${TEACHER_RL_OUT}"
    --out-dir "${SFT_DATA_DIR}" --min-hard-score "${MIN_HARD_SCORE:-0.35}"
    --source-policy "${ROLLOUT_SOURCE_POLICY}"
    --min-distinct-agents "${ROLLOUT_MIN_AGENTS_BEFORE_STOP}"
    --min-handoffs "${ROLLOUT_MIN_HANDOFFS_BEFORE_STOP}")
  if [[ "${REQUIRE_TEACHER_ROLLOUT}" == 1 ]]; then
    builder_args+=(--require-source-policy --require-teacher-3x8b)
  fi
  run "${builder_args[@]}"
}

phase_sft() {
  ensure "${SFT_DATA_DIR}/A1.jsonl" A1-SFT-data
  ensure "${SFT_DATA_DIR}/A2.jsonl" A2-SFT-data
  ensure "${SFT_DATA_DIR}/A3.jsonl" A3-SFT-data
  run env SFT_DATA_DIR="${SFT_DATA_DIR}" RUN_ID="${TAG}" RUN_DIR="${SFT_RUN_DIR}" \
    SOURCE_POLICY="${ROLLOUT_SOURCE_POLICY}" \
    NUM_EPOCHS="${SFT_NUM_EPOCHS}" \
    SKIP_COMPLETED="${SKIP_COMPLETED}" \
    MODEL_A1="${STUDENT_MODEL_A1}" MODEL_A2="${STUDENT_MODEL_A2}" MODEL_A3="${STUDENT_MODEL_A3}" \
    bash "${ROOT}/05_training/train_sft.sh"
}

phase_rl_rollout() {
  ensure "${RL_TRAIN_DATA}" rl-train-data
  ensure "${SFT_RUN_DIR}/A1/final/adapter_model.safetensors" A1-SFT-adapter
  ensure "${SFT_RUN_DIR}/A2/final/adapter_model.safetensors" A2-SFT-adapter
  ensure "${SFT_RUN_DIR}/A3/final/adapter_model.safetensors" A3-SFT-adapter
  [[ "${RL_ROLLOUT_OUT}" != "${ROLL_OUT}" ]] || {
    echo "[fatal] RL rollout output must be separate from teacher rollout output" >&2
    exit 2
  }
  if rollout_complete_for "${RL_TRAIN_DATA}" "${RL_ROLLOUT_OUT}" "${RL_ROLLOUT_START}" "${RL_ROLLOUT_LIMIT}" \
      "${RL_ROLLOUT_NUM_ROLLOUTS}" "${RL_ROLLOUT_START_AGENT}" "${RL_ROLLOUT_START_AGENT_SEED}" \
      0 "${RL_ROLLOUT_SOURCE_POLICY}" "rl_from_sft"; then
    echo "[resume] independent SFT-policy RL rollout is complete; skipping generation"
    return 0
  fi
  local attempt attempts=$((RL_ROLLOUT_PHASE_RETRIES + 1))
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    if run env DATA_PATH="${RL_TRAIN_DATA}" START="${RL_ROLLOUT_START}" LIMIT="${RL_ROLLOUT_LIMIT}" \
      NUM_ROLLOUTS="${RL_ROLLOUT_NUM_ROLLOUTS}" MODE="${RL_ROLLOUT_MODE}" ROLLOUT_IS_TEACHER=0 \
      ROLLOUT_SOURCE_POLICY="${RL_ROLLOUT_SOURCE_POLICY}" ROLLOUT_STAGE="rl_from_sft" \
      MODEL_A1="${STUDENT_MODEL_A1}" MODEL_A2="${STUDENT_MODEL_A2}" MODEL_A3="${STUDENT_MODEL_A3}" \
      ADAPTER_A1="${SFT_RUN_DIR}/A1/final" ADAPTER_A2="${SFT_RUN_DIR}/A2/final" ADAPTER_A3="${SFT_RUN_DIR}/A3/final" \
      START_AGENT="${RL_ROLLOUT_START_AGENT}" START_AGENT_SEED="${RL_ROLLOUT_START_AGENT_SEED}" SEED="${RL_ROLLOUT_SEED}" \
      MIN_AGENTS_BEFORE_STOP="${RL_ROLLOUT_MIN_AGENTS_BEFORE_STOP}" \
      MIN_HANDOFFS_BEFORE_STOP="${RL_ROLLOUT_MIN_HANDOFFS_BEFORE_STOP}" \
      TEMPERATURE="${RL_ROLLOUT_TEMPERATURE}" TOP_P="${RL_ROLLOUT_TOP_P}" \
      FORCE_HANDOFF_UNTIL_FINAL="${RL_ROLLOUT_FORCE_HANDOFF_UNTIL_FINAL}" \
      MAX_CONCURRENCY="${MAX_CONCURRENCY}" MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" \
      PROTOCOL_MAX_NEW_TOKENS="${PROTOCOL_MAX_NEW_TOKENS}" GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
      VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS}" VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS}" \
      VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER}" VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING}" \
      VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL}" VLLM_PERFORMANCE_MODE="${VLLM_PERFORMANCE_MODE}" \
      VLLM_GENERATION_CONFIG="${VLLM_GENERATION_CONFIG}" VLLM_DISABLE_LOG_STATS="${VLLM_DISABLE_LOG_STATS}" \
      VLLM_ASYNC_SCHEDULING="${VLLM_ASYNC_SCHEDULING}" VLLM_DISABLE_UVICORN_ACCESS_LOG="${VLLM_DISABLE_UVICORN_ACCESS_LOG}" \
      VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS}" SHARED_VLLM=0 \
      ROLE_REPLICA_MODE="${ROLE_REPLICA_MODE}" ROLE_REPLICA_PORT_BASE="${ROLE_REPLICA_PORT_BASE}" \
      ROLE_REPLICA_ROLE_PORT_STRIDE="${ROLE_REPLICA_ROLE_PORT_STRIDE}" ROLE_REPLICA_PORT_STRIDE="${ROLE_REPLICA_PORT_STRIDE}" \
      ROLE_REPLICA_PORT_AUTO_SHIFT="${ROLE_REPLICA_PORT_AUTO_SHIFT}" ROLE_REPLICA_PORT_SCAN_BLOCKS="${ROLE_REPLICA_PORT_SCAN_BLOCKS}" \
      ROLE_REPLICA_MAX_NUM_SEQS="${ROLE_REPLICA_MAX_NUM_SEQS}" ROLE_REPLICA_MAX_NUM_BATCHED_TOKENS="${ROLE_REPLICA_MAX_NUM_BATCHED_TOKENS}" \
      ROLE_REPLICA_START_RETRIES="${ROLE_REPLICA_START_RETRIES}" ROLE_REPLICA_OMP_NUM_THREADS="${ROLE_REPLICA_OMP_NUM_THREADS}" \
      JCA_ENDPOINT_RETRIES="${JCA_ENDPOINT_RETRIES}" PORT_WAIT_TIMEOUT="${PORT_WAIT_TIMEOUT}" PORT_WAIT_MAX_SLEEP="${PORT_WAIT_MAX_SLEEP}" \
      SERVER_HEALTH_INTERVAL="${SERVER_HEALTH_INTERVAL}" \
      OUTPUT="${RL_ROLLOUT_OUT}" SCORED_OUTPUT="${RL_SCORED_OUT}" RL_OUTPUT="${RL_OUT}" \
      SCORE_OUTPUTS=0 JUDGE_MODE=deterministic RUN_ID="${TAG}_rl_rollout" \
      bash "${ROOT}/06_evaluation/run_conifer_eval.sh"; then
      return 0
    fi
    if ((attempt < attempts)); then
      echo "[retry] independent RL rollout attempt ${attempt}/${attempts} failed; restarting servers and resuming unfinished keys" >&2
      sleep 3
    fi
  done
  echo "[fatal] independent RL rollout failed after ${attempts} attempts" >&2
  return 1
}

phase_rl_score() {
  score_paths "${RL_ROLLOUT_OUT}" "${RL_SCORED_OUT}" "${RL_OUT}" "${TAG}_rl_judge"
}

phase_rl() {
  ensure "${RL_OUT}" scored-turn-data
  ensure "${SFT_RUN_DIR}/A1/final/adapter_model.safetensors" A1-SFT-adapter
  ensure "${SFT_RUN_DIR}/A2/final/adapter_model.safetensors" A2-SFT-adapter
  ensure "${SFT_RUN_DIR}/A3/final/adapter_model.safetensors" A3-SFT-adapter
  run env RL_DATA="${RL_OUT}" SFT_ROOT="${SFT_RUN_DIR}" RUN_ID="${TAG}" RUN_DIR="${RL_RUN_DIR}" \
    MODEL_A1="${STUDENT_MODEL_A1}" MODEL_A2="${STUDENT_MODEL_A2}" MODEL_A3="${STUDENT_MODEL_A3}" \
    NUM_EPOCHS="${RL_NUM_EPOCHS}" PER_DEVICE_BATCH="${RL_PER_DEVICE_BATCH}" GRAD_ACCUM="${RL_GRAD_ACCUM}" MAX_SEQ="${RL_MAX_SEQ}" \
    SAVE_STEPS="${RL_SAVE_STEPS}" RESUME_FROM_CHECKPOINT="${RL_RESUME_FROM_CHECKPOINT}" \
    REQUIRE_SFT_INIT="${REQUIRE_SFT_INIT}" SKIP_COMPLETED="${SKIP_COMPLETED}" \
    bash "${ROOT}/05_training/train_rl.sh"
}

phase_eval() {
  ensure "${TEST_DATA}" test-data
  ensure "${RL_RUN_DIR}/A1/final/adapter_model.safetensors" A1-RL-adapter
  ensure "${RL_RUN_DIR}/A2/final/adapter_model.safetensors" A2-RL-adapter
  ensure "${RL_RUN_DIR}/A3/final/adapter_model.safetensors" A3-RL-adapter
  if [[ -s "${EVAL_DIR}/trajectories.jsonl" && -s "${EVAL_DIR}/scored.jsonl" && -s "${EVAL_DIR}/scored_stats.json" ]]; then
    echo "[resume] holdout evaluation artifacts are complete; skipping evaluation"
    return 0
  fi
  run env DATA_PATH="${TEST_DATA}" MODE=sft_lora_mas RUN_ID="${TAG}_test" \
    START_AGENT="${EVAL_START_AGENT:-balanced}" START_AGENT_SEED="${EVAL_START_AGENT_SEED:-42}" \
    NUM_ROLLOUTS="${NUM_ROLLOUTS}" \
    MODEL_A1="${STUDENT_MODEL_A1}" MODEL_A2="${STUDENT_MODEL_A2}" MODEL_A3="${STUDENT_MODEL_A3}" \
    ADAPTER_A1="${RL_RUN_DIR}/A1/final" ADAPTER_A2="${RL_RUN_DIR}/A2/final" ADAPTER_A3="${RL_RUN_DIR}/A3/final" \
    MAX_CONCURRENCY="${EVAL_MAX_CONCURRENCY}" GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
    VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS}" VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS}" \
    VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER}" VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING}" \
    VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL}" VLLM_PERFORMANCE_MODE="${VLLM_PERFORMANCE_MODE}" \
    VLLM_GENERATION_CONFIG="${VLLM_GENERATION_CONFIG}" VLLM_DISABLE_LOG_STATS="${VLLM_DISABLE_LOG_STATS}" \
    VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS}" SHARED_VLLM=0 \
    ROLE_REPLICA_MODE="${ROLE_REPLICA_MODE}" ROLE_REPLICA_PORT_BASE="${ROLE_REPLICA_PORT_BASE}" \
    ROLE_REPLICA_ROLE_PORT_STRIDE="${ROLE_REPLICA_ROLE_PORT_STRIDE}" ROLE_REPLICA_PORT_SCAN_BLOCKS="${ROLE_REPLICA_PORT_SCAN_BLOCKS}" \
    ROLE_REPLICA_PORT_AUTO_SHIFT="${ROLE_REPLICA_PORT_AUTO_SHIFT}" ROLE_REPLICA_PORT_STRIDE="${ROLE_REPLICA_PORT_STRIDE}" \
    ROLE_REPLICA_MAX_NUM_SEQS="${ROLE_REPLICA_MAX_NUM_SEQS}" \
    ROLE_REPLICA_MAX_NUM_BATCHED_TOKENS="${ROLE_REPLICA_MAX_NUM_BATCHED_TOKENS}" \
    ROLE_REPLICA_START_RETRIES="${ROLE_REPLICA_START_RETRIES}" ROLE_REPLICA_OMP_NUM_THREADS="${ROLE_REPLICA_OMP_NUM_THREADS}" \
    JCA_ENDPOINT_RETRIES="${JCA_ENDPOINT_RETRIES}" PORT_WAIT_TIMEOUT="${PORT_WAIT_TIMEOUT}" PORT_WAIT_MAX_SLEEP="${PORT_WAIT_MAX_SLEEP}" \
    OUTPUT="${EVAL_DIR}/trajectories.jsonl" SCORED_OUTPUT="${EVAL_DIR}/scored.jsonl" \
    RL_OUTPUT="${EVAL_DIR}/rl_records.jsonl" JUDGE_MODE="${EVAL_JUDGE_MODE:-deterministic}" \
    bash "${ROOT}/06_evaluation/run_conifer_eval.sh"
}

phase_ablations() {
  run env DATA_PATH="${TEST_DATA}" RUN_GROUP="${TAG}" \
    NUM_ROLLOUTS="${NUM_ROLLOUTS}" \
    JUDGE_MODE="${EVAL_ABLATION_JUDGE_MODE:-deterministic}" \
    EVAL_MAX_CONCURRENCY="${EVAL_MAX_CONCURRENCY}" JCA_HTTP_KEEPALIVE="${JCA_HTTP_KEEPALIVE}" \
    JCA_HTTP_POOL_MAXSIZE="${JCA_HTTP_POOL_MAXSIZE}" \
    VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS}" VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS}" \
    VLLM_ASYNC_SCHEDULING="${VLLM_ASYNC_SCHEDULING}" VLLM_DISABLE_UVICORN_ACCESS_LOG="${VLLM_DISABLE_UVICORN_ACCESS_LOG}" \
    ROLE_REPLICA_MODE="${ROLE_REPLICA_MODE}" ROLE_REPLICA_PORT_BASE="${ROLE_REPLICA_PORT_BASE}" \
    ROLE_REPLICA_ROLE_PORT_STRIDE="${ROLE_REPLICA_ROLE_PORT_STRIDE}" ROLE_REPLICA_PORT_STRIDE="${ROLE_REPLICA_PORT_STRIDE}" \
    ROLE_REPLICA_PORT_AUTO_SHIFT="${ROLE_REPLICA_PORT_AUTO_SHIFT}" ROLE_REPLICA_PORT_SCAN_BLOCKS="${ROLE_REPLICA_PORT_SCAN_BLOCKS}" \
    ROLE_REPLICA_MAX_NUM_SEQS="${ROLE_REPLICA_MAX_NUM_SEQS}" \
    ROLE_REPLICA_MAX_NUM_BATCHED_TOKENS="${ROLE_REPLICA_MAX_NUM_BATCHED_TOKENS}" \
    ROLE_REPLICA_START_RETRIES="${ROLE_REPLICA_START_RETRIES}" ROLE_REPLICA_OMP_NUM_THREADS="${ROLE_REPLICA_OMP_NUM_THREADS}" \
    JCA_ENDPOINT_RETRIES="${JCA_ENDPOINT_RETRIES}" PORT_WAIT_TIMEOUT="${PORT_WAIT_TIMEOUT}" PORT_WAIT_MAX_SLEEP="${PORT_WAIT_MAX_SLEEP}" \
    START_AGENT="${EVAL_START_AGENT:-balanced}" START_AGENT_SEED="${EVAL_START_AGENT_SEED:-42}" \
    MODEL_A1="${STUDENT_MODEL_A1}" MODEL_A2="${STUDENT_MODEL_A2}" MODEL_A3="${STUDENT_MODEL_A3}" \
    VARIANTS="${EVAL_ABLATION_VARIANTS:-zero_shot,single_a3,fixed_three_turn,sft}" \
    SFT_ADAPTER_A1="${SFT_RUN_DIR}/A1/final" SFT_ADAPTER_A2="${SFT_RUN_DIR}/A2/final" SFT_ADAPTER_A3="${SFT_RUN_DIR}/A3/final" \
    bash "${ROOT}/07_ablation/run_conifer_ablation_suite.sh"
}

phase_train_ablations() {
  run env TRAIN_DATA="${TRAIN_DATA}" TAG="${TAG}" \
    MAX_CONCURRENCY="${MAX_CONCURRENCY}" JCA_HTTP_KEEPALIVE="${JCA_HTTP_KEEPALIVE}" \
    JCA_HTTP_POOL_MAXSIZE="${JCA_HTTP_POOL_MAXSIZE}" VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS}" \
    VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS}" VLLM_ASYNC_SCHEDULING="${VLLM_ASYNC_SCHEDULING}" \
    VLLM_DISABLE_UVICORN_ACCESS_LOG="${VLLM_DISABLE_UVICORN_ACCESS_LOG}" PORT_WAIT_TIMEOUT="${PORT_WAIT_TIMEOUT}" PORT_WAIT_MAX_SLEEP="${PORT_WAIT_MAX_SLEEP}" \
    MODEL_A1="${STUDENT_MODEL_A1}" MODEL_A2="${STUDENT_MODEL_A2}" MODEL_A3="${STUDENT_MODEL_A3}" \
    ROLLOUT_MODEL_A1="${ROLLOUT_MODEL_A1}" ROLLOUT_MODEL_A2="${ROLLOUT_MODEL_A2}" ROLLOUT_MODEL_A3="${ROLLOUT_MODEL_A3}" \
    ROLLOUT_SOURCE_POLICY="${ROLLOUT_SOURCE_POLICY}" ROLLOUT_START_AGENT="${ROLLOUT_START_AGENT}" \
    ROLLOUT_START_AGENT_SEED="${ROLLOUT_START_AGENT_SEED}" \
    bash "${ROOT}/07_ablation/run_conifer_training_ablation_suite.sh"
}

phase_official() {
  run bash "${ROOT}/00_setup/download_official_evals.sh"
  run env PHASE=all BENCHMARK="${OFFICIAL_BENCHMARK:-ifeval}" MOCK="${MOCK:-0}" \
    bash "${ROOT}/06_evaluation/run_official_benchmark.sh"
}

case "${MODE}" in
  prepare) phase_prepare ;;
  rollout) phase_rollout ;;
  score) phase_score ;;
  build_sft) phase_build_sft ;;
  sft) phase_sft ;;
  rl_rollout) phase_rl_rollout ;;
  rl_score) phase_rl_score ;;
  rl) phase_rl ;;
  eval) phase_eval ;;
  ablations) phase_ablations ;;
  train_ablations) phase_train_ablations ;;
  official) phase_official ;;
  all)
    if [[ "${SKIP_PREPARE}" != 1 ]]; then phase_prepare; fi
    phase_rollout; phase_score; phase_build_sft; phase_sft; phase_rl_rollout; phase_rl_score; phase_rl; phase_eval; phase_ablations
    ;;
  *) echo "[fatal] unknown MODE=${MODE}" >&2; exit 2 ;;
esac
echo "[ok] Conifer pipeline phase ${MODE} complete (tag=${TAG})"

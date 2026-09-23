#!/usr/bin/env bash
# Run the corrected v13 MAS evaluation once per start-agent seed, serially.
#
# Example:
#   INFINITE=0 SEEDS="42 43 44" bash scripts/run_gsm_v13_seed_sweep.sh
#
# The per-seed role-batched launcher waits for all GPUs to be idle, starts one
# LoRA server at a time (A1/A2/A3), finalizes a 132-row JSONL, and then tears
# the server down before the next seed starts.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/wangyuheng/jca}"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/qwen35/bin/python}"
PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-/data/wangyuheng}"
LAUNCHER="${LAUNCHER:-${PROJECT_ROOT}/gsm/scripts/run_gsm_vllm_role_batched_8gpu.sh}"
RUNNER="${RUNNER:-${PROJECT_ROOT}/gsm/scripts/run_mas_role_batched_thinking_hidden_self_handoff.py}"

DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/Math/data/GSM-HARD/splits/gsmhardv2_dev.jsonl}"
MODEL_A1="${MODEL_A1:-/data/wangyuheng/models/Qwen3-1.7B}"
MODEL_A2="${MODEL_A2:-/data/wangyuheng/models/Qwen3-4B}"
MODEL_A3="${MODEL_A3:-/data/wangyuheng/models/Qwen3-8B}"
ADAPTER_ROOT="${ADAPTER_ROOT:-${PROJECT_ROOT}/rl_runs/gsm_judge_rl/gsm_judge_rl_v13_14b_role_c2c_1_05_1}"

# By default keep evaluating seed=42,43,... forever. Set INFINITE=0 and
# provide SEEDS="42 43 ..." for a finite run, or set START_SEED to resume at
# a later seed.
SEEDS="${SEEDS:-}"
START_SEED="${START_SEED:-42}"
INFINITE="${INFINITE:-1}"
START="${START:-0}"
LIMIT="${LIMIT:-132}"
T_MAX="${T_MAX:-8}"
START_AGENT="${START_AGENT:-random}"
MIN_AGENTS_BEFORE_STOP="${MIN_AGENTS_BEFORE_STOP:-1}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8192}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-0.95}"
# The 0.7045 reference used generation seed 42. This keeps the seed sweep
# focused on START_AGENT_SEED; override explicitly when reproducing a variant.
GENERATION_SEED="${GENERATION_SEED:-42}"
JSON_TRANSPORT="${JSON_TRANSPORT:-none}"
ENABLE_THINKING="${ENABLE_THINKING:-1}"
ENFORCE_COLLABORATION_POLICY="${ENFORCE_COLLABORATION_POLICY:-0}"
GLOBAL_TURN_BATCHING="${GLOBAL_TURN_BATCHING:-0}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-1}"
GROUP_RETRIES="${GROUP_RETRIES:-5}"
STEP_RETRIES="${STEP_RETRIES:-4}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-512}"
API_TIMEOUT="${API_TIMEOUT:-900}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-8}"
WAIT_FOR_IDLE_GPUS="${WAIT_FOR_IDLE_GPUS:-1}"
GPU_IDLE_TIMEOUT_SECONDS="${GPU_IDLE_TIMEOUT_SECONDS:-86400}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-10}"
GPU_IDLE_CHECKS="${GPU_IDLE_CHECKS:-2}"
RESUME_EXISTING="${RESUME_EXISTING:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
RETRY_FAILED_SECONDS="${RETRY_FAILED_SECONDS:-60}"

SWEEP_TAG="${SWEEP_TAG:-gsm_judge_rl_v13_seed_sweep_$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/gsm_eval/role_batched/${SWEEP_TAG}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/gsm_eval/role_batched/${SWEEP_TAG}}"
SUMMARY_PATH="${SUMMARY_PATH:-${OUTPUT_ROOT}/summary.tsv}"
LAUNCHER_LOG="${LAUNCHER_LOG:-${LOG_ROOT}/sweep.log}"

fatal() {
  echo "[fatal] $*" >&2
  exit 1
}

is_nonnegative_integer() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

validate() {
  [[ -x "$PYTHON_BIN" ]] || fatal "PYTHON_BIN is not executable: $PYTHON_BIN"
  [[ -x "$LAUNCHER" ]] || fatal "role-batched launcher is not executable: $LAUNCHER"
  [[ -f "$RUNNER" ]] || fatal "runner not found: $RUNNER"
  [[ -s "$DATA_PATH" ]] || fatal "dataset not found or empty: $DATA_PATH"
  for path in "$MODEL_A1" "$MODEL_A2" "$MODEL_A3" \
    "$ADAPTER_ROOT/A1/final" "$ADAPTER_ROOT/A2/final" "$ADAPTER_ROOT/A3/final"; do
    [[ -e "$path" ]] || fatal "required path not found: $path"
  done
  for agent in A1 A2 A3; do
    [[ -s "$ADAPTER_ROOT/$agent/final/adapter_model.safetensors" ]] || \
      fatal "missing v13 adapter weights: $ADAPTER_ROOT/$agent/final"
  done
  is_nonnegative_integer "$START" || fatal "START must be non-negative"
  is_nonnegative_integer "$LIMIT" || fatal "LIMIT must be non-negative"
  [[ "$LIMIT" -gt 0 ]] || fatal "LIMIT must be positive"
  is_nonnegative_integer "$T_MAX" || fatal "T_MAX must be a positive integer"
  [[ "$T_MAX" -gt 0 ]] || fatal "T_MAX must be positive"
  is_nonnegative_integer "$MIN_AGENTS_BEFORE_STOP" || fatal "MIN_AGENTS_BEFORE_STOP must be non-negative"
  [[ "$MIN_AGENTS_BEFORE_STOP" -ge 1 && "$MIN_AGENTS_BEFORE_STOP" -le 3 ]] || \
    fatal "MIN_AGENTS_BEFORE_STOP must be in [1,3]"
  [[ "$START_AGENT" == "random" ]] || fatal "this sweep requires START_AGENT=random"
  [[ "$ENABLE_THINKING" == "0" || "$ENABLE_THINKING" == "1" ]] || fatal "ENABLE_THINKING must be 0 or 1"
  [[ "$RESUME_EXISTING" == "0" || "$RESUME_EXISTING" == "1" ]] || fatal "RESUME_EXISTING must be 0 or 1"
  [[ "$CONTINUE_ON_ERROR" == "0" || "$CONTINUE_ON_ERROR" == "1" ]] || fatal "CONTINUE_ON_ERROR must be 0 or 1"
  [[ "$INFINITE" == "0" || "$INFINITE" == "1" ]] || fatal "INFINITE must be 0 or 1"
  is_nonnegative_integer "$START_SEED" || fatal "START_SEED must be non-negative"
  is_nonnegative_integer "$RETRY_FAILED_SECONDS" || fatal "RETRY_FAILED_SECONDS must be non-negative"
  if [[ "$INFINITE" == "0" ]]; then
    [[ -n "${SEEDS//[[:space:]]/}" ]] || fatal "SEEDS is empty when INFINITE=0"
  fi
}

write_summary_header() {
  mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
  if [[ ! -e "$SUMMARY_PATH" ]]; then
    printf 'seed\tstatus\tn\tcorrect\tem\toutput\trun_dir\tstarted_at\tfinished_at\n' >"$SUMMARY_PATH"
  fi
}

summarize_output() {
  local seed="$1" status="$2" output="$3" run_dir="$4" started_at="$5" finished_at="$6"
  "$PYTHON_BIN" - "$seed" "$status" "$output" "$run_dir" "$started_at" "$finished_at" "$SUMMARY_PATH" <<'PY'
import json
import sys
from pathlib import Path

seed, status, output, run_dir, started, finished, summary = sys.argv[1:]
rows = []
path = Path(output)
if path.is_file():
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
correct = sum(float(row.get("em", 0.0)) == 1.0 for row in rows)
em = correct / len(rows) if rows else 0.0
with Path(summary).open("a", encoding="utf-8") as handle:
    handle.write("\t".join([
        seed, status, str(len(rows)), str(correct), f"{em:.6f}",
        output, run_dir, started, finished,
    ]) + "\n")
print(f"[summary] seed={seed} status={status} n={len(rows)} correct={correct}/{len(rows)} em={em:.6f}")
PY
}

run_seed() {
  local seed="$1"
  local run_id="${SWEEP_TAG}_seed${seed}_dev${LIMIT}_thinking_hidden_self_handoff_ctx${MAX_MODEL_LEN}"
  local run_dir="${LOG_ROOT}/${run_id}"
  local output="${OUTPUT_ROOT}/${run_id}.jsonl"
  local state="${run_dir}/trajectory_state.json"
  local started_at finished_at resume=0

  if [[ -s "$output" ]]; then
    local rows
    rows="$(wc -l <"$output")"
    if [[ "$rows" -eq "$LIMIT" ]]; then
      echo "[skip] seed=$seed already has a complete output: $output"
      started_at="existing"
      finished_at="existing"
      summarize_output "$seed" "existing" "$output" "$run_dir" "$started_at" "$finished_at"
      return 0
    fi
    fatal "seed=$seed has an incomplete output ($rows/$LIMIT lines): $output; remove or inspect it before retrying"
  fi
  if [[ "$RESUME_EXISTING" == "1" && -s "$state" ]]; then
    resume=1
    echo "[resume] seed=$seed using existing state: $state"
  elif [[ -e "$state" ]]; then
    fatal "seed=$seed has an existing state but RESUME_EXISTING=0: $state"
  fi

  started_at="$(date --iso-8601=seconds)"
  echo
  echo "================ v13 seed=$seed (${started_at}) ================"
  if env \
    PROJECT_ROOT="$PROJECT_ROOT" \
    PYTHON_BIN="$PYTHON_BIN" \
    PYTHONPATH_ROOT="$PYTHONPATH_ROOT" \
    RUNNER="$RUNNER" \
    DATA_PATH="$DATA_PATH" START="$START" LIMIT="$LIMIT" T_MAX="$T_MAX" \
    START_AGENT="$START_AGENT" START_AGENT_SEED="$seed" \
    MIN_AGENTS_BEFORE_STOP="$MIN_AGENTS_BEFORE_STOP" \
    MAX_CONCURRENCY="$MAX_CONCURRENCY" MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    MAX_MODEL_LEN="$MAX_MODEL_LEN" TEMPERATURE="$TEMPERATURE" TOP_P="$TOP_P" \
    GENERATION_SEED="$GENERATION_SEED" JSON_TRANSPORT="$JSON_TRANSPORT" \
    ENABLE_THINKING="$ENABLE_THINKING" \
    ENFORCE_COLLABORATION_POLICY="$ENFORCE_COLLABORATION_POLICY" \
    GLOBAL_TURN_BATCHING="$GLOBAL_TURN_BATCHING" NUM_ROLLOUTS="$NUM_ROLLOUTS" \
    GROUP_RETRIES="$GROUP_RETRIES" STEP_RETRIES="$STEP_RETRIES" \
    CHECKPOINT_EVERY="$CHECKPOINT_EVERY" API_TIMEOUT="$API_TIMEOUT" \
    MODEL_A1="$MODEL_A1" MODEL_A2="$MODEL_A2" MODEL_A3="$MODEL_A3" \
    ADAPTER_A1="$ADAPTER_ROOT/A1/final" \
    ADAPTER_A2="$ADAPTER_ROOT/A2/final" \
    ADAPTER_A3="$ADAPTER_ROOT/A3/final" \
    GPU_IDS="$GPU_IDS" TENSOR_PARALLEL_SIZE="$TENSOR_PARALLEL_SIZE" \
    DATA_PARALLEL_SIZE="$DATA_PARALLEL_SIZE" WAIT_FOR_IDLE_GPUS="$WAIT_FOR_IDLE_GPUS" \
    GPU_IDLE_TIMEOUT_SECONDS="$GPU_IDLE_TIMEOUT_SECONDS" GPU_POLL_SECONDS="$GPU_POLL_SECONDS" \
    GPU_IDLE_CHECKS="$GPU_IDLE_CHECKS" RESUME="$resume" \
    RUN_ID="$run_id" RUN_DIR="$run_dir" OUTPUT_PATH="$output" \
    LOG_DIR="$LOG_ROOT" OUTPUT_DIR="$OUTPUT_ROOT" \
    bash "$LAUNCHER"; then
    local rc=0
  else
    local rc=$?
  fi
  finished_at="$(date --iso-8601=seconds)"
  if [[ "$rc" -eq 0 && -s "$output" && "$(wc -l <"$output")" -eq "$LIMIT" ]]; then
    summarize_output "$seed" "completed" "$output" "$run_dir" "$started_at" "$finished_at"
    return 0
  fi
  summarize_output "$seed" "failed_rc${rc}" "$output" "$run_dir" "$started_at" "$finished_at"
  if [[ "$CONTINUE_ON_ERROR" == "1" ]]; then
    echo "[warn] seed=$seed failed (rc=$rc); continuing to next seed"
    return 0
  fi
  return "$rc"
}

validate
write_summary_header

printf '[sweep] tag=%s infinite=%s start_seed=%s\n' "$SWEEP_TAG" "$INFINITE" "$START_SEED"
printf '[sweep] outputs=%s\n[sweep] summary=%s\n' "$OUTPUT_ROOT" "$SUMMARY_PATH"
if [[ "$INFINITE" == "1" ]]; then
  seed="$START_SEED"
  while true; do
    if run_seed "$seed"; then
      seed=$((seed + 1))
      continue
    fi
    rc=$?
    echo "[warn] seed=$seed failed (rc=$rc); preserving state and retrying after ${RETRY_FAILED_SECONDS}s"
    sleep "$RETRY_FAILED_SECONDS"
  done
fi

read -r -a seed_list <<<"$SEEDS"
printf '[sweep] finite seeds=%s\n' "${seed_list[*]}"
for seed in "${seed_list[@]}"; do
  is_nonnegative_integer "$seed" || fatal "invalid seed: $seed"
  run_seed "$seed"
done
echo
echo "[sweep] all requested seeds finished: $(date '+%F %T %Z')"
echo "[sweep] summary: $SUMMARY_PATH"

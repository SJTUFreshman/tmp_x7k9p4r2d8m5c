#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=config.env
source "${EXPERIMENT_ROOT}/config.env"

fatal() {
  echo "[fatal] $*" >&2
  exit 1
}

log() {
  echo "[$(date '+%F %T %Z')] $*"
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

require_path() {
  [[ -e "$1" ]] || fatal "required path not found: $1"
}

require_file() {
  [[ -s "$1" ]] || fatal "required file missing or empty: $1"
}

adapter_complete() {
  [[ -s "$1/adapter_config.json" && -s "$1/adapter_model.safetensors" ]]
}

require_adapter() {
  adapter_complete "$1" || fatal "adapter is incomplete: $1"
}

require_positive_int() {
  [[ "$2" =~ ^[1-9][0-9]*$ ]] || fatal "$1 must be a positive integer: $2"
}

require_nonnegative_int() {
  [[ "$2" =~ ^[0-9]+$ ]] || fatal "$1 must be a non-negative integer: $2"
}

require_bool() {
  [[ "$2" == "0" || "$2" == "1" ]] || fatal "$1 must be 0 or 1: $2"
}

require_probability() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import math
import sys

name, raw = sys.argv[1:]
try:
    value = float(raw)
except ValueError as exc:
    raise SystemExit(f"[fatal] {name} must be numeric: {raw}") from exc
if not math.isfinite(value) or not 0.0 <= value <= 1.0:
    raise SystemExit(f"[fatal] {name} must be in [0, 1]: {raw}")
PY
}

require_positive_number() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import math
import sys

name, raw = sys.argv[1:]
try:
    value = float(raw)
except ValueError as exc:
    raise SystemExit(f"[fatal] {name} must be numeric: {raw}") from exc
if not math.isfinite(value) or value <= 0.0:
    raise SystemExit(f"[fatal] {name} must be positive: {raw}")
PY
}

validate_static_config() {
  [[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] || fatal "TAG contains unsupported characters"
  for pair in \
    "RESUME:$RESUME" "DRY_RUN:$DRY_RUN" "WAIT_FOR_GPUS:$WAIT_FOR_GPUS" \
    "SAMPLE_ENABLE_THINKING:$SAMPLE_ENABLE_THINKING" \
    "SAMPLE_REQUIRE_THINKING:$SAMPLE_REQUIRE_THINKING" \
    "TRAIN_ENABLE_THINKING:$TRAIN_ENABLE_THINKING" \
    "REUSE_JUDGE_SCORES:$REUSE_JUDGE_SCORES" \
    "RL_REUSE_LEGACY_SAMPLED_DATA:$RL_REUSE_LEGACY_SAMPLED_DATA" \
    "ENABLE_THINKING:$ENABLE_THINKING" "REQUIRE_THINKING:$REQUIRE_THINKING" \
    "ALLOW_THINKING_MODE_MISMATCH:$ALLOW_THINKING_MODE_MISMATCH" \
    "EVAL_CHECKPOINT_SWEEP:$EVAL_CHECKPOINT_SWEEP"; do
    require_bool "${pair%%:*}" "${pair#*:}"
  done
  require_positive_int SFT_SAMPLES_PER_AGENT "$SFT_SAMPLES_PER_AGENT"
  (( SFT_SAMPLES_PER_AGENT % 40 == 0 )) || \
    fatal "SFT_SAMPLES_PER_AGENT must be a multiple of 40"
  require_positive_int SFT_ROLLOUT_LIMIT "$SFT_ROLLOUT_LIMIT"
  (( SFT_ROLLOUT_LIMIT % 40 == 0 )) || \
    fatal "SFT_ROLLOUT_LIMIT must be a multiple of 40"
  require_nonnegative_int SFT_ROLLOUT_MAX_MISSING "$SFT_ROLLOUT_MAX_MISSING"
  (( SFT_ROLLOUT_MAX_MISSING < SFT_ROLLOUT_LIMIT )) || \
    fatal "SFT_ROLLOUT_MAX_MISSING must be smaller than SFT_ROLLOUT_LIMIT"
  require_positive_int SFT_ROLLOUT_MAX_CONCURRENCY "$SFT_ROLLOUT_MAX_CONCURRENCY"
  require_positive_int SFT_ROLLOUT_MAX_TRAJECTORY_ATTEMPTS \
    "$SFT_ROLLOUT_MAX_TRAJECTORY_ATTEMPTS"
  require_positive_int SFT_ROLLOUT_LAUNCH_ATTEMPTS "$SFT_ROLLOUT_LAUNCH_ATTEMPTS"
  require_probability SFT_CORRECTION_FRACTION "$SFT_CORRECTION_FRACTION"
  require_positive_number REFERENCE_COEF "$REFERENCE_COEF"
  require_positive_number PROTOCOL_REFERENCE_COEF "$PROTOCOL_REFERENCE_COEF"
  require_positive_number PROTOCOL_REFERENCE_MARGIN "$PROTOCOL_REFERENCE_MARGIN"
  require_positive_number NEGATIVE_MARGIN "$NEGATIVE_MARGIN"
  [[ "$NEGATIVE_MASK_POLICY" == "role_aware" ]] || \
    fatal "NEGATIVE_MASK_POLICY must remain role_aware"
  require_positive_int NUM_ROLLOUTS "$NUM_ROLLOUTS"
  [[ "$NUM_ROLLOUTS" == "8" ]] || fatal "NUM_ROLLOUTS must remain 8"
  [[ "$SAMPLE_START_AGENT" == "A1" ]] || fatal "training rollout must start at A1"
  [[ "$SAMPLE_ENABLE_THINKING" == "$TRAIN_ENABLE_THINKING" ]] || \
    fatal "sampling and RL target thinking modes must match"
  [[ "$SAMPLE_REQUIRE_THINKING" != "1" || "$SAMPLE_ENABLE_THINKING" == "1" ]] || \
    fatal "SAMPLE_REQUIRE_THINKING=1 requires SAMPLE_ENABLE_THINKING=1"
  [[ "$REQUIRE_THINKING" != "1" || "$ENABLE_THINKING" == "1" ]] || \
    fatal "REQUIRE_THINKING=1 requires ENABLE_THINKING=1"
  if [[ "$ALLOW_THINKING_MODE_MISMATCH" != "1" && \
        "$ENABLE_THINKING" != "$TRAIN_ENABLE_THINKING" ]]; then
    fatal "evaluation and training thinking modes differ"
  fi
  [[ "$SAMPLE_JSON_TRANSPORT" == "json_object" ]] || \
    fatal "SAMPLE_JSON_TRANSPORT must be json_object"
  require_positive_int NUM_GPUS "$NUM_GPUS"
  require_positive_int EVAL_LIMIT "$EVAL_LIMIT"
  [[ "$EVAL_LIMIT" == "500" ]] || fatal "the comparison contract requires EVAL_LIMIT=500"
  require_path "$BUNDLED_PROJECT_ROOT"
  require_path "$BUNDLED_PYTHONPATH_ROOT"
  require_file "$BUNDLED_PROJECT_ROOT/scripts/math_gpu_guard.sh"
  require_path "$UPSTREAM_MATH_ROOT"
  require_path "$PYTHON_BIN"
  require_path "$MATH_DATA_ROOT"
  require_path "$MODEL_A1"
  require_path "$MODEL_A2"
  require_path "$MODEL_A3"
  require_path "$SFT_ROLLOUT_MODEL"
}

wait_for_idle_gpus() {
  local label="$1"
  [[ "$WAIT_FOR_GPUS" == "1" ]] || return 0
  # shellcheck source=/dev/null
  source "${BUNDLED_PROJECT_ROOT}/scripts/math_gpu_guard.sh"
  gpu_guard_wait_idle "$label" "$GPU_IDS" "$GPU_IDLE_MAX_MEMORY_MIB" \
    "$GPU_IDLE_CHECKS" "$GPU_POLL_SECONDS" "$GPU_IDLE_TIMEOUT_SECONDS" || \
    fatal "GPUs did not become idle before $label"
}

acquire_gpu_lock() {
  [[ "$DRY_RUN" == "1" ]] && return 0
  mkdir -p "$(dirname "$LOCK_PATH")"
  exec 9>"$LOCK_PATH"
  log "waiting for shared GPU lock: $LOCK_PATH"
  flock 9
  log "shared GPU lock acquired"
}

run_or_print() {
  print_command "$@"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

archive_partial_dir() {
  local path="$1"
  [[ -d "$path" ]] || return 0
  local archive_root="${path%/*}/_incomplete"
  local destination="${archive_root}/$(basename "$path").$(date +%Y%m%d_%H%M%S).$$"
  mkdir -p "$archive_root"
  mv -- "$path" "$destination"
  log "archived incomplete run: $destination"
}

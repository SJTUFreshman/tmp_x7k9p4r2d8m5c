#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=common.sh
source "${EXPERIMENT_ROOT}/common.sh"
if [[ "$TAG" == "math_3x4b_v1" ]]; then
  reference_root="${EXPERIMENT_ROOT}/resources/reference_eval_20260919"
  export EVAL_SHARD_FILE="$reference_root/shard_04.jsonl"
  export EVAL_LIMIT=500
  "$PYTHON_BIN" "$reference_root/verify_eval_reference.py" \
    --source-root "$MATH_DATA_ROOT" --reference "$EVAL_SHARD_FILE" \
    --expected-sha256 3d4b0649a1a4f6198ed22b138fb82b31d7339be65997af4175bf9bdcc6343183
fi
validate_static_config

SHARD_BUILDER="${UPSTREAM_MATH_ROOT}/build_eval_shard_root.py"
EVAL_VARIANT="${EXPERIMENT_ROOT}/eval_variant.sh"
SUMMARIZER="${EXPERIMENT_ROOT}/summarize_evals.py"
require_file "$SHARD_BUILDER"
require_file "$EVAL_VARIANT"
require_file "$SUMMARIZER"
require_file "$EVAL_SHARD_FILE"

build_eval_shard() {
  local command=(
    env "PYTHONPATH=$PYTHONPATH_ROOT" "$PYTHON_BIN" -u "$SHARD_BUILDER"
    --source-root "$MATH_DATA_ROOT" --shard-file "$EVAL_SHARD_FILE"
    --output-root "$EVAL_DATA_ROOT" --manifest "$EVAL_DATA_ROOT/manifest.json"
  )
  if [[ -s "$EVAL_DATA_ROOT/manifest.json" ]]; then
    command+=(--validate-existing)
  elif [[ -e "$EVAL_DATA_ROOT" ]]; then
    fatal "partial evaluation shard root exists: $EVAL_DATA_ROOT"
  fi
  print_command "${command[@]}"
  [[ "$DRY_RUN" == "1" ]] || "${command[@]}"
}

run_variant() {
  local name="$1"
  shift
  GPU_LOCK_HELD=1 DRY_RUN="$DRY_RUN" RESUME="$RESUME" \
    bash "$EVAL_VARIANT" "$name" "$@"
}

echo "MATH fixed-shard final RL evaluation"
echo "  shard:       $EVAL_SHARD_FILE"
echo "  count:       $EVAL_LIMIT"
echo "  variant:     rl_all_final only"
echo "  thinking:    disabled for every variant"

if [[ "$DRY_RUN" != "1" ]]; then
  require_adapter "$RL_ROOT/A1/final"
  require_adapter "$RL_ROOT/A2/final"
  require_adapter "$RL_ROOT/A3/final"
fi

build_eval_shard
acquire_gpu_lock
run_variant rl_all_final \
  "$MODEL_A1" "$MODEL_A2" "$MODEL_A3" \
  "$RL_ROOT/A1/final" "$RL_ROOT/A2/final" "$RL_ROOT/A3/final"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] no evaluation was started"
else
  "$PYTHON_BIN" "$SUMMARIZER" --eval-root "$EVAL_ROOT" --expected-count "$EVAL_LIMIT"
  echo "[done] summary: $EVAL_ROOT/summary.md"
fi

#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=common.sh
source "${EXPERIMENT_ROOT}/common.sh"
validate_static_config

NORMALIZER="${EXPERIMENT_ROOT}/normalize_reused_math_rollouts.py"
require_file "$NORMALIZER"
require_file "$REUSED_RAW_SOURCE"

NORMALIZE_CMD=(
  env "PYTHONPATH=$PYTHONPATH_ROOT" "$PYTHON_BIN" -u "$NORMALIZER"
  --input "$REUSED_RAW_SOURCE" --output "$RAW_ROLLOUT"
  --stats-output "$REUSE_NORMALIZATION_STATS" --data-root "$MATH_DATA_ROOT"
  --expected-rollouts "$NUM_ROLLOUTS"
  --sampling-policy-lineage "$RL_SAMPLING_POLICY_LINEAGE"
)

echo "Reused MATH rollout normalization"
echo "  source:          $REUSED_RAW_SOURCE"
echo "  source policy:   $RL_SAMPLING_POLICY_LINEAGE"
echo "  output:          $RAW_ROLLOUT"
if [[ "$REUSE_JUDGE_SCORES" == "1" ]]; then
  echo "  judge reuse:     strict migration from $REUSED_JUDGED_SOURCE"
else
  echo "  judge reuse:     disabled; stage 03 will rescore every group"
fi
print_command "${NORMALIZE_CMD[@]}"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] reused rollout normalization not started"
  exit 0
fi

if [[ "$RESUME" == "1" && -s "$RAW_ROLLOUT" && -s "$REUSE_NORMALIZATION_STATS" ]]; then
  "${NORMALIZE_CMD[@]}" --validate-existing
  echo "[resume] reused raw rollout already normalized and validated"
  exit 0
fi
if [[ -e "$RAW_ROLLOUT" || -e "$REUSE_NORMALIZATION_STATS" ]]; then
  fatal "partial reused-rollout normalization exists; archive it or choose a new TAG"
fi

mkdir -p "$(dirname "$RAW_ROLLOUT")"
"${NORMALIZE_CMD[@]}"
"${NORMALIZE_CMD[@]}" --validate-existing
echo "[done] reused raw rollout is normalized to the current evaluator contract"

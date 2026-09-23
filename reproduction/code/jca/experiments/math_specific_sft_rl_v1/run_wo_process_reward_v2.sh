#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=common.sh
source "${EXPERIMENT_ROOT}/common.sh"
validate_static_config

WOPR_ROOT="${WOPR_ROOT:-${ARTIFACT_ROOT}/07_ablations/wo_process_reward_v2_protocol_floor}"
WOPR_BASE_ROOT="${WOPR_BASE_ROOT:-${WOPR_ROOT}/04_prepare/base_no_process_reward}"
WOPR_BASE_TRAIN="${WOPR_BASE_TRAIN:-${WOPR_BASE_ROOT}/train.jsonl}"
WOPR_BASE_HOLDOUT="${WOPR_BASE_HOLDOUT:-${WOPR_BASE_ROOT}/holdout.jsonl}"
WOPR_BASE_STATS="${WOPR_BASE_STATS:-${WOPR_BASE_ROOT}/stats.json}"
WOPR_BASE_MANIFEST="${WOPR_BASE_MANIFEST:-${WOPR_BASE_ROOT}/manifest.json}"
WOPR_TRAIN_DATA="${WOPR_TRAIN_DATA:-${WOPR_ROOT}/04_prepare/awr/train.jsonl}"
WOPR_HOLDOUT_DATA="${WOPR_HOLDOUT_DATA:-${WOPR_ROOT}/04_prepare/awr/holdout.jsonl}"
WOPR_PREPARE_STATS="${WOPR_PREPARE_STATS:-${WOPR_ROOT}/04_prepare/awr/stats.json}"
WOPR_MANIFEST="${WOPR_MANIFEST:-${WOPR_ROOT}/manifest.json}"
WOPR_RL_ROOT="${WOPR_RL_ROOT:-${WOPR_ROOT}/05_train/adapters}"
WOPR_EVAL_ROOT="${WOPR_EVAL_ROOT:-${WOPR_ROOT}/06_eval}"

REWARD_BUILDER="${UPSTREAM_MATH_ROOT}/build_without_process_reward.py"
MANIFEST_TOOL="${EXPERIMENT_ROOT}/make_wo_process_reward_manifest.py"
ANNOTATOR="${EXPERIMENT_ROOT}/annotate_math_awr.py"
TRAIN_STAGE="${EXPERIMENT_ROOT}/05_train_rl.sh"
EVAL_VARIANT="${EXPERIMENT_ROOT}/eval_variant.sh"
for path in "$REWARD_BUILDER" "$MANIFEST_TOOL" "$ANNOTATOR" "$TRAIN_STAGE" "$EVAL_VARIANT"; do
  require_file "$path"
done

WOPR_SAMPLE_SOURCE="${WOPR_SAMPLE_SOURCE:-${RL_SAMPLE_SOURCE_V4}_wo_process_reward_v2}"
BUILD_CMD=(
  env "PYTHONPATH=$PYTHONPATH_ROOT" "$PYTHON_BIN" -u "$REWARD_BUILDER"
  --train-input "$PREPARE_BASE_TRAIN" --holdout-input "$PREPARE_BASE_HOLDOUT"
  --train-output "$WOPR_BASE_TRAIN" --holdout-output "$WOPR_BASE_HOLDOUT"
  --stats-output "$WOPR_BASE_STATS" --transition-boost "$TRANSITION_BOOST"
  --correct-to-correct-coef-a1 "$CORRECT_TO_CORRECT_COEF_A1"
  --correct-to-correct-coef-a2 "$CORRECT_TO_CORRECT_COEF_A2"
  --correct-to-correct-coef-a3 "$CORRECT_TO_CORRECT_COEF_A3"
)
BASE_MANIFEST_CMD=(
  env "PYTHONPATH=$PYTHONPATH_ROOT" "$PYTHON_BIN" -u "$MANIFEST_TOOL"
  --source-manifest "$PREPARE_BASE_MANIFEST"
  --source-train "$PREPARE_BASE_TRAIN" --source-holdout "$PREPARE_BASE_HOLDOUT"
  --train "$WOPR_BASE_TRAIN" --holdout "$WOPR_BASE_HOLDOUT"
  --stats "$WOPR_BASE_STATS" --output "$WOPR_BASE_MANIFEST"
)
ANNOTATE_CMD=(
  env "PYTHONPATH=$PYTHONPATH_ROOT" "$PYTHON_BIN" -u "$ANNOTATOR"
  --train-input "$WOPR_BASE_TRAIN" --holdout-input "$WOPR_BASE_HOLDOUT"
  --base-stats "$WOPR_BASE_STATS" --base-manifest "$WOPR_BASE_MANIFEST"
  --sft-stats "$SFT_DATA_DIR/stats.json"
  --sft-a1 "$SFT_A1" --sft-a2 "$SFT_A2" --sft-a3 "$SFT_A3"
  --train-output "$WOPR_TRAIN_DATA" --holdout-output "$WOPR_HOLDOUT_DATA"
  --stats-output "$WOPR_PREPARE_STATS" --manifest-output "$WOPR_MANIFEST"
  --sample-source-v4 "$WOPR_SAMPLE_SOURCE"
  --sampling-policy-lineage "$RL_SAMPLING_POLICY_LINEAGE"
)
if [[ "$RL_REUSE_LEGACY_SAMPLED_DATA" == "1" ]]; then
  ANNOTATE_CMD+=(--reuse-legacy-sampled-data)
else
  ANNOTATE_CMD+=(--no-reuse-legacy-sampled-data)
fi

echo "MATH wo_process_reward v2"
echo "  invariant:    paired main data, SFT init/reference, trainer, LR, and protocol floor"
echo "  only change:  process/judge reward weight 0.65 -> 0.0; correctness weight -> 1.0"
echo "  trainer:      conservative signed-AWR v4 with protocol floor"
echo "  reference:    $SFT_RUN_ROOT/{A1,A2,A3}/final"
echo "  output:       $WOPR_ROOT"
print_command "${BUILD_CMD[@]}"
print_command "${BASE_MANIFEST_CMD[@]}"
print_command "${ANNOTATE_CMD[@]}"

if [[ "$DRY_RUN" != "1" ]]; then
  for path in "$PREPARE_BASE_TRAIN" "$PREPARE_BASE_HOLDOUT" \
    "$PREPARE_BASE_STATS" "$PREPARE_BASE_MANIFEST" "$SFT_DATA_DIR/stats.json"; do
    require_file "$path"
  done
  require_adapter "$SFT_A1"
  require_adapter "$SFT_A2"
  require_adapter "$SFT_A3"
  mkdir -p "$WOPR_BASE_ROOT"
  if [[ -s "$WOPR_BASE_STATS" ]]; then
    "${BUILD_CMD[@]}" --validate-existing
  else
    if [[ -e "$WOPR_BASE_TRAIN" || -e "$WOPR_BASE_HOLDOUT" || -e "$WOPR_BASE_STATS" ]]; then
      fatal "partial wo_process_reward base data exists: $WOPR_BASE_ROOT"
    fi
    "${BUILD_CMD[@]}"
  fi
  if [[ -s "$WOPR_BASE_MANIFEST" ]]; then
    "${BASE_MANIFEST_CMD[@]}" --validate-existing
  else
    "${BASE_MANIFEST_CMD[@]}"
  fi
  if [[ -s "$WOPR_MANIFEST" ]]; then
    "${ANNOTATE_CMD[@]}" --validate-existing
  else
    if [[ -e "$WOPR_TRAIN_DATA" || -e "$WOPR_HOLDOUT_DATA" || \
          -e "$WOPR_PREPARE_STATS" || -e "$WOPR_MANIFEST" ]]; then
      fatal "partial wo_process_reward AWR data exists: $WOPR_ROOT/04_prepare/awr"
    fi
    "${ANNOTATE_CMD[@]}"
  fi
else
  echo "[dry-run] reward transformation and annotation are deferred until main artifacts exist"
fi

env \
  "EXPERIMENT_ROOT=$EXPERIMENT_ROOT" "TAG=${TAG}_wo_process_reward_v2" \
  "ARTIFACT_ROOT=$WOPR_ROOT" "LOG_ROOT=${WOPR_ROOT}/logs" \
  "PREPARE_BASE_ROOT=$WOPR_BASE_ROOT" "PREPARE_BASE_TRAIN=$WOPR_BASE_TRAIN" \
  "PREPARE_BASE_HOLDOUT=$WOPR_BASE_HOLDOUT" "PREPARE_BASE_STATS=$WOPR_BASE_STATS" \
  "PREPARE_BASE_MANIFEST=$WOPR_BASE_MANIFEST" \
  "TRAIN_DATA=$WOPR_TRAIN_DATA" "HOLDOUT_DATA=$WOPR_HOLDOUT_DATA" \
  "PREPARE_STATS=$WOPR_PREPARE_STATS" "MANIFEST=$WOPR_MANIFEST" \
  "RL_ROOT=$WOPR_RL_ROOT" "RL_SAMPLE_SOURCE_V4=$WOPR_SAMPLE_SOURCE" \
  "SFT_DATA_DIR=$SFT_DATA_DIR" "SFT_RUN_ROOT=$SFT_RUN_ROOT" \
  "SFT_A1=$SFT_A1" "SFT_A2=$SFT_A2" "SFT_A3=$SFT_A3" \
  "MODEL_A1=$MODEL_A1" "MODEL_A2=$MODEL_A2" "MODEL_A3=$MODEL_A3" \
  "RL_SAMPLING_POLICY_LINEAGE=$RL_SAMPLING_POLICY_LINEAGE" \
  "RL_REUSE_LEGACY_SAMPLED_DATA=$RL_REUSE_LEGACY_SAMPLED_DATA" \
  "REFERENCE_COEF=$REFERENCE_COEF" \
  "PROTOCOL_REFERENCE_COEF=$PROTOCOL_REFERENCE_COEF" \
  "PROTOCOL_REFERENCE_MARGIN=$PROTOCOL_REFERENCE_MARGIN" \
  "NEGATIVE_MARGIN=$NEGATIVE_MARGIN" "NEGATIVE_MASK_POLICY=$NEGATIVE_MASK_POLICY" \
  "RL_LR=$RL_LR" "RL_NUM_EPOCHS=$RL_NUM_EPOCHS" \
  "RESUME=$RESUME" "DRY_RUN=$DRY_RUN" \
  bash "$TRAIN_STAGE"

env \
  "EXPERIMENT_ROOT=$EXPERIMENT_ROOT" "TAG=${TAG}_wo_process_reward_v2" \
  "ARTIFACT_ROOT=$ARTIFACT_ROOT" "EVAL_ROOT=$WOPR_EVAL_ROOT" \
  "RL_ROOT=$WOPR_RL_ROOT" "RESUME=$RESUME" "DRY_RUN=$DRY_RUN" \
  bash "$EVAL_VARIANT" wo_process_reward_v2 \
  "$MODEL_A1" "$MODEL_A2" "$MODEL_A3" \
  "$WOPR_RL_ROOT/A1/final" "$WOPR_RL_ROOT/A2/final" "$WOPR_RL_ROOT/A3/final"

echo "[done] wo_process_reward v2 training and fixed-shard evaluation complete"

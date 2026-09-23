#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=common.sh
source "${EXPERIMENT_ROOT}/common.sh"
validate_static_config

TRAINER="${PROJECT_ROOT}/scripts/rl_train_signed_awr_v4.py"
CONTRACT_TOOL="${EXPERIMENT_ROOT}/training_contract.py"
ANNOTATOR="${EXPERIMENT_ROOT}/annotate_math_awr.py"
require_file "$TRAINER"
require_file "$CONTRACT_TOOL"
if [[ "$DRY_RUN" != "1" ]]; then
  require_file "$TRAIN_DATA"
  require_file "$HOLDOUT_DATA"
  require_file "$PREPARE_STATS"
  require_file "$MANIFEST"
  require_adapter "$SFT_A1"
  require_adapter "$SFT_A2"
  require_adapter "$SFT_A3"
  ANNOTATE_VALIDATE_CMD=(
    env "PYTHONPATH=$PYTHONPATH_ROOT" "$PYTHON_BIN" "$ANNOTATOR"
    --train-input "$PREPARE_BASE_TRAIN" --holdout-input "$PREPARE_BASE_HOLDOUT" \
    --base-stats "$PREPARE_BASE_STATS" --base-manifest "$PREPARE_BASE_MANIFEST" \
    --sft-stats "$SFT_DATA_DIR/stats.json" \
    --sft-a1 "$SFT_A1" --sft-a2 "$SFT_A2" --sft-a3 "$SFT_A3" \
    --train-output "$TRAIN_DATA" --holdout-output "$HOLDOUT_DATA" \
    --stats-output "$PREPARE_STATS" --manifest-output "$MANIFEST" \
    --sample-source-v4 "$RL_SAMPLE_SOURCE_V4" \
    --sampling-policy-lineage "$RL_SAMPLING_POLICY_LINEAGE"
  )
  if [[ "$RL_REUSE_LEGACY_SAMPLED_DATA" == "1" ]]; then
    ANNOTATE_VALIDATE_CMD+=(--reuse-legacy-sampled-data)
  else
    ANNOTATE_VALIDATE_CMD+=(--no-reuse-legacy-sampled-data)
  fi
  ANNOTATE_VALIDATE_CMD+=(--validate-existing)
  "${ANNOTATE_VALIDATE_CMD[@]}"
fi
[[ "$MIXED_PRECISION" == "bf16" ]] || \
  fatal "conservative signed-AWR currently requires MIXED_PRECISION=bf16"
require_positive_int RL_MAX_ATTEMPTS "$RL_MAX_ATTEMPTS"

PARAMETERS="$($PYTHON_BIN - "$REFERENCE_COEF" "$PROTOCOL_REFERENCE_COEF" \
  "$PROTOCOL_REFERENCE_MARGIN" "$NEGATIVE_MARGIN" \
  "$NEGATIVE_MASK_POLICY" "$DANGEROUS_DECISION_COEF" \
  "$POSITIVE_STOP_DECISION_COEF" "$RL_NUM_EPOCHS" "$RL_LR" \
  "$RL_PER_DEVICE_BATCH" "$RL_GRAD_ACCUM" "$RL_MAX_SEQ" \
  "$RL_SAVE_STEPS" "$RL_SAVE_TOTAL_LIMIT" <<'PY'
import json
import sys

(
    reference,
    protocol_reference,
    protocol_margin,
    margin,
    mask,
    dangerous,
    positive_stop,
    epochs,
    lr,
    batch,
    accum,
    max_seq,
    save_steps,
    save_limit,
) = sys.argv[1:]
print(json.dumps({
    "algorithm": "conservative_signed_awr_v4_protocol_floor_v1",
    "reference_coef": float(reference),
    "protocol_reference_coef": float(protocol_reference),
    "protocol_reference_margin": float(protocol_margin),
    "negative_margin": float(margin),
    "negative_mask_policy": mask,
    "dangerous_decision_coef": float(dangerous),
    "positive_stop_decision_coef": float(positive_stop),
    "num_epochs": int(epochs),
    "learning_rate": float(lr),
    "per_device_batch": int(batch),
    "gradient_accumulation": int(accum),
    "max_seq_length": int(max_seq),
    "save_steps": int(save_steps),
    "save_total_limit": int(save_limit),
}, sort_keys=True, separators=(",", ":")))
PY
)"

contract_command() {
  local agent="$1" model="$2" adapter="$3" output="$4"
  CONTRACT_CMD=(
    "$PYTHON_BIN" "$CONTRACT_TOOL"
    --output "$output/training_contract.json"
    --agent "$agent" --model "$model" --sft-adapter "$adapter"
    --train "$TRAIN_DATA" --holdout "$HOLDOUT_DATA"
    --prepare-manifest "$MANIFEST" --parameters "$PARAMETERS"
  )
}

validate_final() {
  local agent="$1" output="$2"
  require_adapter "$output/final"
  require_file "$output/final/rl_train_summary.json"
  "$PYTHON_BIN" - "$output/final/rl_train_summary.json" "$agent" \
    "$REFERENCE_COEF" "$PROTOCOL_REFERENCE_COEF" \
    "$PROTOCOL_REFERENCE_MARGIN" <<'PY'
import json
import math
import sys

(
    path,
    agent,
    expected_reference_text,
    expected_protocol_reference_text,
    expected_protocol_margin_text,
) = sys.argv[1:]
value = json.load(open(path, encoding="utf-8"))
if (
    value.get("agent") != agent
    or value.get("algorithm") != "conservative_signed_awr_v4_protocol_floor_v1"
):
    raise SystemExit("RL summary contract mismatch")
if int(value.get("positive_records", 0)) <= 0 or int(value.get("negative_records", 0)) <= 0:
    raise SystemExit("RL summary lacks both reward signs")
if not math.isclose(float(value.get("reference_coef", -1)), float(expected_reference_text)):
    raise SystemExit("RL summary reference coefficient mismatch")
if not math.isclose(
    float(value.get("protocol_reference_coef", -1)),
    float(expected_protocol_reference_text),
):
    raise SystemExit("RL summary protocol reference coefficient mismatch")
if not math.isclose(
    float(value.get("protocol_reference_margin", -1)),
    float(expected_protocol_margin_text),
):
    raise SystemExit("RL summary protocol reference margin mismatch")
drift = value.get("validation", {}).get("reference_token_logp_mse")
if not isinstance(drift, (int, float)) or not math.isfinite(drift) or drift < 0:
    raise SystemExit("RL summary lacks a finite reference-drift metric")
validation = value.get("validation", {})
for name in (
    "protocol_reference_floor_loss",
    "protocol_protected_token_fraction",
    "protocol_downward_drift_mean",
    "protocol_downward_drift_rms",
    "protocol_downward_drift_max",
    "protocol_floor_violation_fraction",
):
    metric = validation.get(name)
    if not isinstance(metric, (int, float)) or not math.isfinite(metric) or metric < 0:
        raise SystemExit(f"RL summary lacks finite non-negative {name}")
protected = validation.get("protocol_protected_tokens")
completion = validation.get("protocol_completion_tokens")
if not isinstance(protected, (int, float)) or protected <= 0:
    raise SystemExit("RL summary contains no protected protocol tokens")
if not isinstance(completion, (int, float)) or completion < protected:
    raise SystemExit("RL summary protocol/completion token counts are inconsistent")
if not 0 < validation["protocol_protected_token_fraction"] <= 1:
    raise SystemExit("RL summary protected-token fraction is outside (0, 1]")
if not 0 <= validation["protocol_floor_violation_fraction"] <= 1:
    raise SystemExit("RL summary protocol-floor violation fraction is outside [0, 1]")
PY
}

train_agent() {
  local agent="$1" model="$2" adapter="$3"
  local output="$RL_ROOT/$agent"
  contract_command "$agent" "$model" "$adapter" "$output"
  if adapter_complete "$output/final"; then
    "${CONTRACT_CMD[@]}" --validate-existing
    validate_final "$agent" "$output"
    log "[resume] RL $agent already complete"
    return 0
  fi
  if [[ -e "$output" ]]; then
    [[ "$RESUME" == "1" ]] || fatal "partial RL output exists and RESUME=0: $output"
    if [[ "$DRY_RUN" == "1" ]]; then
      log "[dry-run] would archive incomplete RL directory: $output"
    else
      archive_partial_dir "$output"
    fi
  fi

  local launch_args=(--num_processes "$NUM_GPUS" --mixed_precision "$MIXED_PRECISION")
  [[ "$NUM_GPUS" == "1" ]] || launch_args+=(--multi_gpu)
  local command=(
    env "PYTHONPATH=$PYTHONPATH_ROOT" "$ACCELERATE" launch
    "${launch_args[@]}"
    "$TRAINER"
    --agent "$agent" --rollout "$TRAIN_DATA" --validation-rollout "$HOLDOUT_DATA"
    --sft-adapter "$adapter" --out-dir "$output" --model-name-or-path "$model"
    --advantage-field train_weight --mask-field loss_mask_mode_v4
    --negative-mask-policy "$NEGATIVE_MASK_POLICY"
    --dangerous-decision-coef "$DANGEROUS_DECISION_COEF"
    --positive-stop-decision-coef "$POSITIVE_STOP_DECISION_COEF"
    --reference-coef "$REFERENCE_COEF" --negative-margin "$NEGATIVE_MARGIN"
    --protocol-reference-coef "$PROTOCOL_REFERENCE_COEF"
    --protocol-reference-margin "$PROTOCOL_REFERENCE_MARGIN"
    --num-epochs "$RL_NUM_EPOCHS" --learning-rate "$RL_LR"
    --per-device-batch-size "$RL_PER_DEVICE_BATCH"
    --gradient-accumulation-steps "$RL_GRAD_ACCUM"
    --max-seq-length "$RL_MAX_SEQ" --seed "$SEED"
    --logging-steps "$RL_LOGGING_STEPS" --save-steps "$RL_SAVE_STEPS"
    --save-total-limit "$RL_SAVE_TOTAL_LIMIT" --gradient-checkpointing --bf16
  )
  echo "RL $agent"
  echo "  policy init: $adapter"
  echo "  frozen ref:  $adapter"
  echo "  output:      $output"
  print_command "${command[@]}"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi

  local attempt status
  for ((attempt=1; attempt<=RL_MAX_ATTEMPTS; attempt++)); do
    if [[ -e "$output" ]]; then
      archive_partial_dir "$output"
    fi
    contract_command "$agent" "$model" "$adapter" "$output"
    "${CONTRACT_CMD[@]}"
    wait_for_idle_gpus "MATH conservative-AWR $agent attempt $attempt/$RL_MAX_ATTEMPTS"
    set +e
    "${command[@]}"
    status=$?
    set -e
    if (( status == 0 )); then
      validate_final "$agent" "$output"
      "${CONTRACT_CMD[@]}" --validate-existing
      return 0
    fi
    log "RL $agent attempt $attempt failed with exit code $status"
  done
  fatal "RL $agent failed after $RL_MAX_ATTEMPTS clean attempts"
}

echo "MATH-specific conservative RL training"
echo "  data:      $TRAIN_DATA"
echo "  objective: signed AWR with frozen identical MATH-SFT reference"
echo "  drift reg: sampled-token MSE coefficient=$REFERENCE_COEF"
echo "  protocol:  one-sided SFT log-prob floor coefficient=$PROTOCOL_REFERENCE_COEF margin=$PROTOCOL_REFERENCE_MARGIN"
echo "  recovery:  interrupted attempts are archived; retries restart cleanly from MATH SFT"
echo "  adapters:  $RL_ROOT/{A1,A2,A3}/final"

acquire_gpu_lock
train_agent A1 "$MODEL_A1" "$SFT_A1"
train_agent A2 "$MODEL_A2" "$SFT_A2"
train_agent A3 "$MODEL_A3" "$SFT_A3"
echo "[done] all MATH RL adapters completed with immutable MATH-SFT contracts"

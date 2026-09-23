#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
RUN_ID="${RUN_ID:-math_sas14b_self_rl_dense_2100_2200_20260908}"
RUN_DIR="${RUN_DIR:-$EXPERIMENT_ROOT/artifacts/14b_self_rl/$RUN_ID}"
SOURCE_RUN="${SOURCE_RUN:-$EXPERIMENT_ROOT/artifacts/14b_self_rl/math_sas14b_self_rl_step_sweep_2000_3654_20260907}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-$SOURCE_RUN/03_train/adapter/checkpoint-2100}"
JUDGED_ROLLOUT="${JUDGED_ROLLOUT:-$EXPERIMENT_ROOT/artifacts/14b_self_rl/math_sas14b_self_rl_full_20260831_1313/02_score/train_judged.jsonl}"
TRAIN_OUT="${TRAIN_OUT:-$RUN_DIR/03_train/adapter}"
TRAIN_GPU_IDS="${TRAIN_GPU_IDS:?TRAIN_GPU_IDS is required}"
EVAL_GPU_IDS="${EVAL_GPU_IDS:?EVAL_GPU_IDS is required}"
TRAIN_NUM_GPUS="${TRAIN_NUM_GPUS:-2}"
EVAL_NUM_GPUS="${EVAL_NUM_GPUS:?EVAL_NUM_GPUS is required}"
DENSE_START_STEP="${DENSE_START_STEP:-2100}"
DENSE_END_STEP="${DENSE_END_STEP:-2200}"
DENSE_STEP_INTERVAL="${DENSE_STEP_INTERVAL:-10}"
EVAL_CONCURRENCY="${EVAL_CONCURRENCY:-64}"
TARGET_EM_MIN="${TARGET_EM_MIN:-0.74}"
TARGET_EM_MAX="${TARGET_EM_MAX:-0.78}"
RUNTIME_ENV_ROOT="${RUNTIME_ENV_ROOT:-/data/home/scwb515/.conda/envs/swift}"
PYTHON_BIN="$RUNTIME_ENV_ROOT/bin/python"

fatal() {
  echo "[fatal] $*" >&2
  exit 1
}

complete_checkpoint() {
  local checkpoint="$1"
  [[ -s "$checkpoint/training_state.json" && \
     -s "$checkpoint/optimizer.bin" && \
     -s "$checkpoint/custom_checkpoint_0.pkl" && \
     -s "$checkpoint/policy_adapter/adapter_model.safetensors" ]]
}

[[ "$DENSE_START_STEP" == "2100" ]] || fatal "DENSE_START_STEP must remain 2100"
[[ "$DENSE_END_STEP" == "2200" ]] || fatal "DENSE_END_STEP must remain 2200"
[[ "$DENSE_STEP_INTERVAL" == "10" ]] || fatal "DENSE_STEP_INTERVAL must remain 10"
[[ "$TRAIN_NUM_GPUS" == "2" ]] || fatal "training must use two GPUs to preserve the checkpoint contract"
[[ -x "$PYTHON_BIN" ]] || fatal "missing Python runtime: $PYTHON_BIN"
complete_checkpoint "$SOURCE_CHECKPOINT" || fatal "incomplete source checkpoint: $SOURCE_CHECKPOINT"
[[ -s "$JUDGED_ROLLOUT" ]] || fatal "missing judged rollout: $JUDGED_ROLLOUT"

marker="$RUN_DIR/.dense_2100_2200_v1"
if [[ -e "$RUN_DIR" && ! -f "$marker" ]]; then
  fatal "refusing to reuse an unmarked run directory: $RUN_DIR"
fi
mkdir -p "$RUN_DIR/03_train" "$RUN_DIR/04_eval"
if [[ ! -e "$marker" ]]; then
  printf '%s\n' \
    "source_checkpoint=$(readlink -f "$SOURCE_CHECKPOINT")" \
    "judged_rollout=$(readlink -f "$JUDGED_ROLLOUT")" \
    "start_step=$DENSE_START_STEP" \
    "end_step=$DENSE_END_STEP" \
    "interval=$DENSE_STEP_INTERVAL" >"$marker"
fi
exec > >(tee -a "$RUN_DIR/dense_sweep.log") 2>&1

echo "========== MATH SELF-RL DENSE 2100-2200 SWEEP =========="
echo "run=$RUN_DIR"
echo "source_checkpoint=$SOURCE_CHECKPOINT"
echo "training_checkpoints=2110,2120,...,2200"
echo "evaluation_checkpoints=2100,2110,...,2200"
echo "target_em=[$TARGET_EM_MIN,$TARGET_EM_MAX]"

if complete_checkpoint "$TRAIN_OUT/checkpoint-$DENSE_END_STEP"; then
  echo "[resume] dense training already reached step $DENSE_END_STEP"
else
  env \
    RUN_ID="$RUN_ID" RUN_DIR="$RUN_DIR" TRAIN_OUT="$TRAIN_OUT" \
    JUDGED_ROLLOUT="$JUDGED_ROLLOUT" TRAIN_RESUME_CHECKPOINT="$SOURCE_CHECKPOINT" \
    GPU_IDS="$TRAIN_GPU_IDS" CUDA_VISIBLE_DEVICES="$TRAIN_GPU_IDS" \
    NUM_GPUS=2 DP_SIZE=2 TP_SIZE=1 \
    NUM_EPOCHS=1 KL_COEF=0.1 LR=3e-6 WARMUP_RATIO=0.05 \
    PER_DEVICE_BATCH=1 GRAD_ACCUM=8 AUTO_SCALE_GRAD_ACCUM=0 TARGET_EFFECTIVE_BATCH=16 \
    TRAIN_MAX_SEQ=8192 LORA_RANK=64 LORA_ALPHA=64 LORA_DROPOUT=0.05 \
    GENERATION_SEED=42 TRAIN_SAVE_STEPS=10 TRAIN_SAVE_TOTAL_LIMIT=12 \
    MAX_TRAIN_STEPS=2200 RESUME=1 ALLOW_WORLD_SIZE_CHANGE_RESUME=0 \
    GPU_GUARD_USE_ALLOCATION=1 WAIT_FOR_IDLE_GPUS=0 \
    bash "$SCRIPT_DIR/02_train.sh"
fi

step=$((DENSE_START_STEP + DENSE_STEP_INTERVAL))
while (( step <= DENSE_END_STEP )); do
  complete_checkpoint "$TRAIN_OUT/checkpoint-$step" || \
    fatal "missing complete dense checkpoint: $TRAIN_OUT/checkpoint-$step"
  step=$((step + DENSE_STEP_INTERVAL))
done

results="$RUN_DIR/04_eval/dense_sweep_results.tsv"
if [[ ! -s "$results" ]]; then
  printf 'step\tem\tmath_soft_f1\tnumeric_soft_f1\tvalid\tprotocol_failures\tsummary\n' >"$results"
fi

validate_summary() {
  local step="$1"
  local summary="$2"
  "$PYTHON_BIN" - "$step" "$summary" <<'PY'
import json
import sys
from pathlib import Path

step, summary_path = sys.argv[1:]
summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
expected = {
    "phase": "rollout",
    "math_eval_version": "math_eval_symbolic_v5",
    "split": "test",
    "problems": 500,
    "expected": 500,
    "written": 500,
    "missing": 0,
    "denominator": 500,
}
for key, value in expected.items():
    if summary.get(key) != value:
        raise SystemExit(
            f"step {step}: invalid summary field {key}={summary.get(key)!r}, expected {value!r}"
        )
config = summary.get("config") or {}
if config.get("temperature") != 0.0:
    raise SystemExit(f"step {step}: evaluation temperature is not zero")
if config.get("enable_thinking") is not False or config.get("require_thinking") is not False:
    raise SystemExit(f"step {step}: evaluation thinking contract changed")
if not str(config.get("problem_ids_file", "")).endswith("/test_10x500_seed42/shard_04.jsonl"):
    raise SystemExit(f"step {step}: evaluation shard changed")
PY
}

record_result() {
  local step="$1"
  local summary="$2"
  "$PYTHON_BIN" - "$step" "$summary" "$results" <<'PY'
import json
import sys
from pathlib import Path

step, summary_path, results_path = sys.argv[1:]
summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
row = "\t".join(
    [
        step,
        str(summary["em"]),
        str(summary["math_soft_f1"]),
        str(summary["numeric_soft_f1"]),
        str(summary["valid"]),
        str(summary["protocol_failures"]),
        summary_path,
    ]
)
path = Path(results_path)
lines = path.read_text(encoding="utf-8").splitlines()
lines = [line for line in lines if not line.startswith(f"{step}\t")]
lines.append(row)
header, body = lines[0], lines[1:]
body.sort(key=lambda line: int(line.split("\t", 1)[0]))
path.write_text("\n".join([header, *body]) + "\n", encoding="utf-8")
print(
    f"[result] step={step} em={summary['em']:.3f} "
    f"math_soft_f1={summary['math_soft_f1']:.3f} "
    f"valid={summary['valid']} protocol_failures={summary['protocol_failures']}"
)
PY
}

evaluate_step() {
  local step="$1"
  local adapter
  if (( step == DENSE_START_STEP )); then
    adapter="$SOURCE_CHECKPOINT/policy_adapter"
  else
    adapter="$TRAIN_OUT/checkpoint-$step/policy_adapter"
  fi
  local output="$RUN_DIR/04_eval/step${step}_test.jsonl"
  local summary="$RUN_DIR/04_eval/step${step}_summary.json"
  [[ -s "$adapter/adapter_model.safetensors" ]] || fatal "missing adapter for step $step: $adapter"
  if [[ ! -s "$summary" ]]; then
    env \
      RUN_ID="$RUN_ID" RUN_DIR="$RUN_DIR" TRAIN_OUT="$TRAIN_OUT" \
      GPU_IDS="$EVAL_GPU_IDS" CUDA_VISIBLE_DEVICES="$EVAL_GPU_IDS" \
      NUM_GPUS="$EVAL_NUM_GPUS" DP_SIZE="$EVAL_NUM_GPUS" TP_SIZE=1 \
      EVAL_VARIANT=rl EVAL_TAG="step${step}" EVAL_ADAPTER="$adapter" \
      EVAL_OUTPUT="$output" EVAL_STATS="$summary" \
      EVAL_START=0 EVAL_LIMIT=500 EVAL_CONCURRENCY="$EVAL_CONCURRENCY" \
      EVAL_TEMPERATURE=0.0 EVAL_TOP_P=0.95 EVAL_MAX_NEW_TOKENS=8192 \
      EVAL_ENABLE_THINKING=0 EVAL_REQUIRE_THINKING=0 \
      GENERATION_SEED=42 PROTOCOL_RETRIES=4 RESUME=1 \
      GPU_GUARD_USE_ALLOCATION=1 WAIT_FOR_IDLE_GPUS=0 \
      bash "$SCRIPT_DIR/03_eval.sh"
  fi
  validate_summary "$step" "$summary"
  record_result "$step" "$summary"
}

source_start_summary="$SOURCE_RUN/04_eval/step${DENSE_START_STEP}_summary.json"
[[ -s "$source_start_summary" ]] || fatal "missing source endpoint summary: $source_start_summary"
validate_summary "$DENSE_START_STEP" "$source_start_summary"
record_result "$DENSE_START_STEP" "$source_start_summary"

step=$((DENSE_START_STEP + DENSE_STEP_INTERVAL))
while (( step <= DENSE_END_STEP )); do
  evaluate_step "$step"
  step=$((step + DENSE_STEP_INTERVAL))
done

"$PYTHON_BIN" - "$results" "$RUN_DIR/04_eval/target_steps.txt" "$TARGET_EM_MIN" "$TARGET_EM_MAX" <<'PY'
import csv
import sys
from pathlib import Path

results_path, target_path, lower, upper = sys.argv[1:]
lower, upper = float(lower), float(upper)
with Path(results_path).open(encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle, delimiter="\t"))
targets = [row["step"] for row in rows if lower <= float(row["em"]) <= upper]
Path(target_path).write_text("".join(f"{step}\n" for step in targets), encoding="utf-8")
print(f"[target] steps={','.join(targets) if targets else 'none'} range=[{lower},{upper}]")
PY

echo "[done] dense checkpoint and evaluation sweep complete: $results"

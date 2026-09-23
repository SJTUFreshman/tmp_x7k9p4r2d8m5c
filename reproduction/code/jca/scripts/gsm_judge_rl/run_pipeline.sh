#!/usr/bin/env bash
# Fixed-A1 GSM offline RL: successful trajectory mix + signed RWR.
#
# This is intentionally a simple, teacher-free pipeline.  The data builder
# selects complete trajectories in one shared four-class mix, then each agent
# is trained independently with the same scripts/rl_train.py objective and
# hyperparameters.  Evaluation always starts at A1 and never selects an
# adapter based on the dev result.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/wangyuheng/jca}"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/qwen35/bin/python}"
PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-/data/wangyuheng}"
PYTHON_BIN_DIR="${PYTHON_BIN%/*}"
ACCELERATE="${ACCELERATE:-${PYTHON_BIN_DIR}/accelerate}"
PYTHON_LAUNCHER="${PYTHON_LAUNCHER:-${PROJECT_ROOT}/gsm/scripts/prepare_gsm_judge_rl_v5_success_data.py}"
TRAIN_LAUNCHER="${TRAIN_LAUNCHER:-${PROJECT_ROOT}/scripts/rl_train.sh}"
EVAL_LAUNCHER="${EVAL_LAUNCHER:-${PROJECT_ROOT}/scripts/run_gsm_sft_eval_fixed_a1.sh}"
SUMMARY_SCRIPT="${SUMMARY_SCRIPT:-${PROJECT_ROOT}/gsm/scripts/summarize_gsm_agent_ablation.py}"

SFT_TAG="${SFT_TAG:-gsm_fixed_a1_corr30_20260723_v3}"
SFT_ROOT="${SFT_ROOT:-${PROJECT_ROOT}/sft_runs/${SFT_TAG}}"
TAG="${TAG:-${SFT_TAG}_judge_rl_v5_success}"
SOURCE_INPUT="${SOURCE_INPUT:-${PROJECT_ROOT}/rl_data/gsm/gsm_fixed_a1_corr30_20260723_v3_judge_rl_full1187_v4_rejudged.jsonl}"

TRAIN_DATA="${TRAIN_DATA:-${PROJECT_ROOT}/rl_data/gsm/${TAG}_train.jsonl}"
HOLDOUT_DATA="${HOLDOUT_DATA:-${PROJECT_ROOT}/rl_data/gsm/${TAG}_holdout.jsonl}"
DATA_STATS="${DATA_STATS:-${PROJECT_ROOT}/rl_data/gsm/${TAG}_stats.json}"
RL_RUNS_DIR="${RL_RUNS_DIR:-${PROJECT_ROOT}/rl_runs/${TAG}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/gsm_eval}"
BASELINE_OUTPUT="${BASELINE_OUTPUT:-${OUTPUT_DIR}/${TAG}_SFT_baseline_dev132.jsonl}"
A1_ONLY_OUTPUT="${A1_ONLY_OUTPUT:-${OUTPUT_DIR}/${TAG}_A1_only_dev132.jsonl}"
A2_ONLY_OUTPUT="${A2_ONLY_OUTPUT:-${OUTPUT_DIR}/${TAG}_A2_only_dev132.jsonl}"
A3_ONLY_OUTPUT="${A3_ONLY_OUTPUT:-${OUTPUT_DIR}/${TAG}_A3_only_dev132.jsonl}"
ALL_AGENTS_OUTPUT="${ALL_AGENTS_OUTPUT:-${OUTPUT_DIR}/${TAG}_all_agents_dev132.jsonl}"
REPORT_OUTPUT="${REPORT_OUTPUT:-${OUTPUT_DIR}/${TAG}_agent_ablation_report.json}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${PROJECT_ROOT}/logs/gsm_judge_rl/${TAG}}"
EVAL_LOG_ROOT="${EVAL_LOG_ROOT:-${PROJECT_ROOT}/logs/gsm_eval/${TAG}}"
RUN_LOG="${RUN_LOG_DIR}/pipeline.log"

EXPECTED_ROLLOUTS="${EXPECTED_ROLLOUTS:-8}"
HOLDOUT_FRACTION="${HOLDOUT_FRACTION:-0.1}"
MIX_A1_CORRECT="${MIX_A1_CORRECT:-0.30}"
MIX_NO_CORRECTION="${MIX_NO_CORRECTION:-0.20}"
MIX_CORRECTION_SUCCESS="${MIX_CORRECTION_SUCCESS:-0.35}"
MIX_CORRECTION_FAILED="${MIX_CORRECTION_FAILED:-0.15}"
MAX_TRAIN_TRAJECTORIES="${MAX_TRAIN_TRAJECTORIES:-0}"
MAX_HOLDOUT_TRAJECTORIES="${MAX_HOLDOUT_TRAJECTORIES:-0}"
BASE_REWARD_WEIGHT="${BASE_REWARD_WEIGHT:-0.35}"
JUDGE_REWARD_WEIGHT="${JUDGE_REWARD_WEIGHT:-0.65}"
TRANSITION_BOOST="${TRANSITION_BOOST:-1.25}"
MIN_ALIGNED_JUDGE="${MIN_ALIGNED_JUDGE:-0.0}"
CORRECT_TO_CORRECT_COEF="${CORRECT_TO_CORRECT_COEF:-1.0}"
CORRECT_TO_CORRECT_COEF_A1="${CORRECT_TO_CORRECT_COEF_A1:-}"
CORRECT_TO_CORRECT_COEF_A2="${CORRECT_TO_CORRECT_COEF_A2:-}"
CORRECT_TO_CORRECT_COEF_A3="${CORRECT_TO_CORRECT_COEF_A3:-}"
CORRECT_TO_CORRECT_HANDOFF_COEF="${CORRECT_TO_CORRECT_HANDOFF_COEF:-1.0}"
WRONG_TO_CORRECT_MULTIPLIER="${WRONG_TO_CORRECT_MULTIPLIER:-}"
CORRECT_TO_WRONG_MULTIPLIER="${CORRECT_TO_WRONG_MULTIPLIER:-}"
DROP_WRONG_TO_WRONG_STOP="${DROP_WRONG_TO_WRONG_STOP:-0}"
SEED="${SEED:-42}"
TRAIN_SEED="${TRAIN_SEED:-42}"
REWARD_FIELD="${REWARD_FIELD:-reward}"

# One objective and one set of hyperparameters for all three agents.
KL_COEF="${KL_COEF:-0.1}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LR="${LR:-3e-6}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
MAX_SEQ="${MAX_SEQ:-8192}"
NUM_GPUS="${NUM_GPUS:-8}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-64}"
WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"
GPU_IDLE_CHECKS="${GPU_IDLE_CHECKS:-2}"

MODEL_A1="${MODEL_A1:-/data/wangyuheng/models/Qwen3-1.7B}"
MODEL_A2="${MODEL_A2:-/data/wangyuheng/models/Qwen3-4B}"
MODEL_A3="${MODEL_A3:-/data/wangyuheng/models/Qwen3-8B}"
SFT_A1="${SFT_A1:-${SFT_ROOT}/A1/final}"
SFT_A2="${SFT_A2:-${SFT_ROOT}/A2/final}"
SFT_A3="${SFT_A3:-${SFT_ROOT}/A3/final}"

RESUME="${RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"
RETRY_EMPTY_AGENT_DIR="${RETRY_EMPTY_AGENT_DIR:-1}"
RUN_EVALS="${RUN_EVALS:-1}"

fatal() {
  echo "[fatal] $*" >&2
  exit 1
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

count_lines() {
  awk 'END { print NR }' "$1"
}

check_eval_complete() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  [[ "$(count_lines "$path")" -eq 132 ]]
}

agent_dir_is_empty_except_launch_log() {
  local dir="$1" entry name
  [[ -d "$dir" ]] || return 1
  for entry in "$dir"/*; do
    [[ -e "$entry" ]] || continue
    name="$(basename "$entry")"
    [[ "$name" == "launch.log" ]] || return 1
  done
  return 0
}

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] || fatal "TAG contains unsupported characters"
[[ "$DRY_RUN" == "0" || "$DRY_RUN" == "1" ]] || fatal "DRY_RUN must be 0 or 1"
[[ "$RESUME" == "0" || "$RESUME" == "1" ]] || fatal "RESUME must be 0 or 1"
[[ "$RETRY_EMPTY_AGENT_DIR" == "0" || "$RETRY_EMPTY_AGENT_DIR" == "1" ]] || fatal "RETRY_EMPTY_AGENT_DIR must be 0 or 1"
[[ "$RUN_EVALS" == "0" || "$RUN_EVALS" == "1" ]] || fatal "RUN_EVALS must be 0 or 1"
[[ "$WAIT_FOR_GPUS" == "0" || "$WAIT_FOR_GPUS" == "1" ]] || fatal "WAIT_FOR_GPUS must be 0 or 1"
[[ "$DROP_WRONG_TO_WRONG_STOP" == "0" || "$DROP_WRONG_TO_WRONG_STOP" == "1" ]] || fatal "DROP_WRONG_TO_WRONG_STOP must be 0 or 1"
[[ "$GPU_POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || fatal "GPU_POLL_SECONDS must be positive"
[[ "$GPU_IDLE_CHECKS" =~ ^[1-9][0-9]*$ ]] || fatal "GPU_IDLE_CHECKS must be positive"
[[ "$SEED" =~ ^[0-9]+$ ]] || fatal "SEED must be a non-negative integer"
[[ "$TRAIN_SEED" =~ ^[0-9]+$ ]] || fatal "TRAIN_SEED must be a non-negative integer"
for path in "$PYTHON_BIN" "$ACCELERATE" "$PYTHON_LAUNCHER" "$TRAIN_LAUNCHER" "$EVAL_LAUNCHER" \
  "$SUMMARY_SCRIPT" "$SOURCE_INPUT" "$MODEL_A1" "$MODEL_A2" "$MODEL_A3" \
  "$SFT_A1" "$SFT_A2" "$SFT_A3"; do
  # A dry-run may intentionally point at a not-yet-generated judge cache;
  # every runtime dependency is still checked for real executions.
  if [[ ! -e "$path" ]]; then
    [[ "$DRY_RUN" == "1" && "$path" == "$SOURCE_INPUT" ]] || \
      fatal "required path not found: $path"
  fi
done
for adapter in "$SFT_A1" "$SFT_A2" "$SFT_A3"; do
  [[ -s "$adapter/adapter_model.safetensors" ]] || fatal "missing SFT adapter: $adapter"
done

PREP_CMD=(
  env "PYTHONPATH=$PYTHONPATH_ROOT" "$PYTHON_BIN" -u "$PYTHON_LAUNCHER"
  --input "$SOURCE_INPUT" --train-output "$TRAIN_DATA"
  --holdout-output "$HOLDOUT_DATA" --stats-output "$DATA_STATS"
  --expected-rollouts-per-problem "$EXPECTED_ROLLOUTS"
  --drop-all-failed-problems
  --holdout-fraction "$HOLDOUT_FRACTION"
  --a1-correct-ratio "$MIX_A1_CORRECT"
  --no-correction-ratio "$MIX_NO_CORRECTION"
  --correction-success-ratio "$MIX_CORRECTION_SUCCESS"
  --correction-failed-ratio "$MIX_CORRECTION_FAILED"
  --max-train-trajectories "$MAX_TRAIN_TRAJECTORIES"
  --max-holdout-trajectories "$MAX_HOLDOUT_TRAJECTORIES"
  --base-reward-weight "$BASE_REWARD_WEIGHT"
  --judge-reward-weight "$JUDGE_REWARD_WEIGHT"
  --transition-boost "$TRANSITION_BOOST"
  --min-aligned-judge "$MIN_ALIGNED_JUDGE"
  --correct-to-correct-coef "$CORRECT_TO_CORRECT_COEF"
  --correct-to-correct-handoff-coef "$CORRECT_TO_CORRECT_HANDOFF_COEF"
  --seed "$SEED" --overwrite
)
if [[ -n "$CORRECT_TO_CORRECT_COEF_A1" ]]; then
  PREP_CMD+=(--correct-to-correct-coef-a1 "$CORRECT_TO_CORRECT_COEF_A1")
fi
if [[ -n "$CORRECT_TO_CORRECT_COEF_A2" ]]; then
  PREP_CMD+=(--correct-to-correct-coef-a2 "$CORRECT_TO_CORRECT_COEF_A2")
fi
if [[ -n "$CORRECT_TO_CORRECT_COEF_A3" ]]; then
  PREP_CMD+=(--correct-to-correct-coef-a3 "$CORRECT_TO_CORRECT_COEF_A3")
fi
if [[ -n "$WRONG_TO_CORRECT_MULTIPLIER" ]]; then
  PREP_CMD+=(--wrong-to-correct-multiplier "$WRONG_TO_CORRECT_MULTIPLIER")
fi
if [[ -n "$CORRECT_TO_WRONG_MULTIPLIER" ]]; then
  PREP_CMD+=(--correct-to-wrong-multiplier "$CORRECT_TO_WRONG_MULTIPLIER")
fi
if [[ "$DROP_WRONG_TO_WRONG_STOP" == "1" ]]; then
  PREP_CMD+=(--drop-wrong-to-wrong-stop)
else
  PREP_CMD+=(--no-drop-wrong-to-wrong-stop)
fi

train_command() {
  local agent="$1"
  TRAIN_CMD=(
    env "PROJECT_ROOT=$PROJECT_ROOT" "PYTHONPATH_ROOT=$PYTHONPATH_ROOT"
    "ROLLOUT=$TRAIN_DATA" "AGENTS=$agent" "TAG=$TAG"
    "RL_RUNS_DIR=$RL_RUNS_DIR" "PYTHON_BIN=$PYTHON_BIN"
    "SFT_A1=$SFT_A1" "SFT_A2=$SFT_A2" "SFT_A3=$SFT_A3"
    "MODEL_A1=$MODEL_A1" "MODEL_A2=$MODEL_A2" "MODEL_A3=$MODEL_A3"
    "KL_COEF=$KL_COEF" "NUM_EPOCHS=$NUM_EPOCHS" "LR=$LR"
    "REWARD_FIELD=$REWARD_FIELD"
    "SEED=$TRAIN_SEED"
    "PER_DEVICE_BATCH=$PER_DEVICE_BATCH" "GRAD_ACCUM=$GRAD_ACCUM"
    "MAX_SEQ=$MAX_SEQ" "NUM_GPUS=$NUM_GPUS"
    "MIXED_PRECISION=$MIXED_PRECISION" "ACCELERATE=$ACCELERATE"
    bash "$TRAIN_LAUNCHER"
  )
}

eval_command() {
  local variant="$1" output="$2" adapter_a1="$3" adapter_a2="$4" adapter_a3="$5"
  EVAL_CMD=(
    env "PROJECT_ROOT=$PROJECT_ROOT" "PYTHON_BIN=$PYTHON_BIN"
    "PYTHONPATH_ROOT=$PYTHONPATH_ROOT" "SFT_TAG=$SFT_TAG"
    "RUN_ID=${TAG}_${variant}_dev132" "OUTPUT_PATH=$output"
    "MAX_CONCURRENCY=$MAX_CONCURRENCY" "MODEL_A1=$MODEL_A1"
    "MODEL_A2=$MODEL_A2" "MODEL_A3=$MODEL_A3"
    "ADAPTER_A1=$adapter_a1" "ADAPTER_A2=$adapter_a2" "ADAPTER_A3=$adapter_a3"
    "LOG_DIR=${EVAL_LOG_ROOT}/${variant}"
    bash "$EVAL_LAUNCHER"
  )
}

run_eval() {
  local output="$1"
  shift
  if check_eval_complete "$output"; then
    echo "[resume] evaluation already complete: $output"
    return
  fi
  if [[ -e "$output" ]]; then
    fatal "existing evaluation is incomplete (expected 132 rows): $output"
  fi
  wait_for_idle_gpus "evaluation $(basename "$output")"
  "$@"
}

wait_for_idle_gpus() {
  local label="$1"
  if [[ "$WAIT_FOR_GPUS" != "1" ]]; then
    echo "[gpu] waiting disabled before $label"
    return
  fi
  local idle_checks=0 gpu_processes
  echo "[gpu] waiting for stable all-GPU idle window before $label"
  while (( idle_checks < GPU_IDLE_CHECKS )); do
    gpu_processes="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits || true)"
    if [[ -z "${gpu_processes//[[:space:]]/}" ]]; then
      idle_checks=$((idle_checks + 1))
      echo "[$(date '+%F %T')] all GPUs idle ($idle_checks/$GPU_IDLE_CHECKS)"
    else
      idle_checks=0
      echo "[$(date '+%F %T')] GPUs occupied; waiting before $label"
    fi
    if (( idle_checks < GPU_IDLE_CHECKS )); then
      sleep "$GPU_POLL_SECONDS"
    fi
  done
}

echo "GSM fixed-A1 offline RL success-mix pipeline"
echo "  tag:              $TAG"
echo "  source:           $SOURCE_INPUT"
echo "  trajectory mix:   a1_correct=$MIX_A1_CORRECT no_correction=$MIX_NO_CORRECTION correction_success=$MIX_CORRECTION_SUCCESS correction_failed=$MIX_CORRECTION_FAILED"
echo "  reward:           base=$BASE_REWARD_WEIGHT judge=$JUDGE_REWARD_WEIGHT transition_boost=$TRANSITION_BOOST aligned_min=$MIN_ALIGNED_JUDGE"
echo "  causal credit:    correct_to_correct_coef=$CORRECT_TO_CORRECT_COEF A1=${CORRECT_TO_CORRECT_COEF_A1:-global} A2=${CORRECT_TO_CORRECT_COEF_A2:-global} A3=${CORRECT_TO_CORRECT_COEF_A3:-global} handoff=$CORRECT_TO_CORRECT_HANDOFF_COEF drop_wrong_to_wrong_stop=$DROP_WRONG_TO_WRONG_STOP"
echo "  asymmetric transition: wrong_to_correct=${WRONG_TO_CORRECT_MULTIPLIER:-legacy} correct_to_wrong=${CORRECT_TO_WRONG_MULTIPLIER:-legacy} reward_field=$REWARD_FIELD"
echo "  objective:        plain signed RWR, full response, reward weights as above, no replay/teacher"
echo "  protocol:         fixed A1 start, temp=0 eval, no A3-start"
echo "  train:            epochs=$NUM_EPOCHS lr=$LR kl=$KL_COEF seed=$TRAIN_SEED gpus=$NUM_GPUS batch=$PER_DEVICE_BATCH accum=$GRAD_ACCUM max_seq=$MAX_SEQ"
echo "  pipeline evals:   $RUN_EVALS"
echo "  prepare command:"
print_command "${PREP_CMD[@]}"

if [[ "$RUN_EVALS" == "1" ]]; then
  eval_command "SFT_baseline" "$BASELINE_OUTPUT" "$SFT_A1" "$SFT_A2" "$SFT_A3"
  echo "  baseline eval command:"
  print_command "${EVAL_CMD[@]}"
fi
for agent in A1 A2 A3; do
  train_command "$agent"
  echo "  $agent train command:"
  print_command "${TRAIN_CMD[@]}"
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] no data writes, training, or evaluation started"
  exit 0
fi

mkdir -p "$RUN_LOG_DIR" "$OUTPUT_DIR" "$RL_RUNS_DIR" "$(dirname "$TRAIN_DATA")"
exec > >(tee -a "$RUN_LOG") 2>&1

echo "[phase 1/9] prepare success-mix trajectory data"
if [[ "$RESUME" == "1" && -s "$TRAIN_DATA" && -s "$HOLDOUT_DATA" && -s "$DATA_STATS" ]]; then
  echo "[resume] data already complete: $TRAIN_DATA"
else
  if [[ "$RESUME" == "1" && ( -e "$TRAIN_DATA" || -e "$HOLDOUT_DATA" || -e "$DATA_STATS" ) ]]; then
    fatal "partial data outputs exist; remove only this tag's partial files or choose a new TAG"
  fi
  "${PREP_CMD[@]}"
fi

if [[ "$RUN_EVALS" == "1" ]]; then
  echo "[phase 2/9] fresh SFT baseline evaluation"
  run_eval "$BASELINE_OUTPUT" "${EVAL_CMD[@]}"
fi

for agent in A1 A2 A3; do
  case "$agent" in
    A1) only_output="$A1_ONLY_OUTPUT"; eval_command A1_only "$only_output" "$RL_RUNS_DIR/A1/final" "$SFT_A2" "$SFT_A3" ;;
    A2) only_output="$A2_ONLY_OUTPUT"; eval_command A2_only "$only_output" "$SFT_A1" "$RL_RUNS_DIR/A2/final" "$SFT_A3" ;;
    A3) only_output="$A3_ONLY_OUTPUT"; eval_command A3_only "$only_output" "$SFT_A1" "$SFT_A2" "$RL_RUNS_DIR/A3/final" ;;
  esac
  echo "[phase] train $agent with the shared plain signed-RWR objective"
  train_command "$agent"
  if [[ "$RESUME" == "1" && -s "$RL_RUNS_DIR/$agent/final/adapter_model.safetensors" ]]; then
    echo "[resume] $agent adapter already complete: $RL_RUNS_DIR/$agent/final"
  else
    if [[ "$RESUME" == "1" && -d "$RL_RUNS_DIR/$agent" ]]; then
      if [[ "$RETRY_EMPTY_AGENT_DIR" == "1" ]] && agent_dir_is_empty_except_launch_log "$RL_RUNS_DIR/$agent"; then
        echo "[retry] $agent directory contains only launch.log; retrying failed startup"
      else
        fatal "partial $agent run exists; choose a new TAG rather than mixing checkpoints"
      fi
    fi
    wait_for_idle_gpus "training $agent"
    "${TRAIN_CMD[@]}"
  fi
  if [[ "$RUN_EVALS" == "1" ]]; then
    echo "[phase] isolated fixed-A1 evaluation for $agent"
    run_eval "$only_output" "${EVAL_CMD[@]}"
  fi
done

if [[ "$RUN_EVALS" == "0" ]]; then
  echo "GSM fixed-A1 offline RL training complete; pipeline evaluations disabled"
  echo "  train data: $TRAIN_DATA"
  echo "  data stats:  $DATA_STATS"
  echo "  adapters:    $RL_RUNS_DIR/{A1,A2,A3}/final"
  echo "  log:         $RUN_LOG"
  exit 0
fi

echo "[phase 8/9] all-agent fixed-A1 evaluation"
eval_command all_agents "$ALL_AGENTS_OUTPUT" \
  "$RL_RUNS_DIR/A1/final" "$RL_RUNS_DIR/A2/final" "$RL_RUNS_DIR/A3/final"
run_eval "$ALL_AGENTS_OUTPUT" "${EVAL_CMD[@]}"

echo "[phase 9/9] attribution report"
SUMMARY_CMD=(
  env "PYTHONPATH=$PYTHONPATH_ROOT" "$PYTHON_BIN" -u "$SUMMARY_SCRIPT"
  --baseline "$BASELINE_OUTPUT" --a1-only "$A1_ONLY_OUTPUT"
  --a2-only "$A2_ONLY_OUTPUT" --a3-only "$A3_ONLY_OUTPUT"
  --all-agents "$ALL_AGENTS_OUTPUT" --output "$REPORT_OUTPUT"
)
"${SUMMARY_CMD[@]}"
echo "GSM fixed-A1 offline RL pipeline complete"
echo "  train data: $TRAIN_DATA"
echo "  data stats:  $DATA_STATS"
echo "  adapters:    $RL_RUNS_DIR/{A1,A2,A3}/final"
echo "  report:      $REPORT_OUTPUT"
echo "  log:         $RUN_LOG"

#!/usr/bin/env bash
# GSM-HARD 3x4B homogeneous team — FULL chain on ParaCloud Zhongwei-1.
#
#   01 sft      correction-SFT distill from the existing corr30 v3 demonstrations
#   02 sfteval  deterministic dev132 eval of the SFT policy
#   03 rollout  second sampling from the NEW 3x4B SFT policy, judged inline by a
#               local Qwen3-14B
#   04 rl       quality gate -> v13 rejudge -> v8 success mix -> RWR -> dev132
#
# Usage:  bash run_gsm_3x4b_cluster.sh 01 02 03 04
#         DRY_RUN=1 bash run_gsm_3x4b_cluster.sh 01
#
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

JCAROOT="${JCAROOT:-/data/run01/scwb515/yangrunde/jcaroot}"
JCA="${JCA:-$JCAROOT/jca}"
ENV_ROOT="${ENV_ROOT:-/data/home/scwb515/.conda/envs/swift}"
M4B="${M4B:-$JCAROOT/models/Qwen3-4B}"
M14B="${M14B:-$JCAROOT/models/Qwen3-14B}"

export SFT_TAG="${SFT_TAG:-gsm_3x4b_v1}"
export TAG="${TAG:-${SFT_TAG}_judge_rl_v1}"

# --- re-root the runtime onto the cluster's conda env ------------------------
export PROJECT_ROOT="$JCA"
export PYTHONPATH_ROOT="$JCAROOT"
export PYTHON_BIN="${PYTHON_BIN:-$ENV_ROOT/bin/python}"
export VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-$ENV_ROOT/bin/python}"
export VLLM_LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH:-$ENV_ROOT/lib}"
export ACCELERATE="${ACCELERATE:-$ENV_ROOT/bin/accelerate}"
export PY="${PY:-$PYTHON_BIN}"
export TRAIN_PROJ_ROOT="${TRAIN_PROJ_ROOT:-$JCAROOT}"

# --- models: homogeneous 3x4B students, 14B judge ----------------------------
export MODEL_A1="$M4B"
export MODEL_A2="$M4B"
export MODEL_A3="$M4B"

SFT_DATA="${SFT_DATA:-$JCA/sft_data/gsm_fixed_a1_corr30_20260723_v3}"
SFT_ROOT="$JCA/sft_runs/$SFT_TAG"
RL_RUNS_ROOT="$JCA/rl_runs/$TAG"

# Under $JCA, not $JCAROOT: run_gsm_judge_rl_end_to_end.sh:16-17 roots these at
# $PROJECT_ROOT, and stages 02/03 have to name the same files it will.
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-$JCA/Math/data/GSM-HARD/splits/gsmhardv2_train.jsonl}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-$JCA/Math/data/GSM-HARD/splits/gsmhardv2_dev.jsonl}"
SFT_EVAL_OUTPUT="$JCA/outputs/gsm_eval/${SFT_TAG}_fixed_a1_dev132.jsonl"
ROLLOUT_FILE="$JCA/rl_data/gsm/${TAG}.jsonl"
RL_EVAL_OUTPUT="$JCA/outputs/gsm_eval/${TAG}_dev132.jsonl"

ROLLOUT_LIMIT="${ROLLOUT_LIMIT:-1187}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-8}"
V13_DATA_ROOT="$JCA/rl_data/gsm/judge_rl/${TAG}_v13_14b_v8"
V13_SOURCE="$V13_DATA_ROOT/source_rejudged.jsonl"
V8_OUTPUT_ROOT="$JCA/outputs/gsm_eval/gsm_judge_rl/${TAG}_v13_14b_v8"

# --- judge sidecar -----------------------------------------------------------
export JUDGE14B_MODEL_PATH="$M14B"
export JUDGE14B_PORT="${JUDGE14B_PORT:-8400}"   # gsm's own port, per rejudge_gsm_qwen14b.sh
export JUDGE14B_GPUS="${JUDGE14B_GPUS:-6,7}"
export JUDGE14B_TP="${JUDGE14B_TP:-2}"
export JUDGE14B_SERVED_NAME="${JUDGE14B_SERVED_NAME:-qwen14b_gsm_judge}"
export JUDGE14B_MAX_MODEL_LEN="${JUDGE14B_MAX_MODEL_LEN:-16384}"
export JUDGE14B_MAX_TOKENS="${JUDGE14B_MAX_TOKENS:-4096}"
# shellcheck source=judge14b.sh
source "$HERE/judge14b.sh"

# --- GPUs --------------------------------------------------------------------
export NUM_GPUS="${NUM_GPUS:-8}"
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-0}"
export GPU_GUARD_USE_ALLOCATION="${GPU_GUARD_USE_ALLOCATION:-1}"

derive_accum() {  # target per_device gpus -> accum (nearest, >= 1)
  local target="$1" per_device="$2" gpus="$3" denom lower upper
  denom=$(( gpus * per_device ))
  lower=$(( target / denom )); (( lower >= 1 )) || lower=1
  upper=$(( lower + 1 ))
  if (( target - lower * denom <= upper * denom - target )); then echo "$lower"; else echo "$upper"; fi
}

fatal() { echo "[fatal] $*" >&2; exit 2; }

# --- preflight ---------------------------------------------------------------
[[ -d "$SFT_DATA" ]] || fatal "missing SFT data dir: $SFT_DATA"
[[ -s "$M4B/config.json"  ]] || fatal "missing model: $M4B"
[[ -s "$M14B/config.json" ]] || fatal "missing judge model: $M14B"
[[ -x "$PYTHON_BIN" ]] || fatal "not executable: $PYTHON_BIN"
[[ -x "$ACCELERATE" ]] || fatal "not executable: $ACCELERATE"
[[ -s "$TRAIN_DATA_PATH" ]] || fatal "missing train split: $TRAIN_DATA_PATH"
[[ -s "$EVAL_DATA_PATH"  ]] || fatal "missing dev split: $EVAL_DATA_PATH"
for s in scripts/run_gsm_correction_sft.sh scripts/run_gsm_sft_eval_fixed_a1.sh \
         scripts/gsm_judge_rl/rejudge_gsm_qwen14b_v13.sh scripts/gsm_judge_rl/run_pipeline.sh \
         gsm/scripts/prepare_gsm_judge_rl_v5_success_data.py gsm/scripts/rejudge_gsm_collaboration_v4.py \
         scripts/validate_gsm_judge_rl_data.py \
         gsm/scripts/run_gsm_judge_rl_rollout_8gpu.sh gsm/scripts/run_gsm_vllm_8gpu.sh; do
  [[ -f "$JCA/$s" ]] || fatal "missing script: $JCA/$s"
done
train_count="$(awk 'NF {count++} END {print count+0}' "$TRAIN_DATA_PATH")"
[[ "$ROLLOUT_LIMIT" == "$train_count" && "$NUM_ROLLOUTS" == "8" ]] || \
  fatal "v13_14b_v8 requires all $train_count train problems x8; got $ROLLOUT_LIMIT x$NUM_ROLLOUTS"
mkdir -p "$JCA/rl_data/gsm" "$JCA/outputs/gsm_eval" "$JCA/logs"

adapters_ready() {
  local root="$1" a
  for a in A1 A2 A3; do
    [[ -s "$root/$a/final/adapter_model.safetensors" ]] || return 1
  done
  return 0
}

# =============================================================================
stage_01() {
  if adapters_ready "$SFT_ROOT"; then
    echo "[skip] 01 sft already complete: $SFT_ROOT"
    return 0
  fi
  local ga
  ga="$(derive_accum 32 1 "$NUM_GPUS")"   # main arm: 8 x 1 x 4 = 32
  echo "[01] correction sft  gpus=$NUM_GPUS per_device=1 grad_accum=$ga"
  cd "$JCA"
  PROJECT_ROOT="$JCA" TRAIN_PROJ_ROOT="$JCAROOT" \
  PY="$PYTHON_BIN" ACCELERATE="$ACCELERATE" \
  SFT_TRAIN_SH="$JCA/scripts/sft_train.sh" \
  DATA_DIR="$SFT_DATA" TAG="$SFT_TAG" \
  MODEL_A1="$M4B" MODEL_A2="$M4B" MODEL_A3="$M4B" \
  AGENTS="${AGENTS:-A1 A2 A3}" \
  NUM_EPOCHS=3 LR=1e-5 EVAL_FRACTION=0.05 \
  PER_DEVICE_BATCH=1 GRAD_ACCUM="$ga" MAX_SEQ=4096 \
  LORA_RANK=64 LORA_ALPHA=64 LORA_DROPOUT=0.05 \
  EARLY_STOP_PATIENCE=3 LOAD_BEST_MODEL_AT_END=0 \
  NUM_GPUS="$NUM_GPUS" MIXED_PRECISION=bf16 REPORT_TO=tensorboard \
  DATALOADER_NUM_WORKERS=0 \
    bash "$JCA/scripts/run_gsm_correction_sft.sh"
  adapters_ready "$SFT_ROOT" || fatal "01 finished but adapters are incomplete under $SFT_ROOT"
}

# =============================================================================
stage_02() {
  adapters_ready "$SFT_ROOT" || fatal "02 needs stage 01's adapters: $SFT_ROOT"
  if [[ -s "$SFT_EVAL_OUTPUT" ]]; then
    echo "[skip] 02 sft dev132 eval already present: $SFT_EVAL_OUTPUT"
    return 0
  fi
  echo "[02] sft dev132 eval (fixed A1 start)"
  cd "$JCA"
  PROJECT_ROOT="$JCA" PYTHONPATH_ROOT="$JCAROOT" PYTHON_BIN="$PYTHON_BIN" \
  SFT_TAG="$SFT_TAG" DATA_PATH="$EVAL_DATA_PATH" \
  OUTPUT_PATH="$SFT_EVAL_OUTPUT" \
  MODEL_A1="$M4B" MODEL_A2="$M4B" MODEL_A3="$M4B" \
  ADAPTER_A1="$SFT_ROOT/A1/final" \
  ADAPTER_A2="$SFT_ROOT/A2/final" \
  ADAPTER_A3="$SFT_ROOT/A3/final" \
    bash "$JCA/scripts/run_gsm_sft_eval_fixed_a1.sh"
  [[ -s "$SFT_EVAL_OUTPUT" ]] || fatal "02 produced no eval at $SFT_EVAL_OUTPUT"
}

# =============================================================================
stage_03() {
  adapters_ready "$SFT_ROOT" || fatal "03 needs stage 01's adapters: $SFT_ROOT"
  [[ "$NUM_GPUS" == "8" ]] || fatal "03 needs all 8 GPUs (3x4B policies at TP=2 + a TP=2 14B judge); got $NUM_GPUS"
  if [[ -s "$ROLLOUT_FILE" ]]; then
    echo "[03] resuming into existing rollout: $ROLLOUT_FILE ($(wc -l <"$ROLLOUT_FILE") rows)"
  fi

  judge14b_start

  echo "[03] rollout train 0..$ROLLOUT_LIMIT x$NUM_ROLLOUTS from the 3x4B SFT policy, judged by Qwen3-14B"
  cd "$JCA"
  SFT_TAG="$SFT_TAG" \
  SFT_A1="$SFT_ROOT/A1/final" SFT_A2="$SFT_ROOT/A2/final" SFT_A3="$SFT_ROOT/A3/final" \
  ADAPTER_A1="$SFT_ROOT/A1/final" ADAPTER_A2="$SFT_ROOT/A2/final" ADAPTER_A3="$SFT_ROOT/A3/final" \
  MODEL_A1="$M4B" MODEL_A2="$M4B" MODEL_A3="$M4B" \
  DATA_PATH="$TRAIN_DATA_PATH" START=0 LIMIT="$ROLLOUT_LIMIT" \
  NUM_ROLLOUTS="$NUM_ROLLOUTS" T_MAX=8 START_AGENT=A1 \
  MIN_AGENTS_BEFORE_STOP=1 ENFORCE_COLLABORATION_POLICY=0 \
  MAX_CONCURRENCY="${ROLLOUT_MAX_CONCURRENCY:-8}" \
  MAX_NEW_TOKENS=1024 TEMPERATURE=0.9 TOP_P=0.95 \
  JUDGE_MODEL="$JUDGE14B_SERVED_NAME" \
  JUDGE_TEMPERATURE=0.0 JUDGE_TOP_P=0.95 \
  JUDGE_MAX_TOKENS="$JUDGE14B_MAX_TOKENS" \
  JUDGE_REASONING_EFFORT=low JUDGE_PARSE_RETRIES=2 ALLOW_JUDGE_FAILURE=0 \
  ALPHA=0.6 TASK_METRIC=em RESUME=1 \
  GROUP_RETRIES=5 STEP_RETRIES=4 \
  A1_GPUS=0,1 A2_GPUS=2,3 A3_GPUS=4,5 A1_TP=2 A2_TP=2 A3_TP=2 \
  MAX_MODEL_LEN=8192 \
  OUTPUT_PATH="$ROLLOUT_FILE" RUN_ID="${TAG}_rollout_judge" \
  LOG_DIR="$JCA/logs/gsm_judge_rl_rollout" \
    bash "$JCA/gsm/scripts/run_gsm_judge_rl_rollout_8gpu.sh"
  [[ -s "$ROLLOUT_FILE" ]] || fatal "03 produced no rollout at $ROLLOUT_FILE"

  # Free 6,7 before stage 04 wants all eight cards.
  judge14b_stop
  echo "[03] judged rollout: $ROLLOUT_FILE ($(wc -l <"$ROLLOUT_FILE") rows)"
}

# =============================================================================
stage_04() {
  [[ -s "$ROLLOUT_FILE" ]] || fatal "04 needs stage 03's judged rollout: $ROLLOUT_FILE"
  [[ -s "$SFT_EVAL_OUTPUT" ]] || fatal "04 needs stage 02's SFT eval: $SFT_EVAL_OUTPUT"
  echo "[04] full-train gate + v13 compact judge + v8 success mix + RWR + dev132 suite"
  cd "$JCA"
  mkdir -p "$V13_DATA_ROOT"
  PYTHONPATH="$JCAROOT" "$PYTHON_BIN" "$JCA/scripts/validate_gsm_judge_rl_data.py" \
    --input "$ROLLOUT_FILE" --expected-problems "$train_count" \
    --num-rollouts 8 --min-records-per-agent "$(((train_count * 8 + 2) / 3))" \
    --alpha 0.6 --task-metric em --expected-start-agent A1 \
    --require-reward-polarity --stats-output "$V13_DATA_ROOT/raw_gate_stats.json"

  local pass_index temperature max_tokens judge_complete=0
  for pass_index in 1 2 3; do
    case "$pass_index" in
      1) temperature=0.0; max_tokens=8192 ;;
      2) temperature=0.0; max_tokens=12288 ;;
      3) temperature=0.6; max_tokens=4096 ;;
    esac
    if PROJECT_ROOT="$JCA" PYTHONPATH_ROOT="$JCAROOT" \
      MODEL_PATH="$M14B" SERVED_MODEL_NAME=qwen14b_gsm_judge \
      INPUT_PATH="$ROLLOUT_FILE" DATA_PATH="$TRAIN_DATA_PATH" \
      V13_14B_SOURCE_OUTPUT="$V13_SOURCE" \
      V13_14B_STATS_OUTPUT="$V13_DATA_ROOT/source_rejudge_stats.json" \
      V13_14B_LOG_DIR="$JCA/logs/gsm_judge_rl/${TAG}_v13_rejudge" \
      V13_14B_RUN_ID="${TAG}_v13_pass${pass_index}" \
      JUDGE_GPUS="$GPU_IDS" JUDGE_TP="$NUM_GPUS" JUDGE_DP=1 \
      MAX_MODEL_LEN=32768 JUDGE_MAX_TOKENS="$max_tokens" \
      JUDGE_TEMPERATURE="$temperature" JUDGE_TOP_P=0.95 \
      JUDGE_CONCURRENCY=16 GROUP_RETRIES=2 JUDGE_PARSE_RETRIES=2 \
      EXPECTED_ROLLOUTS_PER_PROBLEM=8 LIMIT_GROUPS=0 \
      DROP_ALL_FAILED_PROBLEMS=0 ALLOW_PARTIAL_ROLLOUTS=0 \
      PROJECT_CONFLICTING_POSITIVE_SCORES=0 \
      JCA_JUDGE_CONFIDENT_EXTREMES=0 JCA_JUDGE_COMPACT_FALLBACK_DISABLE_THINKING=1 \
      RUNTIME_ROOT_OVERRIDE="${RUNTIME_ROOT:-${TMPDIR:-/tmp}/gsm_${TAG}}/judge_v13" \
      WAIT_FOR_GPUS=0 RESUME=1 AUTO_RESUME_PASSES=1 \
        bash "$JCA/scripts/gsm_judge_rl/rejudge_gsm_qwen14b_v13.sh"; then
      judge_complete=1
      break
    fi
    echo "[04] v13 judge pass $pass_index incomplete; next pass resumes failed groups"
  done
  [[ "$judge_complete" == "1" ]] || fatal "v13 judge failed after all original-recipe retry passes"

  PROJECT_ROOT="$JCA" PYTHONPATH_ROOT="$JCAROOT" \
  SFT_TAG="$SFT_TAG" TAG="$TAG" SFT_ROOT="$SFT_ROOT" \
  SFT_A1="$SFT_ROOT/A1/final" SFT_A2="$SFT_ROOT/A2/final" SFT_A3="$SFT_ROOT/A3/final" \
  MODEL_A1="$M4B" MODEL_A2="$M4B" MODEL_A3="$M4B" \
  SOURCE_INPUT="$V13_SOURCE" TRAIN_DATA="$V13_DATA_ROOT/train.jsonl" \
  HOLDOUT_DATA="$V13_DATA_ROOT/holdout.jsonl" DATA_STATS="$V13_DATA_ROOT/stats.json" \
  RL_RUNS_DIR="$RL_RUNS_ROOT" OUTPUT_DIR="$V8_OUTPUT_ROOT" \
  BASELINE_OUTPUT="$SFT_EVAL_OUTPUT" ALL_AGENTS_OUTPUT="$RL_EVAL_OUTPUT" \
  A1_ONLY_OUTPUT="$V8_OUTPUT_ROOT/a1_only_dev132.jsonl" \
  A2_ONLY_OUTPUT="$V8_OUTPUT_ROOT/a2_only_dev132.jsonl" \
  A3_ONLY_OUTPUT="$V8_OUTPUT_ROOT/a3_only_dev132.jsonl" \
  REPORT_OUTPUT="$V8_OUTPUT_ROOT/agent_ablation_report.json" \
  RUN_LOG_DIR="$JCA/logs/gsm_judge_rl/${TAG}_v13_14b_v8" \
  EVAL_LOG_ROOT="$JCA/logs/gsm_eval/${TAG}_v13_14b_v8" \
  EXPECTED_ROLLOUTS=8 HOLDOUT_FRACTION=0.1 \
  MIX_A1_CORRECT=0.30 MIX_NO_CORRECTION=0.20 MIX_CORRECTION_SUCCESS=0.35 MIX_CORRECTION_FAILED=0.15 \
  MAX_TRAIN_TRAJECTORIES=0 MAX_HOLDOUT_TRAJECTORIES=0 \
  BASE_REWARD_WEIGHT=0.35 JUDGE_REWARD_WEIGHT=0.65 TRANSITION_BOOST=1.25 MIN_ALIGNED_JUDGE=0.0 \
  CORRECT_TO_CORRECT_COEF=0.5 CORRECT_TO_CORRECT_COEF_A1= CORRECT_TO_CORRECT_COEF_A2= \
  CORRECT_TO_CORRECT_COEF_A3= CORRECT_TO_CORRECT_HANDOFF_COEF=1.0 \
  WRONG_TO_CORRECT_MULTIPLIER= CORRECT_TO_WRONG_MULTIPLIER= DROP_WRONG_TO_WRONG_STOP=0 \
  REWARD_FIELD=reward SEED=42 TRAIN_SEED=42 KL_COEF=0.1 NUM_EPOCHS=1 LR=3e-6 \
  PER_DEVICE_BATCH=1 GRAD_ACCUM=2 MAX_SEQ=8192 \
  NUM_GPUS="$NUM_GPUS" MIXED_PRECISION=bf16 \
  DATA_PATH="$EVAL_DATA_PATH" START=0 LIMIT=132 T_MAX=8 START_AGENT=A1 \
  MIN_AGENTS_BEFORE_STOP=1 ENFORCE_COLLABORATION_POLICY=0 \
  MAX_CONCURRENCY=64 MAX_NEW_TOKENS=1024 TEMPERATURE=0.0 TOP_P=0.95 \
  WAIT_FOR_GPUS=0 RUN_EVALS=1 RESUME=1 \
    bash "$JCA/scripts/gsm_judge_rl/run_pipeline.sh"
  adapters_ready "$RL_RUNS_ROOT" || fatal "04 finished but RL adapters are incomplete under $RL_RUNS_ROOT"
  [[ -s "$RL_EVAL_OUTPUT" ]] || fatal "04 produced no eval at $RL_EVAL_OUTPUT"
  echo "[04] results: $RL_EVAL_OUTPUT"
}

# =============================================================================
declare -A STAGE_FN=( [01]=stage_01 [02]=stage_02 [03]=stage_03 [04]=stage_04 )

(( $# > 0 )) || fatal "no stages given (try: 01 02 03 04)"
for s in "$@"; do
  [[ -n "${STAGE_FN[$s]:-}" ]] || fatal "unknown stage: $s"
done

echo "=== gsm 3x4B (cluster) sft_tag=$SFT_TAG tag=$TAG ==="
echo "    students : $M4B x3"
echo "    judge    : $M14B (local vLLM, TP=$JUDGE14B_TP on GPUs $JUDGE14B_GPUS, no gpt-5)"
echo "    sft data : $SFT_DATA"
echo "    rollout  : $ROLLOUT_FILE"
echo "    coverage : all $train_count train problems x$NUM_ROLLOUTS; dev132"
echo "    target   : v13_14b_v8; compact v4 judge; success mix; RWR ep1 lr3e-6 kl0.1 batch16"
echo "    gpus     : $NUM_GPUS ($GPU_IDS)"
echo "    stages   : $*"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "[dry-run] preflight passed; not executing stages"
  exit 0
fi

for s in "$@"; do
  echo "=== [$(date '+%F %T')] gsm 3x4B stage $s start ==="
  "${STAGE_FN[$s]}"
  echo "=== [$(date '+%F %T')] gsm 3x4B stage $s done ==="
done
echo "=== [$(date '+%F %T')] gsm 3x4B (cluster) all requested stages done ==="

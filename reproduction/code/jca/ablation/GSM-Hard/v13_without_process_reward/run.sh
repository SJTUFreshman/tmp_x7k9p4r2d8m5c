#!/usr/bin/env bash
# v13 ablation: train from the same SFT adapters without process reward.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/wangyuheng/jca}"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/qwen35/bin/python}"
PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-/data/wangyuheng}"
PYTHON_BIN_DIR="${PYTHON_BIN%/*}"
ACCELERATE="${ACCELERATE:-${PYTHON_BIN_DIR}/accelerate}"
BUILD_SCRIPT="${BUILD_SCRIPT:-${PROJECT_ROOT}/ablation/GSM-Hard/v13_without_process_reward/build_data.py}"
TRAIN_LAUNCHER="${TRAIN_LAUNCHER:-${PROJECT_ROOT}/scripts/rl_train.sh}"

SOURCE_INPUT="${SOURCE_INPUT:-${PROJECT_ROOT}/rl_data/gsm/judge_rl/v13_14b/source_rejudged.jsonl}"
ORIGINAL_TRAIN="${ORIGINAL_TRAIN:-${PROJECT_ROOT}/rl_data/gsm/judge_rl/gsm_judge_rl_v13_14b_role_c2c_1_05_1/train.jsonl}"
EXPECTED_SOURCE_SHA256="f530cb6a697ddf0ebe669cebde78a3e18d776dd2eca049257b5cb1ec757f4228"
EXPECTED_ORIGINAL_TRAIN_SHA256="54edcb7ca00716199e2feb3b79ca11452cfb8cd4e3a15b43e5de7ad3ab97a2a0"
EXPECTED_SFT_A1_WEIGHTS_SHA256="788e116f37c5ecf9ab6b0c1f3c858ee78d98928f44c168fc0a43eeb4c90ca544"
EXPECTED_SFT_A2_WEIGHTS_SHA256="ac3c078b8cf1d7b9ed907bda41502298c4d4c790c6ab5cede022805445351b07"
EXPECTED_SFT_A3_WEIGHTS_SHA256="c2ba2d4dacb2ac5b78d91483cbc578861f8a383de5faab70ea0eec7a34bfe876"
EXPECTED_SFT_A1_CONFIG_SHA256="47e846193f050dc08f5c6e1b038626d42855a9a0e35c1cb918548b5e4becc07f"
EXPECTED_SFT_A2_CONFIG_SHA256="2491b118ac581d9068eb4d1e47e1f1ac6969328153137ef58c25e59b392f3684"
EXPECTED_SFT_A3_CONFIG_SHA256="884c85cab20d428ae5599a30a8d61fc46159951ba1d75abef58f58891bb20229"

TAG="${TAG:-gsm_judge_rl_v13_ablation_without_process_reward}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/rl_data/gsm/judge_rl/${TAG}}"
TRAIN_DATA="${TRAIN_DATA:-${DATA_DIR}/train.jsonl}"
HOLDOUT_DATA="${HOLDOUT_DATA:-${DATA_DIR}/holdout.jsonl}"
DATA_STATS="${DATA_STATS:-${DATA_DIR}/stats.json}"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/rl_runs/gsm_judge_rl/${TAG}}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${PROJECT_ROOT}/logs/gsm_judge_rl/${TAG}}"

MODEL_A1="/data/wangyuheng/models/Qwen3-1.7B"
MODEL_A2="/data/wangyuheng/models/Qwen3-4B"
MODEL_A3="/data/wangyuheng/models/Qwen3-8B"
SFT_A1="${PROJECT_ROOT}/sft_runs/gsm_fixed_a1_corr30_20260723_v3/A1/final"
SFT_A2="${PROJECT_ROOT}/sft_runs/gsm_fixed_a1_corr30_20260723_v3/A2/final"
SFT_A3="${PROJECT_ROOT}/sft_runs/gsm_fixed_a1_corr30_20260723_v3/A3/final"

# Freeze the actual v13 experiment. Only operational paths and resume/wait
# behavior are configurable in this launcher.
KL_COEF=0.1
NUM_EPOCHS=1
LR=3e-6
TRAIN_SEED=42
PER_DEVICE_BATCH=1
GRAD_ACCUM=2
MAX_SEQ=8192
WARMUP_RATIO=0.05
LORA_RANK=64
LORA_ALPHA=64
LORA_DROPOUT=0.05
NUM_GPUS=8
MIXED_PRECISION=bf16
REWARD_FIELD=reward

DRY_RUN="${DRY_RUN:-0}"
RESUME="${RESUME:-1}"
WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-1}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"
GPU_IDLE_CHECKS="${GPU_IDLE_CHECKS:-2}"

fatal() {
  echo "[fatal] $*" >&2
  exit 1
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

wait_for_idle_gpus() {
  local label="$1" idle_checks=0 gpu_processes
  if [[ "$WAIT_FOR_GPUS" != "1" ]]; then
    echo "[gpu] waiting disabled before $label"
    return
  fi
  echo "[gpu] waiting for a stable all-GPU idle window before $label"
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

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] || fatal "TAG contains unsupported characters"
[[ "$DRY_RUN" == "0" || "$DRY_RUN" == "1" ]] || fatal "DRY_RUN must be 0 or 1"
[[ "$RESUME" == "0" || "$RESUME" == "1" ]] || fatal "RESUME must be 0 or 1"
[[ "$WAIT_FOR_GPUS" == "0" || "$WAIT_FOR_GPUS" == "1" ]] || fatal "WAIT_FOR_GPUS must be 0 or 1"
[[ "$TRAIN_SEED" =~ ^[0-9]+$ ]] || fatal "TRAIN_SEED must be non-negative"
[[ "$GPU_POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || fatal "GPU_POLL_SECONDS must be positive"
[[ "$GPU_IDLE_CHECKS" =~ ^[1-9][0-9]*$ ]] || fatal "GPU_IDLE_CHECKS must be positive"
if [[ "$WAIT_FOR_GPUS" == "1" ]]; then
  NVIDIA_SMI_BIN="$(command -v nvidia-smi || true)"
  [[ -n "$NVIDIA_SMI_BIN" ]] || fatal "nvidia-smi is required when WAIT_FOR_GPUS=1"
fi

for path in "$PYTHON_BIN" "$ACCELERATE" "$BUILD_SCRIPT" "$TRAIN_LAUNCHER" \
  "$SOURCE_INPUT" "$ORIGINAL_TRAIN" "$MODEL_A1" "$MODEL_A2" "$MODEL_A3" \
  "$SFT_A1" "$SFT_A2" "$SFT_A3"; do
  [[ -e "$path" ]] || fatal "required path not found: $path"
done
for adapter in "$SFT_A1" "$SFT_A2" "$SFT_A3"; do
  [[ -s "$adapter/adapter_model.safetensors" ]] || fatal "missing SFT adapter weights: $adapter"
done
[[ "$(sha256_file "$SOURCE_INPUT")" == "$EXPECTED_SOURCE_SHA256" ]] || fatal "v13 source SHA256 mismatch"
[[ "$(sha256_file "$ORIGINAL_TRAIN")" == "$EXPECTED_ORIGINAL_TRAIN_SHA256" ]] || fatal "original v13 train SHA256 mismatch"
[[ "$(sha256_file "$SFT_A1/adapter_model.safetensors")" == "$EXPECTED_SFT_A1_WEIGHTS_SHA256" ]] || fatal "A1 SFT adapter weights SHA256 mismatch"
[[ "$(sha256_file "$SFT_A2/adapter_model.safetensors")" == "$EXPECTED_SFT_A2_WEIGHTS_SHA256" ]] || fatal "A2 SFT adapter weights SHA256 mismatch"
[[ "$(sha256_file "$SFT_A3/adapter_model.safetensors")" == "$EXPECTED_SFT_A3_WEIGHTS_SHA256" ]] || fatal "A3 SFT adapter weights SHA256 mismatch"
[[ "$(sha256_file "$SFT_A1/adapter_config.json")" == "$EXPECTED_SFT_A1_CONFIG_SHA256" ]] || fatal "A1 SFT adapter config SHA256 mismatch"
[[ "$(sha256_file "$SFT_A2/adapter_config.json")" == "$EXPECTED_SFT_A2_CONFIG_SHA256" ]] || fatal "A2 SFT adapter config SHA256 mismatch"
[[ "$(sha256_file "$SFT_A3/adapter_config.json")" == "$EXPECTED_SFT_A3_CONFIG_SHA256" ]] || fatal "A3 SFT adapter config SHA256 mismatch"

BUILD_CMD=(
  env "PYTHONPATH=$PYTHONPATH_ROOT" "$PYTHON_BIN" -u "$BUILD_SCRIPT"
  --project-root "$PROJECT_ROOT" --source "$SOURCE_INPUT"
  --original-train "$ORIGINAL_TRAIN" --train-output "$TRAIN_DATA"
  --holdout-output "$HOLDOUT_DATA" --stats-output "$DATA_STATS"
)

train_command() {
  local agent="$1"
  TRAIN_CMD=(
    env "PROJECT_ROOT=$PROJECT_ROOT" "PYTHON_BIN=$PYTHON_BIN"
    "PYTHONPATH_ROOT=$PYTHONPATH_ROOT" "ACCELERATE=$ACCELERATE"
    "ROLLOUT=$TRAIN_DATA" "REWARD_FIELD=$REWARD_FIELD"
    "SFT_A1=$SFT_A1" "SFT_A2=$SFT_A2" "SFT_A3=$SFT_A3"
    "MODEL_A1=$MODEL_A1" "MODEL_A2=$MODEL_A2" "MODEL_A3=$MODEL_A3"
    "KL_COEF=$KL_COEF" "NUM_EPOCHS=$NUM_EPOCHS" "LR=$LR"
    "SEED=$TRAIN_SEED" "PER_DEVICE_BATCH=$PER_DEVICE_BATCH"
    "GRAD_ACCUM=$GRAD_ACCUM" "MAX_SEQ=$MAX_SEQ"
    "WARMUP_RATIO=$WARMUP_RATIO" "LORA_RANK=$LORA_RANK"
    "LORA_ALPHA=$LORA_ALPHA" "LORA_DROPOUT=$LORA_DROPOUT"
    "NUM_GPUS=$NUM_GPUS" "MIXED_PRECISION=$MIXED_PRECISION"
    "AGENTS=$agent" "TAG=$TAG" "RL_RUNS_DIR=$RUN_DIR"
    bash "$TRAIN_LAUNCHER"
  )
}

echo "GSM v13 ablation: without process reward"
echo "  source:          $SOURCE_INPUT"
echo "  original train:  $ORIGINAL_TRAIN"
echo "  initialization:  unchanged v13 SFT adapters"
echo "  reward:          deterministic sign, judge weight=0, judge threshold disabled"
echo "  self-handoff:    legacy v13 rows preserved by the local builder"
echo "  expected data:   train=10184 (A1=3814 A2=3568 A3=2802), holdout=1226"
echo "  train:           epochs=$NUM_EPOCHS lr=$LR kl=$KL_COEF seed=$TRAIN_SEED gpus=$NUM_GPUS batch=$PER_DEVICE_BATCH accum=$GRAD_ACCUM max_seq=$MAX_SEQ"
echo "  output:          $RUN_DIR/{A1,A2,A3}/final"
echo "  prepare command:"
print_command "${BUILD_CMD[@]}"
for agent in A1 A2 A3; do
  train_command "$agent"
  echo "  $agent training command:"
  print_command "${TRAIN_CMD[@]}"
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] hashes and paths validated; no data, logs, adapters, or training were created"
  exit 0
fi

mkdir -p "$RUN_LOG_DIR"
exec > >(tee -a "$RUN_LOG_DIR/launcher.log") 2>&1

if [[ -s "$TRAIN_DATA" && -s "$HOLDOUT_DATA" && -s "$DATA_STATS" ]]; then
  [[ "$RESUME" == "1" ]] || fatal "data outputs already exist and RESUME=0"
  echo "[resume] validating existing ablation data"
  "${BUILD_CMD[@]}" --validate-existing
elif [[ -e "$TRAIN_DATA" || -e "$HOLDOUT_DATA" || -e "$DATA_STATS" ]]; then
  fatal "partial data outputs exist; choose a new TAG or remove only those partial outputs"
else
  echo "[phase 1/4] build and validate strict without-process-reward data"
  "${BUILD_CMD[@]}"
fi

for agent in A1 A2 A3; do
  if [[ -s "$RUN_DIR/$agent/final/adapter_model.safetensors" && -s "$RUN_DIR/$agent/final/adapter_config.json" ]]; then
    [[ "$RESUME" == "1" ]] || fatal "completed output already exists: $RUN_DIR/$agent/final"
    echo "[resume] $agent already complete: $RUN_DIR/$agent/final"
    continue
  fi
  [[ ! -e "$RUN_DIR/$agent" ]] || fatal "partial training output exists: $RUN_DIR/$agent"
  wait_for_idle_gpus "training $agent"
  echo "[phase] train $agent"
  train_command "$agent"
  "${TRAIN_CMD[@]}"
  [[ -s "$RUN_DIR/$agent/final/adapter_model.safetensors" \
    && -s "$RUN_DIR/$agent/final/adapter_config.json" ]] \
    || fatal "$agent trainer exited without a complete final adapter"
done

echo "Ablation training complete: $RUN_DIR/{A1,A2,A3}/final"

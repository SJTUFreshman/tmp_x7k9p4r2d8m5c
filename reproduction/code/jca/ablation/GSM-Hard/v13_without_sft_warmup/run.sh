#!/usr/bin/env bash
# v13 ablation: run the same offline signed-RWR training from bare base models.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/wangyuheng/jca}"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/qwen35/bin/python}"
PYTHONPATH_ROOT="${PYTHONPATH_ROOT:-/data/wangyuheng}"
PYTHON_BIN_DIR="${PYTHON_BIN%/*}"
ACCELERATE="${ACCELERATE:-${PYTHON_BIN_DIR}/accelerate}"
INIT_SCRIPT="${INIT_SCRIPT:-${PROJECT_ROOT}/ablation/GSM-Hard/init_bare_lora_adapter.py}"
TRAIN_LAUNCHER="${TRAIN_LAUNCHER:-${PROJECT_ROOT}/scripts/rl_train.sh}"

V13_TAG="gsm_judge_rl_v13_14b_role_c2c_1_05_1"
V13_TRAIN_DATA="${V13_TRAIN_DATA:-${PROJECT_ROOT}/rl_data/gsm/judge_rl/${V13_TAG}/train.jsonl}"
EXPECTED_V13_TRAIN_SHA256="54edcb7ca00716199e2feb3b79ca11452cfb8cd4e3a15b43e5de7ad3ab97a2a0"
TAG="${TAG:-gsm_judge_rl_v13_ablation_without_sft_warmup}"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/rl_runs/gsm_judge_rl/${TAG}}"
INIT_ROOT="${INIT_ROOT:-${RUN_DIR}/_bare_lora_init}"

MODEL_A1="/data/wangyuheng/models/Qwen3-1.7B"
MODEL_A2="/data/wangyuheng/models/Qwen3-4B"
MODEL_A3="/data/wangyuheng/models/Qwen3-8B"
INIT_A1="${INIT_A1:-${INIT_ROOT}/A1}"
INIT_A2="${INIT_A2:-${INIT_ROOT}/A2}"
INIT_A3="${INIT_A3:-${INIT_ROOT}/A3}"

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

adapter_complete() {
  local path="$1"
  [[ -s "$path/adapter_model.safetensors" ]] \
    && [[ -s "$path/adapter_config.json" ]] \
    && [[ -s "$path/ablation_init.json" ]]
}

validate_init_adapter() {
  local path="$1" model_path="$2"
  "$PYTHON_BIN" - "$path" "$model_path" "$LORA_RANK" "$LORA_ALPHA" \
    "$LORA_DROPOUT" "$TRAIN_SEED" <<'PY'
import json
import sys
from pathlib import Path

from safetensors import safe_open

adapter_dir = Path(sys.argv[1])
model_path = sys.argv[2]
rank = int(sys.argv[3])
alpha = int(sys.argv[4])
dropout = float(sys.argv[5])
seed = int(sys.argv[6])
expected_targets = {"q_proj", "k_proj", "v_proj", "o_proj"}

metadata = json.loads((adapter_dir / "ablation_init.json").read_text(encoding="utf-8"))
config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
expected_metadata = {
    "ablation": "without_sft_warmup",
    "initial_policy": "bare_base_model_plus_fresh_noop_lora",
    "base_model": model_path,
    "lora_rank": rank,
    "lora_alpha": alpha,
    "lora_dropout": dropout,
    "seed": seed,
    "lora_b_zero_verified": True,
}
for key, expected in expected_metadata.items():
    if metadata.get(key) != expected:
        raise SystemExit(
            f"invalid initialization metadata {adapter_dir}: "
            f"{key}={metadata.get(key)!r}, expected {expected!r}"
        )
if set(metadata.get("target_modules") or []) != expected_targets:
    raise SystemExit(f"invalid initialization target modules: {adapter_dir}")

expected_config = {
    "base_model_name_or_path": model_path,
    "r": rank,
    "lora_alpha": alpha,
    "lora_dropout": dropout,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}
for key, expected in expected_config.items():
    if config.get(key) != expected:
        raise SystemExit(
            f"invalid adapter config {adapter_dir}: "
            f"{key}={config.get(key)!r}, expected {expected!r}"
        )
if set(config.get("target_modules") or []) != expected_targets:
    raise SystemExit(f"invalid adapter target modules: {adapter_dir}")

weights_path = adapter_dir / "adapter_model.safetensors"
with safe_open(weights_path, framework="pt", device="cpu") as handle:
    lora_b_keys = [key for key in handle.keys() if "lora_B" in key]
    if not lora_b_keys:
        raise SystemExit(f"no LoRA B matrices found: {weights_path}")
    for key in lora_b_keys:
        if handle.get_tensor(key).count_nonzero().item():
            raise SystemExit(f"nonzero LoRA B matrix in bare initialization: {key}")
PY
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

[[ "$TAG" =~ ^[A-Za-z0-9._-]+$ ]] || fatal "TAG contains unsupported characters"
[[ "$DRY_RUN" == "0" || "$DRY_RUN" == "1" ]] || fatal "DRY_RUN must be 0 or 1"
[[ "$RESUME" == "0" || "$RESUME" == "1" ]] || fatal "RESUME must be 0 or 1"
[[ "$WAIT_FOR_GPUS" == "0" || "$WAIT_FOR_GPUS" == "1" ]] || fatal "WAIT_FOR_GPUS must be 0 or 1"
[[ "$GPU_POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || fatal "GPU_POLL_SECONDS must be positive"
[[ "$GPU_IDLE_CHECKS" =~ ^[1-9][0-9]*$ ]] || fatal "GPU_IDLE_CHECKS must be positive"
[[ "$TRAIN_SEED" =~ ^[0-9]+$ ]] || fatal "TRAIN_SEED must be non-negative"
for path in "$PYTHON_BIN" "$ACCELERATE" "$INIT_SCRIPT" "$TRAIN_LAUNCHER" \
  "$V13_TRAIN_DATA" "$MODEL_A1" "$MODEL_A2" "$MODEL_A3"; do
  [[ -e "$path" ]] || fatal "required path not found: $path"
done
actual_v13_train_sha256="$(sha256sum "$V13_TRAIN_DATA" | awk '{print $1}')"
[[ "$actual_v13_train_sha256" == "$EXPECTED_V13_TRAIN_SHA256" ]] || \
  fatal "v13 train SHA256 mismatch: expected $EXPECTED_V13_TRAIN_SHA256, got $actual_v13_train_sha256"
[[ "$(awk 'END { print NR }' "$V13_TRAIN_DATA")" == "9995" ]] || \
  fatal "v13 train data must contain exactly 9995 rows"

echo "GSM v13 ablation: without SFT warmup"
echo "  train data:      $V13_TRAIN_DATA"
echo "  train SHA256:    $actual_v13_train_sha256"
echo "  initialization:  bare base model + fresh zero-effect LoRA"
echo "  KL reference:    the same bare-model policy at initialization"
echo "  objective:       unchanged v13 signed RWR over reward=$REWARD_FIELD"
echo "  train:           epochs=$NUM_EPOCHS lr=$LR kl=$KL_COEF seed=$TRAIN_SEED gpus=$NUM_GPUS batch=$PER_DEVICE_BATCH accum=$GRAD_ACCUM max_seq=$MAX_SEQ"
echo "  output:          $RUN_DIR/{A1,A2,A3}/final"

for agent in A1 A2 A3; do
  model_var="MODEL_${agent}"
  init_var="INIT_${agent}"
  model_path="${!model_var}"
  init_path="${!init_var}"
  INIT_CMD=(
    env "PYTHONPATH=$PYTHONPATH_ROOT" "$PYTHON_BIN" -u "$INIT_SCRIPT"
    --model-name-or-path "$model_path" --output-dir "$init_path"
    --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA"
    --lora-dropout "$LORA_DROPOUT" --seed "$TRAIN_SEED"
  )
  echo "  $agent initialization command:"
  print_command "${INIT_CMD[@]}"
done

for agent in A1 A2 A3; do
  TRAIN_CMD=(
    env "PROJECT_ROOT=$PROJECT_ROOT" "PYTHON_BIN=$PYTHON_BIN"
    "PYTHONPATH_ROOT=$PYTHONPATH_ROOT" "ACCELERATE=$ACCELERATE"
    "ROLLOUT=$V13_TRAIN_DATA" "REWARD_FIELD=$REWARD_FIELD"
    "SFT_A1=$INIT_A1" "SFT_A2=$INIT_A2" "SFT_A3=$INIT_A3"
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
  echo "  $agent training command:"
  print_command "${TRAIN_CMD[@]}"
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] no adapter initialization or training started"
  exit 0
fi

mkdir -p "$RUN_DIR"
for agent in A1 A2 A3; do
  model_var="MODEL_${agent}"
  init_var="INIT_${agent}"
  model_path="${!model_var}"
  init_path="${!init_var}"
  if adapter_complete "$init_path"; then
    [[ "$RESUME" == "1" ]] || fatal "initial adapter already exists: $init_path"
    validate_init_adapter "$init_path" "$model_path"
    echo "[resume] fresh $agent LoRA already initialized: $init_path"
    continue
  fi
  [[ ! -e "$init_path" ]] || fatal "partial initial adapter exists: $init_path"
  env PYTHONPATH="$PYTHONPATH_ROOT" "$PYTHON_BIN" -u "$INIT_SCRIPT" \
    --model-name-or-path "$model_path" --output-dir "$init_path" \
    --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" \
    --lora-dropout "$LORA_DROPOUT" --seed "$TRAIN_SEED"
  validate_init_adapter "$init_path" "$model_path"
done

for agent in A1 A2 A3; do
  if [[ -s "$RUN_DIR/$agent/final/adapter_model.safetensors" \
    && -s "$RUN_DIR/$agent/final/adapter_config.json" ]]; then
    [[ "$RESUME" == "1" ]] || fatal "completed output already exists: $RUN_DIR/$agent/final"
    echo "[resume] $agent already complete: $RUN_DIR/$agent/final"
    continue
  fi
  [[ ! -e "$RUN_DIR/$agent" ]] || fatal "partial training output exists: $RUN_DIR/$agent"
  wait_for_idle_gpus "training $agent"
  env \
    PROJECT_ROOT="$PROJECT_ROOT" PYTHON_BIN="$PYTHON_BIN" \
    PYTHONPATH_ROOT="$PYTHONPATH_ROOT" ACCELERATE="$ACCELERATE" \
    ROLLOUT="$V13_TRAIN_DATA" REWARD_FIELD="$REWARD_FIELD" \
    SFT_A1="$INIT_A1" SFT_A2="$INIT_A2" SFT_A3="$INIT_A3" \
    MODEL_A1="$MODEL_A1" MODEL_A2="$MODEL_A2" MODEL_A3="$MODEL_A3" \
    KL_COEF="$KL_COEF" NUM_EPOCHS="$NUM_EPOCHS" LR="$LR" \
    SEED="$TRAIN_SEED" PER_DEVICE_BATCH="$PER_DEVICE_BATCH" \
    GRAD_ACCUM="$GRAD_ACCUM" MAX_SEQ="$MAX_SEQ" \
    WARMUP_RATIO="$WARMUP_RATIO" LORA_RANK="$LORA_RANK" \
    LORA_ALPHA="$LORA_ALPHA" LORA_DROPOUT="$LORA_DROPOUT" \
    NUM_GPUS="$NUM_GPUS" MIXED_PRECISION="$MIXED_PRECISION" \
    AGENTS="$agent" TAG="$TAG" RL_RUNS_DIR="$RUN_DIR" \
    bash "$TRAIN_LAUNCHER"
  [[ -s "$RUN_DIR/$agent/final/adapter_model.safetensors" \
    && -s "$RUN_DIR/$agent/final/adapter_config.json" ]] \
    || fatal "$agent trainer exited without a complete final adapter"
done

echo "Ablation training complete: $RUN_DIR/{A1,A2,A3}/final"

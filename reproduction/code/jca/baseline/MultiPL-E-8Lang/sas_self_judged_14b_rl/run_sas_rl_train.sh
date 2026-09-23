#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/wangyuheng/jca}"
PYTHON_BIN="${PYTHON_BIN:-/data/conda_envs/qwen35/bin/python}"
ACCELERATE="${ACCELERATE:-/data/conda_envs/qwen35/bin/accelerate}"
ROLLOUT="${ROLLOUT:?set ROLLOUT to judged SAS JSONL}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/rl_runs/multipl_e_8lang_sas_14b_self_rl}"
MODEL="${MODEL:-/data/wangyuheng/models/Qwen3-14B}"

cd "$PROJECT_ROOT"
"$ACCELERATE" launch --num_processes "${NUM_GPUS:-8}" --mixed_precision bf16 scripts/rl_train.py \
  --agent SAS \
  --rollout "$ROLLOUT" \
  --reward-field reward \
  --model-name-or-path "$MODEL" \
  --out-dir "$OUT_DIR" \
  --kl-coef "${KL_COEF:-0.05}" \
  --num-epochs "${NUM_EPOCHS:-1}" \
  --learning-rate "${LR:-3e-6}" \
  --seed "${SEED:-42}" \
  --per-device-batch-size "${PER_DEVICE_BATCH:-2}" \
  --gradient-accumulation-steps "${GRAD_ACCUM:-8}" \
  --max-seq-length "${MAX_SEQ:-12000}" \
  --lora-rank "${LORA_RANK:-32}" \
  --lora-alpha "${LORA_ALPHA:-32}" \
  --lora-dropout "${LORA_DROPOUT:-0.05}" \
  --warmup-ratio "${WARMUP_RATIO:-0.05}" \
  --bf16 --gradient-checkpointing

#!/usr/bin/env bash
#SBATCH --job-name=maporl-seedsweep
#SBATCH --partition=hp_a800,gpu_a800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=3
#SBATCH --time=7-00:00:00
#SBATCH --output=/data/home/scwb515/run/yangrunde/maporl_baseline/slurm/logs/maporl-seedsweep-%j.log
set -Eeo pipefail
source /data/home/scwb515/run/yangrunde/maporl_baseline/deploy/common.sh

TEMP="${TEMP:-1.0}"
LIMIT="${LIMIT:-0}"
SEED_END="${SEED_END:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-96}"
EVAL_MAX_CONCURRENCY="${EVAL_MAX_CONCURRENCY:-96}"
RUN_DIR="$ROOT/runs/musique/paracloud_4gpu_48h_163146"
CONFIG="$RUN_DIR/configs/musique.yaml"
SWEEP_DIR="$RUN_DIR/eval/seedsweep_t${TEMP}"

echo "[sweep-job] MAPORL only node=$(hostname) job=$SLURM_JOB_ID temp=$TEMP start=$(date '+%F %T')"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
"$PYTHON_BIN" - <<'PY'
import os
import torch
allocated = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
if len(allocated) != 3 or not all(allocated) or torch.cuda.device_count() != 3:
    raise SystemExit(f"expected three allocated GPUs, got {allocated}")
print(f"[sweep-job] physical GPU mapping A1={allocated[0]} A2={allocated[1]} A3={allocated[2]}", flush=True)
print(f"[sweep-job] allowed CPU affinity={sorted(os.sched_getaffinity(0))}", flush=True)
PY

SANITY="$SWEEP_DIR/_sanity_maporl_only"
mkdir -p "$SANITY"
"$PYTHON_BIN" scripts/eval_maporl.py --config "$CONFIG" --run-dir "$RUN_DIR" --iteration last --seed 999999 --temperature "$TEMP" --limit 16 --output-dir "$SANITY" --batch-size "$EVAL_BATCH_SIZE" --max-concurrency "$EVAL_MAX_CONCURRENCY"
"$PYTHON_BIN" deploy/check_sanity_temp.py "$SANITY/summary.json" "$TEMP"
echo "[sweep-job] MAPORL temperature sanity gate passed"

TEMP="$TEMP" LIMIT="$LIMIT" SEED_START=1 SEED_END="$SEED_END" EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" EVAL_MAX_CONCURRENCY="$EVAL_MAX_CONCURRENCY" bash deploy/seed_sweep_musique.sh
RC=$?
echo "[sweep-job] driver exited rc=$RC at $(date '+%F %T')"

if [[ -f "$SWEEP_DIR/STOP" ]]; then
  echo "[sweep-job] STOP sentinel present, not resubmitting"
elif [[ "$SEED_END" != 0 ]]; then
  echo "[sweep-job] bounded run (SEED_END=$SEED_END) finished, not resubmitting"
else
  cd "$ROOT"
  NEXT=$(TEMP="$TEMP" LIMIT="$LIMIT" EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" EVAL_MAX_CONCURRENCY="$EVAL_MAX_CONCURRENCY" sbatch --parsable deploy/seed_sweep_musique.sbatch)
  echo "[sweep-job] resubmitted as $NEXT"
fi

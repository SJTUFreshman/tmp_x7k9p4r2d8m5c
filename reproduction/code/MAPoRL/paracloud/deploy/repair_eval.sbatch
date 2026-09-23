#!/usr/bin/env bash
#SBATCH --partition=hp_a800,gpu_a800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --time=2-00:00:00
#SBATCH --output=/data/run01/scwb515/yangrunde/maporl_baseline/slurm/logs/maporl-repair-%x-%j.log
set -Eeo pipefail
source /data/run01/scwb515/yangrunde/maporl_baseline/deploy/common.sh
DATASET="${1:?dataset argument is required}"
RUN_DIR="${2:?run directory is required}"
CONFIG="$RUN_DIR/configs/$DATASET.yaml"
OUT="$RUN_DIR/eval/repaired_${SLURM_JOB_ID}"
exec "$PYTHON_BIN" scripts/eval_maporl.py --config "$CONFIG" --run-dir "$RUN_DIR" --iteration last --include-base --output-dir "$OUT"

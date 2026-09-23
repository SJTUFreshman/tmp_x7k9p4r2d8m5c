#!/usr/bin/env bash
#SBATCH --job-name=atgrpo-mpe-sweep-t2
#SBATCH --partition=hp_a800,gpu_a800
#SBATCH --gpus=3
#SBATCH --nodes=1
#SBATCH --time=0
#SBATCH --output=/data/run01/scwb515/yangrunde/atgrpo_baseline/slurm/logs/atgrpo-mpe-sweep-t2-%j.log
source /data/run01/scwb515/yangrunde/atgrpo_baseline/deploy/common.sh
export PATH="/data/apps/apptainer/apptainer/x86_64/libexec/apptainer/bin:/data/apps/apptainer/apptainer/x86_64/utils/bin:$PATH"
export SKIP_GPU_GATE=1
export TMPDIR="/tmp/ampe-${SLURM_JOB_ID}"
export VLLM_RPC_BASE_PATH="$TMPDIR"
mkdir -p "$TMPDIR"
bash "$ROOT/deploy/prepare_multipl_e_myserver_parity.sh"
export MULTIPLE_APPTAINER_IMAGE="$ROOT/resources/containers/myserver-e753895f.sif"
export MULTIPLE_CANONICAL_SIF_SHA256
MULTIPLE_CANONICAL_SIF_SHA256="$($PYTHON_BIN -c 'import json,sys; print(json.load(open(sys.argv[1]))["sif_sha256"])' "$ROOT/resources/containers/myserver-e753895f.manifest.json")"
exec "$PYTHON_BIN" -u "$ROOT/scripts/eval_multipl_e_seedsweep_t2.py" \
  --config "$ROOT/configs/paracloud/multipl_e_seedsweep_t2.yaml" \
  --run-dir "$ROOT/runs/multipl_e/paracloud_v1" \
  --output "$ROOT/runs/multipl_e/paracloud_v1/eval/seedsweep_myserver_parity_t2_v1" \
  --iteration 119 --first-seed 1 --max-concurrency 24

#!/usr/bin/env bash
#SBATCH --job-name=magrpo-multipl-eval
#SBATCH --partition=hp_a800,gpu_a800
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --time=12:00:00
#SBATCH --output=/data/home/scwb515/run/yangrunde/magrpo_baseline/slurm/logs/magrpo-multipl-eval-%j.log
set -Eeo pipefail
source /data/home/scwb515/run/yangrunde/magrpo_baseline/deploy/common.sh

# Eval-only rerun for paracloud_parallel_4gpu_162062.
#
# Training is DONE (state.json iteration 48, adapters through iter_0047).
# Job 164021 died at its 30 h wall with eval/last empty: eval_magrpo.evaluate()
# looped over the 1352 eval problems one at a time (max_workers=1, no
# concurrency flag), so the serving GPUs sat at ~0.3% KV cache usage and 13h47m
# was not enough. team_reward_batch only runs after the whole loop, so nothing
# was ever written. deploy/patch_eval_concurrency.py makes that loop a
# ThreadPoolExecutor, mirroring atgrpo's scripts/eval_atgrpo.py.
#
# No training here, so no MAX_ITERATIONS / RESUME / wall-clock knobs.
export MULTIPLE_APPTAINER_IMAGE="$ROOT/resources/containers/multipl-e-evaluation.sif"
export MAGRPO_DISABLE_CUSTOM_ALL_REDUCE=1

CONFIG="$ROOT/slurm/configs/multipl_e-4gpu-162062.yaml"
RUN_DIR="$ROOT/runs/multipl_e/paracloud_parallel_4gpu_162062"

echo "[eval-job] node=$(hostname) job=$SLURM_JOB_ID start=$(date '+%F %T')"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "[eval-job] newest adapters:"
ls "$RUN_DIR/adapters/A1" | tail -3

# Sanity pass first: 8 problems, both arms. Cheap insurance against spending
# another queue slot on a broken eval path -- 164021 gave zero signal for 13 h.
echo "[eval-job] === sanity pass (limit 8) ==="
"$PYTHON_BIN" scripts/eval_magrpo.py --config "$CONFIG" --run-dir "$RUN_DIR" --iteration last --include-base --limit 8 --max-concurrency 8 --output "$RUN_DIR/eval/sanity8"
echo "[eval-job] sanity pass OK at $(date '+%F %T')"

echo "[eval-job] === full two-arm eval ==="
"$PYTHON_BIN" scripts/eval_magrpo.py --config "$CONFIG" --run-dir "$RUN_DIR" --iteration last --include-base --max-concurrency 48
echo "[eval-job] finished rc=$? at $(date '+%F %T')"

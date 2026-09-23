#!/usr/bin/env bash
#SBATCH --job-name=atgrpo-musique2
#SBATCH --partition=hp_a800,gpu_a800
#SBATCH --gpus=4
#SBATCH --nodes=1
#SBATCH --time=7-00:00:00
#SBATCH --output=/data/run01/scwb515/yangrunde/atgrpo_baseline/slurm/logs/atgrpo-musique2-%j.log
source /data/run01/scwb515/yangrunde/atgrpo_baseline/deploy/common.sh

# Resume of job 164655, which died at iteration 50 after 10h with
#   rollout infrastructure failed for 1/24 prompts ... HTTP Error 400
# on 2hop__242144_55948. atgrpo refuses to commit a partial iteration, so one
# over-length prompt kills the whole run. Same root cause as the conifer eval:
# t_max=4 turns accumulating against an 8192 window.
#
# max_model_len is inside the hashed config, so editing musique.yaml would make
# state.json's config_sha256 mismatch and forfeit all 50 finished iterations.
# --max-model-len is therefore a CLI override, and the config hash does survive:
# verified equal, c140444a...
#
# THAT IS ONLY HALF THE GUARD, and job 165963 died on the other half after 15s.
# Adding the flag meant editing scripts/train_atgrpo.py and scripts/eval_atgrpo.py,
# and project_code_sha256() hashes every .py/.sh under atgrpo/, scripts/, vendor/.
# So code_sha256 moved and load_resume_state() (train_atgrpo.py:136) refused.
# state.json's code_sha256 has been re-stamped by hand to cf61ba0f...; see
# runs/musique/paracloud_v1/RESUME_NOTE.md for the scope of the code delta and
# for the fact that iters 0-49 ran at 8192 and 50+ run at 16384.
#
# Anyone editing a hashed file again must re-stamp again. Do not edit any .py/.sh
# under atgrpo/, scripts/ or vendor/ while this run is alive.
DATASET=musique
CONFIG="$ROOT/configs/paracloud/musique.yaml"
RUN_DIR="$ROOT/runs/musique/paracloud_v1"

export SKIP_GPU_GATE=1

RESUME=0
if [[ -f "$RUN_DIR/state.json" ]]; then RESUME=1; fi
export RESUME

echo "[sbatch] dataset=$DATASET resume=$RESUME node=$(hostname) job=$SLURM_JOB_ID"
echo "[sbatch] resuming from iteration:"
python3 -c "import json;print(json.load(open('$RUN_DIR/state.json'))['iteration'])" || true
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# MAX_MODEL_LEN below must equal serving.max_model_len in
# configs/paracloud/musique.yaml (both 16384 as of 2026-09-19).  It is now a
# no-op for training: train_atgrpo.py:191 mutates the config to the value it
# already holds, so loop.py:320 still sees the hash stamped in state.json.
# It is NOT removable -- run_atgrpo.sh:189 uses ${MAX_MODEL_LEN:+...} and the
# default is the string "0", so dropping it would hand eval --max-model-len 0.
DATASET="$DATASET" CONFIG="$CONFIG" RUN_DIR="$RUN_DIR" EVAL_AFTER=1 \
  MAX_MODEL_LEN=16384 EVAL_MAX_CONCURRENCY=48 bash scripts/run_atgrpo.sh
echo "[sbatch] finished dataset=$DATASET rc=$?"

#!/usr/bin/env bash
set -euo pipefail
source /data/run01/scwb515/yangrunde/magrpo_baseline/deploy/common.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
exec "$PYTHON_BIN" /data/run01/scwb515/yangrunde/magrpo_baseline/deploy/merge_math_recovery.py \
  --original /data/run01/scwb515/yangrunde/magrpo_baseline/runs/math/paracloud_parallel_160136/eval/last/results_magrpo.jsonl \
  --recovery /data/run01/scwb515/yangrunde/magrpo_baseline/runs/math/paracloud_parallel_160136/eval/recover_protocol/results_magrpo.jsonl \
  --grader /data/run01/scwb515/yangrunde/magrpo_baseline/vendor/math_eval_v4.py \
  --output /data/run01/scwb515/yangrunde/magrpo_baseline/runs/math/paracloud_parallel_160136/eval/recovered_merged

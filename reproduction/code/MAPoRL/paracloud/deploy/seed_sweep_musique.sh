#!/usr/bin/env bash
set -Eeo pipefail
source /data/home/scwb515/run/yangrunde/maporl_baseline/deploy/common.sh

TEMP="${TEMP:-1.0}"
LIMIT="${LIMIT:-0}"
SEED_START="${SEED_START:-1}"
SEED_END="${SEED_END:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-96}"
EVAL_MAX_CONCURRENCY="${EVAL_MAX_CONCURRENCY:-96}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/musique/paracloud_4gpu_48h_163146}"
CONFIG="${CONFIG:-$RUN_DIR/configs/musique.yaml}"
SWEEP_DIR="${SWEEP_DIR:-$RUN_DIR/eval/seedsweep_t${TEMP}}"
LEDGER="$SWEEP_DIR/sweep.jsonl"

mkdir -p "$SWEEP_DIR"
exec 9>"$SWEEP_DIR/.sweep.lock"
flock -n 9 || { echo "[sweep] another sweep owns $SWEEP_DIR" >&2; exit 1; }
echo "[sweep] MAPORL only temp=$TEMP limit=$LIMIT seeds from $SEED_START batch=$EVAL_BATCH_SIZE concurrency=$EVAL_MAX_CONCURRENCY ledger=$LEDGER"

seed="$SEED_START"
while true; do
  if [[ -f "$SWEEP_DIR/STOP" ]]; then
    echo "[sweep] STOP sentinel present, exiting cleanly before seed $seed"
    exit 0
  fi
  if [[ "$SEED_END" != 0 && "$seed" -gt "$SEED_END" ]]; then
    echo "[sweep] reached SEED_END=$SEED_END, done"
    exit 0
  fi

  OUT="$SWEEP_DIR/seed_${seed}"
  if [[ -f "$OUT/summary.json" ]] && "$PYTHON_BIN" deploy/sweep_record.py "$OUT/summary.json" "$LEDGER" "$seed" --check-only; then
    echo "[sweep] seed $seed already has a complete MAPORL summary, skipping"
  else
    echo "[sweep] === seed $seed temp $TEMP start $(date '+%F %T') ==="
    EVAL_ARGS=(--config "$CONFIG" --run-dir "$RUN_DIR" --iteration last --seed "$seed" --temperature "$TEMP" --output-dir "$OUT" --batch-size "$EVAL_BATCH_SIZE" --max-concurrency "$EVAL_MAX_CONCURRENCY")
    if [[ "$LIMIT" != 0 ]]; then EVAL_ARGS+=(--limit "$LIMIT"); fi
    "$PYTHON_BIN" scripts/eval_maporl.py "${EVAL_ARGS[@]}"
    echo "[sweep] === seed $seed done $(date '+%F %T') ==="
  fi

  "$PYTHON_BIN" deploy/sweep_record.py "$OUT/summary.json" "$LEDGER" "$seed"
  seed=$((seed + 1))
done

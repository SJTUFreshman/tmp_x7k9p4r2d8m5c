#!/usr/bin/env bash
# Re-evaluate an ALREADY SEARCHED Conifer AFlow workflow N times on the test
# split, to put an error bar on a number that has only ever been measured once.
#
# The search is NOT repeated.  It is the expensive half (~2h for 20x20) and
# re-running it would answer a different question -- how much the MCTS search
# varies -- rather than how much the reported test number varies.  SEARCH_RUN_DIR
# must therefore point at a completed search, and RUN_SEARCH=0 is forced below
# so an accidental empty directory cannot silently start a fresh 2h search.
#
# Each roll re-resolves the workflow from that search's search_metrics.json
# rather than taking a hardcoded path, so the file this evaluates is by
# construction the one the search actually selected.
#
# The spread here is genuine sampling noise, not batching jitter: the AFlow ops
# run at BASELINE_AFLOW_TEMPERATURE=0.7 (run_conifer_eval.sh:47).
#
# Usage:
#   SEARCH_RUN_DIR=logs/baseline_queue/rq09110842/030_conifer_aflow_search/search \
#   REPEATS=5 bash conifer_training_hub/03_rollout/aflow/run_eval_repeat.sh
#
# As a queue RUNNER: run_baseline_queue.sh injects RUN_ID and RUN_DIR.  The
# injected RUN_ID becomes the batch name; RUN_DIR is REPLACED per roll, because
# the queue points it at the one shared job directory and all N rolls would
# otherwise overwrite each other's trajectories.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
HUB_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${HUB_ROOT}/.." && pwd)}"

BASE_SCRIPT="${BASE_SCRIPT:-${SCRIPT_DIR}/run_search_then_eval.sh}"
AGGREGATOR="${AGGREGATOR:-${PROJECT_ROOT}/scripts/baseline_queues/summarize_conifer_eval_batch.py}"
AGGREGATOR_PYTHON="${AGGREGATOR_PYTHON:-/data/conda_envs/qwen35/bin/python}"

REPEATS="${REPEATS:-5}"
SLEEP_SECONDS="${SLEEP_SECONDS:-30}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

SEARCH_RUN_DIR="${SEARCH_RUN_DIR:-}"
TEST_DATA="${TEST_DATA:-${HUB_ROOT}/01_dataset/processed/test.jsonl}"

BATCH_NAME="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_conifer_aflow_eval_repeat}"
BATCH_DIR="${BATCH_DIR:-${RUN_DIR:-${HUB_ROOT}/11_runs/aflow_eval_repeat/${BATCH_NAME}}}"
MASTER_LOG="${BATCH_DIR}/master.log"
STATUS_TSV="${BATCH_DIR}/status.tsv"
SUMMARY_JSON="${SUMMARY_JSON:-${BATCH_DIR}/summary.json}"

fatal() { echo "[fatal] $*" >&2; exit 2; }

[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || fatal "REPEATS must be a positive integer: $REPEATS"
[[ "$SLEEP_SECONDS" =~ ^[0-9]+$ ]] || fatal "SLEEP_SECONDS must be a non-negative integer"
[[ -f "$BASE_SCRIPT" ]] || fatal "base script missing: $BASE_SCRIPT"
[[ -n "$SEARCH_RUN_DIR" ]] || fatal "SEARCH_RUN_DIR is required -- point it at a completed search"
[[ -s "${SEARCH_RUN_DIR}/search_metrics.json" ]] \
  || fatal "no completed search at ${SEARCH_RUN_DIR}/search_metrics.json"
[[ -s "$TEST_DATA" ]] || fatal "test data missing: $TEST_DATA"

mkdir -p "$BATCH_DIR"
exec > >(tee -a "$MASTER_LOG") 2>&1

SELECTED="$("$AGGREGATOR_PYTHON" -c "
import json,sys
m=json.load(open('${SEARCH_RUN_DIR}/search_metrics.json'))
print(m.get('selected_source'), m.get('selected_round'), m.get('selected_hard_score'))
")"

echo "============ Conifer AFlow eval repeat (search reused) ============"
echo "time:        $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "batch_name:  $BATCH_NAME"
echo "batch_dir:   $BATCH_DIR"
echo "search_dir:  $SEARCH_RUN_DIR"
echo "selected:    $SELECTED"
echo "test_data:   $TEST_DATA"
echo "repeats:     $REPEATS"
echo "==================================================================="
echo

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  # run_baseline_queue.sh preflights with DRY_RUN=1.  run_search_then_eval.sh
  # falls through to stage 2 when the search is already complete, so one pass
  # actually validates the eval command instead of REPEATS no-ops.
  echo "dry-run: validating one eval roll instead of ${REPEATS}"
  env RUN_SEARCH=0 RUN_EVAL=1 DRY_RUN=1 \
    RUN_ID="${BATCH_NAME}_preflight" \
    RUN_DIR="${BATCH_DIR}/.preflight" \
    SEARCH_RUN_DIR="$SEARCH_RUN_DIR" \
    EVAL_TAG="${BATCH_NAME}_preflight" \
    TEST_DATA="$TEST_DATA" DATA_PATH="$TEST_DATA" \
    OUTPUT="${BATCH_DIR}/.preflight/trajectories.jsonl" \
    SCORED_OUTPUT="${BATCH_DIR}/.preflight/scored.jsonl" \
    bash "$BASE_SCRIPT"
  exit $?
fi

if [[ ! -s "$STATUS_TSV" ]]; then
  printf 'index\trun_id\tstatus\tscored_output\tstarted_at\tended_at\telapsed_seconds\n' > "$STATUS_TSV"
fi

failures=0
for (( i = 1; i <= REPEATS; i++ )); do
  roll="$(printf 'r%02d' "$i")"
  run_id="${BATCH_NAME}_${roll}"
  roll_dir="${BATCH_DIR}/${roll}"
  scored_output="${roll_dir}/scored.jsonl"
  started_at="$(date '+%Y-%m-%d %H:%M:%S %Z')"
  started_epoch="$(date +%s)"
  mkdir -p "$roll_dir"

  echo "---------------- roll $i/$REPEATS ----------------"
  echo "run_id:  $run_id"
  echo "outputs: $roll_dir"
  echo

  status="ok"
  # RUN_SEARCH=0 is hardcoded, not defaulted: this wrapper exists to reuse a
  # search, and a stray RUN_SEARCH=1 in the environment would burn two hours
  # re-running one.  OUTPUT/SCORED_OUTPUT/RUN_DIR are per-roll for the same
  # reason the MultiPL-E wrapper overrides RUN_DIR -- the queue presets them to
  # a single shared directory.
  if ! env \
      RUN_SEARCH=0 RUN_EVAL=1 \
      RUN_ID="$run_id" \
      RUN_DIR="$roll_dir" \
      LOG_DIR="$roll_dir" \
      SEARCH_RUN_DIR="$SEARCH_RUN_DIR" \
      EVAL_TAG="$run_id" \
      TEST_DATA="$TEST_DATA" \
      DATA_PATH="$TEST_DATA" \
      OUTPUT="${roll_dir}/trajectories.jsonl" \
      SCORED_OUTPUT="$scored_output" \
      bash "$BASE_SCRIPT"; then
    status="failed"
    (( failures += 1 ))
    echo
    echo "roll $i failed: $run_id"
  fi

  ended_at="$(date '+%Y-%m-%d %H:%M:%S %Z')"
  elapsed="$(( $(date +%s) - started_epoch ))"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$i" "$run_id" "$status" "$scored_output" "$started_at" "$ended_at" "$elapsed" >> "$STATUS_TSV"

  echo
  echo "roll $i/$REPEATS finished: $status in ${elapsed}s"
  echo

  if [[ "$status" == "failed" && "$CONTINUE_ON_ERROR" != "1" ]]; then
    echo "stopping batch because CONTINUE_ON_ERROR=$CONTINUE_ON_ERROR"
    break
  fi

  if (( i < REPEATS )) && (( SLEEP_SECONDS > 0 )); then
    echo "sleeping ${SLEEP_SECONDS}s so the GPUs drain before the next roll..."
    sleep "$SLEEP_SECONDS"
    echo
  fi
done

echo "================ aggregating ================"
if [[ -f "$AGGREGATOR" ]]; then
  "$AGGREGATOR_PYTHON" "$AGGREGATOR" --status "$STATUS_TSV" --output "$SUMMARY_JSON" \
    --search-metrics "${SEARCH_RUN_DIR}/search_metrics.json" || {
    echo "WARNING: aggregation failed; per-roll artifacts are still intact." >&2
  }
else
  echo "WARNING: aggregator not found at $AGGREGATOR; skipping summary.json" >&2
fi

echo
echo "batch_dir:    $BATCH_DIR"
echo "status_tsv:   $STATUS_TSV"
echo "summary:      $SUMMARY_JSON"
echo "failed rolls: $failures/$REPEATS"

(( failures < REPEATS )) || exit 1

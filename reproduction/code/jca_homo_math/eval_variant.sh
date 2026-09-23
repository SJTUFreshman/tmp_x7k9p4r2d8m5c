#!/usr/bin/env bash
set -euo pipefail

if (( $# != 7 )); then
  echo "usage: $0 NAME MODEL_A1 MODEL_A2 MODEL_A3 ADAPTER_A1 ADAPTER_A2 ADAPTER_A3" >&2
  exit 2
fi

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
# shellcheck source=common.sh
source "${EXPERIMENT_ROOT}/common.sh"
validate_static_config

variant="$1"
model_a1="$2"
model_a2="$3"
model_a3="$4"
adapter_a1="$5"
adapter_a2="$6"
adapter_a3="$7"
[[ "$variant" =~ ^[A-Za-z0-9._-]+$ ]] || fatal "invalid variant name: $variant"

EVAL_STAGE="${UPSTREAM_MATH_ROOT}/05_eval.sh"
variant_root="${EVAL_ROOT}/variants/${variant}"
output="${variant_root}/results.jsonl"
state="${variant_root}/trajectory_state.json"
metrics="${variant_root}/metrics.json"
fingerprint="${state}.evaluator.json"
variant_manifest="${variant_root}/variant.json"
require_file "$EVAL_STAGE"
require_path "$model_a1"
require_path "$model_a2"
require_path "$model_a3"
if [[ "$DRY_RUN" != "1" ]]; then
  require_adapter "$adapter_a1"
  require_adapter "$adapter_a2"
  require_adapter "$adapter_a3"
  require_file "$EVAL_DATA_ROOT/manifest.json"
fi

EVAL_CMD=(
  env
  "EXPERIMENT_ROOT=$UPSTREAM_MATH_ROOT" "PROJECT_ROOT=$PROJECT_ROOT"
  "PYTHONPATH_ROOT=$PYTHONPATH_ROOT" "PYTHON_BIN=$PYTHON_BIN"
  "VLLM_PYTHON_BIN=$VLLM_PYTHON_BIN"
  "VLLM_LD_LIBRARY_PATH=$VLLM_LD_LIBRARY_PATH"
  "TAG=${TAG}_eval_${variant}" "ARTIFACT_ROOT=$ARTIFACT_ROOT"
  "MATH_DATA_ROOT=$EVAL_DATA_ROOT" "LOG_ROOT=${variant_root}/logs"
  "EVAL_OUTPUT=$output" "EVAL_STATE=$state" "EVAL_METRICS=$metrics"
  "EVAL_FINGERPRINT_FILE=$fingerprint"
  "EVAL_START=$EVAL_START" "EVAL_LIMIT=$EVAL_LIMIT" "EVAL_T_MAX=$EVAL_T_MAX"
  "EVAL_BOOTSTRAP_AGENT=$EVAL_BOOTSTRAP_AGENT"
  "EVAL_START_AGENT=$EVAL_BOOTSTRAP_AGENT"
  "EVAL_START_AGENT_SEED=$EVAL_START_AGENT_SEED"
  "EVAL_GENERATION_SEED=$EVAL_GENERATION_SEED"
  "EVAL_TEMPERATURE=$EVAL_TEMPERATURE" "EVAL_TOP_P=$EVAL_TOP_P"
  "EVAL_MAX_NEW_TOKENS=$EVAL_MAX_NEW_TOKENS"
  "EVAL_MAX_CONCURRENCY=$EVAL_MAX_CONCURRENCY"
  "EVAL_GROUP_RETRIES=$EVAL_GROUP_RETRIES" "EVAL_STEP_RETRIES=$EVAL_STEP_RETRIES"
  "EVAL_SCHEDULER_MODE=shared_all_gpu" "EVAL_ADAPTIVE_LAYOUT=0"
  "EVAL_ADAPTIVE_ALLOW_ZERO=0" "EVAL_REBALANCE_AFTER_ADVANCED=64"
  "EVAL_MAX_ADVANCED_PER_PASS=0" "EVAL_MAX_INFLIGHT=128"
  "EVAL_SHARED_GPU_IDS=$GPU_IDS" "EVAL_SHARED_DP=$NUM_GPUS"
  "EVAL_A1_GPU_IDS=$GPU_IDS" "EVAL_A2_GPU_IDS=$GPU_IDS" "EVAL_A3_GPU_IDS=$GPU_IDS"
  "EVAL_A1_DP=$NUM_GPUS" "EVAL_A2_DP=$NUM_GPUS" "EVAL_A3_DP=$NUM_GPUS"
  "EVAL_A1_GPU_MEMORY_UTILIZATION=${EVAL_A1_GPU_MEMORY_UTILIZATION:-0.28}"
  "EVAL_A2_GPU_MEMORY_UTILIZATION=${EVAL_A2_GPU_MEMORY_UTILIZATION:-0.28}"
  "EVAL_A3_GPU_MEMORY_UTILIZATION=${EVAL_A3_GPU_MEMORY_UTILIZATION:-0.28}"
  "EVAL_A1_MAX_CONCURRENCY=384" "EVAL_A2_MAX_CONCURRENCY=384"
  "EVAL_A3_MAX_CONCURRENCY=384" "EVAL_PROTOCOL_THINKING_MAX_TOKENS=2048"
  "EVAL_ENFORCE_EAGER=0"
  "MODEL_A1=$model_a1" "MODEL_A2=$model_a2" "MODEL_A3=$model_a3"
  "SFT_ROOT=$SFT_RUN_ROOT" "SFT_A1=$adapter_a1" "SFT_A2=$adapter_a2" "SFT_A3=$adapter_a3"
  "RL_ROOT=$RL_ROOT"
  "EVAL_ADAPTER_A1=$adapter_a1" "EVAL_ADAPTER_A2=$adapter_a2"
  "EVAL_ADAPTER_A3=$adapter_a3"
  "SAMPLE_ENABLE_THINKING=$SAMPLE_ENABLE_THINKING"
  "SAMPLE_REQUIRE_THINKING=$SAMPLE_REQUIRE_THINKING"
  "TRAIN_ENABLE_THINKING=$TRAIN_ENABLE_THINKING"
  "ENABLE_THINKING=$ENABLE_THINKING" "REQUIRE_THINKING=$REQUIRE_THINKING"
  "ALLOW_THINKING_MODE_MISMATCH=$ALLOW_THINKING_MODE_MISMATCH"
  "SAMPLE_JSON_TRANSPORT=$SAMPLE_JSON_TRANSPORT"
  "SAMPLE_MAX_NEW_TOKENS=$SAMPLE_MAX_NEW_TOKENS"
  "JUDGE_MAX_TOKENS=$JUDGE_MAX_TOKENS"
  "NUM_ROLLOUTS=$NUM_ROLLOUTS" "T_MAX=$T_MAX"
  "RESUME=$RESUME" "DRY_RUN=$DRY_RUN" "WAIT_FOR_GPUS=$WAIT_FOR_GPUS"
  "GPU_POLL_SECONDS=$GPU_POLL_SECONDS" "GPU_IDLE_CHECKS=$GPU_IDLE_CHECKS"
  bash "$EVAL_STAGE"
)

echo "MATH evaluation variant: $variant"
echo "  shard:    $EVAL_SHARD_FILE"
echo "  rows:     $EVAL_LIMIT"
echo "  thinking: disabled"
echo "  adapters: $adapter_a1 | $adapter_a2 | $adapter_a3"
echo "  output:   $output"
print_command "${EVAL_CMD[@]}"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] evaluation not started"
  exit 0
fi

mkdir -p "$variant_root"
"$PYTHON_BIN" - "$variant_manifest" "$variant" "$EVAL_SHARD_FILE" \
  "$EVAL_DATA_ROOT/manifest.json" "$model_a1" "$model_a2" "$model_a3" \
  "$adapter_a1" "$adapter_a2" "$adapter_a3" "$EVAL_LIMIT" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

(
    output,
    name,
    shard,
    shard_manifest,
    model_a1,
    model_a2,
    model_a3,
    adapter_a1,
    adapter_a2,
    adapter_a3,
    expected_count,
) = sys.argv[1:]


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adapter(path):
    digest = hashlib.sha256()
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        candidate = os.path.join(path, filename)
        if not os.path.isfile(candidate):
            raise SystemExit(f"adapter file missing: {candidate}")
        digest.update(filename.encode())
        digest.update(b"\0")
        digest.update(sha(candidate).encode())
        digest.update(b"\0")
    return {"path": os.path.realpath(path), "sha256": digest.hexdigest()}


models = {"A1": model_a1, "A2": model_a2, "A3": model_a3}
adapters = {"A1": adapter_a1, "A2": adapter_a2, "A3": adapter_a3}
value = {
    "schema_version": 1,
    "name": name,
    "dataset": "MATH test",
    "shard": {"path": os.path.realpath(shard), "sha256": sha(shard), "count": int(expected_count)},
    "shard_manifest": {"path": os.path.realpath(shard_manifest), "sha256": sha(shard_manifest)},
    "models": {
        agent: {"path": os.path.realpath(path), "config_sha256": sha(os.path.join(path, "config.json"))}
        for agent, path in models.items()
    },
    "adapters": {agent: adapter(path) for agent, path in adapters.items()},
    "generation": {
        "temperature": 0.0,
        "thinking_enabled": False,
        "one_rollout_per_problem": True,
    },
}
destination = Path(output)
if destination.exists():
    actual = json.loads(destination.read_text(encoding="utf-8"))
    if actual != value:
        raise SystemExit("evaluation variant contract changed; choose a new variant name")
else:
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
PY

if [[ "${GPU_LOCK_HELD:-0}" != "1" ]]; then
  acquire_gpu_lock
fi
wait_for_idle_gpus "MATH evaluation $variant"
exec "${EVAL_CMD[@]}"

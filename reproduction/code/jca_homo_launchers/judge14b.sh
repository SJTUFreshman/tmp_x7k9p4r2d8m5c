#!/usr/bin/env bash
# Local Qwen3-14B judge sidecar for the 3x4B cluster runners.  SOURCE, don't run.
#
#   judge14b_start   launch the vLLM server, block until it serves, export the
#                    JCA_JUDGE_* env the rollout's judge path reads, and install
#                    an EXIT/INT/TERM trap so the GPUs are always released
#   judge14b_stop    kill it (idempotent)
#
# WHY THIS EXISTS
# ---------------
# The JCA reward has always been produced by a local Qwen3-14B, never by gpt-5:
#   * musique's RL arm trains on rl_data/rl_14b_qwen_0716.jsonl, written by
#     scripts/rl_rescore_qwen14b_vllm.sh
#   * gsm's shipped arm is v13_14b_v8/v9, written by rejudge_gsm_qwen14b.sh
#   * multipl_e uses rejudge_multipl_e_qwen14b_vllm.sh
#   * math is the sas_self_judged_14b_rl tree
# gpt-5 survives only as `JUDGE_MODEL`'s vendored default on the *inline* rollout
# judge, whose score is then thrown away and recomputed by the 14B pass.  So the
# honest -- and cheaper -- shape is to judge with the 14B inline and drop the
# rescore pass entirely.  scripts/rl_rollout.py and scripts/rl_rescore_rollout.py
# both call jca.src.judge with the same alpha and write the same
# reward/judge_score/em/f1 fields, so the two files are interchangeable and
# rl_train.sh --reward-field reward consumes either without modification.
#
# THE ENV BLOCK BELOW IS NOT INVENTED.
# rejudge_gsm_qwen14b.sh:269-276 and rejudge_multipl_e_qwen14b_vllm.sh:261-278
# are the two scripts that already drive this exact code path against a local
# 14B, and they agree line for line:
#     JCA_JUDGE_API_BASE=http://HOST:PORT/v1
#     JCA_JUDGE_API_KEY=EMPTY
#     JCA_JUDGE_DISABLE_REASONING_EFFORT=1     <- vLLM has no reasoning_effort
#     JCA_JUDGE_DISABLE_THINKING=1
#     JCA_JUDGE_ENABLE_THINKING=0
# plus the server-side --{default-,}chat-template-kwargs '{"enable_thinking":false}'.
# Thinking off matters twice over: Qwen3's template defaults it ON, and the judge
# must emit parseable JSON inside JUDGE_MAX_TOKENS or ALLOW_JUDGE_FAILURE=0 aborts
# the run.  rejudge_gsm_qwen14b.sh:55 settles on 4096 tokens against a 16384
# context, and enforces JUDGE_MAX_TOKENS < MAX_MODEL_LEN; both are kept here.
#
# GPU BUDGET: the 14B takes 2 of the 8 cards (TP=2, ~28 GB bf16 weights), leaving
# 0-5 for the three 4B policies at TP=2 each.  Neither rl_rollout_vllm_8gpu.sh nor
# run_gsm_correction_vllm_8gpu.sh has a GPU-idle guard and both read A*_GPUS /
# A*_TP as ${VAR:-default}, so this needs no vendored edits.

JUDGE14B_MODEL_PATH="${JUDGE14B_MODEL_PATH:-}"
JUDGE14B_HOST="${JUDGE14B_HOST:-127.0.0.1}"
JUDGE14B_PORT="${JUDGE14B_PORT:-8300}"
JUDGE14B_GPUS="${JUDGE14B_GPUS:-6,7}"
JUDGE14B_TP="${JUDGE14B_TP:-2}"
JUDGE14B_SERVED_NAME="${JUDGE14B_SERVED_NAME:-qwen14b_judge}"
JUDGE14B_DTYPE="${JUDGE14B_DTYPE:-bfloat16}"
JUDGE14B_GPU_MEM_UTIL="${JUDGE14B_GPU_MEM_UTIL:-0.80}"
JUDGE14B_MAX_MODEL_LEN="${JUDGE14B_MAX_MODEL_LEN:-16384}"
JUDGE14B_MAX_TOKENS="${JUDGE14B_MAX_TOKENS:-4096}"
JUDGE14B_ENFORCE_EAGER="${JUDGE14B_ENFORCE_EAGER:-1}"
JUDGE14B_ENABLE_THINKING="${JUDGE14B_ENABLE_THINKING:-0}"
JUDGE14B_WAIT="${JUDGE14B_WAIT:-2400}"
JUDGE14B_LOG="${JUDGE14B_LOG:-}"

JUDGE14B_PID=""
_JUDGE14B_HELP=""

judge14b_fatal() { echo "[judge14b][fatal] $*" >&2; exit 2; }

# vLLM CLI capability probe, same approach as rl_rescore_qwen14b_vllm.sh: flag
# spellings moved between releases and the cluster env ships vLLM 0.16.0.
_judge14b_supports() {
  local opt="$1"
  if [[ -z "$_JUDGE14B_HELP" ]]; then
    _JUDGE14B_HELP="$(
      LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
      "$VLLM_PYTHON_BIN" -m vllm.entrypoints.openai.api_server --help 2>&1 || true
    )"
  fi
  grep -q -- "$opt" <<<"$_JUDGE14B_HELP"
}

# Returns 0 once the server answers /v1/models with our served name.
judge14b_ready() {
  local url="http://${JUDGE14B_HOST}:${JUDGE14B_PORT}/v1/models"
  if command -v curl >/dev/null 2>&1; then
    curl -sf --noproxy '*' --max-time 5 "$url" 2>/dev/null | grep -q "$JUDGE14B_SERVED_NAME"
  else
    "$VLLM_PYTHON_BIN" - "$url" "$JUDGE14B_SERVED_NAME" <<'PY' >/dev/null 2>&1
import sys, urllib.request
url, name = sys.argv[1], sys.argv[2]
try:
    body = urllib.request.urlopen(url, timeout=5).read().decode()
except Exception:
    sys.exit(1)
sys.exit(0 if name in body else 1)
PY
  fi
}

judge14b_stop() {
  local status=$?
  [[ -n "$JUDGE14B_PID" ]] || return "$status"
  if kill -0 "$JUDGE14B_PID" 2>/dev/null; then
    echo "[judge14b] stopping server pid=$JUDGE14B_PID"
    # setsid put it in its own process group; negate the pid to take the workers
    # (TP=2 forks an engine child) with it.  Never pkill -f: the pattern would
    # match this very shell's command line.
    kill -TERM -"$JUDGE14B_PID" 2>/dev/null || kill -TERM "$JUDGE14B_PID" 2>/dev/null || true
    local waited=0
    while kill -0 "$JUDGE14B_PID" 2>/dev/null && (( waited < 90 )); do
      sleep 3; waited=$(( waited + 3 ))
    done
    kill -KILL -"$JUDGE14B_PID" 2>/dev/null || true
  fi
  JUDGE14B_PID=""
  return "$status"
}

judge14b_start() {
  [[ -n "$JUDGE14B_MODEL_PATH" ]] || judge14b_fatal "JUDGE14B_MODEL_PATH is unset"
  [[ -s "$JUDGE14B_MODEL_PATH/config.json" ]] || judge14b_fatal "no judge model at $JUDGE14B_MODEL_PATH"
  [[ -x "${VLLM_PYTHON_BIN:-}" ]] || judge14b_fatal "VLLM_PYTHON_BIN is not executable: ${VLLM_PYTHON_BIN:-unset}"
  (( JUDGE14B_MAX_TOKENS < JUDGE14B_MAX_MODEL_LEN )) || \
    judge14b_fatal "JUDGE14B_MAX_TOKENS ($JUDGE14B_MAX_TOKENS) must leave prompt headroom under MAX_MODEL_LEN ($JUDGE14B_MAX_MODEL_LEN)"

  if judge14b_ready; then
    echo "[judge14b] reusing a server already serving $JUDGE14B_SERVED_NAME on ${JUDGE14B_HOST}:${JUDGE14B_PORT}"
  else
    JUDGE14B_LOG="${JUDGE14B_LOG:-${TMPDIR:-/tmp}/qwen14b_judge_${JUDGE14B_PORT}.log}"
    mkdir -p "$(dirname "$JUDGE14B_LOG")"

    local cmd=(
      "$VLLM_PYTHON_BIN" -m vllm.entrypoints.openai.api_server
      --host "$JUDGE14B_HOST" --port "$JUDGE14B_PORT"
      --model "$JUDGE14B_MODEL_PATH"
      --served-model-name "$JUDGE14B_SERVED_NAME"
      --tensor-parallel-size "$JUDGE14B_TP"
      --dtype "$JUDGE14B_DTYPE"
      --gpu-memory-utilization "$JUDGE14B_GPU_MEM_UTIL"
      --max-model-len "$JUDGE14B_MAX_MODEL_LEN"
      --trust-remote-code
    )
    # if/fi, not `[[ ]] && cmd`: the callers run under `set -e`, where a false
    # test as a whole statement is itself a failing command.
    if [[ "$JUDGE14B_ENFORCE_EAGER" == "1" ]]; then cmd+=(--enforce-eager); fi

    local ctk
    if [[ "$JUDGE14B_ENABLE_THINKING" == "1" ]]; then
      ctk='{"enable_thinking":true}'
    else
      ctk='{"enable_thinking":false}'
    fi
    if _judge14b_supports "--default-chat-template-kwargs"; then
      cmd+=(--default-chat-template-kwargs "$ctk")
    elif _judge14b_supports "--chat-template-kwargs"; then
      cmd+=(--chat-template-kwargs "$ctk")
    else
      # Not fatal: JCA_JUDGE_DISABLE_THINKING below also suppresses it per-request
      # via llm_client.py's chat_template_kwargs injection.
      echo "[judge14b] note: this vLLM has no chat-template-kwargs flag; relying on the per-request switch"
    fi

    echo "[judge14b] starting Qwen3-14B judge  gpus=$JUDGE14B_GPUS tp=$JUDGE14B_TP port=$JUDGE14B_PORT thinking=$JUDGE14B_ENABLE_THINKING"
    echo "[judge14b] log: $JUDGE14B_LOG"
    setsid env \
      CUDA_VISIBLE_DEVICES="$JUDGE14B_GPUS" \
      LD_LIBRARY_PATH="${VLLM_LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
      no_proxy="${no_proxy:-},127.0.0.1,localhost" NO_PROXY="${NO_PROXY:-},127.0.0.1,localhost" \
      "${cmd[@]}" >"$JUDGE14B_LOG" 2>&1 &
    JUDGE14B_PID="$!"
    trap judge14b_stop EXIT INT TERM

    local waited=0
    until judge14b_ready; do
      if ! kill -0 "$JUDGE14B_PID" 2>/dev/null; then
        echo "[judge14b] --- last 60 lines of $JUDGE14B_LOG ---" >&2
        tail -n 60 "$JUDGE14B_LOG" >&2 || true
        judge14b_fatal "judge server died during startup"
      fi
      (( waited < JUDGE14B_WAIT )) || {
        tail -n 60 "$JUDGE14B_LOG" >&2 || true
        judge14b_fatal "judge server did not come up within ${JUDGE14B_WAIT}s"
      }
      sleep 10; waited=$(( waited + 10 ))
    done
    echo "[judge14b] ready after ${waited}s"
  fi

  # The exact switches rejudge_gsm_qwen14b.sh / rejudge_multipl_e_qwen14b_vllm.sh
  # export around this same code path.  llm_client.py reads every one of them.
  export JCA_JUDGE_API_BASE="http://${JUDGE14B_HOST}:${JUDGE14B_PORT}/v1"
  export JCA_JUDGE_API_KEY="EMPTY"
  export JCA_JUDGE_DISABLE_REASONING_EFFORT=1
  if [[ "$JUDGE14B_ENABLE_THINKING" == "1" ]]; then
    export JCA_JUDGE_DISABLE_THINKING=0 JCA_JUDGE_ENABLE_THINKING=1
  else
    export JCA_JUDGE_DISABLE_THINKING=1 JCA_JUDGE_ENABLE_THINKING=0
  fi
  # Loopback must never go through the corporate proxy.
  export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"
  export NO_PROXY="$no_proxy"
  echo "[judge14b] judge endpoint: $JCA_JUDGE_API_BASE  model=$JUDGE14B_SERVED_NAME  max_tokens=$JUDGE14B_MAX_TOKENS"
}

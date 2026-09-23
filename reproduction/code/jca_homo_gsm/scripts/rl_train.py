"""JCA RL Training -- format-protected token-level signed RWR.

Objective:
  - Each turn is trained with the signed reward written by the rollout scorer.
  - Positive-reward turns reinforce the complete assistant response.
  - Negative-reward turns suppress only semantic thinking/JSON-value tokens;
    protocol punctuation, field names, and scaffolding remain format-protected.
  - A token-level forward KL anchor keeps the policy close to its SFT/base
    reference across the complete assistant response.
  - No trajectory grouping is needed -- every turn is an independent target.

Starting point: SFT LoRA checkpoint. Only LoRA params updated; base frozen.

Input: rollout.jsonl (output of rl_rollout.py)
  Each line: {problem_id, turn, agent_id, messages, response, reward, ...}

Output: updated LoRA adapters per agent

Usage:
    accelerate launch --num_processes 1 \\
        scripts/rl_train.py \\
        --agent A1 \\
        --rollout rl_data/rollout_train_0_500.jsonl \\
        --sft-adapter sft_runs/0706v2/A1/final \\
        --out-dir rl_runs/v1_0707/A1 \\
        --model-name-or-path /data/wangyuheng/models/Qwen3-1.7B
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jca.src.protocol_json import (  # noqa: E402
    protocol_json_decoder as _strict_protocol_json_decoder,
    protocol_json_well_formed as _strict_protocol_json_well_formed,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RWR RL training for one JCA agent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Identity
    p.add_argument("--agent", required=True, choices=["A1", "A2", "A3", "SAS"])
    p.add_argument("--rollout", type=Path, required=True,
                   help="rollout.jsonl from rl_rollout.py")
    p.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Explicit chat-template mode. If omitted, a homogeneous mode is "
            "read from enable_thinking/thinking_enabled in every rollout row."
        ),
    )
    p.add_argument(
        "--reward-field", default="reward",
        help="Numeric signed weight field to optimize (e.g. reward or train_weight).",
    )
    p.add_argument(
        "--sft-adapter",
        type=Path,
        default=None,
        help="Optional LoRA adapter to start from; omit to initialize a fresh LoRA on the base model",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--model-name-or-path", default=None)

    # RWR
    p.add_argument("--kl-coef", type=float, default=0.05,
                   help="KL penalty coefficient against SFT reference (0 = disable)")

    # Training
    p.add_argument("--num-epochs", type=float, default=1.0)
    p.add_argument("--learning-rate", type=float, default=5e-6)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--per-device-batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--max-seq-length", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)

    # LoRA (must match the starting adapter when one is supplied)
    p.add_argument("--lora-rank", type=int, default=64)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    # Logging
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Optional execution limit for recovery/smoke runs; zero means no limit.",
    )
    p.add_argument("--report-to", default="tensorboard")

    # Misc
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Accelerate checkpoint directory, or 'latest' to resume the newest complete checkpoint.",
    )

    return p.parse_args()


def resolve_resume_checkpoint(out_dir: Path, requested: Optional[str]) -> Optional[Path]:
    if not requested:
        return None
    if requested != "latest":
        path = Path(requested).resolve()
        if not (path / "training_state.json").is_file():
            raise ValueError(f"resume checkpoint is incomplete: {path}")
        return path
    candidates = []
    for path in out_dir.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-([0-9]+)", path.name)
        if match and (path / "training_state.json").is_file():
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None))[1]


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def register_compact_peft_checkpoint_hooks(accelerator: Any) -> None:
    """Save only trainable PEFT weights while Accelerate saves other state."""

    def save_peft_model_hook(
        models: List[Any], weights: List[Dict[str, Any]], output_dir: str
    ) -> None:
        if len(models) != 1:
            raise RuntimeError(f"expected one policy model, found {len(models)}")
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(models[0])
            unwrapped.save_pretrained(
                str(Path(output_dir) / "policy_adapter"),
                safe_serialization=True,
            )
        # Tell Accelerate that the model was handled by this hook. Optimizer,
        # scheduler, RNG, and registered custom states are still saved below.
        weights.clear()

    def load_peft_model_hook(models: List[Any], input_dir: str) -> None:
        if len(models) != 1:
            raise RuntimeError(f"expected one policy model, found {len(models)}")
        adapter_dir = Path(input_dir) / "policy_adapter"
        adapter_file = adapter_dir / "adapter_model.safetensors"
        if not adapter_file.is_file():
            raise RuntimeError(f"PEFT checkpoint is missing adapter weights: {adapter_file}")
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        unwrapped = accelerator.unwrap_model(models[0])
        result = set_peft_model_state_dict(
            unwrapped,
            load_file(str(adapter_file), device="cpu"),
            adapter_name="default",
        )
        # Missing base-model keys are expected because this file deliberately
        # contains adapter tensors only. Unexpected adapter keys are not.
        if result.unexpected_keys:
            raise RuntimeError(
                "PEFT checkpoint key mismatch: "
                f"unexpected={result.unexpected_keys[:5]}"
            )
        # Prevent Accelerate from looking for a full model.safetensors file.
        models.clear()

    accelerator.register_save_state_pre_hook(save_peft_model_hook)
    accelerator.register_load_state_pre_hook(load_peft_model_hook)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(out_dir: Path, agent: str) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / f"rl_{agent}.log"
    logger = logging.getLogger("jca.rl")
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    if os.environ.get("LOCAL_RANK", "0") == "0":
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class RolloutRecord:
    problem_id:  str
    turn:        int
    agent_id:    str
    messages:    List[Dict[str, str]]
    response:    str
    reward:      float = 0.0   # signed reward, typically in [-1, 1]
    enable_thinking: Optional[bool] = None

    @property
    def thinking_enabled(self) -> Optional[bool]:
        return self.enable_thinking


_SEMANTIC_JSON_FIELDS = frozenset(
    {
        "reasoning",
        "final_answer",
        "tentative_answer",
        "action",
        "handoff_target",
        "handoff_note",
        "confirmed_answer",
    }
)
_THINK_BLOCK_RE = re.compile(
    r"<think\b[^>]*>(.*?)</think\s*>", re.IGNORECASE | re.DOTALL
)
_ASSISTANT_MARKERS = (
    "<|im_start|>assistant",
    "<|assistant|>",
    "### Assistant:",
    "\nassistant\n",
)


def _protocol_json_well_formed(visible: str) -> bool:
    return _strict_protocol_json_well_formed(visible)


def _coerce_thinking_flag(value: Any, *, field: str, row_number: int) -> Optional[bool]:
    """Parse a thinking flag without treating arbitrary strings as true."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, float) and value in (0.0, 1.0):
        return bool(int(value))
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on", "thinking"}:
            return True
        if normalized in {"0", "false", "no", "off", "nonthinking", "non-thinking"}:
            return False
    raise ValueError(
        f"invalid {field} value at rollout line {row_number}: {value!r}; "
        "expected a boolean"
    )


def _row_thinking_flag(row: Dict[str, Any], row_number: int) -> Optional[bool]:
    """Read both historical thinking-field spellings and reject conflicts."""
    values: List[Tuple[str, Optional[bool]]] = []
    for field in ("enable_thinking", "thinking_enabled"):
        if field in row:
            values.append(
                (field, _coerce_thinking_flag(row.get(field), field=field, row_number=row_number))
            )
    present = [(field, value) for field, value in values if value is not None]
    if not present:
        return None
    unique = {value for _, value in present}
    if len(unique) != 1:
        raise ValueError(
            f"conflicting thinking flags at rollout line {row_number}: "
            + ", ".join(f"{field}={value}" for field, value in present)
        )
    return present[0][1]


def load_rollouts(
    path: Path,
    agent: str,
    reward_field: str = "reward",
    *,
    enable_thinking: Optional[bool] = None,
) -> List[RolloutRecord]:
    """Load all rollout records for one agent. Each turn is one record."""
    records = []

    observed_modes: List[bool] = []
    inferred_modes: List[bool] = []
    missing_mode_rows = 0
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in rollout file at line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(r, dict):
                raise ValueError(
                    f"rollout JSONL row must be an object at line {line_number}"
                )

            if r.get("agent_id") != agent:
                continue
            if not r.get("messages") or not r.get("response"):
                raise ValueError(
                    f"rollout record for agent {agent} is missing messages/response "
                    f"at line {line_number}"
                )

            if reward_field not in r:
                raise ValueError(
                    f"rollout record is missing {reward_field!r} field "
                    f"(problem_id={r.get('problem_id')}, turn={r.get('turn')}). "
                    f"Regenerate data with the current builder — old "
                    f"rollouts written before the per-turn reward change are "
                    f"incompatible."
                )
            try:
                reward = float(r[reward_field])
            except (TypeError, ValueError):
                raise ValueError(
                    f"rollout record has a non-numeric {reward_field!r} field "
                    f"at line {line_number}"
                )
            if not math.isfinite(reward):
                raise ValueError(
                    f"rollout record has a non-finite {reward_field!r} field "
                    f"at line {line_number}"
                )

            row_mode = _row_thinking_flag(r, line_number)
            if row_mode is None:
                missing_mode_rows += 1
                # Legacy rows predate the explicit mode fields.  Infer only
                # from the unambiguous response envelope so direct callers
                # remain compatible; the launcher still passes an explicit
                # mode for production runs.
                row_mode = _infer_response_thinking_flag(
                    r.get("response"), r.get("thinking")
                )
                if row_mode is not None:
                    inferred_modes.append(row_mode)
                elif enable_thinking is not None:
                    # An explicitly requested mode is authoritative for
                    # legacy rows that carry no mode metadata.  The response
                    # validator below still rejects an incompatible envelope.
                    row_mode = bool(enable_thinking)
            else:
                observed_modes.append(row_mode)
            if enable_thinking is not None and row_mode is not None and row_mode != enable_thinking:
                raise ValueError(
                    f"thinking mode mismatch at rollout line {line_number}: "
                    f"row={row_mode}, requested={enable_thinking}"
                )

            resolved_row_mode = bool(row_mode) if row_mode is not None else False
            response = _training_response_for_mode(r, resolved_row_mode, line_number)
            records.append(RolloutRecord(
                problem_id  = r["problem_id"],
                turn        = r["turn"],
                agent_id    = r["agent_id"],
                messages    = r["messages"],
                response    = response,
                reward      = reward,
                enable_thinking = row_mode,
            ))

    if not records:
        return records
    if enable_thinking is None:
        unique_modes = set(observed_modes)
        unique_modes.update(inferred_modes)
        if len(unique_modes) > 1:
            raise ValueError(
                "rollout mixes thinking and non-thinking records; pass an explicit "
                "--enable-thinking/--no-enable-thinking and regenerate a homogeneous file"
            )
        if missing_mode_rows and not inferred_modes:
            raise ValueError(
                f"{missing_mode_rows} rollout records have no thinking_enabled/enable_thinking "
                "field; pass an explicit --enable-thinking or --no-enable-thinking"
            )
        resolved_mode = next(iter(unique_modes), None)
    else:
        resolved_mode = bool(enable_thinking)

    if resolved_mode is None:
        raise ValueError("cannot resolve rollout thinking mode")
    for record in records:
        record.enable_thinking = resolved_mode

    return records


def _infer_response_thinking_flag(
    response: Any, thinking: Any = None
) -> Optional[bool]:
    """Infer a legacy row's mode from tags or a stored thinking trace."""
    if isinstance(thinking, str) and thinking.strip():
        return True
    if not isinstance(response, str) or not response.strip():
        return None
    matches = list(_THINK_BLOCK_RE.finditer(response))
    if matches:
        return True
    if re.search(r"<\/?think\b", response, flags=re.IGNORECASE):
        # A malformed/unclosed tag still clearly indicates the thinking mode.
        return True
    return False


def _training_response_for_mode(row: Dict[str, Any], mode: bool, line_number: int) -> str:
    """Validate/normalize a rollout response for the resolved mode."""
    response = row.get("response")
    if not isinstance(response, str) or not response:
        raise ValueError(f"response must be a non-empty string at rollout line {line_number}")
    matches = list(_THINK_BLOCK_RE.finditer(response))
    has_tag = bool(matches) or bool(re.search(r"<\/?think\b", response, flags=re.IGNORECASE))
    has_nonempty = any(match.group(1).strip() for match in matches)
    if mode:
        if not has_nonempty:
            # Recover traces from legacy rows that stored them in a separate
            # field; never train an empty Qwen thinking scaffold.
            thinking = str(row.get("thinking") or "").strip()
            if thinking:
                if has_tag and not matches:
                    raise ValueError(
                        f"thinking-enabled rollout has malformed think tags at line {line_number}"
                    )
                # Drop empty scaffolds emitted by Qwen's non-thinking branch
                # before restoring a real legacy trace.  Otherwise the target
                # would contain two thinking blocks, one of them empty.
                visible_response = _THINK_BLOCK_RE.sub("", response).lstrip()
                response = f"<think>\n{thinking}\n</think>\n{visible_response}"
            else:
                raise ValueError(
                    f"thinking-enabled rollout has no non-empty thinking trace at line {line_number}"
                )
    elif has_tag:
        raise ValueError(f"non-thinking rollout contains think tags at line {line_number}")
    visible = _THINK_BLOCK_RE.sub("", response).strip()
    if not _protocol_json_well_formed(visible):
        raise ValueError(
            f"rollout response is not exactly one strict JSON object at line {line_number}"
        )
    return response


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def encode_one(
    messages:       List[Dict[str, str]],
    response:       str,
    tokenizer,
    max_seq_length: int,
    enable_thinking: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """Tokenize one turn and return response plus semantic-token masks.

    A separately rendered ``prompt_text`` is useful as a fallback, but its
    token sequence is not assumed to be a prefix of the rendered full
    conversation.  Qwen templates intentionally change the assistant scaffold
    depending on whether earlier turns are present.  We therefore locate the
    final assistant span in the rendered text and use tokenizer offsets (or a
    token-subsequence fallback) to construct the labels.
    """
    if enable_thinking is None:
        raise ValueError(
            "encode_one requires an explicit thinking mode; pass True or False"
        )
    if not isinstance(response, str) or not response:
        return None

    messages_with_response = list(messages) + [
        {"role": "assistant", "content": response}
    ]
    template_kwargs: Dict[str, Any] = {"enable_thinking": bool(enable_thinking)}
    full_text = _apply_chat_template(
        tokenizer,
        messages_with_response,
        add_generation_prompt=False,
        template_kwargs=template_kwargs,
    )
    prompt_text = _apply_chat_template(
        tokenizer,
        messages,
        add_generation_prompt=True,
        template_kwargs=template_kwargs,
    )

    full_encoding = _tokenize_text(
        tokenizer,
        full_text,
        max_seq_length=max_seq_length,
        return_offsets=True,
    )
    full_ids = list(full_encoding["input_ids"])
    prompt_ids = list(
        _tokenize_text(
            tokenizer,
            prompt_text,
            max_seq_length=max_seq_length,
            return_offsets=False,
        )["input_ids"]
    )
    if len(full_ids) >= max_seq_length or not full_ids:
        return None

    assistant_start, assistant_end = _assistant_char_span(
        full_text, response, enable_thinking=bool(enable_thinking)
    )
    offsets = full_encoding.get("offset_mapping")
    response_mask, mask_strategy = _char_or_token_mask(
        tokenizer,
        full_text,
        full_ids,
        offsets,
        assistant_start,
        assistant_end,
        response,
        prompt_ids,
    )
    if not any(response_mask):
        return None

    semantic_spans = _semantic_char_spans(full_text[assistant_start:assistant_end])
    semantic_spans = [
        (assistant_start + start, assistant_start + end)
        for start, end in semantic_spans
    ]
    semantic_mask = _mask_from_offsets(
        offsets,
        len(full_ids),
        semantic_spans,
        require_contained=True,
        fill_gaps=False,
    )
    if not any(semantic_mask):
        # A slow tokenizer has no offsets.  Locate semantic snippets in the
        # rendered assistant segment; if that fails, leave the mask empty so a
        # malformed negative example cannot teach away the protocol format.
        semantic_mask = _semantic_token_mask_fallback(
            tokenizer,
            full_ids,
            full_text[assistant_start:assistant_end],
            response,
            response_mask,
        )
    semantic_mask = [bool(active and target) for active, target in zip(semantic_mask, response_mask)]

    labels = [token_id if active else -100 for token_id, active in zip(full_ids, response_mask)]
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "response_mask": response_mask,
        "semantic_mask": semantic_mask,
        "mask_strategy": mask_strategy,
    }


def _apply_chat_template(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    *,
    add_generation_prompt: bool,
    template_kwargs: Dict[str, Any],
) -> str:
    """Render a chat template while keeping the requested mode explicit."""
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **template_kwargs,
        )
    except TypeError as exc:
        # Some lightweight test/dummy tokenizers have no ``**kwargs``.  Retry
        # only for compatibility; production Qwen tokenizers accept the flag.
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
        except TypeError:
            raise exc
    if not isinstance(rendered, str):
        raise TypeError("chat template must return a string when tokenize=False")
    return rendered


def _tokenize_text(
    tokenizer: Any,
    text: str,
    *,
    max_seq_length: int,
    return_offsets: bool,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "add_special_tokens": False,
        "truncation": True,
        "max_length": max_seq_length,
        "return_attention_mask": False,
    }
    if return_offsets:
        kwargs["return_offsets_mapping"] = True
    try:
        encoding = tokenizer(text, **kwargs)
    except (NotImplementedError, TypeError):
        if not return_offsets:
            raise
        kwargs.pop("return_offsets_mapping", None)
        encoding = tokenizer(text, **kwargs)
    if isinstance(encoding, dict) or (
        hasattr(encoding, "keys") and "input_ids" in encoding
    ):
        return dict(encoding)
    # A few tiny tokenizers return a bare list for input_ids.
    return {"input_ids": encoding}


def _find_subsequence(sequence: List[int], pattern: List[int], start: int = 0) -> int:
    if not pattern or len(pattern) > len(sequence):
        return -1
    limit = len(sequence) - len(pattern) + 1
    for index in range(max(0, start), limit):
        if sequence[index : index + len(pattern)] == pattern:
            return index
    return -1


def _find_last_subsequence(sequence: List[int], pattern: List[int]) -> int:
    if not pattern or len(pattern) > len(sequence):
        return -1
    found = -1
    start = 0
    while True:
        candidate = _find_subsequence(sequence, pattern, start)
        if candidate < 0:
            return found
        found = candidate
        start = candidate + 1


def _assistant_char_span(
    full_text: str,
    response: str,
    *,
    enable_thinking: bool,
) -> Tuple[int, int]:
    """Find the final assistant message boundaries in rendered chat text."""
    if not response:
        return 0, 0
    response_start = -1
    response_end = -1
    visible = response.rsplit("</think>", 1)[-1].lstrip("\n")
    expected_prefixes = tuple(
        prefix[:32]
        for prefix in (response.lstrip(), visible.lstrip(), "<think>", "{")
        if prefix
    )

    def marker_matches_response(position: int, marker: str) -> bool:
        suffix = full_text[position + len(marker) :].lstrip()
        return any(suffix.startswith(prefix) for prefix in expected_prefixes)

    valid_marker_positions = []
    for marker in _ASSISTANT_MARKERS:
        search_end = len(full_text)
        while search_end > 0:
            position = full_text.rfind(marker, 0, search_end)
            if position < 0:
                break
            if marker_matches_response(position, marker):
                valid_marker_positions.append((position, len(marker)))
                break
            search_end = position
    latest_valid_marker = max(valid_marker_positions, default=(-1, 0))

    response_candidates = []
    for candidate_text in (response, visible):
        if not candidate_text:
            continue
        candidate = full_text.rfind(candidate_text)
        if candidate >= 0:
            candidate_end = candidate + len(candidate_text)
            marker_after = latest_valid_marker[0]
            if marker_after > candidate_end:
                continue
            response_candidates.append(
                (candidate_end, candidate, candidate_text == response)
            )
    if response_candidates:
        _, response_start, _ = max(
            response_candidates,
            key=lambda item: (item[0], item[2]),
        )
        response_end = max(
            candidate[0]
            for candidate in response_candidates
            if candidate[1] == response_start
        )

    marker_start = -1
    marker_length = 0
    if response_start >= 0:
        for marker in _ASSISTANT_MARKERS:
            candidate = full_text.rfind(marker, 0, response_start + 1)
            if candidate > marker_start:
                marker_start = candidate
                marker_length = len(marker)
    else:
        marker_start, marker_length = latest_valid_marker
        if marker_start < 0:
            for marker in _ASSISTANT_MARKERS:
                candidate = full_text.rfind(marker)
                if candidate > marker_start:
                    marker_start = candidate
                    marker_length = len(marker)

    if response_start < 0:
        if marker_start < 0:
            response_start = full_text.rfind(response)
            response_end = response_start + len(response) if response_start >= 0 else -1

    if marker_start < 0:
        marker_start = response_start
        marker_length = 0
    if marker_start < 0:
        # Neither the response nor an assistant scaffold can be located.  A
        # zero-origin span would silently supervise the user/system prompt.
        return -1, -1

    content_start = response_start if response_start >= 0 else marker_start + marker_length
    # A Qwen3 template emits an empty ``<think></think>`` scaffold even when
    # the assistant target is plain JSON.  Include that block only when the
    # supplied response actually carries a non-empty thinking trace; otherwise
    # an accidental mode mismatch would train the empty scaffold as a target.
    has_nonempty_thinking = any(
        match.group(1).strip() for match in _THINK_BLOCK_RE.finditer(response)
    )
    if enable_thinking and has_nonempty_thinking:
        think_marker = full_text.find("<think", marker_start, max(content_start, marker_start) + 1)
        if think_marker >= 0:
            content_start = think_marker
    content_end = response_end if response_end >= 0 else len(full_text)
    end_marker = "<|im_end|>"
    end_position = full_text.rfind(end_marker, max(content_end, marker_start))
    if end_position >= content_end:
        assistant_end = end_position + len(end_marker)
    else:
        end_positions = [
            full_text.find(marker, max(content_end, marker_start))
            for marker in ("<|assistant|>", "### Assistant:")
        ]
        end_positions = [position for position in end_positions if position >= 0]
        assistant_end = min(end_positions) if end_positions else content_end
    assistant_start = max(marker_start, min(len(full_text), content_start))
    assistant_end = max(assistant_start, min(len(full_text), assistant_end))
    return assistant_start, assistant_end


def _mask_from_offsets(
    offsets: Any,
    length: int,
    spans: List[Tuple[int, int]],
    *,
    require_contained: bool = False,
    fill_gaps: bool = True,
) -> List[bool]:
    mask = [False] * length
    if offsets is None or not spans:
        return mask
    all_offsets = list(offsets)
    if len(all_offsets) == 1 and isinstance(all_offsets[0], (list, tuple)):
        nested = all_offsets[0]
        if nested and isinstance(nested[0], (list, tuple)):
            all_offsets = list(nested)
    all_offsets = all_offsets[:length]
    for span_start, span_end in spans:
        overlap: List[int] = []
        for index, offset in enumerate(all_offsets):
            try:
                start, end = int(offset[0]), int(offset[1])
            except (TypeError, ValueError, IndexError):
                continue
            overlaps = end > span_start and start < span_end
            contained = start >= span_start and end <= span_end
            if end > start and overlaps and (contained or not require_contained):
                overlap.append(index)
        if overlap:
            if fill_gaps:
                for index in range(min(overlap), max(overlap) + 1):
                    mask[index] = True
            else:
                for index in overlap:
                    mask[index] = True
    return mask


def _char_or_token_mask(
    tokenizer: Any,
    full_text: str,
    full_ids: List[int],
    offsets: Any,
    start: int,
    end: int,
    response: str,
    prompt_ids: List[int],
) -> Tuple[List[bool], str]:
    span_is_valid = start >= 0 and end > start and end <= len(full_text)
    if span_is_valid:
        mask = _mask_from_offsets(offsets, len(full_ids), [(start, end)])
        if any(mask):
            return mask, "assistant_char_offsets"
    else:
        mask = [False] * len(full_ids)

    response_ids = list(
        _tokenize_text(
            tokenizer,
            response,
            max_seq_length=max(len(full_ids), 1),
            return_offsets=False,
        )["input_ids"]
    )
    response_index = _find_last_subsequence(full_ids, response_ids)
    if response_index >= 0:
        if not span_is_valid:
            prompt_is_prefix = bool(prompt_ids) and full_ids[: len(prompt_ids)] == prompt_ids
            response_end = response_index + len(response_ids)
            if prompt_is_prefix and response_index < len(prompt_ids):
                if full_ids != response_ids:
                    return mask, "unresolved"
            elif not prompt_is_prefix and response_end < len(full_ids):
                return mask, "unresolved"
        mask[response_index : response_index + len(response_ids)] = [True] * len(response_ids)
        return mask, "response_token_subsequence"

    # This is deliberately a fallback, not the normal path.  It uses the
    # longest common token prefix only when no rendered assistant anchor can
    # be found, so multi-turn scaffold differences cannot silently shift all
    # labels.
    if span_is_valid and prompt_ids and full_ids[: len(prompt_ids)] == prompt_ids:
        response_start = len(prompt_ids)
        if response_start < len(full_ids):
            mask[response_start:] = [True] * (len(full_ids) - response_start)
            return mask, "verified_prompt_prefix_fallback"
    return mask, "unresolved"


def _json_value_spans(text: str) -> List[Tuple[int, int]]:
    """Return character spans of protocol values, excluding JSON syntax."""
    decoder = _strict_protocol_json_decoder()
    candidates: List[Tuple[int, int, Dict[str, Any]]] = []
    for match in re.finditer(r"\{", text):
        object_start = match.start()
        try:
            parsed, consumed = decoder.raw_decode(text[object_start:])
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if isinstance(parsed, dict):
            candidates.append((object_start, object_start + consumed, parsed))
    if not candidates:
        return []

    # Prefer the object containing the most recognized protocol fields.  A
    # nested object or a JSON-looking fragment inside reasoning can otherwise
    # win merely because its opening brace appears later.
    object_start, object_end, parsed = max(
        candidates,
        key=lambda item: (
            sum(key in _SEMANTIC_JSON_FIELDS for key in item[2]),
            item[1] - item[0],
            item[0],
        ),
    )
    spans: List[Tuple[int, int]] = []
    index = object_start + 1
    while index < object_end:
        while index < object_end and text[index].isspace():
            index += 1
        if index >= object_end or text[index] == "}":
            break
        try:
            key, consumed = decoder.raw_decode(text[index:])
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
        if not isinstance(key, str):
            return []
        index += consumed
        while index < object_end and text[index].isspace():
            index += 1
        if index >= object_end or text[index] != ":":
            return []
        index += 1
        while index < object_end and text[index].isspace():
            index += 1
        value_start = index
        try:
            value, consumed = decoder.raw_decode(text[index:])
        except (TypeError, ValueError, json.JSONDecodeError):
            return []
        value_end = index + consumed
        if key in _SEMANTIC_JSON_FIELDS and key in parsed and value is not None:
            if isinstance(value, str) and value_end - value_start >= 2:
                spans.append((value_start + 1, value_end - 1))
            else:
                spans.append((value_start, value_end))
        index = value_end
        while index < object_end and text[index].isspace():
            index += 1
        if index < object_end and text[index] == ",":
            index += 1
            continue
        if index < object_end and text[index] == "}":
            break
        return []
    return spans


def _semantic_char_spans(assistant_text: str) -> List[Tuple[int, int]]:
    """Identify semantic content while leaving protocol scaffolding untouched."""
    spans = [
        (match.start(1), match.end(1))
        for match in _THINK_BLOCK_RE.finditer(assistant_text)
        if match.group(1).strip()
    ]
    spans.extend(_json_value_spans(assistant_text))
    if not spans and "{" not in assistant_text:
        cleaned = re.sub(r"<\/?think\b[^>]*>", "", assistant_text, flags=re.IGNORECASE)
        if cleaned.strip():
            start = assistant_text.find(cleaned.strip())
            if start >= 0:
                spans.append((start, start + len(cleaned.strip())))
    return spans


def _semantic_token_mask_fallback(
    tokenizer: Any,
    full_ids: List[int],
    assistant_text: str,
    response: str,
    response_mask: List[bool],
) -> List[bool]:
    spans = _semantic_char_spans(assistant_text)
    if not spans:
        # Do not apply a negative policy gradient to an unparsed protocol.
        return [False] * len(full_ids)
    output = [False] * len(full_ids)
    for start, end in spans:
        snippet = assistant_text[start:end]
        if not snippet:
            continue
        snippet_ids = list(
            _tokenize_text(
                tokenizer,
                snippet,
                max_seq_length=max(len(full_ids), 1),
                return_offsets=False,
            )["input_ids"]
        )
        # Prefer the last occurrence: the same semantic value can appear in a
        # user prompt or an earlier assistant turn, while this mask belongs to
        # the final assistant span.
        index = _find_last_subsequence(full_ids, snippet_ids)
        if index < 0:
            continue
        for token_index in range(index, min(len(full_ids), index + len(snippet_ids))):
            output[token_index] = True
    return [active and target for active, target in zip(output, response_mask)]


# ---------------------------------------------------------------------------
# RWR loss
# ---------------------------------------------------------------------------

_DEFAULT_LOGITS_CHUNK_SIZE = 64


def _rwr_logits_chunk_size() -> int:
    """Return a conservative projection chunk size for large vocabularies."""
    value = os.environ.get("JCA_RWR_LOGITS_CHUNK_SIZE", "")
    if not value:
        return _DEFAULT_LOGITS_CHUNK_SIZE
    try:
        parsed = int(value)
    except ValueError:
        return _DEFAULT_LOGITS_CHUNK_SIZE
    return max(1, min(parsed, 512))


def _unwrap_for_introspection(model: Any) -> Any:
    """Unwrap DDP without changing the module used for its forward call."""
    current = model
    while hasattr(current, "module"):
        module = getattr(current, "module")
        if module is current:
            break
        current = module
    return current


def _causal_lm_components(model: Any) -> Optional[Tuple[Any, Any, Any]]:
    """Find the causal-LM, transformer backbone, and vocabulary head."""
    unwrapped = _unwrap_for_introspection(model)
    get_base_model = getattr(unwrapped, "get_base_model", None)
    causal_lm = get_base_model() if callable(get_base_model) else unwrapped
    backbone = getattr(causal_lm, "model", None)
    lm_head = getattr(causal_lm, "lm_head", None)
    if backbone is None or lm_head is None or not callable(backbone) or not callable(lm_head):
        return None
    return causal_lm, backbone, lm_head


def _capture_hidden_states(
    model: Any,
    backbone: Any,
    batch_input_ids: Any,
    batch_attention_mask: Any,
) -> Any:
    """Run the normal model forward while suppressing the full-vocabulary head."""
    captured: Dict[str, Any] = {}

    def capture(_module: Any, _inputs: Any, output: Any) -> None:
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None and isinstance(output, (tuple, list)):
            hidden = output[0]
        # Lightweight causal-LM backbones (and a few older model wrappers)
        # return the hidden tensor directly instead of a ModelOutput object.
        # Treat a rank-3 tensor as the final hidden-state sequence; otherwise
        # the chunked token/KL path would fail before its compatibility
        # fallback can be selected.
        if hidden is None and hasattr(output, "ndim") and int(output.ndim) >= 3:
            hidden = output
        if hidden is not None:
            captured["hidden"] = hidden

    handle = backbone.register_forward_hook(capture)
    outputs = None
    try:
        try:
            # Qwen3 (and recent Transformers causal-LM implementations) uses
            # this argument to compute only one tiny sentinel row of logits.
            # The hook still receives the complete final hidden-state tensor.
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                use_cache=False,
                logits_to_keep=1,
            )
        except TypeError:
            # Older model versions may not expose logits_to_keep.  Calling the
            # backbone directly is still memory-safe and keeps the same LoRA
            # modules active; this branch is mainly for compatibility.
            outputs = backbone(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                use_cache=False,
            )
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None and isinstance(outputs, (tuple, list)):
                hidden = outputs[0]
            if hidden is None and hasattr(outputs, "ndim") and int(outputs.ndim) >= 3:
                hidden = outputs
            if hidden is not None:
                captured["hidden"] = hidden
    finally:
        handle.remove()

    hidden = captured.get("hidden")
    if hidden is None:
        raise RuntimeError("causal-LM forward did not expose last hidden states")
    if outputs is not None:
        del outputs
    return hidden


def _full_token_log_probs(
    model: Any,
    batch_input_ids: Any,
    batch_attention_mask: Any,
    shift_labels: Any,
) -> Any:
    """Compatibility path returning one target log-probability per position."""
    import torch.nn.functional as F

    outputs = model(
        input_ids=batch_input_ids,
        attention_mask=batch_attention_mask,
    )
    shift_logits = outputs.logits[:, :-1, :]
    safe_labels = shift_labels.clamp(min=0)
    return F.log_softmax(shift_logits.float(), dim=-1).gather(
        2, safe_labels.unsqueeze(-1)
    ).squeeze(-1)


def _token_log_probs(
    model: Any,
    batch_input_ids: Any,
    batch_attention_mask: Any,
    shift_labels: Any,
    shift_mask: Any,
) -> Any:
    """Compute target log probabilities without materializing full logits."""
    import torch
    from torch.utils.checkpoint import checkpoint

    components = _causal_lm_components(model)
    if components is None:
        return _full_token_log_probs(
            model,
            batch_input_ids,
            batch_attention_mask,
            shift_labels,
        )

    _causal_lm, backbone, lm_head = components
    valid_positions = torch.nonzero(shift_mask.any(dim=0), as_tuple=False).flatten()
    token_values = torch.zeros(
        batch_input_ids.shape[0],
        shift_labels.shape[1],
        device=batch_input_ids.device,
        dtype=torch.float32,
    )
    if valid_positions.numel() == 0:
        return token_values

    hidden_states = _capture_hidden_states(
        model,
        backbone,
        batch_input_ids,
        batch_attention_mask,
    )
    vocab_size = getattr(lm_head, "out_features", None)
    if vocab_size is None:
        vocab_size = lm_head.weight.shape[0]
    vocab_size = int(vocab_size)
    chunk_size = _rwr_logits_chunk_size()

    def project_chunk(hidden_chunk: Any, labels_chunk: Any) -> Any:
        logits = lm_head(hidden_chunk)
        safe_labels = labels_chunk.clamp(min=0, max=vocab_size - 1)
        target_logits = logits.gather(
            -1, safe_labels.unsqueeze(-1)
        ).squeeze(-1)
        log_norm = torch.logsumexp(logits.float(), dim=-1)
        return target_logits.float() - log_norm

    for start in range(0, valid_positions.numel(), chunk_size):
        positions = valid_positions[start : start + chunk_size]
        hidden_chunk = hidden_states.index_select(1, positions)
        labels_chunk = shift_labels.index_select(1, positions)
        if torch.is_grad_enabled() and hidden_chunk.requires_grad:
            chunk_values = checkpoint(
                project_chunk,
                hidden_chunk,
                labels_chunk,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            chunk_values = project_chunk(hidden_chunk, labels_chunk)
        token_values = token_values.index_copy(1, positions, chunk_values.float())
    return token_values


def _full_sequence_log_probs(
    model: Any,
    batch_input_ids: Any,
    batch_attention_mask: Any,
    shift_labels: Any,
    shift_mask: Any,
) -> Any:
    """Backward-compatible averaged-log-probability helper."""
    return _masked_sequence_log_probs(
        _full_token_log_probs(
            model,
            batch_input_ids,
            batch_attention_mask,
            shift_labels,
        ),
        shift_mask,
    )


def _masked_sequence_log_probs(token_log_probs: Any, mask: Any) -> Any:
    import torch

    mask_float = mask.float()
    lengths = mask_float.sum(dim=1).clamp(min=1.0)
    values = (token_log_probs * mask_float).sum(dim=1) / lengths
    # A zero-length semantic mask must contribute no policy gradient.
    return torch.where(mask_float.sum(dim=1) > 0, values, torch.zeros_like(values))


def _sequence_log_probs(
    model: Any,
    batch_input_ids: Any,
    batch_attention_mask: Any,
    shift_labels: Any,
    shift_mask: Any,
) -> Any:
    """Compute masked mean assistant-token log probabilities."""
    return _masked_sequence_log_probs(
        _token_log_probs(
            model,
            batch_input_ids,
            batch_attention_mask,
            shift_labels,
            shift_mask,
        ),
        shift_mask,
    )


def _token_level_kl_from_logits(policy_logits: Any, reference_logits: Any) -> Any:
    """Exact per-token ``KL(policy || reference)`` from two logit tensors."""
    import torch.nn.functional as F

    if policy_logits.shape != reference_logits.shape:
        raise ValueError(
            "policy and reference logits must have identical shapes for token KL"
        )
    if policy_logits.ndim < 1 or policy_logits.shape[-1] == 0:
        raise ValueError("logit tensors must have a non-empty vocabulary dimension")
    policy_log_probs = F.log_softmax(policy_logits.float(), dim=-1)
    reference_log_probs = F.log_softmax(reference_logits.float(), dim=-1)
    policy_probs = policy_log_probs.exp()
    # Roundoff can produce a tiny negative value for nearly identical logits;
    # the mathematical KL is non-negative and the clamp keeps the loss anchor
    # from becoming an unintended reward.
    return (policy_probs * (policy_log_probs - reference_log_probs)).sum(dim=-1).clamp_min(0.0)


def _token_log_probs_and_kl(
    model: Any,
    reference_model: Any,
    batch_input_ids: Any,
    batch_attention_mask: Any,
    shift_labels: Any,
    shift_mask: Any,
    *,
    shared_base_reference: bool = False,
) -> Tuple[Any, Optional[Any]]:
    """Compute response log-probs and token KL in one policy forward pass.

    The policy hidden states are the expensive autograd-bearing object.  The
    previous implementation built one policy graph for RWR and another for KL,
    which could exhaust GPU memory on long A3 sequences.  This helper projects
    each vocabulary chunk once and derives both quantities from that same
    policy-logit tensor.  The reference branch remains inference-only.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    need_kl = reference_model is not None or shared_base_reference
    policy_components = _causal_lm_components(model)
    reference_components = (
        policy_components
        if shared_base_reference
        else _causal_lm_components(reference_model)
        if need_kl
        else None
    )
    valid_positions = torch.nonzero(shift_mask.any(dim=0), as_tuple=False).flatten()
    token_values = torch.zeros(
        batch_input_ids.shape[0],
        shift_labels.shape[1],
        device=batch_input_ids.device,
        dtype=torch.float32,
    )
    if not need_kl:
        kl_values: Optional[Any] = None
    else:
        lengths = shift_mask.sum(dim=1).clamp(min=1).float()
        kl_values = torch.zeros(
            batch_input_ids.shape[0],
            device=batch_input_ids.device,
            dtype=torch.float32,
        )
    if valid_positions.numel() == 0:
        if kl_values is not None:
            lengths = shift_mask.sum(dim=1).clamp(min=1).float()
            return token_values, kl_values / lengths
        return token_values, None

    # Tiny/mock models and unusual wrappers may not expose a separable
    # backbone and vocabulary head.  Keep the full-logit compatibility path;
    # production Qwen models use the chunked path below.
    if policy_components is None or (need_kl and reference_components is None):
        policy_outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
        )
        policy_logits = policy_outputs.logits[:, :-1, :]
        safe_labels = shift_labels.clamp(min=0)
        policy_log_probs = F.log_softmax(policy_logits.float(), dim=-1)
        token_values = policy_log_probs.gather(
            2, safe_labels.unsqueeze(-1)
        ).squeeze(-1)
        if not need_kl:
            return token_values, None
        if shared_base_reference:
            with _adapter_disabled(model):
                with torch.no_grad():
                    reference_outputs = model(
                        input_ids=batch_input_ids,
                        attention_mask=batch_attention_mask,
                    )
        else:
            with torch.no_grad():
                reference_outputs = reference_model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                )
        reference_logits = reference_outputs.logits[:, :-1, :]
        token_kl = _token_level_kl_from_logits(policy_logits, reference_logits)
        lengths = shift_mask.sum(dim=1).clamp(min=1).float()
        kl_values = (token_kl * shift_mask.float()).sum(dim=1) / lengths
        return token_values, kl_values

    _policy_lm, policy_backbone, policy_head = policy_components
    _reference_lm, reference_backbone, reference_head = reference_components
    policy_hidden = _capture_hidden_states(
        model,
        policy_backbone,
        batch_input_ids,
        batch_attention_mask,
    )
    if need_kl:
        if shared_base_reference:
            with _adapter_disabled(model):
                with torch.no_grad():
                    reference_hidden = _capture_hidden_states(
                        model,
                        reference_backbone,
                        batch_input_ids,
                        batch_attention_mask,
                    )
        else:
            with torch.no_grad():
                reference_hidden = _capture_hidden_states(
                    reference_model,
                    reference_backbone,
                    batch_input_ids,
                    batch_attention_mask,
                )
    else:
        reference_hidden = None

    vocab_size = getattr(policy_head, "out_features", None)
    if vocab_size is None:
        vocab_size = policy_head.weight.shape[0]
    vocab_size = int(vocab_size)
    if need_kl:
        reference_vocab = getattr(reference_head, "out_features", None)
        if reference_vocab is None:
            reference_vocab = reference_head.weight.shape[0]
        if int(reference_vocab) != vocab_size:
            raise ValueError("policy and reference vocabulary sizes differ")
    chunk_size = _rwr_logits_chunk_size()

    def project_chunk(
        policy_chunk: Any,
        labels_chunk: Any,
        reference_chunk: Any,
        mask_chunk: Any,
    ) -> Tuple[Any, Any]:
        policy_logits = policy_head(policy_chunk)
        safe_labels = labels_chunk.clamp(min=0, max=vocab_size - 1)
        target_logits = policy_logits.gather(
            -1, safe_labels.unsqueeze(-1)
        ).squeeze(-1)
        policy_log_probs = policy_logits.float() - torch.logsumexp(
            policy_logits.float(), dim=-1, keepdim=True
        )
        chunk_log_probs = policy_log_probs.gather(
            -1, safe_labels.unsqueeze(-1)
        ).squeeze(-1)
        if not need_kl:
            return chunk_log_probs, torch.zeros(
                policy_chunk.shape[0],
                device=policy_chunk.device,
                dtype=torch.float32,
            )
        with torch.no_grad():
            reference_logits = reference_head(reference_chunk)
            reference_log_probs = F.log_softmax(reference_logits.float(), dim=-1)
        policy_probs = policy_log_probs.exp()
        token_kl = (
            policy_probs * (policy_log_probs - reference_log_probs)
        ).sum(dim=-1).clamp_min(0.0)
        return chunk_log_probs, (token_kl * mask_chunk.float()).sum(dim=1)

    for start in range(0, valid_positions.numel(), chunk_size):
        positions = valid_positions[start : start + chunk_size]
        policy_chunk = policy_hidden.index_select(1, positions)
        labels_chunk = shift_labels.index_select(1, positions)
        if need_kl:
            reference_chunk = reference_hidden.index_select(1, positions)
        else:
            reference_chunk = policy_chunk
        mask_chunk = shift_mask.index_select(1, positions)
        if torch.is_grad_enabled() and policy_chunk.requires_grad:
            chunk_log_probs, chunk_kl = checkpoint(
                project_chunk,
                policy_chunk,
                labels_chunk,
                reference_chunk,
                mask_chunk,
                use_reentrant=False,
                preserve_rng_state=False,
            )
        else:
            chunk_log_probs, chunk_kl = project_chunk(
                policy_chunk,
                labels_chunk,
                reference_chunk,
                mask_chunk,
            )
        token_values = token_values.index_copy(
            1, positions, chunk_log_probs.float()
        )
        if kl_values is not None:
            kl_values = kl_values + chunk_kl.float()

    if kl_values is not None:
        lengths = shift_mask.sum(dim=1).clamp(min=1).float()
        kl_values = kl_values / lengths
    return token_values, kl_values


@contextlib.contextmanager
def _adapter_disabled(model: Any):
    """Temporarily disable a PEFT adapter while preserving train/eval state."""
    unwrapped = _unwrap_for_introspection(model)
    disable = getattr(unwrapped, "disable_adapter", None)
    if not callable(disable):
        yield
        return
    was_training = bool(unwrapped.training)
    unwrapped.eval()
    try:
        with disable():
            yield
    finally:
        unwrapped.train(was_training)


def _token_level_kl(
    model: Any,
    reference_model: Any,
    batch_input_ids: Any,
    batch_attention_mask: Any,
    shift_labels: Any,
    shift_mask: Any,
    *,
    shared_base_reference: bool = False,
) -> Any:
    """Return the exact token-level KL for each example."""
    import torch

    _token_values, per_example_kl = _token_log_probs_and_kl(
        model,
        reference_model,
        batch_input_ids,
        batch_attention_mask,
        shift_labels,
        shift_mask,
        shared_base_reference=shared_base_reference,
    )
    if per_example_kl is None:
        return torch.zeros(batch_input_ids.shape[0], device=batch_input_ids.device)
    return per_example_kl


def compute_rwr_loss(
    model,
    ref_model,
    batch_input_ids,      # (N, L)
    batch_attention_mask, # (N, L)
    batch_labels,         # (N, L)
    rewards,              # (N,)
    *,
    kl_coef: float,
    shared_base_reference: bool = False,
    semantic_masks=None,
    skip_kl: bool = False,
):
    """Format-preserving signed RWR with an exact token-level KL anchor.

    Positive rewards reinforce the complete assistant response.  Negative
    rewards only suppress semantic value tokens selected by ``semantic_masks``;
    JSON punctuation, field names, and assistant protocol scaffolding receive
    no negative policy gradient.  The KL anchor still covers every assistant
    token, including those protected format tokens.  ``skip_kl`` is reserved
    for synthetic synchronization batches that contain no real response.
    """
    import torch

    shift_labels = batch_labels[:, 1:]
    shift_mask = shift_labels != -100
    needs_kl = (
        not skip_kl
        and kl_coef > 0
        and (ref_model is not None or shared_base_reference)
    )
    if needs_kl:
        # Reuse the policy hidden-state graph for both signed-RWR and KL.
        # Running the two helpers independently doubles long-sequence
        # activations and was the source of the observed A3 OOMs.
        token_log_probs, per_example_kl = _token_log_probs_and_kl(
            model,
            ref_model,
            batch_input_ids,
            batch_attention_mask,
            shift_labels,
            shift_mask,
            shared_base_reference=shared_base_reference,
        )
    else:
        per_example_kl = None
        token_log_probs = _token_log_probs(
            model,
            batch_input_ids,
            batch_attention_mask,
            shift_labels,
            shift_mask,
        )
    response_logp = _masked_sequence_log_probs(token_log_probs, shift_mask)

    if semantic_masks is None:
        # Never guess that protocol syntax is semantic.  Callers using the
        # format-protected objective must provide the mask produced by
        # ``encode_one``; omitting it disables negative suppression rather than
        # silently reverting to whole-response signed RWR.
        semantic_shift_mask = torch.zeros_like(shift_mask)
    else:
        semantic_masks = torch.as_tensor(
            semantic_masks, device=batch_labels.device, dtype=torch.bool
        )
        if semantic_masks.shape == batch_labels.shape:
            semantic_shift_mask = semantic_masks[:, 1:]
        elif semantic_masks.shape == shift_labels.shape:
            semantic_shift_mask = semantic_masks
        else:
            raise ValueError(
                "semantic_masks must have the same shape as batch_labels or shifted labels"
            )
        semantic_shift_mask = semantic_shift_mask & shift_mask
    semantic_logp = _masked_sequence_log_probs(token_log_probs, semantic_shift_mask)

    weights = rewards.to(response_logp.device).float()
    if weights.ndim != 1 or weights.shape[0] != response_logp.shape[0]:
        raise ValueError("rewards must be a one-dimensional tensor matching the batch")
    if not torch.isfinite(weights).all():
        raise ValueError("rewards must contain only finite values")
    positive = weights > 0
    negative = weights < 0
    policy_terms = torch.zeros_like(response_logp)
    policy_terms = torch.where(positive, -weights * response_logp, policy_terms)
    policy_terms = torch.where(negative, -weights * semantic_logp, policy_terms)
    policy_loss = policy_terms.mean()

    kl_loss = torch.tensor(0.0, device=response_logp.device)
    token_kl_mean = torch.tensor(0.0, device=response_logp.device)
    if needs_kl:
        if per_example_kl is None:
            raise RuntimeError("token-level KL helper returned no per-example values")
        token_kl_mean = per_example_kl.mean()
        kl_loss = kl_coef * token_kl_mean

    response_lengths = shift_mask.sum(dim=1).float()
    semantic_lengths = semantic_shift_mask.sum(dim=1).float()
    format_lengths = (shift_mask & ~semantic_shift_mask).sum(dim=1).float()
    return policy_loss + kl_loss, {
        "policy_loss": policy_loss.item(),
        "kl_loss": kl_loss.item(),
        "token_kl_mean": token_kl_mean.item(),
        "reward_mean": weights.mean().item(),
        "reward_min": weights.min().item(),
        "reward_max": weights.max().item(),
        "response_tokens": response_lengths.mean().item(),
        "semantic_tokens": semantic_lengths.mean().item(),
        "format_tokens": format_lengths.mean().item(),
        "negative_semantic_examples": float((negative & (semantic_lengths > 0)).sum().item()),
    }


# ---------------------------------------------------------------------------
# Collate & pad
# ---------------------------------------------------------------------------

def collate_batch(
    records:        List[RolloutRecord],
    tokenizer,
    max_seq_length: int,
) -> Optional[Tuple]:
    """Tokenize a batch and carry the semantic negative-gradient mask."""
    import torch

    encoded_list     = []
    reward_list      = []

    for rec in records:
        enc = encode_one(
            rec.messages,
            rec.response,
            tokenizer,
            max_seq_length,
            rec.enable_thinking,
        )
        if enc is None:
            continue
        encoded_list.append(enc)
        reward_list.append(rec.reward)

    if not encoded_list:
        return None

    rewards      = torch.tensor(reward_list, dtype=torch.float32)
    max_len      = max(len(e["input_ids"]) for e in encoded_list)
    pad_id       = tokenizer.pad_token_id

    input_ids_list      = []
    attention_mask_list = []
    labels_list         = []
    semantic_mask_list  = []

    for enc in encoded_list:
        pad_len = max_len - len(enc["input_ids"])
        input_ids_list.append(enc["input_ids"] + [pad_id] * pad_len)
        attention_mask_list.append(enc["attention_mask"] + [0] * pad_len)
        labels_list.append(enc["labels"] + [-100] * pad_len)
        semantic_mask_list.append(enc["semantic_mask"] + [False] * pad_len)

    return (
        torch.tensor(input_ids_list,      dtype=torch.long),
        torch.tensor(attention_mask_list, dtype=torch.long),
        torch.tensor(labels_list,         dtype=torch.long),
        torch.tensor(semantic_mask_list,  dtype=torch.bool),
        rewards,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args    = parse_args()
    out_dir = Path(args.out_dir)
    logger  = setup_logging(out_dir, args.agent)
    resume_checkpoint = resolve_resume_checkpoint(out_dir, args.resume_from_checkpoint)

    if args.model_name_or_path is None:
        from jca.src.agents import AGENT_CONFIGS
        args.model_name_or_path = AGENT_CONFIGS[args.agent].base_model

    logger.info(f"Agent:       {args.agent}")
    logger.info(f"Base model:  {args.model_name_or_path}")
    logger.info(f"SFT adapter: {args.sft_adapter}")
    logger.info(f"Rollout:     {args.rollout}")
    logger.info(f"Reward field: {args.reward_field}")
    logger.info(f"Output:      {out_dir}")
    if resume_checkpoint is not None:
        logger.info(f"Resume:      {resume_checkpoint}")
    elif args.resume_from_checkpoint:
        start_label = "the SFT adapter" if args.sft_adapter is not None else "a fresh LoRA on the base model"
        logger.info(f"Resume:      no complete checkpoint found; starting from {start_label}")

    all_records = load_rollouts(
        args.rollout,
        args.agent,
        args.reward_field,
        enable_thinking=args.enable_thinking,
    )
    logger.info(f"Loaded {len(all_records)} turn records for agent {args.agent}")
    if not all_records:
        raise SystemExit("No rollout records found for this agent.")
    rewards = [r.reward for r in all_records]
    logger.info(
        f"Reward stats: mean={sum(rewards)/len(rewards):.4f} "
        f"min={min(rewards):.4f} max={max(rewards):.4f}"
    )
    training_thinking_mode = bool(all_records[0].enable_thinking)
    logger.info(
        "Thinking contract: "
        f"enabled={int(training_thinking_mode)} (explicit template argument)"
    )

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, PeftModel, get_peft_model
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, set_seed

    ddp_kwargs = DistributedDataParallelKwargs(broadcast_buffers=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
    )
    # Initialize every stochastic training component before models are loaded.
    # device_specific keeps DDP dropout streams independent across ranks.
    set_seed(args.seed, device_specific=True)
    is_main = accelerator.is_main_process
    world_size = accelerator.num_processes
    rank = accelerator.process_index
    logger.info(f"Accelerator: rank={rank}/{world_size}  device={accelerator.device}")
    logger.info(f"Training seed: {args.seed}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype  = torch.bfloat16 if args.bf16 else torch.float32
    policy = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=dtype, trust_remote_code=True
    )
    if args.sft_adapter is not None:
        policy = PeftModel.from_pretrained(policy, str(args.sft_adapter), is_trainable=True)
    else:
        policy = get_peft_model(
            policy,
            LoraConfig(
                r=args.lora_rank,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    if args.gradient_checkpointing:
        policy.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        policy.enable_input_require_grads()
    if is_main:
        policy.print_trainable_parameters()

    ref_model = None
    shared_base_reference = False
    if args.kl_coef > 0:
        if args.sft_adapter is None:
            # The reference for fresh-LoRA SAS is exactly the frozen base
            # already held by policy. A second 14B copy leaves no activation
            # headroom on an 80 GiB GPU, so compute reference logits by
            # temporarily disabling the policy adapter.
            shared_base_reference = True
            logger.info(
                "Frozen reference: shared policy base weights with LoRA disabled"
            )
        else:
            logger.info("Loading frozen reference model (starting adapter)...")
            ref_base = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path, torch_dtype=dtype, trust_remote_code=True
            )
            ref_model = PeftModel.from_pretrained(
                ref_base, str(args.sft_adapter), is_trainable=False
            )
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad_(False)

    from transformers import get_cosine_schedule_with_warmup
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )

    n_records       = len(all_records)
    rollout_path = args.rollout.resolve()
    rollout_sha256 = file_sha256(rollout_path)
    # effective batch = world_size * per_device_batch * grad_accum
    effective_batch = world_size * args.per_device_batch_size * args.gradient_accumulation_steps
    steps_per_epoch = max(1, math.ceil(n_records / effective_batch))
    total_steps  = int(steps_per_epoch * args.num_epochs)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    # Important: do not pass the scheduler through accelerator.prepare().
    # Accelerate wraps schedulers and, with split_batches=False, advances them
    # once per process on each optimizer step. Our total_steps already accounts
    # for world_size via effective_batch, so that wrapper makes the LR schedule
    # run ~world_size times too fast and creates the observed oscillating logs.
    #
    # We prepare model + optimizer first, then build a plain scheduler on the
    # wrapped optimizer so one scheduler.step() matches one optimizer step.
    policy, optimizer = accelerator.prepare(policy, optimizer)

    # Accelerate's default model checkpoint contains the full frozen base
    # model. For PEFT training that makes every A3 checkpoint ~16 GB even
    # though only LoRA parameters change. Override model serialization while
    # retaining Accelerate's optimizer, scheduler, scaler, and RNG state.
    register_compact_peft_checkpoint_hooks(accelerator)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    accelerator.register_for_checkpointing(scheduler)
    logger.info(
        f"total_records={n_records}  world_size={world_size}  "
        f"effective_batch={effective_batch}  total_steps={total_steps}  warmup={warmup_steps}"
    )
    if ref_model is not None:
        ref_model = ref_model.to(accelerator.device)

    checkpoint_contract = {
        "agent": args.agent,
        "rollout": str(rollout_path),
        "rollout_size": rollout_path.stat().st_size,
        "rollout_sha256": rollout_sha256,
        "reward_field": args.reward_field,
        "thinking_enabled": training_thinking_mode,
        "loss_version": "format_protected_signed_rwr_token_kl_v2",
        "negative_mask": "semantic_json_values_and_thinking",
        "kl_mode": "token_level_forward_kl",
        "model_name_or_path": str(Path(args.model_name_or_path).resolve()),
        "sft_adapter": str(args.sft_adapter.resolve()) if args.sft_adapter is not None else None,
        "kl_coef": args.kl_coef,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "per_device_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_seq_length": args.max_seq_length,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "bf16": args.bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "seed": args.seed,
        "world_size": world_size,
        "total_records": n_records,
        "total_steps": total_steps,
    }
    resume_epoch = 0
    resume_micro_idx = 0
    global_step = 0
    if resume_checkpoint is not None:
        resume_payload = json.loads(
            (resume_checkpoint / "training_state.json").read_text(encoding="utf-8")
        )
        stored_contract = resume_payload.get("contract")
        if stored_contract != checkpoint_contract:
            differing = sorted(
                key
                for key in set(checkpoint_contract) | set(stored_contract or {})
                if checkpoint_contract.get(key) != (stored_contract or {}).get(key)
            )
            raise ValueError(
                "checkpoint does not match this training invocation; differing fields: "
                + ", ".join(differing)
            )
        accelerator.load_state(str(resume_checkpoint))
        global_step = int(resume_payload["global_step"])
        resume_epoch = int(resume_payload["next_epoch"])
        resume_micro_idx = int(resume_payload["next_micro_idx"])
        logger.info(
            f"Resumed checkpoint at global_step={global_step}, "
            f"next_epoch={resume_epoch}, next_micro_idx={resume_micro_idx}"
        )

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    import random
    rng = random.Random(args.seed)

    accum_loss   = 0.0
    accum_metrics: Dict[str, float] = defaultdict(float)
    accum_count  = 0
    accum_real_batches = 0
    max_steps_reached = False
    optimizer.zero_grad()

    epochs     = max(1, int(args.num_epochs))
    batch_size = args.per_device_batch_size

    def save_training_checkpoint(next_epoch: int, next_micro_idx: int) -> None:
        import shutil

        ckpt_dir = out_dir / f"checkpoint-{global_step}"
        accelerator.wait_for_everyone()
        if is_main and ckpt_dir.exists() and not (ckpt_dir / "training_state.json").is_file():
            shutil.rmtree(ckpt_dir)
        accelerator.wait_for_everyone()
        accelerator.save_state(str(ckpt_dir), safe_serialization=True)
        accelerator.wait_for_everyone()
        if is_main:
            tokenizer.save_pretrained(str(ckpt_dir))
            write_json_atomic(
                ckpt_dir / "training_state.json",
                {
                    "version": 1,
                    "global_step": global_step,
                    "next_epoch": next_epoch,
                    "next_micro_idx": next_micro_idx,
                    "contract": checkpoint_contract,
                },
            )
            logger.info(f"Saved resumable checkpoint: {ckpt_dir}")
            ckpts = sorted(
                (
                    path for path in out_dir.glob("checkpoint-*")
                    if re.fullmatch(r"checkpoint-[0-9]+", path.name)
                    and (path / "training_state.json").is_file()
                ),
                key=lambda path: int(path.name.split("-")[1]),
            )
            for old in ckpts[: -args.save_total_limit]:
                shutil.rmtree(old, ignore_errors=True)
        accelerator.wait_for_everyone()

    for epoch in range(epochs):
        # Shuffle deterministically so all ranks see the same order, then pad
        # to a multiple of the distributed effective micro-batch. This keeps
        # every rank on the same number of complete accumulation windows.
        rng.shuffle(all_records)
        shard_multiple = (
            world_size * batch_size * args.gradient_accumulation_steps
        )
        pad_to = math.ceil(len(all_records) / shard_multiple) * shard_multiple
        # A single slice is insufficient when the padding needed is larger
        # than the dataset itself (for example a one-row smoke test on 8
        # ranks). Repeat the shuffled records until every rank receives the
        # same number of complete accumulation windows.
        repeat_count = math.ceil(pad_to / len(all_records))
        padded_records = (all_records * repeat_count)[:pad_to]
        my_records = padded_records[rank::world_size]
        if is_main:
            logger.info(
                f"Epoch {epoch+1}/{epochs}  total_n={len(all_records)}  "
                f"padded_n={len(padded_records)}  per_rank_n={len(my_records)}"
            )

        num_micro_batches = math.ceil(len(my_records) / batch_size)
        if epoch < resume_epoch:
            continue
        first_micro_idx = resume_micro_idx if epoch == resume_epoch else 0
        if first_micro_idx > num_micro_batches:
            raise ValueError(
                f"checkpoint micro-batch offset {first_micro_idx} exceeds epoch size {num_micro_batches}"
            )
        epoch_start_time = time.perf_counter()
        processed_tokens = 0
        for micro_idx, step_idx in enumerate(range(0, len(my_records), batch_size)):
            if micro_idx < first_micro_idx:
                continue
            batch   = my_records[step_idx: step_idx + batch_size]
            tensors = collate_batch(batch, tokenizer, args.max_seq_length)
            # A failed tokenization still needs to execute the same DDP
            # backward path as the other ranks.  Keep the tiny zero-reward
            # batch below connected to the policy graph, but never apply the
            # KL anchor to its synthetic label: there is no real response to
            # regularize and doing so would create an update from padding.
            dummy_batch = tensors is None
            if not dummy_batch:
                accum_real_batches += 1

            if dummy_batch:
                # tokenize failure: use a tiny dummy batch to keep DDP in sync.
                # The synthetic label keeps backward connected; its zero
                # reward and disabled KL coefficient contribute no update.
                pad_id = tokenizer.pad_token_id
                dummy_len = 8
                input_ids = torch.full((1, dummy_len), pad_id, dtype=torch.long)
                attention_mask = torch.ones((1, dummy_len), dtype=torch.long)
                labels = torch.full((1, dummy_len), -100, dtype=torch.long)
                # give one real label to avoid /0
                labels[0, -1] = pad_id
                semantic_masks = torch.zeros((1, dummy_len), dtype=torch.bool)
                rewards = torch.zeros((1,), dtype=torch.float32)
            else:
                input_ids, attention_mask, labels, semantic_masks, rewards = tensors

            input_ids      = input_ids.to(accelerator.device)
            attention_mask = attention_mask.to(accelerator.device)
            labels         = labels.to(accelerator.device)
            semantic_masks = semantic_masks.to(accelerator.device)
            rewards        = rewards.to(accelerator.device)
            processed_tokens += int(attention_mask.sum().item())

            with accelerator.accumulate(policy):
                loss, metrics = compute_rwr_loss(
                    policy, ref_model,
                    input_ids, attention_mask, labels, rewards,
                    kl_coef=args.kl_coef,
                    shared_base_reference=shared_base_reference,
                    semantic_masks=semantic_masks,
                    skip_kl=False,
                )
                if dummy_batch:
                    loss = loss * 0.0

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        [p for p in policy.parameters() if p.requires_grad],
                        max_norm=1.0,
                    )
                if accelerator.sync_gradients:
                    real_batch_count = accelerator.gather(
                        torch.tensor([accum_real_batches], device=accelerator.device)
                    ).sum().item()
                    if real_batch_count > 0:
                        optimizer.step()
                        scheduler.step()
                        global_step += 1
                        if args.max_steps > 0 and global_step >= args.max_steps:
                            max_steps_reached = True
                    accum_real_batches = 0
                optimizer.zero_grad()

            accum_loss += loss.item()
            for k, v in metrics.items():
                accum_metrics[k] += v
            accum_count += 1

            if accelerator.sync_gradients:
                if global_step % args.logging_steps == 0 and is_main:
                    n = max(accum_count, 1)
                    logger.info(
                        f"step {global_step}"
                        f"  loss={accum_loss/n:.4f}"
                        f"  kl={accum_metrics['kl_loss']/n:.4f}"
                        f"  reward_mean={accum_metrics['reward_mean']/n:.3f}"
                        f"  tokens_per_sec={processed_tokens / max(time.perf_counter() - epoch_start_time, 1e-6):.1f}"
                        f"  lr={scheduler.get_last_lr()[0]:.2e}"
                    )
                if global_step % args.logging_steps == 0:
                    accum_loss    = 0.0
                    accum_metrics = defaultdict(float)
                    accum_count   = 0

                if global_step % args.save_steps == 0:
                    next_epoch = epoch
                    next_micro_idx = micro_idx + 1
                    if next_micro_idx >= num_micro_batches:
                        next_epoch = epoch + 1
                        next_micro_idx = 0
                    save_training_checkpoint(next_epoch, next_micro_idx)

            if max_steps_reached:
                break

        if max_steps_reached:
            break

    accelerator.wait_for_everyone()
    if is_main:
        final_dir = out_dir / "final"
        unwrapped = accelerator.unwrap_model(policy)
        unwrapped.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        logger.info(f"Final adapter saved: {final_dir}")
        adapter_path = final_dir / "adapter_model.safetensors"
        if not adapter_path.is_file():
            raise RuntimeError(f"final adapter weights were not written: {adapter_path}")
        adapter_sha256 = file_sha256(adapter_path)
        write_json_atomic(
            final_dir / "rl_train_summary.json",
            {
                "summary_version": 2,
                "agent":         args.agent,
                "global_step":   global_step,
                "total_records": n_records,
                "algorithm":     "rwr",
                "kl_coef":       args.kl_coef,
                "thinking_enabled": training_thinking_mode,
                "loss_version": "format_protected_signed_rwr_token_kl_v2",
                "negative_mask": "semantic_json_values_and_thinking",
                "kl_mode": "token_level_forward_kl",
                "world_size":    world_size,
                "completed":     True,
                "rollout":       str(rollout_path),
                "rollout_size":  rollout_path.stat().st_size,
                "rollout_sha256": rollout_sha256,
                "adapter_sha256": adapter_sha256,
                "model_name_or_path": str(Path(args.model_name_or_path).resolve()),
                "sft_adapter": str(args.sft_adapter.resolve()) if args.sft_adapter is not None else None,
                "contract":      checkpoint_contract,
            },
        )

    logger.info("RL training complete.")


if __name__ == "__main__":
    main()

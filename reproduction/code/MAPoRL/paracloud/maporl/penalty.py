"""The response-quality gate that can replace a score with -10.

Provenance:
  check_repeated_sequences -- /tmp/maporl_ref/utils/utils_cooperateLLM.py:425-470
  penalty gate             -- ppov2_trainer_multi_different_model.py:1463-1502

Despite its name ``non_eos_penalty`` does not check EOS -- it cannot, because
the official generation config forces ``min_new_tokens == max_new_tokens`` so no
response ever contains one. What it actually gates is:

    penalty_condition = (length >= min_output_length)
                      & (not degenerate repetition)
                      & (answer is extractable)      [if task_training or turn != 0]

Rows failing the gate have their score **replaced** by ``penalty_reward_value``
(-10), not decremented. That replacement happens *before* ``score_rule`` runs, so
a single bad turn-2 response drags turn 0 and turn 1 for **every** agent through
the all-agent mean. ``reward_rules.combine``'s ``!= -1`` mask does not catch -10.

``non_box_penalty`` and ``repeat_penalty`` exist in the official config but are
assigned and never read (:342-343). They are dead and are not ported.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

# ppov2_trainer_multi_different_model.py:1497 / config...reloadF.py:140
DEFAULT_PENALTY_REWARD_VALUE = -10.0
DEFAULT_MIN_OUTPUT_LENGTH = 50


@dataclass
class PenaltyResult:
    passed: bool
    cause: str | None = None  # "length" | "repetition" | "no_answer"


def check_repeated_sequences(
    token_ids: Sequence[int],
    *,
    decode=None,
    min_seq_length: int = 1,
    max_seq_length: int = 20,
    max_repeats: int = 3,
) -> bool:
    """True when some subsequence repeats **consecutively** more than ``max_repeats``.

    Verbatim port. Two details that are easy to get wrong:

    * Repeats must be *adjacent* -- the scan breaks on the first mismatch, so
      "ab X ab X ab X ab" does not trigger but "ab ab ab ab" does.
    * Subsequences that decode to pure digits are skipped, so a long numeric
      answer is not mistaken for degeneration.

    ``decode`` maps a token-id list to text. Callers should pass a memoized
    tokenizer decode: the scan is O(L x max_seq_length) and at L=300 that is
    ~6000 decode calls per response.
    """
    seq = list(token_ids)
    seq_len = len(seq)
    if seq_len == 0:
        return False

    for length in range(min_seq_length, min(max_seq_length + 1, seq_len + 1)):
        for start in range(seq_len - length + 1):
            subseq = seq[start : start + length]
            if decode is not None:
                text = decode(tuple(subseq))
                if text.strip().isdigit():
                    continue
            repeat_count = 1
            pos = start + length
            while pos + length <= seq_len:
                if seq[pos : pos + length] == subseq:
                    repeat_count += 1
                    if repeat_count > max_repeats:
                        return True
                    pos += length
                else:
                    break
    return False


def memoized_decoder(tokenizer):
    """Cache ``tokenizer.decode`` over token tuples -- the scan repeats heavily."""
    cache: dict[tuple[int, ...], str] = {}

    def decode(token_ids: tuple[int, ...]) -> str:
        hit = cache.get(token_ids)
        if hit is None:
            hit = tokenizer.decode(list(token_ids))
            cache[token_ids] = hit
        return hit

    return decode


def evaluate_penalty(
    *,
    token_ids: Sequence[int],
    sequence_length: int,
    answer_extracted: bool,
    turn: int,
    task_training: bool,
    min_output_length: int = DEFAULT_MIN_OUTPUT_LENGTH,
    require_answer: bool = True,
    decode=None,
) -> PenaltyResult:
    """The :1463-1502 gate, generalized.

    The official check is a ``\\boxed{...}`` regex. Only two of our five tasks
    use ``\\boxed`` at all, so the caller passes ``answer_extracted`` -- "the
    task's own parser found an answer" -- which is the same intent. Turn 0 is
    exempt unless ``task_training`` is on, exactly as upstream (:1494-1495).
    """
    if sequence_length < min_output_length:
        return PenaltyResult(False, "length")
    if check_repeated_sequences(token_ids, decode=decode):
        return PenaltyResult(False, "repetition")
    if require_answer and (task_training or turn != 0) and not answer_extracted:
        return PenaltyResult(False, "no_answer")
    return PenaltyResult(True)


def apply_penalty(
    score: float,
    result: PenaltyResult,
    *,
    penalty_reward_value: float = DEFAULT_PENALTY_REWARD_VALUE,
) -> float:
    """Replace, do not decrement (:1497-1502)."""
    return score if result.passed else penalty_reward_value


def summarize_causes(results: Iterable[PenaltyResult]) -> dict[str, Any]:
    """Per-cause counts, so a spike is attributable rather than just visible."""
    causes = {"length": 0, "repetition": 0, "no_answer": 0}
    total = failed = 0
    for result in results:
        total += 1
        if not result.passed:
            failed += 1
            if result.cause in causes:
                causes[result.cause] += 1
    return {
        "penalty_total": total,
        "penalty_failed": failed,
        "penalty_frac": failed / total if total else 0.0,
        **{f"penalty_{k}": v for k, v in causes.items()},
    }

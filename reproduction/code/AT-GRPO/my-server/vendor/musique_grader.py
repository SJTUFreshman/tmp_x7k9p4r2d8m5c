"""Grading: extract answer + EM/F1 vs MuSiQue ground truth.

Public API:
    is_correct(predicted, problem) -> bool
    compute_em_f1(predicted, problem) -> (em, f1)
    extract_boxed_answer(text) -> Optional[str]

See `code-guide.md` §2.2.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional, Tuple

from .musique_data import MuSiQueProblem


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the LAST \\boxed{...} content with balanced braces.

    Same logic as frozen_baseline_probe.py:extract_answer.
    Returns None if no \\boxed{} found AND no fallback line is non-empty.
    """
    if not text:
        return None

    matches = []
    idx = 0
    while True:
        i = text.find(r"\boxed{", idx)
        if i < 0:
            break
        start = i + len(r"\boxed{")
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    matches.append(text[start:j])
                    break
            j += 1
        idx = i + 1

    if matches:
        return _strip_latex_answer_wrappers(matches[-1])

    # Fallback: last non-empty line
    for line in reversed(text.splitlines()):
        s = line.strip()
        if s:
            return _strip_latex_answer_wrappers(s)
    return None


def _strip_latex_answer_wrappers(text: str) -> str:
    """Remove simple LaTeX text wrappers around an extracted answer."""
    answer = text.strip()
    wrappers = ("text", "mathrm", "operatorname", "mbox", "textbf", "emph")

    while True:
        unwrapped = None
        for wrapper in wrappers:
            unwrapped = _unwrap_latex_wrapper(answer, wrapper)
            if unwrapped is not None:
                break
        if unwrapped is None:
            return answer
        answer = unwrapped.strip()


def _unwrap_latex_wrapper(text: str, wrapper: str) -> Optional[str]:
    prefix = "\\" + wrapper + "{"
    if not text.startswith(prefix):
        return None

    start = len(prefix)
    depth = 1
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                if text[idx + 1 :].strip():
                    return None
                return text[start:idx]
    return None


def normalize_answer(s: str) -> str:
    """MuSiQue's official normalization (lowercase, strip articles, punct)."""
    if not s:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_correct(predicted: str, problem: MuSiQueProblem) -> bool:
    """EM check: predicted (extracted from \\boxed{}) matches answer or any alias.
    """
    pred = extract_boxed_answer(predicted) if predicted else None
    if pred is None:
        return False
    pred_norm = normalize_answer(pred)
    candidates = [problem.answer] + list(problem.answer_aliases or [])
    return any(pred_norm == normalize_answer(c) for c in candidates)


def compute_em_f1(predicted: str, problem: MuSiQueProblem) -> Tuple[float, float]:
    """Token-level EM and F1 vs ground truth.

    Scores against the ground-truth answer and aliases, returning the best F1
    over all acceptable answers.
    """
    pred = extract_boxed_answer(predicted) if predicted else None
    if pred is None:
        return 0.0, 0.0

    candidates = [problem.answer] + list(problem.answer_aliases or [])
    em = 1.0 if any(_exact_match_score(pred, c) for c in candidates) else 0.0
    f1 = max((_f1_score(pred, c) for c in candidates), default=0.0)
    return em, f1


def _exact_match_score(prediction: str, ground_truth: str) -> bool:
    """Return normalized exact match."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def _f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 after MuSiQue-style normalization."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    if not prediction_tokens and not ground_truth_tokens:
        return 1.0
    if not prediction_tokens or not ground_truth_tokens:
        return 0.0

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return 2 * precision * recall / (precision + recall)

"""Numeric grader for GSM-HARD."""
from __future__ import annotations

from typing import Optional, Tuple

from .gsm_data import GSMProblem, extract_number


def numeric_values_match(
    prediction: float,
    target: float,
    tolerance: float = 1e-6,
) -> bool:
    """Return whether two numeric values match under the legacy GSM grader."""
    try:
        prediction = float(prediction)
        target = float(target)
    except (TypeError, ValueError, OverflowError):
        return False

    if abs(prediction - target) <= tolerance:
        return True

    if abs(target) > 1e-9:
        rel_err = abs(prediction - target) / abs(target)
        if rel_err < 1e-6:
            return True

    return False


def compute_em_f1(
    prediction: str,
    problem: GSMProblem,
    tolerance: float = 1e-6,
) -> Tuple[float, float]:
    """Compute EM and F1 for a numeric prediction.

    For numeric answers:
    - EM = 1.0 if prediction matches target exactly (within tolerance)
    - F1 = same as EM (either right or wrong for numbers)

    Returns (em, f1) both in [0, 1].
    """
    if not prediction:
        return 0.0, 0.0

    pred_num = extract_number(prediction)
    if pred_num is None:
        return 0.0, 0.0

    target = problem.answer

    if numeric_values_match(pred_num, target, tolerance):
        return 1.0, 1.0

    # Partial credit: how close is the magnitude?
    # Use log-scale similarity as a soft F1 proxy
    try:
        if target == 0 and pred_num == 0:
            return 1.0, 1.0
        if target == 0 or pred_num == 0:
            return 0.0, 0.0
        # ratio-based soft score
        ratio = min(abs(pred_num), abs(target)) / max(abs(pred_num), abs(target))
        # same sign?
        if (pred_num > 0) != (target > 0):
            return 0.0, 0.0
        f1 = ratio  # continuous score 0-1
        return 0.0, round(f1, 4)
    except Exception:
        return 0.0, 0.0


def is_correct(prediction: str, problem: GSMProblem, tolerance: float = 1e-6) -> bool:
    em, _ = compute_em_f1(prediction, problem, tolerance)
    return em == 1.0

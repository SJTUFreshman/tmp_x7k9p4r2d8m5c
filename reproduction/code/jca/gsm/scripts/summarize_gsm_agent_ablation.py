#!/usr/bin/env python3
"""Summarize isolated A1/A2/A3 adapter evaluations against one SFT baseline."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, Mapping, Sequence

from jca.gsm.src.data import GSMProblem
from jca.gsm.src.grader import compute_em_f1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--a1-only", type=Path, required=True)
    parser.add_argument("--a2-only", type=Path, required=True)
    parser.add_argument("--a3-only", type=Path, required=True)
    parser.add_argument("--all-agents", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_results(path: Path) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            problem = row.get("problem") or {}
            if not isinstance(problem, Mapping):
                raise ValueError(f"invalid problem object at {path}:{line_number}")
            problem_id = str(problem.get("id") or row.get("problem_id") or "")
            if not problem_id or problem_id in results:
                raise ValueError(f"missing or duplicate problem at {path}:{line_number}")
            trajectory = row.get("trajectory") or {}
            steps = trajectory.get("steps") or []
            if steps and str(steps[0].get("active_agent")) != "A1":
                raise ValueError(f"non-A1 trajectory start at {path}:{line_number}")
            results[problem_id] = row
    if not results:
        raise ValueError(f"no results in {path}")
    return results


def exact_mcnemar_p(gained: int, lost: int) -> float:
    discordant = gained + lost
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(gained, lost) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _first_present(
    mapping: Mapping[str, Any],
    field_names: Sequence[str],
) -> Any:
    for field_name in field_names:
        if field_name in mapping and mapping[field_name] is not None:
            return mapping[field_name]
    return None


def _problem_from_row(row: Mapping[str, Any]) -> GSMProblem:
    raw_problem = row.get("problem") or {}
    if not isinstance(raw_problem, Mapping):
        raise ValueError("result row has a non-object problem field")

    reference = _first_present(
        raw_problem,
        ("answer", "answer_str", "target", "reference_answer", "gold_answer"),
    )
    if reference is None:
        reference = _first_present(
            row,
            ("reference_answer", "gold_answer", "target", "answer"),
        )
    problem_id = str(raw_problem.get("id") or row.get("problem_id") or "")
    if reference is None:
        raise ValueError(f"missing reference answer for {problem_id or '<unknown>'}")

    answer_str = str(reference)
    try:
        answer = float(answer_str.replace(",", ""))
    except ValueError as exc:
        raise ValueError(
            f"invalid numeric answer for {problem_id or '<unknown>'}: {answer_str!r}"
        ) from exc
    return GSMProblem(
        id=problem_id,
        question=str(raw_problem.get("question") or row.get("question") or ""),
        answer=answer,
        answer_str=answer_str,
    )


def _final_prediction(row: Mapping[str, Any]) -> str:
    trajectory = row.get("trajectory") or {}
    if "final_answer" in row:
        prediction = row["final_answer"]
        return "" if prediction is None else str(prediction)
    if isinstance(trajectory, Mapping) and "final_answer" in trajectory:
        prediction = trajectory["final_answer"]
        return "" if prediction is None else str(prediction)

    prediction_fields = ("prediction", "predicted_answer", "model_answer")
    prediction = _first_present(row, prediction_fields)
    if prediction is None and isinstance(trajectory, Mapping):
        prediction = _first_present(trajectory, prediction_fields)
        steps = trajectory.get("steps") or []
        if prediction is None and steps and isinstance(steps[-1], Mapping):
            prediction = _first_present(
                steps[-1],
                ("final_answer", "confirmed_answer", "tentative_answer"),
            )
    return "" if prediction is None else str(prediction)


def final_em_f1(row: Mapping[str, Any]) -> tuple[float, float]:
    """Recompute final metrics from saved answers with the current grader."""
    return compute_em_f1(_final_prediction(row), _problem_from_row(row))


def initial_a1_em(row: Dict[str, Any]) -> float:
    steps = (row.get("trajectory") or {}).get("steps") or []
    if not steps:
        return 0.0
    tentative_answer = str(steps[0].get("tentative_answer") or "")
    return float(compute_em_f1(tentative_answer, _problem_from_row(row))[0])


def summarize(
    baseline: Dict[str, Dict[str, Any]],
    candidate: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    if set(candidate) != set(baseline):
        missing = sorted(set(baseline) - set(candidate))[:5]
        extra = sorted(set(candidate) - set(baseline))[:5]
        raise ValueError(f"evaluation coverage mismatch: missing={missing} extra={extra}")
    problem_ids = sorted(baseline)
    baseline_em = [final_em_f1(baseline[key])[0] for key in problem_ids]
    candidate_scores = [final_em_f1(candidate[key]) for key in problem_ids]
    candidate_em = [score[0] for score in candidate_scores]
    candidate_f1 = [score[1] for score in candidate_scores]
    candidate_initial = [initial_a1_em(candidate[key]) for key in problem_ids]
    gained = sum(
        before == 0.0 and after == 1.0
        for before, after in zip(baseline_em, candidate_em)
    )
    lost = sum(
        before == 1.0 and after == 0.0
        for before, after in zip(baseline_em, candidate_em)
    )
    initial_wrong_final_correct = sum(
        initial == 0.0 and final == 1.0
        for initial, final in zip(candidate_initial, candidate_em)
    )
    initial_correct_final_wrong = sum(
        initial == 1.0 and final == 0.0
        for initial, final in zip(candidate_initial, candidate_em)
    )
    return {
        "rows": len(problem_ids),
        "em": fmean(candidate_em),
        "em_correct": int(sum(candidate_em)),
        "f1": fmean(candidate_f1),
        "a1_initial_em": fmean(candidate_initial),
        "a1_initial_correct": int(sum(candidate_initial)),
        "initial_wrong_to_final_correct": initial_wrong_final_correct,
        "initial_correct_to_final_wrong": initial_correct_final_wrong,
        "gained_vs_sft": gained,
        "lost_vs_sft": lost,
        "net_em_items_vs_sft": gained - lost,
        "paired_mcnemar_p": exact_mcnemar_p(gained, lost),
    }


def build_report(paths: Dict[str, Path]) -> Dict[str, Any]:
    loaded = {name: load_results(path) for name, path in paths.items()}
    baseline = loaded["sft_baseline"]
    report = {
        "protocol": {
            "fixed_start_agent": "A1",
            "single_deterministic_trajectory": True,
            "agent_adapters_selected_by_dev": False,
        },
        "paths": {name: str(path) for name, path in paths.items()},
        "results": {},
    }
    for name, rows in loaded.items():
        report["results"][name] = summarize(baseline, rows)
    return report


def print_report(report: Dict[str, Any]) -> None:
    print("variant       EM       F1       A1-init  gain  loss  net")
    for name in ("sft_baseline", "a1_only", "a2_only", "a3_only", "all_agents"):
        item = report["results"][name]
        print(
            f"{name:<13} {item['em']:.4f}   {item['f1']:.4f}   "
            f"{item['a1_initial_em']:.4f}   {item['gained_vs_sft']:>4}  "
            f"{item['lost_vs_sft']:>4}  {item['net_em_items_vs_sft']:>+4}"
        )


def main() -> None:
    args = parse_args()
    paths = {
        "sft_baseline": args.baseline,
        "a1_only": args.a1_only,
        "a2_only": args.a2_only,
        "a3_only": args.a3_only,
        "all_agents": args.all_agents,
    }
    report = build_report(paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print_report(report)
    print(f"report: {args.output}")


if __name__ == "__main__":
    main()

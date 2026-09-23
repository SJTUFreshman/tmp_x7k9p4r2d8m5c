#!/usr/bin/env python3
"""Rejudge immutable GSM rollout groups with the collaboration-focused v3 judge."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PARENT = REPO_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from jca.gsm.scripts.run_mas import answers_match  # noqa: E402
from jca.gsm.src.data import GSMProblem, load_gsm_hard  # noqa: E402
from jca.gsm.src.grader import compute_em_f1  # noqa: E402
from jca.gsm.src.judge_v3 import (  # noqa: E402
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL,
    JUDGE_PARSE_RETRIES,
    JUDGE_REASONING_EFFORT,
    JUDGE_TEMPERATURE,
    JUDGE_TOP_P,
    CollaborationTrajectoryScore,
    build_judge_prompt,
    judge_trajectory,
)


GroupKey = Tuple[str, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rejudge GSM rollout groups for collaboration-aware RL v3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path)
    parser.add_argument("--max-concurrency", type=int, default=16)
    parser.add_argument("--group-retries", type=int, default=2)
    parser.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", JUDGE_MODEL))
    parser.add_argument("--judge-temperature", type=float, default=JUDGE_TEMPERATURE)
    parser.add_argument("--judge-top-p", type=float, default=JUDGE_TOP_P)
    parser.add_argument("--judge-max-tokens", type=int, default=JUDGE_MAX_TOKENS)
    parser.add_argument(
        "--judge-reasoning-effort",
        default=JUDGE_REASONING_EFFORT,
    )
    parser.add_argument("--judge-parse-retries", type=int, default=JUDGE_PARSE_RETRIES)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--limit-groups", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate source groups and construct one judge prompt without calling a judge or writing output.",
    )
    return parser.parse_args()


def group_key(row: Dict[str, Any]) -> GroupKey:
    return str(row.get("problem_id", "")), int(row.get("rollout_idx", 0))


def read_groups(
    path: Path,
    *,
    tolerate_invalid_lines: bool = False,
) -> Dict[GroupKey, List[Dict[str, Any]]]:
    groups: Dict[GroupKey, List[Dict[str, Any]]] = defaultdict(list)
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                if tolerate_invalid_lines:
                    continue
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            groups[group_key(row)].append(row)
    for key, rows in groups.items():
        rows.sort(key=lambda row: int(row.get("turn", -1)))
        turns = [int(row.get("turn", -1)) for row in rows]
        if turns != list(range(len(rows))):
            raise ValueError(f"non-contiguous turns for group={key}: {turns}")
    return dict(groups)


def parse_response(row: Dict[str, Any]) -> Dict[str, Any]:
    response = row.get("response")
    if isinstance(response, dict):
        return dict(response)
    if not isinstance(response, str):
        raise ValueError("response is not a JSON string")
    parsed = json.loads(response)
    if not isinstance(parsed, dict):
        raise ValueError("response JSON is not an object")
    return parsed


def build_result(rows: List[Dict[str, Any]], problem: GSMProblem) -> Dict[str, Any]:
    steps: List[Dict[str, Any]] = []
    final_answer = ""
    for row in rows:
        response = parse_response(row)
        tentative = str(response.get("tentative_answer") or "").strip()
        confirmed = response.get("confirmed_answer")
        if confirmed is not None and str(confirmed).strip():
            final_answer = str(confirmed).strip()
        elif tentative:
            final_answer = tentative
        steps.append({
            "turn": int(row["turn"]),
            "active_agent": str(row["agent_id"]),
            "reasoning": str(response.get("reasoning") or ""),
            "tentative_answer": tentative,
            "action": str(response.get("action") or row.get("action") or ""),
            "handoff_target": response.get("handoff_target"),
            "handoff_note": response.get("handoff_note"),
            "confirmed_answer": confirmed,
        })
    em, f1 = compute_em_f1(final_answer, problem)
    return {
        "problem": {
            "id": problem.id,
            "question": problem.question,
            "answer": problem.answer_str,
        },
        "trajectory": {"problem_id": problem.id, "steps": steps},
        "final_answer": final_answer,
        "em": em,
        "f1": f1,
    }


def deterministic_states(
    rows: List[Dict[str, Any]],
    problem: GSMProblem,
) -> Dict[int, Dict[str, Any]]:
    states: Dict[int, Dict[str, Any]] = {}
    previous_answer: Optional[str] = None
    previous_correct: Optional[bool] = None
    for row in rows:
        response = parse_response(row)
        current_answer = str(response.get("tentative_answer") or "").strip()
        current_correct = compute_em_f1(current_answer, problem)[0] == 1.0
        changed = bool(
            previous_answer is not None
            and not answers_match(previous_answer, current_answer)
        )
        if previous_correct is None:
            answer_state = "proposal_correct" if current_correct else "proposal_wrong"
        else:
            answer_state = (
                ("correct" if previous_correct else "wrong")
                + "_to_"
                + ("correct" if current_correct else "wrong")
            )
        states[int(row["turn"])] = {
            "previous_answer": previous_answer,
            "current_answer": current_answer,
            "previous_answer_correct": previous_correct,
            "current_answer_correct": current_correct,
            "answer_changed": changed,
            "answer_state": answer_state,
        }
        if current_answer:
            previous_answer = current_answer
            previous_correct = current_correct
    return states


def merge_scores(
    rows: List[Dict[str, Any]],
    problem: GSMProblem,
    judged: CollaborationTrajectoryScore,
    judge_model: str,
) -> List[Dict[str, Any]]:
    score_by_turn = {score.turn: score for score in judged.turn_scores}
    states = deterministic_states(rows, problem)
    merged: List[Dict[str, Any]] = []
    for row in rows:
        turn = int(row["turn"])
        score = score_by_turn[turn]
        state = states[turn]
        disagreements = []
        for field in (
            "previous_answer_correct",
            "current_answer_correct",
            "answer_changed",
        ):
            if getattr(score, field) != state[field]:
                disagreements.append(field)
        output = dict(row)
        output["collaboration_judge_v3"] = {
            "reasoning_score": round(score.reasoning_score, 4),
            "verification_score": round(score.verification_score, 4),
            "action_score": round(score.action_score, 4),
            "route_score": round(score.route_score, 4),
            "judge_score": round(score.judge_score, 4),
            "previous_answer_correct": score.previous_answer_correct,
            "current_answer_correct": score.current_answer_correct,
            "answer_changed": score.answer_changed,
            "independent_verification": score.independent_verification,
            "recommended_action": score.recommended_action,
            "comment": score.comment,
        }
        output["deterministic_state_v3"] = state
        output["judge_state_disagreements_v3"] = disagreements
        output["collaboration_judge_model_v3"] = judge_model
        output["collaboration_judge_status_v3"] = "scored"
        merged.append(output)
    return merged


def output_group_complete(rows: List[Dict[str, Any]]) -> bool:
    if not rows:
        return False
    turns = sorted(int(row.get("turn", -1)) for row in rows)
    return turns == list(range(len(rows))) and all(
        row.get("collaboration_judge_status_v3") == "scored"
        and isinstance(row.get("collaboration_judge_v3"), dict)
        and isinstance(row.get("deterministic_state_v3"), dict)
        for row in rows
    )


def prepare_resume_output(output: Path, resume: bool) -> set[GroupKey]:
    if not output.exists():
        return set()
    if not resume:
        raise FileExistsError(f"output exists: {output}; use --resume or a new path")
    existing = read_groups(output, tolerate_invalid_lines=True)
    complete = {key for key, rows in existing.items() if output_group_complete(rows)}
    temporary = output.with_suffix(output.suffix + ".resume.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for key in sorted(complete):
            for row in existing[key]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output)
    return complete


def judge_group(
    rows: List[Dict[str, Any]],
    problem: GSMProblem,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    result = build_result(rows, problem)
    last_error: Optional[Exception] = None
    for _attempt in range(args.group_retries + 1):
        try:
            judged = judge_trajectory(
                result,
                model=args.judge_model,
                temperature=args.judge_temperature,
                top_p=args.judge_top_p,
                max_tokens=args.judge_max_tokens,
                reasoning_effort=args.judge_reasoning_effort,
                parse_retries=args.judge_parse_retries,
            )
            if judged is None:
                raise RuntimeError("judge returned no parseable score")
            return merge_scores(rows, problem, judged, args.judge_model)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"v3 judge failed after retries: {last_error}")


def summarize_output(path: Path) -> Dict[str, Any]:
    groups = read_groups(path)
    rows = [row for group_rows in groups.values() for row in group_rows]
    disagreements = Counter(
        field
        for row in rows
        for field in row.get("judge_state_disagreements_v3", [])
    )
    return {
        "groups": len(groups),
        "rows": len(rows),
        "agents": dict(Counter(str(row.get("agent_id")) for row in rows)),
        "judge_state_disagreements": dict(disagreements),
        "judge_model": next(
            (
                row.get("collaboration_judge_model_v3")
                for row in rows
                if row.get("collaboration_judge_model_v3")
            ),
            None,
        ),
    }


def write_rows(handle, rows: Iterable[Dict[str, Any]]) -> None:
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    handle.write(payload)
    handle.flush()


def main() -> None:
    args = parse_args()
    if args.max_concurrency <= 0 or args.group_retries < 0:
        raise SystemExit("concurrency must be positive and retries non-negative")
    if args.judge_temperature != 0.0:
        raise SystemExit("v3 judge temperature must remain 0.0")

    source_groups = read_groups(args.input)
    if args.limit_groups > 0:
        selected_keys = sorted(source_groups)[: args.limit_groups]
        source_groups = {key: source_groups[key] for key in selected_keys}
    problem_map = {problem.id: problem for problem in load_gsm_hard(args.data_path)}
    missing = sorted({key[0] for key in source_groups} - set(problem_map))
    if missing:
        raise SystemExit(f"source references unknown problems: {missing[:5]}")

    if args.dry_run:
        sample_key = min(source_groups)
        sample_rows = source_groups[sample_key]
        sample_result = build_result(sample_rows, problem_map[sample_key[0]])
        prompt = build_judge_prompt(sample_result)
        states = deterministic_states(sample_rows, problem_map[sample_key[0]])
        print("GSM collaboration-focused v3 rejudge dry-run")
        print(
            f"groups={len(source_groups)} sample_group={sample_key} "
            f"turns={len(sample_rows)} prompt_chars={len(prompt)} states={len(states)}"
        )
        print("dry-run complete; no judge calls or output writes")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = prepare_resume_output(args.output, args.resume)
    pending = [key for key in sorted(source_groups) if key not in completed]
    print("=" * 72)
    print("GSM collaboration-focused v3 rejudge")
    print(f"groups={len(source_groups)} completed={len(completed)} pending={len(pending)}")
    print(
        f"judge={args.judge_model} temperature={args.judge_temperature} "
        f"top_p={args.judge_top_p} concurrency={args.max_concurrency}"
    )
    print(f"output={args.output}")
    print("=" * 72)

    started = time.time()
    errors: List[Tuple[GroupKey, str]] = []
    if pending:
        with args.output.open("a", encoding="utf-8") as handle:
            with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
                futures = {
                    executor.submit(
                        judge_group,
                        source_groups[key],
                        problem_map[key[0]],
                        args,
                    ): key
                    for key in pending
                }
                for index, future in enumerate(as_completed(futures), 1):
                    key = futures[future]
                    try:
                        write_rows(handle, future.result())
                    except Exception as exc:
                        errors.append((key, str(exc)))
                        print(f"[error] group={key}: {exc}")
                    if index % args.progress_every == 0 or index == len(pending):
                        elapsed = max(time.time() - started, 1e-6)
                        rate = index / elapsed
                        eta = (len(pending) - index) / max(rate, 1e-9)
                        print(
                            f"progress={index}/{len(pending)} errors={len(errors)} "
                            f"elapsed={elapsed / 60:.1f}m eta={eta / 60:.1f}m"
                        )

    if errors:
        raise SystemExit(
            f"{len(errors)} groups failed; rerun the same command with --resume"
        )
    stats = summarize_output(args.output)
    if stats["groups"] != len(source_groups):
        raise SystemExit(
            f"output has {stats['groups']} complete groups, expected {len(source_groups)}"
        )
    if args.stats_output:
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
        args.stats_output.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Rejudge fixed GSM rollouts with the compact canonical-state judge."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PARENT = REPO_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from jca.gsm.scripts.rejudge_gsm_collaboration_v3 import (  # noqa: E402
    GroupKey,
    build_result,
    deterministic_states,
    read_groups,
)
from jca.gsm.scripts.prepare_gsm_judge_rl_v4_data import (  # noqa: E402
    all_failed_problem_ids,
)
from jca.gsm.src.data import load_gsm_hard  # noqa: E402
from jca.gsm.src.judge_v4 import (  # noqa: E402
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL,
    JUDGE_PARSE_RETRIES,
    JUDGE_REASONING_EFFORT,
    JUDGE_TEMPERATURE,
    JUDGE_TOP_P,
    CollaborationTrajectoryScore,
    CompactTurnScore,
    build_judge_prompt,
    get_judge_prompt_variant,
    judge_trajectory,
    parse_turn_scores,
    scores_respect_exact_verifier,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rejudge GSM fixed-A1 rollouts with compact canonical scoring.",
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
    parser.add_argument("--judge-reasoning-effort", default=JUDGE_REASONING_EFFORT)
    parser.add_argument("--judge-parse-retries", type=int, default=JUDGE_PARSE_RETRIES)
    parser.add_argument(
        "--project-conflicting-positive-scores",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "After all judge repair retries fail, project only positive scores "
            "that contradict an exact-verifier incorrect state to 0.0 and "
            "record the projection in the validation audit."
        ),
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--limit-groups", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument(
        "--drop-all-failed-problems",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Drop every trajectory for problems whose complete rollout set "
            "has terminal EM 0, matching downstream training-data filtering."
        ),
    )
    parser.add_argument("--expected-rollouts-per-problem", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _complete_group(rows: List[Dict[str, Any]]) -> bool:
    if not rows:
        return False
    turns = sorted(int(row.get("turn", -1)) for row in rows)
    return turns == list(range(len(rows))) and all(
        row.get("collaboration_judge_status_v4") == "scored"
        and isinstance(row.get("collaboration_judge_v4"), dict)
        and isinstance(row.get("deterministic_state_v3"), dict)
        for row in rows
    )


def _prepare_resume_output(
    output: Path,
    resume: bool,
    allowed_keys: Optional[set[GroupKey]] = None,
) -> set[GroupKey]:
    if not output.exists():
        return set()
    if not resume:
        raise FileExistsError(f"output exists: {output}; use --resume or a new path")
    existing = read_groups(output, tolerate_invalid_lines=True)
    complete = {
        key
        for key, rows in existing.items()
        if _complete_group(rows)
        and (allowed_keys is None or key in allowed_keys)
    }
    temporary = output.with_suffix(output.suffix + ".resume.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for key in sorted(complete):
            _write_rows(handle, existing[key])
    temporary.replace(output)
    return complete


def _filter_all_failed_source_groups(
    source_groups: Dict[GroupKey, List[Dict[str, Any]]],
    *,
    enabled: bool,
    expected_rollouts_per_problem: int,
) -> Tuple[Dict[GroupKey, List[Dict[str, Any]]], set[str]]:
    if not enabled:
        return source_groups, set()
    source_rows = [row for rows in source_groups.values() for row in rows]
    dropped_problem_ids = all_failed_problem_ids(
        source_rows,
        expected_rollouts_per_problem=expected_rollouts_per_problem,
    )
    filtered = {
        key: rows
        for key, rows in source_groups.items()
        if key[0] not in dropped_problem_ids
    }
    return filtered, dropped_problem_ids


def _merge_scores(
    rows: List[Dict[str, Any]],
    judged: CollaborationTrajectoryScore,
    states: Dict[int, Dict[str, Any]],
    judge_model: str,
    validation_audit: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    score_by_turn = {score.turn: score for score in judged.turn_scores}
    merged: List[Dict[str, Any]] = []
    for row in rows:
        turn = int(row["turn"])
        score = score_by_turn[turn]
        payload = {
            "process_score": round(score.process_score, 4),
            "judge_score": round(score.judge_score, 4),
            "score_scale": "discrete[-1,-0.5,0,0.5,1]",
            "schema": "compact_canonical_v4",
        }
        output = dict(row)
        # Keep this compatibility field because v5-success data preparation
        # intentionally consumes only judge_score and deterministic state.
        output["collaboration_judge_v3"] = dict(payload)
        output["collaboration_judge_v4"] = dict(payload)
        output["deterministic_state_v3"] = states[turn]
        output["judge_state_disagreements_v3"] = []
        output["collaboration_judge_model_v3"] = judge_model
        output["collaboration_judge_status_v3"] = "scored"
        output["collaboration_judge_model_v4"] = judge_model
        output["collaboration_judge_status_v4"] = "scored"
        output["collaboration_judge_schema_v4"] = "compact_canonical_v4"
        output["collaboration_judge_prompt_variant_v4"] = (
            get_judge_prompt_variant()
        )
        output["collaboration_judge_validation_repair_count_v4"] = len(
            validation_audit
        )
        projections = [
            entry
            for entry in validation_audit
            if entry.get("issue")
            == "conflicting_positive_scores_projected_to_zero"
        ]
        if projections:
            output["collaboration_judge_validation_projection_v4"] = dict(
                projections[-1]
            )
        if turn == 0 and validation_audit:
            output["collaboration_judge_validation_audit_v4"] = list(
                validation_audit
            )
        merged.append(output)
    return merged


def _project_conflicting_positive_scores(
    result: Dict[str, Any],
    states: Dict[int, Dict[str, Any]],
    validation_audit: List[Dict[str, Any]],
) -> Optional[CollaborationTrajectoryScore]:
    steps = result.get("trajectory", {}).get("steps", [])
    for entry in reversed(validation_audit):
        if entry.get("issue") != "positive_score_for_incorrect_candidate":
            continue
        parsed = parse_turn_scores(str(entry.get("response") or ""), steps)
        if parsed is None:
            continue
        conflicting_turns = [
            score.turn
            for score in parsed
            if states.get(score.turn, {}).get("current_answer_correct") is False
            and score.process_score > 0.0
        ]
        if not conflicting_turns:
            continue
        conflicting_set = set(conflicting_turns)
        projected = [
            CompactTurnScore(
                turn=score.turn,
                agent_id=score.agent_id,
                process_score=(
                    0.0 if score.turn in conflicting_set else score.process_score
                ),
            )
            for score in parsed
        ]
        if not scores_respect_exact_verifier(projected, states):
            continue
        validation_audit.append({
            "issue": "conflicting_positive_scores_projected_to_zero",
            "conflicting_turns": conflicting_turns,
            "original_scores": {
                str(score.turn): score.process_score for score in parsed
            },
            "projected_scores": {
                str(score.turn): score.process_score for score in projected
            },
            "policy": "nearest_nonpositive_discrete_score",
        })
        problem = result.get("problem", {})
        trajectory = result.get("trajectory", {})
        return CollaborationTrajectoryScore(
            problem_id=str(
                problem.get("id") or trajectory.get("problem_id") or ""
            ),
            turn_scores=projected,
        )
    return None


def _judge_group(
    rows: List[Dict[str, Any]],
    problem: Any,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    result = build_result(rows, problem)
    states = deterministic_states(rows, problem)
    last_error: Optional[Exception] = None
    validation_audit: List[Dict[str, Any]] = []
    for _attempt in range(args.group_retries + 1):
        try:
            judged = judge_trajectory(
                result,
                states,
                model=args.judge_model,
                temperature=args.judge_temperature,
                top_p=args.judge_top_p,
                max_tokens=args.judge_max_tokens,
                reasoning_effort=args.judge_reasoning_effort,
                parse_retries=args.judge_parse_retries,
                validation_audit=validation_audit,
            )
            if judged is None:
                raise RuntimeError("judge returned no parseable score")
            return _merge_scores(
                rows,
                judged,
                states,
                args.judge_model,
                validation_audit,
            )
        except Exception as exc:
            last_error = exc
    if args.project_conflicting_positive_scores:
        projected = _project_conflicting_positive_scores(
            result,
            states,
            validation_audit,
        )
        if projected is not None:
            return _merge_scores(
                rows,
                projected,
                states,
                args.judge_model,
                validation_audit,
            )
    last_validation = validation_audit[-1] if validation_audit else None
    raise RuntimeError(
        "compact canonical judge failed after retries: "
        f"{last_error}; last_validation="
        f"{json.dumps(last_validation, ensure_ascii=False)}"
    )


def _summarize_output(path: Path) -> Dict[str, Any]:
    groups = read_groups(path)
    rows = [row for group_rows in groups.values() for row in group_rows]
    scores = [
        float(row["collaboration_judge_v4"]["judge_score"])
        for row in rows
    ]
    repair_counts = [
        max(
            int(row.get("collaboration_judge_validation_repair_count_v4", 0))
            for row in group_rows
        )
        for group_rows in groups.values()
    ]
    return {
        "groups": len(groups),
        "rows": len(rows),
        "agents": dict(Counter(str(row.get("agent_id")) for row in rows)),
        "score_counts": dict(Counter(str(score) for score in scores)),
        "validation_repair_groups": sum(count > 0 for count in repair_counts),
        "validation_repair_attempts": sum(repair_counts),
        "judge_model": next(
            (
                row.get("collaboration_judge_model_v4")
                for row in rows
                if row.get("collaboration_judge_model_v4")
            ),
            None,
        ),
        "schema": "compact_canonical_v4",
        "prompt_variant": next(
            (
                row.get("collaboration_judge_prompt_variant_v4")
                for row in rows
                if row.get("collaboration_judge_prompt_variant_v4")
            ),
            None,
        ),
    }


def _write_rows(handle: Any, rows: Iterable[Dict[str, Any]]) -> None:
    handle.write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    handle.flush()


def main() -> None:
    args = parse_args()
    if (
        args.max_concurrency <= 0
        or args.group_retries < 0
        or args.expected_rollouts_per_problem <= 0
    ):
        raise SystemExit("concurrency must be positive and retries non-negative")
    if not 0.0 <= args.judge_temperature <= 2.0:
        raise SystemExit("judge temperature must be between 0.0 and 2.0")

    all_source_groups = read_groups(args.input)
    eligible_source_groups, dropped_problem_ids = _filter_all_failed_source_groups(
        all_source_groups,
        enabled=args.drop_all_failed_problems,
        expected_rollouts_per_problem=args.expected_rollouts_per_problem,
    )
    source_groups = eligible_source_groups
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
        sample_problem = problem_map[sample_key[0]]
        result = build_result(sample_rows, sample_problem)
        states = deterministic_states(sample_rows, sample_problem)
        prompt = build_judge_prompt(result, states)
        print("GSM compact canonical rejudge dry-run")
        print(
            f"input_groups={len(all_source_groups)} eligible_groups={len(source_groups)} "
            f"dropped_problems={len(dropped_problem_ids)} "
            f"sample_group={sample_key} "
            f"turns={len(sample_rows)} prompt_chars={len(prompt)} states={len(states)}"
        )
        print("dry-run complete; no judge calls or output writes")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = _prepare_resume_output(
        args.output,
        args.resume,
        allowed_keys=set(source_groups),
    )
    pending = [key for key in sorted(source_groups) if key not in completed]
    print("=" * 72)
    print("GSM compact canonical rejudge")
    print(
        f"input_groups={len(all_source_groups)} "
        f"dropped_all_failed_problems={len(dropped_problem_ids)} "
        f"eligible_groups={len(source_groups)}"
    )
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
                        _judge_group,
                        source_groups[key],
                        problem_map[key[0]],
                        args,
                    ): key
                    for key in pending
                }
                for index, future in enumerate(as_completed(futures), 1):
                    key = futures[future]
                    try:
                        _write_rows(handle, future.result())
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
    stats = _summarize_output(args.output)
    if stats["groups"] != len(source_groups):
        raise SystemExit(
            f"output has {stats['groups']} complete groups, expected {len(source_groups)}"
        )
    stats.update({
        "input_groups": len(all_source_groups),
        "input_problems": len({key[0] for key in all_source_groups}),
        "drop_all_failed_problems": args.drop_all_failed_problems,
        "dropped_all_failed_problem_count": len(dropped_problem_ids),
        "dropped_all_failed_trajectory_count": (
            len(all_source_groups) - len(eligible_source_groups)
        ),
        "eligible_groups": len(eligible_source_groups),
    })
    if args.stats_output:
        args.stats_output.parent.mkdir(parents=True, exist_ok=True)
        args.stats_output.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

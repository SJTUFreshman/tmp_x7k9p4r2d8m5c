#!/usr/bin/env python3
"""Build MATH data with the v13 success-mix signed-RWR contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PARENT = REPO_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from jca.gsm.scripts.prepare_gsm_judge_rl_v5_success_data import build_data  # noqa: E402
from jca.src.math_eval import (  # noqa: E402
    MATH_EVAL_VERSION,
    compute_math_em,
    load_math_problems,
    math_answers_equivalent,
)
from jca.experiments.math_rl_mas_thinking.math_artifact_validator import (  # noqa: E402
    EVALUATOR_FINGERPRINT,
    MATH_EVAL_FINGERPRINT,
)
from jca.experiments.math_rl_mas_thinking.sampling_coverage import validate_sampling_coverage
from jca.src.protocol_json import parse_protocol_object, validate_protocol_object  # noqa: E402


Identity = Tuple[str, int, int]
THINK_RE = re.compile(r"<think\b[^>]*>(.*?)</think\s*>", re.IGNORECASE | re.DOTALL)


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-standard JSON constant: {value}")


def _reject_duplicate_keys(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _strict_json_loads(line: str, *, path: Path, line_no: int) -> Any:
    try:
        return json.loads(
            line,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON at {path}:{line_no}") from exc


def is_documented_thinking_cross_mode(
    sample_enable_thinking: bool,
    eval_enable_thinking: bool,
    eval_require_thinking: bool,
) -> bool:
    """Return whether modes match the recorded GSM-HARD v13 contract."""
    return (
        not sample_enable_thinking
        and eval_enable_thinking
        and eval_require_thinking
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare MATH v13 signed-RWR data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--holdout-output", type=Path, required=True)
    parser.add_argument("--stats-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--math-data-root", type=Path, required=True)
    parser.add_argument("--expected-rollouts-per-problem", type=int, default=8)
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--a1-correct-ratio", type=float, default=0.30)
    parser.add_argument("--no-correction-ratio", type=float, default=0.20)
    parser.add_argument("--correction-success-ratio", type=float, default=0.35)
    parser.add_argument("--correction-failed-ratio", type=float, default=0.15)
    parser.add_argument("--max-train-trajectories", type=int, default=0)
    parser.add_argument("--max-holdout-trajectories", type=int, default=0)
    parser.add_argument("--base-reward-weight", type=float, default=0.35)
    parser.add_argument("--judge-reward-weight", type=float, default=0.65)
    parser.add_argument("--transition-boost", type=float, default=1.25)
    parser.add_argument("--correct-to-correct-coef-a1", type=float, default=1.0)
    parser.add_argument("--correct-to-correct-coef-a2", type=float, default=0.5)
    parser.add_argument("--correct-to-correct-coef-a3", type=float, default=1.0)
    parser.add_argument("--sample-start-agent", default="A1")
    parser.add_argument("--sample-t-max", type=int, default=8)
    parser.add_argument("--sample-temperature", type=float, default=0.9)
    parser.add_argument("--sample-top-p", type=float, default=0.95)
    parser.add_argument("--sample-max-new-tokens", type=int, default=8192)
    parser.add_argument("--sample-generation-seed", type=int, default=42)
    parser.add_argument(
        "--sample-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--sample-require-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-top-p", type=float, default=0.95)
    parser.add_argument("--judge-max-tokens", type=int, default=8192)
    parser.add_argument("--judge-auto-resume-passes", type=int, default=5)
    parser.add_argument("--train-max-seq", type=int, default=8192)
    parser.add_argument("--train-learning-rate", type=float, default=3e-6)
    parser.add_argument("--train-kl-coef", type=float, default=0.1)
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--eval-start-agent", default="random")
    parser.add_argument("--eval-start-agent-seed", type=int, default=43)
    parser.add_argument("--eval-generation-seed", type=int, default=42)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--eval-top-p", type=float, default=0.95)
    parser.add_argument("--eval-max-new-tokens", type=int, default=8192)
    parser.add_argument(
        "--eval-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--eval-require-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--allow-thinking-mode-mismatch",
        action="store_true",
        help="Allow an intentional cross-mode evaluation ablation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = []
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = _strict_json_loads(line, path=path, line_no=line_no)
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row is not an object at {path}:{line_no}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
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


def identity(row: Dict[str, Any]) -> Identity:
    return (
        str(row.get("problem_id") or ""),
        int(row.get("rollout_idx", -1)),
        int(row.get("turn", -1)),
    )


def protocol_payload(row: Dict[str, Any], *, strict: bool = False) -> str:
    value = row.get("protocol_response")
    if value is None:
        response = row.get("response")
        if not isinstance(response, str):
            raise ValueError("missing response")
        value = THINK_RE.sub("", response).strip()
    if not strict and row.get("agent_id") is None and "trajectory" not in row:
        candidate = parse_protocol_object(value)
    else:
        candidate = validate_protocol_object(value, active_agent=row.get("agent_id"))
    return json.dumps(candidate, ensure_ascii=False, allow_nan=False)


def row_thinking_mode(row: Dict[str, Any]) -> Optional[bool]:
    values = []
    for field in ("thinking_enabled", "enable_thinking"):
        if field not in row or row.get(field) is None:
            continue
        value = row.get(field)
        if not isinstance(value, bool):
            raise ValueError(f"{field} must be boolean at {identity(row)}")
        values.append(value)
    if not values:
        return None
    if len(set(values)) != 1:
        raise ValueError(f"conflicting thinking flags at {identity(row)}")
    return values[0]


def assert_sampling_contract(
    row: Dict[str, Any],
    *,
    thinking_enabled: bool,
    thinking_required: bool,
) -> None:
    response = row.get("response")
    thinking = str(row.get("thinking") or "").strip()
    if not isinstance(response, str):
        raise ValueError(f"response missing at {identity(row)}")
    if re.search(r"<think\b[^>]*>(?:(?!</think\s*>).)*$", response, re.IGNORECASE | re.DOTALL):
        raise ValueError(f"unclosed thinking tag at {identity(row)}")
    if row_thinking_mode(row) is not thinking_enabled:
        raise ValueError(f"thinking mode mismatch at {identity(row)}")
    matches = list(THINK_RE.finditer(response))
    if thinking_enabled:
        if not thinking:
            raise ValueError(f"thinking missing at {identity(row)}")
        if not matches:
            raise ValueError(f"thinking tags missing from response at {identity(row)}")
        if not any(match.group(1).strip() for match in matches):
            raise ValueError(f"thinking trace is empty at {identity(row)}")
        if not any(match.group(1).strip() == thinking for match in matches):
            raise ValueError(f"stored thinking disagrees with response at {identity(row)}")
        if row.get("thinking_retained_in_response") is not True:
            raise ValueError(f"thinking provenance missing at {identity(row)}")
    else:
        if thinking or matches or re.search(r"<\/?think\b", response, re.IGNORECASE):
            raise ValueError(f"non-thinking response contains thinking at {identity(row)}")
        if row.get("thinking_retained_in_response") is not False:
            raise ValueError(f"non-thinking response has inconsistent thinking provenance at {identity(row)}")
    protocol_payload(row, strict=row.get("agent_id") is not None)


def assert_judge_contract(row: Dict[str, Any]) -> None:
    """Reject judged rows produced by an older evaluator or partial write."""
    if row.get("math_eval_version") != MATH_EVAL_VERSION:
        raise ValueError(f"stale math evaluator version at {identity(row)}")
    if row.get("math_eval_fingerprint") != MATH_EVAL_FINGERPRINT:
        raise ValueError(f"stale math evaluator fingerprint at {identity(row)}")
    if row.get("evaluator_fingerprint") != EVALUATOR_FINGERPRINT:
        raise ValueError(f"stale runtime evaluator fingerprint at {identity(row)}")
    if row.get("collaboration_judge_status_v3") != "scored" or row.get(
        "collaboration_judge_status_v4"
    ) != "scored":
        raise ValueError(f"unscored judged row at {identity(row)}")
    if row.get("collaboration_judge_schema_v4") != "compact_canonical_math_v1":
        raise ValueError(f"unexpected judge schema at {identity(row)}")
    if row.get("judge_thinking_enabled") is not True:
        raise ValueError(f"judge thinking contract missing at {identity(row)}")
    state = row.get("deterministic_state_v3")
    if not isinstance(state, dict) or state.get("math_eval_version") != MATH_EVAL_VERSION:
        raise ValueError(f"stale deterministic state at {identity(row)}")
    if state.get("math_eval_fingerprint") not in {None, MATH_EVAL_FINGERPRINT}:
        raise ValueError(f"stale deterministic-state fingerprint at {identity(row)}")
    if state.get("evaluator_fingerprint") not in {None, EVALUATOR_FINGERPRINT}:
        raise ValueError(f"stale deterministic-state runtime fingerprint at {identity(row)}")
    if not isinstance(row.get("judge_state_disagreements_v3"), list):
        raise ValueError(f"invalid judge disagreement ledger at {identity(row)}")
    for field in ("collaboration_judge_v3", "collaboration_judge_v4"):
        payload = row.get(field)
        if not isinstance(payload, dict):
            raise ValueError(f"missing {field} at {identity(row)}")
        if payload.get("schema") != "compact_canonical_math_v1":
            raise ValueError(f"unexpected {field} schema at {identity(row)}")
        if payload.get("math_eval_version") != MATH_EVAL_VERSION:
            raise ValueError(f"stale {field} at {identity(row)}")
        if payload.get("math_eval_fingerprint") not in {None, MATH_EVAL_FINGERPRINT}:
            raise ValueError(f"stale {field} fingerprint at {identity(row)}")
        if payload.get("evaluator_fingerprint") not in {None, EVALUATOR_FINGERPRINT}:
            raise ValueError(f"stale {field} runtime fingerprint at {identity(row)}")
        score = payload.get("process_score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or float(score) not in {-1.0, -0.5, 0.0, 0.5, 1.0}
        ):
            raise ValueError(f"invalid {field} score at {identity(row)}")
        if payload.get("score_scale") not in {None, "discrete[-1,-0.5,0,0.5,1]"}:
            raise ValueError(f"invalid {field} score scale at {identity(row)}")
        if "judge_score" in payload and payload.get("judge_score") != float(score):
            raise ValueError(f"{field} judge_score disagrees at {identity(row)}")
    for field in ("collaboration_judge_model_v3", "collaboration_judge_model_v4"):
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(f"missing {field} at {identity(row)}")
    if not isinstance(row.get("thinking_enabled"), bool) or not isinstance(
        row.get("enable_thinking"), bool
    ) or row["thinking_enabled"] != row["enable_thinking"]:
        raise ValueError(f"thinking aliases are missing or conflicting at {identity(row)}")


def assert_terminal_em_contract(row: Dict[str, Any], gold_answer: str) -> None:
    """Reject terminal labels that do not match symbolic MATH evaluation."""
    final_answer = row.get("final_answer")
    if final_answer is None:
        final_answer = ""
    elif not isinstance(final_answer, str):
        raise ValueError(f"final_answer must be a string or null at {identity(row)}")
    expected = compute_math_em(final_answer, gold_answer)
    actual = row.get("em")
    if isinstance(actual, bool) or not isinstance(actual, (int, float)):
        raise ValueError(f"missing or invalid terminal EM at {identity(row)}")
    try:
        actual_value = float(actual)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid terminal EM at {identity(row)}") from exc
    if actual_value not in {0.0, 1.0}:
        raise ValueError(f"terminal EM must be binary at {identity(row)}")
    if actual_value != expected:
        raise ValueError(
            f"terminal EM disagrees with {MATH_EVAL_VERSION} at {identity(row)}: "
            f"stored={actual_value} expected={expected}"
        )


def assert_trajectory_terminal_contract(
    rows: list[Dict[str, Any]],
    gold_answer: str,
) -> None:
    """Validate trajectory-level terminal and handoff bindings before selection."""
    if not rows:
        raise ValueError("empty trajectory group")
    ordered = sorted(rows, key=lambda row: int(row["turn"]))
    turns = [int(row["turn"]) for row in ordered]
    if turns != list(range(len(turns))):
        raise ValueError("trajectory turns are not contiguous")
    statuses = {row.get("terminated_by") for row in ordered}
    if len(statuses) != 1:
        raise ValueError("trajectory termination status differs across turns")
    status = next(iter(statuses))
    if status not in {"stop", "truncated", "rejected_quality", "exception"}:
        raise ValueError(f"invalid trajectory termination status: {status!r}")
    parsed_steps = []
    final_answers = []
    expected_agent = None
    for index, row in enumerate(ordered):
        agent = row.get("agent_id")
        if agent not in {"A1", "A2", "A3"}:
            raise ValueError(f"invalid trajectory agent at turn {index}")
        if index == 0 and row.get("start_agent") in {"A1", "A2", "A3"} and agent != row.get("start_agent"):
            raise ValueError("trajectory first agent disagrees with start_agent")
        if expected_agent is not None and agent != expected_agent:
            raise ValueError("trajectory active-agent sequence disagrees")
        value = row.get("protocol_response")
        if value is None:
            response = row.get("response")
            if not isinstance(response, str):
                raise ValueError("trajectory response is missing")
            value = THINK_RE.sub("", response).strip()
        parsed = validate_protocol_object(value, active_agent=agent)
        if row.get("action") not in {None, parsed["action"]}:
            raise ValueError("trajectory row action disagrees with protocol")
        parsed_steps.append(parsed)
        final = row.get("final_answer")
        if final is None:
            final = ""
        if not isinstance(final, str):
            raise ValueError("trajectory final_answer must be a string or null")
        final_answers.append(final.strip())
        expected_agent = parsed.get("handoff_target") if parsed["action"] == "handoff" else None
        if parsed["action"] == "confirm_stop" and index + 1 < len(ordered):
            raise ValueError("trajectory contains a turn after confirm_stop")
    terminal = final_answers[-1]
    for final in final_answers:
        if bool(final) != bool(terminal) or (final and not math_answers_equivalent(final, terminal)):
            raise ValueError("trajectory rows disagree on final_answer")
    if status == "stop":
        if parsed_steps[-1]["action"] != "confirm_stop" or not terminal:
            raise ValueError("stop trajectory is not bound to a confirmation")
        confirmed = str(parsed_steps[-1].get("confirmed_answer") or "").strip()
        if not confirmed or not math_answers_equivalent(terminal, confirmed):
            raise ValueError("terminal final_answer disagrees with confirmed_answer")
        tentative = str(parsed_steps[-1].get("tentative_answer") or "").strip()
        if not tentative or not math_answers_equivalent(terminal, tentative):
            raise ValueError("terminal final_answer disagrees with tentative_answer")
        if compute_math_em(terminal, gold_answer) not in {0.0, 1.0}:
            raise ValueError("terminal EM is invalid")
    elif terminal:
        raise ValueError("non-stop trajectory has a terminal final_answer")


def main() -> None:
    args = parse_args()
    if args.sample_require_thinking and not args.sample_enable_thinking:
        raise SystemExit("--sample-require-thinking requires --sample-enable-thinking")
    if args.eval_require_thinking and not args.eval_enable_thinking:
        raise SystemExit("--eval-require-thinking requires --eval-enable-thinking")
    documented_cross_mode = is_documented_thinking_cross_mode(
        args.sample_enable_thinking,
        args.eval_enable_thinking,
        args.eval_require_thinking,
    )
    if (
        not args.allow_thinking_mode_mismatch
        and args.sample_enable_thinking != args.eval_enable_thinking
        and not documented_cross_mode
    ):
        raise SystemExit(
            "sampling/training and evaluation thinking modes differ outside the "
            "documented non-thinking-train/thinking-eval contract; use matching "
            "flags or explicitly pass --allow-thinking-mode-mismatch"
        )
    outputs = (
        args.train_output,
        args.holdout_output,
        args.stats_output,
        args.manifest_output,
    )
    if not args.overwrite and any(path.exists() for path in outputs):
        raise SystemExit("output exists; pass --overwrite or choose another tag")

    original = read_jsonl(args.input)
    if not original:
        raise SystemExit("input is empty")
    problems = load_math_problems(args.math_data_root, split="train")
    problem_map = {problem.problem_id: problem for problem in problems}
    original_by_id: Dict[Identity, Dict[str, Any]] = {}
    original_groups: Dict[Tuple[str, int], list[Dict[str, Any]]] = {}
    transformed = []
    for source in original:
        if source.get("agent_id") not in {"A1", "A2", "A3"}:
            raise ValueError(f"source has an invalid agent_id at {identity(source)}")
        assert_sampling_contract(
            source,
            thinking_enabled=args.sample_enable_thinking,
            thinking_required=args.sample_require_thinking,
        )
        assert_judge_contract(source)
        problem_id = str(source.get("problem_id") or "")
        problem = problem_map.get(problem_id)
        if problem is None:
            raise ValueError(f"unknown MATH problem: {problem_id}")
        assert_terminal_em_contract(source, problem.gold_answer)
        key = identity(source)
        if key in original_by_id:
            raise ValueError(f"duplicate row identity: {key}")
        original_by_id[key] = source
        original_groups.setdefault((key[0], key[1]), []).append(source)
        row = dict(source)
        # The unchanged v13 builder consumes protocol JSON. The original
        # policy target is restored after selection and reward construction.
        row["response"] = protocol_payload(source, strict=True)
        transformed.append(row)

    for (problem_id, _rollout_idx), rows in original_groups.items():
        assert_trajectory_terminal_contract(rows, problem_map[problem_id].gold_answer)

    failure_ledger = os.environ.get("MATH_FAILURE_LEDGER")
    sample_state = os.environ.get("MATH_SAMPLE_STATE")
    sampling_coverage = validate_sampling_coverage(
        original_groups,
        expected_rollouts=args.expected_rollouts_per_problem,
        data_root=args.math_data_root,
        failure_ledger=Path(failure_ledger) if failure_ledger else None,
        sample_state=Path(sample_state) if sample_state else None,
    )
    builder_rollout_count = (
        None
        if failure_ledger and sample_state and sampling_coverage["zero_step_failures"]
        else args.expected_rollouts_per_problem
    )

    ratios = {
        "a1_correct": args.a1_correct_ratio,
        "no_correction": args.no_correction_ratio,
        "correction_success": args.correction_success_ratio,
        "correction_failed": args.correction_failed_ratio,
    }
    train, holdout, stats = build_data(
        transformed,
        expected_rollouts_per_problem=builder_rollout_count,
        drop_all_failed_problems=True,
        holdout_fraction=args.holdout_fraction,
        ratios=ratios,
        max_train_trajectories=args.max_train_trajectories,
        max_holdout_trajectories=args.max_holdout_trajectories,
        base_reward_weight=args.base_reward_weight,
        judge_reward_weight=args.judge_reward_weight,
        transition_boost=args.transition_boost,
        min_aligned_judge=0.0,
        seed=args.seed,
        correct_to_correct_coef=1.0,
        drop_wrong_to_wrong_stop=False,
        correct_to_correct_coef_by_agent={
            "A1": args.correct_to_correct_coef_a1,
            "A2": args.correct_to_correct_coef_a2,
            "A3": args.correct_to_correct_coef_a3,
        },
        correct_to_correct_handoff_coef=1.0,
        wrong_to_correct_multiplier=None,
        correct_to_wrong_multiplier=None,
        expected_start_agent="A1",
    )

    def restore(rows: list[Dict[str, Any]]) -> None:
        for row in rows:
            source = original_by_id[identity(row)]
            assert_sampling_contract(
                source,
                thinking_enabled=args.sample_enable_thinking,
                thinking_required=args.sample_require_thinking,
            )
            row["protocol_response"] = protocol_payload(source, strict=True)
            row["response"] = source["response"]
            row["raw_response"] = source.get("raw_response")
            row["raw_outputs"] = source.get("raw_outputs", [])
            row["thinking"] = source["thinking"]
            row["thinking_enabled"] = args.sample_enable_thinking
            row["enable_thinking"] = args.sample_enable_thinking
            row["thinking_retained_in_response"] = bool(
                source.get("thinking_retained_in_response")
            )
            row["math_eval_version"] = MATH_EVAL_VERSION
            row["math_eval_fingerprint"] = MATH_EVAL_FINGERPRINT
            row["evaluator_fingerprint"] = EVALUATOR_FINGERPRINT
            sample_mode = "thinking" if args.sample_enable_thinking else "nonthinking"
            row["sample_source"] = f"math_v13_{sample_mode}_signed_rwr"
            row["reward_source"] = "math_exact_state_plus_qwen14b_process"
            row["dataset"] = "MATH"

    restore(train)
    restore(holdout)
    if not train or not holdout:
        raise ValueError("prepared split is empty")
    if args.sample_enable_thinking:
        if not all("<think>" in str(row["response"]) for row in train + holdout):
            raise ValueError("prepared data lost thinking targets")
    elif any("<think>" in str(row["response"]) for row in train + holdout):
        raise ValueError("prepared non-thinking data contains thinking targets")

    train_problem_ids = {str(row["problem_id"]) for row in train}
    holdout_problem_ids = {str(row["problem_id"]) for row in holdout}
    if train_problem_ids & holdout_problem_ids:
        raise ValueError("problem-level train/holdout leakage")
    math_train_ids = set(problem_map)
    if (train_problem_ids | holdout_problem_ids) - math_train_ids:
        raise ValueError("prepared data contains a problem outside MATH train")

    sample_mode = "thinking" if args.sample_enable_thinking else "nonthinking"
    target_format = (
        "<think>...</think> followed by protocol JSON"
        if args.sample_enable_thinking
        else "protocol JSON"
    )
    stats["version"] = f"math_rl_mas_v13_{sample_mode}_sample"
    stats["dataset"] = "MATH"
    stats["sampling_coverage"] = sampling_coverage
    stats["thinking_contract"] = {
        "sampling_enable_thinking": args.sample_enable_thinking,
        "nonempty_thinking_required": args.sample_require_thinking,
        "thinking_retained_in_training_response": args.sample_enable_thinking,
        "training_enable_thinking": args.sample_enable_thinking,
        "evaluation_enable_thinking": args.eval_enable_thinking,
        "evaluation_require_thinking": args.eval_require_thinking,
        "documented_cross_mode": documented_cross_mode,
        "training_target_format": target_format,
        "prepared_train_rows": len(train),
        "prepared_holdout_rows": len(holdout),
    }
    stats["source_schema"] = "math_rl_mas_turns_v1"
    stats["math_eval_version"] = MATH_EVAL_VERSION
    stats["math_eval_fingerprint"] = MATH_EVAL_FINGERPRINT
    stats["evaluator_fingerprint"] = EVALUATOR_FINGERPRINT
    stats["judge_schema"] = "compact_canonical_math_v1"
    input_path = args.input.resolve()
    stats["source_input"] = {
        "path": str(input_path),
        "size": input_path.stat().st_size,
        "sha256": file_sha256(input_path),
    }

    manifest = {
        "experiment": "MATH reproduction of GSM RL-MAS v13",
        "reference_gsm_result": {"em": 0.7197, "f1": 0.8115},
        "base_models": {
            "A1": "Qwen3-1.7B",
            "A2": "Qwen3-4B",
            "A3": "Qwen3-8B",
        },
        "dataset": {
            "name": "MATH",
            "source_split": "train",
            "evaluation_split": "test",
            "source_problem_count": len(math_train_ids),
        },
        "sampling": {
            "rollouts_per_problem": args.expected_rollouts_per_problem,
            "coverage": sampling_coverage,
            "start_agent": args.sample_start_agent,
            "t_max": args.sample_t_max,
            "temperature": args.sample_temperature,
            "top_p": args.sample_top_p,
            "max_new_tokens": args.sample_max_new_tokens,
            "generation_seed": args.sample_generation_seed,
            "thinking_enabled": args.sample_enable_thinking,
            "thinking_required": args.sample_require_thinking,
            "target_format": target_format,
        },
        "math_eval_version": MATH_EVAL_VERSION,
        "math_eval_fingerprint": MATH_EVAL_FINGERPRINT,
        "evaluator_fingerprint": EVALUATOR_FINGERPRINT,
        "judge": {
            "model": "Qwen3-14B",
            "temperature": args.judge_temperature,
            "top_p": args.judge_top_p,
            "max_tokens": args.judge_max_tokens,
            "auto_resume_passes": args.judge_auto_resume_passes,
            "thinking_enabled": True,
            "schema": "compact_canonical_math_v1",
        },
        "training": {
            "objective": "format-protected signed-RWR with token-level forward KL",
            "max_seq_length": args.train_max_seq,
            "learning_rate": args.train_learning_rate,
            "kl_coef": args.train_kl_coef,
            "epochs": args.train_epochs,
            "target_includes_thinking": args.sample_enable_thinking,
        },
        "evaluation": {
            "start_agent": args.eval_start_agent,
            "start_agent_seed": args.eval_start_agent_seed,
            "generation_seed": args.eval_generation_seed,
            "temperature": args.eval_temperature,
            "top_p": args.eval_top_p,
            "max_new_tokens": args.eval_max_new_tokens,
            "thinking_enabled": args.eval_enable_thinking,
            "thinking_required": args.eval_require_thinking,
        },
        "reward": stats["reward"],
        "objective": stats["objective"],
        "outputs": {
            "train": str(args.train_output),
            "holdout": str(args.holdout_output),
            "stats": str(args.stats_output),
        },
        "source_input": {
            "path": str(input_path),
            "size": input_path.stat().st_size,
            "sha256": stats["source_input"]["sha256"],
        },
    }
    write_jsonl(args.train_output, train)
    write_jsonl(args.holdout_output, holdout)
    write_json(args.stats_output, stats)
    write_json(args.manifest_output, manifest)
    print(
        f"MATH v13 {sample_mode} data ready: train_rows={len(train)} "
        f"holdout_rows={len(holdout)} thinking_retained={int(args.sample_enable_thinking)}"
    )


if __name__ == "__main__":
    main()

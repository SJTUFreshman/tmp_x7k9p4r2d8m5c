#!/usr/bin/env python3
"""Collect fixed-A1 MATH protocol SFT trajectories from base models."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import re
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Optional, Sequence


BUNDLED_PACKAGE_PARENT = Path(__file__).resolve().parent / "vendor"
if str(BUNDLED_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(BUNDLED_PACKAGE_PARENT))

from jca.gsm.scripts import run_mas as protocol  # noqa: E402
from jca.src.inference import GenerationOptions, OpenAIChatLLMCaller  # noqa: E402
from jca.src.math_eval import (  # noqa: E402
    MathProblem,
    format_math_problem_as_prompt,
    load_math_problems,
    math_answers_equivalent,
    render_math_mas_system_prompt,
)


COLLECTOR_VERSION = "math_specific_fixed_a1_rollout_v2"
ROUTES = (("A1", "A2", "A3"), ("A1", "A3", "A2"))
ERROR_TYPES = (
    "sign",
    "offset",
    "scale",
    "reciprocal",
    "square",
    "zero_substitution",
)
ERROR_DESCRIPTIONS = {
    "sign": "a sign or subtraction-direction mistake",
    "offset": "an off-by-one or missing-constant mistake",
    "scale": "a dropped or duplicated multiplicative factor",
    "reciprocal": "an inversion or reciprocal mistake",
    "square": "a mistaken square or missing square root",
    "zero_substitution": "an invalid zero or unit substitution",
}
ERROR_PATTERNS = {
    "sign": re.compile(r"negative|positive|sign|subtract|difference|opposite", re.I),
    "offset": re.compile(r"integer|consecutive|constant|remainder|more|less|difference", re.I),
    "scale": re.compile(r"ratio|product|times|scale|area|volume|similar|percent|fraction", re.I),
    "reciprocal": re.compile(r"reciprocal|slope|rate|fraction|divide|quotient|inverse", re.I),
    "square": re.compile(r"square|quadratic|root|distance|radius|area|pythag", re.I),
    "zero_substitution": re.compile(r"zero|root|solution|value|evaluate|intercept|constant", re.I),
}
CORRECTION_TERMS = {
    "sign": ("sign", "negative", "positive", "subtract", "direction"),
    "offset": ("offset", "constant", "off-by-one", "missing term", "added"),
    "scale": ("scale", "factor", "multiply", "coefficient", "duplicated"),
    "reciprocal": ("reciprocal", "invert", "inverse", "denominator", "division"),
    "square": ("square", "squared", "root", "exponent", "power"),
    "zero_substitution": ("zero", "substitution", "unit", "invalid value", "domain"),
}


@dataclass(frozen=True)
class RolloutProblem:
    id: str
    question: str
    answer_str: str
    subject: str
    level: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=3160)
    parser.add_argument("--max-missing", type=int, default=0)
    parser.add_argument("--plan-offset", type=int, default=0)
    parser.add_argument("--api-base-a1", default="http://127.0.0.1:8201/v1")
    parser.add_argument("--api-base-a2", default="http://127.0.0.1:8202/v1")
    parser.add_argument("--api-base-a3", default="http://127.0.0.1:8203/v1")
    parser.add_argument("--api-model-a1", default="A1")
    parser.add_argument("--api-model-a2", default="A2")
    parser.add_argument("--api-model-a3", default="A3")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--api-timeout", type=int, default=600)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-concurrency", type=int, default=64)
    parser.add_argument("--max-step-retries", type=int, default=2)
    parser.add_argument("--max-trajectory-attempts", type=int, default=20)
    parser.add_argument("--max-verifier-similarity", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validate-existing", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.start < 0 or args.plan_offset < 0:
        parser.error("--start and --plan-offset must be non-negative")
    if args.limit <= 0 or args.limit % 40:
        parser.error("--limit must be a positive multiple of 40")
    if not 0 <= args.max_missing < args.limit:
        parser.error("--max-missing must be in [0, limit)")
    if args.max_concurrency <= 0 or args.max_step_retries < 0:
        parser.error("concurrency must be positive and step retries non-negative")
    if args.max_trajectory_attempts <= 0:
        parser.error("--max-trajectory-attempts must be positive")
    if not 0.0 <= args.max_verifier_similarity <= 1.0:
        parser.error("--max-verifier-similarity must be in [0, 1]")
    return args


def as_rollout_problem(problem: MathProblem) -> RolloutProblem:
    return RolloutProblem(
        id=problem.problem_id,
        question=problem.prompt,
        answer_str=problem.gold_answer,
        subject=problem.subject,
        level=problem.level,
    )


def select_plan(index: int) -> tuple[str, tuple[str, str, str], Optional[str]]:
    cycle_position = index % 20
    cycle_number = index // 20
    route_index = cycle_position % 2
    route_copy = cycle_position // 2
    route = ROUTES[route_index]
    if route_copy >= 3:
        return "regular", route, None
    error_index = (2 * route_copy + route_index + cycle_number) % len(ERROR_TYPES)
    return "correction", route, ERROR_TYPES[error_index]


def eligible_for_error(problem: RolloutProblem, error_type: str) -> bool:
    return bool(ERROR_PATTERNS[error_type].search(problem.question))


def build_error_pools(
    problems: Sequence[RolloutProblem],
) -> dict[str, list[RolloutProblem]]:
    pools = {
        error_type: [problem for problem in problems if eligible_for_error(problem, error_type)]
        for error_type in ERROR_TYPES
    }
    sparse = {name: len(values) for name, values in pools.items() if len(values) < 100}
    if sparse:
        raise ValueError(f"MATH error-type pools are unexpectedly sparse: {sparse}")
    return pools


def mutate_answer(gold: str, error_type: str) -> str:
    value = gold.strip()
    candidates = {
        "sign": [f"-({value})", f"({value})+1"],
        "offset": [f"({value})+1", f"({value})-1"],
        "scale": [f"2({value})", f"\\frac{{{value}}}{{2}}"],
        "reciprocal": [f"\\frac{{1}}{{{value}}}", f"({value})+1"],
        "square": [f"({value})^2", f"\\sqrt{{{value}}}"],
        "zero_substitution": ["0", "1", f"({value})+1"],
    }[error_type]
    for candidate in candidates:
        if not math_answers_equivalent(candidate, value):
            return candidate
    raise ValueError(f"could not make a non-equivalent {error_type} answer from {value!r}")


def reasoning_similarity(left: str, right: str) -> float:
    normalized_left = " ".join((left or "").lower().split())
    normalized_right = " ".join((right or "").lower().split())
    if not normalized_left or not normalized_right:
        return 0.0
    return difflib.SequenceMatcher(None, normalized_left, normalized_right).ratio()


def correct_step_rejection(
    step: protocol.GSMStep,
    *,
    gold_answer: str,
    prior_reasonings: Sequence[str],
    max_similarity: float,
) -> Optional[str]:
    if not protocol.has_parseable_json_object(step.raw_output):
        return "the response was not a parseable JSON object"
    if protocol.is_empty_step(step):
        return "the response did not contain reasoning and a tentative answer"
    if not math_answers_equivalent(step.tentative_answer, gold_answer):
        return "the tentative answer did not match the trusted MATH answer"
    if step.confirmed_answer and not math_answers_equivalent(
        step.confirmed_answer, gold_answer
    ):
        return "the confirmed answer did not match the trusted MATH answer"
    normalized = " ".join(step.reasoning.lower().split())
    if any(
        phrase in normalized
        for phrase in (
            "reference answer",
            "trusted answer",
            "provided answer",
            "hidden instruction",
            "control instruction",
        )
    ):
        return "the reasoning revealed generation-only answer guidance"
    if any(
        reasoning_similarity(step.reasoning, previous) >= max_similarity
        for previous in prior_reasonings
        if previous.strip()
    ):
        return "the reasoning repeated an earlier agent instead of independently checking"
    return None


def generate_step(
    *,
    caller: OpenAIChatLLMCaller,
    messages: list[dict[str, str]],
    active_agent: str,
    prior_tentative: bool,
    prior_reasonings: Sequence[str],
    gold_answer: str,
    request_instruction: str,
    retries: int,
    max_similarity: float,
    expected_wrong_answer: Optional[str] = None,
    correction_error_type: Optional[str] = None,
) -> protocol.GSMStep:
    rejection: Optional[str] = None
    for _ in range(retries + 1):
        request_messages = [dict(message) for message in messages]
        instruction = request_instruction
        if rejection:
            instruction += (
                f" The previous draft was rejected because {rejection}. "
                "Recompute and output one new JSON object only."
            )
        request_messages.append({"role": "user", "content": instruction})
        raw = caller(request_messages)
        candidate = protocol.parse_action(raw, active_agent, prior_tentative)
        if expected_wrong_answer is not None:
            normalized = " ".join(candidate.reasoning.lower().split())
            if not protocol.has_parseable_json_object(raw):
                rejection = "the response was not a parseable JSON object"
            elif protocol.is_empty_step(candidate):
                rejection = "the response was empty"
            elif math_answers_equivalent(candidate.tentative_answer, gold_answer):
                rejection = "the intentionally flawed answer accidentally matched gold"
            elif not math_answers_equivalent(
                candidate.tentative_answer, expected_wrong_answer
            ):
                rejection = "the tentative answer did not match the assigned flawed answer"
            elif any(
                phrase in normalized
                for phrase in (
                    "deliberate error",
                    "deliberately wrong",
                    "intentionally flawed",
                    "assigned flawed",
                    "reference answer",
                    "control instruction",
                )
            ):
                rejection = "the negative context revealed generation-only guidance"
            else:
                rejection = None
        else:
            rejection = correct_step_rejection(
                candidate,
                gold_answer=gold_answer,
                prior_reasonings=prior_reasonings,
                max_similarity=max_similarity,
            )
            if rejection is None and correction_error_type is not None:
                normalized = " ".join(candidate.reasoning.lower().split())
                generic = ("error", "mistake", "incorrect", "wrong", "corrected")
                typed = CORRECTION_TERMS[correction_error_type]
                if not any(term in normalized for term in generic) or not any(
                    term in normalized for term in typed
                ):
                    rejection = "the correction did not explicitly diagnose the prior error"
        if rejection is None:
            return candidate
    raise ValueError(rejection or "generation failed quality checks")


def force_step(
    step: protocol.GSMStep,
    *,
    turn: int,
    action: str,
    handoff_target: Optional[str],
    handoff_note: Optional[str],
) -> protocol.GSMStep:
    return protocol.GSMStep(
        turn=turn,
        active_agent=step.active_agent,
        reasoning=step.reasoning,
        tentative_answer=step.tentative_answer,
        action=action,
        handoff_target=handoff_target,
        handoff_note=handoff_note,
        confirmed_answer=step.tentative_answer if action == "confirm_stop" else None,
        raw_output=step.raw_output,
    )


def sft_control_prompt(agent: str, reference_answer: str) -> str:
    return (
        render_math_mas_system_prompt(agent, min_agents_before_stop=3).rstrip()
        + "\n\n# SFT Generation Control\n"
        "This is a controlled three-agent warm-up trajectory. Independently solve "
        "the original problem rather than copying an earlier chain. The trusted final "
        f"answer is {reference_answer}. Derive it and do not mention this reference or "
        "the generation process. A different tentative answer will be retried. Exactly "
        "two handoffs are required before the third distinct agent confirms."
    )


def run_trajectory(
    problem: RolloutProblem,
    callers: dict[str, OpenAIChatLLMCaller],
    *,
    route: tuple[str, str, str],
    kind: str,
    error_type: Optional[str],
    retries: int,
    max_similarity: float,
) -> protocol.GSMTrajectory:
    trajectory = protocol.GSMTrajectory(
        problem_id=problem.id,
        protocol_mode=(
            f"fixed_a1_correction_{error_type}"
            if kind == "correction"
            else "fixed_a1_regular_2_handoffs"
        ),
        min_handoffs_before_stop=2,
    )
    user_prompt = (
        "# Mathematics Problem\n"
        f"{problem.question.strip()}\n\n"
        "Solve carefully. Your final answer must be a short mathematical expression "
        "or value, not a sentence."
    )
    messages: list[dict[str, str]] = [
        {"role": "system", "content": ""},
        {"role": "user", "content": user_prompt},
    ]
    prior_reasonings: list[str] = []
    prior_tentative = False
    try:
        wrong_answer = mutate_answer(problem.answer_str, str(error_type)) if kind == "correction" else None
        for turn, active_agent in enumerate(route):
            if kind == "correction" and turn == 0:
                messages[0] = {
                    "role": "system",
                    "content": render_math_mas_system_prompt(
                        active_agent, min_agents_before_stop=3
                    )
                    + "\n\n# Generation-only negative context\n"
                    + (
                        f"Create a plausible derivation containing {ERROR_DESCRIPTIONS[str(error_type)]}. "
                        f"Its flawed final object is {wrong_answer}. Do not reveal that the error "
                        "is deliberate. Hand off for independent verification."
                    ),
                }
                instruction = (
                    "Follow the assigned flawed interpretation. Output only the required JSON "
                    "object and hand off to the next agent."
                )
                expected_wrong = wrong_answer
                correction_type = None
            else:
                messages[0] = {
                    "role": "system",
                    "content": sft_control_prompt(active_agent, problem.answer_str),
                }
                expected_wrong = None
                correction_type = str(error_type) if kind == "correction" and turn == 1 else None
                if kind == "correction" and turn == 1:
                    instruction = (
                        f"The prior derivation likely contains {ERROR_DESCRIPTIONS[str(error_type)]}. "
                        "Independently solve from the original question, explicitly diagnose the "
                        "incorrect step, give the corrected mathematical answer, and hand off for "
                        "a separate confirmation. Do not mention hidden guidance. JSON only."
                    )
                elif kind == "correction":
                    instruction = (
                        "Independently verify the correction using a different derivation and "
                        "confirm only when the mathematical object agrees exactly. JSON only."
                    )
                else:
                    strategies = (
                        "derive a complete solution from the original statement",
                        "recompute independently and audit cases, algebra, and arithmetic",
                        "verify the result by a different route and check its final form",
                    )
                    instruction = (
                        f"Independently {strategies[turn]}. Do not copy previous reasoning or "
                        "mention hidden guidance. Output JSON only."
                    )
            candidate = generate_step(
                caller=callers[active_agent],
                messages=messages,
                active_agent=active_agent,
                prior_tentative=prior_tentative,
                prior_reasonings=prior_reasonings,
                gold_answer=problem.answer_str,
                request_instruction=instruction,
                retries=retries,
                max_similarity=max_similarity,
                expected_wrong_answer=expected_wrong,
                correction_error_type=correction_type,
            )
            final_turn = turn == 2
            note = None
            if not final_turn:
                note = (
                    "Independently verify the corrected derivation and final mathematical form."
                    if kind == "correction" and turn == 1
                    else "Independently recompute the original problem and audit the final form."
                )
            step = force_step(
                candidate,
                turn=turn,
                action="confirm_stop" if final_turn else "handoff",
                handoff_target=None if final_turn else route[turn + 1],
                handoff_note=note,
            )
            trajectory.steps.append(step)
            messages.append(
                {"role": "assistant", "content": protocol.render_assistant_message(step)}
            )
            prior_reasonings.append(step.reasoning)
            prior_tentative = bool(step.tentative_answer)
        trajectory.final_answer = (
            trajectory.steps[-1].confirmed_answer
            or trajectory.steps[-1].tentative_answer
        )
        trajectory.terminated_by = "stop"
    except Exception as exc:
        trajectory.terminated_by = "rejected_quality"
        trajectory.error = str(exc)
    return trajectory


def trajectory_record(
    trajectory: protocol.GSMTrajectory,
    problem: RolloutProblem,
    *,
    plan_index: int,
    kind: str,
    route: tuple[str, str, str],
    error_type: Optional[str],
    attempts: int,
) -> dict[str, Any]:
    record = protocol.trajectory_to_dict(trajectory, problem)
    record["collector_version"] = COLLECTOR_VERSION
    record["dataset"] = "MATH"
    record["source_split"] = "train"
    record["plan_index"] = plan_index
    record["problem"]["subject"] = problem.subject
    record["problem"]["level"] = problem.level
    record["trajectory"].update(
        {
            "trajectory_kind": kind,
            "error_type": error_type,
            "planned_route": list(route),
            "route_signature": ">".join(route),
            "error_problem_eligible": (
                eligible_for_error(problem, str(error_type))
                if kind == "correction"
                else None
            ),
            "generation_attempts": attempts,
        }
    )
    record["em"] = 1.0
    record["f1"] = 1.0
    return record


def read_existing(path: Path, limit: int, plan_offset: int) -> dict[int, dict[str, Any]]:
    existing: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return existing
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank line in rollout at {path}:{line_number}")
            row = json.loads(line)
            index = int(row.get("plan_index", -1))
            if index in existing:
                raise ValueError(f"duplicate plan_index {index} in {path}")
            if not plan_offset <= index < plan_offset + limit:
                raise ValueError(f"out-of-range plan_index {index} in {path}")
            if row.get("collector_version") != COLLECTOR_VERSION:
                raise ValueError(f"stale collector row at {path}:{line_number}")
            if row.get("terminated_by") != "stop" or float(row.get("em", 0.0)) != 1.0:
                raise ValueError(f"non-accepted rollout at {path}:{line_number}")
            expected_kind, expected_route, expected_error = select_plan(index)
            trajectory = row.get("trajectory") or {}
            if (
                trajectory.get("trajectory_kind") != expected_kind
                or tuple(trajectory.get("planned_route") or ()) != expected_route
                or trajectory.get("error_type") != expected_error
            ):
                raise ValueError(f"plan metadata mismatch at {path}:{line_number}")
            existing[index] = row
    return existing


def validate_coverage(
    existing: dict[int, dict[str, Any]],
    *,
    limit: int,
    plan_offset: int,
    max_missing: int,
) -> dict[str, Any]:
    missing = [
        index
        for index in range(plan_offset, plan_offset + limit)
        if index not in existing
    ]
    if len(missing) > max_missing:
        raise ValueError(
            f"rollout coverage mismatch: {len(existing)}/{limit} accepted plans; "
            f"missing={len(missing)} exceeds max_missing={max_missing}"
        )
    return {
        "accepted": len(existing),
        "planned": limit,
        "missing": len(missing),
        "max_missing": max_missing,
        "missing_plan_indices": missing,
    }


def execute_bounded(
    work_items: Sequence[tuple[Any, ...]],
    worker: Any,
    *,
    max_workers: int,
    on_record: Any,
) -> list[str]:
    iterator = iter(work_items)
    failures: list[str] = []
    pending: dict[Future[Any], tuple[Any, ...]] = {}

    def submit_next(executor: ThreadPoolExecutor) -> bool:
        try:
            item = next(iterator)
        except StopIteration:
            return False
        pending[executor.submit(worker, *item)] = item
        return True

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _ in range(min(max_workers, len(work_items))):
            submit_next(executor)
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                item = pending.pop(future)
                try:
                    record = future.result()
                except Exception as exc:
                    plan_index = item[0] if item else "unknown"
                    message = f"plan={plan_index}: {type(exc).__name__}: {exc}"
                    failures.append(message)
                    print(f"[warning] exhausted rollout plan: {message}", file=sys.stderr)
                else:
                    on_record(record)
                submit_next(executor)
    return failures


def main() -> None:
    args = parse_args()
    math_problems = load_math_problems(args.data_root, split="train")
    problems = [as_rollout_problem(problem) for problem in math_problems]
    if not problems:
        raise SystemExit("no MATH train problems were loaded")
    error_pools = build_error_pools(problems)
    existing = read_existing(args.output, args.limit, args.plan_offset)
    if args.validate_existing:
        try:
            coverage = validate_coverage(
                existing,
                limit=args.limit,
                plan_offset=args.plan_offset,
                max_missing=args.max_missing,
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        print(json.dumps({"status": "valid", **coverage}))
        return
    if args.plan_only:
        counts: dict[str, int] = {"regular": 0, "correction": 0}
        for offset in range(args.limit):
            kind, _, _ = select_plan(args.plan_offset + offset)
            counts[kind] += 1
        print(
            json.dumps(
                {
                    "status": "plan-valid",
                    "math_train_problems": len(problems),
                    "error_pools": {key: len(value) for key, value in error_pools.items()},
                    "plans": counts,
                }
            )
        )
        return
    if args.output.exists() and not args.resume:
        raise SystemExit(f"output exists and --no-resume was requested: {args.output}")

    generation = GenerationOptions(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=False,
    )
    callers = {
        "A1": OpenAIChatLLMCaller(
            args.api_base_a1,
            args.api_model_a1,
            generation=generation,
            timeout=args.api_timeout,
            api_key=args.api_key,
        ),
        "A2": OpenAIChatLLMCaller(
            args.api_base_a2,
            args.api_model_a2,
            generation=generation,
            timeout=args.api_timeout,
            api_key=args.api_key,
        ),
        "A3": OpenAIChatLLMCaller(
            args.api_base_a3,
            args.api_model_a3,
            generation=generation,
            timeout=args.api_timeout,
            api_key=args.api_key,
        ),
    }
    work_items = []
    for output_offset in range(args.limit):
        plan_index = args.plan_offset + output_offset
        if plan_index in existing:
            continue
        kind, route, error_type = select_plan(plan_index)
        work_items.append((plan_index, output_offset, kind, route, error_type))
    if not work_items:
        print(json.dumps({"status": "already-complete", "accepted": len(existing)}))
        return

    def process_one(
        plan_index: int,
        output_offset: int,
        kind: str,
        route: tuple[str, str, str],
        error_type: Optional[str],
    ) -> dict[str, Any]:
        last_error = "unknown generation failure"
        for attempt in range(1, args.max_trajectory_attempts + 1):
            selector_payload = (
                f"{args.seed}:{args.start}:{output_offset}:{attempt}:{error_type or 'regular'}"
            ).encode("utf-8")
            selector = int.from_bytes(hashlib.sha256(selector_payload).digest()[:8], "big")
            pool = error_pools[str(error_type)] if kind == "correction" else problems
            problem = pool[selector % len(pool)]
            trajectory = run_trajectory(
                problem,
                callers,
                route=route,
                kind=kind,
                error_type=error_type,
                retries=args.max_step_retries,
                max_similarity=args.max_verifier_similarity,
            )
            if (
                trajectory.terminated_by == "stop"
                and math_answers_equivalent(
                    trajectory.final_answer or "", problem.answer_str
                )
            ):
                return trajectory_record(
                    trajectory,
                    problem,
                    plan_index=plan_index,
                    kind=kind,
                    route=route,
                    error_type=error_type,
                    attempts=attempt,
                )
            last_error = trajectory.error or "final answer did not match MATH gold"
        raise RuntimeError(
            f"plan={plan_index} kind={kind} route={'>'.join(route)} "
            f"error_type={error_type} failed after {args.max_trajectory_attempts} "
            f"attempts: {last_error}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.output.exists() else "w"
    started = time.monotonic()
    completed = len(existing)
    attempts_seen: list[int] = []
    with args.output.open(mode, encoding="utf-8") as handle:
        def accept(record: dict[str, Any]) -> None:
            nonlocal completed
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            completed += 1
            attempts_seen.append(int(record["trajectory"]["generation_attempts"]))
            elapsed = max(time.monotonic() - started, 1e-6)
            if completed % 20 == 0 or completed == args.limit:
                print(
                    f"[{completed}/{args.limit}] accepted "
                    f"new_rate={(completed - len(existing)) / elapsed:.2f}/s "
                    f"mean_attempts={mean(attempts_seen):.2f}"
                )

        failures = execute_bounded(
            work_items,
            process_one,
            max_workers=args.max_concurrency,
            on_record=accept,
        )
    final = read_existing(args.output, args.limit, args.plan_offset)
    try:
        coverage = validate_coverage(
            final,
            limit=args.limit,
            plan_offset=args.plan_offset,
            max_missing=args.max_missing,
        )
    except ValueError as exc:
        first_failure = failures[0] if failures else "no recorded worker failure"
        raise RuntimeError(
            f"{len(failures)} rollout plans exhausted their attempts; "
            f"accepted rows were preserved for resume; first failure: {first_failure}; {exc}"
        ) from exc
    status = "complete" if not coverage["missing"] else "accepted_partial"
    print(json.dumps({"status": status, **coverage}))


if __name__ == "__main__":
    main()

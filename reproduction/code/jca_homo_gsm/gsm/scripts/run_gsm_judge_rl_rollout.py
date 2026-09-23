"""Collect GSM-HARD trajectories and score every turn with an LLM judge.

The output schema is compatible with scripts/rl_train.py: one JSON object per
agent turn with messages, response, and a signed mixed reward.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PARENT = REPO_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from jca.gsm.scripts.run_mas import (  # noqa: E402
    GSMStep,
    GSMTrajectory,
    enforce_collaboration_policy,
    is_empty_step,
    parse_action,
    render_assistant_message,
    trajectory_to_dict,
)
from jca.gsm.src.agents import AGENT_IDS, render_system_prompt  # noqa: E402
from jca.gsm.src.data import (  # noqa: E402
    GSMProblem,
    format_problem_as_prompt,
    load_gsm_hard,
)
from jca.gsm.src.grader import compute_em_f1  # noqa: E402
from jca.gsm.src.judge import (  # noqa: E402
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL,
    JUDGE_PARSE_RETRIES,
    JUDGE_REASONING_EFFORT,
    JUDGE_TEMPERATURE,
    JUDGE_TOP_P,
    TrajectoryReward,
    judge_trajectory,
)
from jca.src.inference import GenerationOptions, OpenAIChatLLMCaller  # noqa: E402


GroupKey = Tuple[str, int]


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def normalized_step_response(step: GSMStep) -> str:
    """Serialize the effective protocol action used by the environment.

    The environment can turn a premature confirm_stop into a handoff. Training
    must use that effective action rather than reinforce the rejected raw action.
    """
    return json.dumps(
        {
            "reasoning": step.reasoning,
            "tentative_answer": step.tentative_answer,
            "action": step.action,
            "handoff_target": step.handoff_target,
            "handoff_note": step.handoff_note,
            "confirmed_answer": step.confirmed_answer,
        },
        ensure_ascii=False,
    )


def run_trajectory_with_messages(
    problem: GSMProblem,
    callers: Dict[str, OpenAIChatLLMCaller],
    *,
    t_max: int,
    start_agent: str,
    min_agents_before_stop: int,
    enforce_collaboration_policy_runtime: bool,
    step_retries: int,
    sampling_seed_base: int,
) -> Tuple[GSMTrajectory, List[List[Dict[str, str]]]]:
    """Run the same inference protocol as run_mas.py and retain turn inputs."""
    trajectory = GSMTrajectory(problem_id=problem.id)
    messages: List[Dict[str, str]] = [
        {
            "role": "system",
            "content": render_system_prompt(
                start_agent,
                min_agents_before_stop=min_agents_before_stop,
            ),
        },
        {"role": "user", "content": format_problem_as_prompt(problem)},
    ]
    turn_messages: List[List[Dict[str, str]]] = []
    current_agent = start_agent
    prior_tentative = False
    prior_tentative_answer: Optional[str] = None

    try:
        for turn in range(t_max):
            turn_messages.append([dict(message) for message in messages])
            step: Optional[GSMStep] = None
            attempts = step_retries + 1
            for attempt in range(attempts):
                request_messages = [dict(message) for message in messages]
                if attempt:
                    request_messages.append({
                        "role": "user",
                        "content": (
                            "The previous response was empty or invalid. Recompute "
                            "the problem, fill reasoning and tentative_answer, and "
                            "output exactly one complete JSON object matching the "
                            "required protocol."
                        ),
                    })
                seed = (sampling_seed_base + turn * 1009 + attempt) % 2147483647
                caller = callers[current_agent]
                if isinstance(caller, OpenAIChatLLMCaller):
                    raw_output = caller.generate(request_messages, seed=seed)
                else:
                    raw_output = caller(request_messages)
                candidate = parse_action(
                    raw_output,
                    active_agent=current_agent,
                    prior_tentative=prior_tentative,
                    repair_premature_confirm=enforce_collaboration_policy_runtime,
                )
                candidate.turn = turn
                if not is_empty_step(candidate):
                    step = candidate
                    break

            if step is None:
                trajectory.terminated_by = "rejected_quality"
                trajectory.error = (
                    f"turn {turn} agent {current_agent} returned {attempts} "
                    "empty or invalid responses"
                )
                turn_messages.pop()
                return trajectory, turn_messages

            if enforce_collaboration_policy_runtime:
                step = enforce_collaboration_policy(
                    step,
                    seen_agents_before=trajectory.active_agents,
                    min_agents_before_stop=min_agents_before_stop,
                    prior_tentative_answer=prior_tentative_answer,
                )
            trajectory.steps.append(step)

            if step.tentative_answer:
                prior_tentative = True
                prior_tentative_answer = step.tentative_answer

            messages.append(
                {"role": "assistant", "content": render_assistant_message(step)}
            )

            if step.confirmed_answer is not None or (
                step.action == "confirm_stop" and step.tentative_answer
            ):
                trajectory.final_answer = (
                    step.confirmed_answer or step.tentative_answer
                )
                trajectory.terminated_by = "stop"
                return trajectory, turn_messages

            if step.action == "handoff" and step.handoff_target:
                current_agent = step.handoff_target
                messages[0] = {
                    "role": "system",
                    "content": render_system_prompt(
                        current_agent,
                        min_agents_before_stop=min_agents_before_stop,
                    ),
                }

        trajectory.terminated_by = "truncated"
        return trajectory, turn_messages
    except Exception as exc:
        trajectory.terminated_by = "exception"
        trajectory.error = str(exc)
        return trajectory, turn_messages[: len(trajectory.steps)]


def select_start_agent(sample_index: int, configured: str) -> str:
    if configured == "balanced":
        return AGENT_IDS[sample_index % len(AGENT_IDS)]
    return configured


def build_rl_records(
    problem: GSMProblem,
    trajectory: GSMTrajectory,
    turn_messages: List[List[Dict[str, str]]],
    reward: Optional[TrajectoryReward],
    *,
    rollout_idx: int,
    start_agent: str,
    em: float,
    f1: float,
    task_metric: str,
    alpha: float,
    judge_model: str,
    judge_temperature: float,
) -> List[Dict[str, Any]]:
    task_value = em if task_metric == "em" else f1
    task_reward = 2.0 * task_value - 1.0
    score_by_turn: Dict[int, Dict[str, Any]] = {}
    if reward is not None:
        for score in reward.turn_scores:
            score_by_turn[score.turn] = {
                "reasoning_score": round(score.reasoning_score, 4),
                "action_score": round(score.action_score, 4),
                "judge_score": round(score.judge_score, 4),
                "comment": score.comment,
            }

    records: List[Dict[str, Any]] = []
    for step, messages in zip(trajectory.steps, turn_messages):
        turn_scores = score_by_turn.get(step.turn, {})
        judge_failed = not bool(turn_scores)
        judge_score = float(turn_scores.get("judge_score", 0.0))
        total_reward = (
            task_reward
            if judge_failed
            else alpha * task_reward + (1.0 - alpha) * judge_score
        )
        response = normalized_step_response(step)
        records.append(
            {
                "problem_id": problem.id,
                "rollout_idx": rollout_idx,
                "turn": step.turn,
                "agent_id": step.active_agent,
                "messages": messages,
                "response": response,
                "raw_response": step.raw_output,
                "response_normalized": response.strip() != step.raw_output.strip(),
                "reward": round(total_reward, 4),
                "task_reward": round(task_reward, 4),
                "task_metric": task_metric,
                "judge_score": round(judge_score, 4),
                "turn_scores": turn_scores,
                "judge_failed": judge_failed,
                "judge_status": "failed" if judge_failed else "scored",
                "judge_model": judge_model,
                "judge_temperature": judge_temperature,
                "reward_source": "gsm_llm_judge_rwr",
                "action": step.action,
                "start_agent": start_agent,
                "terminated_by": trajectory.terminated_by,
                "em": em,
                "f1": f1,
            }
        )
    return records


def _process_one_attempt(
    problem: GSMProblem,
    problem_index: int,
    rollout_idx: int,
    callers: Dict[str, OpenAIChatLLMCaller],
    args: argparse.Namespace,
    *,
    sampling_attempt: int,
) -> List[Dict[str, Any]]:
    sample_index = problem_index * args.num_rollouts + rollout_idx
    start_agent = select_start_agent(sample_index, args.start_agent)
    sampling_seed_base = (
        (sample_index + 1) * 1_000_003 + sampling_attempt * 100_003
    ) % 2147483647
    trajectory, turn_messages = run_trajectory_with_messages(
        problem,
        callers,
        t_max=args.t_max,
        start_agent=start_agent,
        min_agents_before_stop=args.min_agents_before_stop,
        enforce_collaboration_policy_runtime=args.enforce_collaboration_policy,
        step_retries=args.step_retries,
        sampling_seed_base=sampling_seed_base,
    )
    if trajectory.terminated_by != "stop":
        raise RuntimeError(
            f"trajectory terminated_by={trajectory.terminated_by}: "
            f"{trajectory.error or 'no final answer'}"
        )
    em, f1 = compute_em_f1(trajectory.final_answer or "", problem)
    result = trajectory_to_dict(trajectory, problem)
    result["em"] = em
    result["f1"] = f1

    reward = None if args.skip_judge else judge_trajectory(
        result,
        model=args.judge_model,
        temperature=args.judge_temperature,
        top_p=args.judge_top_p,
        max_tokens=args.judge_max_tokens,
        reasoning_effort=args.judge_reasoning_effort,
        parse_retries=args.judge_parse_retries,
    )
    if reward is None and not (args.allow_judge_failure or args.skip_judge):
        raise RuntimeError(
            f"judge failed for problem={problem.id} rollout={rollout_idx}"
        )

    return build_rl_records(
        problem,
        trajectory,
        turn_messages,
        reward,
        rollout_idx=rollout_idx,
        start_agent=start_agent,
        em=em,
        f1=f1,
        task_metric=args.task_metric,
        alpha=args.alpha,
        judge_model=args.judge_model,
        judge_temperature=args.judge_temperature,
    )


def process_one(
    problem: GSMProblem,
    problem_index: int,
    rollout_idx: int,
    callers: Dict[str, OpenAIChatLLMCaller],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    last_error: Optional[Exception] = None
    for attempt in range(args.group_retries + 1):
        try:
            return _process_one_attempt(
                problem,
                problem_index,
                rollout_idx,
                callers,
                args,
                sampling_attempt=attempt,
            )
        except Exception as exc:
            last_error = exc
    assert last_error is not None
    raise RuntimeError(
        f"failed after {args.group_retries + 1} group attempts: {last_error}"
    ) from last_error


def group_key(record: Dict[str, Any]) -> GroupKey:
    return str(record.get("problem_id", "")), int(record.get("rollout_idx", 0))


def _group_is_complete(records: List[Dict[str, Any]], allow_failed: bool) -> bool:
    if not records:
        return False
    turns = sorted(int(record.get("turn", -1)) for record in records)
    if turns != list(range(len(records))):
        return False
    if allow_failed:
        return True
    return all(
        record.get("judge_status") == "scored"
        and record.get("judge_failed") is False
        for record in records
    )


def prepare_resume_output(
    output: Path,
    expected_keys: set[GroupKey],
    *,
    resume: bool,
    allow_failed: bool,
    expected_start_agent: Optional[str] = None,
) -> set[GroupKey]:
    if not output.exists():
        return set()
    if not resume:
        raise FileExistsError(f"output already exists: {output}; use --resume")

    grouped: Dict[GroupKey, List[Dict[str, Any]]] = {}
    order: List[GroupKey] = []
    found_start_agents: set[str] = set()
    with output.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
                key = group_key(record)
            except Exception:
                continue
            if key not in expected_keys:
                continue
            found_start_agents.add(str(record.get("start_agent", "")))
            if key not in grouped:
                grouped[key] = []
                order.append(key)
            grouped[key].append(record)

    if expected_start_agent is not None and found_start_agents != {expected_start_agent}:
        raise ValueError(
            "resume output start-agent mismatch: "
            f"expected only {expected_start_agent}, found {sorted(found_start_agents)}; "
            "use a new output path"
        )

    completed = {
        key
        for key, records in grouped.items()
        if _group_is_complete(records, allow_failed)
    }
    temporary = output.with_suffix(output.suffix + ".resume.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for key in order:
            if key not in completed:
                continue
            for record in sorted(grouped[key], key=lambda row: int(row["turn"])):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(output)
    return completed


def write_records(handle, records: Iterable[Dict[str, Any]]) -> None:
    for record in records:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GSM-HARD LLM-judge rollout for offline RWR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-path",
        default=str(
            REPO_ROOT / "Math/data/GSM-HARD/splits/gsmhardv2_train.jsonl"
        ),
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--num-rollouts", type=int, default=_env_int("NUM_ROLLOUTS", 8))
    parser.add_argument("--t-max", type=int, default=8)
    parser.add_argument(
        "--start-agent",
        choices=[*AGENT_IDS, "balanced"],
        default="A1",
    )
    parser.add_argument("--min-agents-before-stop", type=int, default=1)
    parser.add_argument(
        "--enforce-collaboration-policy",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("ENFORCE_COLLABORATION_POLICY", False),
        help="Rewrite model actions to enforce collaboration minima during rollout.",
    )

    parser.add_argument("--api-base-a1", default="http://127.0.0.1:8201/v1")
    parser.add_argument("--api-base-a2", default="http://127.0.0.1:8202/v1")
    parser.add_argument("--api-base-a3", default="http://127.0.0.1:8203/v1")
    parser.add_argument("--api-model-a1", default="A1")
    parser.add_argument("--api-model-a2", default="A2")
    parser.add_argument("--api-model-a3", default="A3")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--api-timeout", type=int, default=900)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument(
        "--group-retries",
        type=int,
        default=_env_int("GROUP_RETRIES", 2),
        help="Regenerate an incomplete or unjudgeable trajectory group.",
    )
    parser.add_argument(
        "--step-retries",
        type=int,
        default=_env_int("STEP_RETRIES", 4),
        help="Regenerate an empty or invalid agent turn with a repair prompt.",
    )

    parser.add_argument(
        "--judge-model",
        default=os.environ.get("JUDGE_MODEL", JUDGE_MODEL),
    )
    parser.add_argument("--alpha", type=float, default=_env_float("ALPHA", 0.6))
    parser.add_argument(
        "--task-metric",
        choices=["em", "f1"],
        default=os.environ.get("TASK_METRIC", "em"),
    )
    parser.add_argument(
        "--judge-temperature",
        type=float,
        default=_env_float("JUDGE_TEMPERATURE", JUDGE_TEMPERATURE),
    )
    parser.add_argument(
        "--judge-top-p",
        type=float,
        default=_env_float("JUDGE_TOP_P", JUDGE_TOP_P),
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=_env_int("JUDGE_MAX_TOKENS", JUDGE_MAX_TOKENS),
    )
    parser.add_argument(
        "--judge-reasoning-effort",
        default=os.environ.get("JUDGE_REASONING_EFFORT", JUDGE_REASONING_EFFORT),
    )
    parser.add_argument(
        "--judge-parse-retries",
        type=int,
        default=_env_int("JUDGE_PARSE_RETRIES", JUDGE_PARSE_RETRIES),
    )
    parser.add_argument(
        "--allow-judge-failure",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("ALLOW_JUDGE_FAILURE", False),
    )
    parser.add_argument(
        "--skip-judge",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("SKIP_JUDGE", False),
        help="Write complete raw rollout groups without calling the legacy judge.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("RESUME", True),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "rl_data/gsm/gsm_judge_rollout.jsonl",
    )
    parser.add_argument("--log-raw-chars", type=int, default=0)

    # Compatibility with the independent GSM vLLM launcher used by the shell wrapper.
    parser.add_argument("--correction-fraction", type=float, default=0.3, help=argparse.SUPPRESS)
    parser.add_argument("--sft-plan-offset", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument(
        "--cycle-data",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.start < 0:
        raise SystemExit("--start must be non-negative")
    for name in ("limit", "num_rollouts", "t_max", "max_concurrency"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if not 1 <= args.min_agents_before_stop <= len(AGENT_IDS):
        raise SystemExit("--min-agents-before-stop must be in [1, 3]")
    if not 0.0 <= args.alpha <= 1.0:
        raise SystemExit("--alpha must be in [0, 1]")
    if not 0.0 <= args.temperature <= 2.0:
        raise SystemExit("--temperature must be in [0, 2]")
    if not 0.0 <= args.judge_temperature <= 2.0:
        raise SystemExit("--judge-temperature must be in [0, 2]")
    if not 0.0 < args.top_p <= 1.0 or not 0.0 < args.judge_top_p <= 1.0:
        raise SystemExit("top-p values must be in (0, 1]")
    if args.judge_parse_retries < 0:
        raise SystemExit("--judge-parse-retries must be non-negative")
    if args.group_retries < 0:
        raise SystemExit("--group-retries must be non-negative")
    if args.step_retries < 0:
        raise SystemExit("--step-retries must be non-negative")


def main() -> None:
    args = parse_args()
    validate_args(args)
    problems = load_gsm_hard(args.data_path)
    selected = problems[args.start : args.start + args.limit]
    if len(selected) != args.limit:
        raise SystemExit(
            f"requested {args.limit} problems at start={args.start}, "
            f"but only {len(selected)} are available"
        )

    generation = GenerationOptions(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=False,
    )
    callers = {
        "A1": OpenAIChatLLMCaller(
            args.api_base_a1, args.api_model_a1,
            generation=generation, timeout=args.api_timeout, api_key=args.api_key,
            response_format={"type": "json_object"},
        ),
        "A2": OpenAIChatLLMCaller(
            args.api_base_a2, args.api_model_a2,
            generation=generation, timeout=args.api_timeout, api_key=args.api_key,
            response_format={"type": "json_object"},
        ),
        "A3": OpenAIChatLLMCaller(
            args.api_base_a3, args.api_model_a3,
            generation=generation, timeout=args.api_timeout, api_key=args.api_key,
            response_format={"type": "json_object"},
        ),
    }

    jobs = [
        (problem, args.start + offset, rollout_idx)
        for offset, problem in enumerate(selected)
        for rollout_idx in range(args.num_rollouts)
    ]
    expected_keys = {(problem.id, rollout_idx) for problem, _, rollout_idx in jobs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed_keys = prepare_resume_output(
        args.output,
        expected_keys,
        resume=args.resume,
        allow_failed=args.allow_judge_failure or args.skip_judge,
        expected_start_agent=(
            None if args.start_agent == "balanced" else args.start_agent
        ),
    )
    pending = [job for job in jobs if (job[0].id, job[2]) not in completed_keys]

    print("=" * 72)
    print("GSM-HARD LLM-judge offline-RL rollout")
    print(f"problems={len(selected)} rollouts_per_problem={args.num_rollouts}")
    print(f"groups={len(jobs)} completed={len(completed_keys)} pending={len(pending)}")
    print(
        f"rollout_temperature={args.temperature} top_p={args.top_p} "
        f"t_max={args.t_max} start_agent={args.start_agent}"
    )
    print(
        f"step_retries={args.step_retries} group_retries={args.group_retries} "
        "json_mode=on seeded_retries=on"
    )
    if args.skip_judge:
        print("judge=skipped (raw trajectories will be judged once downstream)")
    else:
        print(
            f"judge={args.judge_model} temperature={args.judge_temperature} "
            f"top_p={args.judge_top_p} alpha={args.alpha} "
            f"task_metric={args.task_metric}"
        )
    print(f"output={args.output}")
    print("=" * 72)

    started = time.monotonic()
    written_groups = 0
    written_rows = 0
    errors: List[str] = []
    reward_sum = 0.0
    reward_count = 0

    with args.output.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=args.max_concurrency) as executor:
            future_to_job = {
                executor.submit(
                    process_one,
                    problem,
                    problem_index,
                    rollout_idx,
                    callers,
                    args,
                ): (problem.id, rollout_idx)
                for problem, problem_index, rollout_idx in pending
            }
            for future in as_completed(future_to_job):
                key = future_to_job[future]
                try:
                    records = future.result()
                    if not records:
                        raise RuntimeError("trajectory produced no trainable turns")
                except Exception as exc:
                    message = f"group={key}: {exc}"
                    errors.append(message)
                    print(f"[error] {message}", file=sys.stderr, flush=True)
                    continue

                write_records(handle, records)
                written_groups += 1
                written_rows += len(records)
                for record in records:
                    reward_sum += float(record["reward"])
                    reward_count += 1

                done = written_groups + len(errors)
                if done == 1 or done % 10 == 0 or done == len(pending):
                    elapsed = time.monotonic() - started
                    rate = done / max(elapsed, 1e-9)
                    eta = (len(pending) - done) / max(rate, 1e-9)
                    reward_mean = reward_sum / max(reward_count, 1)
                    print(
                        f"progress={done}/{len(pending)} rows={written_rows} "
                        f"errors={len(errors)} reward_mean={reward_mean:.4f} "
                        f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
                        flush=True,
                    )

    if errors:
        raise SystemExit(
            f"rollout incomplete: {len(errors)} groups failed; rerun with RESUME=1"
        )

    total_groups = len(completed_keys) + written_groups
    if total_groups != len(jobs) or not math.isfinite(reward_sum):
        raise SystemExit(
            f"rollout completeness check failed: {total_groups}/{len(jobs)} groups"
        )
    print("GSM judge rollout complete")
    print(f"groups={total_groups} new_rows={written_rows} output={args.output}")


if __name__ == "__main__":
    main()

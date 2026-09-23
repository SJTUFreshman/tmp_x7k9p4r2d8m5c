"""GSM-HARD MAS inference script.

Runs multi-agent collaborative solving on GSM-HARD using vLLM OpenAI-compatible servers.
Same protocol as MuSiQue but adapted for math: no paragraphs, numeric grader.

Usage:
    python jca/gsm/scripts/run_mas.py \
        --data-path jca/MATH-data/GSM-HARD/gsmhardv2.jsonl \
        --start 0 --limit 100 \
        --api-base-a1 http://127.0.0.1:8201/v1 \
        --api-base-a2 http://127.0.0.1:8202/v1 \
        --api-base-a3 http://127.0.0.1:8203/v1 \
        --output results/gsm_mas_test.jsonl
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jca.gsm.src.data import GSMProblem, load_gsm_hard, format_problem_as_prompt, extract_number
from jca.gsm.src.grader import compute_em_f1, is_correct, numeric_values_match
from jca.gsm.src.agents import (
    AGENT_IDS,
    render_sft_rollout_system_prompt,
    render_system_prompt,
)
from jca.src.inference import GenerationOptions, OpenAIChatLLMCaller, response_attempts


# ---------------------------------------------------------------------------
# Trajectory structures
# ---------------------------------------------------------------------------

@dataclass
class GSMStep:
    turn: int
    active_agent: str
    reasoning: str
    tentative_answer: str
    action: str              # "handoff" or "confirm_stop"
    handoff_target: Optional[str]
    handoff_note: Optional[str]
    confirmed_answer: Optional[str]
    raw_output: str
    raw_outputs: List[str] = field(default_factory=list)
    visible_raw_output: str = ""


@dataclass
class GSMTrajectory:
    problem_id: str
    generation_seed: Optional[int] = None
    protocol_mode: str = "default"
    min_handoffs_before_stop: int = 0
    steps: List[GSMStep] = field(default_factory=list)
    generation_attempts: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: Optional[str] = None
    terminated_by: str = "truncated"
    error: Optional[str] = None

    @property
    def active_agents(self) -> List[str]:
        agents: List[str] = []
        for step in self.steps:
            if step.active_agent not in agents:
                agents.append(step.active_agent)
        return agents

    @property
    def n_handoffs(self) -> int:
        return sum(1 for s in self.steps if s.action == "handoff")


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def parse_action(
    raw: str,
    active_agent: str,
    prior_tentative: bool,
    *,
    repair_premature_confirm: bool = True,
) -> GSMStep:
    """Parse model output into a GSMStep."""
    text = raw.strip()

    # Extract JSON
    obj = {}
    start = text.find("{")
    if start >= 0:
        try:
            import json as _json
            decoder = _json.JSONDecoder()
            obj, _ = decoder.raw_decode(text[start:])
        except Exception:
            # fallback: try to extract fields with regex
            for field_name, pattern in [
                ("reasoning",        r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"'),
                ("tentative_answer", r'"tentative_answer"\s*:\s*"([^"]*)"'),
                ("action",           r'"action"\s*:\s*"([^"]*)"'),
                ("handoff_target",   r'"handoff_target"\s*:\s*"([^"]*)"'),
                ("handoff_note",     r'"handoff_note"\s*:\s*"([^"]*)"'),
                ("confirmed_answer", r'"confirmed_answer"\s*:\s*"([^"]*)"'),
            ]:
                m = re.search(pattern, text, re.DOTALL)
                if m:
                    obj[field_name] = m.group(1)

    reasoning  = str(obj.get("reasoning", "")).strip()
    tentative  = str(obj.get("tentative_answer", "")).strip()
    action     = str(obj.get("action", "")).strip().lower()
    h_target   = obj.get("handoff_target")
    h_note     = obj.get("handoff_note")
    confirmed  = obj.get("confirmed_answer")

    # Normalize
    if isinstance(h_target, str) and h_target.lower() in ("null", "none", ""):
        h_target = None
    if isinstance(h_note, str) and h_note.lower() in ("null", "none", ""):
        h_note = None
    if isinstance(confirmed, str) and confirmed.lower() in ("null", "none", ""):
        confirmed = None

    # If action=handoff, confirmed_answer must be null regardless of what model wrote
    if action == "handoff":
        confirmed = None

    # Validate action
    if action not in ("handoff", "confirm_stop"):
        action = "handoff" if not prior_tentative else "confirm_stop"

    # confirm_stop requires prior tentative
    if action == "confirm_stop" and not prior_tentative and repair_premature_confirm:
        action = "handoff"
        confirmed = None
        if h_target is None:
            others = [a for a in AGENT_IDS if a != active_agent]
            h_target = others[0] if others else None

    if action == "handoff" and h_target not in AGENT_IDS:
        others = [a for a in AGENT_IDS if a != active_agent]
        h_target = others[0] if others else None
    if action == "handoff" and not h_note:
        h_note = "Please independently recompute the answer from the original problem."

    confirmed_value = None
    if confirmed is not None:
        confirmed_value = str(confirmed).strip()
        if not confirmed_value:
            confirmed_value = None

    return GSMStep(
        turn=0,
        active_agent=active_agent,
        reasoning=reasoning,
        tentative_answer=tentative,
        action=action,
        handoff_target=h_target,
        handoff_note=str(h_note).strip() if h_note else None,
        confirmed_answer=confirmed_value,
        raw_output=raw,
    )


def is_empty_step(step: GSMStep) -> bool:
    return not step.reasoning and not step.tentative_answer


def step_parse_failure_reason(step: GSMStep) -> Optional[str]:
    """Classify outputs that did not provide usable protocol fields."""
    if not is_empty_step(step):
        return None
    if not step.raw_output.strip():
        return "the response was empty"
    return (
        "the response was non-empty but did not contain parseable reasoning "
        "or tentative_answer fields"
    )


def answers_match(left: Optional[str], right: Optional[str]) -> bool:
    if not left or not right:
        return left == right
    left_number = extract_number(left)
    right_number = extract_number(right)
    if left_number is not None and right_number is not None:
        return numeric_values_match(left_number, right_number)
    return left.strip() == right.strip()


def choose_handoff_target(
    active_agent: str,
    seen_agents: List[str],
    requested_target: Optional[str],
    *,
    balanced: bool = False,
) -> Optional[str]:
    others = [agent for agent in AGENT_IDS if agent != active_agent]
    unseen = [agent for agent in others if agent not in seen_agents]
    if balanced and unseen:
        return unseen[0]
    if balanced and others:
        active_index = AGENT_IDS.index(active_agent)
        return next(
            AGENT_IDS[(active_index + offset) % len(AGENT_IDS)]
            for offset in range(1, len(AGENT_IDS))
            if AGENT_IDS[(active_index + offset) % len(AGENT_IDS)] != active_agent
        )
    if requested_target in others:
        return requested_target
    if unseen:
        return unseen[0]
    return others[0] if others else None


def enforce_collaboration_policy(
    step: GSMStep,
    *,
    seen_agents_before: List[str],
    min_agents_before_stop: int,
    handoffs_before: int = 0,
    min_handoffs_before_stop: int = 0,
    prior_tentative_answer: Optional[str] = None,
    balanced_routing: bool = False,
    stop_when_ready: bool = False,
) -> GSMStep:
    seen_with_current = list(seen_agents_before)
    if step.active_agent not in seen_with_current:
        seen_with_current.append(step.active_agent)

    needs_more_agents = len(seen_with_current) < min_agents_before_stop
    needs_more_handoffs = handoffs_before < min_handoffs_before_stop
    candidate_answer = step.confirmed_answer or step.tentative_answer
    needs_correction_verification = bool(
        prior_tentative_answer
        and candidate_answer
        and not answers_match(candidate_answer, prior_tentative_answer)
    )
    ready_to_stop = bool(
        prior_tentative_answer
        and not needs_more_agents
        and not needs_more_handoffs
        and not needs_correction_verification
    )
    if stop_when_ready and ready_to_stop:
        return GSMStep(
            turn=step.turn,
            active_agent=step.active_agent,
            reasoning=step.reasoning,
            tentative_answer=step.tentative_answer,
            action="confirm_stop",
            handoff_target=None,
            handoff_note=None,
            confirmed_answer=step.tentative_answer,
            raw_output=step.raw_output,
            raw_outputs=step.raw_outputs,
            visible_raw_output=step.visible_raw_output,
        )

    if step.action == "confirm_stop" and (
        needs_more_agents or needs_more_handoffs or needs_correction_verification
    ):
        target = choose_handoff_target(
            step.active_agent,
            seen_with_current,
            None,
            balanced=balanced_routing,
        )
        requirements = []
        if needs_more_agents:
            requirements.append(f"{min_agents_before_stop} distinct agents")
        if needs_more_handoffs:
            requirements.append(f"{min_handoffs_before_stop} completed handoffs")
        if needs_correction_verification:
            requirements.append("a later check of the corrected answer")
        note = (
            "Do one more independent verification before finalizing; require "
            + " and ".join(requirements)
            + "."
        )
        return GSMStep(
            turn=step.turn,
            active_agent=step.active_agent,
            reasoning=step.reasoning,
            tentative_answer=step.confirmed_answer or step.tentative_answer,
            action="handoff",
            handoff_target=target,
            handoff_note=note,
            confirmed_answer=None,
            raw_output=step.raw_output,
            raw_outputs=step.raw_outputs,
            visible_raw_output=step.visible_raw_output,
        )

    if step.action == "handoff":
        forced_target = choose_handoff_target(
            step.active_agent,
            seen_with_current,
            step.handoff_target,
            balanced=balanced_routing,
        )
        if forced_target is not None and forced_target != step.handoff_target:
            note = step.handoff_note or "Please independently recompute and verify the arithmetic."
            return GSMStep(
                turn=step.turn,
                active_agent=step.active_agent,
                reasoning=step.reasoning,
                tentative_answer=step.tentative_answer,
                action=step.action,
                handoff_target=forced_target,
                handoff_note=note,
                confirmed_answer=step.confirmed_answer,
                raw_output=step.raw_output,
                raw_outputs=step.raw_outputs,
                visible_raw_output=step.visible_raw_output,
            )

    return step


def reasoning_similarity(left: str, right: str) -> float:
    normalized_left = " ".join((left or "").lower().split())
    normalized_right = " ".join((right or "").lower().split())
    if not normalized_left or not normalized_right:
        return 0.0
    return difflib.SequenceMatcher(None, normalized_left, normalized_right).ratio()


def has_parseable_json_object(raw: str) -> bool:
    start = raw.find("{")
    if start < 0:
        return False
    try:
        obj, _ = json.JSONDecoder().raw_decode(raw[start:])
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return isinstance(obj, dict)


def sft_step_quality_reason(
    step: GSMStep,
    *,
    gold_answer: str,
    prior_reasonings: List[str],
    max_verifier_similarity: float,
) -> Optional[str]:
    if not has_parseable_json_object(step.raw_output):
        return "output is not a parseable JSON object"
    if is_empty_step(step):
        return "empty or unparseable output"
    if not answers_match(step.tentative_answer, gold_answer):
        return "tentative_answer does not match the trusted numeric answer"
    if step.confirmed_answer and not answers_match(step.confirmed_answer, gold_answer):
        return "confirmed_answer does not match the trusted numeric answer"
    if step.confirmed_answer and not answers_match(
        step.confirmed_answer,
        step.tentative_answer,
    ):
        return "confirmed_answer conflicts with tentative_answer"
    normalized_reasoning = " ".join(step.reasoning.lower().split())
    if any(phrase in normalized_reasoning for phrase in (
        "reference answer",
        "trusted answer",
        "provided answer",
        "hidden instruction",
    )):
        return "reasoning reveals generation-only answer guidance"
    if any(
        reasoning_similarity(step.reasoning, previous) >= max_verifier_similarity
        for previous in prior_reasonings
        if previous.strip()
    ):
        return "reasoning repeats an earlier agent instead of independently recomputing"
    return None


def render_assistant_message(step: GSMStep) -> str:
    parts = [f"[{step.active_agent}]"]
    if step.reasoning:
        parts.append(step.reasoning)
    if step.tentative_answer:
        parts.append(f"tentative_answer: {step.tentative_answer}")
    if step.action == "handoff" and step.handoff_target:
        line = f"→ handoff to {step.handoff_target}"
        if step.handoff_note:
            line += f": {step.handoff_note}"
        parts.append(line)
    elif step.action == "confirm_stop":
        final_value = step.confirmed_answer or step.tentative_answer
        if final_value:
            parts.append(f"confirmed_answer: {final_value}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Trajectory runner
# ---------------------------------------------------------------------------

_REQUEST_SEED_MODULUS = 2**31 - 1


def derive_generation_request_seed(
    generation_seed: Optional[int],
    problem_id: str,
    turn: int,
    attempt: int,
    current_agent: str,
) -> Optional[int]:
    if generation_seed is None:
        return None
    payload = json.dumps(
        [generation_seed, problem_id, turn, attempt, current_agent],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % _REQUEST_SEED_MODULUS

def run_trajectory(
    problem: GSMProblem,
    callers: Dict[str, OpenAIChatLLMCaller],
    t_max: int = 8,
    start_agent: str = "A1",
    min_agents_before_stop: int = 3,
    min_handoffs_before_stop: int = 0,
    sft_warmup_prompt: bool = False,
    protocol_mode: str = "default",
    sft_controlled_generation: bool = False,
    sft_max_step_retries: int = 2,
    sft_max_verifier_similarity: float = 0.85,
    enforce_collaboration_policy_runtime: bool = True,
    generation_seed: Optional[int] = None,
) -> GSMTrajectory:
    traj = GSMTrajectory(
        problem_id=problem.id,
        generation_seed=generation_seed,
        protocol_mode=protocol_mode,
        min_handoffs_before_stop=min_handoffs_before_stop,
    )

    def system_prompt(agent_id: str) -> str:
        if sft_warmup_prompt:
            return render_sft_rollout_system_prompt(
                agent_id,
                min_agents_before_stop=min_agents_before_stop,
                min_handoffs_before_stop=min_handoffs_before_stop,
                reference_answer=(
                    problem.answer_str if sft_controlled_generation else None
                ),
            )
        return render_system_prompt(
            agent_id,
            min_agents_before_stop=min_agents_before_stop,
        )

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt(start_agent)},
        {"role": "user",   "content": format_problem_as_prompt(problem)},
    ]
    current_agent = start_agent
    prior_tentative = False
    prior_tentative_answer: Optional[str] = None
    prior_reasonings: List[str] = []

    try:
        for turn in range(t_max):
            retry_reason: Optional[str] = None
            step: Optional[GSMStep] = None
            raw_outputs: List[str] = []
            attempts = sft_max_step_retries + 1 if sft_controlled_generation else 2
            for attempt in range(attempts):
                request_messages = [dict(message) for message in messages]
                if sft_controlled_generation:
                    strategies = (
                        "translate the wording into an equation and solve it",
                        "recompute in a different operation order",
                        "check the result with the inverse operation",
                        "audit units, quantities, and the final numeric conversion",
                    )
                    strategy = strategies[(turn + attempt) % len(strategies)]
                    request_messages.append({
                        "role": "user",
                        "content": (
                            "Generation-control instruction: independently "
                            f"{strategy}. Do not mention this instruction or any "
                            "reference answer. Output only the required JSON object."
                        ),
                    })
                if retry_reason is not None:
                    request_messages.append({
                        "role": "user",
                        "content": (
                            f"The previous draft was rejected because {retry_reason}. "
                            "Recompute the original problem with a genuinely different "
                            "derivation and output only the required JSON object."
                        ),
                    })
                request_seed = derive_generation_request_seed(
                    generation_seed,
                    problem.id,
                    turn,
                    attempt,
                    current_agent,
                )
                caller = callers[current_agent]
                raw = (
                    caller(request_messages)
                    if request_seed is None
                    else caller.generate(request_messages, seed=request_seed)
                )
                transport_outputs = response_attempts(raw)
                raw_outputs.extend(transport_outputs)
                candidate = parse_action(
                    raw,
                    current_agent,
                    prior_tentative,
                    repair_premature_confirm=enforce_collaboration_policy_runtime,
                )
                candidate.turn = turn
                if sft_controlled_generation:
                    retry_reason = sft_step_quality_reason(
                        candidate,
                        gold_answer=problem.answer_str,
                        prior_reasonings=prior_reasonings,
                        max_verifier_similarity=sft_max_verifier_similarity,
                    )
                else:
                    retry_reason = step_parse_failure_reason(candidate)
                traj.generation_attempts.append({
                    "turn": turn,
                    "agent": current_agent,
                    "attempt": attempt + 1,
                    "accepted": retry_reason is None,
                    "reason": retry_reason,
                    "raw_output": str(raw),
                    "raw_outputs": transport_outputs,
                    "visible_raw_output": str(raw),
                })
                if retry_reason is None:
                    candidate.raw_outputs = list(raw_outputs)
                    candidate.visible_raw_output = raw
                    step = candidate
                    break

            if step is None:
                traj.terminated_by = "rejected_quality"
                traj.error = (
                    f"turn {turn} agent {current_agent} failed generation quality "
                    f"after {attempts} attempts: {retry_reason}"
                )
                return traj

            if enforce_collaboration_policy_runtime:
                step = enforce_collaboration_policy(
                    step,
                    seen_agents_before=traj.active_agents,
                    min_agents_before_stop=min_agents_before_stop,
                    handoffs_before=traj.n_handoffs,
                    min_handoffs_before_stop=min_handoffs_before_stop,
                    prior_tentative_answer=prior_tentative_answer,
                    balanced_routing=sft_controlled_generation,
                    stop_when_ready=sft_controlled_generation,
                )
            traj.steps.append(step)

            if step.tentative_answer:
                prior_tentative = True
                prior_tentative_answer = step.tentative_answer
            prior_reasonings.append(step.reasoning)

            messages.append({"role": "assistant", "content": render_assistant_message(step)})

            if step.confirmed_answer is not None or (
                step.action == "confirm_stop" and step.tentative_answer
            ):
                final = step.confirmed_answer or step.tentative_answer
                traj.final_answer = final
                traj.terminated_by = "stop"
                return traj

            if step.handoff_target and step.action == "handoff":
                current_agent = step.handoff_target
                messages[0] = {
                    "role": "system",
                    "content": system_prompt(current_agent),
                }

        traj.terminated_by = "truncated"
        return traj

    except Exception as exc:
        traj.terminated_by = "exception"
        traj.error = str(exc)
        return traj


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def trajectory_to_dict(traj: GSMTrajectory, problem: GSMProblem) -> Dict[str, Any]:
    trajectory = {
        "problem_id":    traj.problem_id,
        "protocol_mode": traj.protocol_mode,
        "min_handoffs_before_stop": traj.min_handoffs_before_stop,
        "steps": [
            {
                "turn":             s.turn,
                "active_agent":     s.active_agent,
                "reasoning":        s.reasoning,
                "tentative_answer": s.tentative_answer,
                "action":           s.action,
                "handoff_target":   s.handoff_target,
                "handoff_note":     s.handoff_note,
                "confirmed_answer": s.confirmed_answer,
                "raw_output":       s.raw_output,
                "raw_outputs":      s.raw_outputs,
                "visible_raw_output": s.visible_raw_output,
            }
            for s in traj.steps
        ],
        "generation_attempts": traj.generation_attempts,
        "final_answer":  traj.final_answer,
        "terminated_by": traj.terminated_by,
        "error":         traj.error,
        "active_agents": traj.active_agents,
        "n_handoffs":    traj.n_handoffs,
    }
    record = {
        "problem": {
            "id":       problem.id,
            "question": problem.question,
            "answer":   problem.answer_str,
        },
        "trajectory": trajectory,
        "final_answer":  traj.final_answer,
        "terminated_by": traj.terminated_by,
    }
    if traj.generation_seed is not None:
        trajectory["generation_seed"] = traj.generation_seed
        record["generation_seed"] = traj.generation_seed
    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-path", default="/data/wangyuheng/jca/Math/data/GSM-HARD/splits/gsmhardv2_dev.jsonl")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--t-max", type=int, default=8)
    p.add_argument(
        "--start-agent",
        choices=[*AGENT_IDS, "balanced", "random"],
        default="A1",
        help=(
            "Use 'balanced' to rotate by problem index or 'random' for a "
            "seeded deterministic draw per absolute problem index."
        ),
    )
    p.add_argument("--start-agent-seed", type=int, default=42)
    p.add_argument("--api-base-a1", default="http://127.0.0.1:8201/v1")
    p.add_argument("--api-base-a2", default="http://127.0.0.1:8202/v1")
    p.add_argument("--api-base-a3", default="http://127.0.0.1:8203/v1")
    p.add_argument("--api-model-a1", default="A1")
    p.add_argument("--api-model-a2", default="A2")
    p.add_argument("--api-model-a3", default="A3")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--api-timeout", type=int, default=120)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument(
        "--generation-seed",
        type=int,
        default=None,
        help="Optional base seed; each generation request gets a stable derived seed.",
    )
    p.add_argument("--output", type=Path, default=Path("results/gsm_mas.jsonl"))
    p.add_argument("--log-raw-chars", type=int, default=500)
    p.add_argument("--max-concurrency", type=int, default=1,
                   help="Number of problems to run in parallel.")
    p.add_argument("--min-agents-before-stop", type=int, default=3)
    p.add_argument(
        "--enforce-collaboration-policy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rewrite premature stop/correction actions to satisfy runtime collaboration minima.",
    )
    p.add_argument("--sft-warmup-prompt", action="store_true")
    p.add_argument("--sft-extended-fraction", type=float, default=0.4)
    p.add_argument("--sft-protocol-seed", type=int, default=42)
    p.add_argument(
        "--sft-controlled-generation",
        action="store_true",
        help=(
            "Use gold-anchored retries, balanced routing, and stop as soon as the "
            "configured SFT protocol is complete."
        ),
    )
    p.add_argument("--sft-max-step-retries", type=int, default=2)
    p.add_argument("--sft-max-verifier-similarity", type=float, default=0.85)
    p.add_argument("--sft-standard-min-handoffs", type=int, default=2)
    p.add_argument(
        "--sft-extended-min-handoffs",
        type=int,
        nargs="+",
        default=[3, 4],
    )
    return p.parse_args()


def format_duration(sec: float) -> str:
    if sec < 60: return f"{sec:.1f}s"
    m, s = divmod(sec, 60)
    if m < 60: return f"{int(m)}m{int(s):02d}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h{int(m):02d}m{int(s):02d}s"


def select_sft_protocol(
    problem_index: int,
    *,
    extended_fraction: float,
    seed: int,
    standard_min_handoffs: int,
    extended_min_handoffs: List[int],
) -> tuple[str, int]:
    digest = hashlib.sha256(f"{seed}:{problem_index}".encode("utf-8")).digest()
    draw = int.from_bytes(digest[:8], "big") / 2**64
    if draw >= extended_fraction:
        return "standard", standard_min_handoffs
    choice = int.from_bytes(digest[8:16], "big") % len(extended_min_handoffs)
    min_handoffs = extended_min_handoffs[choice]
    return f"extended_{min_handoffs}_handoffs", min_handoffs


def select_start_agent(
    problem_index: int,
    configured_start_agent: str,
    seed: int = 42,
) -> str:
    if configured_start_agent == "balanced":
        return AGENT_IDS[problem_index % len(AGENT_IDS)]
    if configured_start_agent == "random":
        digest = hashlib.sha256(
            f"{seed}:{problem_index}".encode("utf-8")
        ).digest()
        return AGENT_IDS[int.from_bytes(digest[:8], "big") % len(AGENT_IDS)]
    return configured_start_agent


def main() -> None:
    args = parse_args()
    if args.limit <= 0 or args.max_concurrency <= 0:
        raise SystemExit("--limit and --max-concurrency must be positive")
    if args.start < 0:
        raise SystemExit("--start must be non-negative")
    if args.start_agent_seed < 0:
        raise SystemExit("--start-agent-seed must be non-negative")
    if not 0.0 <= args.sft_extended_fraction <= 1.0:
        raise SystemExit("--sft-extended-fraction must be in [0, 1]")
    if args.sft_standard_min_handoffs < 1:
        raise SystemExit("--sft-standard-min-handoffs must be positive")
    if not args.sft_extended_min_handoffs or min(args.sft_extended_min_handoffs) < 1:
        raise SystemExit("--sft-extended-min-handoffs values must be positive")
    if args.sft_max_step_retries < 0:
        raise SystemExit("--sft-max-step-retries must be non-negative")
    if not 0.0 <= args.sft_max_verifier_similarity <= 1.0:
        raise SystemExit("--sft-max-verifier-similarity must be in [0, 1]")
    if args.sft_controlled_generation and not args.sft_warmup_prompt:
        raise SystemExit("--sft-controlled-generation requires --sft-warmup-prompt")
    if args.sft_warmup_prompt:
        largest_min_handoffs = max(
            [args.sft_standard_min_handoffs, *args.sft_extended_min_handoffs]
        )
        if args.t_max <= largest_min_handoffs:
            raise SystemExit("--t-max must exceed every SFT minimum handoff count")

    problems = load_gsm_hard(args.data_path)
    selected = problems[args.start: args.start + args.limit]
    if not selected:
        raise SystemExit("No problems selected.")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    generation = GenerationOptions(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=False,
    )
    callers = {
        "A1": OpenAIChatLLMCaller(args.api_base_a1, args.api_model_a1,
                                   generation=generation, timeout=args.api_timeout,
                                   api_key=args.api_key),
        "A2": OpenAIChatLLMCaller(args.api_base_a2, args.api_model_a2,
                                   generation=generation, timeout=args.api_timeout,
                                   api_key=args.api_key),
        "A3": OpenAIChatLLMCaller(args.api_base_a3, args.api_model_a3,
                                   generation=generation, timeout=args.api_timeout,
                                   api_key=args.api_key),
    }

    print(f"GSM-HARD MAS Inference")
    print(f"  n={len(selected)}  start={args.start}  t_max={args.t_max}")
    print(
        f"  max_concurrency={args.max_concurrency}"
        f"  min_agents_before_stop={args.min_agents_before_stop}"
    )
    if args.generation_seed is not None:
        print(f"  generation_seed={args.generation_seed}")
    if args.sft_warmup_prompt:
        print(
            "  sft_warmup_prompt=1"
            f"  controlled_generation={int(args.sft_controlled_generation)}"
            f"  extended_fraction={args.sft_extended_fraction}"
            f"  standard_min_handoffs={args.sft_standard_min_handoffs}"
            f"  extended_min_handoffs={args.sft_extended_min_handoffs}"
        )
    print(f"  output={args.output}")

    records = []
    run_start = time.monotonic()

    def _process_one(problem_index, problem):
        protocol_mode = "default"
        min_handoffs_before_stop = 0
        if args.sft_warmup_prompt:
            protocol_mode, min_handoffs_before_stop = select_sft_protocol(
                problem_index,
                extended_fraction=args.sft_extended_fraction,
                seed=args.sft_protocol_seed,
                standard_min_handoffs=args.sft_standard_min_handoffs,
                extended_min_handoffs=args.sft_extended_min_handoffs,
            )
        trajectory_start_agent = select_start_agent(
            problem_index,
            args.start_agent,
            args.start_agent_seed,
        )
        traj = run_trajectory(problem, callers,
                               t_max=args.t_max,
                               start_agent=trajectory_start_agent,
                               min_agents_before_stop=args.min_agents_before_stop,
                               min_handoffs_before_stop=min_handoffs_before_stop,
                               sft_warmup_prompt=args.sft_warmup_prompt,
                               protocol_mode=protocol_mode,
                               sft_controlled_generation=args.sft_controlled_generation,
                               sft_max_step_retries=args.sft_max_step_retries,
                               sft_max_verifier_similarity=args.sft_max_verifier_similarity,
                               enforce_collaboration_policy_runtime=args.enforce_collaboration_policy,
                               generation_seed=args.generation_seed)
        em, f1 = compute_em_f1(traj.final_answer or "", problem)
        record = trajectory_to_dict(traj, problem)
        record["start_agent"] = trajectory_start_agent
        record["em"] = em
        record["f1"] = f1
        return problem, traj, record, em, f1

    import threading
    write_lock = threading.Lock()

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with args.output.open("w", encoding="utf-8") as handle:
        if args.max_concurrency <= 1:
            # Sequential
            for idx, problem in enumerate(selected, start=1):
                t0 = time.monotonic()
                _, traj, record, em, f1 = _process_one(args.start + idx - 1, problem)
                with write_lock:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                records.append(record)
                elapsed = time.monotonic() - run_start
                avg = elapsed / idx
                eta  = avg * (len(selected) - idx)
                last = time.monotonic() - t0
                print(
                    f"\n[{idx}/{len(selected)}  {idx/len(selected)*100:.1f}%"
                    f"  elapsed={format_duration(elapsed)}"
                    f"  avg={avg:.1f}s/item  eta={format_duration(eta)}"
                    f"  last={format_duration(last)}] {problem.id}"
                )
                print(f"  question={problem.question[:80]}")
                print(f"  gold={problem.answer_str}")
                print(f"  terminated={traj.terminated_by}"
                      f"  agents={traj.active_agents}"
                      f"  handoffs={traj.n_handoffs}"
                      f"  em={em:.3f}  f1={f1:.3f}")
                print(f"  final={traj.final_answer}")
                for s in traj.steps:
                    raw_short = s.raw_output[:args.log_raw_chars]
                    print(f"    turn={s.turn} {s.active_agent} action={s.action}")
                    print(f"      tentative={s.tentative_answer}")
                    if s.handoff_target:
                        print(f"      handoff→{s.handoff_target}: {s.handoff_note}")
                    if raw_short:
                        print(f"      raw={raw_short}")
        else:
            # Concurrent
            done = 0
            next_record_index = 0
            pending_records: Dict[int, Dict[str, Any]] = {}
            with ThreadPoolExecutor(max_workers=args.max_concurrency) as pool:
                futures = {
                    pool.submit(_process_one, args.start + offset, problem): offset
                    for offset, problem in enumerate(selected)
                }
                for fut in as_completed(futures):
                    problem, traj, record, em, f1 = fut.result()
                    pending_records[futures[fut]] = record
                    done += 1
                    elapsed = time.monotonic() - run_start
                    rate = done / max(elapsed, 1e-6)
                    print(
                        f"  [{done}/{len(selected)}]"
                        f"  em={em:.3f}  f1={f1:.3f}"
                        f"  final={traj.final_answer}"
                        f"  agents={traj.active_agents}"
                        f"  rate={rate:.2f}/s"
                    )
                    while next_record_index in pending_records:
                        ordered_record = pending_records.pop(next_record_index)
                        with write_lock:
                            handle.write(json.dumps(ordered_record, ensure_ascii=False) + "\n")
                            handle.flush()
                        records.append(ordered_record)
                        next_record_index += 1

    print(f"\nDone.")
    print(f"  EM:  {mean(r['em'] for r in records):.4f}")
    print(f"  F1:  {mean(r['f1'] for r in records):.4f}")
    print(f"  output: {args.output}")


if __name__ == "__main__":
    main()

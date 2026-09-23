"""AgentVerse baseline for MuSiQue.

Faithful adaptation of Chen et al. 2023 (ICLR'24) AgentVerse for our
setup:

    Meta (Recruiter + Evaluator) = Qwen3-8B (frozen, no LoRA)
    Agent A1 = Qwen3-1.7B  (no LoRA, pure base)
    Agent A2 = Qwen3-4B    (no LoRA, pure base)
    Agent A3 = Qwen3-8B    (no LoRA, pure base)

Pipeline (per problem, up to N iterations):

    1. Recruit:  Recruiter reads the question and outputs exactly 3 roles,
                 each tagged with capacity ∈ {low, mid, high}. Roles are
                 capacity-matched to agents (low→A1, mid→A2, high→A3).

    2. Decide:   The 3 agents each answer under their assigned role. On
                 iterations > 0, they also see the previous round's
                 attempts and the evaluator's feedback.

    3. Action:   The team's answer for this iteration is the majority
                 vote across the 3 agents' answers (normalized), with
                 tie-break preferring high-capacity → mid → low.

    4. Evaluate: The Evaluator scores the round in 0–10. If score >= 8,
                 exit early. Otherwise loop back to step 2 with the
                 feedback. Recruitment (step 1) is NOT redone.

Public API:
    run_agentverse_for_problem(problem, callers, ...) -> AgentVerseRecord
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from jca.src.agents import AGENT_IDS
from jca.src.inference import response_attempts
from jca.src.data import MuSiQueProblem, format_problem_as_prompt
from jca.src.grader import normalize_answer


PROMPT_RECRUITER_PATH = Path(__file__).parent / "prompts" / "recruiter.md"
PROMPT_AGENT_PATH = Path(__file__).parent / "prompts" / "agent.md"
PROMPT_EVALUATOR_PATH = Path(__file__).parent / "prompts" / "evaluator.md"


LLMCaller = Callable[[List[Dict[str, str]]], str]


CAPACITY_TO_AGENT: Dict[str, str] = {
    "low": "A1",
    "mid": "A2",
    "high": "A3",
}


# ============================================================================
# Data structures
# ============================================================================


@dataclass
class Role:
    """One recruited role: name + capacity + description + assigned agent."""

    name: str
    capacity: str
    description: str
    agent_id: str  # A1 / A2 / A3 after capacity mapping


@dataclass
class AgentAnswer:
    """One agent's output in one iteration."""

    agent_id: str
    role_name: str
    role_capacity: str
    iteration: int
    reasoning: str
    answer: str
    raw_output: str
    parse_ok: bool
    retried: bool = False
    raw_outputs: List[str] = field(default_factory=list)


@dataclass
class EvaluationTurn:
    """One evaluator judgment for one iteration."""

    iteration: int
    score: int
    feedback: str
    raw_output: str
    parse_ok: bool
    retried: bool = False
    raw_outputs: List[str] = field(default_factory=list)


@dataclass
class AgentVerseRecord:
    problem_id: str
    max_iterations: int
    score_threshold: int
    roles: List[Role] = field(default_factory=list)
    recruit_raw_output: str = ""
    recruit_raw_outputs: List[str] = field(default_factory=list)
    recruit_parse_ok: bool = False
    recruit_retried: bool = False
    iterations: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: Optional[str] = None
    final_iteration: Optional[int] = None
    tie_break: bool = False
    exit_reason: str = "running"
    error: Optional[str] = None


# ============================================================================
# Prompt loading
# ============================================================================


def _load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _render_agent_system_prompt(role: Role) -> str:
    template = _load_prompt(PROMPT_AGENT_PATH)
    return (
        template
        .replace("{ROLE_NAME}", role.name)
        .replace("{ROLE_CAPACITY}", role.capacity)
        .replace("{ROLE_DESCRIPTION}", role.description)
    )


# ============================================================================
# JSON extraction
# ============================================================================


def _load_json_object(raw_output: str) -> Optional[Any]:
    """Robust JSON extraction. Returns dict/list on success, else None."""
    if not isinstance(raw_output, str):
        return None
    text = re.sub(
        r"<think\b[^>]*>.*?</think>",
        "",
        raw_output,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start >= 0:
            try:
                decoder = json.JSONDecoder()
                obj, _ = decoder.raw_decode(text[start:])
                return obj
            except json.JSONDecodeError:
                pass
        start = text.find("[")
        if start >= 0:
            try:
                decoder = json.JSONDecoder()
                obj, _ = decoder.raw_decode(text[start:])
                return obj
            except json.JSONDecodeError:
                pass
    return None


def _parse_agent_json(raw_output: str) -> Optional[Tuple[str, str]]:
    """Parse an agent's {"reasoning": ..., "answer": ...} response."""
    payload = _load_json_object(raw_output)
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        payload = payload[0]
    if isinstance(payload, dict):
        reasoning = payload.get("reasoning")
        answer = payload.get("answer")
        if isinstance(reasoning, str) and isinstance(answer, str):
            answer_s = answer.strip()
            if answer_s:
                return reasoning.strip(), answer_s

    reasoning_m = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_output, re.DOTALL)
    answer_m = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_output, re.DOTALL)
    if reasoning_m and answer_m:
        answer = answer_m.group(1).strip()
        if answer:
            return reasoning_m.group(1).strip(), answer
    return None


def _parse_recruiter_json(raw_output: str) -> Optional[List[Role]]:
    """Parse recruiter output into exactly 3 Roles with distinct capacities."""
    payload = _load_json_object(raw_output)
    if not isinstance(payload, dict):
        return None
    roles_raw = payload.get("roles")
    if not isinstance(roles_raw, list) or len(roles_raw) != 3:
        return None

    parsed: List[Role] = []
    seen_capacities: set = set()
    for item in roles_raw:
        if not isinstance(item, dict):
            return None
        name = item.get("name")
        capacity = item.get("capacity")
        description = item.get("description")
        if not (isinstance(name, str) and isinstance(capacity, str)
                and isinstance(description, str)):
            return None
        capacity = capacity.strip().lower()
        if capacity not in CAPACITY_TO_AGENT:
            return None
        if capacity in seen_capacities:
            return None
        seen_capacities.add(capacity)
        parsed.append(Role(
            name=name.strip(),
            capacity=capacity,
            description=description.strip(),
            agent_id=CAPACITY_TO_AGENT[capacity],
        ))

    if seen_capacities != set(CAPACITY_TO_AGENT.keys()):
        return None

    parsed.sort(key=lambda r: ["low", "mid", "high"].index(r.capacity))
    return parsed


def _parse_evaluator_json(raw_output: str) -> Optional[Tuple[int, str]]:
    """Parse evaluator's {"score": int, "feedback": str}."""
    payload = _load_json_object(raw_output)
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        payload = payload[0]
    if isinstance(payload, dict):
        score = payload.get("score")
        feedback = payload.get("feedback")
        if isinstance(score, (int, float)) and isinstance(feedback, str):
            s = max(0, min(10, int(round(score))))
            return s, feedback.strip()

    score_m = re.search(r'"score"\s*:\s*(-?\d+)', raw_output)
    feedback_m = re.search(r'"feedback"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_output, re.DOTALL)
    if score_m and feedback_m:
        s = max(0, min(10, int(score_m.group(1))))
        return s, feedback_m.group(1).strip()
    return None


def _retry_instruction(schema_hint: str) -> str:
    return (
        "Your previous output could not be parsed. Return ONLY a valid JSON "
        f"object of the form {schema_hint}. No prose, no code fences."
    )


# ============================================================================
# Stage 1: Recruit
# ============================================================================


def _stage_recruit(
    caller: LLMCaller,
    problem: MuSiQueProblem,
) -> Tuple[Optional[List[Role]], str, bool, List[str]]:
    """Run recruiter, preserving every raw response including retries."""
    messages = [
        {"role": "system", "content": _load_prompt(PROMPT_RECRUITER_PATH)},
        {"role": "user", "content": format_problem_as_prompt(problem)},
    ]
    raw_output = caller([dict(m) for m in messages])
    raw_outputs = response_attempts(raw_output)
    parsed = _parse_recruiter_json(raw_output)
    if parsed is not None:
        return parsed, raw_output, False, raw_outputs

    schema_hint = (
        '{"roles":[{"name":"...","capacity":"low","description":"..."},'
        '{"name":"...","capacity":"mid","description":"..."},'
        '{"name":"...","capacity":"high","description":"..."}]}'
    )
    retry_messages = list(messages) + [
        {"role": "user", "content": _retry_instruction(schema_hint)},
    ]
    raw_output = caller(retry_messages)
    raw_outputs.extend(response_attempts(raw_output))
    parsed = _parse_recruiter_json(raw_output)
    return parsed, raw_output, True, raw_outputs


# ============================================================================
# Stage 2: Decide (agents answer under their roles, concurrently)
# ============================================================================


def _format_prior_context(
    prior_answers: List[AgentAnswer],
    prior_evaluation: Optional[EvaluationTurn],
) -> str:
    if not prior_answers or prior_evaluation is None:
        return ""
    parts = ["# Previous Attempts"]
    for ans in prior_answers:
        parts.append(
            f"## Role: {ans.role_name} (capacity: {ans.role_capacity})\n"
            f"Reasoning: {ans.reasoning}\n"
            f"Answer: {ans.answer}"
        )
    parts.append(
        f"\n# Evaluator Feedback (previous round score={prior_evaluation.score}/10)\n"
        f"{prior_evaluation.feedback}"
    )
    return "\n\n".join(parts)


def _run_one_agent(
    caller: LLMCaller,
    role: Role,
    iteration: int,
    user_content: str,
) -> AgentAnswer:
    system_prompt = _render_agent_system_prompt(role)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    raw_output = caller([dict(m) for m in messages])
    raw_outputs = response_attempts(raw_output)
    parsed = _parse_agent_json(raw_output)
    retried = False

    if parsed is None:
        retry_messages = list(messages) + [
            {"role": "user", "content": _retry_instruction(
                '{"reasoning": "...", "answer": "..."}'
            )},
        ]
        raw_output_retry = caller(retry_messages)
        raw_outputs.extend(response_attempts(raw_output_retry))
        parsed_retry = _parse_agent_json(raw_output_retry)
        retried = True
        if parsed_retry is not None:
            reasoning, answer = parsed_retry
            return AgentAnswer(
                agent_id=role.agent_id,
                role_name=role.name,
                role_capacity=role.capacity,
                iteration=iteration,
                reasoning=reasoning,
                answer=answer,
                raw_output=raw_output_retry,
                parse_ok=True,
                retried=True,
                raw_outputs=raw_outputs,
            )
        return AgentAnswer(
            agent_id=role.agent_id,
            role_name=role.name,
            role_capacity=role.capacity,
            iteration=iteration,
            reasoning="",
            answer="",
            raw_output=raw_output_retry,
            parse_ok=False,
            retried=True,
            raw_outputs=raw_outputs,
        )

    reasoning, answer = parsed
    return AgentAnswer(
        agent_id=role.agent_id,
        role_name=role.name,
        role_capacity=role.capacity,
        iteration=iteration,
        reasoning=reasoning,
        answer=answer,
        raw_output=raw_output,
        parse_ok=True,
        retried=retried,
        raw_outputs=raw_outputs,
    )


def _stage_decide(
    callers_agents: Dict[str, LLMCaller],
    roles: List[Role],
    problem: MuSiQueProblem,
    iteration: int,
    prior_answers: List[AgentAnswer],
    prior_evaluation: Optional[EvaluationTurn],
) -> List[AgentAnswer]:
    prior_context = _format_prior_context(prior_answers, prior_evaluation)
    user_content = format_problem_as_prompt(problem)
    if prior_context:
        user_content = f"{user_content}\n\n{prior_context}"

    with ThreadPoolExecutor(max_workers=len(roles)) as executor:
        futures = {
            role.agent_id: executor.submit(
                _run_one_agent,
                callers_agents[role.agent_id],
                role,
                iteration,
                user_content,
            )
            for role in roles
        }
        results = {agent_id: fut.result() for agent_id, fut in futures.items()}

    return [results[role.agent_id] for role in roles]


# ============================================================================
# Stage 3: Action (majority vote with capacity tie-break)
# ============================================================================


def _majority_vote(
    answers: List[AgentAnswer],
) -> Tuple[Optional[str], bool]:
    """Return (winning_answer, tie_break_used).

    Bucketize by normalize_answer; tie-break by capacity high → mid → low.
    """
    non_empty = [a for a in answers if a.answer and normalize_answer(a.answer)]
    if not non_empty:
        return None, False

    buckets: Dict[str, List[AgentAnswer]] = {}
    for ans in non_empty:
        key = normalize_answer(ans.answer)
        buckets.setdefault(key, []).append(ans)

    max_count = max(len(v) for v in buckets.values())
    winners = [key for key, v in buckets.items() if len(v) == max_count]
    tie_break_used = len(winners) > 1

    priority = ["high", "mid", "low"]

    def _priority_score(bucket_key: str) -> int:
        bucket = buckets[bucket_key]
        return min(priority.index(a.role_capacity) for a in bucket)

    winning_key = min(winners, key=_priority_score)
    winning_bucket = buckets[winning_key]
    best_ans = min(winning_bucket, key=lambda a: priority.index(a.role_capacity))
    return best_ans.answer, tie_break_used


# ============================================================================
# Stage 4: Evaluate
# ============================================================================


def _stage_evaluate(
    caller: LLMCaller,
    problem: MuSiQueProblem,
    iteration: int,
    answers: List[AgentAnswer],
    team_answer: Optional[str],
) -> EvaluationTurn:
    agents_block = "\n\n".join(
        f"## Role: {a.role_name} (capacity: {a.role_capacity})\n"
        f"Reasoning: {a.reasoning}\n"
        f"Answer: {a.answer}"
        for a in answers
    )
    user_content = (
        f"{format_problem_as_prompt(problem)}\n\n"
        f"# This Round's Agent Outputs\n{agents_block}\n\n"
        f"# Team Majority Answer (this round)\n{team_answer or '<empty>'}\n\n"
        f"Score this round and give feedback."
    )
    messages = [
        {"role": "system", "content": _load_prompt(PROMPT_EVALUATOR_PATH)},
        {"role": "user", "content": user_content},
    ]
    raw_output = caller([dict(m) for m in messages])
    raw_outputs = response_attempts(raw_output)
    parsed = _parse_evaluator_json(raw_output)

    if parsed is None:
        retry_messages = list(messages) + [
            {"role": "user", "content": _retry_instruction(
                '{"score": <int 0-10>, "feedback": "..."}'
            )},
        ]
        raw_output_retry = caller(retry_messages)
        raw_outputs.extend(response_attempts(raw_output_retry))
        parsed_retry = _parse_evaluator_json(raw_output_retry)
        if parsed_retry is None:
            return EvaluationTurn(
                iteration=iteration,
                score=0,
                feedback="",
                raw_output=raw_output_retry,
                parse_ok=False,
                retried=True,
                raw_outputs=raw_outputs,
            )
        score, feedback = parsed_retry
        return EvaluationTurn(
            iteration=iteration,
            score=score,
            feedback=feedback,
            raw_output=raw_output_retry,
            parse_ok=True,
            retried=True,
            raw_outputs=raw_outputs,
        )

    score, feedback = parsed
    return EvaluationTurn(
        iteration=iteration,
        score=score,
        feedback=feedback,
        raw_output=raw_output,
        parse_ok=True,
        retried=False,
        raw_outputs=raw_outputs,
    )


# ============================================================================
# Full problem run
# ============================================================================


def run_agentverse_for_problem(
    problem: MuSiQueProblem,
    callers_agents: Dict[str, LLMCaller],
    caller_meta: LLMCaller,
    *,
    max_iterations: int = 3,
    score_threshold: int = 8,
) -> AgentVerseRecord:
    """Run the AgentVerse pipeline for one MuSiQue problem.

    Args:
        callers_agents: {agent_id -> caller} for A1/A2/A3.
        caller_meta: caller for the 8B model that plays both recruiter
                     and evaluator.
        max_iterations: hard cap on Decide/Evaluate loops.
        score_threshold: stop early if evaluator score >= this.
    """
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    for agent_id in AGENT_IDS:
        if agent_id not in callers_agents:
            raise ValueError(f"Missing agent caller for {agent_id}")

    record = AgentVerseRecord(
        problem_id=problem.id,
        max_iterations=max_iterations,
        score_threshold=score_threshold,
    )

    try:
        roles, recruit_raw, recruit_retried, recruit_raw_outputs = _stage_recruit(
            caller_meta, problem)
        record.recruit_raw_output = recruit_raw
        record.recruit_raw_outputs = recruit_raw_outputs
        record.recruit_retried = recruit_retried
        if roles is None:
            record.recruit_parse_ok = False
            record.exit_reason = "recruit_failed"
            record.error = "Recruiter output could not be parsed after retry."
            return record
        record.recruit_parse_ok = True
        record.roles = roles

        prior_answers: List[AgentAnswer] = []
        prior_evaluation: Optional[EvaluationTurn] = None

        for iteration in range(max_iterations):
            answers = _stage_decide(
                callers_agents, roles, problem, iteration,
                prior_answers, prior_evaluation,
            )
            team_answer, tie_break = _majority_vote(answers)
            evaluation = _stage_evaluate(
                caller_meta, problem, iteration, answers, team_answer,
            )
            record.iterations.append({
                "iteration": iteration,
                "answers": [asdict(a) for a in answers],
                "team_answer": team_answer,
                "tie_break": tie_break,
                "evaluation": asdict(evaluation),
            })

            record.final_answer = (
                f"\\boxed{{{team_answer}}}" if team_answer else None
            )
            record.final_iteration = iteration
            record.tie_break = tie_break

            if evaluation.parse_ok and evaluation.score >= score_threshold:
                record.exit_reason = "score_threshold"
                return record

            prior_answers = answers
            prior_evaluation = evaluation

        record.exit_reason = "max_iterations"
        return record

    except Exception as exc:
        record.error = f"{type(exc).__name__}: {exc}"
        record.exit_reason = "exception"
        return record


# ============================================================================
# Serialization
# ============================================================================


def agentverse_record_to_dict(record: AgentVerseRecord) -> Dict[str, Any]:
    return {
        "problem_id": record.problem_id,
        "max_iterations": record.max_iterations,
        "score_threshold": record.score_threshold,
        "roles": [asdict(r) for r in record.roles],
        "recruit_raw_output": record.recruit_raw_output,
        "recruit_raw_outputs": record.recruit_raw_outputs,
        "recruit_parse_ok": record.recruit_parse_ok,
        "recruit_retried": record.recruit_retried,
        "iterations": record.iterations,
        "final_answer": record.final_answer,
        "final_iteration": record.final_iteration,
        "tie_break": record.tie_break,
        "exit_reason": record.exit_reason,
        "error": record.error,
    }

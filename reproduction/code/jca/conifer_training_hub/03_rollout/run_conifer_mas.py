#!/usr/bin/env python3
"""Run dynamic JCA trajectories on normalized Conifer examples.

The default backend talks to three OpenAI-compatible vLLM servers.  ``--mock``
is intentionally included for CPU-only protocol/data smoke tests and never
pretends to be a quality result.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import random
import re
import sys
from threading import Condition, Lock
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
for path in (PROJECT_ROOT, PROJECT_ROOT.parent, ROOT / "02_protocol", ROOT / "04_judge"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from conifer_protocol import (  # noqa: E402
    AGENT_IDS,
    ConiferStep,
    ConiferTrajectory,
    enforce_policy,
    format_problem_as_prompt,
    render_assistant_message,
    render_system_prompt,
    response_schema,
    step_to_dict,
    trajectory_to_dict,
    parse_step,
)
from conifer_scoring import check_constraints  # noqa: E402
from jca.src.inference import OpenAIChatLLMCaller, response_attempts  # noqa: E402


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _atomic_append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _termination(item: dict[str, Any]) -> str:
    trajectory = item.get("trajectory")
    if not isinstance(trajectory, dict):
        return ""
    return str(trajectory.get("terminated_by") or "")


def _rewrite_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.resume.tmp")
    with temporary.open("w", encoding="utf-8", buffering=1024 * 1024) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class MockCaller:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        seed: int | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        schema_name = str((response_format or {}).get("json_schema", {}).get("name") or "")
        if schema_name == "agentverse_recruitment":
            return json.dumps({
                "roles": [
                    {"name": "evidence finder", "capacity": "low", "description": "Enumerate explicit requirements."},
                    {"name": "verifier", "capacity": "mid", "description": "Find omissions and contradictions."},
                    {"name": "synthesizer", "capacity": "high", "description": "Produce the complete final answer."},
                ],
            })
        if schema_name == "agentverse_evaluation":
            return json.dumps({"score": 9, "feedback": "The smoke-test answer covers the explicit constraints."})
        prior = sum(1 for message in messages if message.get("role") == "assistant")
        # A deterministic, valid envelope makes parser/policy tests possible
        # without a model.  The answer is deliberately marked as a smoke value.
        answer = "SMOKE TEST ANSWER: satisfy the explicit constraints in the user instruction."
        if prior == 0:
            action, target, note, confirmed = "handoff", next(a for a in AGENT_IDS if a != self.agent_id), "Check every explicit content and format constraint.", None
        else:
            action, target, note, confirmed = "confirm_stop", None, None, answer
        return json.dumps({
            "reasoning": f"{self.agent_id} inspected the instruction and enumerated its constraints.",
            "tentative_answer": answer,
            "action": action,
            "handoff_target": target,
            "handoff_note": note,
            "confirmed_answer": confirmed,
        }, ensure_ascii=False)

    def __call__(self, messages: list[dict[str, str]]) -> str:
        return self.generate(messages)


class EndpointPoolCaller:
    def __init__(self, callers: list[Any]) -> None:
        if not callers:
            raise ValueError("endpoint pool must contain at least one caller")
        self.callers = callers
        self._active = [0] * len(callers)
        self._failures = [0] * len(callers)
        self._cooldown_until = [0.0] * len(callers)
        self._next_index = 0
        self._condition = Condition(Lock())

    def __call__(self, messages: list[dict[str, str]]) -> str:
        return self.generate(messages)

    def _acquire(self, excluded: set[int] | None = None) -> int:
        excluded = excluded or set()
        with self._condition:
            now = time.monotonic()
            candidates = [
                index for index in range(len(self._active))
                if index not in excluded and self._cooldown_until[index] <= now
            ]
            if not candidates:
                candidates = [index for index in range(len(self._active)) if index not in excluded]
            if not candidates:
                candidates = list(range(len(self._active)))
            minimum = min(self._active[index] for index in candidates)
            for offset in range(len(self._active)):
                index = (self._next_index + offset) % len(self._active)
                if index in candidates and self._active[index] == minimum:
                    self._active[index] += 1
                    self._next_index = (index + 1) % len(self._active)
                    return index
        raise RuntimeError("endpoint pool selection failed")

    def _release(self, index: int) -> None:
        with self._condition:
            if not 0 <= index < len(self._active) or self._active[index] <= 0:
                raise RuntimeError(f"invalid endpoint release: {index}")
            self._active[index] -= 1
            self._condition.notify()

    def _mark_success(self, index: int) -> None:
        with self._condition:
            self._failures[index] = 0
            self._cooldown_until[index] = 0.0

    def _mark_failure(self, index: int) -> None:
        with self._condition:
            self._failures[index] += 1
            self._cooldown_until[index] = time.monotonic() + min(10.0, 0.25 * (2 ** min(self._failures[index] - 1, 5)))

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        seed: int | None = None,
        temperature: float | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        try:
            retry_budget = max(0, int(os.environ.get("JCA_ENDPOINT_RETRIES", "2")))
        except ValueError:
            retry_budget = 2
        attempts = min(len(self.callers), retry_budget + 1)
        last_error: RuntimeError | None = None
        tried: set[int] = set()
        for _ in range(attempts):
            index = self._acquire(tried)
            tried.add(index)
            try:
                caller = self.callers[index]
                if hasattr(caller, "generate"):
                    result = str(caller.generate(
                        messages,
                        seed=seed,
                        temperature=temperature,
                        response_format=response_format,
                    ))
                else:
                    result = str(caller(messages))
                self._mark_success(index)
                return result
            except RuntimeError as exc:
                last_error = exc
                self._mark_failure(index)
            finally:
                self._release(index)
        if last_error is not None:
            raise last_error
        raise RuntimeError("endpoint pool call failed")


def _parse_api_bases(value: str | None) -> tuple[str, ...]:
    bases = tuple(part.strip().rstrip("/") for part in (value or "").split(",") if part.strip())
    if not bases:
        raise ValueError("API base must contain at least one endpoint")
    if len(set(bases)) != len(bases):
        raise ValueError("API base pool contains duplicate endpoints")
    return bases


def _build_callers(args: argparse.Namespace) -> dict[str, Any]:
    if args.mock:
        return {agent: MockCaller(agent) for agent in AGENT_IDS}
    try:
        from jca.src.inference import GenerationOptions, OpenAIChatLLMCaller
    except ImportError as exc:
        raise SystemExit("OpenAI backend requires the project Python path and src/inference.py") from exc
    bases = {"A1": args.api_base_a1, "A2": args.api_base_a2, "A3": args.api_base_a3}
    models = {"A1": args.model_a1, "A2": args.model_a2, "A3": args.model_a3}
    missing = [agent for agent, value in bases.items() if not value]
    if missing:
        raise SystemExit("Missing API base(s): " + ", ".join(missing))
    generation = GenerationOptions(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=args.enable_thinking,
    )
    callers: dict[str, Any] = {}
    endpoint_pools: dict[tuple[tuple[str, ...], str, str], EndpointPoolCaller] = {}
    for agent in AGENT_IDS:
        response_format = None
        if args.json_transport == "json_schema":
            response_format = response_schema(agent)
        elif args.json_transport == "json_object":
            response_format = {"type": "json_object"}
        api_bases = _parse_api_bases(bases[agent])
        endpoint_key = (api_bases, models[agent], str(response_format))
        pool = endpoint_pools.get(endpoint_key)
        if pool is None:
            pool = EndpointPoolCaller([
                OpenAIChatLLMCaller(
                    api_base, models[agent], generation=generation,
                    timeout=args.api_timeout, api_key=args.api_key,
                    max_model_len=args.max_model_len, response_format=response_format,
                    length_retries=args.length_retries,
                )
                for api_base in api_bases
            ])
            endpoint_pools[endpoint_key] = pool
        callers[agent] = pool
    return callers


def _select_start(row_index: int, rollout_index: int, configured: str, seed: int) -> str:
    if configured == "balanced":
        return AGENT_IDS[(row_index + rollout_index + seed) % len(AGENT_IDS)]
    if configured not in AGENT_IDS:
        raise ValueError(f"invalid start agent: {configured}")
    return configured


def run_trajectory(
    row: dict[str, Any],
    caller_by_agent: dict[str, Any],
    *,
    start_agent: str,
    t_max: int,
    min_agents_before_stop: int,
    min_handoffs_before_stop: int,
    enforce_runtime_policy: bool,
    step_retries: int,
    seed: int,
    include_reference: bool = False,
    routing: str = "dynamic",
    force_handoff_until_final: bool = False,
    context_turns: int = 2,
    include_source_assistant_context: bool = False,
) -> tuple[ConiferTrajectory, list[list[dict[str, str]]]]:
    trajectory = ConiferTrajectory(problem_id=str(row["problem_id"]), start_agent=start_agent)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": render_system_prompt(start_agent, min_agents_before_stop=min_agents_before_stop, min_handoffs_before_stop=min_handoffs_before_stop)},
        {"role": "user", "content": format_problem_as_prompt(
            row,
            include_reference=include_reference,
            context_turns=context_turns,
            include_source_assistant_context=include_source_assistant_context,
        )},
    ]
    turn_messages: list[list[dict[str, str]]] = []
    current_agent = start_agent
    prior_tentative = False
    prior_answer: str | None = None
    seen_agents: list[str] = []
    handoffs = 0

    try:
        for turn in range(t_max):
            turn_messages.append([dict(message) for message in messages])
            step: ConiferStep | None = None
            raw = ""
            for attempt in range(step_retries + 1):
                request = [dict(message) for message in messages]
                if attempt:
                    request.append({"role": "user", "content": "Your previous output was invalid. Return exactly one complete JSON protocol object with a non-empty reasoning and tentative_answer."})
                request_seed = (seed + turn * 1009 + attempt) % 2_147_483_647
                caller = caller_by_agent[current_agent]
                if hasattr(caller, "generate"):
                    raw = str(caller.generate(request, seed=request_seed))
                else:
                    raw = str(caller(request))
                candidate = parse_step(raw, active_agent=current_agent, turn=turn, prior_tentative=prior_tentative, repair=True)
                if candidate is not None:
                    step = candidate
                    break
            if step is None:
                trajectory.terminated_by = "invalid_response"
                trajectory.error = f"agent {current_agent} returned no parseable response after {step_retries + 1} attempts"
                turn_messages.pop()
                return trajectory, turn_messages
            if enforce_runtime_policy:
                step = enforce_policy(step, seen_agents_before=seen_agents, handoffs_before=handoffs,
                                      min_agents_before_stop=min_agents_before_stop,
                                      min_handoffs_before_stop=min_handoffs_before_stop,
                                      prior_answer=prior_answer)
            if routing == "cycle" and step.action == "handoff":
                next_agent = AGENT_IDS[(AGENT_IDS.index(current_agent) + 1) % len(AGENT_IDS)]
                step.handoff_target = next_agent
            if force_handoff_until_final and turn < t_max - 1:
                next_agent = AGENT_IDS[(AGENT_IDS.index(current_agent) + 1) % len(AGENT_IDS)]
                step.action = "handoff"
                step.handoff_target = next_agent
                step.handoff_note = step.handoff_note or "Fixed-route ablation: verify the draft before the final turn."
                step.confirmed_answer = None
            elif force_handoff_until_final and turn == t_max - 1:
                step.action = "confirm_stop"
                step.handoff_target = None
                step.handoff_note = None
                step.confirmed_answer = step.confirmed_answer or step.tentative_answer
            step.hard_checks = check_constraints(row, step.confirmed_answer or step.tentative_answer)
            step.hard_score = float(step.hard_checks.get("hard_score", 0.0))
            trajectory.steps.append(step)
            if current_agent not in seen_agents:
                seen_agents.append(current_agent)
            if step.tentative_answer:
                prior_tentative = True
                prior_answer = step.tentative_answer
            messages.append({"role": "assistant", "content": render_assistant_message(step)})
            if step.action == "confirm_stop":
                trajectory.final_answer = step.confirmed_answer or step.tentative_answer
                trajectory.terminated_by = "stop"
                return trajectory, turn_messages
            if step.action == "handoff" and step.handoff_target:
                handoffs += 1
                current_agent = step.handoff_target
                messages[0] = {"role": "system", "content": render_system_prompt(current_agent, min_agents_before_stop=min_agents_before_stop, min_handoffs_before_stop=min_handoffs_before_stop)}
        trajectory.terminated_by = "truncated"
        if trajectory.steps:
            trajectory.final_answer = trajectory.steps[-1].tentative_answer
        return trajectory, turn_messages
    except Exception as exc:  # preserve partial trajectory for resume/audit
        trajectory.terminated_by = "exception"
        trajectory.error = f"{type(exc).__name__}: {exc}"
        return trajectory, turn_messages[: len(trajectory.steps)]


def _baseline_call_step(
    caller: Any,
    messages: list[dict[str, str]],
    *,
    active_agent: str,
    turn: int,
    seed: int,
    step_retries: int,
    request_log: list[list[dict[str, str]]],
    temperature: float,
) -> ConiferStep:
    """Call one baseline node and always return an auditable protocol step.

    Every attempt's response is kept in ``raw_outputs``: a retry burns real
    tokens, so dropping the failed attempts would make the cost accounting
    unrecoverable.  ``raw_output`` still holds the last attempt, unchanged.
    """
    last_raw = ""
    raw_outputs: list[str] = []
    for attempt in range(step_retries + 1):
        request = [dict(message) for message in messages]
        if attempt:
            request.append({
                "role": "user",
                "content": (
                    "Return exactly one complete JSON protocol object with non-empty "
                    "reasoning and tentative_answer. Do not add markdown."
                ),
            })
        request_log.append(request)
        request_seed = (seed + turn * 1009 + attempt) % 2_147_483_647
        try:
            if hasattr(caller, "generate"):
                last_raw = str(caller.generate(
                    request,
                    seed=request_seed,
                    temperature=temperature,
                ))
            else:
                last_raw = str(caller(request))
        except Exception as exc:
            last_raw = ""
            raw_outputs.append(f"{type(exc).__name__}: {exc}")
            if attempt >= step_retries:
                failed = ConiferStep(
                    turn=turn,
                    active_agent=active_agent,
                    reasoning="baseline call failed",
                    tentative_answer="",
                    action="confirm_stop",
                    handoff_target=None,
                    handoff_note=None,
                    confirmed_answer=None,
                    raw_output=f"{type(exc).__name__}: {exc}",
                )
                failed.raw_outputs = raw_outputs
                return failed
            continue
        raw_outputs.extend(response_attempts(last_raw))
        parsed = parse_step(
            last_raw,
            active_agent=active_agent,
            turn=turn,
            prior_tentative=True,
            repair=False,
        )
        if parsed is not None:
            parsed.raw_outputs = raw_outputs
            return parsed
    # A malformed model response should not make an otherwise resumable
    # baseline disappear.  Keep the raw text as a draft and let deterministic
    # scoring mark it appropriately.
    fallback = last_raw.strip()
    unparsed = ConiferStep(
        turn=turn,
        active_agent=active_agent,
        reasoning="The baseline response was not valid protocol JSON.",
        tentative_answer=fallback,
        action="confirm_stop",
        handoff_target=None,
        handoff_note=None,
        confirmed_answer=fallback or None,
        raw_output=last_raw,
    )
    unparsed.raw_outputs = raw_outputs
    return unparsed


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    text = re.sub(
        r"<think\b[^>]*>.*?</think>",
        "",
        str(raw or ""),
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            return None
        try:
            payload, _ = json.JSONDecoder().raw_decode(text[start:])
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            return None


def _baseline_call_json(
    caller: Any,
    messages: list[dict[str, str]],
    *,
    turn: int,
    seed: int,
    step_retries: int,
    request_log: list[list[dict[str, str]]],
    temperature: float,
    response_format: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    last_raw = ""
    for attempt in range(step_retries + 1):
        request = [dict(message) for message in messages]
        if attempt:
            request.append({
                "role": "user",
                "content": "Your previous output was invalid. Return exactly one JSON object matching the required schema.",
            })
        request_log.append(request)
        request_seed = (seed + turn * 1009 + attempt) % 2_147_483_647
        try:
            if hasattr(caller, "generate"):
                last_raw = str(caller.generate(
                    request,
                    seed=request_seed,
                    temperature=temperature,
                    response_format=response_format,
                ))
            else:
                last_raw = str(caller(request))
        except Exception:
            last_raw = ""
            continue
        payload = _extract_json_object(last_raw)
        if payload is not None:
            return payload, last_raw
    return None, last_raw


def _agentverse_recruitment_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "agentverse_recruitment",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "roles": {
                        "type": "array",
                        "minItems": 3,
                        "maxItems": 3,
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "capacity": {"type": "string", "enum": ["low", "mid", "high"]},
                                "description": {"type": "string"},
                            },
                            "required": ["name", "capacity", "description"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["roles"],
                "additionalProperties": False,
            },
        },
    }


def _agentverse_evaluation_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "agentverse_evaluation",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "score": {"type": "integer", "enum": list(range(11))},
                    "feedback": {"type": "string"},
                },
                "required": ["score", "feedback"],
                "additionalProperties": False,
            },
        },
    }


def _baseline_messages(
    row: dict[str, Any],
    *,
    strategy: str,
    agent_id: str,
    instruction: str,
    context: str = "",
) -> list[dict[str, str]]:
    system = render_system_prompt(
        agent_id,
        min_agents_before_stop=1,
        min_handoffs_before_stop=0,
    )
    system += (
        "\n\nYou are running the " + strategy +
        " inference baseline. Follow the baseline node's role exactly; "
        "the final answer will be selected by the baseline controller.\n" + instruction
    )
    user = format_problem_as_prompt(row, context_turns=2)
    if context:
        user += "\n\n" + context
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _baseline_vote(
    answers_by_agent: dict[str, str],
    *,
    priority: tuple[str, ...] = ("A3", "A2", "A1"),
) -> tuple[str, bool]:
    """Majority vote with deterministic capacity tie-break for open answers."""
    buckets: dict[str, list[tuple[str, str]]] = {}
    for agent_id, answer in answers_by_agent.items():
        answer = str(answer or "").strip()
        if not answer:
            continue
        key = re.sub(r"[^\w]+", " ", answer.casefold(), flags=re.UNICODE).strip()
        if key:
            buckets.setdefault(key, []).append((agent_id, answer))
    if not buckets:
        return "", False
    maximum = max(len(items) for items in buckets.values())
    winners = [key for key, items in buckets.items() if len(items) == maximum]
    tie_break = len(winners) > 1
    if tie_break:
        for preferred in priority:
            for key in winners:
                for agent_id, answer in buckets[key]:
                    if agent_id == preferred:
                        return answer, True
    winning_key = winners[0]
    for preferred in priority:
        for agent_id, answer in buckets[winning_key]:
            if agent_id == preferred:
                return answer, tie_break
    return buckets[winning_key][0][1], tie_break


def _append_baseline_step(
    trajectory: ConiferTrajectory,
    row: dict[str, Any],
    step: ConiferStep,
    *,
    action: str,
    handoff_target: str | None = None,
    confirmed_answer: str | None = None,
) -> None:
    step.action = action
    step.handoff_target = handoff_target if action == "handoff" else None
    step.handoff_note = (
        step.handoff_note or "Pass the draft to the next baseline node for an independent check."
    ) if action == "handoff" else None
    step.confirmed_answer = confirmed_answer if action == "confirm_stop" else None
    answer = step.confirmed_answer or step.tentative_answer
    step.hard_checks = check_constraints(row, answer)
    step.hard_score = float(step.hard_checks.get("hard_score", 0.0))
    trajectory.steps.append(step)


def _run_baseline_mad(
    row: dict[str, Any],
    caller_by_agent: dict[str, Any],
    trajectory: ConiferTrajectory,
    *,
    start_agent: str,
    rounds: int,
    step_retries: int,
    seed: int,
    request_log: list[list[dict[str, str]]],
    temperature_round0: float,
    temperature_debate: float,
) -> None:
    previous: dict[str, ConiferStep] = {}
    agent_order = AGENT_IDS
    rounds = max(1, rounds)
    for round_index in range(rounds):
        contexts: dict[str, str] = {}
        for agent_id in agent_order:
            if round_index == 0:
                contexts[agent_id] = (
                    "# MAD round 0\nProduce an independent draft; do not assume any peer answer."
                )
            else:
                peers = [previous[peer] for peer in agent_order if peer != agent_id]
                peer_blocks = []
                for peer_index, peer in enumerate(peers, start=1):
                    peer_blocks.append(
                        f"## Anonymous peer {peer_index}\n"
                        f"Reasoning: {peer.reasoning}\nDraft: {peer.tentative_answer}"
                    )
                contexts[agent_id] = (
                    f"# MAD debate round {round_index}\n"
                    "Recheck your own draft against these anonymous peer drafts, then improve it.\n"
                    + "\n\n".join(peer_blocks)
                )
        current: dict[str, ConiferStep] = {}
        with ThreadPoolExecutor(max_workers=len(AGENT_IDS)) as pool:
            futures = {
                pool.submit(
                    _baseline_call_step,
                    caller_by_agent[agent_id],
                    _baseline_messages(
                        row,
                        strategy="MAD",
                        agent_id=agent_id,
                        instruction="Independent debate and evidence checking.",
                        context=contexts[agent_id],
                    ),
                    active_agent=agent_id,
                    turn=round_index * len(AGENT_IDS) + agent_index,
                    seed=seed + round_index * 10007 + agent_index * 97,
                    step_retries=step_retries,
                    request_log=request_log,
                    temperature=(temperature_round0 if round_index == 0 else temperature_debate),
                ): agent_id
                for agent_index, agent_id in enumerate(agent_order)
            }
            for future, agent_id in ((future, futures[future]) for future in futures):
                current[agent_id] = future.result()
        for agent_index, agent_id in enumerate(agent_order):
            target = agent_order[(agent_order.index(agent_id) + 1) % len(agent_order)]
            _append_baseline_step(
                trajectory,
                row,
                current[agent_id],
                action="handoff",
                handoff_target=target,
            )
        previous = current
    final_answer, tie_break = _baseline_vote(
        {agent_id: step.tentative_answer for agent_id, step in previous.items()}
    )
    final_agent = agent_order[-1]
    final_step = ConiferStep(
        turn=len(trajectory.steps),
        active_agent=final_agent,
        reasoning="MAD controller selected the final draft by majority vote with capacity tie-break.",
        tentative_answer=final_answer,
        action="confirm_stop",
        handoff_target=None,
        handoff_note=None,
        confirmed_answer=final_answer,
        raw_output="",
    )
    _append_baseline_step(trajectory, row, final_step, action="confirm_stop", confirmed_answer=final_answer)
    trajectory.final_answer = final_answer
    trajectory.terminated_by = "stop"
    trajectory.quality = {
        "baseline": "mad", "rounds": rounds, "tie_break": tie_break,
        "start_agent": start_agent,
    }


def _run_baseline_agentverse(
    row: dict[str, Any],
    caller_by_agent: dict[str, Any],
    trajectory: ConiferTrajectory,
    *,
    start_agent: str,
    iterations: int,
    step_retries: int,
    seed: int,
    request_log: list[list[dict[str, str]]],
    agent_temperature: float,
    meta_temperature: float,
    score_threshold: int,
) -> None:
    iterations = max(1, iterations)
    default_role_descriptions = {
        "A1": "low-capacity evidence finder: enumerate explicit requirements and concrete facts",
        "A2": "mid-capacity verifier: identify omissions, contradictions, and unsafe claims",
        "A3": "high-capacity synthesizer: produce a complete, audience-appropriate answer",
    }
    route = AGENT_IDS
    recruit_payload, recruit_raw = _baseline_call_json(
        caller_by_agent["A3"],
        [
            {
                "role": "system",
                "content": (
                    "You are the AgentVerse Recruiter. Recruit exactly three complementary roles, "
                    "one for each capacity low, mid, and high. Return only the required JSON object."
                ),
            },
            {"role": "user", "content": format_problem_as_prompt(row, context_turns=2)},
        ],
        turn=0,
        seed=seed,
        step_retries=step_retries,
        request_log=request_log,
        temperature=meta_temperature,
        response_format=_agentverse_recruitment_schema(),
    )
    role_descriptions = dict(default_role_descriptions)
    capacities = {"low": "A1", "mid": "A2", "high": "A3"}
    recruited_roles = (recruit_payload or {}).get("roles")
    if isinstance(recruited_roles, list) and len(recruited_roles) == 3:
        parsed_descriptions: dict[str, str] = {}
        for recruited_role in recruited_roles:
            if not isinstance(recruited_role, dict):
                continue
            capacity = str(recruited_role.get("capacity") or "").strip().lower()
            agent_id = capacities.get(capacity)
            name = str(recruited_role.get("name") or "").strip()
            description = str(recruited_role.get("description") or "").strip()
            if agent_id and name and description:
                parsed_descriptions[agent_id] = f"{capacity}-capacity {name}: {description}"
        if set(parsed_descriptions) == set(AGENT_IDS):
            role_descriptions = parsed_descriptions
    recruitment_text = json.dumps(recruit_payload, ensure_ascii=False) if recruit_payload else recruit_raw
    previous_answers: dict[str, ConiferStep] = {}
    feedback = recruitment_text
    turn_index = 1
    evaluator_scores: list[int] = []
    exit_reason = "max_iterations"
    for iteration in range(iterations):
        contexts = {}
        for agent_id in route:
            prior = ""
            if previous_answers:
                prior = "\n\n".join(
                    f"## {peer}\n{step.tentative_answer}"
                    for peer, step in previous_answers.items()
                )
            contexts[agent_id] = (
                f"# AgentVerse iteration {iteration}\nRole: {role_descriptions[agent_id]}\n"
                f"Recruiter plan / previous evaluator feedback:\n{feedback}\n"
                f"Previous team attempts:\n{prior or '[none]'}\n"
                "Improve the answer under your assigned role."
            )
        current: dict[str, ConiferStep] = {}
        with ThreadPoolExecutor(max_workers=len(AGENT_IDS)) as pool:
            future_map = {
                pool.submit(
                    _baseline_call_step,
                    caller_by_agent[agent_id],
                    _baseline_messages(
                        row,
                        strategy="AgentVerse",
                        agent_id=agent_id,
                        instruction=role_descriptions[agent_id],
                        context=contexts[agent_id],
                    ),
                    active_agent=agent_id,
                    turn=turn_index + agent_index,
                    seed=seed + iteration * 10007 + agent_index * 97,
                    step_retries=step_retries,
                    request_log=request_log,
                    temperature=agent_temperature,
                ): agent_id
                for agent_index, agent_id in enumerate(route)
            }
            for future in future_map:
                current[future_map[future]] = future.result()
        for agent_index, agent_id in enumerate(route):
            target = route[(route.index(agent_id) + 1) % len(route)]
            _append_baseline_step(trajectory, row, current[agent_id], action="handoff", handoff_target=target)
        turn_index += len(AGENT_IDS)
        team_answer, _ = _baseline_vote({agent: step.tentative_answer for agent, step in current.items()})
        evaluator_context = (
            f"# AgentVerse evaluator iteration {iteration}\n"
            "Score the team answer from 0 to 10 against every explicit constraint and give concise repair feedback.\n"
            + "\n\n".join(f"## {agent}\n{step.tentative_answer}" for agent, step in current.items())
            + f"\n\nTeam answer:\n{team_answer}"
        )
        evaluator_payload, evaluator_raw = _baseline_call_json(
            caller_by_agent[route[-1]],
            [
                {
                    "role": "system",
                    "content": (
                        "You are the AgentVerse Evaluator. Judge requirement coverage, correctness, "
                        "and requested format. Return only the required score and feedback JSON object."
                    ),
                },
                {
                    "role": "user",
                    "content": format_problem_as_prompt(row, context_turns=2) + "\n\n" + evaluator_context,
                },
            ],
            turn=turn_index,
            seed=seed + iteration * 313,
            step_retries=step_retries,
            request_log=request_log,
            temperature=meta_temperature,
            response_format=_agentverse_evaluation_schema(),
        )
        raw_score = (evaluator_payload or {}).get("score")
        evaluator_score = int(raw_score) if isinstance(raw_score, (int, float)) else 0
        evaluator_score = max(0, min(10, evaluator_score))
        evaluator_feedback = str((evaluator_payload or {}).get("feedback") or "").strip()
        evaluator = ConiferStep(
            turn=turn_index,
            active_agent="A3",
            reasoning=f"AgentVerse evaluator score: {evaluator_score}/10. {evaluator_feedback}".strip(),
            tentative_answer=team_answer,
            action="handoff",
            handoff_target="A1",
            handoff_note=evaluator_feedback or "Revise the team answer against every explicit constraint.",
            confirmed_answer=None,
            raw_output=evaluator_raw,
        )
        _append_baseline_step(trajectory, row, evaluator, action="handoff", handoff_target=route[0])
        turn_index += 1
        feedback = evaluator_feedback
        previous_answers = current
        evaluator_scores.append(evaluator_score)
        if evaluator_score >= score_threshold:
            exit_reason = "score_threshold"
            break
    final_answer, tie_break = _baseline_vote(
        {agent: step.tentative_answer for agent, step in previous_answers.items()}
    )
    final_step = ConiferStep(
        turn=len(trajectory.steps), active_agent=route[-1],
        reasoning="AgentVerse controller finalized the last recruited team answer.",
        tentative_answer=final_answer, action="confirm_stop", handoff_target=None,
        handoff_note=None, confirmed_answer=final_answer, raw_output="",
    )
    _append_baseline_step(trajectory, row, final_step, action="confirm_stop", confirmed_answer=final_answer)
    trajectory.final_answer = final_answer
    trajectory.terminated_by = "stop"
    trajectory.quality = {
        "baseline": "agentverse", "iterations": len(evaluator_scores), "tie_break": tie_break,
        "start_agent": start_agent,
        "recruitment": recruitment_text, "roles": role_descriptions,
        "score_threshold": score_threshold,
        "evaluator_scores": evaluator_scores,
        "exit_reason": exit_reason,
    }


def _run_baseline_gptswarm(
    row: dict[str, Any],
    caller_by_agent: dict[str, Any],
    trajectory: ConiferTrajectory,
    *,
    start_agent: str,
    step_retries: int,
    seed: int,
    request_log: list[list[dict[str, str]]],
    temperature: float,
) -> None:
    node_specs = [
        ("io_0", "A1", "Extract concrete facts and explicit constraints independently.", ()),
        ("cot_0", "A2", "Reason through a complete candidate answer step by step.", ()),
        ("cot_1", "A2", "Try an independent alternative and stress-test likely pitfalls.", ()),
        ("debate_0", "A3", "Debate and repair all predecessor outputs.", ("io_0", "cot_0", "cot_1")),
        ("cot_2", "A2", "Re-solve the problem using the debate output as evidence.", ("debate_0",)),
        ("io_1", "A1", "Check formatting and requirement coverage from the debate output.", ("debate_0",)),
        ("aggregator", "A3", "Synthesize the strongest complete final answer from every predecessor.", ("debate_0", "cot_2", "io_1")),
    ]
    outputs: dict[str, ConiferStep] = {}
    turn_index = 0
    for layer in ((node_specs[0:3]), (node_specs[3:4]), (node_specs[4:6]), (node_specs[6:7])):
        current: dict[str, ConiferStep] = {}
        with ThreadPoolExecutor(max_workers=len(layer)) as pool:
            future_map = {}
            for node_name, agent_id, instruction, predecessors in layer:
                predecessor_text = "\n\n".join(
                    f"## {name}\nReasoning: {outputs[name].reasoning}\nDraft: {outputs[name].tentative_answer}"
                    for name in predecessors
                    if name in outputs
                )
                context = f"# GPTSwarm node: {node_name}\n{predecessor_text or '[no predecessors]'}"
                future_map[pool.submit(
                    _baseline_call_step,
                    caller_by_agent[agent_id],
                    _baseline_messages(
                        row, strategy="GPTSwarm", agent_id=agent_id,
                        instruction=instruction, context=context,
                    ),
                    active_agent=agent_id, turn=turn_index,
                    seed=seed + turn_index * 97, step_retries=step_retries,
                    request_log=request_log,
                    temperature=temperature,
                )] = (node_name, agent_id)
                turn_index += 1
            for future in future_map:
                node_name, agent_id = future_map[future]
                current[node_name] = future.result()
        for node_name, step in current.items():
            outputs[node_name] = step
            is_sink = node_name == "aggregator"
            if is_sink:
                _append_baseline_step(
                    trajectory, row, step, action="confirm_stop",
                    confirmed_answer=step.tentative_answer,
                )
            else:
                _append_baseline_step(
                    trajectory, row, step, action="handoff",
                    handoff_target=AGENT_IDS[(AGENT_IDS.index(step.active_agent) + 1) % len(AGENT_IDS)],
                )
    final_answer = outputs.get("aggregator", ConiferStep(0, "A3", "", "", "", None, None, None, "")).tentative_answer
    trajectory.final_answer = final_answer
    trajectory.terminated_by = "stop" if final_answer else "invalid_response"
    trajectory.quality = {
        "baseline": "gptswarm",
        "dag_layers": [[name for name, *_ in layer] for layer in ((node_specs[0:3]), (node_specs[3:4]), (node_specs[4:6]), (node_specs[6:7]))],
        "node_routing": {name: agent_id for name, agent_id, *_ in node_specs},
    }


def _run_baseline_aflow(
    row: dict[str, Any],
    caller_by_agent: dict[str, Any],
    trajectory: ConiferTrajectory,
    *,
    start_agent: str,
    step_retries: int,
    seed: int,
    request_log: list[list[dict[str, str]]],
    temperature: float,
) -> None:
    solve_specs = (
        ("solve_a1", "A1", "Solve independently by extracting every explicit content and format requirement."),
        ("solve_a2", "A2", "Produce an independent complete answer and audit factual and logical coverage."),
        ("solve_a3", "A3", "Solve independently, stress-test edge cases, and satisfy every constraint."),
    )
    candidates: dict[str, ConiferStep] = {}
    with ThreadPoolExecutor(max_workers=len(solve_specs)) as pool:
        future_map = {
            pool.submit(
                _baseline_call_step,
                caller_by_agent[agent_id],
                _baseline_messages(
                    row,
                    strategy="AFlow",
                    agent_id=agent_id,
                    instruction=instruction,
                    context=f"# AFlow node: {node_name}\nGenerate one independent candidate.",
                ),
                active_agent=agent_id,
                turn=turn_index,
                seed=seed + turn_index * 97,
                step_retries=step_retries,
                request_log=request_log,
                temperature=temperature,
            ): node_name
            for turn_index, (node_name, agent_id, instruction) in enumerate(solve_specs)
        }
        for future, node_name in ((future, future_map[future]) for future in future_map):
            candidates[node_name] = future.result()
    for index, (node_name, _, _) in enumerate(solve_specs):
        _append_baseline_step(
            trajectory,
            row,
            candidates[node_name],
            action="handoff",
            handoff_target=AGENT_IDS[(index + 1) % len(AGENT_IDS)],
        )

    candidate_text = "\n\n".join(
        f"## Candidate {index}\nReasoning: {candidates[node_name].reasoning}\n"
        f"Answer: {candidates[node_name].tentative_answer}"
        for index, (node_name, _, _) in enumerate(solve_specs)
    )
    ensemble = _baseline_call_step(
        caller_by_agent["A3"],
        _baseline_messages(
            row,
            strategy="AFlow",
            agent_id="A3",
            instruction=(
                "Select the strongest candidate. Begin tentative_answer with exactly "
                "`CHOSEN_INDEX: N`, where N is 0, 1, or 2, and explain the choice."
            ),
            context=f"# AFlow node: ensemble\n{candidate_text}",
        ),
        active_agent="A3",
        turn=3,
        seed=seed + 3 * 97,
        step_retries=step_retries,
        request_log=request_log,
        temperature=temperature,
    )
    choice_match = re.search(
        r"\bCHOSEN_INDEX\s*[:=]\s*([0-2])\b",
        "\n".join((ensemble.tentative_answer, ensemble.reasoning, ensemble.raw_output)),
        flags=re.IGNORECASE,
    )
    chosen_index = int(choice_match.group(1)) if choice_match else 2
    chosen_name = solve_specs[chosen_index][0]
    chosen = candidates[chosen_name]
    _append_baseline_step(trajectory, row, ensemble, action="handoff", handoff_target="A3")

    answer_generate = _baseline_call_step(
        caller_by_agent["A3"],
        _baseline_messages(
            row,
            strategy="AFlow",
            agent_id="A3",
            instruction="Return the chosen candidate as a polished final answer with every constraint preserved.",
            context=(
                "# AFlow node: answer_generate\n"
                f"Chosen candidate {chosen_index}:\nReasoning: {chosen.reasoning}\n"
                f"Answer: {chosen.tentative_answer}"
            ),
        ),
        active_agent="A3",
        turn=4,
        seed=seed + 4 * 97,
        step_retries=step_retries,
        request_log=request_log,
        temperature=temperature,
    )
    _append_baseline_step(
        trajectory,
        row,
        answer_generate,
        action="confirm_stop",
        confirmed_answer=answer_generate.tentative_answer,
    )
    trajectory.final_answer = answer_generate.tentative_answer
    trajectory.terminated_by = "stop" if trajectory.final_answer else "invalid_response"
    trajectory.quality = {
        "baseline": "aflow",
        "workflow": ["solve_a1", "solve_a2", "solve_a3", "ensemble", "answer_generate"],
        "chosen_index": chosen_index,
        "choice_parsed": choice_match is not None,
        "executor_routing": {"solve_a1": "A1", "solve_a2": "A2", "solve_a3": "A3"},
        "judge_agent": "A3",
    }


_AFLOW_SEARCHED_CACHE: dict[str, Any] = {}
_AFLOW_UNCONSTRAINED_CACHE: dict[int, Any] = {}


def _unconstrained_caller(caller: Any) -> Any:
    """Return a clone of ``caller`` with guided decoding switched off.

    The callers built by :func:`build_callers` pin ``response_format`` to the
    Conifer protocol schema (``reasoning``/``tentative_answer``/``action``).
    AFlow operators use their own, different schemas, so reusing those callers
    forces every operator response into the wrong envelope and the operator
    parser rejects all of them.  Everything else -- endpoint, model, generation
    options, retries -- is preserved, so the only difference from the other
    Conifer baselines is the decoding constraint.

    See AFLOW_SEARCH_MIGRATION.md 4.2; ``run_search_conifer.py`` leaves
    ``response_format`` unset for the same reason during the search stage.
    """
    cached = _AFLOW_UNCONSTRAINED_CACHE.get(id(caller))
    if cached is not None:
        return cached
    if isinstance(caller, EndpointPoolCaller):
        clone: Any = EndpointPoolCaller([_unconstrained_caller(inner) for inner in caller.callers])
    elif isinstance(caller, OpenAIChatLLMCaller):
        if caller.response_format is None:
            clone = caller
        else:
            clone = OpenAIChatLLMCaller(
                caller.base_url,
                caller.model_name,
                generation=caller.generation,
                timeout=caller.timeout,
                api_key=caller.api_key,
                max_model_len=caller.max_model_len,
                response_format=None,
                length_retries=caller.length_retries,
                thinking_fallback=caller.thinking_fallback,
                thinking_max_tokens=caller.thinking_max_tokens,
                thinking_stop=caller.thinking_stop,
                fallback_bad_words=caller.fallback_bad_words,
                fallback_stop=caller.fallback_stop,
                connection_pool=caller.connection_pool,
                connection_pool_maxsize=caller.connection_pool_maxsize,
            )
    else:
        # MockCaller and friends ignore response_format already.
        clone = caller
    _AFLOW_UNCONSTRAINED_CACHE[id(caller)] = clone
    return clone


def _load_searched_aflow_workflow(workflow_file: str):
    """Load a searched AFlow workflow through the shared sandbox, once."""
    cached = _AFLOW_SEARCHED_CACHE.get(workflow_file)
    if cached is not None:
        return cached
    aflow_dir = Path(__file__).resolve().parent / "aflow"
    if str(aflow_dir) not in sys.path:
        sys.path.insert(0, str(aflow_dir))
    from conifer_aflow_adapter import configure  # noqa: E402

    modules = configure()
    workflow_module = modules["workflow"]
    loaded = (
        workflow_module,
        modules["ConiferProblem"],
        workflow_module.load_workflow_from_file(Path(workflow_file), round_id=0),
    )
    _AFLOW_SEARCHED_CACHE[workflow_file] = loaded
    return loaded


def _run_baseline_aflow_searched(
    row: dict[str, Any],
    caller_by_agent: dict[str, Any],
    trajectory: ConiferTrajectory,
    *,
    workflow_file: str,
    request_log: list[list[dict[str, str]]],
) -> None:
    """Execute a searched AFlow workflow and project it onto the protocol.

    The workflow is a searched artifact, so its shape is not known ahead of
    time: every operator call becomes one protocol step (``handoff`` except for
    the last, which is ``confirm_stop``).  Downstream scoring and summarization
    only look at ``ConiferTrajectory``, so they need no changes.
    """
    workflow_module, conifer_problem_cls, workflow = _load_searched_aflow_workflow(workflow_file)
    problem = conifer_problem_cls.from_row(row)
    a1, a2, a3 = (_unconstrained_caller(caller_by_agent[agent]) for agent in AGENT_IDS)
    callers = workflow_module.ExecutorCallers([a1, a2, a3], a3, ["A1", "A2", "A3"], "A3")
    record = workflow_module.run_workflow_on_problem(workflow, problem, callers)
    final_answer = record.final_answer or ""

    op_calls = list(record.op_calls)
    for index, call in enumerate(op_calls):
        is_last = index == len(op_calls) - 1
        agent = call.caller_id if call.caller_id in AGENT_IDS else "A3"
        raw_outputs = list(call.raw_outputs or [])
        step = ConiferStep(
            turn=index,
            active_agent=agent,
            reasoning=f"AFlow op `{call.op}` ({'ok' if call.ok else 'failed'}).",
            tentative_answer=final_answer if is_last else "",
            action="confirm_stop",
            handoff_target=None,
            handoff_note=None,
            confirmed_answer=None,
            raw_output=raw_outputs[-1] if raw_outputs else "",
        )
        step.raw_outputs = raw_outputs
        request_log.append([{"role": "system", "content": f"aflow_op:{call.op}"}])
        _append_baseline_step(
            trajectory,
            row,
            step,
            action="confirm_stop" if is_last else "handoff",
            handoff_target=None if is_last else AGENT_IDS[(index + 1) % len(AGENT_IDS)],
            confirmed_answer=final_answer if is_last else None,
        )

    trajectory.final_answer = final_answer
    if record.error:
        trajectory.terminated_by = "exception"
        trajectory.error = record.error
    else:
        trajectory.terminated_by = "stop" if final_answer else "invalid_response"
    trajectory.quality = {
        "baseline": "aflow",
        "aflow_mode": "searched",
        "workflow_file": workflow_file,
        "workflow_code_hash": workflow.code_hash,
        "workflow": [call.op for call in op_calls],
        "n_ops": len(op_calls),
        "failed_ops": [call.op for call in op_calls if not call.ok],
        "executor_routing": {"solve": "A1/A2/A3 round-robin"},
        "judge_agent": "A3",
    }


def run_baseline_trajectory(
    row: dict[str, Any],
    caller_by_agent: dict[str, Any],
    *,
    strategy: str,
    start_agent: str,
    t_max: int,
    step_retries: int,
    seed: int,
    baseline_rounds: int,
    baseline_iterations: int,
    baseline_mad_temperature_round0: float,
    baseline_mad_temperature_debate: float,
    baseline_agent_temperature: float,
    baseline_meta_temperature: float,
    baseline_score_threshold: int,
    baseline_gptswarm_temperature: float,
    baseline_aflow_temperature: float,
    baseline_aflow_workflow_file: str = "",
) -> tuple[ConiferTrajectory, list[list[dict[str, str]]]]:
    if strategy not in {"mad", "agentverse", "gptswarm", "aflow"}:
        raise ValueError(f"unknown baseline strategy: {strategy}")
    trajectory = ConiferTrajectory(problem_id=str(row["problem_id"]), start_agent=start_agent)
    request_log: list[list[dict[str, str]]] = []
    try:
        if strategy == "mad":
            _run_baseline_mad(
                row, caller_by_agent, trajectory, start_agent=start_agent,
                rounds=min(max(1, t_max), baseline_rounds),
                step_retries=step_retries, seed=seed, request_log=request_log,
                temperature_round0=baseline_mad_temperature_round0,
                temperature_debate=baseline_mad_temperature_debate,
            )
        elif strategy == "agentverse":
            _run_baseline_agentverse(
                row, caller_by_agent, trajectory, start_agent=start_agent,
                iterations=min(max(1, t_max), baseline_iterations),
                step_retries=step_retries, seed=seed, request_log=request_log,
                agent_temperature=baseline_agent_temperature,
                meta_temperature=baseline_meta_temperature,
                score_threshold=baseline_score_threshold,
            )
        elif strategy == "gptswarm":
            _run_baseline_gptswarm(
                row, caller_by_agent, trajectory, start_agent=start_agent,
                step_retries=step_retries, seed=seed, request_log=request_log,
                temperature=baseline_gptswarm_temperature,
            )
        elif baseline_aflow_workflow_file:
            # Searched workflow: the real AFlow method, with its MCTS product.
            _run_baseline_aflow_searched(
                row, caller_by_agent, trajectory,
                workflow_file=baseline_aflow_workflow_file,
                request_log=request_log,
            )
        else:
            # Hardcoded 5-op graph, byte-for-byte the previous behaviour.
            _run_baseline_aflow(
                row, caller_by_agent, trajectory, start_agent=start_agent,
                step_retries=step_retries, seed=seed, request_log=request_log,
                temperature=baseline_aflow_temperature,
            )
    except Exception as exc:
        trajectory.terminated_by = "exception"
        trajectory.error = f"{type(exc).__name__}: {exc}"
    return trajectory, request_log


def _serialize_result(
    row: dict[str, Any],
    trajectory: ConiferTrajectory,
    turn_messages: list[list[dict[str, str]]],
    rollout_index: int,
    seed: int,
    sampling: dict[str, Any] | None = None,
) -> dict[str, Any]:
    final_answer = trajectory.final_answer or ""
    final_checks = check_constraints(row, final_answer)
    return {
        "schema_version": 1,
        "problem_id": row["problem_id"],
        "group_id": row.get("group_id"),
        "rollout_idx": rollout_index,
        "seed": seed,
        "start_agent": trajectory.start_agent,
        "source_policy": sampling.get("source_policy"),
        "teacher_rollout": bool(sampling.get("teacher_rollout", False)),
        "rollout_stage": sampling.get("rollout_stage", "unknown"),
        "sampling": sampling or {},
        "problem": row,
        "trajectory": trajectory_to_dict(trajectory),
        "turn_messages": turn_messages,
        "final_answer": final_answer,
        "final_checks": final_checks,
        "hard_score": final_checks["hard_score"],
        "reference_lexical_f1": final_checks.get("reference_lexical_f1"),
        "created_at": time.time(),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-path", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--num-rollouts", type=int, default=1)
    p.add_argument("--t-max", type=int, default=6)
    p.add_argument(
        "--start-agent",
        choices=[*AGENT_IDS, "balanced"],
        default="balanced",
        help="Use a fixed first agent or deterministically rotate A1/A2/A3 across rollout keys.",
    )
    p.add_argument("--start-agent-seed", type=int, default=42, help="Phase offset for balanced rotation.")
    p.add_argument("--min-agents-before-stop", type=int, default=3)
    p.add_argument("--min-handoffs-before-stop", type=int, default=2)
    p.add_argument("--enforce-runtime-policy", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--step-retries", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens", type=int, default=1536)
    p.add_argument(
        "--protocol-max-new-tokens",
        type=int,
        default=0,
        help="Initial JSON protocol budget; zero uses --max-new-tokens and length retries can expand it.",
    )
    p.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--json-transport", choices=["json_schema", "json_object", "none"], default="json_schema")
    p.add_argument("--api-base-a1", default=os.environ.get("CONIFER_API_BASE_A1"))
    p.add_argument("--api-base-a2", default=os.environ.get("CONIFER_API_BASE_A2"))
    p.add_argument("--api-base-a3", default=os.environ.get("CONIFER_API_BASE_A3"))
    p.add_argument("--model-a1", default=os.environ.get("CONIFER_MODEL_A1", "A1"))
    p.add_argument("--model-a2", default=os.environ.get("CONIFER_MODEL_A2", "A2"))
    p.add_argument("--model-a3", default=os.environ.get("CONIFER_MODEL_A3", "A3"))
    p.add_argument("--model-path-a1", default=os.environ.get("CONIFER_MODEL_PATH_A1", ""))
    p.add_argument("--model-path-a2", default=os.environ.get("CONIFER_MODEL_PATH_A2", ""))
    p.add_argument("--model-path-a3", default=os.environ.get("CONIFER_MODEL_PATH_A3", ""))
    p.add_argument("--adapter-path-a1", default=os.environ.get("CONIFER_ADAPTER_PATH_A1", ""))
    p.add_argument("--adapter-path-a2", default=os.environ.get("CONIFER_ADAPTER_PATH_A2", ""))
    p.add_argument("--adapter-path-a3", default=os.environ.get("CONIFER_ADAPTER_PATH_A3", ""))
    p.add_argument("--student-model-path-a1", default=os.environ.get("CONIFER_STUDENT_MODEL_PATH_A1", ""))
    p.add_argument("--student-model-path-a2", default=os.environ.get("CONIFER_STUDENT_MODEL_PATH_A2", ""))
    p.add_argument("--student-model-path-a3", default=os.environ.get("CONIFER_STUDENT_MODEL_PATH_A3", ""))
    p.add_argument("--source-policy", default=os.environ.get("CONIFER_SOURCE_POLICY", "unknown"))
    p.add_argument("--rollout-stage", default=os.environ.get("CONIFER_ROLLOUT_STAGE", "unknown"))
    p.add_argument("--teacher-rollout", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--api-key", default=os.environ.get("CONIFER_API_KEY", os.environ.get("OPENAI_API_KEY", "EMPTY")))
    p.add_argument("--api-timeout", type=float, default=600.0)
    p.add_argument("--max-model-len", type=int, default=16384)
    p.add_argument("--length-retries", type=int, default=1)
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=512,
        help="Maximum trajectories in flight; size this to keep vLLM replicas busy.",
    )
    p.add_argument(
        "--progress-every",
        type=int,
        default=32,
        help="Emit one progress line after this many completed trajectories.",
    )
    p.add_argument(
        "--flush-every",
        type=int,
        default=32,
        help="Flush the resumable rollout file after this many results.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--include-reference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Expose the dataset reference only for controlled SFT smoke generation.",
    )
    p.add_argument("--context-turns", type=int, default=2)
    p.add_argument(
        "--include-source-assistant-context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Expose native source answers in context; keep disabled for unbiased evaluation.",
    )
    p.add_argument("--routing", choices=["dynamic", "cycle"], default="dynamic")
    p.add_argument("--force-handoff-until-final", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument(
        "--baseline-strategy",
        choices=["none", "mad", "agentverse", "gptswarm", "aflow"],
        default="none",
        help="Run a named training-free baseline controller instead of the sequential JCA controller.",
    )
    p.add_argument("--baseline-rounds", type=int, default=3)
    p.add_argument("--baseline-iterations", type=int, default=3)
    p.add_argument("--baseline-mad-temperature-round0", type=float, default=0.9)
    p.add_argument("--baseline-mad-temperature-debate", type=float, default=0.3)
    p.add_argument("--baseline-agent-temperature", type=float, default=0.7)
    p.add_argument("--baseline-meta-temperature", type=float, default=0.0)
    p.add_argument("--baseline-score-threshold", type=int, default=8)
    p.add_argument("--baseline-gptswarm-temperature", type=float, default=0.7)
    p.add_argument("--baseline-aflow-temperature", type=float, default=0.7)
    p.add_argument(
        "--baseline-aflow-workflow-file",
        default="",
        help=(
            "Execute this searched AFlow workflow instead of the hardcoded 5-op graph. "
            "Produced by 03_rollout/aflow/run_search_conifer.py. Empty keeps the old behaviour."
        ),
    )
    p.add_argument("--mock", action="store_true", help="Use deterministic local responses; never use for quality claims.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.start < 0 or args.limit < 0 or args.num_rollouts <= 0 or args.t_max <= 0:
        raise SystemExit("start/limit must be non-negative and rollout/t-max must be positive")
    rows = _jsonl(args.data_path)
    selected = rows[args.start : (args.start + args.limit if args.limit else None)]
    if not selected:
        raise SystemExit("No rows selected")
    if args.max_concurrency <= 0 or args.progress_every <= 0 or args.flush_every <= 0:
        raise SystemExit("concurrency and progress/flush intervals must be positive")
    if args.max_new_tokens <= 0 or args.protocol_max_new_tokens < 0:
        raise SystemExit("protocol-max-new-tokens must be non-negative")
    if args.baseline_rounds <= 0 or args.baseline_iterations <= 0:
        raise SystemExit("baseline rounds/iterations must be positive")
    if not 0 <= args.baseline_score_threshold <= 10:
        raise SystemExit("baseline score threshold must be in [0, 10]")
    baseline_temperatures = (
        args.baseline_mad_temperature_round0,
        args.baseline_mad_temperature_debate,
        args.baseline_agent_temperature,
        args.baseline_meta_temperature,
        args.baseline_gptswarm_temperature,
        args.baseline_aflow_temperature,
    )
    if any(value < 0 for value in baseline_temperatures):
        raise SystemExit("baseline temperatures must be non-negative")
    requested_max_new_tokens = args.max_new_tokens
    protocol_max_new_tokens = args.protocol_max_new_tokens or args.max_new_tokens
    args.max_new_tokens = protocol_max_new_tokens
    callers = _build_callers(args)
    sampling = {
        "schema_version": 1,
        "source_policy": args.source_policy,
        "teacher_rollout": bool(args.teacher_rollout),
        "api_models": {"A1": args.model_a1, "A2": args.model_a2, "A3": args.model_a3},
        "model_paths": {"A1": args.model_path_a1, "A2": args.model_path_a2, "A3": args.model_path_a3},
        "adapter_paths": {"A1": args.adapter_path_a1, "A2": args.adapter_path_a2, "A3": args.adapter_path_a3},
        "student_models": {
            "A1": args.student_model_path_a1,
            "A2": args.student_model_path_a2,
            "A3": args.student_model_path_a3,
        },
        "teacher_models": (
            {"A1": args.model_path_a1, "A2": args.model_path_a2, "A3": args.model_path_a3}
            if args.teacher_rollout else {}
        ),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "requested_max_new_tokens": requested_max_new_tokens,
        "t_max": args.t_max,
        "start_agent_policy": args.start_agent,
        "start_agent_seed": args.start_agent_seed,
        "min_agents_before_stop": args.min_agents_before_stop,
        "min_handoffs_before_stop": args.min_handoffs_before_stop,
        "routing": args.routing,
        "force_handoff_until_final": bool(args.force_handoff_until_final),
        "rollout_stage": args.rollout_stage,
        "baseline_strategy": args.baseline_strategy,
        "baseline_rounds": args.baseline_rounds,
        "baseline_iterations": args.baseline_iterations,
        "baseline_temperatures": {
            "mad_round0": args.baseline_mad_temperature_round0,
            "mad_debate": args.baseline_mad_temperature_debate,
            "agentverse_agent": args.baseline_agent_temperature,
            "agentverse_meta": args.baseline_meta_temperature,
            "gptswarm": args.baseline_gptswarm_temperature,
            "aflow": args.baseline_aflow_temperature,
        },
        "baseline_score_threshold": args.baseline_score_threshold,
        "baseline_aflow_mode": "searched" if args.baseline_aflow_workflow_file else "initial_5op",
        "baseline_aflow_workflow_file": args.baseline_aflow_workflow_file,
    }
    existing: dict[tuple[str, int], str] = {}
    retained_rows: list[dict[str, Any]] = []
    retained_terminations: Counter[str] = Counter()
    if args.resume and args.output.is_file():
        retryable_rows = 0
        prior_rows: dict[tuple[str, int], dict[str, Any]] = {}
        for item in _jsonl(args.output):
            key = (str(item.get("problem_id")), int(item.get("rollout_idx", 0)))
            actual_start = str(item.get("start_agent") or (item.get("trajectory") or {}).get("start_agent") or "")
            previous = prior_rows.get(key)
            previous_start = str(
                previous.get("start_agent") or (previous.get("trajectory") or {}).get("start_agent") or ""
            ) if previous is not None else None
            if previous_start is not None and previous_start != actual_start:
                raise SystemExit(f"Conflicting existing start agents for rollout key {key}: {previous_start} vs {actual_start}")
            if previous is not None and _termination(previous) != "exception" and _termination(item) != "exception":
                raise SystemExit(f"Duplicate completed rollout key in existing output: {key}")
            if previous is None or _termination(previous) == "exception":
                prior_rows[key] = item
        for key, item in prior_rows.items():
            if _termination(item) == "exception":
                retryable_rows += 1
                continue
            actual_start = str(item.get("start_agent") or (item.get("trajectory") or {}).get("start_agent") or "")
            existing[key] = actual_start
            retained_rows.append(item)
            retained_terminations[_termination(item)] += 1
        if retryable_rows:
            _rewrite_jsonl(args.output, retained_rows)
            print(json.dumps({
                "resume_pruned_retryable": retryable_rows,
                "resume_retained": len(retained_rows),
            }, ensure_ascii=False), flush=True)
    jobs = []
    requested_start_counts: Counter[str] = Counter()
    pending_start_counts: Counter[str] = Counter()
    for row_index, row in enumerate(selected, start=args.start):
        for rollout_index in range(args.num_rollouts):
            start_agent = _select_start(row_index, rollout_index, args.start_agent, args.start_agent_seed)
            requested_start_counts[start_agent] += 1
            key = (str(row["problem_id"]), rollout_index)
            if key in existing:
                if existing[key] != start_agent:
                    raise SystemExit(
                        f"Existing rollout {key} starts with {existing[key] or '<missing>'}, but the current "
                        f"{args.start_agent!r} policy requires {start_agent}. Use a new output path/TAG."
                    )
                continue
            jobs.append((row, row_index, rollout_index, start_agent))
            pending_start_counts[start_agent] += 1
    if args.start_agent == "balanced":
        requested = [requested_start_counts[agent] for agent in AGENT_IDS]
        if max(requested) - min(requested) > 1:
            raise SystemExit(f"Balanced start schedule is unexpectedly skewed: {dict(requested_start_counts)}")
    print(json.dumps({
        "start_agent_policy": args.start_agent,
        "start_agent_seed": args.start_agent_seed,
        "requested_start_counts": {agent: requested_start_counts[agent] for agent in AGENT_IDS},
        "pending_start_counts": {agent: pending_start_counts[agent] for agent in AGENT_IDS},
    }, ensure_ascii=False), flush=True)
    if not jobs:
        print("All requested rollout keys already exist; nothing to do.")
        return

    def one(job: tuple[dict[str, Any], int, int, str]) -> dict[str, Any]:
        row, row_index, rollout_index, start_agent = job
        seed = args.seed + row_index * 100_003 + rollout_index * 1_009
        if args.baseline_strategy == "none":
            trajectory, turn_messages = run_trajectory(
                row, callers, start_agent=start_agent, t_max=args.t_max,
                min_agents_before_stop=args.min_agents_before_stop,
                min_handoffs_before_stop=args.min_handoffs_before_stop,
                enforce_runtime_policy=args.enforce_runtime_policy,
                step_retries=args.step_retries, seed=seed,
                include_reference=args.include_reference,
                routing=args.routing,
                force_handoff_until_final=args.force_handoff_until_final,
                context_turns=args.context_turns,
                include_source_assistant_context=args.include_source_assistant_context,
            )
        else:
            trajectory, turn_messages = run_baseline_trajectory(
                row, callers, strategy=args.baseline_strategy,
                start_agent=start_agent, t_max=args.t_max,
                step_retries=args.step_retries, seed=seed,
                baseline_rounds=args.baseline_rounds,
                baseline_iterations=args.baseline_iterations,
                baseline_mad_temperature_round0=args.baseline_mad_temperature_round0,
                baseline_mad_temperature_debate=args.baseline_mad_temperature_debate,
                baseline_agent_temperature=args.baseline_agent_temperature,
                baseline_meta_temperature=args.baseline_meta_temperature,
                baseline_score_threshold=args.baseline_score_threshold,
                baseline_gptswarm_temperature=args.baseline_gptswarm_temperature,
                baseline_aflow_temperature=args.baseline_aflow_temperature,
                baseline_aflow_workflow_file=args.baseline_aflow_workflow_file,
            )
        return _serialize_result(row, trajectory, turn_messages, rollout_index, seed, sampling)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    termination_counts = Counter(retained_terminations)
    started_at = time.monotonic()
    with args.output.open("a", encoding="utf-8", buffering=1024 * 1024) as output_handle:
        worker_count = max(1, min(args.max_concurrency, len(jobs)))
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            job_iter = iter(jobs)
            futures = {}
            for _ in range(worker_count):
                try:
                    futures[pool.submit(one, next(job_iter))] = True
                except StopIteration:
                    break
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    del futures[future]
                    result = future.result()
                    output_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                    termination_counts[_termination(result)] += 1
                    completed += 1
                    try:
                        futures[pool.submit(one, next(job_iter))] = True
                    except StopIteration:
                        pass
                    if completed % args.flush_every == 0 or completed == len(jobs):
                        output_handle.flush()
                    if completed % args.progress_every == 0 or completed == len(jobs):
                        elapsed = max(time.monotonic() - started_at, 1e-6)
                        print(json.dumps({
                            "completed": completed,
                            "total": len(jobs),
                            "rate_rollouts_per_sec": round(completed / elapsed, 3),
                        }, ensure_ascii=False), flush=True)
    print(json.dumps({
        "rollout_termination_counts": dict(sorted(termination_counts.items())),
        "total_rollouts": sum(termination_counts.values()),
    }, ensure_ascii=False), flush=True)
    if termination_counts.get("exception", 0):
        raise SystemExit(
            f"Infrastructure exceptions remain in rollout output: {termination_counts['exception']}; rerun with resume enabled"
        )


if __name__ == "__main__":
    main()

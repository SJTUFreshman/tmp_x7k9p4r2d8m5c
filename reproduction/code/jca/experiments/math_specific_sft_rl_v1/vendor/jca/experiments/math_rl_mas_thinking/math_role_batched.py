#!/usr/bin/env python3
"""Stateful MATH MAS inference with global-turn role batching."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from threading import Condition, Event, Lock, Thread
from typing import Any, Callable, Dict, Optional

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import math_rollout as base
from jca.src.inference import GenerationOptions, OpenAIChatLLMCaller
from jca.src.math_eval import (
    AGENT_IDS,
    MathProblem,
    extract_last_boxed,
    format_math_problem_as_prompt,
    load_math_problems,
    math_answers_equivalent,
    render_math_mas_system_prompt,
)
from jca.src.protocol_json import (
    PROTOCOL_FIELDS,
    json_object_well_formed,
    protocol_schema_error,
    protocol_json_schema_well_formed as strict_protocol_json_well_formed,
    validate_protocol_object,
)


STATE_VERSION = 1
JOURNAL_VERSION = 1
FAILURE_LEDGER_VERSION = 1
DIRECT_SOLVER_SYSTEM_PROMPT = (
    "You are an expert mathematical problem solver. Solve the problem carefully. "
    "Return a concise derivation and end the visible response with exactly one "
    "final answer in \\boxed{...}. A response without a final \\boxed{...} answer "
    "is unusable and will be retried."
)
DEGENERATE_THINKING_STOP = "$ $ $ $ $ $ $ $ $ $ $ $"
DEGENERATE_FALLBACK_BAD_WORDS = (
    "$$$$",
    "$$$ $",
    "$$ $$",
    "$$ $ $",
    "$ $$$",
    "$ $$ $",
    "$ $ $$",
    "$ $ $ $",
)
DEGENERATE_RECOVERY_AGENT = "A3"
DEGENERATE_RECOVERY_TARGET_ORDER = ("A2", "A1", "A3")
PROTOCOL_REASONING_MAX_CHARS = 512
PROTOCOL_ANSWER_MAX_CHARS = 128
PROTOCOL_NOTE_MAX_CHARS = 256
DEFAULT_PROTOCOL_MAX_TOKENS = 1024
JOURNAL_ITEM_FIELDS = (
    "current_agent",
    "group_attempt",
    "prior_tentative",
    "status",
    "last_error",
    "trajectory",
    "messages",
    "generation_attempts",
    "generation_rejections",
    "last_rejected_raw",
    "bootstrap",
    "pending_recovery",
)

_STATE_STATUSES = frozenset({"pending", "retry", "failed", "done"})
_TERMINAL_REASONS = frozenset(
    {"stop", "rejected_quality", "truncated", "exception"}
)
_CONFIG_DEFAULTS: Dict[str, Any] = {
    # These defaults are only for states written by the pre-journal runner.
    # New states always persist every field explicitly.
    "allow_first_turn_stop": False,
    # The original MATH runner defaulted to a plain protocol response when
    # the thinking flag was omitted.  Materialize that historical default so
    # an interrupted pre-journal state can still be resumed safely.
    "enable_thinking": False,
    "retry_failed_groups": False,
    "output_mode": "turns",
    "json_transport": "none",
    "start_agent_seed": 42,
    "split": "train",
    "subjects": [],
    "bootstrap_agent": None,
    "bootstrap_handoff_target": None,
    "preserve_reasonable_incumbent": False,
    "lock_upstream_answer_agents": [],
}
_PROBLEM_FIELDS = frozenset(
    {"problem_id", "subject", "level", "prompt", "solution", "gold_answer"}
)
_ITEM_FIELDS = frozenset(
    {
        "problem_index",
        "rollout_idx",
        "problem",
        "start_agent",
        "current_agent",
        "group_attempt",
        "prior_tentative",
        "status",
        "last_error",
        "trajectory",
        "messages",
    }
)
_OPTIONAL_ITEM_FIELDS = frozenset(
    {
        "generation_attempts",
        "generation_rejections",
        "last_rejected_raw",
        "bootstrap",
        "pending_recovery",
    }
)
_PENDING_RECOVERY_FIELDS = frozenset(
    {
        "collapsed_agent",
        "turn",
        "reason",
        "exact_tentative_answer",
        "retry_failed_group",
    }
)


class EndpointLoadBalancer:
    """Keep concurrent requests evenly distributed across explicit endpoints."""

    def __init__(self, endpoint_count: int) -> None:
        if endpoint_count <= 0:
            raise ValueError("endpoint_count must be positive")
        self._active = [0] * endpoint_count
        self._next_index = 0
        self._lock = Lock()

    def acquire(self) -> int:
        with self._lock:
            minimum = min(self._active)
            for offset in range(len(self._active)):
                index = (self._next_index + offset) % len(self._active)
                if self._active[index] == minimum:
                    self._active[index] += 1
                    self._next_index = (index + 1) % len(self._active)
                    return index
        raise RuntimeError("endpoint selection failed")

    def release(self, index: int) -> None:
        with self._lock:
            if not 0 <= index < len(self._active) or self._active[index] <= 0:
                raise RuntimeError(f"invalid endpoint release: {index}")
            self._active[index] -= 1


class OpenAIEndpointPoolCaller:
    """Route each complete generation attempt to one least-active endpoint."""

    def __init__(
        self,
        callers: list[OpenAIChatLLMCaller],
        load_balancer: EndpointLoadBalancer,
    ) -> None:
        if not callers:
            raise ValueError("callers must be non-empty")
        self.callers = callers
        self.load_balancer = load_balancer

    def __call__(self, messages: list[Dict[str, str]]) -> str:
        return self.generate(messages)

    def generate(
        self,
        messages: list[Dict[str, str]],
        *,
        seed: int | None = None,
    ) -> str:
        index = self.load_balancer.acquire()
        try:
            return self.callers[index].generate(messages, seed=seed)
        finally:
            self.load_balancer.release(index)


class LiveEndpointRegistry:
    """Share endpoint admission and active-request state with the shell manager."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        control_path: Path,
        status_path: Path,
        endpoint_capacity: int,
        *,
        poll_seconds: float = 0.2,
    ) -> None:
        if endpoint_capacity <= 0:
            raise ValueError("endpoint_capacity must be positive")
        self.control_path = control_path
        self.status_path = status_path
        self.endpoint_capacity = endpoint_capacity
        self.poll_seconds = max(0.05, poll_seconds)
        self._condition = Condition(Lock())
        self._endpoints: Dict[str, Dict[str, Any]] = {}
        self._control_revision = -1
        self._next_index = {agent: 0 for agent in AGENT_IDS}
        self._workload: Optional[Dict[str, Any]] = None
        self._watch_error: Optional[BaseException] = None
        self._runner_active = True
        self._stop = Event()
        with self._condition:
            self._refresh_locked(force=True)
            self._write_status_locked()
        self._watcher = Thread(
            target=self._watch_control,
            name="math-live-endpoint-registry",
            daemon=True,
        )
        self._watcher.start()

    def _load_control(self) -> Dict[str, Any]:
        with self.control_path.open("r", encoding="utf-8") as handle:
            control = json.load(handle)
        if not isinstance(control, dict):
            raise ValueError("live endpoint control must be a JSON object")
        if control.get("schema_version") != self.SCHEMA_VERSION:
            raise ValueError("unsupported live endpoint control schema")
        revision = control.get("revision")
        if not isinstance(revision, int) or revision <= 0:
            raise ValueError("live endpoint control revision must be positive")
        records = control.get("endpoints")
        if not isinstance(records, list) or not records:
            raise ValueError("live endpoint control needs at least one endpoint")
        normalized: Dict[str, Dict[str, Any]] = {}
        gpus: set[int] = set()
        for record in records:
            if not isinstance(record, dict):
                raise ValueError("invalid live endpoint record")
            endpoint_id = record.get("id")
            gpu = record.get("gpu")
            agent = record.get("agent")
            base_url = record.get("base_url")
            accepting = record.get("accepting")
            if not isinstance(endpoint_id, str) or not endpoint_id:
                raise ValueError("live endpoint id must be non-empty")
            if not isinstance(gpu, int) or gpu < 0 or endpoint_id != f"gpu{gpu}":
                raise ValueError(f"invalid live endpoint GPU identity: {record!r}")
            if agent not in AGENT_IDS:
                raise ValueError(f"invalid live endpoint agent: {agent!r}")
            if not isinstance(base_url, str) or not base_url.rstrip("/").endswith("/v1"):
                raise ValueError(f"invalid live endpoint base URL: {base_url!r}")
            if not isinstance(accepting, bool):
                raise ValueError("live endpoint accepting flag must be boolean")
            if endpoint_id in normalized or gpu in gpus:
                raise ValueError("duplicate live endpoint id or GPU")
            normalized[endpoint_id] = {
                "id": endpoint_id,
                "gpu": gpu,
                "agent": agent,
                "base_url": base_url.rstrip("/"),
                "accepting": accepting,
                "replacement_agent": record.get("replacement_agent"),
                "active": 0,
            }
            gpus.add(gpu)
        return {"revision": revision, "endpoints": normalized}

    def _refresh_locked(self, *, force: bool = False) -> bool:
        control = self._load_control()
        revision = int(control["revision"])
        if not force and revision == self._control_revision:
            return False
        if revision < self._control_revision:
            raise ValueError("live endpoint control revision moved backwards")
        incoming = control["endpoints"]
        for endpoint_id, current in self._endpoints.items():
            active = int(current["active"])
            replacement = incoming.get(endpoint_id)
            if replacement is None and active:
                raise RuntimeError(f"active endpoint removed from control: {endpoint_id}")
            if active and replacement is not None and (
                replacement["agent"] != current["agent"]
                or replacement["base_url"] != current["base_url"]
            ):
                raise RuntimeError(f"active endpoint reassigned by control: {endpoint_id}")
        for endpoint_id, endpoint in incoming.items():
            if endpoint_id in self._endpoints:
                endpoint["active"] = int(self._endpoints[endpoint_id]["active"])
        self._endpoints = incoming
        self._control_revision = revision
        self._write_status_locked()
        self._condition.notify_all()
        return True

    def _raise_watch_error_locked(self) -> None:
        if self._watch_error is not None:
            raise RuntimeError("live endpoint registry watcher failed") from self._watch_error

    def _write_status_locked(self) -> None:
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.status_path.with_name(
            f".{self.status_path.name}.tmp.{os.getpid()}"
        )
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "runner_pid": os.getpid(),
            "runner_active": self._runner_active,
            "control_revision": self._control_revision,
            "endpoint_capacity": self.endpoint_capacity,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "endpoints": [
                dict(endpoint)
                for endpoint in sorted(
                    self._endpoints.values(), key=lambda item: int(item["gpu"])
                )
            ],
            "workload": copy.deepcopy(self._workload),
        }
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, sort_keys=True)
            handle.write("\n")
            handle.flush()
        os.replace(temporary, self.status_path)

    def _watch_control(self) -> None:
        while not self._stop.wait(self.poll_seconds):
            try:
                with self._condition:
                    self._refresh_locked()
            except BaseException as exc:
                with self._condition:
                    self._watch_error = exc
                    self._condition.notify_all()
                return

    def acquire(self, agent: str) -> Dict[str, Any]:
        with self._condition:
            while True:
                self._raise_watch_error_locked()
                self._refresh_locked()
                candidates = sorted(
                    (
                        endpoint
                        for endpoint in self._endpoints.values()
                        if endpoint["agent"] == agent
                        and endpoint["accepting"]
                        and int(endpoint["active"]) < self.endpoint_capacity
                    ),
                    key=lambda endpoint: str(endpoint["id"]),
                )
                if candidates:
                    minimum = min(int(endpoint["active"]) for endpoint in candidates)
                    least_active = [
                        endpoint
                        for endpoint in candidates
                        if int(endpoint["active"]) == minimum
                    ]
                    index = self._next_index[agent] % len(least_active)
                    endpoint = least_active[index]
                    self._next_index[agent] += 1
                    endpoint["active"] = int(endpoint["active"]) + 1
                    self._write_status_locked()
                    return dict(endpoint)
                if not self._runner_active:
                    raise RuntimeError("live endpoint registry is closed")
                self._condition.wait(timeout=self.poll_seconds)

    def release(self, endpoint_id: str) -> None:
        with self._condition:
            endpoint = self._endpoints.get(endpoint_id)
            if endpoint is None or int(endpoint["active"]) <= 0:
                raise RuntimeError(f"invalid live endpoint release: {endpoint_id}")
            endpoint["active"] = int(endpoint["active"]) - 1
            self._write_status_locked()
            self._condition.notify_all()

    def capacity(self, agent: str) -> int:
        with self._condition:
            self._raise_watch_error_locked()
            self._refresh_locked()
            return self.endpoint_capacity * sum(
                endpoint["agent"] == agent and endpoint["accepting"]
                for endpoint in self._endpoints.values()
            )

    def publish_workload(
        self,
        state: Dict[str, Any],
        active_by_agent: Dict[str, int],
        *,
        advanced: int,
        in_flight: int,
    ) -> None:
        workload = {
            "advanced": advanced,
            "in_flight": in_flight,
            "active_by_agent": dict(active_by_agent),
            "bootstrap_pending": bootstrap_pending_count(state),
            "recovery_pending": recovery_pending_count(state),
            "pending_total": pending_count(state),
            "pending_by_agent": {
                agent: pending_count(state, agent) for agent in AGENT_IDS
            },
            "retry": status_count(state, "retry"),
            "failed": status_count(state, "failed"),
            "done": status_count(state, "done"),
        }
        with self._condition:
            self._raise_watch_error_locked()
            self._refresh_locked()
            self._workload = workload
            self._write_status_locked()

    def close(self) -> None:
        self._stop.set()
        self._watcher.join(timeout=max(1.0, self.poll_seconds * 5))
        with self._condition:
            self._runner_active = False
            self._write_status_locked()
            self._condition.notify_all()


class DynamicOpenAIEndpointCaller:
    """Build callers lazily while endpoint ownership changes at runtime."""

    def __init__(
        self,
        registry: LiveEndpointRegistry,
        agent: str,
        caller_factory: Callable[[str], OpenAIChatLLMCaller],
    ) -> None:
        self.registry = registry
        self.agent = agent
        self.caller_factory = caller_factory
        self._callers: Dict[tuple[str, str], OpenAIChatLLMCaller] = {}
        self._caller_lock = Lock()

    def __call__(self, messages: list[Dict[str, str]]) -> str:
        return self.generate(messages)

    def generate(
        self,
        messages: list[Dict[str, str]],
        *,
        seed: int | None = None,
    ) -> str:
        endpoint = self.registry.acquire(self.agent)
        endpoint_id = str(endpoint["id"])
        base_url = str(endpoint["base_url"])
        try:
            key = (endpoint_id, base_url)
            with self._caller_lock:
                caller = self._callers.get(key)
                if caller is None:
                    caller = self.caller_factory(base_url)
                    self._callers[key] = caller
            return caller.generate(messages, seed=seed)
        finally:
            self.registry.release(endpoint_id)


def parse_api_base_pool(value: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    bases = tuple(part.strip().rstrip("/") for part in value.split(",") if part.strip())
    if not bases:
        if allow_empty:
            return ()
        raise ValueError("API base pool must be non-empty")
    if len(set(bases)) != len(bases):
        raise ValueError("API base pool contains duplicate endpoints")
    return bases


def protocol_max_new_tokens(config: Dict[str, Any]) -> int:
    """Return the bounded completion budget used for protocol turns.

    Bootstrap solving keeps the larger ``max_new_tokens`` budget, but protocol
    responses are schema-bounded short JSON objects.  Applying the protocol
    cap even when thinking is disabled prevents a malformed/degenerate answer
    from consuming the full bootstrap budget and blocking an entire batch.
    """

    configured = int(config.get("max_new_tokens", DEFAULT_PROTOCOL_MAX_TOKENS))
    cap = int(config.get("protocol_max_tokens", DEFAULT_PROTOCOL_MAX_TOKENS))
    if configured <= 0 or cap <= 0:
        raise ValueError("protocol generation token budgets must be positive")
    return min(configured, cap)
_BOOTSTRAP_FIELDS = frozenset(
    {
        "agent",
        "status",
        "system_prompt",
        "thinking",
        "visible_output",
        "final_answer",
        "raw_output",
        "raw_outputs",
        "attempts",
        "error",
    }
)
_TRAJECTORY_FIELDS = frozenset(
    {
        "problem_id",
        "rollout_idx",
        "start_agent",
        "steps",
        "turn_messages",
        "final_answer",
        "terminated_by",
        "error",
    }
)
_STEP_FIELDS = frozenset(
    {
        "turn",
        "active_agent",
        "reasoning",
        "tentative_answer",
        "action",
        "handoff_target",
        "handoff_note",
        "confirmed_answer",
        "raw_output",
        "visible_output",
        "thinking",
        "raw_outputs",
    }
)

_FAILURE_LEDGER_FIELDS = frozenset(
    {
        "version",
        "schema_version",
        "problem_id",
        "problem_index",
        "rollout",
        "rollout_idx",
        "n_steps",
        "problem",
        "start_agent",
        "status",
        "group_attempt",
        "terminated_by",
        "termination",
        "error",
        "generation_attempts",
        "generation_rejections",
        "attempt_diagnostics",
        "last_rejected_raw",
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: str) -> str:
    return value.encode("utf-16", "surrogatepass").decode("utf-16", "replace")


def sanitize_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return normalize_text(value)
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, dict):
        return {
            sanitize_json_value(key): sanitize_json_value(item)
            for key, item in value.items()
        }
    return value


def problem_to_dict(problem: MathProblem) -> Dict[str, Any]:
    return {
        "problem_id": problem.problem_id,
        "subject": problem.subject,
        "level": problem.level,
        "prompt": problem.prompt,
        "solution": problem.solution,
        "gold_answer": problem.gold_answer,
    }


def problem_from_dict(payload: Dict[str, Any]) -> MathProblem:
    return MathProblem(
        problem_id=str(payload["problem_id"]),
        subject=str(payload["subject"]),
        level=str(payload["level"]),
        prompt=str(payload["prompt"]),
        solution=str(payload["solution"]),
        gold_answer=str(payload["gold_answer"]),
    )


def select_start_agent(index: int, configured: str, seed: int) -> str:
    if configured == "balanced":
        return AGENT_IDS[index % len(AGENT_IDS)]
    if configured == "random":
        digest = hashlib.sha256(f"{seed}:{index}".encode()).digest()
        return AGENT_IDS[int.from_bytes(digest[:8], "big") % len(AGENT_IDS)]
    if configured not in AGENT_IDS:
        raise ValueError(f"invalid start agent: {configured}")
    return configured


def select_protocol_start_agent(
    index: int,
    configured: str,
    seed: int,
    bootstrap_agent: Optional[str],
) -> str:
    if bootstrap_agent is not None:
        if bootstrap_agent not in AGENT_IDS:
            raise ValueError(f"invalid bootstrap protocol agent: {bootstrap_agent}")
        return bootstrap_agent
    return select_start_agent(index, configured, seed)


def request_seed(base_seed: int, item: Dict[str, Any], turn: int, attempt: int) -> int:
    text = (
        f"{base_seed}:{item['problem_index']}:{item['rollout_idx']}:{turn}:"
        f"{attempt}:{item.get('group_attempt', 0)}"
    )
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big") & 0x7FFFFFFF


def active_agents(item: Dict[str, Any]) -> list[str]:
    output: list[str] = []
    for step in item["trajectory"]["steps"]:
        agent = str(step["active_agent"])
        if agent not in output:
            output.append(agent)
    return output


def handoff_count(item: Dict[str, Any]) -> int:
    return sum(step.get("action") == "handoff" for step in item["trajectory"]["steps"])


def current_turn(item: Dict[str, Any]) -> int:
    return len(item["trajectory"]["steps"])


def exact_answer_json_literal(answer: str) -> str:
    """Encode backslashes as JSON Unicode escapes for reliable literal copying."""

    return json.dumps(answer, ensure_ascii=True).replace("\\\\", r"\u005c")


def protocol_system_prompt(agent: str, config: Dict[str, Any]) -> str:
    prompt = render_math_mas_system_prompt(
        agent,
        min_agents_before_stop=int(config["min_agents_before_stop"]),
    )
    if agent in config.get("lock_upstream_answer_agents", []):
        return (
            f"{prompt}\n\n"
            "# Immutable Upstream Answer\n"
            "The upstream tentative_answer is immutable for this turn. You MUST copy "
            "it character-for-character into your tentative_answer, and if you choose "
            "confirm_stop, confirmed_answer MUST be that same answer. You may verify or "
            "explain the upstream answer in reasoning, but you MUST NOT correct, rewrite, "
            "normalize, simplify, restyle, or replace it, even with a mathematically "
            "equivalent expression and even if you believe it is wrong."
        )
    if not bool(config.get("preserve_reasonable_incumbent", False)):
        return prompt
    return (
        f"{prompt}\n\n"
        "# Incumbent Answer Preservation\n"
        "Treat the existing tentative_answer as the incumbent. Check it carefully "
        "before changing it. If it remains mathematically reasonable and no specific "
        "error is found, preserve it exactly as the next tentative_answer or confirmed_answer. "
        "Do not change it merely to restyle an equivalent expression, to demonstrate a "
        "contribution, or because a different presentation is possible. Change it only "
        "after identifying a concrete mathematical error, and state that error and the "
        "basis for the correction in reasoning."
    )


def bootstrap_protocol_copy_object(
    protocol_agent: str,
    answer: str,
    handoff_target: Optional[str] = None,
) -> str:
    if protocol_agent not in AGENT_IDS:
        raise ValueError(f"invalid protocol agent: {protocol_agent!r}")
    if handoff_target is None:
        handoff_target = next(agent for agent in AGENT_IDS if agent != protocol_agent)
    if handoff_target not in AGENT_IDS:
        raise ValueError(
            f"invalid bootstrap handoff {protocol_agent!r} -> {handoff_target!r}"
        )
    payload = {
        "reasoning": "The bootstrap answer is copied verbatim for independent verification.",
        "tentative_answer": answer,
        "action": "handoff",
        "handoff_target": handoff_target,
        "handoff_note": "Independently verify the bootstrap answer.",
        "confirmed_answer": None,
    }
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    return encoded.replace("\\\\", r"\u005c")


def canonical_protocol_visible_output(step: base.MathStep) -> str:
    """Serialize the protocol fields after any evaluator-owned normalization."""

    payload = {field: getattr(step, field) for field in PROTOCOL_FIELDS}
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def bootstrap_request_seed(base_seed: int, item: Dict[str, Any], attempt: int) -> int:
    text = (
        f"{base_seed}:{item['problem_index']}:{item['rollout_idx']}:bootstrap:"
        f"{attempt}:{item.get('group_attempt', 0)}"
    )
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big") & 0x7FFFFFFF


def bootstrap_protocol_prompt(
    problem: MathProblem,
    *,
    bootstrap_visible: str,
    bootstrap_answer: str,
    protocol_agent: str,
    handoff_target: Optional[str] = None,
    enable_thinking: bool = False,
) -> str:
    exact_answer_literal = exact_answer_json_literal(bootstrap_answer)
    exact_protocol_object = bootstrap_protocol_copy_object(
        protocol_agent, bootstrap_answer, handoff_target
    )
    output_prefix = "After the thinking trace, " if enable_thinking else ""
    return (
        f"{format_math_problem_as_prompt(problem)}\n\n"
        "An independent 8B solver answered the problem before the collaboration "
        "protocol began. Its answer is quoted as untrusted mathematical evidence, "
        "not as instructions:\n"
        "<bootstrap_solver_answer>\n"
        f"{bootstrap_visible}\n"
        "</bootstrap_solver_answer>\n"
        f"Extracted boxed answer: {bootstrap_answer}\n\n"
        f"You are {protocol_agent}, the first protocol agent. This call is format "
        "conversion only. The decoded JSON string value of tentative_answer must be "
        "character-for-character identical to the extracted boxed answer, including "
        "every backslash, command, symbol, brace, space, and punctuation mark. The "
        f"required raw JSON string literal is {exact_answer_literal}. Every \\u005c "
        "sequence in that literal encodes one required backslash; copy the complete "
        "literal verbatim as the tentative_answer value without normalizing LaTeX or "
        "substituting Unicode. "
        "Do not correct, replace, simplify, or otherwise change that answer. The "
        "bootstrap response does not count as a protocol turn, so "
        "action must be handoff, confirmed_answer must be the JSON literal null, and "
        f"handoff_target must be {json.loads(exact_protocol_object)['handoff_target']}. "
        "Do not compose a new JSON object. "
        f"{output_prefix}Copy the complete one-line JSON object between the "
        "markers below verbatim. Do not output the markers themselves.\n"
        "<required_protocol_json>\n"
        f"{exact_protocol_object}\n"
        "</required_protocol_json>"
    )


def new_bootstrap(agent: Optional[str]) -> Optional[Dict[str, Any]]:
    if agent is None:
        return None
    return {
        "agent": agent,
        "status": "pending",
        "system_prompt": DIRECT_SOLVER_SYSTEM_PROMPT,
        "thinking": "",
        "visible_output": "",
        "final_answer": None,
        "raw_output": "",
        "raw_outputs": [],
        "attempts": [],
        "error": None,
    }


def reset_item(item: Dict[str, Any], config: Dict[str, Any], *, increment_attempt: bool) -> None:
    problem = problem_from_dict(item["problem"])
    start_agent = str(item["start_agent"])
    if increment_attempt:
        item["group_attempt"] = int(item.get("group_attempt", 0)) + 1
    item["status"] = "pending"
    item["current_agent"] = start_agent
    item["prior_tentative"] = False
    item["last_error"] = None
    item["trajectory"] = {
        "problem_id": problem.problem_id,
        "rollout_idx": int(item["rollout_idx"]),
        "start_agent": start_agent,
        "steps": [],
        "turn_messages": [],
        "final_answer": None,
        "terminated_by": "truncated",
        "error": None,
    }
    item["messages"] = [
        {
            "role": "system",
            "content": protocol_system_prompt(start_agent, config),
        },
        {"role": "user", "content": format_math_problem_as_prompt(problem)},
    ]
    item["bootstrap"] = new_bootstrap(config.get("bootstrap_agent"))
    item["pending_recovery"] = None
    item.setdefault("generation_attempts", [])
    item.setdefault("generation_rejections", [])
    item.pop("last_rejected_raw", None)


def create_state(
    problems: list[MathProblem],
    *,
    start: int,
    data_path: str,
    t_max: int,
    start_agent: str,
    start_agent_seed: int,
    min_agents_before_stop: int,
    allow_first_turn_stop: bool,
    num_rollouts: int,
    generation_seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool,
    require_thinking: bool,
    api_timeout: float,
    group_retries: int,
    retry_failed_groups: bool,
    step_retries: int,
    json_transport: str,
    output_mode: str,
    split: str = "train",
    subjects: Optional[list[str]] = None,
    bootstrap_agent: Optional[str] = None,
    protocol_thinking_max_tokens: Optional[int] = None,
    protocol_max_tokens: Optional[int] = None,
    bootstrap_handoff_target: Optional[str] = None,
    preserve_reasonable_incumbent: bool = False,
    lock_upstream_answer_agents: Optional[list[str]] = None,
) -> Dict[str, Any]:
    normalized_subjects = sorted({str(value).strip() for value in (subjects or []) if str(value).strip()})
    requested_locked_agents = list(lock_upstream_answer_agents or [])
    if (
        any(agent not in AGENT_IDS for agent in requested_locked_agents)
        or len(set(requested_locked_agents)) != len(requested_locked_agents)
    ):
        raise ValueError(
            f"lock_upstream_answer_agents must contain unique agent IDs: "
            f"{requested_locked_agents!r}"
        )
    normalized_locked_agents = [
        agent for agent in AGENT_IDS if agent in requested_locked_agents
    ]
    if split not in {"train", "test"}:
        raise ValueError(f"invalid split: {split}")
    config = {
        "data_path": data_path,
        "split": split,
        "subjects": normalized_subjects,
        "start": start,
        "limit": len(problems),
        "t_max": t_max,
        "start_agent": start_agent,
        "start_agent_seed": start_agent_seed,
        "min_agents_before_stop": min_agents_before_stop,
        "allow_first_turn_stop": allow_first_turn_stop,
        "num_rollouts": num_rollouts,
        "generation_seed": generation_seed,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "enable_thinking": enable_thinking,
        "require_thinking": require_thinking,
        "api_timeout": api_timeout,
        "group_retries": group_retries,
        "retry_failed_groups": retry_failed_groups,
        "step_retries": step_retries,
        "json_transport": json_transport,
        "output_mode": output_mode,
        "bootstrap_agent": bootstrap_agent,
        "bootstrap_handoff_target": bootstrap_handoff_target,
        "preserve_reasonable_incumbent": preserve_reasonable_incumbent,
        "lock_upstream_answer_agents": normalized_locked_agents,
        "protocol_thinking_max_tokens": (
            max_new_tokens
            if protocol_thinking_max_tokens is None
            else protocol_thinking_max_tokens
        ),
        "protocol_max_tokens": (
            DEFAULT_PROTOCOL_MAX_TOKENS
            if protocol_max_tokens is None
            else protocol_max_tokens
        ),
    }
    state = {
        "version": STATE_VERSION,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "phase_count": 0,
        "config": config,
        "history": [],
        "items": [],
    }
    for offset, problem in enumerate(problems):
        index = start + offset
        for rollout_idx in range(num_rollouts):
            first = select_protocol_start_agent(
                index,
                start_agent,
                start_agent_seed,
                bootstrap_agent,
            )
            item = {
                "problem_index": index,
                "rollout_idx": rollout_idx,
                "problem": problem_to_dict(problem),
                "start_agent": first,
                "current_agent": first,
                "group_attempt": 0,
                "prior_tentative": False,
                "status": "pending",
                "last_error": None,
                "trajectory": {},
                "messages": [],
                "generation_attempts": [],
                "generation_rejections": [],
                "bootstrap": None,
                "pending_recovery": None,
            }
            reset_item(item, config, increment_attempt=False)
            state["items"].append(item)
    return state


def state_journal_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.journal.jsonl")


def _strict_state_json_loads(raw: bytes, *, source: Path, line_no: int) -> Any:
    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-standard JSON constant {value}")

    def reject_duplicate_keys(pairs: Any) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON in {source} at line {line_no}: {exc}") from exc


def apply_state_journal(path: Path, state: Dict[str, Any]) -> int:
    """Replay completed item updates left by an interrupted agent phase."""
    journal = state_journal_path(path)
    if not journal.exists():
        return 0
    size = journal.stat().st_size
    applied = 0
    truncate_at: Optional[int] = None
    journal_line_no = 0
    with journal.open("rb") as handle:
        while True:
            record_start = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            journal_line_no += 1
            at_eof = handle.tell() == size
            # Every durable record is newline terminated. A SIGTERM may leave
            # a syntactically valid JSON object without its final newline; do
            # not replay that uncommitted record, otherwise a later append
            # would concatenate two objects and poison the journal.
            if at_eof and not raw.endswith(b"\n"):
                truncate_at = record_start
                print(
                    "[resume] discarding unterminated final journal record",
                    file=sys.stderr,
                )
                break
            try:
                record = _strict_state_json_loads(
                    raw, source=journal, line_no=journal_line_no
                )
            except ValueError as exc:
                # SIGTERM can interrupt the final append. All earlier records
                # are complete newline-delimited JSON objects and remain safe.
                if at_eof:
                    truncate_at = record_start
                    print(f"[resume] ignoring truncated final journal record: {exc}", file=sys.stderr)
                    break
                raise ValueError(f"corrupt state journal before EOF: {journal}") from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"invalid state journal record in {journal} at line {journal_line_no}"
                )
            if record.get("version") != JOURNAL_VERSION:
                raise ValueError(f"unsupported state journal version: {record.get('version')}")
            if not _is_int(record.get("index")):
                raise ValueError(f"invalid journal item index at record {journal_line_no}")
            index = int(record["index"])
            if not 0 <= index < len(state["items"]):
                raise ValueError(f"journal item index out of range: {index}")
            item = state["items"][index]
            if (
                not _is_int(record.get("problem_index"))
                or not _is_int(record.get("rollout_idx"))
                or int(record["problem_index"]) != int(item["problem_index"])
                or int(record["rollout_idx"]) != int(item["rollout_idx"])
            ):
                raise ValueError(f"journal identity mismatch at item {index}")
            update = record.get("update")
            optional_fields = _OPTIONAL_ITEM_FIELDS
            required_fields = set(JOURNAL_ITEM_FIELDS) - optional_fields
            update_fields = set(update) if isinstance(update, dict) else set()
            if (
                not isinstance(update, dict)
                or not required_fields.issubset(update_fields)
                or not update_fields.issubset(set(JOURNAL_ITEM_FIELDS))
            ):
                raise ValueError(f"invalid journal update at item {index}")
            # Journals written before retry diagnostics were introduced omit
            # those fields.  Materialize the same defaults used by state
            # normalization so an old interrupted run remains resumable.
            for field in optional_fields:
                update.setdefault(
                    field,
                    item.get(
                        field,
                        [] if field in {"generation_attempts", "generation_rejections"} else None,
                    ),
                )
            item.update(update)
            applied += 1
    if truncate_at is not None:
        # Repair the tail before StateJournal opens the file in append mode.
        # Keeping only complete records makes repeated stop/resume cycles safe.
        with journal.open("r+b") as handle:
            handle.truncate(truncate_at)
            handle.flush()
            os.fsync(handle.fileno())
    if applied:
        print(f"[resume] replayed {applied} completed item updates from {journal}", file=sys.stderr)
    return applied


class StateJournal:
    """Durably record phase progress without rewriting the very large state file."""

    def __init__(self, state_path: Path, fsync_every: int) -> None:
        self.path = state_journal_path(state_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a", encoding="utf-8", errors="replace", buffering=1)
        self.fsync_every = max(1, fsync_every)
        self.count = 0

    def append(self, index: int, item: Dict[str, Any]) -> None:
        defaults = {
            "generation_attempts": [],
            "generation_rejections": [],
            "last_rejected_raw": None,
            "bootstrap": None,
            "pending_recovery": None,
        }
        record = {
            "version": JOURNAL_VERSION,
            "index": index,
            "problem_index": int(item["problem_index"]),
            "rollout_idx": int(item["rollout_idx"]),
            # Accept hand-built/legacy state items that predate the audit
            # fields; ``load_state`` will materialize the same defaults.
            "update": {
                field: item.get(field, defaults.get(field))
                for field in JOURNAL_ITEM_FIELDS
            },
        }
        self.handle.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n")
        self.count += 1
        if self.count % self.fsync_every == 0:
            self.handle.flush()
            os.fsync(self.handle.fileno())

    def close(self) -> None:
        if self.handle.closed:
            return
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()

    def __enter__(self) -> "StateJournal":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def save_state(path: Path, state: Dict[str, Any]) -> None:
    # Never make an invalid state durable.  This is intentionally performed
    # before creating the temporary file so a failed validation leaves the
    # previous checkpoint untouched and the caller can inspect the exception.
    _validate_state(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    # The full state now contains every replayed update. Removing the journal
    # after os.replace makes compaction crash-safe (replaying it twice is also
    # harmless if interruption happens between these two operations).
    state_journal_path(path).unlink(missing_ok=True)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_message_list(value: Any, location: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be a list")
    for index, message in enumerate(value):
        if not isinstance(message, dict):
            raise ValueError(f"{location}[{index}] must be an object")
        if not isinstance(message.get("role"), str) or not message["role"].strip():
            raise ValueError(f"{location}[{index}] has an invalid role")
        if not isinstance(message.get("content"), str):
            raise ValueError(f"{location}[{index}] has non-string content")


def _validate_attempt_records(value: Any, location: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be a list")
    for index, record in enumerate(value):
        if not isinstance(record, dict):
            raise ValueError(f"{location}[{index}] must be an object")
        for field in ("turn", "agent", "attempt", "accepted", "reason", "raw_output", "raw_outputs", "visible_raw_output"):
            if field not in record:
                raise ValueError(f"{location}[{index}] missing field: {field}")
        if not _is_int(record["turn"]) or int(record["turn"]) < 0:
            raise ValueError(f"{location}[{index}] has invalid turn")
        if not isinstance(record["agent"], str) or record["agent"] not in AGENT_IDS:
            raise ValueError(f"{location}[{index}] has invalid agent")
        if not _is_int(record["attempt"]) or int(record["attempt"]) <= 0:
            raise ValueError(f"{location}[{index}] has invalid attempt")
        if not isinstance(record["accepted"], bool):
            raise ValueError(f"{location}[{index}].accepted must be boolean")
        if record["reason"] is not None and not isinstance(record["reason"], str):
            raise ValueError(f"{location}[{index}].reason must be string or null")
        if not isinstance(record["raw_output"], str) or not isinstance(record["visible_raw_output"], str):
            raise ValueError(f"{location}[{index}] raw output fields must be strings")
        if not isinstance(record["raw_outputs"], list) or not all(
            isinstance(raw, str) for raw in record["raw_outputs"]
        ):
            raise ValueError(f"{location}[{index}].raw_outputs must be a list of strings")


def _normalize_legacy_state(state: Dict[str, Any]) -> None:
    """Fill fields that were absent in the pre-journal role runner.

    This deliberately handles only fields with an unambiguous historical
    default.  Missing trajectory content or identity fields remain errors and
    are rejected by ``_validate_state`` instead of being guessed.
    """
    config = state["config"]
    for key, default in _CONFIG_DEFAULTS.items():
        config.setdefault(key, copy.deepcopy(default))
    config.setdefault("require_thinking", bool(config.get("enable_thinking", False)))
    config.setdefault("group_retries", 0)
    config.setdefault("step_retries", 0)
    config.setdefault("max_new_tokens", 1)
    config.setdefault(
        "protocol_thinking_max_tokens", int(config.get("max_new_tokens", 1))
    )
    config.setdefault(
        "protocol_max_tokens",
        min(
            int(config.get("protocol_thinking_max_tokens", config.get("max_new_tokens", 1))),
            DEFAULT_PROTOCOL_MAX_TOKENS,
        ),
    )
    config.setdefault("temperature", 0.0)
    config.setdefault("top_p", 1.0)
    config.setdefault("api_timeout", 1.0)
    config.setdefault("min_agents_before_stop", 1)
    config.setdefault("num_rollouts", 1)
    config.setdefault("start", 0)
    config.setdefault("limit", len(state.get("items", [])))
    config.setdefault("generation_seed", 42)
    for item in state.get("items", []):
        if not isinstance(item, dict):
            continue
        if "start_agent" not in item:
            problem_index = item.get("problem_index")
            if _is_int(problem_index):
                item["start_agent"] = select_start_agent(
                    int(problem_index),
                    str(config.get("start_agent", "A1")),
                    int(config.get("start_agent_seed", 42)),
                )
        item.setdefault("group_attempt", 0)
        item.setdefault("last_error", None)
        item.setdefault("generation_attempts", [])
        item.setdefault("generation_rejections", [])
        item.setdefault("last_rejected_raw", None)
        item.setdefault("bootstrap", None)
        item.setdefault("pending_recovery", None)
        trajectory = item.get("trajectory")
        if not isinstance(trajectory, dict):
            continue
        if "rollout_idx" not in trajectory and _is_int(item.get("rollout_idx")):
            trajectory["rollout_idx"] = int(item["rollout_idx"])
        if "start_agent" not in trajectory and item.get("start_agent") in AGENT_IDS:
            trajectory["start_agent"] = item["start_agent"]
        trajectory.setdefault("turn_messages", [])
        trajectory.setdefault("final_answer", None)
        trajectory.setdefault("terminated_by", "truncated")
        trajectory.setdefault("error", item.get("last_error"))


def _validate_bootstrap(item: Dict[str, Any], item_index: int, config: Dict[str, Any]) -> None:
    expected_agent = config.get("bootstrap_agent")
    bootstrap = item.get("bootstrap")
    location = f"item {item_index}.bootstrap"
    if expected_agent is None:
        if bootstrap is not None:
            raise ValueError(f"{location} must be null when bootstrap is disabled")
        return
    if expected_agent not in AGENT_IDS:
        raise ValueError(f"invalid bootstrap agent: {expected_agent!r}")
    if not isinstance(bootstrap, dict) or set(bootstrap) != _BOOTSTRAP_FIELDS:
        raise ValueError(f"{location} has invalid fields")
    if bootstrap["agent"] != expected_agent:
        raise ValueError(f"{location} agent disagrees with config")
    if bootstrap["status"] not in {"pending", "done", "failed"}:
        raise ValueError(f"{location} has invalid status")
    if bootstrap["system_prompt"] != DIRECT_SOLVER_SYSTEM_PROMPT:
        raise ValueError(f"{location} system prompt disagrees with direct-solver contract")
    for field in ("thinking", "visible_output", "raw_output"):
        if not isinstance(bootstrap[field], str):
            raise ValueError(f"{location}.{field} must be a string")
    if not isinstance(bootstrap["raw_outputs"], list) or not all(
        isinstance(value, str) for value in bootstrap["raw_outputs"]
    ):
        raise ValueError(f"{location}.raw_outputs must be a list of strings")
    if not isinstance(bootstrap["attempts"], list):
        raise ValueError(f"{location}.attempts must be a list")
    for attempt_index, attempt in enumerate(bootstrap["attempts"]):
        attempt_location = f"{location}.attempts[{attempt_index}]"
        if not isinstance(attempt, dict) or set(attempt) != {
            "attempt", "accepted", "reason", "raw_output", "raw_outputs", "visible_raw_output"
        }:
            raise ValueError(f"{attempt_location} has invalid fields")
        if not _is_int(attempt["attempt"]) or int(attempt["attempt"]) != attempt_index + 1:
            raise ValueError(f"{attempt_location} has invalid attempt number")
        if not isinstance(attempt["accepted"], bool):
            raise ValueError(f"{attempt_location}.accepted must be boolean")
        if attempt["reason"] is not None and not isinstance(attempt["reason"], str):
            raise ValueError(f"{attempt_location}.reason must be string or null")
        for field in ("raw_output", "visible_raw_output"):
            if not isinstance(attempt[field], str):
                raise ValueError(f"{attempt_location}.{field} must be a string")
        if not isinstance(attempt["raw_outputs"], list) or not all(
            isinstance(value, str) for value in attempt["raw_outputs"]
        ):
            raise ValueError(f"{attempt_location}.raw_outputs must be a list of strings")
    final_answer = bootstrap["final_answer"]
    if final_answer is not None and not isinstance(final_answer, str):
        raise ValueError(f"{location}.final_answer must be string or null")
    error = bootstrap["error"]
    if error is not None and not isinstance(error, str):
        raise ValueError(f"{location}.error must be string or null")
    if bootstrap["status"] == "pending":
        if any((bootstrap["thinking"], bootstrap["visible_output"], bootstrap["raw_output"])):
            raise ValueError(f"{location} pending record contains generated output")
        if final_answer is not None or error is not None or bootstrap["attempts"]:
            raise ValueError(f"{location} pending record contains terminal fields")
    elif bootstrap["status"] == "done":
        if not bootstrap["visible_output"].strip() or not str(final_answer or "").strip():
            raise ValueError(f"{location} done record lacks answer text")
        if bool(config.get("require_thinking")) and not bootstrap["thinking"].strip():
            raise ValueError(f"{location} lacks required thinking")
        if extract_last_boxed(bootstrap["visible_output"]) != final_answer:
            raise ValueError(f"{location} boxed answer disagrees")
        if error is not None or not bootstrap["attempts"] or not bootstrap["attempts"][-1]["accepted"]:
            raise ValueError(f"{location} done record has inconsistent attempts")
    else:
        if not isinstance(error, str) or not error.strip() or final_answer is not None:
            raise ValueError(f"{location} failed record has inconsistent terminal fields")
        if item.get("status") not in {"done", "failed"}:
            raise ValueError(f"{location} failed while item remains active")


def _validate_step(
    step: Any,
    *,
    item_index: int,
    step_index: int,
    config: Dict[str, Any],
    expected_agent: str,
    seen_agents: list[str],
) -> str:
    location = f"item {item_index} trajectory.steps[{step_index}]"
    if not isinstance(step, dict):
        raise ValueError(f"{location} must be an object")
    missing = _STEP_FIELDS - set(step)
    if missing:
        raise ValueError(f"{location} missing fields: {sorted(missing)}")
    unknown = set(step) - _STEP_FIELDS
    if unknown:
        raise ValueError(f"{location} has unknown fields: {sorted(unknown)}")
    if not _is_int(step["turn"]) or int(step["turn"]) != step_index:
        raise ValueError(f"{location} has non-contiguous turn")
    if step["active_agent"] != expected_agent or expected_agent not in AGENT_IDS:
        raise ValueError(f"{location} has unexpected active_agent: {step.get('active_agent')}")
    for field in ("reasoning", "tentative_answer", "raw_output", "visible_output", "thinking"):
        if not isinstance(step[field], str):
            raise ValueError(f"{location}.{field} must be a string")
    if not step["reasoning"].strip() or not step["tentative_answer"].strip():
        raise ValueError(f"{location} has an empty reasoning or tentative_answer")
    if not bool(config.get("enable_thinking", False)) and step["thinking"].strip():
        raise ValueError(f"{location} contains thinking while thinking is disabled")
    if bool(config.get("require_thinking", False)) and not step["thinking"].strip():
        raise ValueError(f"{location} is missing required thinking")
    if not isinstance(step["raw_outputs"], list) or not all(
        isinstance(value, str) for value in step["raw_outputs"]
    ):
        raise ValueError(f"{location}.raw_outputs must be a list of strings")
    for field in ("handoff_target", "handoff_note", "confirmed_answer"):
        value = step[field]
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{location}.{field} must be a string or null")
    action = step["action"]
    if action not in {"handoff", "confirm_stop"}:
        raise ValueError(f"{location} has invalid action: {action!r}")
    allow_bootstrap_self_handoff = False
    if action == "handoff":
        target = step["handoff_target"]
        allow_bootstrap_self_handoff = (
            step_index == 0
            and target == step["active_agent"]
            and target == config.get("bootstrap_agent")
            and target == config.get("bootstrap_handoff_target")
        )
        if target not in AGENT_IDS or (
            target == step["active_agent"] and not allow_bootstrap_self_handoff
        ):
            raise ValueError(f"{location} has invalid handoff_target: {target!r}")
        if step["confirmed_answer"] is not None:
            raise ValueError(f"{location} handoff has confirmed_answer")
        next_agent = str(target)
    else:
        if step["handoff_target"] is not None:
            raise ValueError(f"{location} confirm_stop has handoff_target")
        if not str(step["confirmed_answer"] or "").strip():
            raise ValueError(f"{location} confirm_stop has no confirmed_answer")
        if not bool(config.get("allow_first_turn_stop", False)) and not seen_agents:
            raise ValueError(f"{location} confirm_stop is not allowed on the first turn")
        distinct_agents = set(seen_agents)
        distinct_agents.add(step["active_agent"])
        if len(distinct_agents) < int(config["min_agents_before_stop"]):
            raise ValueError(f"{location} confirm_stop is before the minimum agent count")
        if not math_answers_equivalent(
            str(step["confirmed_answer"]), str(step["tentative_answer"])
        ):
            raise ValueError(f"{location} tentative/confirmed answers disagree")
        next_agent = ""
    if "visible_output" in step:
        visible_output = step["visible_output"]
        if re.search(r"</?think\b", visible_output, re.IGNORECASE):
            raise ValueError(f"{location} visible_output contains thinking tags")
        try:
            parsed = validate_protocol_object(
                visible_output,
                active_agent=step["active_agent"],
                allow_self_handoff=allow_bootstrap_self_handoff,
            )
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"{location} visible_output is not valid protocol JSON") from exc
        for field in (
            "action",
            "handoff_target",
            "handoff_note",
            "confirmed_answer",
        ):
            if parsed.get(field) != step.get(field):
                raise ValueError(f"{location} visible_output disagrees on {field}")
        for field in ("reasoning", "tentative_answer"):
            if str(parsed.get(field) or "").strip() != str(step.get(field) or "").strip():
                raise ValueError(f"{location} visible_output disagrees on {field}")
    raw_output = step.get("raw_output", "")
    raw_thinking = list(re.finditer(r"<think\b[^>]*>(.*?)</think\s*>", raw_output, re.I | re.S))
    if not bool(config.get("enable_thinking", False)) and any(
        match.group(1).strip() for match in raw_thinking
    ):
        raise ValueError(f"{location} raw_output contains a non-empty thinking trace")
    if step["active_agent"] not in seen_agents:
        seen_agents.append(step["active_agent"])
    return next_agent


def _is_audited_degenerate_recovery(
    item: Dict[str, Any],
    *,
    turn: int,
    scheduled_agent: str,
    actual_agent: str,
) -> bool:
    if actual_agent != DEGENERATE_RECOVERY_AGENT or scheduled_agent == actual_agent:
        return False
    rejected = any(
        int(record.get("turn", -1)) == turn
        and record.get("agent") == scheduled_agent
        and str(record.get("reason") or "").startswith("degenerate repetition")
        for record in item.get("generation_rejections", [])
    )
    recovered = any(
        int(record.get("turn", -1)) == turn
        and record.get("agent") == actual_agent
        and record.get("accepted") is True
        for record in item.get("generation_attempts", [])
    )
    return rejected and recovered


def _validate_item(item: Any, item_index: int, state: Dict[str, Any]) -> tuple[int, int]:
    config = state["config"]
    if not isinstance(item, dict):
        raise ValueError(f"item {item_index} must be an object")
    missing = _ITEM_FIELDS - set(item)
    if missing:
        raise ValueError(f"item {item_index} missing fields: {sorted(missing)}")
    unknown = set(item) - _ITEM_FIELDS - _OPTIONAL_ITEM_FIELDS
    # ``generation_attempts`` and similar diagnostics appeared in some GSM
    # states, but MATH states have no supported extra item fields.  Rejecting
    # them catches accidental cross-run state mixing early.
    if unknown:
        raise ValueError(f"item {item_index} has unknown fields: {sorted(unknown)}")

    problem_index = item["problem_index"]
    rollout_idx = item["rollout_idx"]
    if not _is_int(problem_index) or not _is_int(rollout_idx):
        raise ValueError(f"item {item_index} has non-integer identity")
    if int(rollout_idx) < 0 or int(rollout_idx) >= int(config["num_rollouts"]):
        raise ValueError(f"item {item_index} rollout_idx is out of range")
    problem = item["problem"]
    if not isinstance(problem, dict):
        raise ValueError(f"item {item_index}.problem must be an object")
    if not _PROBLEM_FIELDS.issubset(problem):
        raise ValueError(
            f"item {item_index}.problem missing fields: {sorted(_PROBLEM_FIELDS - set(problem))}"
        )
    for field in _PROBLEM_FIELDS:
        if not isinstance(problem[field], str):
            raise ValueError(f"item {item_index}.problem.{field} must be a string")
    if not problem["problem_id"].strip():
        raise ValueError(f"item {item_index} has an empty problem_id")

    start_agent = item["start_agent"]
    if start_agent not in AGENT_IDS:
        raise ValueError(f"item {item_index} has invalid start_agent: {start_agent!r}")
    if config.get("bootstrap_agent") is not None and start_agent != config["bootstrap_agent"]:
        raise ValueError(
            f"item {item_index} bootstrap protocol must start with "
            f"{config['bootstrap_agent']}"
        )
    status = item["status"]
    if status not in _STATE_STATUSES:
        raise ValueError(f"item {item_index} has invalid status: {status!r}")
    current_agent = item["current_agent"]
    if status == "pending":
        if current_agent not in AGENT_IDS:
            raise ValueError(f"pending item {item_index} has invalid current_agent")
    elif current_agent is not None:
        raise ValueError(f"terminal item {item_index} must not have current_agent")
    if not _is_int(item["group_attempt"]) or int(item["group_attempt"]) < 0:
        raise ValueError(f"item {item_index} has invalid group_attempt")
    if int(item["group_attempt"]) > int(config["group_retries"]):
        raise ValueError(f"item {item_index} group_attempt exceeds group_retries")
    if not isinstance(item["prior_tentative"], bool):
        raise ValueError(f"item {item_index}.prior_tentative must be boolean")
    if item["last_error"] is not None and not isinstance(item["last_error"], str):
        raise ValueError(f"item {item_index}.last_error must be string or null")
    _validate_attempt_records(
        item.get("generation_attempts", []), f"item {item_index}.generation_attempts"
    )
    _validate_attempt_records(
        item.get("generation_rejections", []), f"item {item_index}.generation_rejections"
    )
    if item.get("last_rejected_raw") is not None and not isinstance(item["last_rejected_raw"], str):
        raise ValueError(f"item {item_index}.last_rejected_raw must be string or null")
    pending_recovery = item.get("pending_recovery")
    if pending_recovery is not None:
        location = f"item {item_index}.pending_recovery"
        if (
            not isinstance(pending_recovery, dict)
            or set(pending_recovery) != _PENDING_RECOVERY_FIELDS
        ):
            raise ValueError(f"{location} has invalid fields")
        if pending_recovery["collapsed_agent"] not in AGENT_IDS:
            raise ValueError(f"{location} has invalid collapsed_agent")
        if (
            not _is_int(pending_recovery["turn"])
            or int(pending_recovery["turn"]) < 0
        ):
            raise ValueError(f"{location} has invalid turn")
        if (
            not isinstance(pending_recovery["reason"], str)
            or not pending_recovery["reason"].startswith("degenerate repetition")
        ):
            raise ValueError(f"{location} has invalid reason")
        exact_answer = pending_recovery["exact_tentative_answer"]
        if exact_answer is not None and (
            not isinstance(exact_answer, str) or not exact_answer.strip()
        ):
            raise ValueError(f"{location} has invalid exact_tentative_answer")
        if not isinstance(pending_recovery["retry_failed_group"], bool):
            raise ValueError(f"{location}.retry_failed_group must be boolean")
    _validate_bootstrap(item, item_index, config)
    _validate_message_list(item["messages"], f"item {item_index}.messages")
    if len(item["messages"]) < 2:
        raise ValueError(f"item {item_index}.messages must contain system and user prompts")
    if item["messages"][0]["role"] != "system" or item["messages"][1]["role"] != "user":
        raise ValueError(f"item {item_index}.messages must start with system/user")

    trajectory = item["trajectory"]
    if not isinstance(trajectory, dict):
        raise ValueError(f"item {item_index}.trajectory must be an object")
    missing = _TRAJECTORY_FIELDS - set(trajectory)
    if missing:
        raise ValueError(f"item {item_index}.trajectory missing fields: {sorted(missing)}")
    unknown = set(trajectory) - _TRAJECTORY_FIELDS
    if unknown:
        raise ValueError(f"item {item_index}.trajectory has unknown fields: {sorted(unknown)}")
    if trajectory["problem_id"] != problem["problem_id"]:
        raise ValueError(f"item {item_index} trajectory/problem identity mismatch")
    if not _is_int(trajectory["rollout_idx"]) or int(trajectory["rollout_idx"]) != int(rollout_idx):
        raise ValueError(f"item {item_index} trajectory rollout identity mismatch")
    if trajectory["start_agent"] != start_agent:
        raise ValueError(f"item {item_index} trajectory start-agent mismatch")
    if trajectory["terminated_by"] not in _TERMINAL_REASONS:
        raise ValueError(
            f"item {item_index} has invalid terminated_by: {trajectory['terminated_by']!r}"
        )
    if trajectory["error"] is not None and not isinstance(trajectory["error"], str):
        raise ValueError(f"item {item_index}.trajectory.error must be string or null")
    if trajectory["final_answer"] is not None and not isinstance(trajectory["final_answer"], str):
        raise ValueError(f"item {item_index}.trajectory.final_answer must be string or null")
    steps = trajectory["steps"]
    if not isinstance(steps, list):
        raise ValueError(f"item {item_index}.trajectory.steps must be a list")
    if len(steps) > int(config["t_max"]):
        raise ValueError(f"item {item_index} exceeds t_max")
    turn_messages = trajectory["turn_messages"]
    if not isinstance(turn_messages, list) or len(turn_messages) != len(steps):
        raise ValueError(f"item {item_index} turn_messages/steps length mismatch")
    for turn_index, messages in enumerate(turn_messages):
        _validate_message_list(messages, f"item {item_index}.turn_messages[{turn_index}]")

    expected_agent = start_agent
    seen_agents: list[str] = []
    for step_index, step in enumerate(steps):
        step_expected_agent = expected_agent
        if step.get("active_agent") != expected_agent:
            if not _is_audited_degenerate_recovery(
                item,
                turn=step_index,
                scheduled_agent=expected_agent,
                actual_agent=str(step.get("active_agent")),
            ):
                raise ValueError(
                    f"item {item_index}.trajectory.steps[{step_index}] has "
                    f"unexpected active_agent: {step.get('active_agent')}"
                )
            step_expected_agent = str(step["active_agent"])
        next_agent = _validate_step(
            step,
            item_index=item_index,
            step_index=step_index,
            config=config,
            expected_agent=step_expected_agent,
            seen_agents=seen_agents,
        )
        if step_index + 1 < len(steps):
            if not next_agent:
                raise ValueError(f"item {item_index} has a step after confirm_stop")
            expected_agent = next_agent
        elif next_agent:
            # A handoff at the end of a pending trajectory must be reflected
            # in current_agent.  Terminal truncation/rejection clears the
            # field below, so this value is only used for pending items.
            expected_agent = next_agent
    if bool(steps) != bool(item["prior_tentative"]):
        raise ValueError(f"item {item_index}.prior_tentative disagrees with steps")
    if len(item["messages"]) != len(steps) + 2:
        raise ValueError(f"item {item_index}.messages/steps length mismatch")
    reason = trajectory["terminated_by"]
    if pending_recovery is not None:
        collapsed_agent = str(pending_recovery["collapsed_agent"])
        recovery_turn = int(pending_recovery["turn"])
        if status != "pending" or item["current_agent"] != collapsed_agent:
            raise ValueError(
                f"item {item_index}.pending_recovery disagrees with active item state"
            )
        if recovery_turn != len(steps):
            raise ValueError(f"item {item_index}.pending_recovery turn is not current")
        bootstrap = item.get("bootstrap")
        if isinstance(bootstrap, dict) and bootstrap.get("status") != "done":
            raise ValueError(
                f"item {item_index}.pending_recovery exists before bootstrap completion"
            )
        if not any(
            int(record.get("turn", -1)) == recovery_turn
            and record.get("agent") == collapsed_agent
            and str(record.get("reason") or "").startswith("degenerate repetition")
            for record in item.get("generation_rejections", [])
        ):
            raise ValueError(
                f"item {item_index}.pending_recovery lacks collapse audit record"
            )
    if status == "pending":
        bootstrap = item.get("bootstrap")
        if isinstance(bootstrap, dict) and bootstrap.get("status") == "pending" and steps:
            raise ValueError(f"pending item {item_index} has protocol steps before bootstrap")
        if reason != "truncated" or trajectory["final_answer"] is not None:
            raise ValueError(f"pending item {item_index} has terminal trajectory fields")
        if len(steps) >= int(config["t_max"]):
            raise ValueError(f"pending item {item_index} reached t_max")
        if item["current_agent"] != expected_agent:
            raise ValueError(
                f"pending item {item_index} current_agent does not match the last handoff"
            )
        if trajectory.get("error") is not None or item.get("last_error") is not None:
            raise ValueError(f"pending item {item_index} retains a terminal error")
    elif reason == "stop":
        if status != "done" or not steps or steps[-1]["action"] != "confirm_stop":
            raise ValueError(f"item {item_index} stop state is inconsistent")
        if not str(trajectory["final_answer"] or "").strip():
            raise ValueError(f"item {item_index} stop state has no final answer")
        confirmed = steps[-1].get("confirmed_answer")
        tentative = steps[-1].get("tentative_answer")
        if not math_answers_equivalent(str(trajectory["final_answer"]), str(confirmed)):
            raise ValueError(f"item {item_index} final answer disagrees with confirmation")
        if not math_answers_equivalent(str(trajectory["final_answer"]), str(tentative)):
            raise ValueError(f"item {item_index} final answer disagrees with tentative answer")
    elif status not in {"done", "retry", "failed"}:
        raise ValueError(f"item {item_index} non-stop trajectory has invalid status")
    if reason == "truncated" and status != "pending":
        if len(steps) != int(config["t_max"]):
            raise ValueError(f"item {item_index} truncated before t_max")
        tentative = steps[-1].get("tentative_answer") if steps else None
        if not isinstance(tentative, str) or not tentative.strip():
            raise ValueError(f"item {item_index} truncated without a tentative answer")
        if trajectory["final_answer"] != tentative:
            raise ValueError(
                f"item {item_index} truncated final answer is not the last tentative answer"
            )
    elif reason in {"rejected_quality", "exception"} and trajectory["final_answer"] is not None:
        raise ValueError(f"item {item_index} failed trajectory has a final answer")
    if status in {"retry", "failed"} and not bool(config.get("retry_failed_groups", False)):
        raise ValueError(f"item {item_index} retries present while group retry is disabled")
    if status == "retry" and int(item["group_attempt"]) >= int(config["group_retries"]):
        raise ValueError(f"item {item_index} retry is already exhausted")
    if status == "failed" and int(item["group_attempt"]) < int(config["group_retries"]):
        raise ValueError(f"item {item_index} failed before exhausting group retries")
    return int(problem_index), int(rollout_idx)


def _validate_state(state: Dict[str, Any]) -> None:
    """Validate a MATH role-batched state before it is resumed or saved."""
    if not isinstance(state, dict):
        raise ValueError("MATH role-batched state must be an object")
    if state.get("version") != STATE_VERSION:
        raise ValueError(
            f"unsupported state version {state.get('version')}; expected {STATE_VERSION}"
        )
    config = state.get("config")
    items = state.get("items")
    if not isinstance(config, dict) or not isinstance(items, list):
        raise ValueError("invalid MATH role-batched state structure")
    required_config = {
        "data_path", "split", "subjects", "start", "limit", "t_max", "start_agent", "start_agent_seed",
        "min_agents_before_stop", "allow_first_turn_stop", "num_rollouts",
        "generation_seed", "max_new_tokens", "protocol_thinking_max_tokens", "protocol_max_tokens", "temperature", "top_p",
        "enable_thinking", "require_thinking", "api_timeout", "group_retries",
        "retry_failed_groups", "step_retries", "json_transport", "output_mode",
        "bootstrap_agent",
        "bootstrap_handoff_target", "preserve_reasonable_incumbent",
        "lock_upstream_answer_agents",
    }
    missing = required_config - set(config)
    if missing:
        raise ValueError(f"state config missing fields: {sorted(missing)}")
    if not isinstance(config["data_path"], str) or not config["data_path"].strip():
        raise ValueError("state config data_path must be a non-empty string")
    if config["split"] not in {"train", "test"}:
        raise ValueError(f"invalid state config split: {config['split']!r}")
    if not isinstance(config["subjects"], list) or any(
        not isinstance(subject, str) or not subject.strip() for subject in config["subjects"]
    ) or len(set(config["subjects"])) != len(config["subjects"]):
        raise ValueError("state config subjects must be a list of unique non-empty strings")
    for field in ("start", "limit", "t_max", "start_agent_seed", "min_agents_before_stop", "num_rollouts", "max_new_tokens", "protocol_thinking_max_tokens", "protocol_max_tokens", "group_retries", "step_retries"):
        if not _is_int(config[field]):
            raise ValueError(f"state config {field} must be an integer")
    if int(config["start"]) < 0 or int(config["limit"]) <= 0 or int(config["t_max"]) <= 0:
        raise ValueError("state config start/limit/t_max has an invalid value")
    if int(config["start_agent_seed"]) < 0 or int(config["num_rollouts"]) <= 0:
        raise ValueError("state config has an invalid seed or rollout count")
    if int(config["min_agents_before_stop"]) < 1 or int(config["min_agents_before_stop"]) > len(AGENT_IDS):
        raise ValueError("state config min_agents_before_stop is out of range")
    if int(config["max_new_tokens"]) <= 0 or int(config["protocol_thinking_max_tokens"]) <= 0 or int(config["protocol_max_tokens"]) <= 0 or int(config["group_retries"]) < 0 or int(config["step_retries"]) < 0:
        raise ValueError("state config has an invalid retry/token value")
    if config["start_agent"] not in {*AGENT_IDS, "balanced", "random"}:
        raise ValueError(f"invalid state config start_agent: {config['start_agent']!r}")
    if config.get("bootstrap_agent") is not None and config.get("bootstrap_agent") not in AGENT_IDS:
        raise ValueError(f"invalid state config bootstrap_agent: {config.get('bootstrap_agent')!r}")
    bootstrap_handoff_target = config.get("bootstrap_handoff_target")
    if bootstrap_handoff_target is not None:
        if config.get("bootstrap_agent") is None:
            raise ValueError("bootstrap_handoff_target requires bootstrap_agent")
        if bootstrap_handoff_target not in AGENT_IDS:
            raise ValueError(
                f"invalid state config bootstrap_handoff_target: {bootstrap_handoff_target!r}"
            )
    for field in (
        "allow_first_turn_stop", "enable_thinking", "require_thinking",
        "retry_failed_groups", "preserve_reasonable_incumbent",
    ):
        if not isinstance(config[field], bool):
            raise ValueError(f"state config {field} must be boolean")
    locked_agents = config["lock_upstream_answer_agents"]
    if (
        not isinstance(locked_agents, list)
        or any(agent not in AGENT_IDS for agent in locked_agents)
        or len(set(locked_agents)) != len(locked_agents)
        or locked_agents != [agent for agent in AGENT_IDS if agent in locked_agents]
    ):
        raise ValueError(
            "state config lock_upstream_answer_agents must be a canonical list of "
            "unique agent IDs"
        )
    if config["require_thinking"] and not config["enable_thinking"]:
        raise ValueError("state config require_thinking requires enable_thinking")
    if config["json_transport"] not in {"json_schema", "json_object", "none"}:
        raise ValueError(f"invalid state config json_transport: {config['json_transport']!r}")
    if config["output_mode"] not in {"turns", "trajectories"}:
        raise ValueError(f"invalid state config output_mode: {config['output_mode']!r}")
    for field in ("temperature", "top_p", "api_timeout"):
        if not _is_number(config[field]) or not math.isfinite(float(config[field])):
            raise ValueError(f"state config {field} must be finite numeric")
    if float(config["temperature"]) < 0 or not 0 < float(config["top_p"]) <= 1 or float(config["api_timeout"]) <= 0:
        raise ValueError("state config temperature/top_p/api_timeout is out of range")
    expected_count = int(config["limit"]) * int(config["num_rollouts"])
    if len(items) != expected_count:
        raise ValueError(f"state item count {len(items)} != expected {expected_count}")
    identities: set[tuple[int, int]] = set()
    for index, item in enumerate(items):
        identity = _validate_item(item, index, state)
        if identity in identities:
            raise ValueError(f"duplicate state item identity: {identity}")
        identities.add(identity)
    expected_identities = {
        (int(config["start"]) + offset, rollout_idx)
        for offset in range(int(config["limit"]))
        for rollout_idx in range(int(config["num_rollouts"]))
    }
    if identities != expected_identities:
        missing = sorted(expected_identities - identities)[:5]
        extra = sorted(identities - expected_identities)[:5]
        raise ValueError(f"state item identities do not match config (missing={missing}, extra={extra})")
    if not isinstance(state.get("history", []), list):
        raise ValueError("state history must be a list")
    if not _is_int(state.get("phase_count", 0)) or int(state.get("phase_count", 0)) < 0:
        raise ValueError("state phase_count must be a non-negative integer")


def load_state(path: Path) -> Dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            state = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load MATH role-batched state {path}: {exc}") from exc
    if not isinstance(state, dict):
        raise ValueError(f"invalid MATH role-batched state root: {path}")
    if state.get("version") != STATE_VERSION:
        raise ValueError(f"unsupported state version: {state.get('version')}")
    if not isinstance(state.get("config"), dict) or not isinstance(state.get("items"), list):
        raise ValueError("invalid MATH role-batched state")
    _normalize_legacy_state(state)
    # Older non-retrying runs persisted transient ``retry``/``failed`` labels
    # even though those labels were terminal for that runner.  Normalize them
    # before the first validation pass; journal replay below may reintroduce a
    # newer retry update and is normalized once more afterwards.
    if not bool(state["config"].get("retry_failed_groups", False)):
        finalize_legacy_retries(state)
    # Validate the immutable configuration before replaying any journal.  The
    # item-level pass below runs again after replay, catching torn updates.
    _validate_state(state)
    apply_state_journal(path, state)
    if not bool(state["config"].get("retry_failed_groups", False)):
        finalize_legacy_retries(state)
    _validate_state(state)
    return state


def pending_count(state: Dict[str, Any], agent: Optional[str] = None, turn: Optional[int] = None) -> int:
    count = 0
    for item in state["items"]:
        work = _ready_work(item)
        if work is None or work[0] == "bootstrap":
            continue
        if agent is not None and work[1] != agent:
            continue
        if turn is not None and current_turn(item) != turn:
            continue
        count += 1
    return count


def recovery_pending_count(state: Dict[str, Any]) -> int:
    return sum(
        item.get("status") == "pending"
        and isinstance(item.get("pending_recovery"), dict)
        for item in state["items"]
    )


def bootstrap_pending_count(state: Dict[str, Any], agent: Optional[str] = None) -> int:
    return sum(
        item.get("status") == "pending"
        and isinstance(item.get("bootstrap"), dict)
        and item["bootstrap"].get("status") == "pending"
        and (agent is None or item["bootstrap"].get("agent") == agent)
        for item in state["items"]
    )


def status_count(state: Dict[str, Any], status: str) -> int:
    return sum(item.get("status") == status for item in state["items"])


def total_steps(state: Dict[str, Any]) -> int:
    return sum(current_turn(item) for item in state["items"])


def terminate(
    item: Dict[str, Any],
    reason: str,
    error: Optional[str] = None,
    *,
    retry_failed_group: bool = False,
) -> None:
    if reason == "truncated":
        steps = item["trajectory"].get("steps", [])
        if not steps:
            raise ValueError("cannot truncate a trajectory without a tentative answer")
        tentative_answer = steps[-1].get("tentative_answer")
        if not isinstance(tentative_answer, str) or not tentative_answer.strip():
            raise ValueError("cannot truncate a trajectory with an empty tentative answer")
        item["trajectory"]["final_answer"] = tentative_answer
    item["trajectory"]["terminated_by"] = reason
    item["trajectory"]["error"] = error
    item["last_error"] = error
    item["status"] = (
        "retry" if retry_failed_group and reason != "stop" else "done"
    )
    item["current_agent"] = None
    item["pending_recovery"] = None


def build_response_format(
    transport: str,
    *,
    enable_thinking: bool = False,
    active_agent: Optional[str] = None,
    allow_self_handoff: bool = False,
) -> Optional[Dict[str, Any]]:
    if transport == "none":
        return None
    if transport == "json_object":
        return {"type": "json_object"}
    if transport != "json_schema":
        raise ValueError(f"unsupported JSON transport: {transport}")
    nullable_note = {
        "anyOf": [
            {"type": "string", "maxLength": PROTOCOL_NOTE_MAX_CHARS},
            {"type": "null"},
        ]
    }
    if active_agent is not None and active_agent not in AGENT_IDS:
        raise ValueError(f"invalid active agent for protocol schema: {active_agent!r}")
    handoff_targets = [
        agent
        for agent in AGENT_IDS
        if active_agent is None or agent != active_agent or allow_self_handoff
    ]

    def action_schema(action: str) -> Dict[str, Any]:
        handoff = action == "handoff"
        properties: Dict[str, Any] = {
            "reasoning": {
                "type": "string",
                "minLength": 1,
                "maxLength": PROTOCOL_REASONING_MAX_CHARS,
            },
            "tentative_answer": {
                "type": "string",
                "minLength": 1,
                "maxLength": PROTOCOL_ANSWER_MAX_CHARS,
            },
            "action": {"type": "string", "enum": [action]},
            "handoff_target": (
                {"type": "string", "enum": handoff_targets}
                if handoff
                else {"type": "null"}
            ),
            "handoff_note": nullable_note,
            "confirmed_answer": (
                {"type": "null"}
                if handoff
                else {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": PROTOCOL_ANSWER_MAX_CHARS,
                }
            ),
        }
        return {
            "type": "object",
            "properties": properties,
            "required": list(PROTOCOL_FIELDS),
            "additionalProperties": False,
        }

    schema = {
        "anyOf": [
            action_schema("handoff"),
            action_schema("confirm_stop"),
        ]
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "math_multi_agent_protocol",
            "strict": True,
            "schema": schema,
        },
    }


def fallback_thinking_trace(visible: str, reasoning: str) -> str:
    prefix = visible.split("{", 1)[0].strip()
    return prefix or reasoning.strip()


def degenerate_protocol_reason(candidate: base.MathStep) -> Optional[str]:
    for field in (
        "reasoning",
        "tentative_answer",
        "handoff_note",
        "confirmed_answer",
    ):
        value = getattr(candidate, field)
        if isinstance(value, str):
            dollar_skeleton = re.sub(r"[^$A-Za-z0-9]+", "", value)
            if re.search(r"(?:\$){12,}", dollar_skeleton):
                return f"degenerate repetition in protocol field {field}"
    return None


def degenerate_recovery_target(collapsed_agent: str) -> str:
    if collapsed_agent not in AGENT_IDS:
        raise ValueError(f"invalid collapsed agent: {collapsed_agent!r}")
    for candidate in DEGENERATE_RECOVERY_TARGET_ORDER:
        if candidate not in {collapsed_agent, DEGENERATE_RECOVERY_AGENT}:
            return candidate
    for candidate in DEGENERATE_RECOVERY_TARGET_ORDER:
        if candidate != DEGENERATE_RECOVERY_AGENT:
            return candidate
    raise RuntimeError("no non-recovery handoff target is available")


def degenerate_recovery_prompt(
    collapsed_agent: str,
    recovery_target: str,
    reason: str,
) -> str:
    return (
        f"Agent {collapsed_agent}'s scheduled response was rejected because it "
        f"collapsed: {reason}. You are the trained {DEGENERATE_RECOVERY_AGENT} "
        "recovery agent. Check the current mathematical answer and replace the "
        "collapsed step with a concise, substantive protocol response. Preserve a "
        "correct incumbent answer or correct it if necessary. Output exactly one "
        "complete JSON object. action must be handoff, confirmed_answer must be null, "
        f"and handoff_target must be {recovery_target}. The next verification step "
        "must go to that non-collapsed model. Never repeat empty math delimiters."
    )


def bootstrap_preservation_reason(
    item: Dict[str, Any], candidate: base.MathStep
) -> Optional[str]:
    if candidate.turn != 0 or not isinstance(item.get("bootstrap"), dict):
        return None
    bootstrap_answer = str(item["bootstrap"].get("final_answer") or "").strip()
    if bootstrap_answer and not math_answers_equivalent(
        candidate.tentative_answer, bootstrap_answer
    ):
        return (
            "protocol turn 0 must preserve the extracted bootstrap answer "
            f"{bootstrap_answer!r}"
        )
    return None


def protocol_retry_prompt(
    agent: str,
    reason: str,
    *,
    exact_tentative_answer: Optional[str] = None,
    bootstrap_handoff_target: Optional[str] = None,
    enable_thinking: bool = False,
) -> str:
    allow_bootstrap_self_handoff = (
        exact_tentative_answer is not None and bootstrap_handoff_target == agent
    )
    valid_targets = ", ".join(
        candidate
        for candidate in AGENT_IDS
        if candidate != agent or allow_bootstrap_self_handoff
    )
    targeted_guidance = ""
    if exact_tentative_answer is not None:
        exact_answer_literal = exact_answer_json_literal(exact_tentative_answer)
        exact_protocol_object = bootstrap_protocol_copy_object(
            agent, exact_tentative_answer, bootstrap_handoff_target
        )
        output_prefix = "After the thinking trace, " if enable_thinking else ""
        targeted_guidance += (
            " This is a format-conversion turn: after JSON decoding, tentative_answer "
            "must be character-for-character identical to the extracted bootstrap "
            f"answer. Its required raw JSON string literal is {exact_answer_literal}. "
            "Every \\u005c sequence in that literal encodes one required backslash. "
            "Copy the complete literal verbatim; do not normalize LaTeX, replace commands "
            "with Unicode, or alter whitespace or punctuation. Do not compose a new JSON "
            f"object. {output_prefix}Copy this entire one-line object verbatim "
            "and output nothing else in the visible response: "
            f"{exact_protocol_object}"
        )
    if reason.startswith("malformed JSON protocol"):
        targeted_guidance += (
            r" The visible response after </think> must be exactly one parseable JSON "
            r"object with all six required fields, no Markdown fence, and no trailing "
            r"text. Inside JSON strings, double every LaTeX backslash in the raw JSON "
            r'source (write "\\frac{1}{2}", never "\frac{1}{2}"), and escape '
            r"embedded quotes and newlines. To avoid control-character escapes, use "
            r"plain ASCII words instead of LaTeX backslash commands in reasoning and "
            r"handoff_note. Keep reasoning under 300 characters and handoff_note under "
            r"120 characters so the JSON object is completed."
        )
    self_handoff_rule = (
        "For this bootstrap format-conversion turn only, handing off to yourself is "
        "required. "
        if allow_bootstrap_self_handoff
        else "Never hand off to yourself. "
    )
    return (
        f"The previous response was rejected for this exact reason: {reason}. "
        f"{targeted_guidance} "
        f"You are {agent}. Recompute only as needed and output exactly one complete "
        "JSON object. For action=handoff, handoff_target must be one of "
        f"[{valid_targets}] and confirmed_answer must be the JSON literal null. "
        f"{self_handoff_rule}For action=confirm_stop, handoff_target must be "
        "null and confirmed_answer must be a non-empty final answer. Do not emit "
        "malformed JSON or repeated empty math delimiters such as '$ $ $'."
    )


def parse_protocol_with_labels(
    visible: str, agent: str, turn: int, raw: str
) -> base.MathStep:
    parsed = base.parse_protocol(visible, agent, turn, raw)
    labels = (
        "reasoning",
        "tentative_answer",
        "action",
        "handoff_target",
        "handoff_note",
        "confirmed_answer",
    )

    def capture(label: str) -> str:
        following = "|".join(labels[labels.index(label) + 1 :])
        boundary = rf"(?=^\s*(?:\[[A-Za-z0-9_-]+\]\s*)?(?:{following})\s*:|\Z)"
        match = re.search(
            rf"^\s*(?:\[[A-Za-z0-9_-]+\]\s*)?{label}\s*:\s*(.*?){boundary}",
            visible,
            flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
        )
        return match.group(1).strip().strip("` ") if match else ""

    values = {label: capture(label) for label in labels}
    if not parsed.reasoning and values["reasoning"]:
        parsed.reasoning = values["reasoning"]
    if not parsed.tentative_answer and values["tentative_answer"]:
        parsed.tentative_answer = values["tentative_answer"]
    if not parsed.action and values["action"]:
        parsed.action = values["action"].strip("`\"' ,.;").lower()
    if parsed.handoff_target is None and values["handoff_target"]:
        target = values["handoff_target"].strip("`\"' ,.;")
        parsed.handoff_target = None if target.lower() in {"none", "null"} else target
    if parsed.handoff_note is None and values["handoff_note"]:
        parsed.handoff_note = values["handoff_note"]
    if parsed.confirmed_answer is None and values["confirmed_answer"]:
        answer = values["confirmed_answer"]
        parsed.confirmed_answer = None if answer.lower() in {"none", "null"} else answer
    return parsed


def protocol_json_well_formed(
    visible: str, active_agent: Optional[str] = None
) -> bool:
    """Return whether the visible model output satisfies the protocol schema."""
    return strict_protocol_json_well_formed(
        visible,
        active_agent=active_agent,
        agent_ids=base.AGENT_IDS,
    )


def repair_redundant_protocol_fields(candidate: base.MathStep) -> None:
    """Validate legacy callers without mutating contradictory fields.

    Older callers used this helper to silently clear fields that contradicted
    ``action``.  Keep the name for API compatibility, but fail closed so an
    invalid model response is never repaired into a different response.
    """

    payload = {
        "reasoning": candidate.reasoning,
        "tentative_answer": candidate.tentative_answer,
        "action": candidate.action,
        "handoff_target": candidate.handoff_target,
        "handoff_note": candidate.handoff_note,
        "confirmed_answer": candidate.confirmed_answer,
    }
    validate_protocol_object(payload, active_agent=candidate.active_agent)


def parse_strict_protocol(
    visible: str,
    agent: str,
    turn: int,
    raw: str,
    *,
    allow_self_handoff: bool = False,
) -> base.MathStep:
    """Build a MathStep from an already schema-validated JSON object."""

    payload = validate_protocol_object(
        visible,
        active_agent=agent,
        allow_self_handoff=allow_self_handoff,
    )
    return base.MathStep(
        turn=turn,
        active_agent=agent,
        reasoning=payload["reasoning"].strip(),
        tentative_answer=payload["tentative_answer"].strip(),
        action=payload["action"],
        handoff_target=payload["handoff_target"],
        handoff_note=payload["handoff_note"],
        confirmed_answer=payload["confirmed_answer"],
        raw_output=raw,
        visible_output=visible,
        thinking="",
    )


def advance_bootstrap_one(
    item: Dict[str, Any],
    agent: str,
    caller: OpenAIChatLLMCaller,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    bootstrap = item.get("bootstrap")
    if (
        item.get("status") != "pending"
        or not isinstance(bootstrap, dict)
        or bootstrap.get("status") != "pending"
        or bootstrap.get("agent") != agent
    ):
        return item
    problem = problem_from_dict(item["problem"])
    all_raw_outputs: list[str] = []
    last_reason = "no generation"
    for attempt in range(int(config["step_retries"]) + 1):
        messages = [
            {"role": "system", "content": DIRECT_SOLVER_SYSTEM_PROMPT},
            {"role": "user", "content": format_math_problem_as_prompt(problem)},
        ]
        if attempt:
            thinking_instruction = (
                "include a non-empty thinking trace, and "
                if bool(config["require_thinking"])
                else ""
            )
            messages.append({
                "role": "user",
                "content": (
                    "The previous response was unusable. Recompute the mathematics, "
                    f"{thinking_instruction}end with one final \\boxed{{...}} answer."
                ),
            })
        raw: object = ""
        raw_text = ""
        visible = ""
        attempt_outputs: list[str] = []
        try:
            raw = caller.generate(
                messages,
                seed=bootstrap_request_seed(
                    int(config["generation_seed"]), item, attempt
                ),
            )
            attempt_outputs = [
                normalize_text(text) for text in base.response_attempts(raw)
            ]
            all_raw_outputs.extend(attempt_outputs)
            raw_text = normalize_text(str(raw))
            thinking, visible = base.split_thinking(raw_text)
            final_answer = extract_last_boxed(visible)
            if not bool(config["enable_thinking"]) and thinking.strip():
                last_reason = "thinking mode violation: non-empty thinking in non-thinking mode"
            elif bool(config["require_thinking"]) and not thinking.strip():
                last_reason = "missing non-empty thinking trace"
            elif not final_answer:
                last_reason = "missing final boxed answer"
            else:
                attempt_record = {
                    "attempt": attempt + 1,
                    "accepted": True,
                    "reason": None,
                    "raw_output": raw_text,
                    "raw_outputs": attempt_outputs,
                    "visible_raw_output": visible,
                }
                bootstrap["attempts"].append(attempt_record)
                bootstrap.update({
                    "status": "done",
                    "thinking": thinking if bool(config["enable_thinking"]) else "",
                    "visible_output": visible,
                    "final_answer": final_answer,
                    "raw_output": raw_text,
                    "raw_outputs": list(all_raw_outputs),
                    "error": None,
                })
                item["messages"][1] = {
                    "role": "user",
                    "content": bootstrap_protocol_prompt(
                        problem,
                        bootstrap_visible=visible,
                        bootstrap_answer=final_answer,
                    protocol_agent=str(item["start_agent"]),
                    handoff_target=config.get("bootstrap_handoff_target"),
                    enable_thinking=bool(config["enable_thinking"]),
                    ),
                }
                return item
        except Exception as exc:
            last_reason = normalize_text(str(exc))
        bootstrap["attempts"].append({
            "attempt": attempt + 1,
            "accepted": False,
            "reason": last_reason,
            "raw_output": raw_text or normalize_text(str(raw)),
            "raw_outputs": attempt_outputs,
            "visible_raw_output": visible,
        })
    bootstrap.update({
        "status": "failed",
        "thinking": "",
        "visible_output": "",
        "final_answer": None,
        "raw_output": bootstrap["attempts"][-1]["raw_output"],
        "raw_outputs": list(all_raw_outputs),
        "error": last_reason,
    })
    terminate(
        item,
        "rejected_quality",
        error=f"bootstrap agent {agent}: {last_reason}",
        retry_failed_group=False,
    )
    return item


def recover_degenerate_step(
    item: Dict[str, Any],
    collapsed_agent: str,
    caller: OpenAIChatLLMCaller,
    config: Dict[str, Any],
    *,
    turn: int,
    reason: str,
    exact_tentative_answer: Optional[str] = None,
) -> tuple[Optional[base.MathStep], Optional[list[Dict[str, str]]], str]:
    recovery_agent = DEGENERATE_RECOVERY_AGENT
    allow_bootstrap_self_handoff = (
        turn == 0
        and exact_tentative_answer is not None
        and config.get("bootstrap_agent") == recovery_agent
        and config.get("bootstrap_handoff_target") == recovery_agent
    )
    recovery_target = (
        recovery_agent
        if allow_bootstrap_self_handoff
        else degenerate_recovery_target(collapsed_agent)
    )
    recovery_prompt = degenerate_recovery_prompt(
        collapsed_agent,
        recovery_target,
        reason,
    )
    raw_outputs: list[str] = []
    last_reason = reason
    for attempt in range(int(config["step_retries"]) + 1):
        messages = [dict(message) for message in item["messages"]]
        messages[0] = {
            "role": "system",
            "content": protocol_system_prompt(recovery_agent, config),
        }
        messages.append({"role": "user", "content": recovery_prompt})
        if attempt:
            messages.append({
                "role": "user",
                "content": protocol_retry_prompt(
                    recovery_agent,
                    last_reason,
                    exact_tentative_answer=exact_tentative_answer,
                    bootstrap_handoff_target=(
                        config.get("bootstrap_handoff_target")
                        if turn == 0 else None
                    ),
                    enable_thinking=bool(config["enable_thinking"]),
                ),
            })
        raw: object = ""
        raw_text = ""
        visible = ""
        attempt_outputs: list[str] = []
        try:
            raw = caller.generate(
                messages,
                seed=request_seed(
                    int(config["generation_seed"]), item, turn, attempt + 1000
                ),
            )
            attempt_outputs = [
                normalize_text(text) for text in base.response_attempts(raw)
            ]
            raw_outputs.extend(attempt_outputs)
            raw_text = normalize_text(str(raw))
            thinking, visible = base.split_thinking(raw_text)
            attempt_record = {
                "turn": turn,
                "agent": recovery_agent,
                "attempt": attempt + 1,
                "accepted": False,
                "reason": None,
                "raw_output": raw_text,
                "raw_outputs": attempt_outputs,
                "visible_raw_output": visible,
            }
            if not json_object_well_formed(visible):
                last_reason = "A3 recovery malformed JSON protocol"
            else:
                schema_reason = protocol_schema_error(
                    visible,
                    active_agent=recovery_agent,
                    allow_self_handoff=allow_bootstrap_self_handoff,
                )
                if schema_reason is not None:
                    last_reason = f"A3 recovery {schema_reason}"
                elif not bool(config["enable_thinking"]) and thinking.strip():
                    last_reason = (
                        "A3 recovery thinking mode violation: non-empty thinking "
                        "in non-thinking mode"
                    )
                else:
                    if not bool(config["enable_thinking"]):
                        thinking = ""
                    candidate = parse_strict_protocol(
                        visible,
                        recovery_agent,
                        turn,
                        raw_text,
                        allow_self_handoff=allow_bootstrap_self_handoff,
                    )
                    candidate = base.MathStep(**sanitize_json_value(asdict(candidate)))
                    candidate.thinking = thinking
                    candidate.raw_outputs = list(raw_outputs)
                    if exact_tentative_answer is not None:
                        candidate.tentative_answer = exact_tentative_answer
                    candidate.action = "handoff"
                    candidate.handoff_target = recovery_target
                    candidate.confirmed_answer = None
                    if not str(candidate.handoff_note or "").strip():
                        candidate.handoff_note = (
                            f"Verify the recovered answer after {collapsed_agent} "
                            "generation collapse."
                        )
                    candidate.visible_output = canonical_protocol_visible_output(candidate)
                    last_reason = (
                        degenerate_protocol_reason(candidate)
                        or base.invalid_step_reason(
                            candidate,
                            prior_tentative=(
                                bool(item.get("prior_tentative"))
                                or bool(config.get("allow_first_turn_stop", False))
                            ),
                            seen_agents=active_agents(item),
                            min_agents_before_stop=int(config["min_agents_before_stop"]),
                            require_thinking=bool(config["require_thinking"]),
                            allow_self_handoff=allow_bootstrap_self_handoff,
                        )
                        or ""
                    )
                    if not last_reason:
                        attempt_record["accepted"] = True
                        item.setdefault("generation_attempts", []).append(attempt_record)
                        return candidate, messages, ""
            attempt_record["reason"] = last_reason
            item.setdefault("generation_rejections", []).append(dict(attempt_record))
            item.setdefault("generation_attempts", []).append(attempt_record)
            item["last_rejected_raw"] = raw_text
        except Exception as exc:
            last_reason = f"A3 recovery {normalize_text(str(exc))}"
            attempt_record = {
                "turn": turn,
                "agent": recovery_agent,
                "attempt": attempt + 1,
                "accepted": False,
                "reason": last_reason,
                "raw_output": raw_text or normalize_text(str(raw)),
                "raw_outputs": attempt_outputs,
                "visible_raw_output": visible,
            }
            item.setdefault("generation_rejections", []).append(dict(attempt_record))
            item.setdefault("generation_attempts", []).append(attempt_record)
            item["last_rejected_raw"] = attempt_record["raw_output"]
    return None, None, last_reason


def _commit_accepted_step(
    item: Dict[str, Any],
    accepted: base.MathStep,
    accepted_messages: Optional[list[Dict[str, str]]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    problem = problem_from_dict(item["problem"])
    turn = current_turn(item)
    retry_failed_group = bool(config.get("retry_failed_groups", False))
    item["pending_recovery"] = None
    item["trajectory"]["turn_messages"].append(
        accepted_messages
        if accepted_messages is not None
        else [dict(message) for message in item["messages"]]
    )
    item["trajectory"]["steps"].append(asdict(accepted))
    item["messages"].append(
        {"role": "assistant", "content": base.render_shared_message(accepted)}
    )
    if turn == 0 and isinstance(item.get("bootstrap"), dict):
        item["messages"][1] = {
            "role": "user",
            "content": format_math_problem_as_prompt(problem),
        }
    item["prior_tentative"] = True
    if accepted.action == "confirm_stop":
        item["trajectory"]["steps"][-1] = asdict(accepted)
        item["trajectory"]["final_answer"] = accepted.confirmed_answer
        terminate(item, "stop")
        return item
    if turn + 1 >= int(config["t_max"]):
        terminate(item, "truncated", retry_failed_group=retry_failed_group)
        return item
    item["current_agent"] = accepted.handoff_target
    item["messages"][0] = {
        "role": "system",
        "content": protocol_system_prompt(str(accepted.handoff_target), config),
    }
    return item


def advance_one(
    item: Dict[str, Any],
    agent: str,
    caller: OpenAIChatLLMCaller,
    config: Dict[str, Any],
    *,
    recovery_caller: Optional[OpenAIChatLLMCaller] = None,
    defer_degenerate_recovery: bool = False,
) -> Dict[str, Any]:
    if item.get("status") != "pending" or item.get("current_agent") != agent:
        return item
    if item.get("pending_recovery") is not None:
        return item
    if isinstance(item.get("bootstrap"), dict) and item["bootstrap"].get("status") != "done":
        return item
    problem = problem_from_dict(item["problem"])
    turn = current_turn(item)
    retry_failed_group = bool(config.get("retry_failed_groups", False))
    if turn >= int(config["t_max"]):
        terminate(item, "truncated", retry_failed_group=retry_failed_group)
        return item

    last_reason = "no generation"
    accepted = None
    accepted_messages: Optional[list[Dict[str, str]]] = None
    collapse_detected = False
    raw_outputs: list[str] = []
    output_attempts = 0
    malformed_json_attempts = 0
    protocol_schema_violation_attempts = 0
    degenerate_output_attempts = 0
    thinking_mode_violation_attempts = 0
    generation_error_attempts = 0
    exact_tentative_answer = None
    if turn == 0 and isinstance(item.get("bootstrap"), dict):
        bootstrap_answer = item["bootstrap"].get("final_answer")
        if isinstance(bootstrap_answer, str) and bootstrap_answer.strip():
            exact_tentative_answer = bootstrap_answer
    allow_bootstrap_self_handoff = (
        exact_tentative_answer is not None
        and config.get("bootstrap_handoff_target") == agent
        and config.get("bootstrap_agent") == agent
    )
    for attempt in range(int(config["step_retries"]) + 1):
        messages = [dict(message) for message in item["messages"]]
        if attempt:
            messages.append({
                "role": "user",
                "content": protocol_retry_prompt(
                    agent,
                    last_reason,
                    exact_tentative_answer=exact_tentative_answer,
                    bootstrap_handoff_target=(
                        config.get("bootstrap_handoff_target")
                        if turn == 0 else None
                    ),
                    enable_thinking=bool(config["enable_thinking"]),
                ),
            })
        raw = ""
        try:
            raw = caller.generate(
                messages,
                seed=request_seed(int(config["generation_seed"]), item, turn, attempt),
            )
            # ``GeneratedText`` carries every response made while recovering
            # from a length/context limit.  Read that metadata before casting
            # to ``str``; the cast otherwise discards the earlier attempts.
            attempt_outputs = [
                normalize_text(text) for text in base.response_attempts(raw)
            ]
            raw_text = normalize_text(str(raw))
            thinking, visible = base.split_thinking(raw_text)
            output_attempts += 1
            raw_outputs.extend(attempt_outputs)
            attempt_record = {
                "turn": turn,
                "agent": agent,
                "attempt": attempt + 1,
                "accepted": False,
                "reason": None,
                "raw_output": raw_text,
                "raw_outputs": attempt_outputs,
                "visible_raw_output": visible,
            }
            if not json_object_well_formed(visible):
                malformed_json_attempts += 1
                last_reason = "malformed JSON protocol"
                attempt_record["reason"] = last_reason
                item.setdefault("generation_rejections", []).append(dict(attempt_record))
                item.setdefault("generation_attempts", []).append(attempt_record)
                item["last_rejected_raw"] = raw_text
                continue
            schema_reason = protocol_schema_error(
                visible,
                active_agent=agent,
                allow_self_handoff=allow_bootstrap_self_handoff,
            )
            if schema_reason is not None:
                protocol_schema_violation_attempts += 1
                last_reason = schema_reason
                attempt_record["reason"] = last_reason
                item.setdefault("generation_rejections", []).append(dict(attempt_record))
                item.setdefault("generation_attempts", []).append(attempt_record)
                item["last_rejected_raw"] = raw_text
                continue
            if not bool(config["enable_thinking"]) and thinking.strip():
                thinking_mode_violation_attempts += 1
                last_reason = (
                    "thinking mode violation: non-empty thinking in non-thinking mode"
                )
                attempt_record["reason"] = last_reason
                item.setdefault("generation_rejections", []).append(dict(attempt_record))
                item.setdefault("generation_attempts", []).append(attempt_record)
                item["last_rejected_raw"] = raw_text
                continue
            if not bool(config["enable_thinking"]):
                thinking = ""
            candidate = parse_strict_protocol(
                visible,
                agent,
                turn,
                raw_text,
                allow_self_handoff=allow_bootstrap_self_handoff,
            )
            candidate = base.MathStep(**sanitize_json_value(asdict(candidate)))
            candidate.thinking = thinking
            candidate.raw_outputs = list(raw_outputs)
            if exact_tentative_answer is not None:
                candidate.tentative_answer = exact_tentative_answer
                bootstrap_handoff_target = config.get("bootstrap_handoff_target")
                if bootstrap_handoff_target is not None:
                    candidate.action = "handoff"
                    candidate.handoff_target = str(bootstrap_handoff_target)
                    candidate.confirmed_answer = None
                candidate.visible_output = canonical_protocol_visible_output(candidate)
            last_reason = (
                degenerate_protocol_reason(candidate)
                or base.invalid_step_reason(
                candidate,
                prior_tentative=(
                    bool(item.get("prior_tentative"))
                    or bool(config.get("allow_first_turn_stop", False))
                ),
                seen_agents=active_agents(item),
                min_agents_before_stop=int(config["min_agents_before_stop"]),
                require_thinking=bool(config["require_thinking"]),
                allow_self_handoff=allow_bootstrap_self_handoff,
                )
                or ""
            )
            if last_reason.startswith("degenerate repetition"):
                degenerate_output_attempts += 1
            attempt_record["reason"] = last_reason or None
            if not last_reason:
                attempt_record["accepted"] = True
                item.setdefault("generation_attempts", []).append(attempt_record)
                candidate.raw_outputs = list(raw_outputs)
                accepted = candidate
                accepted_messages = [dict(message) for message in item["messages"]]
                break
            item.setdefault("generation_rejections", []).append(dict(attempt_record))
            item.setdefault("generation_attempts", []).append(attempt_record)
            item["last_rejected_raw"] = raw_text
            if last_reason.startswith("degenerate repetition"):
                collapse_detected = True
                break
        except Exception as exc:
            generation_error_attempts += 1
            last_reason = normalize_text(str(exc))
            attempt_record = {
                "turn": turn,
                "agent": agent,
                "attempt": attempt + 1,
                "accepted": False,
                "reason": last_reason,
                "raw_output": normalize_text(str(raw)),
                "raw_outputs": [],
                "visible_raw_output": "",
            }
            item.setdefault("generation_rejections", []).append(attempt_record)
            item.setdefault("generation_attempts", []).append(dict(attempt_record))
            item["last_rejected_raw"] = attempt_record["raw_output"]
    only_protocol_violations = (
        output_attempts > 0
        and (
            malformed_json_attempts
            + protocol_schema_violation_attempts
            + degenerate_output_attempts
            + thinking_mode_violation_attempts
            == output_attempts
        )
        and generation_error_attempts == 0
    )
    if accepted is None and collapse_detected and recovery_caller is not None:
        if defer_degenerate_recovery:
            item["pending_recovery"] = {
                "collapsed_agent": agent,
                "turn": turn,
                "reason": last_reason,
                "exact_tentative_answer": exact_tentative_answer,
                "retry_failed_group": (
                    retry_failed_group and not only_protocol_violations
                ),
            }
            return item
        accepted, accepted_messages, recovery_reason = recover_degenerate_step(
            item,
            agent,
            recovery_caller,
            config,
            turn=turn,
            reason=last_reason,
            exact_tentative_answer=exact_tentative_answer,
        )
        if accepted is None:
            last_reason = recovery_reason
    if accepted is None:
        # A malformed protocol is a terminal failed turn, not a trajectory
        # retry.  Group retries remain available for transport/service errors
        # and for well-formed JSON that fails a semantic collaboration rule.
        terminate(
            item,
            "rejected_quality",
            error=f"turn {turn} agent {agent}: {last_reason}",
            retry_failed_group=retry_failed_group and not only_protocol_violations,
        )
        return item

    return _commit_accepted_step(item, accepted, accepted_messages, config)


def advance_degenerate_recovery_one(
    item: Dict[str, Any],
    caller: OpenAIChatLLMCaller,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    pending = item.get("pending_recovery")
    if item.get("status") != "pending" or not isinstance(pending, dict):
        return item
    collapsed_agent = str(pending["collapsed_agent"])
    turn = int(pending["turn"])
    accepted, accepted_messages, last_reason = recover_degenerate_step(
        item,
        collapsed_agent,
        caller,
        config,
        turn=turn,
        reason=str(pending["reason"]),
        exact_tentative_answer=pending["exact_tentative_answer"],
    )
    item["pending_recovery"] = None
    if accepted is None:
        terminate(
            item,
            "rejected_quality",
            error=f"turn {turn} agent {collapsed_agent}: {last_reason}",
            retry_failed_group=bool(pending["retry_failed_group"]),
        )
        return item
    return _commit_accepted_step(item, accepted, accepted_messages, config)


def run_bootstrap_phase(
    state: Dict[str, Any],
    agent: str,
    caller: OpenAIChatLLMCaller,
    max_concurrency: int,
    journal: Optional[StateJournal] = None,
) -> int:
    indices = [
        index
        for index, item in enumerate(state["items"])
        if item.get("status") == "pending"
        and isinstance(item.get("bootstrap"), dict)
        and item["bootstrap"].get("status") == "pending"
        and item["bootstrap"].get("agent") == agent
    ]

    def advance(index: int) -> tuple[int, Dict[str, Any]]:
        return index, advance_bootstrap_one(
            copy.deepcopy(state["items"][index]), agent, caller, state["config"]
        )

    started = time.monotonic()
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, min(max_concurrency, len(indices) or 1))) as pool:
        futures = [pool.submit(advance, index) for index in indices]
        for future in as_completed(futures):
            index, item = future.result()
            state["items"][index] = item
            if journal is not None:
                journal.append(index, item)
            done += 1
            if done == 1 or done % 100 == 0 or done == len(indices):
                print(
                    f"  bootstrap_progress={done}/{len(indices)} agent={agent} "
                    f"rate={done / max(time.monotonic() - started, 1e-6):.2f}/s",
                    flush=True,
                )
    state["phase_count"] = int(state.get("phase_count", 0)) + 1
    state["updated_at"] = utc_now()
    state.setdefault("history", []).append({
        "phase": state["phase_count"],
        "kind": "bootstrap",
        "agent": agent,
        "advanced": len(indices),
        "bootstrap_pending_after": bootstrap_pending_count(state),
        "pending_after": pending_count(state),
        "retry_after": status_count(state, "retry"),
        "total_steps_after": total_steps(state),
        "completed_at": state["updated_at"],
    })
    return len(indices)


def run_agent_phase(
    state: Dict[str, Any],
    agent: str,
    caller: OpenAIChatLLMCaller,
    max_concurrency: int,
    turn: Optional[int] = None,
    journal: Optional[StateJournal] = None,
) -> int:
    indices = [
        index for index, item in enumerate(state["items"])
        if item.get("status") == "pending"
        and item.get("current_agent") == agent
        and (turn is None or current_turn(item) == turn)
    ]

    def advance(index: int) -> tuple[int, Dict[str, Any]]:
        return index, advance_one(copy.deepcopy(state["items"][index]), agent, caller, state["config"])

    started = time.monotonic()
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, min(max_concurrency, len(indices) or 1))) as pool:
        futures = [pool.submit(advance, index) for index in indices]
        for future in as_completed(futures):
            index, item = future.result()
            state["items"][index] = item
            if journal is not None:
                journal.append(index, item)
            done += 1
            if done == 1 or done % 100 == 0 or done == len(indices):
                print(
                    f"  progress={done}/{len(indices)} agent={agent} "
                    f"turn={turn if turn is not None else 'all'} "
                    f"rate={done / max(time.monotonic() - started, 1e-6):.2f}/s",
                    flush=True,
                )
    state["phase_count"] = int(state.get("phase_count", 0)) + 1
    state["updated_at"] = utc_now()
    state.setdefault("history", []).append({
        "phase": state["phase_count"],
        "kind": "protocol",
        "agent": agent,
        "advanced": len(indices),
        "pending_after": pending_count(state),
        "retry_after": status_count(state, "retry"),
        "total_steps_after": total_steps(state),
        "completed_at": state["updated_at"],
    })
    return len(indices)


def _ready_work(item: Dict[str, Any]) -> Optional[tuple[str, str]]:
    if item.get("status") != "pending":
        return None
    bootstrap = item.get("bootstrap")
    if isinstance(bootstrap, dict) and bootstrap.get("status") == "pending":
        return "bootstrap", str(bootstrap["agent"])
    if isinstance(item.get("pending_recovery"), dict):
        return "recovery", DEGENERATE_RECOVERY_AGENT
    agent = item.get("current_agent")
    if agent in AGENT_IDS:
        return "protocol", str(agent)
    return None


def run_concurrent_pipeline(
    state: Dict[str, Any],
    *,
    bootstrap_caller: Any,
    protocol_callers: Dict[str, Optional[Any]],
    concurrency_by_agent: Dict[str, int],
    capacity_provider: Optional[Callable[[str], int]] = None,
    scheduler_observer: Optional[
        Callable[[Dict[str, Any], Dict[str, int], int, int], None]
    ] = None,
    journal: Optional[StateJournal] = None,
    progress_every: int = 25,
    rebalance_after_advanced: int = 0,
    max_advanced_per_pass: int = 0,
    max_inflight: int = 0,
) -> int:
    if max_advanced_per_pass < 0:
        raise ValueError("max_advanced_per_pass must be non-negative")
    if max_inflight < 0:
        raise ValueError("max_inflight must be non-negative")
    def current_capacity(agent: str) -> int:
        if capacity_provider is None:
            return int(concurrency_by_agent.get(agent, 0))
        return int(capacity_provider(agent))

    for agent in AGENT_IDS:
        capacity = current_capacity(agent)
        if capacity < 0:
            raise ValueError(f"invalid concurrency for {agent}")
        if capacity and (
            agent not in protocol_callers or protocol_callers[agent] is None
        ):
            raise ValueError(f"missing protocol caller for {agent}")
    bootstrap_agent = state["config"].get("bootstrap_agent")
    if bootstrap_agent is not None and bootstrap_agent != "A3":
        raise ValueError("concurrent pipeline currently requires bootstrap_agent=A3")
    if bootstrap_pending_count(state) and bootstrap_caller is None:
        raise ValueError("bootstrap work is pending but no bootstrap endpoint is available")

    in_flight: set[int] = set()
    active_by_agent = {agent: 0 for agent in AGENT_IDS}
    futures: Dict[Future[tuple[int, Dict[str, Any]]], tuple[int, str, str]] = {}
    started = time.monotonic()
    advanced = 0
    draining_for_rebalance = False

    def notify_scheduler() -> None:
        if scheduler_observer is not None:
            scheduler_observer(state, active_by_agent, advanced, len(futures))

    def run_work(index: int, kind: str, agent: str) -> tuple[int, Dict[str, Any]]:
        item = copy.deepcopy(state["items"][index])
        if kind == "bootstrap":
            return index, advance_bootstrap_one(
                item, agent, bootstrap_caller, state["config"]
            )
        caller = protocol_callers.get(agent)
        if caller is None:
            raise RuntimeError(f"no protocol endpoint is available for {agent}")
        if kind == "recovery":
            return index, advance_degenerate_recovery_one(
                item, caller, state["config"]
            )
        return index, advance_one(
            item,
            agent,
            caller,
            state["config"],
            recovery_caller=protocol_callers.get(DEGENERATE_RECOVERY_AGENT),
            defer_degenerate_recovery=True,
        )

    def next_index(agent: str) -> Optional[tuple[int, str]]:
        # Keep A3 producing bootstrap answers until they have all been
        # submitted. Completed answers immediately unlock A1/A2 protocol work,
        # while free A3 slots begin protocol work during the bootstrap tail.
        for preferred_kind in ("bootstrap", "recovery", "protocol"):
            for index, item in enumerate(state["items"]):
                if index in in_flight:
                    continue
                work = _ready_work(item)
                if work == (preferred_kind, agent):
                    return index, preferred_kind
        return None

    if capacity_provider is not None and max_inflight > 0:
        max_workers = max_inflight
    else:
        max_workers = max(
            1, sum(int(concurrency_by_agent[agent]) for agent in AGENT_IDS)
        )
    inflight_limit = max_workers if max_inflight == 0 else max_inflight
    if inflight_limit <= 0:
        raise ValueError("max_inflight must be positive when specified")
    positive_capacities = [
        int(concurrency_by_agent[agent])
        for agent in AGENT_IDS
        if int(concurrency_by_agent[agent]) > 0
    ]
    base_capacity = min(positive_capacities) if positive_capacities else 0

    def idle_overprovisioned_agents() -> list[str]:
        idle: list[str] = []
        for agent in AGENT_IDS:
            if current_capacity(agent) <= base_capacity:
                continue
            if active_by_agent[agent] == 0 and next_index(agent) is None:
                idle.append(agent)
        return idle

    notify_scheduler()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        while True:
            scheduled = 0
            if (
                max_advanced_per_pass > 0
                and advanced >= max_advanced_per_pass
            ):
                draining_for_rebalance = True
            if not draining_for_rebalance:
                for agent in AGENT_IDS:
                    capacity = current_capacity(agent)
                    if capacity <= 0:
                        continue
                    while (
                        active_by_agent[agent] < capacity
                        and len(futures) < inflight_limit
                    ):
                        candidate = next_index(agent)
                        if candidate is None:
                            break
                        index, kind = candidate
                        future = pool.submit(run_work, index, kind, agent)
                        futures[future] = (index, kind, agent)
                        in_flight.add(index)
                        active_by_agent[agent] += 1
                        scheduled += 1
            if scheduled:
                notify_scheduler()
            if not futures:
                remaining = [
                    index
                    for index, item in enumerate(state["items"])
                    if _ready_work(item) is not None
                ]
                if draining_for_rebalance:
                    break
                if remaining:
                    unavailable = []
                    for index in remaining:
                        work = _ready_work(state["items"][index])
                        if work is not None and current_capacity(work[1]) <= 0:
                            unavailable.append(work[1])
                    if not unavailable:
                        raise RuntimeError(
                            f"concurrent scheduler stalled with {len(remaining)} ready items"
                        )
                break
            completed, _ = wait(set(futures), return_when=FIRST_COMPLETED)
            for future in completed:
                index, kind, agent = futures.pop(future)
                active_by_agent[agent] -= 1
                in_flight.remove(index)
                result_index, item = future.result()
                if result_index != index:
                    raise RuntimeError("concurrent scheduler item identity drift")
                state["items"][index] = item
                if journal is not None:
                    journal.append(index, item)
                advanced += 1
                if advanced == 1 or advanced % max(1, progress_every) == 0:
                    elapsed = max(time.monotonic() - started, 1e-6)
                    print(
                        f"  pipeline_progress={advanced} rate={advanced / elapsed:.2f}/s "
                        f"bootstrap_pending={bootstrap_pending_count(state)} "
                        f"recovery_pending={recovery_pending_count(state)} "
                        f"protocol_pending={pending_count(state)} "
                        f"active=A1:{active_by_agent['A1']},A2:{active_by_agent['A2']},"
                        f"A3:{active_by_agent['A3']}",
                        flush=True,
                    )
            if completed:
                notify_scheduler()
            unavailable_ready_agents = sorted(
                {
                    work[1]
                    for index, item in enumerate(state["items"])
                    if index not in in_flight
                    and (work := _ready_work(item)) is not None
                    and current_capacity(work[1]) <= 0
                }
            )
            if unavailable_ready_agents and not draining_for_rebalance:
                print(
                    "  draining for elastic rebalance; no endpoint capacity for "
                    f"{','.join(unavailable_ready_agents)}",
                    flush=True,
                )
                draining_for_rebalance = True
            idle_agents = idle_overprovisioned_agents()
            remaining_work_agents = {
                agent
                for _, _, agent in futures.values()
            }
            remaining_work_agents.update(
                work[1]
                for index, item in enumerate(state["items"])
                if index not in in_flight
                and (work := _ready_work(item)) is not None
            )
            bottleneck_work_remains = any(
                agent not in idle_agents for agent in remaining_work_agents
            )
            if (
                capacity_provider is None
                and
                rebalance_after_advanced > 0
                and advanced >= rebalance_after_advanced
                and idle_agents
                and bottleneck_work_remains
                and not draining_for_rebalance
            ):
                print(
                    "  draining for tail rebalance after overprovisioned agents "
                    f"became idle: {','.join(idle_agents)}; active bottleneck="
                    f"{','.join(sorted(remaining_work_agents - set(idle_agents)))}",
                    flush=True,
                )
                draining_for_rebalance = True
            if scheduled == 0 and not completed and futures:
                raise RuntimeError("concurrent scheduler made no progress")

    state["phase_count"] = int(state.get("phase_count", 0)) + 1
    state["updated_at"] = utc_now()
    state.setdefault("history", []).append({
        "phase": state["phase_count"],
        "kind": "concurrent_pipeline",
        "advanced": advanced,
        "concurrency_by_agent": {
            agent: current_capacity(agent) for agent in AGENT_IDS
        },
        "dynamic_endpoint_capacity": capacity_provider is not None,
        "max_advanced_per_pass": max_advanced_per_pass,
        "max_inflight": max_inflight,
        "drained_for_rebalance": draining_for_rebalance,
        "unavailable_agents": [
            agent
            for agent in AGENT_IDS
            if int(concurrency_by_agent.get(agent, 0)) <= 0
        ],
        "bootstrap_pending_after": bootstrap_pending_count(state),
        "recovery_pending_after": recovery_pending_count(state),
        "pending_after": pending_count(state),
        "retry_after": status_count(state, "retry"),
        "total_steps_after": total_steps(state),
        "completed_at": state["updated_at"],
    })
    return advanced


def finalize_legacy_retries(state: Dict[str, Any]) -> int:
    finalized = 0
    for item in state["items"]:
        if item.get("status") not in {"retry", "failed"}:
            continue
        item["status"] = "done"
        item["current_agent"] = None
        finalized += 1
    if finalized:
        state["updated_at"] = utc_now()
    return finalized


def reset_retries(state: Dict[str, Any]) -> int:
    if not bool(state["config"].get("retry_failed_groups", False)):
        return finalize_legacy_retries(state)
    reset = 0
    max_attempts = int(state["config"]["group_retries"])
    for item in state["items"]:
        if item.get("status") != "retry":
            continue
        if int(item.get("group_attempt", 0)) >= max_attempts:
            item["status"] = "failed"
            continue
        reset_item(item, state["config"], increment_attempt=True)
        reset += 1
    state["updated_at"] = utc_now()
    return reset


def reset_rejected_items(state: Dict[str, Any]) -> int:
    reset = 0
    for item in state["items"]:
        trajectory = item.get("trajectory")
        if (
            item.get("status") != "done"
            or not isinstance(trajectory, dict)
            or trajectory.get("terminated_by") != "rejected_quality"
        ):
            continue
        preserved_bootstrap = copy.deepcopy(item.get("bootstrap"))
        reset_item(item, state["config"], increment_attempt=False)
        if (
            isinstance(preserved_bootstrap, dict)
            and preserved_bootstrap.get("status") == "done"
        ):
            item["bootstrap"] = preserved_bootstrap
            problem = problem_from_dict(item["problem"])
            item["messages"][1] = {
                "role": "user",
                "content": bootstrap_protocol_prompt(
                    problem,
                    bootstrap_visible=str(preserved_bootstrap["visible_output"]),
                    bootstrap_answer=str(preserved_bootstrap["final_answer"]),
                    protocol_agent=str(item["start_agent"]),
                    handoff_target=state["config"].get("bootstrap_handoff_target"),
                    enable_thinking=bool(state["config"]["enable_thinking"]),
                ),
            }
        reset += 1
    if reset:
        state["updated_at"] = utc_now()
    return reset


def failure_ledger_path(output: Path) -> Path:
    """Return the deterministic sidecar path for terminal zero-step failures."""

    return Path(f"{Path(output)}.failures.jsonl")


def _is_zero_step_terminal(item: Dict[str, Any]) -> bool:
    trajectory = item.get("trajectory")
    if item.get("status") not in {"done", "failed"} or not isinstance(trajectory, dict):
        return False
    return (
        not trajectory.get("steps")
        and trajectory.get("terminated_by") != "stop"
    )


def _zero_step_failure_items(state: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Return turn-mode terminal items that have no usable training row."""

    if state["config"].get("output_mode", "turns") != "turns":
        return []
    failures: list[Dict[str, Any]] = []
    for item in state["items"]:
        # A freshly initialized (or interrupted) pending item also has zero
        # steps and a default ``truncated`` marker, but it is not a terminal
        # failure and must remain eligible for generation/resume.
        if not _is_zero_step_terminal(item):
            continue
        failures.append(item)
    return failures


def zero_step_failure_records(state: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Build auditable records for turn-format trajectories with zero steps.

    These records deliberately stay outside the training JSONL: a malformed
    response cannot be represented as a valid turn.  The sidecar is the
    durable quarantine record used by ``finalize --resume``.
    """

    records: list[Dict[str, Any]] = []
    for item in _zero_step_failure_items(state):
        problem = problem_from_dict(item["problem"])
        trajectory = item["trajectory"]
        terminated_by = trajectory.get("terminated_by", "truncated")
        attempts = copy.deepcopy(item.get("generation_attempts", []))
        rejections = copy.deepcopy(item.get("generation_rejections", []))
        error = trajectory.get("error")
        if error is None:
            error = item.get("last_error")
        if not isinstance(error, str) or not error.strip():
            error = "unknown terminal failure"
        record = {
            "version": FAILURE_LEDGER_VERSION,
            "schema_version": FAILURE_LEDGER_VERSION,
            "problem_id": problem.problem_id,
            "problem_index": int(item["problem_index"]),
            "rollout": int(item["rollout_idx"]),
            "rollout_idx": int(item["rollout_idx"]),
            "n_steps": 0,
            # Persist the canonical problem projection rather than arbitrary
            # legacy metadata that may have been attached to a state item.
            "problem": problem_to_dict(problem),
            "start_agent": item.get("start_agent"),
            "status": item.get("status"),
            "group_attempt": int(item.get("group_attempt", 0)),
            "terminated_by": terminated_by,
            # ``termination`` is an explicit human-facing alias.  Keeping it
            # equal to ``terminated_by`` makes accidental edits detectable.
            "termination": terminated_by,
            "error": error,
            "generation_attempts": attempts,
            "generation_rejections": rejections,
            # Keep a compact, sampler-independent view for tooling that
            # consumes both legacy and role-batched sidecars.
            "attempt_diagnostics": attempts,
            "last_rejected_raw": copy.deepcopy(item.get("last_rejected_raw")),
        }
        records.append(record)
    return records


# Short alias for callers that use the terminology from the runbook.
failure_ledger_records = zero_step_failure_records


def _validate_failure_ledger_record(record: Any, *, path: Path, line_no: int) -> None:
    location = f"failure ledger {path} line {line_no}"
    if not isinstance(record, dict):
        raise ValueError(f"{location} must be an object")
    missing = _FAILURE_LEDGER_FIELDS - set(record)
    unknown = set(record) - _FAILURE_LEDGER_FIELDS
    if missing:
        raise ValueError(f"{location} missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{location} has unknown fields: {sorted(unknown)}")
    if record["version"] != FAILURE_LEDGER_VERSION or record["schema_version"] != FAILURE_LEDGER_VERSION:
        raise ValueError(
            f"{location} has unsupported version: {record['version']!r}"
        )
    if not isinstance(record["problem_id"], str) or not record["problem_id"].strip():
        raise ValueError(f"{location} has invalid problem_id")
    for field in ("problem_index", "rollout_idx", "group_attempt", "rollout", "n_steps"):
        if not _is_int(record[field]) or int(record[field]) < 0:
            raise ValueError(f"{location} has invalid {field}")
    if record["rollout"] != record["rollout_idx"]:
        raise ValueError(f"{location} rollout aliases disagree")
    if record["n_steps"] != 0:
        raise ValueError(f"{location} is not a zero-step failure")
    if not isinstance(record["problem"], dict):
        raise ValueError(f"{location}.problem must be an object")
    if set(record["problem"]) != _PROBLEM_FIELDS:
        raise ValueError(f"{location}.problem has invalid fields")
    for field in _PROBLEM_FIELDS:
        if not isinstance(record["problem"][field], str):
            raise ValueError(f"{location}.problem.{field} must be a string")
    if record["problem"]["problem_id"] != record["problem_id"]:
        raise ValueError(f"{location} problem_id does not match problem payload")
    if record["start_agent"] not in AGENT_IDS:
        raise ValueError(f"{location} has invalid start_agent")
    if record["status"] not in _STATE_STATUSES:
        raise ValueError(f"{location} has invalid status")
    if record["terminated_by"] not in _TERMINAL_REASONS:
        raise ValueError(f"{location} has invalid terminated_by")
    if record["terminated_by"] == "stop":
        raise ValueError(f"{location} cannot quarantine a stop trajectory")
    if record["termination"] != record["terminated_by"]:
        raise ValueError(f"{location} termination aliases disagree")
    if not isinstance(record["error"], str) or not record["error"].strip():
        raise ValueError(f"{location}.error must be a non-empty string")
    _validate_attempt_records(record["generation_attempts"], f"{location}.generation_attempts")
    _validate_attempt_records(
        record["generation_rejections"], f"{location}.generation_rejections"
    )
    if record["attempt_diagnostics"] != record["generation_attempts"]:
        raise ValueError(f"{location} attempt diagnostics disagree")
    if record["last_rejected_raw"] is not None and not isinstance(
        record["last_rejected_raw"], str
    ):
        raise ValueError(f"{location}.last_rejected_raw must be string or null")


def read_failure_ledger(path: Path) -> list[Dict[str, Any]]:
    """Read and strictly validate a failure sidecar."""

    path = Path(path)
    if not path.exists():
        return []
    entries: list[Dict[str, Any]] = []
    with path.open("rb") as handle:
        line_no = 0
        file_size = path.stat().st_size
        while True:
            raw = handle.readline()
            if not raw:
                break
            line_no += 1
            if not raw.endswith(b"\n"):
                raise ValueError(f"failure ledger {path} has unterminated line {line_no}")
            record = _strict_state_json_loads(raw, source=path, line_no=line_no)
            _validate_failure_ledger_record(record, path=path, line_no=line_no)
            entries.append(record)
        if handle.tell() != file_size:
            raise ValueError(f"failure ledger {path} could not be read completely")
    identities = [(entry["problem_index"], entry["rollout_idx"]) for entry in entries]
    if len(set(identities)) != len(identities):
        raise ValueError(f"failure ledger {path} contains duplicate trajectory identities")
    return entries


def validate_failure_ledger(
    path: Path, state: Dict[str, Any], *, require_present: bool = False
) -> list[Dict[str, Any]]:
    """Validate a sidecar against the current state, failing closed on drift."""

    path = Path(path)
    expected = zero_step_failure_records(state)
    if not path.exists():
        if require_present and expected:
            raise ValueError(
                f"failure ledger missing for {len(expected)} zero-step trajectory(s): {path}"
            )
        if expected:
            return expected
        return []
    actual = read_failure_ledger(path)
    if actual != expected:
        raise ValueError(
            f"failure ledger does not match state: {path} "
            f"(expected {len(expected)} entries, found {len(actual)})"
        )
    return actual


def write_failure_ledger(path: Path, entries: list[Dict[str, Any]]) -> None:
    """Atomically write a newline-delimited failure sidecar."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    identities = [
        (entry.get("problem_index"), entry.get("rollout_idx")) for entry in entries
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("failure ledger contains duplicate trajectory identities")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for entry in entries:
                _validate_failure_ledger_record(entry, path=path, line_no=1)
                handle.write(
                    json.dumps(entry, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def records(
    state: Dict[str, Any], *, allow_zero_step_failures: bool = False
) -> list[Dict[str, Any]]:
    if bootstrap_pending_count(state):
        raise ValueError("cannot finalize while bootstrap generations are pending")
    if bool(state["config"].get("retry_failed_groups", False)):
        if pending_count(state) or status_count(state, "retry"):
            raise ValueError("cannot finalize while trajectories are pending or retrying")
        if status_count(state, "failed"):
            if not allow_zero_step_failures:
                raise ValueError("cannot finalize trajectories exhausted group retries")
            non_quarantinable = [
                item
                for item in state["items"]
                if item.get("status") == "failed"
                and (
                    state["config"].get("output_mode", "turns") != "turns"
                    or not _is_zero_step_terminal(item)
                )
            ]
            if non_quarantinable:
                raise ValueError(
                    "cannot finalize trajectories with exhausted retries and valid turns; "
                    f"first item index={state['items'].index(non_quarantinable[0])}"
                )
    else:
        finalize_legacy_retries(state)
        if pending_count(state):
            raise ValueError("cannot finalize while trajectories are pending")
    output: list[Dict[str, Any]] = []
    for item in state["items"]:
        problem = problem_from_dict(item["problem"])
        if (
            state["config"].get("output_mode", "turns") == "turns"
            and not item["trajectory"].get("steps")
            and item["trajectory"].get("terminated_by") != "stop"
        ):
            if allow_zero_step_failures:
                continue
            raise ValueError(
                "cannot finalize a turn-format trajectory without a valid turn: "
                f"{problem.problem_id}/{item['rollout_idx']} "
                f"({item['trajectory'].get('terminated_by')})"
            )
        trajectory = base.MathTrajectory(
            problem_id=problem.problem_id,
            rollout_idx=int(item["rollout_idx"]),
            start_agent=str(item["start_agent"]),
            steps=[base.MathStep(**step) for step in item["trajectory"]["steps"]],
            turn_messages=item["trajectory"].get("turn_messages", []),
            final_answer=item["trajectory"].get("final_answer"),
            terminated_by=item["trajectory"].get("terminated_by", "truncated"),
            error=item["trajectory"].get("error"),
        )
        if state["config"].get("output_mode", "turns") == "trajectories":
            record = base.trajectory_record(
                problem,
                trajectory,
                thinking_enabled=bool(state["config"]["enable_thinking"]),
            )
            if item.get("bootstrap") is not None:
                bootstrap_record = copy.deepcopy(item["bootstrap"])
                record["bootstrap"] = bootstrap_record
                record["trajectory"]["bootstrap"] = copy.deepcopy(bootstrap_record)
            has_recovery_handoff = any(
                step.action == "handoff"
                and step.handoff_target != trajectory.steps[index + 1].active_agent
                for index, step in enumerate(trajectory.steps[:-1])
            )
            # Keep rejected generation material available for terminal
            # failures and for successful trajectories whose active-agent
            # sequence changed because A3 replaced a collapsed model.  The
            # latter audit is required to distinguish an intentional recovery
            # from a corrupted handoff chain in the finalized artifact.
            if (
                trajectory.terminated_by != "stop"
                or not trajectory.steps
                or has_recovery_handoff
            ):
                attempts = copy.deepcopy(item.get("generation_attempts", []))
                rejections = copy.deepcopy(item.get("generation_rejections", []))
                rejected_raw = item.get("last_rejected_raw")
                record["generation_attempts"] = attempts
                record["generation_rejections"] = rejections
                record["last_rejected_raw"] = rejected_raw
                audit_field = (
                    "degenerate_recovery_audit"
                    if has_recovery_handoff
                    else "terminal_failure_audit"
                )
                record[audit_field] = {
                    "attempts": attempts,
                    "rejections": rejections,
                    "last_rejected_raw": rejected_raw,
                }
                record["trajectory"][audit_field] = copy.deepcopy(record[audit_field])
            output.append(record)
        else:
            output.extend(
                base.turn_records(
                    problem,
                    trajectory,
                    thinking_enabled=bool(state["config"]["enable_thinking"]),
                )
            )
    return output


def write_jsonl(path: Path, rows: list[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MATH global-turn role-batched runner")
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--state", type=Path, required=True)
    init.add_argument("--data-root", type=Path, required=True)
    init.add_argument("--split", choices=["train", "test"], required=True)
    init.add_argument("--subjects", default="")
    init.add_argument("--start", type=int, default=0)
    init.add_argument("--limit", type=int, required=True)
    init.add_argument("--num-rollouts", type=int, default=1)
    init.add_argument("--t-max", type=int, default=8)
    init.add_argument("--start-agent", choices=[*AGENT_IDS, "balanced", "random"], default="A1")
    init.add_argument("--bootstrap-agent", choices=AGENT_IDS)
    init.add_argument("--bootstrap-handoff-target", choices=AGENT_IDS)
    init.add_argument(
        "--preserve-reasonable-incumbent",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    init.add_argument(
        "--lock-upstream-answer-agents",
        default="",
        help="comma-separated agents that must copy the upstream answer unchanged",
    )
    init.add_argument("--start-agent-seed", type=int, default=42)
    init.add_argument("--min-agents-before-stop", type=int, default=1)
    init.add_argument(
        "--allow-first-turn-stop", action=argparse.BooleanOptionalAction, default=False
    )
    init.add_argument("--generation-seed", type=int, default=42)
    init.add_argument("--max-new-tokens", type=int, default=8192)
    init.add_argument("--protocol-thinking-max-tokens", type=int)
    init.add_argument("--protocol-max-tokens", type=int)
    init.add_argument("--temperature", type=float, default=0.9)
    init.add_argument("--top-p", type=float, default=0.95)
    init.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    init.add_argument("--require-thinking", action=argparse.BooleanOptionalAction, default=False)
    init.add_argument("--api-timeout", type=float, default=900)
    init.add_argument("--group-retries", type=int, default=5)
    init.add_argument(
        "--retry-failed-groups", action=argparse.BooleanOptionalAction, default=False
    )
    init.add_argument("--step-retries", type=int, default=4)
    init.add_argument(
        "--json-transport",
        choices=["json_schema", "json_object", "none"],
        default="json_object",
    )
    init.add_argument("--output-mode", choices=["turns", "trajectories"], default="turns")

    for command in (
        "pending",
        "bootstrap-pending",
        "steps",
        "status",
        "reset-retries",
        "reset-rejected",
    ):
        item = sub.add_parser(command)
        item.add_argument("--state", type=Path, required=True)
        if command == "pending":
            item.add_argument("--agent", choices=[*AGENT_IDS, "all"], default="all")
            item.add_argument("--turn", type=int)
        elif command == "bootstrap-pending":
            item.add_argument("--agent", choices=[*AGENT_IDS, "all"], default="all")
    run = sub.add_parser("run-agent")
    run.add_argument("--state", type=Path, required=True)
    run.add_argument("--agent", choices=AGENT_IDS, required=True)
    run.add_argument("--api-base", required=True)
    run.add_argument("--api-model", required=True)
    run.add_argument("--api-key", default="EMPTY")
    run.add_argument("--turn", type=int)
    run.add_argument("--max-concurrency", type=int, default=128)
    run.add_argument("--log-raw-chars", type=int, default=0)
    run.add_argument("--journal-fsync-every", type=int, default=50)
    run_bootstrap = sub.add_parser("run-bootstrap")
    run_bootstrap.add_argument("--state", type=Path, required=True)
    run_bootstrap.add_argument("--agent", choices=AGENT_IDS, required=True)
    run_bootstrap.add_argument("--api-base", required=True)
    run_bootstrap.add_argument("--api-model", required=True)
    run_bootstrap.add_argument("--api-key", default="EMPTY")
    run_bootstrap.add_argument("--max-concurrency", type=int, default=128)
    run_bootstrap.add_argument("--journal-fsync-every", type=int, default=50)
    run_concurrent = sub.add_parser("run-concurrent")
    run_concurrent.add_argument("--state", type=Path, required=True)
    for agent in AGENT_IDS:
        key = agent.lower()
        run_concurrent.add_argument(
            f"--api-base-{key}",
            required=True,
            help="one API base or a comma-separated endpoint pool",
        )
        run_concurrent.add_argument(f"--api-model-{key}", default=agent)
        run_concurrent.add_argument(f"--max-concurrency-{key}", type=int, required=True)
    run_concurrent.add_argument("--api-key", default="EMPTY")
    run_concurrent.add_argument("--progress-every", type=int, default=25)
    run_concurrent.add_argument("--journal-fsync-every", type=int, default=25)
    run_concurrent.add_argument("--rebalance-after-advanced", type=int, default=0)
    run_concurrent.add_argument("--max-advanced-per-pass", type=int, default=0)
    run_concurrent.add_argument("--max-inflight", type=int, default=0)
    run_concurrent.add_argument("--live-endpoint-registry", type=Path)
    run_concurrent.add_argument("--live-endpoint-status", type=Path)
    run_concurrent.add_argument("--endpoint-concurrency", type=int, default=0)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--state", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "init":
        if args.state.exists():
            raise SystemExit(f"state exists: {args.state}")
        if args.limit <= 0 or args.num_rollouts <= 0 or args.t_max <= 0:
            raise SystemExit("limit, num-rollouts, and t-max must be positive")
        subjects = [value.strip() for value in args.subjects.split(",") if value.strip()]
        lock_upstream_answer_agents = [
            value.strip()
            for value in args.lock_upstream_answer_agents.split(",")
            if value.strip()
        ]
        all_problems = load_math_problems(args.data_root, split=args.split, subjects=subjects)
        selected = all_problems[args.start : args.start + args.limit]
        if len(selected) != args.limit:
            raise SystemExit(f"requested {args.limit} problems but selected {len(selected)}")
        if args.require_thinking and not args.enable_thinking:
            raise SystemExit("--require-thinking requires --enable-thinking")
        state = create_state(
            selected,
            start=args.start,
            data_path=str(args.data_root.resolve()),
            split=args.split,
            subjects=subjects,
            t_max=args.t_max,
            start_agent=args.start_agent,
            start_agent_seed=args.start_agent_seed,
            min_agents_before_stop=args.min_agents_before_stop,
            allow_first_turn_stop=args.allow_first_turn_stop,
            num_rollouts=args.num_rollouts,
            generation_seed=args.generation_seed,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            enable_thinking=args.enable_thinking,
            require_thinking=args.require_thinking,
            api_timeout=args.api_timeout,
            group_retries=args.group_retries,
            retry_failed_groups=args.retry_failed_groups,
            step_retries=args.step_retries,
            json_transport=args.json_transport,
            output_mode=args.output_mode,
            bootstrap_agent=args.bootstrap_agent,
            bootstrap_handoff_target=args.bootstrap_handoff_target,
            preserve_reasonable_incumbent=args.preserve_reasonable_incumbent,
            lock_upstream_answer_agents=lock_upstream_answer_agents,
            protocol_thinking_max_tokens=args.protocol_thinking_max_tokens,
            protocol_max_tokens=args.protocol_max_tokens,
        )
        save_state(args.state, state)
        print(f"Initialized {len(state['items'])} trajectories; json_transport={args.json_transport}")
        return

    state = load_state(args.state)
    if args.command == "pending":
        agent = None if args.agent == "all" else args.agent
        print(pending_count(state, agent, args.turn))
        return
    if args.command == "bootstrap-pending":
        agent = None if args.agent == "all" else args.agent
        print(bootstrap_pending_count(state, agent))
        return
    if args.command == "steps":
        print(total_steps(state))
        return
    if args.command == "status":
        zero_step_failures = len(zero_step_failure_records(state))
        failed_zero_step_failures = sum(
            state["config"].get("output_mode", "turns") == "turns"
            and item.get("status") == "failed"
            and _is_zero_step_terminal(item)
            for item in state["items"]
        )
        print(json.dumps({
            "items": len(state["items"]),
            "phase_count": state.get("phase_count", 0),
            "total_steps": total_steps(state),
            "pending_total": pending_count(state),
            "bootstrap_pending": bootstrap_pending_count(state),
            "recovery_pending": recovery_pending_count(state),
            "pending_by_agent": {agent: pending_count(state, agent) for agent in AGENT_IDS},
            "retry": status_count(state, "retry"),
            "failed": status_count(state, "failed"),
            "done": status_count(state, "done"),
            "zero_step_failures": zero_step_failures,
            "failed_zero_step_failures": failed_zero_step_failures,
            "json_transport": state["config"].get("json_transport"),
        }, sort_keys=True))
        return
    if args.command == "run-bootstrap":
        config = state["config"]
        if config.get("bootstrap_agent") != args.agent:
            raise SystemExit(
                f"bootstrap agent mismatch: state={config.get('bootstrap_agent')} cli={args.agent}"
            )
        caller = OpenAIChatLLMCaller(
            args.api_base,
            args.api_model,
            generation=GenerationOptions(
                max_new_tokens=int(config["max_new_tokens"]),
                temperature=float(config["temperature"]),
                top_p=float(config["top_p"]),
                enable_thinking=bool(config["enable_thinking"]),
            ),
            timeout=float(config["api_timeout"]),
            api_key=args.api_key,
            response_format=None,
        )
        queued = bootstrap_pending_count(state, args.agent)
        started = time.monotonic()
        with StateJournal(args.state, args.journal_fsync_every) as journal:
            advanced = run_bootstrap_phase(
                state, args.agent, caller, args.max_concurrency, journal
            )
        save_state(args.state, state)
        print(
            f"bootstrap phase agent={args.agent} queued={queued} advanced={advanced} "
            f"bootstrap_pending={bootstrap_pending_count(state)} "
            f"pending={pending_count(state)} "
            f"rate={advanced / max(time.monotonic() - started, 1e-6):.2f}/s"
        )
        return
    if args.command == "run-concurrent":
        config = state["config"]
        concurrency = {
            agent: int(getattr(args, f"max_concurrency_{agent.lower()}"))
            for agent in AGENT_IDS
        }
        endpoint_load_balancers: Dict[tuple[str, ...], EndpointLoadBalancer] = {}

        if bool(args.live_endpoint_registry) != bool(args.live_endpoint_status):
            raise SystemExit(
                "--live-endpoint-registry and --live-endpoint-status must be used together"
            )
        if args.live_endpoint_registry and args.endpoint_concurrency <= 0:
            raise SystemExit("--endpoint-concurrency must be positive in live endpoint mode")

        def make_openai_caller(
            agent: str,
            api_base: str,
            *,
            bootstrap: bool,
        ) -> OpenAIChatLLMCaller:
            key = agent.lower()
            return OpenAIChatLLMCaller(
                api_base,
                getattr(args, f"api_model_{key}"),
                generation=GenerationOptions(
                    max_new_tokens=(
                        int(config["max_new_tokens"])
                        if bootstrap
                        else protocol_max_new_tokens(config)
                    ),
                    temperature=float(config["temperature"]),
                    top_p=float(config["top_p"]),
                    enable_thinking=bool(config["enable_thinking"]),
                ),
                timeout=float(config["api_timeout"]),
                api_key=args.api_key,
                response_format=(
                    None
                    if bootstrap
                    else build_response_format(
                        str(config.get("json_transport", "none")),
                        enable_thinking=bool(config.get("enable_thinking", False)),
                        active_agent=agent,
                        allow_self_handoff=(
                            config.get("bootstrap_agent") == agent
                            and config.get("bootstrap_handoff_target") == agent
                        ),
                    )
                ),
                length_retries=1 if bootstrap else 0,
                thinking_fallback=not bootstrap,
                thinking_max_tokens=(
                    None
                    if bootstrap
                    else int(config["protocol_thinking_max_tokens"])
                ),
                thinking_stop=None if bootstrap else [DEGENERATE_THINKING_STOP],
                fallback_bad_words=(
                    None if bootstrap else DEGENERATE_FALLBACK_BAD_WORDS
                ),
            )

        def make_caller(agent: str, *, bootstrap: bool = False) -> Any:
            key = agent.lower()
            api_bases = parse_api_base_pool(
                getattr(args, f"api_base_{key}"),
                allow_empty=concurrency.get(agent, 0) == 0,
            )
            if not api_bases:
                return None
            load_balancer = endpoint_load_balancers.get(api_bases)
            if load_balancer is None:
                load_balancer = EndpointLoadBalancer(len(api_bases))
                endpoint_load_balancers[api_bases] = load_balancer
            endpoint_callers = [
                make_openai_caller(agent, api_base, bootstrap=bootstrap)
                for api_base in api_bases
            ]
            if len(endpoint_callers) == 1:
                return endpoint_callers[0]
            return OpenAIEndpointPoolCaller(endpoint_callers, load_balancer)

        live_registry: Optional[LiveEndpointRegistry] = None
        if args.live_endpoint_registry is not None:
            live_registry = LiveEndpointRegistry(
                args.live_endpoint_registry,
                args.live_endpoint_status,
                args.endpoint_concurrency,
            )

            def make_dynamic_caller(agent: str, *, bootstrap: bool = False) -> Any:
                def factory(api_base: str) -> OpenAIChatLLMCaller:
                    return make_openai_caller(agent, api_base, bootstrap=bootstrap)

                return DynamicOpenAIEndpointCaller(live_registry, agent, factory)

            callers = {agent: make_dynamic_caller(agent) for agent in AGENT_IDS}
            bootstrap_caller = make_dynamic_caller("A3", bootstrap=True)
            capacity_provider: Optional[Callable[[str], int]] = live_registry.capacity

            def scheduler_observer(
                observed_state: Dict[str, Any],
                active_by_agent: Dict[str, int],
                advanced: int,
                in_flight: int,
            ) -> None:
                live_registry.publish_workload(
                    observed_state,
                    active_by_agent,
                    advanced=advanced,
                    in_flight=in_flight,
                )
        else:
            callers = {agent: make_caller(agent) for agent in AGENT_IDS}
            bootstrap_caller = make_caller("A3", bootstrap=True)
            capacity_provider = None
            scheduler_observer = None
        started = time.monotonic()
        try:
            with StateJournal(args.state, args.journal_fsync_every) as journal:
                advanced = run_concurrent_pipeline(
                    state,
                    bootstrap_caller=bootstrap_caller,
                    protocol_callers=callers,
                    concurrency_by_agent=concurrency,
                    capacity_provider=capacity_provider,
                    scheduler_observer=scheduler_observer,
                    journal=journal,
                    progress_every=args.progress_every,
                    rebalance_after_advanced=(
                        0 if live_registry is not None else args.rebalance_after_advanced
                    ),
                    max_advanced_per_pass=args.max_advanced_per_pass,
                    max_inflight=args.max_inflight,
                )
        finally:
            if live_registry is not None:
                live_registry.close()
        save_state(args.state, state)
        print(
            f"concurrent pipeline advanced={advanced} "
            f"bootstrap_pending={bootstrap_pending_count(state)} "
            f"recovery_pending={recovery_pending_count(state)} "
            f"pending={pending_count(state)} retry={status_count(state, 'retry')} "
            f"done={status_count(state, 'done')} "
            f"rate={advanced / max(time.monotonic() - started, 1e-6):.2f}/s"
        )
        return
    if args.command == "reset-retries":
        print(f"reset={reset_retries(state)}")
        save_state(args.state, state)
        return
    if args.command == "reset-rejected":
        print(f"reset={reset_rejected_items(state)}")
        save_state(args.state, state)
        return
    if args.command == "run-agent":
        config = state["config"]
        response_format = build_response_format(
            str(config.get("json_transport", "none")),
            enable_thinking=bool(config.get("enable_thinking", False)),
            active_agent=args.agent,
        )
        print(
            f"[transport] protocol={config.get('json_transport')} "
            f"thinking_cap={config.get('protocol_thinking_max_tokens')}",
            flush=True,
        )
        caller = OpenAIChatLLMCaller(
            args.api_base,
            args.api_model,
            generation=GenerationOptions(
                max_new_tokens=protocol_max_new_tokens(config),
                temperature=float(config["temperature"]),
                top_p=float(config["top_p"]),
                enable_thinking=bool(config["enable_thinking"]),
            ),
            timeout=float(config["api_timeout"]),
            api_key=args.api_key,
            response_format=response_format,
            length_retries=0,
            thinking_fallback=True,
            thinking_max_tokens=int(config["protocol_thinking_max_tokens"]),
            thinking_stop=[DEGENERATE_THINKING_STOP],
            fallback_bad_words=DEGENERATE_FALLBACK_BAD_WORDS,
        )
        queued = pending_count(state, args.agent, args.turn)
        started = time.monotonic()
        with StateJournal(args.state, args.journal_fsync_every) as journal:
            advanced = run_agent_phase(
                state,
                args.agent,
                caller,
                args.max_concurrency,
                args.turn,
                journal,
            )
        save_state(args.state, state)
        print(
            f"phase agent={args.agent} queued={queued} advanced={advanced} "
            f"pending={pending_count(state)} retry={status_count(state, 'retry')} "
            f"rate={advanced / max(time.monotonic() - started, 1e-6):.2f}/s"
        )
        return
    if args.command == "finalize":
        if state_journal_path(args.state).exists():
            save_state(args.state, state)
            print(f"[resume] compacted state journal into {args.state}")
        # Turn mode cannot encode a zero-step malformed/failed generation as a
        # training row.  Finalization quarantines those groups in a sidecar;
        # direct callers of records() retain the historical fail-closed error.
        rows = records(state, allow_zero_step_failures=True)
        failures = zero_step_failure_records(state)
        ledger = failure_ledger_path(args.output)
        if args.output.exists():
            if not args.resume:
                raise SystemExit(f"output exists: {args.output}")
            with args.output.open(encoding="utf-8") as handle:
                for index, expected in enumerate(rows, 1):
                    line = handle.readline()
                    if not line:
                        raise SystemExit(
                            f"existing output is incomplete at row {index}: {args.output}"
                        )
                    try:
                        actual = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise SystemExit(
                            f"existing output has invalid JSON at row {index}: {args.output}"
                        ) from exc
                    if actual != expected:
                        raise SystemExit(
                            f"existing output differs from state at row {index}: {args.output}"
                        )
                if any(line.strip() for line in handle):
                    raise SystemExit(f"existing output has extra rows: {args.output}")
            try:
                validate_failure_ledger(ledger, state, require_present=bool(failures))
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            if failures:
                print(
                    f"[resume] output is reproducible but incomplete: rows={len(rows)} "
                    f"zero_step_failures={len(failures)} ledger={ledger}",
                    file=sys.stderr,
                )
                raise SystemExit(1)
            if ledger.exists():
                raise SystemExit(f"stale failure ledger exists for a complete state: {ledger}")
            print(
                f"[resume] finalized output complete: rows={len(rows)} output={args.output}"
            )
            return

        # If a sidecar survived an interrupted output write, validate it before
        # reusing it.  A mismatched stale ledger is never silently overwritten.
        if ledger.exists():
            try:
                validate_failure_ledger(ledger, state, require_present=bool(failures))
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
        if failures:
            write_failure_ledger(ledger, failures)
        elif ledger.exists():
            raise SystemExit(f"stale failure ledger exists for a complete state: {ledger}")
        write_jsonl(args.output, rows)
        if failures:
            print(
                f"MATH role-batched finalize incomplete: rows={len(rows)} "
                f"zero_step_failures={len(failures)} ledger={ledger}",
                file=sys.stderr,
            )
            raise SystemExit(1)
        print(f"MATH role-batched finalize: rows={len(rows)} output={args.output}")
        return
    raise AssertionError(args.command)


if __name__ == "__main__":
    main()

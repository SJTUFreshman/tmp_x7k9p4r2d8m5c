"""Stateful, role-batched GSM-HARD MAS inference.

This runner preserves the trajectory semantics from ``run_mas.py`` while
allowing a launcher to serve only one agent at a time.  A phase advances every
trajectory currently waiting for one role, checkpoints the updated queues, and
then exits so the launcher can replace the vLLM server with the next adapter.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jca.gsm.scripts import run_mas as base
from jca.gsm.src.data import GSMProblem, format_problem_as_prompt, load_gsm_hard
from jca.gsm.src.grader import compute_em_f1
from jca.src.inference import GenerationOptions, OpenAIChatLLMCaller


STATE_VERSION = 1
JSON_TRANSPORTS = ("json_schema", "json_object", "none")


def _strip_thinking(raw: str) -> str:
    """Remove Qwen3 thinking blocks before protocol parsing and persistence."""
    text = str(raw or "")
    text = re.sub(
        r"<think\b[^>]*>.*?</think\s*>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    # A length-truncated response may never emit </think>. Everything after
    # the unmatched opener is private reasoning, not a protocol response.
    unclosed = re.search(r"<think\b[^>]*>", text, flags=re.IGNORECASE)
    if unclosed is not None:
        text = text[:unclosed.start()]
    # Be defensive about a server/parser that leaves only a closing tag.
    text = re.sub(r"</?think\b[^>]*>", "", text, flags=re.IGNORECASE)
    return text.strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_response_format(
    json_transport: str,
    agent_id: str,
) -> Optional[Dict[str, Any]]:
    if json_transport == "none":
        return None
    if json_transport == "json_object":
        return {"type": "json_object"}
    if json_transport != "json_schema":
        raise ValueError(f"unsupported JSON transport: {json_transport}")
    nullable_string = {
        "anyOf": [
            {"type": "string"},
            {"type": "null"},
        ]
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"gsm_mas_{agent_id.lower()}_turn",
            "description": "One structured GSM multi-agent collaboration turn.",
            "schema": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "tentative_answer": {"type": "string"},
                    "action": {
                        "type": "string",
                        "enum": ["handoff", "confirm_stop"],
                    },
                    "handoff_target": {
                        "anyOf": [
                            {
                                "type": "string",
                                "enum": [
                                    target
                                    for target in base.AGENT_IDS
                                    if target != agent_id
                                ],
                            },
                            {"type": "null"},
                        ]
                    },
                    "handoff_note": nullable_string,
                    "confirmed_answer": nullable_string,
                },
                "required": [
                    "reasoning",
                    "tentative_answer",
                    "action",
                    "handoff_target",
                    "handoff_note",
                    "confirmed_answer",
                ],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }


def _problem_to_dict(problem: GSMProblem) -> Dict[str, Any]:
    return {
        "id": problem.id,
        "question": problem.question,
        "answer": problem.answer,
        "answer_str": problem.answer_str,
    }


def _problem_from_dict(payload: Dict[str, Any]) -> GSMProblem:
    return GSMProblem(
        id=str(payload["id"]),
        question=str(payload["question"]),
        answer=float(payload["answer"]),
        answer_str=str(payload["answer_str"]),
    )


def _trajectory_from_item(item: Dict[str, Any]) -> base.GSMTrajectory:
    payload = item["trajectory"]
    return base.GSMTrajectory(
        problem_id=str(payload["problem_id"]),
        generation_seed=payload.get("generation_seed"),
        protocol_mode=str(payload.get("protocol_mode", "default")),
        min_handoffs_before_stop=int(payload.get("min_handoffs_before_stop", 0)),
        steps=[base.GSMStep(**step) for step in payload.get("steps", [])],
        generation_attempts=list(item.get("generation_attempts", [])),
        final_answer=payload.get("final_answer"),
        terminated_by=str(payload.get("terminated_by", "truncated")),
        error=payload.get("error"),
    )


def _item_start_agent(item: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Resolve the immutable first role, including states created before it was stored."""
    stored = str(item.get("start_agent") or "")
    if stored in base.AGENT_IDS:
        return stored
    steps = item.get("trajectory", {}).get("steps", [])
    if steps:
        first_step_agent = str(steps[0].get("active_agent") or "")
        if first_step_agent in base.AGENT_IDS:
            return first_step_agent
    current_agent = str(item.get("current_agent") or "")
    if not steps and current_agent in base.AGENT_IDS:
        return current_agent
    return base.select_start_agent(
        int(item["problem_index"]),
        str(config.get("start_agent", "A1")),
        int(config.get("start_agent_seed", 42)),
    )


def _start_agent_counts(state: Dict[str, Any]) -> Dict[str, int]:
    counts = Counter(
        _item_start_agent(item, state["config"]) for item in state["items"]
    )
    return {agent: counts.get(agent, 0) for agent in base.AGENT_IDS}


def _active_agents(item: Dict[str, Any]) -> list[str]:
    agents: list[str] = []
    for step in item["trajectory"]["steps"]:
        agent = str(step["active_agent"])
        if agent not in agents:
            agents.append(agent)
    return agents


def _handoff_count(item: Dict[str, Any]) -> int:
    return sum(
        1
        for step in item["trajectory"]["steps"]
        if step["action"] == "handoff"
    )


def _system_prompt(
    item: Dict[str, Any],
    agent_id: str,
    config: Dict[str, Any],
) -> str:
    if config["sft_warmup_prompt"]:
        problem = _problem_from_dict(item["problem"])
        return base.render_sft_rollout_system_prompt(
            agent_id,
            min_agents_before_stop=config["min_agents_before_stop"],
            min_handoffs_before_stop=item["trajectory"]["min_handoffs_before_stop"],
            reference_answer=(
                problem.answer_str if config["sft_controlled_generation"] else None
            ),
        )
    return base.render_system_prompt(
        agent_id,
        min_agents_before_stop=config["min_agents_before_stop"],
    )


def create_evaluation_state(
    problems: Iterable[GSMProblem],
    *,
    start_index: int,
    data_path: str,
    t_max: int,
    start_agent: str,
    start_agent_seed: int = 42,
    min_agents_before_stop: int,
    enforce_collaboration_policy: bool,
    generation_seed: Optional[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool = False,
    api_timeout: int,
    json_transport: str = "json_schema",
    sft_warmup_prompt: bool = False,
    sft_extended_fraction: float = 0.4,
    sft_protocol_seed: int = 42,
    sft_controlled_generation: bool = False,
    sft_max_step_retries: int = 2,
    sft_max_verifier_similarity: float = 0.85,
    sft_standard_min_handoffs: int = 2,
    sft_extended_min_handoffs: Optional[list[int]] = None,
) -> Dict[str, Any]:
    if json_transport not in JSON_TRANSPORTS:
        raise ValueError(f"unsupported JSON transport: {json_transport}")
    selected = list(problems)
    extended_handoffs = sft_extended_min_handoffs or [3, 4]
    config: Dict[str, Any] = {
        "data_path": data_path,
        "start": start_index,
        "limit": len(selected),
        "t_max": t_max,
        "start_agent": start_agent,
        "start_agent_seed": start_agent_seed,
        "min_agents_before_stop": min_agents_before_stop,
        "enforce_collaboration_policy": enforce_collaboration_policy,
        "generation_seed": generation_seed,
        "json_transport": json_transport,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "enable_thinking": enable_thinking,
        "api_timeout": api_timeout,
        "sft_warmup_prompt": sft_warmup_prompt,
        "sft_extended_fraction": sft_extended_fraction,
        "sft_protocol_seed": sft_protocol_seed,
        "sft_controlled_generation": sft_controlled_generation,
        "sft_max_step_retries": sft_max_step_retries,
        "sft_max_verifier_similarity": sft_max_verifier_similarity,
        "sft_standard_min_handoffs": sft_standard_min_handoffs,
        "sft_extended_min_handoffs": extended_handoffs,
        "thinking_output_hidden": True,
        "allow_self_handoff": True,
    }
    state: Dict[str, Any] = {
        "version": STATE_VERSION,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "phase_count": 0,
        "config": config,
        "history": [],
        "items": [],
    }

    for offset, problem in enumerate(selected):
        problem_index = start_index + offset
        protocol_mode = "default"
        min_handoffs_before_stop = 0
        if sft_warmup_prompt:
            protocol_mode, min_handoffs_before_stop = base.select_sft_protocol(
                problem_index,
                extended_fraction=sft_extended_fraction,
                seed=sft_protocol_seed,
                standard_min_handoffs=sft_standard_min_handoffs,
                extended_min_handoffs=extended_handoffs,
            )
        trajectory_start_agent = base.select_start_agent(
            problem_index,
            start_agent,
            start_agent_seed,
        )
        item: Dict[str, Any] = {
            "problem_index": problem_index,
            "problem": _problem_to_dict(problem),
            "trajectory": {
                "problem_id": problem.id,
                "generation_seed": generation_seed,
                "protocol_mode": protocol_mode,
                "min_handoffs_before_stop": min_handoffs_before_stop,
                "steps": [],
                "final_answer": None,
                "terminated_by": "truncated",
                "error": None,
            },
            "messages": [],
            "start_agent": trajectory_start_agent,
            "current_agent": trajectory_start_agent,
            "prior_tentative": False,
            "prior_tentative_answer": None,
            "prior_reasonings": [],
            "status": "pending",
        }
        item["messages"] = [
            {
                "role": "system",
                "content": _system_prompt(item, trajectory_start_agent, config),
            },
            {"role": "user", "content": format_problem_as_prompt(problem)},
        ]
        state["items"].append(item)
    return state


def _terminate(
    item: Dict[str, Any],
    terminated_by: str,
    *,
    error: Optional[str] = None,
) -> None:
    item["trajectory"]["terminated_by"] = terminated_by
    item["trajectory"]["error"] = error
    item["status"] = "done"
    item["current_agent"] = None


def _advance_one(
    item: Dict[str, Any],
    agent_id: str,
    caller: Any,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    problem = _problem_from_dict(item["problem"])
    trajectory = item["trajectory"]
    turn = len(trajectory["steps"])
    if item["status"] != "pending" or item["current_agent"] != agent_id:
        raise ValueError(f"trajectory {problem.id} is not pending for {agent_id}")
    if turn >= config["t_max"]:
        _terminate(item, "truncated")
        return item

    try:
        retry_reason: Optional[str] = None
        accepted_step: Optional[base.GSMStep] = None
        raw_outputs: list[str] = []
        controlled = config["sft_controlled_generation"]
        attempts = config["sft_max_step_retries"] + 1 if controlled else 2

        for attempt in range(attempts):
            request_messages = [dict(message) for message in item["messages"]]
            if controlled:
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

            request_seed = base.derive_generation_request_seed(
                config["generation_seed"],
                problem.id,
                turn,
                attempt,
                agent_id,
            )
            raw = (
                caller(request_messages)
                if request_seed is None
                else caller.generate(request_messages, seed=request_seed)
            )
            transport_outputs = base.response_attempts(raw)
            raw_outputs.extend(transport_outputs)
            visible_raw = _strip_thinking(raw)
            candidate = base.parse_action(
                visible_raw,
                agent_id,
                item["prior_tentative"],
                repair_premature_confirm=config["enforce_collaboration_policy"],
            )
            candidate.turn = turn
            candidate_reason: Optional[str] = None
            if controlled:
                controlled_reason = base.sft_step_quality_reason(
                    candidate,
                    gold_answer=problem.answer_str,
                    prior_reasonings=item["prior_reasonings"],
                    max_verifier_similarity=config["sft_max_verifier_similarity"],
                )
                retry_reason = candidate_reason or controlled_reason
            else:
                retry_reason = candidate_reason or base.step_parse_failure_reason(candidate)
            item.setdefault("generation_attempts", []).append({
                "turn": turn,
                "agent": agent_id,
                "attempt": attempt + 1,
                "accepted": retry_reason is None,
                "reason": retry_reason,
                "raw_output": str(raw),
                "raw_outputs": transport_outputs,
                "visible_raw_output": visible_raw,
            })
            if retry_reason is not None:
                item.setdefault("generation_rejections", []).append({
                    "turn": turn,
                    "agent": agent_id,
                    "attempt": attempt + 1,
                    "reason": retry_reason,
                    "raw_output": str(raw),
                    "raw_outputs": transport_outputs,
                    "visible_raw_output": visible_raw,
                })
            if retry_reason is None:
                candidate.raw_output = str(raw)
                candidate.raw_outputs = list(raw_outputs)
                candidate.visible_raw_output = visible_raw
                accepted_step = candidate
                break

        if accepted_step is None:
            _terminate(
                item,
                "rejected_quality",
                error=(
                    f"turn {turn} agent {agent_id} failed generation quality "
                    f"after {attempts} attempts: {retry_reason}"
                ),
            )
            return item

        if config["enforce_collaboration_policy"]:
            accepted_step = base.enforce_collaboration_policy(
                accepted_step,
                seen_agents_before=_active_agents(item),
                min_agents_before_stop=config["min_agents_before_stop"],
                handoffs_before=_handoff_count(item),
                min_handoffs_before_stop=trajectory["min_handoffs_before_stop"],
                prior_tentative_answer=item["prior_tentative_answer"],
                balanced_routing=controlled,
                stop_when_ready=controlled,
            )

        trajectory["steps"].append(asdict(accepted_step))
        if accepted_step.tentative_answer:
            item["prior_tentative"] = True
            item["prior_tentative_answer"] = accepted_step.tentative_answer
        item["prior_reasonings"].append(accepted_step.reasoning)
        item["messages"].append({
            "role": "assistant",
            "content": base.render_assistant_message(accepted_step),
        })

        if accepted_step.confirmed_answer is not None or (
            accepted_step.action == "confirm_stop" and accepted_step.tentative_answer
        ):
            trajectory["final_answer"] = (
                accepted_step.confirmed_answer or accepted_step.tentative_answer
            )
            _terminate(item, "stop")
            return item

        if len(trajectory["steps"]) >= config["t_max"]:
            _terminate(item, "truncated")
            return item

        if accepted_step.action == "handoff" and accepted_step.handoff_target:
            item["current_agent"] = accepted_step.handoff_target
            item["messages"][0] = {
                "role": "system",
                "content": _system_prompt(
                    item,
                    accepted_step.handoff_target,
                    config,
                ),
            }
        return item
    except Exception as exc:
        _terminate(item, "exception", error=str(exc))
        return item


def pending_count(state: Dict[str, Any], agent_id: Optional[str] = None) -> int:
    return sum(
        1
        for item in state["items"]
        if item["status"] == "pending"
        and (agent_id is None or item["current_agent"] == agent_id)
    )


def total_steps(state: Dict[str, Any]) -> int:
    return sum(len(item["trajectory"]["steps"]) for item in state["items"])


def run_agent_phase(
    state: Dict[str, Any],
    agent_id: str,
    caller: Any,
    *,
    max_concurrency: int,
    on_complete: Optional[Callable[[int, Dict[str, Any]], None]] = None,
) -> list[Dict[str, Any]]:
    indices = [
        index
        for index, item in enumerate(state["items"])
        if item["status"] == "pending" and item["current_agent"] == agent_id
    ]
    config = state["config"]
    results: Dict[int, Dict[str, Any]] = {}

    def advance(index: int) -> tuple[int, Dict[str, Any]]:
        updated = _advance_one(
            copy.deepcopy(state["items"][index]),
            agent_id,
            caller,
            config,
        )
        return index, updated

    if max_concurrency <= 1:
        for done, index in enumerate(indices, start=1):
            result_index, item = advance(index)
            results[result_index] = item
            if on_complete is not None:
                on_complete(done, item)
    elif indices:
        with ThreadPoolExecutor(max_workers=min(max_concurrency, len(indices))) as pool:
            futures = {pool.submit(advance, index): index for index in indices}
            done = 0
            for future in as_completed(futures):
                result_index, item = future.result()
                results[result_index] = item
                done += 1
                if on_complete is not None:
                    on_complete(done, item)

    for index, item in results.items():
        state["items"][index] = item
    state["phase_count"] = int(state.get("phase_count", 0)) + 1
    state["updated_at"] = _utc_now()
    state.setdefault("history", []).append({
        "phase": state["phase_count"],
        "agent": agent_id,
        "advanced": len(indices),
        "pending_after": pending_count(state),
        "total_steps_after": total_steps(state),
        "completed_at": state["updated_at"],
    })
    return [results[index] for index in sorted(results)]


def _validate_state(state: Dict[str, Any]) -> None:
    if state.get("version") != STATE_VERSION:
        raise ValueError(
            f"unsupported state version {state.get('version')}; expected {STATE_VERSION}"
        )
    if not isinstance(state.get("config"), dict) or not isinstance(
        state.get("items"), list
    ):
        raise ValueError("invalid role-batched state structure")
    json_transport = state["config"].get("json_transport", "none")
    if json_transport not in JSON_TRANSPORTS:
        raise ValueError(f"invalid JSON transport in state: {json_transport}")
    for item in state["items"]:
        if item.get("status") not in {"pending", "done"}:
            raise ValueError(f"invalid item status: {item.get('status')}")
        if item["status"] == "pending" and item.get("current_agent") not in base.AGENT_IDS:
            raise ValueError(
                f"pending trajectory has invalid agent: {item.get('current_agent')}"
            )
        if _item_start_agent(item, state["config"]) not in base.AGENT_IDS:
            raise ValueError("trajectory has an invalid start agent")


def load_state(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        state = json.load(handle)
    state["config"].setdefault("start_agent_seed", 42)
    state["config"].setdefault("enable_thinking", False)
    for item in state.get("items", []):
        item.setdefault("start_agent", _item_start_agent(item, state["config"]))
    _validate_state(state)
    return state


def save_state(path: Path, state: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_records(state: Dict[str, Any]) -> list[Dict[str, Any]]:
    if pending_count(state):
        raise ValueError("cannot finalize while trajectories are still pending")
    records: list[Dict[str, Any]] = []
    for item in state["items"]:
        problem = _problem_from_dict(item["problem"])
        trajectory = _trajectory_from_item(item)
        em, f1 = compute_em_f1(trajectory.final_answer or "", problem)
        record = base.trajectory_to_dict(trajectory, problem)
        record["start_agent"] = _item_start_agent(item, state["config"])
        record["json_transport"] = state["config"].get("json_transport", "none")
        record["em"] = em
        record["f1"] = f1
        records.append(record)
    return records


def write_records(path: Path, records: list[Dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"evaluation output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_init_args(args: argparse.Namespace) -> None:
    if args.start < 0:
        raise SystemExit("--start must be non-negative")
    if args.limit <= 0 or args.t_max <= 0:
        raise SystemExit("--limit and --t-max must be positive")
    if args.start_agent_seed < 0:
        raise SystemExit("--start-agent-seed must be non-negative")
    if not 1 <= args.min_agents_before_stop <= len(base.AGENT_IDS):
        raise SystemExit(
            f"--min-agents-before-stop must be in [1, {len(base.AGENT_IDS)}]"
        )
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
        largest = max(
            args.sft_standard_min_handoffs,
            *args.sft_extended_min_handoffs,
        )
        if args.t_max <= largest:
            raise SystemExit("--t-max must exceed every SFT minimum handoff count")


def _add_init_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("init", help="Create the trajectory state file.")
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--t-max", type=int, default=8)
    parser.add_argument(
        "--start-agent",
        choices=[*base.AGENT_IDS, "balanced", "random"],
        default="A1",
    )
    parser.add_argument("--start-agent-seed", type=int, default=42)
    parser.add_argument("--min-agents-before-stop", type=int, default=3)
    parser.add_argument(
        "--enforce-collaboration-policy",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--generation-seed", type=int, default=None)
    parser.add_argument(
        "--json-transport",
        choices=JSON_TRANSPORTS,
        default="json_schema",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Request Qwen3 thinking mode for every role phase.",
    )
    parser.add_argument("--api-timeout", type=int, default=120)
    parser.add_argument("--sft-warmup-prompt", action="store_true")
    parser.add_argument("--sft-extended-fraction", type=float, default=0.4)
    parser.add_argument("--sft-protocol-seed", type=int, default=42)
    parser.add_argument("--sft-controlled-generation", action="store_true")
    parser.add_argument("--sft-max-step-retries", type=int, default=2)
    parser.add_argument("--sft-max-verifier-similarity", type=float, default=0.85)
    parser.add_argument("--sft-standard-min-handoffs", type=int, default=2)
    parser.add_argument(
        "--sft-extended-min-handoffs",
        type=int,
        nargs="+",
        default=[3, 4],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Advance GSM MAS trajectories in global role batches."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_init_parser(subparsers)

    pending_parser = subparsers.add_parser("pending", help="Print a pending count.")
    pending_parser.add_argument("--state", type=Path, required=True)
    pending_parser.add_argument("--agent", choices=[*base.AGENT_IDS, "all"], default="all")

    steps_parser = subparsers.add_parser("steps", help="Print the total completed step count.")
    steps_parser.add_argument("--state", type=Path, required=True)

    phase_parser = subparsers.add_parser("run-agent", help="Advance one role queue.")
    phase_parser.add_argument("--state", type=Path, required=True)
    phase_parser.add_argument("--agent", choices=base.AGENT_IDS, required=True)
    phase_parser.add_argument("--api-base", required=True)
    phase_parser.add_argument("--api-model", required=True)
    phase_parser.add_argument("--api-key", default="EMPTY")
    phase_parser.add_argument("--json-transport", choices=JSON_TRANSPORTS)
    phase_parser.add_argument("--max-concurrency", type=int, default=1)
    phase_parser.add_argument("--log-raw-chars", type=int, default=0)

    status_parser = subparsers.add_parser("status", help="Print queue status as JSON.")
    status_parser.add_argument("--state", type=Path, required=True)

    finalize_parser = subparsers.add_parser("finalize", help="Write ordered JSONL results.")
    finalize_parser.add_argument("--state", type=Path, required=True)
    finalize_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _initialize(args: argparse.Namespace) -> None:
    _validate_init_args(args)
    if args.state.exists():
        raise SystemExit(f"state file already exists: {args.state}")
    problems = load_gsm_hard(args.data_path)
    selected = problems[args.start: args.start + args.limit]
    if not selected:
        raise SystemExit("No problems selected.")
    state = create_evaluation_state(
        selected,
        start_index=args.start,
        data_path=str(Path(args.data_path).resolve()),
        t_max=args.t_max,
        start_agent=args.start_agent,
        start_agent_seed=args.start_agent_seed,
        min_agents_before_stop=args.min_agents_before_stop,
        enforce_collaboration_policy=args.enforce_collaboration_policy,
        generation_seed=args.generation_seed,
        json_transport=args.json_transport,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        enable_thinking=args.enable_thinking,
        api_timeout=args.api_timeout,
        sft_warmup_prompt=args.sft_warmup_prompt,
        sft_extended_fraction=args.sft_extended_fraction,
        sft_protocol_seed=args.sft_protocol_seed,
        sft_controlled_generation=args.sft_controlled_generation,
        sft_max_step_retries=args.sft_max_step_retries,
        sft_max_verifier_similarity=args.sft_max_verifier_similarity,
        sft_standard_min_handoffs=args.sft_standard_min_handoffs,
        sft_extended_min_handoffs=args.sft_extended_min_handoffs,
    )
    save_state(args.state, state)
    print(
        f"Initialized {len(selected)} trajectories at {args.state}; "
        f"json_transport={args.json_transport} "
        f"start_agent_counts={_start_agent_counts(state)} "
        f"pending A1={pending_count(state, 'A1')} "
        f"A2={pending_count(state, 'A2')} A3={pending_count(state, 'A3')}"
    )


def _run_agent(args: argparse.Namespace) -> None:
    if args.max_concurrency <= 0:
        raise SystemExit("--max-concurrency must be positive")
    state = load_state(args.state)
    queued = pending_count(state, args.agent)
    if queued == 0:
        raise SystemExit(f"no trajectories are pending for {args.agent}")
    config = state["config"]
    json_transport = str(config.get("json_transport", "none"))
    requested_transport = getattr(args, "json_transport", None)
    if requested_transport is not None and requested_transport != json_transport:
        raise SystemExit(
            "--json-transport does not match the trajectory state: "
            f"requested={requested_transport} state={json_transport}"
        )
    response_format = _build_response_format(json_transport, args.agent)
    generation = GenerationOptions(
        max_new_tokens=config["max_new_tokens"],
        temperature=config["temperature"],
        top_p=config["top_p"],
        enable_thinking=bool(config.get("enable_thinking", False)),
    )
    caller = OpenAIChatLLMCaller(
        args.api_base,
        args.api_model,
        generation=generation,
        timeout=config["api_timeout"],
        api_key=args.api_key,
        response_format=response_format,
    )
    print(
        f"Role phase {args.agent}: queued={queued} "
        f"max_concurrency={args.max_concurrency} "
        f"generation_seed={config.get('generation_seed')} "
        f"enable_thinking={bool(config.get('enable_thinking', False))} "
        f"json_transport={json_transport}"
    )
    print(
        "  response_format="
        + json.dumps(response_format, ensure_ascii=False, sort_keys=True)
    )
    phase_start = time.monotonic()

    def report(done: int, item: Dict[str, Any]) -> None:
        elapsed = time.monotonic() - phase_start
        last_step = item["trajectory"]["steps"][-1] if item["trajectory"]["steps"] else None
        action = last_step["action"] if last_step else item["trajectory"]["terminated_by"]
        target = last_step.get("handoff_target") if last_step else None
        print(
            f"  [{done}/{queued}] id={item['problem']['id']} action={action} "
            f"next={target or '-'} status={item['status']} rate={done / max(elapsed, 1e-6):.2f}/s"
        )
        if args.log_raw_chars > 0 and last_step:
            print(f"    raw={last_step['raw_output'][:args.log_raw_chars]}")
        if item["trajectory"].get("error"):
            print(f"    error={item['trajectory']['error']}")
            current_turn = len(item["trajectory"]["steps"])
            for rejection in item.get("generation_rejections", []):
                if (
                    rejection.get("turn") != current_turn
                    or rejection.get("agent") != args.agent
                ):
                    continue
                raw_output = str(rejection.get("raw_output", ""))
                preview_chars = args.log_raw_chars if args.log_raw_chars > 0 else 240
                preview = raw_output[:preview_chars].replace("\n", "\\n")
                print(
                    f"    failed_attempt={rejection.get('attempt')} "
                    f"raw_chars={len(raw_output)} reason={rejection.get('reason')} "
                    f"raw_preview={preview!r}"
                )

    run_agent_phase(
        state,
        args.agent,
        caller,
        max_concurrency=args.max_concurrency,
        on_complete=report,
    )
    save_state(args.state, state)
    print(
        f"Phase complete: agent={args.agent} advanced={queued} "
        f"pending_total={pending_count(state)} total_steps={total_steps(state)}"
    )


def _status_payload(state: Dict[str, Any]) -> Dict[str, Any]:
    terminated: Dict[str, int] = {}
    for item in state["items"]:
        if item["status"] == "done":
            reason = str(item["trajectory"]["terminated_by"])
            terminated[reason] = terminated.get(reason, 0) + 1
    return {
        "items": len(state["items"]),
        "phase_count": state.get("phase_count", 0),
        "total_steps": total_steps(state),
        "pending_total": pending_count(state),
        "generation_seed": state["config"].get("generation_seed"),
        "enable_thinking": bool(state["config"].get("enable_thinking", False)),
        "start_agent": state["config"].get("start_agent", "A1"),
        "start_agent_seed": state["config"].get("start_agent_seed", 42),
        "start_agent_counts": _start_agent_counts(state),
        "json_transport": state["config"].get("json_transport", "none"),
        "pending_by_agent": {
            agent: pending_count(state, agent) for agent in base.AGENT_IDS
        },
        "terminated": terminated,
    }


def main() -> None:
    args = parse_args()
    if args.command == "init":
        _initialize(args)
        return

    state = load_state(args.state)
    if args.command == "pending":
        print(pending_count(state, None if args.agent == "all" else args.agent))
    elif args.command == "steps":
        print(total_steps(state))
    elif args.command == "status":
        print(json.dumps(_status_payload(state), sort_keys=True))
    elif args.command == "run-agent":
        _run_agent(args)
    elif args.command == "finalize":
        records = build_records(state)
        write_records(args.output, records)
        print("GSM-HARD role-batched evaluation complete")
        print(f"  n:      {len(records)}")
        print(f"  JSON:   {state['config'].get('json_transport', 'none')}")
        print(f"  seed:   {state['config'].get('generation_seed')}")
        print(f"  starts: {_start_agent_counts(state)}")
        print(f"  EM:     {mean(record['em'] for record in records):.4f}")
        print(f"  F1:     {mean(record['f1'] for record in records):.4f}")
        print(f"  output: {args.output}")
    else:
        raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    main()

"""Small protocol adapters shared by coordination-fingerprint scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ProtocolUnit:
    agent: str
    payload: dict[str, Any]
    kind: str


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return list(value.values())
    return list(value or []) if isinstance(value, (list, tuple)) else []


def _agent(payload: dict[str, Any], fallback: str = "unknown") -> str:
    for key in ("active_agent", "agent_id", "agent", "caller_id", "api_model", "model"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    return fallback


def protocol_units(method: str, record: dict[str, Any]) -> list[ProtocolUnit]:
    """Extract ordered model/protocol units from all supported result schemas."""
    normalized = method.lower().replace("_", "-")
    if normalized in {"jca", "zero-shot", "zeroshot"}:
        steps = (record.get("trajectory") or {}).get("steps") or []
        return [ProtocolUnit(_agent(step), step, "step") for step in steps if isinstance(step, dict)]

    if normalized == "mad":
        values = record.get("turns")
        if values is None:
            values = (record.get("mad") or {}).get("turns")
        return [ProtocolUnit(_agent(turn), turn, "turn") for turn in _as_list(values) if isinstance(turn, dict)]

    if normalized == "agentverse":
        units: list[ProtocolUnit] = []
        nested = record.get("agentverse") or {}
        source = nested if nested else record
        recruit = source.get("recruit_raw_outputs")
        recruit_payload = {
            "raw_outputs": recruit,
            "raw_output": source.get("recruit_raw_output"),
            "agent": source.get("meta_agent") or "A3",
        }
        recruit_record = source.get("recruit")
        if isinstance(recruit_record, dict):
            recruit_payload = recruit_record
            recruit_payload = dict(recruit_payload)
            recruit_payload.setdefault("agent", source.get("meta_agent") or "A3")
        if recruit_payload.get("raw_outputs") or recruit_payload.get("raw_output") or recruit_payload.get("attempts"):
            units.append(ProtocolUnit(_agent(recruit_payload, "A3"), recruit_payload, "recruit"))
        for iteration in _as_list(source.get("iterations")):
            if not isinstance(iteration, dict):
                continue
            for answer in _as_list(iteration.get("answers")):
                if isinstance(answer, dict):
                    units.append(ProtocolUnit(_agent(answer), answer, "answer"))
            evaluation = iteration.get("evaluation")
            if isinstance(evaluation, dict) and evaluation:
                units.append(ProtocolUnit(_agent(evaluation, str(source.get("meta_agent") or "A3")), evaluation, "evaluation"))
        return units

    if normalized == "gptswarm":
        nested = record.get("swarm") or {}
        source = nested if nested else record
        outputs = source.get("node_outputs") or []
        node_models = source.get("node_models") or record.get("node_models") or {}
        units = []
        for node in _as_list(outputs):
            if not isinstance(node, dict):
                continue
            payload = node
            node_name = str(node.get("node_name") or "")
            fallback = str(node_models.get(node_name) or "")
            units.append(ProtocolUnit(_agent(payload, fallback or "unknown"), payload, "node"))
        return units

    if normalized == "aflow":
        operations = record.get("op_records") or record.get("op_calls") or []
        return [ProtocolUnit(_agent(operation), operation, "op") for operation in _as_list(operations) if isinstance(operation, dict)]

    if normalized in {"self-rl", "selfrl", "sas"}:
        return [ProtocolUnit("SAS/14B", record, "one-shot")]

    raise ValueError(f"unsupported protocol method: {method}")


def raw_attempts(payload: dict[str, Any]) -> list[Any] | None:
    """Return the persisted attempts/raw outputs, or None when no ledger exists."""
    values = payload.get("raw_outputs")
    if isinstance(values, list):
        return values
    if payload.get("raw_output") not in (None, ""):
        return [payload["raw_output"]]
    if payload.get("raw_response") not in (None, ""):
        return [payload["raw_response"]]
    if payload.get("response") not in (None, ""):
        return [payload["response"]]
    return None


def unit_chain(method: str, record: dict[str, Any]) -> list[str]:
    return [_agent(unit.payload, unit.agent) for unit in protocol_units(method, record)]

"""Strict JSON helpers for the multi-agent response protocol."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple


PROTOCOL_FIELDS = (
    "reasoning",
    "tentative_answer",
    "action",
    "handoff_target",
    "handoff_note",
    "confirmed_answer",
)
PROTOCOL_ACTIONS = frozenset({"handoff", "confirm_stop"})
PROTOCOL_AGENT_IDS = ("A1", "A2", "A3")
REQUIRED_PROTOCOL_FIELDS = PROTOCOL_FIELDS
_NULLABLE_PROTOCOL_FIELDS = (
    "handoff_target",
    "handoff_note",
    "confirmed_answer",
)


def _reject_nonstandard_constant(value: str) -> Any:
    raise ValueError(f"non-standard JSON constant: {value}")


def _reject_duplicate_keys(pairs: Iterable[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def protocol_json_decoder() -> json.JSONDecoder:
    return json.JSONDecoder(
        parse_constant=_reject_nonstandard_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )


def parse_protocol_object(value: Any) -> Dict[str, Any]:
    if not isinstance(value, str):
        if isinstance(value, dict):
            json.dumps(value, allow_nan=False)
            return dict(value)
        raise ValueError("protocol response must be a JSON object string")
    text = value.strip()
    if not text.startswith("{"):
        raise ValueError("protocol response must start with a JSON object")
    parsed, consumed = protocol_json_decoder().raw_decode(text)
    if not isinstance(parsed, dict) or text[consumed:].strip():
        raise ValueError("protocol response must be exactly one JSON object")
    return parsed


def protocol_schema_error(
    value: Any,
    active_agent: Optional[str] = None,
    agent_ids: Sequence[str] = PROTOCOL_AGENT_IDS,
    *,
    allow_self_handoff: bool = False,
) -> Optional[str]:
    """Return a deterministic protocol-schema error, or ``None`` if valid.

    The transport parser deliberately only checks that a value is one complete
    JSON object.  This second layer enforces the collaboration protocol's
    exact six-field contract without coercing values or repairing contradictory
    fields.  ``active_agent`` is optional so callers that parse persisted data
    can still validate the object shape; generation callers should provide it
    to reject self-handoffs at the schema boundary.
    """

    try:
        payload = parse_protocol_object(value)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        return f"malformed JSON protocol: {exc}"

    expected = set(PROTOCOL_FIELDS)
    keys = set(payload)
    missing = sorted(expected - keys)
    if missing:
        return f"protocol schema missing required fields: {missing}"
    unknown = sorted(keys - expected, key=str)
    if unknown:
        return f"protocol schema has unknown fields: {unknown}"

    for field in ("reasoning", "tentative_answer", "action"):
        value_for_field = payload[field]
        if not isinstance(value_for_field, str):
            return f"protocol schema field {field} must be a string"
        if not value_for_field.strip():
            return f"protocol schema field {field} must be non-empty"

    action = payload["action"]
    if action not in PROTOCOL_ACTIONS:
        return f"protocol schema action must be one of {sorted(PROTOCOL_ACTIONS)}"

    for field in _NULLABLE_PROTOCOL_FIELDS:
        value_for_field = payload[field]
        if value_for_field is not None and not isinstance(value_for_field, str):
            return f"protocol schema field {field} must be a string or null"

    target = payload["handoff_target"]
    confirmed = payload["confirmed_answer"]
    allowed_agents = tuple(agent_ids)
    if action == "handoff":
        if target not in allowed_agents:
            return (
                "protocol schema handoff_target must be one of "
                f"{list(allowed_agents)} for handoff"
            )
        if active_agent is not None and target == active_agent and not allow_self_handoff:
            return "protocol schema handoff_target cannot be the active agent"
        if confirmed is not None:
            return "protocol schema handoff must have confirmed_answer=null"
    else:
        if target is not None:
            return "protocol schema confirm_stop must have handoff_target=null"
        if not isinstance(confirmed, str) or not confirmed.strip():
            return "protocol schema confirm_stop requires a non-empty confirmed_answer"

    return None


def validate_protocol_object(
    value: Any,
    active_agent: Optional[str] = None,
    agent_ids: Sequence[str] = PROTOCOL_AGENT_IDS,
    *,
    allow_self_handoff: bool = False,
) -> Dict[str, Any]:
    """Parse and validate one exact collaboration protocol object.

    A fresh dictionary is returned only after every field passes validation;
    callers can therefore safely consume values without ``str()`` coercion.
    ``ValueError`` is raised with a stable, actionable reason on failure.
    """

    payload = parse_protocol_object(value)
    error = protocol_schema_error(
        payload,
        active_agent=active_agent,
        agent_ids=agent_ids,
        allow_self_handoff=allow_self_handoff,
    )
    if error is not None:
        raise ValueError(error)
    return payload


def protocol_json_schema_well_formed(
    value: Any,
    active_agent: Optional[str] = None,
    agent_ids: Sequence[str] = PROTOCOL_AGENT_IDS,
    *,
    allow_self_handoff: bool = False,
) -> bool:
    """Return whether a value satisfies the complete protocol schema."""

    try:
        validate_protocol_object(
            value,
            active_agent=active_agent,
            agent_ids=agent_ids,
            allow_self_handoff=allow_self_handoff,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


validate_protocol_schema = validate_protocol_object
protocol_schema_valid = protocol_json_schema_well_formed
protocol_json_schema_error = protocol_schema_error
is_valid_protocol_object = protocol_json_schema_well_formed


def json_object_well_formed(value: Any) -> bool:
    """Return whether a value is exactly one strict JSON object."""

    try:
        parse_protocol_object(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def protocol_json_well_formed(
    value: Any,
    active_agent: Optional[str] = None,
    agent_ids: Sequence[str] = PROTOCOL_AGENT_IDS,
) -> bool:
    """Return whether a value is exactly one strict JSON object."""

    return json_object_well_formed(value)

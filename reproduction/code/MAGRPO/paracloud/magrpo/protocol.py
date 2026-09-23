"""Protocol parsing shared by every task adapter.

All five datasets speak the same six-field JSON protocol
(``vendor/protocol_json.py``), except MultiPL-E which renames two fields.  The
renaming is normalized here so the *strict* validator still applies everywhere --
using the lenient ``relaxed_json`` parser instead would hide exactly the format
collapse this baseline needs to surface.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from vendor.protocol_json import (  # noqa: F401  (re-exported)
    PROTOCOL_ACTIONS,
    PROTOCOL_AGENT_IDS,
    PROTOCOL_FIELDS,
    parse_protocol_object,
    protocol_schema_error,
)

AGENT_IDS = PROTOCOL_AGENT_IDS

# MultiPL-E's runner renames the answer fields; map them back before validating.
FIELD_ALIASES = {
    "tentative_completion": "tentative_answer",
    "confirmed_completion": "confirmed_answer",
}


def find_json_object(text: str) -> dict[str, Any] | None:
    """Locate the first balanced ``{...}`` block and decode it strictly.

    This only *locates* the object; decoding stays strict (duplicate keys, NaN
    and trailing content are all rejected by ``parse_protocol_object``).
    """
    raw = str(text or "")
    start = raw.find("{")
    while start >= 0:
        decoder = json.JSONDecoder()
        try:
            obj, _ = decoder.raw_decode(raw[start:])
        except ValueError:
            if os.environ.get("MAGRPO_RELAX_PROTOCOL") == "1":
                repaired = _repair_json_string(raw[start:])
                try:
                    obj, _ = decoder.raw_decode(repaired)
                    return obj if isinstance(obj, dict) else None
                except ValueError:
                    pass
            start = raw.find("{", start + 1)
            continue
        return obj if isinstance(obj, dict) else None
    return None


def _repair_json_string(text: str) -> str:
    """Repair common model JSON defects without changing mathematical content."""
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)
    out: list[str] = []
    in_string = False
    index = 0
    while index < len(text):
        char = text[index]
        if char == '"':
            escaped = index > 0 and text[index - 1] == "\\" and not (
                index > 1 and text[index - 2] == "\\"
            )
            if not escaped:
                in_string = not in_string
            out.append(char)
        elif in_string and char == "\\":
            next_char = text[index + 1] if index + 1 < len(text) else ""
            valid = next_char in '"\\/bfnrt' or (
                next_char == "u"
                and index + 5 < len(text)
                and re.fullmatch(r"[0-9a-fA-F]{4}", text[index + 2 : index + 6])
            )
            out.append("\\" if valid else "\\\\")
        elif in_string and ord(char) < 0x20:
            out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}.get(char, " "))
        else:
            out.append(char)
        index += 1
    return "".join(out)


def normalize_aliases(obj: dict[str, Any]) -> dict[str, Any]:
    """Rename MultiPL-E's completion fields onto the canonical protocol names."""
    return {FIELD_ALIASES.get(key, key): value for key, value in obj.items()}


def parse_protocol(
    raw: str,
    *,
    active_agent: str,
    allow_aliases: bool = False,
) -> tuple[dict[str, Any] | None, str | None]:
    """Return ``(protocol_object, error)``. Exactly one is non-None."""
    obj = find_json_object(raw)
    if obj is None:
        return None, "no JSON object found in response"
    if allow_aliases:
        obj = normalize_aliases(obj)
    if os.environ.get("MAGRPO_RELAX_PROTOCOL") == "1":
        obj = _relax_protocol_fields(obj, active_agent)
    error = protocol_schema_error(obj, active_agent)
    if error is not None:
        return None, error
    return obj, None


def _relax_protocol_fields(obj: dict[str, Any], active_agent: str) -> dict[str, Any]:
    """Normalize recoverable schema mistakes for targeted continuation runs."""
    repaired = dict(obj)
    repaired.setdefault("reasoning", "recovered protocol response")
    repaired.setdefault("tentative_answer", "")
    repaired.setdefault("handoff_target", None)
    repaired.setdefault("handoff_note", None)
    repaired.setdefault("confirmed_answer", None)
    for field in ("reasoning", "tentative_answer", "handoff_note", "confirmed_answer"):
        value = repaired.get(field)
        if value is not None and not isinstance(value, str):
            repaired[field] = str(value)
    action = repaired.get("action")
    if action not in PROTOCOL_ACTIONS:
        action = "confirm_stop" if repaired.get("confirmed_answer") else "handoff"
        repaired["action"] = action
    if action == "handoff":
        repaired["confirmed_answer"] = None
        target = repaired.get("handoff_target")
        if target not in AGENT_IDS or target == active_agent:
            repaired["handoff_target"] = next(a for a in AGENT_IDS if a != active_agent)
    else:
        repaired["handoff_target"] = None
        if not repaired.get("confirmed_answer"):
            repaired["confirmed_answer"] = repaired.get("tentative_answer", "")
        repaired["handoff_note"] = None
    return repaired


def answer_of(obj: dict[str, Any] | None) -> str:
    """The agent's current answer: confirmed if it stopped, else tentative."""
    if not obj:
        return ""
    confirmed = obj.get("confirmed_answer")
    if isinstance(confirmed, str) and confirmed.strip():
        return confirmed.strip()
    return str(obj.get("tentative_answer") or "").strip()


def render_assistant_message(obj: dict[str, Any]) -> str:
    """Canonical serialization of a protocol object for conversation history."""
    return json.dumps(
        {field: obj.get(field) for field in PROTOCOL_FIELDS},
        ensure_ascii=False,
    )


def repair_instruction(error: str) -> str:
    """User-turn nudge sent on a retry after a malformed response."""
    fields = ", ".join(PROTOCOL_FIELDS)
    return (
        f"Your previous response was rejected: {error}\n"
        f"Reply with ONLY one JSON object containing exactly these keys: {fields}. "
        'The "action" field must be either "handoff" or "confirm_stop". '
        "No prose, no code fences."
    )


_LABEL_RE = re.compile(r"^\[(A1|A2|A3)\]\s*", re.MULTILINE)


def format_joint_observation(
    responses: dict[str, dict[str, Any] | None],
    *,
    turn: int,
) -> str:
    """Render the previous round's joint response for the synchronous mode.

    Every agent sees all three drafts, labelled by author -- this is the
    "environment transitions on the joint response" part of the Dec-POMDP.
    """
    lines = [f"# Round {turn} drafts from all agents"]
    for agent in AGENT_IDS:
        obj = responses.get(agent)
        if obj is None:
            lines.append(f"\n[{agent}] (no valid response this round)")
            continue
        reasoning = str(obj.get("reasoning") or "").strip()
        answer = answer_of(obj)
        lines.append(f"\n[{agent}] reasoning: {reasoning}")
        lines.append(f"[{agent}] answer: {answer}")
    return "\n".join(lines)

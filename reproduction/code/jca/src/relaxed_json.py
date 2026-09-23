"""Small JSON compatibility helpers for model-generated protocol objects.

Some chat models emit a JSON-looking object containing LaTeX commands such as
``\frac`` without escaping the backslash.  That text is not strict JSON, but
it is common enough that rejecting the whole turn is needlessly lossy.  The
repair below only changes backslashes while inside JSON strings and leaves
normal JSON escapes untouched.
"""

from __future__ import annotations

import json
import re
from typing import Any


_HEX4_RE = re.compile(r"^[0-9a-fA-F]{4}$")
_CONTROL_ESCAPES = frozenset("bfnrt")
_SIMPLE_ESCAPES = frozenset('"\\/')
_LATEX_COMMANDS = frozenset(
    {
        "alpha", "beta", "begin", "mathbf", "boxed", "cdot", "cases", "circ",
        "cos", "end", "exists", "frac", "fbox", "ge", "geq", "gamma", "in",
        "infty", "int", "lambda", "ldots", "left", "le", "leq", "lim", "ln",
        "log", "mathbb", "mathbf", "mathrm", "neq", "nabla", "pi", "pm", "prod",
        "quad", "qquad", "rceil", "right", "rm", "sin", "sqrt", "sum", "tan",
        "text", "theta", "times", "vert", "wedge", "xi", "zeta",
    }
)
_CONTROL_CHARS = {
    "\b": r"\b",
    "\f": r"\f",
    "\n": r"\n",
    "\r": r"\r",
    "\t": r"\t",
}


def _looks_like_latex_command(text: str, next_index: int) -> bool:
    """Return whether a control-looking escape starts a LaTeX command.

    ``\\frac`` starts with ``\\f`` and ``\\text`` starts with ``\\t``;
    both prefixes are technically valid JSON escapes, so merely repairing
    *invalid* escapes would silently turn them into form-feed/tab characters.
    A command is identified by a run of letters after the backslash.  A
    single-letter command is intentionally left to strict JSON's control
    escape rules because ``\\n``/``\\t`` are common serialized whitespace.
    """

    if next_index >= len(text) or not text[next_index].isalpha():
        return False
    end = next_index + 1
    while end < len(text) and text[end].isalpha():
        end += 1
    command = text[next_index:end]
    if command in _LATEX_COMMANDS:
        return True
    if end < len(text) and text[end] in "{}[]()_^,.;:+-=*/":
        return len(command) >= 2
    return False


def repair_json_string_escapes(text: str) -> str:
    """Repair unescaped LaTeX backslashes in a JSON-looking string.

    The returned text is suitable for :class:`json.JSONDecoder`.  Existing
    JSON escapes (including ``\\uXXXX``) remain unchanged.  Invalid escapes
    and LaTeX command backslashes are doubled so that decoding yields the
    original literal backslash.
    """

    output: list[str] = []
    in_string = False
    index = 0
    while index < len(text):
        character = text[index]
        if not in_string:
            output.append(character)
            if character == '"':
                in_string = True
            index += 1
            continue

        if character == '"':
            output.append(character)
            in_string = False
            index += 1
            continue

        if character in _CONTROL_CHARS:
            output.append(_CONTROL_CHARS[character])
            index += 1
            continue

        if character != "\\":
            output.append(character)
            index += 1
            continue

        next_index = index + 1
        if next_index >= len(text):
            output.append("\\\\")
            index = next_index
            continue

        next_character = text[next_index]
        if next_character in _SIMPLE_ESCAPES:
            output.append("\\" + next_character)
            index += 2
            continue
        if next_character == "u" and _HEX4_RE.match(text[next_index + 1 : next_index + 5] or ""):
            output.append(text[index : next_index + 5])
            index = next_index + 5
            continue
        if next_character in _CONTROL_ESCAPES and not _looks_like_latex_command(text, next_index):
            output.append("\\" + next_character)
            index += 2
            continue

        # Invalid JSON escape or a LaTeX command.  Double the slash while
        # retaining the following character exactly as emitted by the model.
        output.append("\\\\")
        index += 1

    return "".join(output)


def _candidate_starts(text: str) -> list[int]:
    return [index for index, character in enumerate(text) if character == "{"]


def find_json_object(text: str) -> dict[str, Any] | None:
    """Find and decode the first JSON object in model output.

    Strict JSON is attempted first.  If that fails, a repaired copy is
    decoded.  Searching later ``{`` positions tolerates a short preamble or a
    fenced response before the object.
    """

    source = str(text or "").strip()
    if not source:
        return None
    decoder = json.JSONDecoder()
    starts = _candidate_starts(source)
    for start in starts:
        candidate = source[start:]
        # Decode the repaired form first.  Strict JSON accepts ``\\f`` and
        # ``\\t`` as control escapes, which would otherwise silently corrupt
        # the common LaTeX commands ``\\frac`` and ``\\text``.
        for candidate_text in (repair_json_string_escapes(candidate), candidate):
            try:
                value, _ = decoder.raw_decode(candidate_text)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    return None

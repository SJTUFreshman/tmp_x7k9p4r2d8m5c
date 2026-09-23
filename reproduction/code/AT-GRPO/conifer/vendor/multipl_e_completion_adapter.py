"""Prompt and response adaptation for chat-model code completion.

MultiPL-E stores a source prefix and expects the model response to contain
only the text that should be appended to that prefix. Chat models often return
a complete answer with prose or Markdown, so the SAS runner uses this small
adapter to preserve the MultiPL-E completion contract.
"""

from __future__ import annotations

import ast
import re


PROMPT_PROTOCOL = "qwen_chat_template_non_thinking_code_continuation_v4"

CODE_COMPLETION_SYSTEM_PROMPT = (
    "You are a strict code completion engine. The caller will append your "
    "entire response directly to a source-code prefix and compile the result. "
    "Return only the missing source-code continuation. Do not return an "
    "explanation, Markdown, code fences, a rewritten complete file, imports, "
    "or test code. Do not repeat any part of the prefix. Generate a complete "
    "function implementation, including final return statements and all "
    "necessary closing delimiters. Start exactly at the cursor and do not "
    "emit a <think> block."
)

_FENCE_RE = re.compile(r"```[^\r\n`]*\r?\n?(.*?)```", re.DOTALL)


def build_chat_user_prompt(
    language: str,
    prefix: str,
    tests: str,
    stop_tokens: list[str] | None = None,
) -> str:
    """Place the source prefix at the end of a focused chat user message."""

    stop_tokens = stop_tokens or []
    if evaluator_supplies_function_closure(tests, stop_tokens):
        boundary_instruction = (
            "The evaluator suffix begins with the function's final closing "
            "delimiter. Do not generate that final delimiter; end immediately "
            "before it. You must still close every nested block that you open."
        )
    else:
        boundary_instruction = (
            "The evaluator does not supply the function's final closing "
            "delimiter. You must generate it so that the function is complete "
            "before the evaluator appends its tests."
        )
    if language == "adb":
        boundary_instruction += (
            " For Ada, the prefix ends immediately after the function "
            "signature: generate the `is` section, declarations if needed, "
            "then `begin` and the executable statements. The evaluator "
            "supplies only the final `end Function_Name;`."
        )

    return (
        f"Target language: {language}\n\n"
        "The text below is the complete source-code prefix. Its final "
        "character is the cursor. Generate only the source text that must be "
        "inserted immediately after that cursor. Do not generate tests or any "
        "text after the function implementation.\n"
        f"{boundary_instruction}\n\n"
        "SOURCE PREFIX:\n"
        f"{prefix}"
    )


def _is_function_closing_stop(stop_token: str) -> bool:
    stripped = stop_token.strip().lower()
    return stripped == "}" or stripped == "end" or stripped.startswith("end ")


def evaluator_supplies_function_closure(
    tests: str, stop_tokens: list[str]
) -> bool:
    """Whether MultiPL-E's appended tests begin with the function closure."""

    # The evaluator inserts one newline before tests. Some datasets already
    # include a leading newline while others rely on the evaluator's newline.
    suffix_variants = (tests, "\n" + tests)
    return any(
        _is_function_closing_stop(stop_token)
        and any(suffix.startswith(stop_token) for suffix in suffix_variants)
        for stop_token in stop_tokens
        if stop_token
    )


def _last_nonempty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return ""


def _find_whitespace_flexible(text: str, needle: str):
    """Find code text while allowing whitespace changes between tokens."""

    needle = needle.strip()
    if not needle:
        return None
    # Splitting only on existing whitespace treats ``f(x){`` and ``f(x) {``
    # as different anchors. Tokenize punctuation separately so a reformatted
    # repeated function signature is still recognized and removed.
    parts = re.findall(r"\w+|[^\w\s]", needle)
    pattern = r"\s*".join(re.escape(part) for part in parts)
    return re.search(pattern, text)


def _remove_reasoning_and_special_tokens(text: str) -> str:
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    elif "<think>" in text:
        text = text.split("<think>", 1)[1]

    for marker in ("<|im_end|>", "<|endoftext|>", "<|end|>"):
        if marker in text:
            text = text.split(marker, 1)[0]
    return text


def _extract_code_block(text: str, prefix: str) -> str:
    blocks = [match.group(1) for match in _FENCE_RE.finditer(text)]
    if not blocks:
        # A response can hit max_tokens before it emits the closing fence.
        # Treat the first fence as the beginning of the code block in that
        # case, and remove its optional language label.
        marker = text.find("```")
        if marker >= 0:
            line_end = text.find("\n", marker + 3)
            if line_end >= 0:
                return text[line_end + 1 :]
        return text

    anchor = _last_nonempty_line(prefix)
    if anchor:
        for block in blocks:
            if _find_whitespace_flexible(block, anchor):
                return block
    return max(blocks, key=len)


def _remove_repeated_prefix(text: str, prefix: str) -> str:
    if not prefix:
        return text

    exact_index = text.find(prefix)
    if exact_index >= 0:
        return text[exact_index + len(prefix) :]

    stripped_prefix = prefix.strip()
    if stripped_prefix:
        stripped_index = text.find(stripped_prefix)
        if stripped_index >= 0:
            return text[stripped_index + len(stripped_prefix) :]

    # A full answer may have reformatted the prefix. The final prefix line is
    # usually the function signature or the last partial statement, which is
    # a stable enough cursor anchor across ordinary whitespace changes.
    # However, Python docstrings end with '"""' and _last_nonempty_line may
    # return a doctest example line like ">>> func(...)" that should NOT be
    # used as an anchor. Skip anchor matching for Python docstrings.
    if prefix.rstrip().endswith('"""'):
        return text
    anchor = _last_nonempty_line(prefix)
    if len(anchor) >= 8:
        match = _find_whitespace_flexible(text, anchor)
        if match:
            return text[match.end() :]
    return text


def _remove_repeated_opening_brace(text: str, prefix: str) -> str:
    """Remove a brace repeated immediately after a prefix ending in a brace."""

    if not prefix.rstrip().endswith("{"):
        return text
    match = re.match(r"(\s*)\{", text)
    if match is None:
        return text
    return match.group(1) + text[match.end() :]


def _ensure_cursor_separator(text: str, prefix: str, language: str) -> str:
    """Prevent adjacent identifier tokens at a whitespace-free cursor."""

    if (
        language == "adb"
        and prefix
        and text
        and not prefix[-1].isspace()
        and not text[0].isspace()
    ):
        return " " + text
    return text


def _scan_braces(text: str, prefix: str) -> tuple[int | None, int, int]:
    """Return function closure index, prefix depth, and final brace depth."""

    source = prefix + text
    prefix_length = len(prefix)
    depth = 0
    state = "normal"
    escaped = False
    line_comment = False
    block_comment = False
    prefix_depth = None
    closure_index = None

    for index, char in enumerate(source):
        next_char = source[index + 1] if index + 1 < len(source) else ""

        if line_comment:
            if char == "\n":
                line_comment = False
            continue
        if block_comment:
            if char == "*" and next_char == "/":
                block_comment = False
            continue
        if state != "normal":
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == state:
                state = "normal"
            continue

        if char == "/" and next_char == "/":
            line_comment = True
            continue
        if char == "/" and next_char == "*":
            block_comment = True
            continue
        if char == "#":
            line_comment = True
            continue
        if char in ("'", '"', "`"):
            state = char
            escaped = False
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            if index >= prefix_length and prefix_depth is not None:
                if depth == prefix_depth and closure_index is None:
                    closure_index = index - prefix_length
            depth = max(0, depth - 1)

        if index + 1 == prefix_length:
            prefix_depth = depth

    return closure_index, prefix_depth or 0, depth


def _find_function_closure(text: str, prefix: str) -> int | None:
    """Find the cursor function's closing brace, ignoring nested braces."""

    closure_index, _, _ = _scan_braces(text, prefix)
    return closure_index


def _is_brace_closing_stop(stop_token: str) -> bool:
    return stop_token.strip() == "}"


def _is_python_indentation_sensitive_stop(stop_token: str) -> bool:
    """Return True for Python stop tokens that must only match at top level."""
    # Python uses "\ndef", "\nif", "\nclass", "\n#" as stop tokens, which can
    # legitimately appear inside function bodies. Only match them at indent 0.
    return stop_token in ("\ndef", "\nif", "\nclass", "\n#")


def _is_at_zero_indent(text: str, position: int) -> bool:
    """Check if position is at the start of a line with zero indentation."""
    if position <= 0:
        return True
    # Walk back to find the start of the line containing position
    line_start = position
    while line_start > 0 and text[line_start - 1] not in "\n\r":
        line_start -= 1
    # Check if there's any non-whitespace before position on this line
    for i in range(line_start, position):
        if text[i] not in " \t":
            return False
    return True


def _python_defines_prompt_function_as_module(text: str, prefix: str) -> bool:
    """Whether text is a safe standalone module redefining the prompt function."""

    try:
        prefix_module = ast.parse(prefix)
        response_module = ast.parse(text)
    except SyntaxError:
        return False

    prompt_functions = [
        node.name
        for node in prefix_module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not prompt_functions:
        return False
    target_name = prompt_functions[-1]

    allowed_top_level = (
        ast.Import,
        ast.ImportFrom,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
    )
    if not all(isinstance(node, allowed_top_level) for node in response_module.body):
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == target_name
        for node in response_module.body
    )


def _truncate_at_stop_tokens(
    text: str,
    prefix: str,
    tests: str,
    stop_tokens: list[str],
    language: str,
) -> str:
    # A standalone Python module defining the requested function remains valid
    # when appended after MultiPL-E's docstring-only function prefix. Preserve
    # it instead of truncating at the blank line before its target ``def``.
    if language == "py" and _python_defines_prompt_function_as_module(text, prefix):
        return text

    # Shell parameter expansions such as ${#value} use braces that the generic
    # C-family scanner cannot distinguish from block delimiters. The evaluator
    # supplies the final function brace, so remove only a final brace-only line.
    if language == "sh" and evaluator_supplies_function_closure(tests, stop_tokens):
        without_final_closure = re.sub(r"\n[ \t]*}[ \t]*\Z", "", text)
        if without_final_closure != text:
            return without_final_closure

    # C/C++/Java-family datasets use a broad "\\n}" stop token. It is meant
    # for the function suffix, but the same text can close an inner block.
    # Only remove it when the completion actually closes the cursor function.
    if any(_is_brace_closing_stop(stop_token) for stop_token in stop_tokens):
        closure_index = _find_function_closure(text, prefix)
        if closure_index is not None:
            return text[:closure_index]

    if language == "adb" and evaluator_supplies_function_closure(
        tests, stop_tokens
    ):
        closure_line = tests.lstrip().splitlines()[0].strip()
        if closure_line:
            closure = re.search(
                rf"(?im)^[ \t]*{re.escape(closure_line)}[ \t]*(?:\n|$)",
                text,
            )
            if closure is not None:
                return text[: closure.start()]

    stop_index = len(text)
    for stop_token in stop_tokens:
        # Whitespace-only stops (notably "\n\n") are generation hints, not
        # reliable evaluator boundaries: valid code commonly contains blank
        # lines. Truncating at one can remove the rest of a function.
        if not stop_token or not stop_token.strip():
            continue
        # Function-closing stops are broad by design. For example, Ada's
        # "end " also prefixes nested "end if" and "end loop" statements.
        if _is_function_closing_stop(stop_token):
            continue
        # Python indentation-sensitive stops must only match at zero indent.
        # Otherwise "\nif" would truncate at every if statement in the body.
        python_indent_sensitive = (
            language == "py"
            and _is_python_indentation_sensitive_stop(stop_token)
        )
        variants = [stop_token]
        without_trailing_newline = stop_token.rstrip("\r\n")
        if without_trailing_newline and without_trailing_newline not in variants:
            variants.append(without_trailing_newline)
        for variant in variants:
            index = text.find(variant)
            if index >= 0:
                if python_indent_sensitive:
                    # For Python, only match if this is at zero indentation
                    if not _is_at_zero_indent(text, index):
                        continue
                stop_index = min(stop_index, index)
    return text[:stop_index]


def _append_missing_function_brace(
    text: str,
    prefix: str,
    tests: str,
    stop_tokens: list[str],
    language: str,
) -> str:
    """Close only a function-level brace omitted at the chat response boundary."""

    if evaluator_supplies_function_closure(tests, stop_tokens):
        return text
    if language not in {"go", "go_test.go", "js", "php", "pl", "r", "ts"}:
        return text

    closure_index, prefix_depth, final_depth = _scan_braces(text, prefix)
    if closure_index is not None:
        return text
    # If final_depth is larger, a nested block is also incomplete. That is a
    # substantive incomplete answer rather than a response-boundary omission.
    if prefix_depth > 0 and final_depth == prefix_depth:
        return text.rstrip() + "\n}"
    return text


def _scan_clojure_parens(text: str) -> int:
    """Count Clojure paren depth while ignoring strings and line comments."""

    depth = 0
    quote = False
    escaped = False
    line_comment = False
    for char in text:
        if line_comment:
            if char == "\n":
                line_comment = False
            continue
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quote = False
            continue
        if char == ";":
            line_comment = True
        elif char == '"':
            quote = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
    return depth


def _append_missing_clojure_function_paren(
    text: str,
    prefix: str,
    tests: str,
    stop_tokens: list[str],
    language: str,
) -> str:
    """Close exactly one outer Clojure function delimiter at the cursor."""

    if language != "clj" or evaluator_supplies_function_closure(tests, stop_tokens):
        return text
    prefix_depth = _scan_clojure_parens(prefix)
    final_depth = _scan_clojure_parens(prefix + text)
    # The prefix starts the defn and the response closed every nested form,
    # leaving only the function-level opening paren for the evaluator boundary.
    if prefix_depth == 1 and final_depth == 1:
        return text.rstrip() + ")"
    return text


def _strip_quoted_text_and_line_comments(text: str) -> str:
    """Blank quoted text and hash comments while preserving token boundaries."""

    result = []
    quote = None
    escaped = False
    line_comment = False
    for char in text:
        if line_comment:
            if char == "\n":
                line_comment = False
                result.append(char)
            else:
                result.append(" ")
            continue
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            result.append(" ")
            continue
        if char == "#":
            line_comment = True
            result.append(" ")
        elif char in ("'", '"'):
            quote = char
            result.append(" ")
        else:
            result.append(char)
    return "".join(result)


def _append_missing_julia_end(text: str, prefix: str, language: str) -> str:
    """Append Julia's function-level end when all nested blocks are closed."""

    if language != "jl":
        return text
    source = _last_nonempty_line(prefix) + "\n" + text
    source = _strip_quoted_text_and_line_comments(source)
    tokens = re.findall(
        r"\b(?:baremodule|begin|do|for|function|if|let|macro|module|quote|"
        r"struct|try|while|end)\b",
        source,
    )
    depth = 0
    for token in tokens:
        if token == "end":
            depth -= 1
        else:
            depth += 1
    if depth == 1:
        return text.rstrip() + "\nend"
    return text


def _append_missing_elixir_ends(text: str, prefix: str, language: str) -> str:
    """Balance function/module ends omitted at an Elixir response boundary."""

    if language != "elixir":
        return text
    source = prefix + text
    source = _strip_quoted_text_and_line_comments(source)
    tokens = re.findall(r"\bdo\b(?!\s*:)|\bfn\b|\bend\b", source)
    depth = 0
    for token in tokens:
        if token == "end":
            depth -= 1
        else:
            depth += 1
    if depth > 0:
        return text.rstrip() + "\n" + "\n".join("end" for _ in range(depth))
    return text


def normalize_completion(
    raw: str,
    prefix: str,
    tests: str,
    stop_tokens: list[str],
    language: str,
) -> str:
    """Convert a chat response into a MultiPL-E continuation string."""

    text = _remove_reasoning_and_special_tokens(raw.replace("\r\n", "\n"))
    text = _extract_code_block(text, prefix)
    text = _remove_repeated_prefix(text, prefix)
    text = _remove_repeated_opening_brace(text, prefix)
    text = _ensure_cursor_separator(text, prefix, language)
    text = _truncate_at_stop_tokens(
        text, prefix, tests, stop_tokens, language
    )
    text = _append_missing_function_brace(
        text, prefix, tests, stop_tokens, language
    )
    text = _append_missing_clojure_function_paren(
        text, prefix, tests, stop_tokens, language
    )
    text = _append_missing_julia_end(text, prefix, language)
    text = _append_missing_elixir_ends(text, prefix, language)

    # Markdown extraction can leave a fence when the model emits an incomplete
    # block. Remove only the fence marker, preserving code indentation.
    text = text.replace("```", "")
    return text.rstrip()

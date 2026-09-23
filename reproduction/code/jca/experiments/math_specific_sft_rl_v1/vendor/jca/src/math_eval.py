"""Utilities for local MATH protocol rollouts and SFT data prep."""

from __future__ import annotations

import ast
import math
import re
import warnings
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional


MATH_EVAL_VERSION = "math_eval_symbolic_v4"
MATH_SOFT_F1_VERSION = "math_soft_f1_v1"
_SYMBOLIC_MAX_LENGTH = 512
_SYMBOLIC_MAX_LITERAL_DIGITS = 18
_SYMBOLIC_MAX_EXPONENT = 128
# Factorials grow super-exponentially and can make an otherwise bounded
# expression trigger unbounded SymPy integer construction (for example
# ``2**(99!)``).  MATH answers that need symbolic factorial equivalence are
# covered well below this safety ceiling; larger factorials are treated as
# unsupported rather than allowed to stall corpus evaluation.
_SYMBOLIC_MAX_FACTORIAL_ARGUMENT = 32
_SYMBOLIC_MAX_OPERATIONS = 96
_SYMBOLIC_MAX_IDENTIFIERS = 24
_SYMBOLIC_MAX_PAREN_DEPTH = 24
_SYMBOLIC_FUNCTION_NAMES = frozenset(
    {
        "abs",
        "arccos",
        "arcsin",
        "arctan",
        "binomial",
        "ceil",
        "cos",
        "cot",
        "csc",
        "exp",
        "floor",
        "gcd",
        "lcm",
        "ln",
        "log",
        "max",
        "min",
        "root",
        "sec",
        "sin",
        "sqrt",
        "tan",
    }
)
_SYMBOLIC_CONSTANT_NAMES = frozenset(
    {"E", "I", "oo", "pi", "theta", "alpha", "beta", "gamma", "delta", "phi", "rho", "tau", "omega"}
)
_UNICODE_SYMBOL_REPLACEMENTS = {
    "θ": "theta",
    "ϑ": "theta",
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "φ": "phi",
    "ϕ": "phi",
    "ρ": "rho",
    "τ": "tau",
    "ω": "omega",
    "ℝ": "R",
    "ℤ": "Z",
    "ℕ": "N",
}
_SUPERSCRIPT_DIGITS = str.maketrans(
    {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "-"}
)
_SYMBOLIC_ALLOWED_CHARS = re.compile(r"^[A-Za-z0-9_\s+*/^().,=<>!\[\]{}\-]+$")
_LATEX_TEXT_COMMANDS = re.compile(
    r"\\(?:text|mathrm|mathbf|mathit|operatorname|textrm|mbox)\s*\{([^{}]*)\}"
)


AGENT_IDS: tuple[str, str, str] = ("A1", "A2", "A3")


@dataclass
class MathProblem:
    problem_id: str
    subject: str
    level: str
    prompt: str
    solution: str
    gold_answer: str


def _load_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required to read local MATH parquet files.") from exc
    return pd


def extract_last_boxed(text: str) -> Optional[str]:
    marker_pos = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if marker_pos < 0:
        return None

    brace_start = text.find("{", marker_pos)
    if brace_start < 0:
        return None

    depth = 0
    for idx in range(brace_start, len(text)):
        ch = text[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1 : idx].strip()
    return None


def fallback_answer(text: str) -> Optional[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None

    last_line = lines[-1].rstrip(".")
    patterns = [
        r"(?i)(?:the answer is|answer:|thus|therefore|so)\s*(.+)$",
        r"(?i)=\s*(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, last_line)
        if match:
            candidate = match.group(1).strip()
            if candidate:
                return candidate
    return None


def normalize_math_answer(text: str) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    s = s.replace("\\boxed{", "").replace("\\fbox{", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("−", "-").replace("–", "-")
    s = s.replace("π", "\\pi").replace("√", "\\sqrt")
    s = re.sub(r"\\(?:text|mathrm|mathbf|mathit|operatorname)\{([^{}]*)\}", r"\1", s)
    s = s.replace("{", "").replace("}", "")
    s = s.replace("$", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\,", "").replace("\\!", "")
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", "", s)
    s = s.rstrip(".")
    return s.lower()


def compute_math_em(prediction: str, gold_answer: str) -> float:
    if not prediction:
        return 0.0
    return 1.0 if math_answers_equivalent(prediction, gold_answer) else 0.0


def _strip_math_wrappers(value: str) -> str:
    value = str(value or "").strip()
    value = value.replace("\r", " ").replace("\n", " ")
    value = value.replace(r"\$", "")
    value = value.replace("$", "")
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\,", " ").replace("\\!", " ")
    value = value.replace("\\;", " ").replace("\\:", " ")
    for _ in range(4):
        unwrapped = re.sub(
            r"^\s*\\(?:boxed|fbox)\s*\{(.*)\}\s*$",
            r"\1",
            value,
            flags=re.DOTALL,
        )
        if unwrapped == value:
            break
        value = unwrapped.strip()
    return value.strip().rstrip(".")


def _replace_braced_command(value: str, command: str, replacement: str) -> str:
    pattern = re.compile(
        rf"\\{command}\s*\{{([^{{}}]*)\}}\s*\{{([^{{}}]*)\}}"
    )
    while True:
        updated, count = pattern.subn(
            lambda match: replacement.format(
                numerator=match.group(1), denominator=match.group(2)
            ),
            value,
        )
        value = updated
        if count == 0:
            return value


def _replace_single_braced_command(value: str, command: str, replacement: str) -> str:
    pattern = re.compile(rf"\\{command}\s*\{{([^{{}}]*)\}}")
    while True:
        updated, count = pattern.subn(
            lambda match: replacement.format(argument=match.group(1)),
            value,
        )
        value = updated
        if count == 0:
            return value


def _replace_mixed_latex_fractions(value: str) -> str:
    r"""Interpret LaTeX mixed numbers as an integer plus a fraction.

    Besides the canonical ``2\frac{1}{3}`` spelling, MATH answers often use
    TeX's single-token shorthand (``6\frac15``) and mix braced and unbraced
    arguments (``6\frac{1}5`` or ``6\frac1{5}``).  An unbraced argument is
    intentionally limited to one numeric token: that matches TeX's token
    semantics and prevents ``\frac123`` from being silently read as
    ``12/3`` instead of ``1/2`` followed by ``3``.
    """
    pattern = re.compile(
        r"(?<![A-Za-z0-9_.)])([+-]?\d+)\s*\\frac\s*"
        r"(?:\{([^{}]*)\}|([+-]?\d))\s*"
        r"(?:\{([^{}]*)\}|([+-]?\d))"
    )

    def replace(match: re.Match[str]) -> str:
        integer = match.group(1)
        numerator = match.group(2) if match.group(2) is not None else match.group(3)
        denominator = match.group(4) if match.group(4) is not None else match.group(5)
        sign = "-" if integer.startswith("-") else ""
        magnitude = integer.lstrip("+-")
        mixed = f"({magnitude}+(({numerator})/({denominator})))"
        return f"-{mixed}" if sign else mixed

    return pattern.sub(replace, value)


def _replace_mixed_plain_fractions(value: str) -> str:
    """Interpret plain mixed numbers such as ``6 3/4`` as ``6 + 3/4``."""
    pattern = re.compile(
        r"(?<![A-Za-z0-9_.)])([+-]?\d+)\s+(\d+)\s*/\s*(\d+)(?![A-Za-z0-9_])"
    )

    def replace(match: re.Match[str]) -> str:
        integer = match.group(1)
        numerator = match.group(2)
        denominator = match.group(3)
        sign = "-" if integer.startswith("-") else ""
        magnitude = integer.lstrip("+-")
        mixed = f"({magnitude}+(({numerator})/({denominator})))"
        return f"-{mixed}" if sign else mixed

    return pattern.sub(replace, value)


def _split_top_level(value: str, delimiter: str = ",") -> List[str]:
    parts: List[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(value):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth < 0:
                return []
        elif char == delimiter and depth == 0:
            part = value[start:index].strip()
            if not part:
                return []
            parts.append(part)
            start = index + 1
    if depth != 0:
        return []
    tail = value[start:].strip()
    if not tail:
        return []
    parts.append(tail)
    return parts


def _balanced_math_delimiters(value: str) -> bool:
    stack: List[str] = []
    pairs = {")": "(", "]": "[", "}": "{",
    }
    for char in value:
        if char in "([{":
            stack.append(char)
        elif char in pairs:
            if not stack or stack.pop() != pairs[char]:
                return False
    return not stack


def _latex_to_sympy_source(text: str) -> Optional[str]:
    value = _strip_math_wrappers(text)
    if not value or len(value) > _SYMBOLIC_MAX_LENGTH:
        return None
    value = value.replace("−", "-").replace("–", "-")
    for source, replacement in _UNICODE_SYMBOL_REPLACEMENTS.items():
        value = value.replace(source, replacement)
    value = re.sub(
        r"[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+",
        lambda match: "^" + match.group(0).translate(_SUPERSCRIPT_DIGITS),
        value,
    )
    value = value.replace("π", "pi").replace("√", "\\sqrt")
    value = value.replace("∞", "\\infty").replace("×", "*")
    value = value.replace("≤", "<=").replace("≥", ">=").replace("≠", "!=")
    value = value.replace("÷", "/").replace("⋅", "*")
    value = value.replace("∪", "\\cup")
    value = (
        value.replace("\\dfrac", "\\frac")
        .replace("\\tfrac", "\\frac")
        .replace("\\cfrac", "\\frac")
    )
    # Currency is presentation, not part of a mathematical value.  Remove
    # only the explicit LaTeX currency escape; arbitrary backslash commands
    # remain rejected below.
    value = value.replace("\\$", "")
    # Keep commas intact: in MATH answers they most often delimit ordered
    # pairs or root sequences.  Treating every digit-comma-digit as a
    # thousands separator turns ``1,2`` into the scalar ``12`` and can create
    # false equivalence with an interval.

    for command in ("text", "mathrm", "mathbf", "mathit", "operatorname", "textrm", "mbox"):
        for _ in range(8):
            updated = _LATEX_TEXT_COMMANDS.sub(r"\1", value)
            if updated == value:
                break
            value = updated

    value = re.sub(
        r"\\(?:arccos|arcsin|arctan|sin|cos|tan|sec|csc|cot|log|ln|exp)"
        r"(?=\\sqrt\b)",
        lambda match: match.group(0) + " ",
        value,
    )
    value = re.sub(
        r"(?<=[A-Za-z0-9_)])"
        r"(?=\\(?:arccos|arcsin|arctan|sin|cos|tan|sec|csc|cot|log|ln|exp|sqrt)\b)",
        "*",
        value,
    )
    value = re.sub(
        r"\\sqrt\s*\[([^\[\]]+)\]\s*\{([^{}]*)\}",
        r"root(\2,\1)",
        value,
    )
    value = _replace_single_braced_command(value, "sqrt", "sqrt({argument})")
    value = _replace_mixed_latex_fractions(value)
    value = _replace_mixed_plain_fractions(value)
    value = _replace_braced_command(value, "frac", "(({numerator})/({denominator}))")
    # A radical may contain a fraction (for example
    # ``\sqrt{\frac{1}{2}}``).  Fractions are reduced above, so run the
    # balanced single-argument rewrite once more after that reduction; the
    # earlier pass cannot see through the nested fraction braces.
    value = _replace_single_braced_command(value, "sqrt", "sqrt({argument})")
    # MATH solutions contain both canonical ``\frac{a}{b}`` and compact
    # variants such as ``\frac9{5}`` or ``\frac{9}5``.  An unbraced TeX
    # argument is one token, so keep the compact form to a single digit; a
    # multi-digit argument must use braces and is handled by the first form.
    fraction_atom = r"[+-]?\d"
    for _ in range(4):
        value = re.sub(
            rf"\\frac\s*\{{([^{{}}]*)\}}\s*({fraction_atom})",
            r"((\1)/(\2))",
            value,
        )
        value = re.sub(
            rf"\\frac\s*({fraction_atom})\s*\{{([^{{}}]*)\}}",
            r"((\1)/(\2))",
            value,
        )
        value = re.sub(
            rf"\\frac\s*({fraction_atom})\s*({fraction_atom})",
            r"((\1)/(\2))",
            value,
        )
        value = re.sub(
            rf"\\frac\s*({fraction_atom})\s+({fraction_atom})(?![A-Za-z0-9_])",
            r"((\1)/(\2))",
            value,
        )
    value = _replace_braced_command(value, "binom", "binomial({numerator},{denominator})")
    value = _replace_single_braced_command(value, "overline", "({argument})")
    value = _replace_single_braced_command(value, "bar", "({argument})")
    value = re.sub(r"\\sqrt\s*([A-Za-z0-9]+)", r"sqrt(\1)", value)
    value = re.sub(r"\\sqrt\s*\(([^()]*)\)", r"sqrt(\1)", value)
    value = re.sub(r"\\sqrt\s*\[([^\[\]]+)\]\s*\(([^()]*)\)", r"root(\2,\1)", value)
    value = re.sub(r"\\infty\b", "oo", value)
    value = re.sub(r"\\(?:cdot|times)\b", "*", value)
    value = re.sub(r"\\(?:div)\b", "/", value)
    value = re.sub(r"\^\s*\{\s*\\circ\s*\}", "", value)
    value = re.sub(r"\{\s*\\circ\s*\}", "", value)
    bare_argument = (
        r"(?:\\frac\s*\{[^{}]*\}\s*\{[^{}]*\}"
        r"|\\sqrt\s*(?:\{[^{}]*\}|\([^()]*\)|[A-Za-z0-9]+)"
        r"|sqrt\s*\([^()]*\)"
        r"|\{[^{}]*\}"
        r"|\\(?:pi|theta|alpha|beta|gamma|delta|phi|rho|tau|omega|infty)"
        r"|[+-]?(?:[A-Za-z](?:_?\d+)?|\d+(?:\.\d+)?))"
    )
    bare_product = rf"{bare_argument}(?:\s*(?![+-]){bare_argument})*"
    for function in (
        "arccos", "arcsin", "arctan", "sin", "cos", "tan", "sec", "csc", "cot",
        "log", "ln", "exp",
    ):
        value = re.sub(
            rf"\\{function}(?![A-Za-z])\s*(?!\()({bare_product})",
            lambda match, name=function: f"{name}({match.group(1).strip()})",
            value,
        )
    value = re.sub(
        r"\\(?:log|ln|sin|cos|tan|arcsin|arccos|arctan|exp)\b",
        lambda match: match.group(0).lstrip("\\"),
        value,
    )
    value = re.sub(r"\\(?:displaystyle|textstyle|scriptstyle)\b", "", value)
    value = re.sub(r"\\mathbb\s*\{([A-Za-z])\}", r"\1", value)
    for command, replacement in {
        "theta": "theta",
        "alpha": "alpha",
        "beta": "beta",
        "gamma": "gamma",
        "delta": "delta",
        "phi": "phi",
        "rho": "rho",
        "tau": "tau",
        "omega": "omega",
    }.items():
        value = re.sub(rf"\\{command}\b", replacement, value)
    value = re.sub(r"\^?\s*\\circ\b", "", value)
    value = value.replace("°", "")
    # Keep trigonometric functions as actual SymPy calls.  Rewriting ``\sec
    # x`` to ``(1/cos) x`` (the old behavior) either failed to parse or meant
    # multiplication by ``x`` instead of secant of x.
    for function in ("sec", "csc", "cot"):
        value = re.sub(
            rf"\\{function}\s*(\([^()]*\)|[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?)",
            rf"{function}(\1)",
            value,
        )
        value = re.sub(rf"\\{function}\b", function, value)
    value = re.sub(r"\\(?:cup)\b", " ", value)
    # A command may be followed immediately by a digit/variable (for
    # example ``\leq82`` in compact MATH answers), so a word-boundary check
    # is too strict here.
    value = re.sub(r"\\(?:le|leq)(?![A-Za-z])", "<=", value)
    value = re.sub(r"\\(?:ge|geq)(?![A-Za-z])", ">=", value)
    value = re.sub(r"\\ne(?![A-Za-z])", "!=", value)
    value = re.sub(r"\\,|\\!|\\;|\\:", " ", value)
    # Remove conventional thousands separators wherever they occur inside an
    # expression (including a fraction denominator), while leaving short
    # comma-separated tuples such as ``1,2`` untouched.
    value = re.sub(r"(?<=\d),(?=\s?\d{3}(?!\d))", "", value)
    value = re.sub(r"(?<=\d)\s+(?=\d{3}(?!\d))", "", value)
    value = re.sub(r"\\(?:pi)\b", "pi", value)
    # LaTeX permits implicit multiplication directly before a command, but
    # stripping the backslash can glue the command name to the preceding
    # symbol (``x\\sqrt{x}`` -> ``xsqrt(x)``).  Separate function calls and
    # named constants before SymPy tokenizes the source.
    function_names = (
        "abs|arccos|arcsin|arctan|binomial|ceil|cos|cot|csc|exp|floor|"
        "gcd|lcm|ln|log|max|min|root|sec|sin|sqrt|tan"
    )
    value = re.sub(
        rf"(?<=[A-Za-z0-9_)])(?=(?<![A-Za-z_])(?:{function_names})\s*\()",
        "*",
        value,
    )
    value = re.sub(
        r"(?<=[A-Za-z0-9_)])(?=(?:pi|oo|I|E)(?![A-Za-z0-9_]))",
        "*",
        value,
    )
    value = re.sub(r"(?<=[0-9)])i(?![A-Za-z])", "*I", value)
    value = re.sub(r"(?<![A-Za-z])i(?![A-Za-z])", "I", value)
    value = value.replace("\\%", "%")
    if "\\pm" in value or "\\mp" in value:
        return None
    value = re.sub(r"(?<![A-Za-z])%", "/100", value)
    value = value.replace("^", "**")
    value = value.replace("{", "(").replace("}", ")")
    if "\\" in value:
        return None
    value = re.sub(r"\s+", " ", value).strip()
    # Treat conventional thousands separators as presentation only, but keep
    # short comma-separated answers (ordered pairs/root lists) intact.
    if re.fullmatch(r"[+-]?\d{1,3}(?:, ?\d{3})+", value):
        value = value.replace(",", "").replace(" ", "")
    elif re.fullmatch(r"[+-]?\d{1,3}(?: \d{3})+", value):
        value = value.replace(" ", "")
    if not value or len(value) > _SYMBOLIC_MAX_LENGTH:
        return None
    if value.count("**") > 16 or re.search(r"\*\*\s*[1-9]\d{3,}", value):
        return None
    factorial_literals = re.finditer(r"(?<![A-Za-z_=])([+-]?\d+)\s*!(?!=)", value)
    for factorial in factorial_literals:
        if abs(int(factorial.group(1))) > _SYMBOLIC_MAX_FACTORIAL_ARGUMENT:
            return None
    if re.search(r"\)\s*!(?!=)|!(?!=)\s*!", value):
        return None
    if any(
        len(match.group(0).lstrip("+-")) > _SYMBOLIC_MAX_LITERAL_DIGITS
        for match in re.finditer(r"(?<![A-Za-z_])[+-]?\d+(?:\.\d+)?", value)
    ):
        return None
    # Bound literal and nested exponents before SymPy sees them.  Parenthesized
    # powers (for example ``2**(1000000)``) used to bypass the old direct-digit
    # check and could trigger enormous integer construction during parsing.
    for power_match in re.finditer(r"\*\*\s*", value):
        exponent_start = power_match.end()
        while exponent_start < len(value) and value[exponent_start].isspace():
            exponent_start += 1
        if exponent_start >= len(value):
            return None
        if value[exponent_start] == "(":
            depth = 0
            exponent_end = -1
            for index in range(exponent_start, len(value)):
                if value[index] == "(":
                    depth += 1
                elif value[index] == ")":
                    depth -= 1
                    if depth == 0:
                        exponent_end = index
                        break
                    if depth < 0:
                        return None
            if exponent_end < 0:
                return None
            exponent = value[exponent_start + 1 : exponent_end].strip()
            if not exponent or "**" in exponent or len(exponent) > 64:
                return None
            literals = re.findall(r"(?<![A-Za-z_])[+-]?\d+", exponent)
            if any(abs(int(literal)) > _SYMBOLIC_MAX_EXPONENT for literal in literals):
                return None
        else:
            literal_match = re.match(r"[+-]?\d+", value[exponent_start:])
            if literal_match is not None and abs(int(literal_match.group(0))) > _SYMBOLIC_MAX_EXPONENT:
                return None
    if sum(value.count(operator) for operator in ("+", "-", "*", "/", "**")) > _SYMBOLIC_MAX_OPERATIONS:
        return None
    if max((len(part) for part in value.split("(")), default=0) > _SYMBOLIC_MAX_LENGTH:
        return None
    if value.count("(") > _SYMBOLIC_MAX_PAREN_DEPTH:
        return None
    if any(fragment in value for fragment in ("__", "'", '"', ";", "`", "\n")):
        return None
    if not _SYMBOLIC_ALLOWED_CHARS.fullmatch(value) or not _balanced_math_delimiters(value):
        return None
    identifiers = set(re.findall(r"\b[A-Za-z_]\w*\b", value))
    if len(identifiers) > _SYMBOLIC_MAX_IDENTIFIERS:
        return None
    for identifier in identifiers:
        if identifier in _SYMBOLIC_FUNCTION_NAMES or identifier in _SYMBOLIC_CONSTANT_NAMES:
            continue
        if re.fullmatch(r"[A-Za-z](?:_?\d+)?", identifier):
            continue
        return None
    return value


def _cheap_numeric_value(source: str):
    """Evaluate a small numeric expression without invoking SymPy.

    This fast path handles the very common ``1/2`` versus ``\\frac{1}{2}``
    cases and rejects unequal numeric candidates immediately.  It deliberately
    accepts only an AST consisting of arithmetic operators and bounded powers;
    anything involving symbols, calls, relations, or collections falls back
    to the restricted symbolic parser.
    """
    if re.search(r"\b[A-Za-z_]\w*\b", source):
        return None
    if any(character in source for character in "<>=!,[]{}"):  # ``!`` may be factorial
        return None
    try:
        tree = ast.parse(source, mode="eval")
    except (SyntaxError, ValueError):
        return None

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if isinstance(node.value, bool):
                return None
            if isinstance(node.value, int):
                return Fraction(node.value)
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = evaluate(node.operand)
            if operand is None:
                return None
            return operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.BinOp):
            left = evaluate(node.left)
            right = evaluate(node.right)
            if left is None or right is None:
                return None
            try:
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    if right == 0:
                        return None
                    return left / right
                if isinstance(node.op, ast.Pow):
                    if right.denominator != 1 or abs(right.numerator) > _SYMBOLIC_MAX_EXPONENT:
                        return None
                    return left ** right.numerator
            except (ArithmeticError, OverflowError, ValueError):
                return None
        return None

    return evaluate(tree)


def _cheap_numeric_equal(left, right) -> bool:
    if left is None or right is None:
        return False
    # Decimal literals are converted to exact Fractions above.  A tolerance
    # here would silently turn a wrong answer such as 1.00000000001 into 1.
    # Approximate transcendental expressions use the separately bounded
    # SymPy path instead.
    return left == right


@lru_cache(maxsize=4096)
def _parse_symbolic_cached(source: str):
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import (
            convert_xor,
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )

        def math_root(argument, index):
            # MATH answers use real-valued roots.  SymPy's ``root`` follows
            # the principal complex branch for a negative radicand, so an
            # odd root such as ``root(-8, 3)`` otherwise fails to match ``-2``.
            # Preserve the principal behavior for even or non-integer roots.
            try:
                if index.is_integer is True and index.is_odd is True:
                    return sp.real_root(argument, index)
            except Exception:
                pass
            return sp.root(argument, index)

        reserved = {
            "pi": sp.pi,
            "I": sp.I,
            "oo": sp.oo,
            "sqrt": sp.sqrt,
            "root": math_root,
            "arcsin": sp.asin,
            "arccos": sp.acos,
            "arctan": sp.atan,
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "sec": sp.sec,
            "csc": sp.csc,
            "cot": sp.cot,
            "log": sp.log,
            "ln": sp.log,
            "exp": sp.exp,
            "E": sp.E,
            "abs": sp.Abs,
            "binomial": sp.binomial,
            "floor": sp.floor,
            "ceil": sp.ceiling,
            "gcd": sp.gcd,
            "lcm": sp.lcm,
            "min": sp.Min,
            "max": sp.Max,
            "Abs": sp.Abs,
        }
        identifiers = set(re.findall(r"\b[A-Za-z_]\w*\b", source))
        function_call_identifiers = set(
            re.findall(r"\b([A-Za-z_]\w*)\s*\(", source)
        )
        for identifier in identifiers:
            if identifier not in reserved:
                reserved[identifier] = (
                    sp.Function(identifier)
                    if identifier in function_call_identifiers
                    else sp.Symbol(identifier)
                )
        transformations = standard_transformations + (
            implicit_multiplication_application,
            convert_xor,
        )
        global_dict = dict(vars(sp))
        global_dict["__builtins__"] = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return parse_expr(
                source,
                local_dict=reserved,
                global_dict=global_dict,
                transformations=transformations,
                evaluate=True,
            )
    except Exception:
        return None


def _parse_math_object(text: str):
    interval_object = _parse_interval_object(text)
    if interval_object is not None:
        return interval_object
    source = _latex_to_sympy_source(text)
    if source is None:
        return None
    relation_matches = list(
        re.finditer(r"(?<![<>=])(<=|>=|!=|=|<|>)(?![<>=])", source)
    )
    if relation_matches:
        segments = []
        segment_start = 0
        for match in relation_matches:
            left_source = source[segment_start : match.start()].strip()
            if not left_source:
                return None
            segments.append((match.group(1), left_source))
            segment_start = match.end()
        right_source = source[segment_start:].strip()
        if not right_source:
            return None
        operands = [left_source for _, left_source in segments] + [right_source]
        parsed_operands = [_parse_symbolic_cached(operand) for operand in operands]
        if any(operand is None for operand in parsed_operands):
            return None
        relations = tuple(
            (segments[index][0], parsed_operands[index], parsed_operands[index + 1])
            for index in range(len(segments))
        )
        if len(relations) == 1:
            operator, left, right = relations[0]
            return ("relation", operator, left, right)
        return ("relation_chain", relations)

    # MATH answers frequently use ordered pairs or comma-separated roots.
    # Parse a parenthesized/list form explicitly; SymPy's implicit
    # multiplication otherwise turns ``(1,2)`` into the scalar ``12``.
    sequence_source = source.strip()
    parts: List[str] = []
    if (
        len(sequence_source) >= 2
        and sequence_source[0] in "([{"
        and sequence_source[-1] == {"(": ")", "[": "]", "{": "}"}[sequence_source[0]]
        and _balanced_math_delimiters(sequence_source)
    ):
        parts = _split_top_level(sequence_source[1:-1])
    if len(parts) <= 1:
        parts = _split_top_level(sequence_source)
    if len(parts) > 1:
        parsed_parts = [_parse_symbolic_cached(part) for part in parts]
        if all(part is not None for part in parsed_parts):
            return ("sequence", tuple(parsed_parts))
    parsed = _parse_symbolic_cached(source)
    if parsed is None:
        return None
    return ("expr", parsed)


def _parse_interval_object(text: str):
    """Parse interval/union notation while comparing endpoints symbolically."""
    import re as _re

    value = _strip_math_wrappers(text)
    value = value.replace("\\left", "").replace("\\right", "").replace("∪", "\\cup")
    pieces = _re.split(r"\\cup\b", value)
    if not pieces or any(not piece.strip() for piece in pieces):
        return None
    intervals = []
    for piece in pieces:
        stripped_piece = piece.strip()
        if len(stripped_piece) < 2 or stripped_piece[0] not in "[(":
            return None
        if stripped_piece[-1] not in ")]" or not _balanced_math_delimiters(
            stripped_piece[1:-1]
        ):
            return None
        # A sequence such as ``(1,2,3)`` is not an interval whose right
        # endpoint happens to contain a comma.  Split only at top-level
        # delimiters and require exactly two endpoint expressions; returning
        # ``None`` for longer sequences lets the normal sequence parser handle
        # them below.
        interval_parts = _split_top_level(stripped_piece[1:-1])
        if len(interval_parts) != 2:
            return None
        left_source = _latex_to_sympy_source(interval_parts[0])
        right_source = _latex_to_sympy_source(interval_parts[1])
        if left_source is None or right_source is None:
            return None
        left = _parse_symbolic_cached(left_source)
        right = _parse_symbolic_cached(right_source)
        if left is None or right is None:
            return None
        intervals.append((left, right, stripped_piece[0] == "[", stripped_piece[-1] == "]"))
    return ("interval_union", tuple(intervals))


def _sympy_equal(left, right) -> bool:
    import sympy as sp

    try:
        if left == right:
            return True
        # Avoid expensive simplification on unexpectedly large symbolic trees.
        # The parser already applies strict textual limits; this second bound
        # catches compact but combinatorially explosive expressions.
        if sp.count_ops(left) > _SYMBOLIC_MAX_OPERATIONS * 4 or sp.count_ops(right) > _SYMBOLIC_MAX_OPERATIONS * 4:
            return False
        difference = sp.simplify(left - right)
        if difference == 0:
            return True
        # ``Expr.equals`` may spend seconds proving a transcendental identity
        # (and returns ``None`` when it cannot).  Simplification is sufficient
        # for the bounded algebraic forms used by MATH; reserve ``equals`` for
        # expressions without undefined/function-valued atoms.
        equals = getattr(difference, "equals", None)
        if callable(equals) and not difference.atoms(sp.Function) and equals(0) is True:
            return True
        if difference.is_number:
            numeric = complex(sp.N(difference, 30))
            return abs(numeric) <= 1e-10
    except Exception:
        return False
    return False


@lru_cache(maxsize=8192)
def _symbolic_equivalent_cached(prediction: str, gold_answer: str) -> bool:
    return _symbolic_equivalent_uncached(prediction, gold_answer)


def _symbolic_equivalent(prediction: str, gold_answer: str) -> bool:
    """Compatibility wrapper for callers of the pre-v3 private helper."""
    return _symbolic_equivalent_cached(str(prediction), str(gold_answer))


def _symbolic_equivalent_uncached(prediction: str, gold_answer: str) -> bool:
    try:
        predicted = _parse_math_object(prediction)
        if predicted is None:
            return False
        expected = _parse_math_object(gold_answer)
    except (ImportError, ModuleNotFoundError):
        return False
    if predicted is None or expected is None or predicted[0] != expected[0]:
        return False
    if predicted[0] == "expr":
        return _sympy_equal(predicted[1], expected[1])
    if predicted[0] == "sequence":
        left, right = predicted[1], expected[1]
        return len(left) == len(right) and all(_sympy_equal(a, b) for a, b in zip(left, right))
    if predicted[0] == "interval_union":
        left_intervals, right_intervals = predicted[1], expected[1]
        if len(left_intervals) != len(right_intervals):
            return False
        return all(
            left[2:] == right[2:]
            and _sympy_equal(left[0], right[0])
            and _sympy_equal(left[1], right[1])
            for left, right in zip(left_intervals, right_intervals)
        )
    if predicted[0] == "relation_chain":
        predicted_relations = predicted[1]
        expected_relations = expected[1]
        if len(predicted_relations) != len(expected_relations):
            return False
        direct = all(
            _relation_equivalent(left, right)
            for left, right in zip(predicted_relations, expected_relations)
        )
        if direct:
            return True
        reversed_expected = tuple(
            _invert_relation(relation)
            for relation in reversed(expected_relations)
        )
        return all(
            _relation_equivalent(left, right)
            for left, right in zip(predicted_relations, reversed_expected)
        )
    if predicted[0] == "relation":
        return _relation_equivalent(predicted[1:], expected[1:])
    return False


_INVERTED_RELATIONS = {"<": ">", ">": "<", "<=": ">=", ">=": "<="}


def _invert_relation(relation):
    operator, left, right = relation
    return (_INVERTED_RELATIONS.get(operator, operator), right, left)


def _relation_equivalent(predicted, expected) -> bool:
    predicted_operator, predicted_left, predicted_right = predicted
    expected_operator, expected_left, expected_right = expected

    def same_orientation(operator, left, right, other_operator, other_left, other_right):
        if operator != other_operator:
            return False
        try:
            left_difference = left - right
            other_difference = other_left - other_right
        except Exception:
            return False
        if operator == "=":
            if _sympy_equal(left_difference, other_difference) or _sympy_equal(
                left_difference, -other_difference
            ):
                return True
            try:
                import sympy as sp

                if other_difference == 0:
                    return False
                ratio = sp.simplify(left_difference / other_difference)
                return bool(
                    ratio.is_number
                    and ratio.is_finite is True
                    and ratio.is_zero is not True
                )
            except Exception:
                return False
        if _sympy_equal(left_difference, other_difference):
            return True
        # Multiplying both sides of an inequality by a positive factor keeps
        # its solution set unchanged.  Restrict the extra equivalence to a
        # provably numeric positive ratio; a sign-changing symbolic factor is
        # not safe to accept without domain analysis.
        try:
            import sympy as sp

            if other_difference == 0:
                return left_difference == 0
            ratio = sp.simplify(left_difference / other_difference)
            if ratio.is_number:
                return bool(ratio.is_positive)
        except Exception:
            pass
        return False

    return same_orientation(
        predicted_operator,
        predicted_left,
        predicted_right,
        expected_operator,
        expected_left,
        expected_right,
    ) or same_orientation(
        predicted_operator,
        predicted_left,
        predicted_right,
        *_invert_relation((expected_operator, expected_left, expected_right)),
    )


def math_answers_equivalent(prediction: str, gold_answer: str) -> bool:
    """Return whether two answer strings are exact or symbolically equivalent."""
    if not prediction or not gold_answer:
        return False
    if str(prediction).strip() == str(gold_answer).strip():
        return True
    # Preserve the historical exact-match behavior for answers that are not
    # valid scalar expressions (for example ``\text{Wednesday}`` or values
    # carrying units).  The symbolic parser below intentionally rejects such
    # prose, but an identical normalized answer is still an unambiguous match.
    normalized_prediction = normalize_math_answer(prediction)
    normalized_gold = normalize_math_answer(gold_answer)
    if normalized_prediction and normalized_prediction == normalized_gold:
        # ``normalize_math_answer`` is intentionally lightweight and removes
        # grouping braces.  That makes it suitable for prose/units that the
        # symbolic parser cannot represent, but it can collide for distinct
        # expressions such as ``\\frac{1}{13}`` and ``\\frac{11}{3}``.
        # Only use the fallback when at least one side is not parseable as a
        # mathematical object; parseable expressions must still be compared
        # semantically below.
        try:
            prediction_object = _parse_math_object(str(prediction))
            gold_object = _parse_math_object(str(gold_answer))
        except Exception:
            prediction_object = None
            gold_object = None
        if prediction_object is None or gold_object is None:
            return True
    # Resolve bounded purely numeric expressions before importing/consulting
    # SymPy.  Besides being much faster for large rollout corpora, this keeps
    # obviously unequal numbers from reaching expensive symbolic simplification.
    try:
        prediction_source = _latex_to_sympy_source(str(prediction))
        gold_source = _latex_to_sympy_source(str(gold_answer))
        if prediction_source is not None and gold_source is not None:
            prediction_numeric = _cheap_numeric_value(prediction_source)
            gold_numeric = _cheap_numeric_value(gold_source)
            if prediction_numeric is not None and gold_numeric is not None:
                return _cheap_numeric_equal(prediction_numeric, gold_numeric)
        # Some structured answers (notably intervals containing ``\\infty``)
        # are intentionally handled by ``_parse_math_object`` even when the
        # scalar source converter declines them.  Fall through to that parser
        # instead of treating one unavailable fast-path source as a mismatch.
    except Exception:
        # The restricted symbolic path below remains the source of truth for
        # expressions outside the cheap arithmetic grammar.
        pass
    try:
        return _symbolic_equivalent_cached(str(prediction), str(gold_answer))
    except Exception:
        return False


def finite_real_scalar(answer: str) -> Optional[Decimal]:
    """Parse a MATH answer as a finite real scalar when possible."""
    if not isinstance(answer, str) or not answer.strip():
        return None
    try:
        parsed = _parse_math_object(answer)
        if parsed is None or parsed[0] != "expr":
            return None
        expression = parsed[1]
        if expression.free_symbols or expression.is_number is not True:
            return None
        numeric = expression.evalf(50)
        if numeric.is_real is not True or numeric.is_finite is not True:
            return None
        value = Decimal(str(numeric))
        if not value.is_finite():
            return None
        return value
    except (ArithmeticError, AttributeError, InvalidOperation, TypeError, ValueError):
        return None


def gsm_numeric_ratio(prediction: Decimal, gold: Decimal) -> float:
    """Return the sign-aware min/max ratio used by GSM soft F1."""
    if prediction == gold:
        return 1.0
    if prediction == 0 or gold == 0:
        return 0.0
    if (prediction > 0) != (gold > 0):
        return 0.0
    ratio = min(abs(prediction), abs(gold)) / max(abs(prediction), abs(gold))
    result = float(ratio)
    return result if math.isfinite(result) and 0.0 <= result <= 1.0 else 0.0


def math_soft_f1(
    prediction: str,
    gold_answer: str,
) -> tuple[float, bool, Optional[float]]:
    """Return overall MATH soft F1 and numeric-subset eligibility/score."""
    if math_answers_equivalent(prediction, gold_answer):
        gold_scalar = finite_real_scalar(gold_answer)
        return 1.0, gold_scalar is not None, 1.0 if gold_scalar is not None else None
    gold_scalar = finite_real_scalar(gold_answer)
    if gold_scalar is None:
        return 0.0, False, None
    prediction_scalar = finite_real_scalar(prediction)
    numeric_score = (
        gsm_numeric_ratio(prediction_scalar, gold_scalar)
        if prediction_scalar is not None
        else 0.0
    )
    return numeric_score, True, numeric_score


def clean_solution_text(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    text = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", text)


def load_math_problems(
    data_root: str | Path,
    *,
    split: str = "train",
    subjects: Iterable[str] | None = None,
) -> List[MathProblem]:
    data_root = Path(data_root)
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported MATH split: {split}")

    subject_filter = {s.strip() for s in (subjects or []) if s and s.strip()}
    parquet_paths = sorted(data_root.glob(f"*/{split}-00000-of-00001.parquet"))
    if subject_filter:
        parquet_paths = [p for p in parquet_paths if p.parent.name in subject_filter]
    if not parquet_paths:
        raise FileNotFoundError(f"No MATH parquet files found under {data_root} for split={split}")

    pd = _load_pandas()
    problems: List[MathProblem] = []
    for parquet_path in parquet_paths:
        subject_dir = parquet_path.parent.name
        df = pd.read_parquet(parquet_path)
        for row_idx, row in enumerate(df.itertuples(index=False)):
            prompt = str(getattr(row, "problem")).strip()
            solution = str(getattr(row, "solution")).strip()
            gold_answer = extract_last_boxed(solution) or fallback_answer(solution)
            if not gold_answer:
                continue

            problems.append(
                MathProblem(
                    problem_id=f"{subject_dir}_{row_idx:05d}",
                    subject=str(getattr(row, "type", subject_dir)).strip(),
                    level=str(getattr(row, "level", "")).strip(),
                    prompt=prompt,
                    solution=clean_solution_text(solution),
                    gold_answer=gold_answer.strip(),
                )
            )
    return problems


def format_math_problem_as_prompt(problem: MathProblem) -> str:
    return (
        "# Mathematics Problem\n"
        f"{problem.prompt.strip()}\n\n"
        "Solve carefully. Your final answer must be a short mathematical expression "
        "or value, not a sentence."
    )


def format_other_agents(agent_id: str) -> str:
    others = [a for a in AGENT_IDS if a != agent_id]
    return f"{others[0]} and {others[1]}"


def render_math_mas_system_prompt(agent_id: str, *, min_agents_before_stop: int = 3) -> str:
    if agent_id not in AGENT_IDS:
        raise KeyError(f"Unknown agent_id: {agent_id}")
    min_agents_line = (
        f"4. Do not use confirm_stop until at least {min_agents_before_stop} distinct agents "
        "have contributed a turn in the conversation."
        if min_agents_before_stop > 1
        else ""
    )
    return f"""You are agent {agent_id}. You collaborate with two other agents: {format_other_agents(agent_id)}.

# The Task
You are solving a mathematics problem. The final answer may be a number,
simplified expression, equation, interval, ordered pair, polynomial, or
other short mathematical object.

# Shared Conversation
You and the other agents share this conversation. Each prior assistant
message is prefixed with the agent ID in brackets, for example: [A1] ...

# Collaborative Verification Protocol
The tentative answer must receive a verifier turn before it can be finalized.
Distinct-agent coverage is not required. A handoff may target any other agent,
including one that already contributed; do not choose a target merely because
it has not participated yet.

1. The first agent computes a tentative_answer and hands off for verification.
2. A verifier checks the existing derivation and tentative_answer.
   - If they agree and the derivation is fully checked: action = "confirm_stop".
   - If they disagree or want another check: give the corrected
     tentative_answer and action = "handoff".
3. confirm_stop is only valid once at least one previous tentative_answer
   exists in the conversation.
{min_agents_line}
5. Later agents should add checking value: test edge cases, validate algebra,
   or verify the final form of the answer.

# Output Format
Output EXACTLY one JSON object and nothing else:

{{
  "reasoning": "<step-by-step derivation>",
  "tentative_answer": "<final answer only, no prose>",
  "action": "handoff" | "confirm_stop",
  "handoff_target": "A1" | "A2" | "A3" | null,
  "handoff_note": "<brief verification note>" | null,
  "confirmed_answer": "<final answer only>" | null
}}

# Field Rules
- reasoning: show the derivation clearly enough to verify the answer.
- tentative_answer: the mathematical answer only; keep it concise.
- handoff:
    * confirmed_answer must be null.
    * handoff_target must be one of the other two agents.
    * never set handoff_target to your own agent ID.
    * handoff_note should name the exact quantity, case split, algebraic
      transformation, or final expression the next agent should check.
- confirm_stop:
    * confirmed_answer must contain the final answer.
    * only use this after a prior tentative_answer already exists.
    * only use this after enough distinct agents have contributed if the
      protocol requires that.
    * do not confirm_stop without checking the math.

# Strategy
- First turn:
    * solve the problem carefully,
    * provide a concrete tentative_answer,
    * hand off with a specific verification request.
- Always emit valid JSON. Never repeat empty math delimiters such as "$ $ $".
- Verifier turn:
    * check the previous reasoning and tentative_answer carefully,
    * if the prior answer is wrong, fix it and hand off,
    * if the prior answer seems right but the derivation is long or subtle,
      either confirm after a full check or hand off for one more useful check;
      the target may be an agent that already participated.
- Final verifier:
    * confirm only after checking the derivation carefully,
    * pay attention to arithmetic errors, sign mistakes, dropped cases,
      simplification mistakes, and answer-format mistakes.
- Collaboration guidance:
    * Use additional turns only when another check would reduce risk; never
      hand off solely to increase the number of distinct participating agents.
    * Handoffs should be actionable and mathematical, not generic.
    * Avoid empty collaboration: no useless handoff and no redundant confirm.
""".strip()

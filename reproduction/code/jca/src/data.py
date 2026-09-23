"""MuSiQue data loading + formatting.

Public API:
    load_musique(split) -> List[MuSiQueProblem]
    format_problem_as_prompt(problem) -> str

See `code-guide.md` §2.1 for the contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

# ============================================================================
# Data structures
# ============================================================================


@dataclass
class MuSiQueProblem:
    """One MuSiQue question."""

    id: str
    question: str
    paragraphs: List[Tuple[str, str]]   # (title, text) tuples; ~20 per question
    answer: str                          # ground truth
    answer_aliases: List[str]            # acceptable equivalents
    hop: int                             # 2, 3, or 4
    supporting_idx: List[int]            # which paragraph indices are gold supporting

    @property
    def n_paragraphs(self) -> int:
        return len(self.paragraphs)


# ============================================================================
# Loading
# ============================================================================


def load_musique(split: str, *, data_dir: str | Path = "musique_data") -> List[MuSiQueProblem]:
    """Load MuSiQue problems for a given split.

    Args:
        split: One of {"train", "dev", "test"}.
        data_dir: Where the raw MuSiQue files live (jsonl format).

    Returns:
        List of MuSiQueProblem.

    Expected local files are produced by scripts/download_musique.py:
        - musique_ans_train.jsonl
        - musique_ans_dev.jsonl
    """
    data_path = _split_to_path(split, Path(data_dir))
    problems: List[MuSiQueProblem] = []

    with data_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{data_path}:{line_no}: invalid JSON: {exc}") from exc
            problems.append(_parse_problem(raw, source=f"{data_path}:{line_no}"))

    return problems


def _split_to_path(split: str, data_dir: Path) -> Path:
    """Resolve a split name to the local MuSiQue-Ans JSONL path."""
    normalized = split.lower()
    split_files = {
        "train": "musique_ans_train.jsonl",
        "dev": "musique_ans_dev.jsonl",
        "valid": "musique_ans_dev.jsonl",
        "validation": "musique_ans_dev.jsonl",
        "test": "musique_ans_test.jsonl",
    }
    if normalized not in split_files:
        raise ValueError(f"Unknown MuSiQue split: {split!r}")

    path = data_dir / split_files[normalized]
    if not path.exists():
        raise FileNotFoundError(
            f"MuSiQue split file not found: {path}. "
            "Run `python scripts/download_musique.py` first."
        )
    return path


def _parse_problem(raw: dict[str, Any], *, source: str) -> MuSiQueProblem:
    """Convert one raw MuSiQue record into the project dataclass."""
    required = ("id", "question", "paragraphs", "answer", "answer_aliases")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"{source}: missing field(s): {missing}")

    paragraphs = _parse_paragraphs(raw["paragraphs"], source=source)
    decomposition = raw.get("question_decomposition") or []
    supporting_idx = _parse_supporting_idx(raw, decomposition)

    return MuSiQueProblem(
        id=str(raw["id"]),
        question=str(raw["question"]),
        paragraphs=paragraphs,
        answer=str(raw["answer"]),
        answer_aliases=[str(a) for a in (raw.get("answer_aliases") or [])],
        hop=len(decomposition) if decomposition else len(supporting_idx),
        supporting_idx=supporting_idx,
    )


def _parse_paragraphs(raw_paragraphs: Any, *, source: str) -> List[Tuple[str, str]]:
    """Parse paragraph objects from HF/official MuSiQue JSONL variants."""
    if not isinstance(raw_paragraphs, list) or not raw_paragraphs:
        raise ValueError(f"{source}: paragraphs must be a non-empty list")

    paragraphs: List[Tuple[str, str]] = []
    for idx, paragraph in enumerate(raw_paragraphs):
        if not isinstance(paragraph, dict):
            raise ValueError(f"{source}: paragraph {idx} must be an object")

        title = paragraph.get("title", "")
        text = paragraph.get("paragraph_text", paragraph.get("text", ""))
        if not isinstance(title, str) or not isinstance(text, str):
            raise ValueError(f"{source}: paragraph {idx} title/text must be strings")
        paragraphs.append((title, text))

    return paragraphs


def _parse_supporting_idx(raw: dict[str, Any], decomposition: Any) -> List[int]:
    """Extract gold supporting paragraph indices, preserving first-seen order."""
    indices: List[int] = []

    if isinstance(decomposition, list):
        for step in decomposition:
            if isinstance(step, dict) and "paragraph_support_idx" in step:
                value = step["paragraph_support_idx"]
                if isinstance(value, int) and value not in indices:
                    indices.append(value)

    if indices:
        return indices

    paragraphs = raw.get("paragraphs")
    if isinstance(paragraphs, list):
        for pos, paragraph in enumerate(paragraphs):
            if isinstance(paragraph, dict) and paragraph.get("is_supporting") is True:
                value = paragraph.get("idx", pos)
                if isinstance(value, int) and value not in indices:
                    indices.append(value)

    return indices


# ============================================================================
# Formatting (problem -> prompt-friendly string)
# ============================================================================


def format_problem_as_prompt(problem: MuSiQueProblem) -> str:
    """Produce the user-message content shown to agents.

    Format:

        # Paragraphs
        [1] (Title 1) text...
        [2] (Title 2) text...
        ...

        # Question
        ...

    Note: includes ALL paragraphs (even non-supporting), since agents share
    the same view—per design.md §1.2 / §7.1.
    """
    lines = ["# Paragraphs"]
    for idx, (title, text) in enumerate(problem.paragraphs, start=1):
        title = title.strip()
        text = " ".join(text.split())
        lines.append(f"[{idx}] ({title}) {text}")

    lines.extend(["", "# Question", problem.question.strip()])
    return "\n".join(lines)

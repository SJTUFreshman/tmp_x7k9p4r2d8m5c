"""GSM-HARD data loading and formatting."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class GSMProblem:
    id: str
    question: str
    answer: float          # numeric answer
    answer_str: str        # string representation for display


def load_gsm_hard(
    data_path: str | Path = "/data/wangyuheng/jca/Math/data/GSM-HARD/splits/gsmhardv2_dev.jsonl",
) -> List[GSMProblem]:
    """Load GSM-HARD problems from jsonl file."""
    path = Path(data_path)
    problems = []
    for i, line in enumerate(path.open(encoding="utf-8")):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        target = float(r["target"])
        # Format answer string: use int if whole number
        if target == int(target):
            answer_str = str(int(target))
        else:
            answer_str = str(target)
        problems.append(GSMProblem(
            id=f"gsm_{i:05d}",
            question=r["input"].strip(),
            answer=target,
            answer_str=answer_str,
        ))
    return problems


def format_problem_as_prompt(problem: GSMProblem) -> str:
    """Format a GSM-HARD problem as the user message."""
    return (
        f"# Math Problem\n{problem.question}\n\n"
        f"Solve step by step. Your final answer must be a single number."
    )


def extract_number(text: str) -> Optional[float]:
    """Extract the final numeric answer from model output.

    Handles formats like:
      - \\boxed{42}
      - The answer is 42
      - 42
      - -9867630
      - 3.14
    """
    if not text:
        return None

    # Try \boxed{...}
    m = re.search(r'\\boxed\{([^}]*)\}', text)
    if m:
        try:
            return float(m.group(1).replace(',', '').strip())
        except ValueError:
            pass

    number_pattern = r"[-+]?(?:(?:\d[\d,]*)(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"

    # Try "the answer is X" pattern
    m = re.search(
        rf'(?:the answer is|answer:|=)\s*({number_pattern})',
        text,
        re.I,
    )
    if m:
        try:
            return float(m.group(1).replace(',', '').strip())
        except ValueError:
            pass

    # Try last number in text
    numbers = re.findall(number_pattern, text)
    if numbers:
        try:
            return float(numbers[-1].replace(',', ''))
        except ValueError:
            pass

    return None

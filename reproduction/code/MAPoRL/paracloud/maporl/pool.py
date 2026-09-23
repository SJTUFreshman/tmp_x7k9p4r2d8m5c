"""Prompt pool with a resumable cursor.

Binary-reward datasets (GSM/MATH/MuSiQue/MultiPL-E) waste most of their rollout
budget on prompts the base team always gets right or always gets wrong: those
groups have zero reward variance, hence zero advantage, hence no gradient.
``scripts/prescreen_pool.py`` filters them out ahead of time and its output is
loaded here.

The cursor and RNG state are serialized so a resumed run draws exactly the same
prompt sequence it would have drawn uninterrupted.
"""
from __future__ import annotations

import base64
import json
import pickle
import random
from pathlib import Path
from typing import Any, Sequence

from .tasks.base import Problem


class PromptPool:
    def __init__(
        self,
        problems: Sequence[Problem],
        *,
        seed: int = 0,
        allowlist: set[str] | None = None,
    ) -> None:
        pool = [p for p in problems if allowlist is None or p.problem_id in allowlist]
        if not pool:
            raise ValueError("prompt pool is empty after filtering")
        self.problems = list(pool)
        self.rng = random.Random(seed)
        self.order = list(range(len(self.problems)))
        self.rng.shuffle(self.order)
        self.cursor = 0
        self.epoch = 0

    def next_batch(self, size: int) -> list[Problem]:
        out: list[Problem] = []
        while len(out) < size:
            if self.cursor >= len(self.order):
                self.rng.shuffle(self.order)
                self.cursor = 0
                self.epoch += 1
            out.append(self.problems[self.order[self.cursor]])
            self.cursor += 1
        return out

    # -- resume -------------------------------------------------------------
    def state(self) -> dict[str, Any]:
        return {
            "cursor": self.cursor,
            "epoch": self.epoch,
            "order": self.order,
            "rng": base64.b64encode(pickle.dumps(self.rng.getstate())).decode("ascii"),
        }

    def load_state(self, payload: dict[str, Any]) -> None:
        self.cursor = int(payload["cursor"])
        self.epoch = int(payload["epoch"])
        self.order = list(payload["order"])
        self.rng.setstate(pickle.loads(base64.b64decode(payload["rng"])))


def load_allowlist(path: Path | None) -> set[str] | None:
    """Read a prescreen result: one JSON object per line with ``problem_id``."""
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    keep: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("keep", True):
                keep.add(str(row["problem_id"]))
    return keep or None

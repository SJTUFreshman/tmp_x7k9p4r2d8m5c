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


def expected_prescreen_count(sample: int, source_count: int) -> int:
    """Return how many rows a prescreen run should emit.

    A zero sample means the whole source split, matching the CLI semantics.
    """
    if sample < 0:
        raise ValueError("prescreen sample must be >= 0")
    if source_count < 0:
        raise ValueError("source problem count must be >= 0")
    return source_count if sample == 0 else min(sample, source_count)


def load_allowlist(
    path: Path | None,
    *,
    expected_group_size: int | None = None,
    expected_count: int | None = None,
    expected_keep_band: tuple[int, int] | None = None,
    source_ids: set[str] | None = None,
) -> set[str] | None:
    """Read a prescreen result and optionally enforce complete coverage."""
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    keep: set[str] = set()
    seen: set[str] = set()
    row_count = 0
    if expected_count is not None and expected_count < 0:
        raise ValueError("expected prescreen row count must be >= 0")
    if expected_keep_band is not None:
        if expected_group_size is None:
            raise ValueError(
                "expected_group_size is required when validating keep_band"
            )
        low, high = expected_keep_band
        if low < 0 or high < low or high > expected_group_size:
            raise ValueError(
                "expected keep_band must satisfy 0 <= low <= high <= group size"
            )
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if expected_group_size is not None:
                actual_group_size = row.get("group_size")
                if actual_group_size is None:
                    raise ValueError(
                        f"prescreen row is missing group_size: {path}"
                    )
                if int(actual_group_size) != expected_group_size:
                    raise ValueError(
                        f"prescreen group_size {actual_group_size} does not match "
                        f"configured K={expected_group_size}: {path}"
                    )
            try:
                problem_id = str(row["problem_id"])
            except (KeyError, TypeError) as exc:
                raise ValueError(
                    f"prescreen row is missing problem_id: {path}"
                ) from exc
            if problem_id in seen:
                raise ValueError(
                    f"prescreen contains duplicate problem_id {problem_id!r}: {path}"
                )
            if source_ids is not None and problem_id not in source_ids:
                raise ValueError(
                    f"prescreen problem_id {problem_id!r} is not in the train split: {path}"
                )
            if expected_keep_band is not None:
                if not isinstance(row.get("keep"), bool):
                    raise ValueError(
                        f"prescreen row keep must be boolean: {path}"
                    )
                try:
                    passes = int(row["passes"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"prescreen row is missing a valid passes value: {path}"
                    ) from exc
                low, high = expected_keep_band
                if not 0 <= passes <= expected_group_size:
                    raise ValueError(
                        f"prescreen passes {passes} is outside 0..{expected_group_size}: {path}"
                    )
                expected_keep = low <= passes <= high
                if row["keep"] != expected_keep:
                    raise ValueError(
                        f"prescreen keep does not match passes={passes} and "
                        f"keep_band={expected_keep_band}: {path}"
                    )
            seen.add(problem_id)
            row_count += 1
            if row.get("keep", True):
                keep.add(problem_id)
    if expected_count is not None and row_count != expected_count:
        raise ValueError(
            f"prescreen has {row_count} rows but expected {expected_count}: {path}"
        )
    return keep

"""Debate trajectory: one record per (turn, agent).

This is the structural difference from the MAGRPO baseline. There, one joint
rollout earns one team reward. Here every ``(turn, agent)`` cell carries its own
verifier score, its own ground-truth correctness, and its own shaped reward, so
the natural container is a table indexed by ``(turn, agent)`` rather than a flat
list of steps.

Two separate signals are stored per cell, matching the official code
(``ppov2_trainer_multi_different_model.py:1305-1310``):

* ``correctness`` -- ground-truth label in {0, 1}. Feeds ``bonus_rule``.
* ``score``       -- the verifier's output, or a copy of ``correctness`` when
  ``reward_feedback=False`` (our setting). Feeds ``score_rule``. This is also
  where ``penalty_reward_value = -10`` is written, *before* ``score_rule`` runs.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

SCHEMA_VERSION = 2

# utils_multi_unified.py:1618 -- "this turn was never approached" marker.
SENTINEL = -1.0


@dataclass
class TurnRecord:
    """One model call: what it saw, what it said, and how it was scored."""

    turn: int
    agent: str
    prompt_messages: list[dict[str, str]]
    response: str
    parsed: dict[str, Any] | None = None
    protocol_error: str | None = None
    answer: str | None = None

    # Ground truth in {0,1} -> bonus_rule.
    correctness: float | None = None
    # Verifier score (== correctness when reward_feedback=False) -> score_rule.
    # Overwritten by penalty_reward_value when the penalty gate fails.
    score: float | None = None
    penalized: bool = False
    penalty_cause: str | None = None
    # Final shaped reward: score_rule(...) + bonus_rule(...).
    reward: float | None = None
    score_detail: dict[str, Any] = field(default_factory=dict)
    finish_reason: str | None = None
    prompt_token_ids: list[int] | None = None
    response_token_ids: list[int] | None = None
    response_logprobs: list[float] | None = None
    model_name: str | None = None
    adapter_version: str | None = None
    generation_config: dict[str, Any] = field(default_factory=dict)
    guided_decoding: bool = False
    stop_reason: str | int | None = None
    generation_attempts: int = 1

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TurnRecord":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})


@dataclass
class DebateTrajectory:
    """One question's full debate: ``round_num x agent_num`` records."""

    problem_id: str
    round_num: int
    agents: tuple[str, ...]
    records: dict[tuple[int, str], TurnRecord] = field(default_factory=dict)
    # -1 == never finished. Official keeps this at -1 for every question because
    # the consensus thresholds are 1.1; we lock that in (see config validation).
    finished_turn: int = -1
    final_answer: str | None = None
    error: str | None = None

    def put(self, record: TurnRecord) -> None:
        self.records[(record.turn, record.agent)] = record

    def get(self, turn: int, agent: str) -> TurnRecord | None:
        return self.records.get((turn, agent))

    def turn_records(self, turn: int) -> list[TurnRecord]:
        return [r for (t, _), r in sorted(self.records.items()) if t == turn]

    def agent_records(self, agent: str) -> list[TurnRecord]:
        return [
            self.records[(t, agent)]
            for t in range(self.round_num)
            if (t, agent) in self.records
        ]

    @property
    def complete(self) -> bool:
        return len(self.records) == self.round_num * len(self.agents)

    def correctness_table(self) -> dict[tuple[int, int], list[float]]:
        """``(turn, agent_index) -> [value]`` in the shape the reward rules want."""
        return {
            (t, i): [float(self.records[(t, a)].correctness or 0.0)]
            for t in range(self.round_num)
            for i, a in enumerate(self.agents)
            if (t, a) in self.records
        }

    def score_table(self) -> dict[tuple[int, int], list[float]]:
        return {
            (t, i): [float(self.records[(t, a)].score or 0.0)]
            for t in range(self.round_num)
            for i, a in enumerate(self.agents)
            if (t, a) in self.records
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "problem_id": self.problem_id,
            "round_num": self.round_num,
            "agents": list(self.agents),
            "records": [r.to_dict() for r in self.records.values()],
            "finished_turn": self.finished_turn,
            "final_answer": self.final_answer,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DebateTrajectory":
        traj = cls(
            problem_id=str(payload["problem_id"]),
            round_num=int(payload["round_num"]),
            agents=tuple(payload["agents"]),
            finished_turn=int(payload.get("finished_turn", -1)),
            final_answer=payload.get("final_answer"),
            error=payload.get("error"),
        )
        for item in payload.get("records", []):
            traj.put(TurnRecord.from_dict(item))
        return traj


def write_trajectories(path: Path, trajectories: Iterable[DebateTrajectory]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with gzip.open(tmp, "wt", encoding="utf-8") as handle:
        for traj in trajectories:
            handle.write(json.dumps(traj.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    tmp.replace(path)
    return count


def read_trajectories(path: Path) -> Iterator[DebateTrajectory]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield DebateTrajectory.from_dict(json.loads(line))

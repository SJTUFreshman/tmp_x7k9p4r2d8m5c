"""Joint-rollout data structures.

MAGRPO's optimization unit is the *joint* response of the whole agent group, so
the unit stored here is a :class:`JointTrajectory` -- one episode produced by all
agents together -- rather than a per-agent record.

Every :class:`AgentTurn` keeps ``prompt_messages``, the conversation snapshot
taken *before* the model was called.  That snapshot is what the trainer encodes,
so it must be captured at rollout time; reconstructing it afterwards from the
protocol is lossy once a chat template has been applied.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

SCHEMA_VERSION = 1


@dataclass
class AgentTurn:
    """One model call: what it saw, what it emitted, and its share of the credit."""

    agent: str
    turn: int
    prompt_messages: list[dict[str, str]]
    response: str
    parsed: dict[str, Any] | None = None
    protocol_error: str | None = None
    # Filled in by advantage.assign_advantages once the group's rewards are known.
    advantage: float | None = None
    # Per-turn shaping reward; stays 0.0 unless reward_shaping is enabled.
    step_reward: float = 0.0
    finish_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "turn": self.turn,
            "prompt_messages": self.prompt_messages,
            "response": self.response,
            "parsed": self.parsed,
            "protocol_error": self.protocol_error,
            "advantage": self.advantage,
            "step_reward": self.step_reward,
            "finish_reason": self.finish_reason,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AgentTurn":
        return cls(
            agent=str(payload["agent"]),
            turn=int(payload["turn"]),
            prompt_messages=list(payload["prompt_messages"]),
            response=str(payload["response"]),
            parsed=payload.get("parsed"),
            protocol_error=payload.get("protocol_error"),
            advantage=payload.get("advantage"),
            step_reward=float(payload.get("step_reward", 0.0)),
            finish_reason=payload.get("finish_reason"),
        )


@dataclass
class JointTrajectory:
    """One joint rollout: the episode that receives a single team reward."""

    problem_id: str
    group_id: str
    rollout_idx: int
    joint_mode: str
    turns: list[AgentTurn] = field(default_factory=list)
    final_answer: str | None = None
    team_reward: float | None = None
    reward_detail: dict[str, Any] = field(default_factory=dict)
    terminated_by: str = "unknown"
    error: str | None = None

    def turns_for(self, agent: str) -> list[AgentTurn]:
        return [t for t in self.turns if t.agent == agent]

    @property
    def n_turns(self) -> int:
        return len(self.turns)

    @property
    def protocol_error_rate(self) -> float:
        if not self.turns:
            return 0.0
        return sum(1 for t in self.turns if t.protocol_error) / len(self.turns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "problem_id": self.problem_id,
            "group_id": self.group_id,
            "rollout_idx": self.rollout_idx,
            "joint_mode": self.joint_mode,
            "turns": [t.to_dict() for t in self.turns],
            "final_answer": self.final_answer,
            "team_reward": self.team_reward,
            "reward_detail": self.reward_detail,
            "terminated_by": self.terminated_by,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "JointTrajectory":
        return cls(
            problem_id=str(payload["problem_id"]),
            group_id=str(payload["group_id"]),
            rollout_idx=int(payload["rollout_idx"]),
            joint_mode=str(payload.get("joint_mode", "synchronous")),
            turns=[AgentTurn.from_dict(t) for t in payload.get("turns", [])],
            final_answer=payload.get("final_answer"),
            team_reward=payload.get("team_reward"),
            reward_detail=payload.get("reward_detail") or {},
            terminated_by=str(payload.get("terminated_by", "unknown")),
            error=payload.get("error"),
        )


def write_trajectories(path: Path, trajectories: Iterable[JointTrajectory]) -> int:
    """Write gzipped JSONL atomically; returns the row count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with gzip.open(tmp, "wt", encoding="utf-8") as handle:
        for traj in trajectories:
            handle.write(json.dumps(traj.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    tmp.replace(path)
    return count


def read_trajectories(path: Path) -> Iterator[JointTrajectory]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield JointTrajectory.from_dict(json.loads(line))

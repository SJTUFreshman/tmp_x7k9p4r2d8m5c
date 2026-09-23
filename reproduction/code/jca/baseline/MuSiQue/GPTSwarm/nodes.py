"""GPTSwarm nodes: 4 kinds of Node backed by a single 8B executor.

Nodes:
    IONode          — fast, direct answer extraction (short reasoning)
    CoTNode         — chain-of-thought reasoning (longer reasoning)
    DebateNode      — critique N predecessor answers, produce corrected answer
    AggregatorNode  — final synthesis over all predecessor answers

Each Node's `run(problem, predecessor_outputs, caller)` method returns a
`NodeOutput` dict with {"reasoning": str, "answer": str}. Nodes are
stateless; the DAG in `swarm.py` handles scheduling.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from jca.src.inference import response_attempts


LLMCaller = Callable[[List[Dict[str, str]]], str]


PROMPT_DIR = Path(__file__).parent / "prompts"
PROMPT_IO_PATH = PROMPT_DIR / "node_io.md"
PROMPT_COT_PATH = PROMPT_DIR / "node_cot.md"
PROMPT_DEBATE_PATH = PROMPT_DIR / "node_debate.md"
PROMPT_AGGREGATOR_PATH = PROMPT_DIR / "node_aggregator.md"


def _load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def _load_json_object(raw_output: str) -> Optional[Any]:
    if not isinstance(raw_output, str):
        return None
    text = re.sub(
        r"<think\b[^>]*>.*?</think>",
        "",
        raw_output,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start >= 0:
            try:
                decoder = json.JSONDecoder()
                obj, _ = decoder.raw_decode(text[start:])
                return obj
            except json.JSONDecodeError:
                pass
    return None


def _parse_node_json(raw: str) -> Optional[Dict[str, str]]:
    payload = _load_json_object(raw)
    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        payload = payload[0]
    if isinstance(payload, dict):
        reasoning = payload.get("reasoning")
        answer = payload.get("answer")
        if isinstance(reasoning, str) and isinstance(answer, str) and answer.strip():
            return {"reasoning": reasoning.strip(), "answer": answer.strip()}
    rm = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, re.DOTALL)
    am = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, re.DOTALL)
    if rm and am and am.group(1).strip():
        return {"reasoning": rm.group(1).strip(), "answer": am.group(1).strip()}
    return None


def _retry_instruction() -> str:
    return (
        "Your previous output could not be parsed. Return ONLY a valid "
        'JSON object of the form {"reasoning": "...", "answer": "..."} '
        "with a non-empty answer. No prose, no code fences."
    )


# ---------------------------------------------------------------------------
# Node output
# ---------------------------------------------------------------------------


@dataclass
class NodeOutput:
    node_name: str
    node_type: str
    reasoning: str
    answer: str
    raw_output: str
    parse_ok: bool
    retried: bool = False
    n_predecessors: int = 0
    raw_outputs: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Base Node
# ---------------------------------------------------------------------------


@dataclass
class Node:
    name: str
    node_type: str  # "IO" | "CoT" | "Debate" | "Aggregator"

    def _system_prompt(self) -> str:
        raise NotImplementedError

    def _uses_predecessors(self) -> bool:
        """Whether this node consumes upstream node outputs in its prompt."""
        return False

    def _format_predecessors(self, preds: List[NodeOutput]) -> str:
        if not preds:
            return ""
        blocks = []
        for i, p in enumerate(preds, start=1):
            blocks.append(
                f"## Predecessor {i}\n"
                f"Reasoning: {p.reasoning}\n"
                f"Answer: {p.answer}"
            )
        return "\n\n".join(blocks)

    def _build_user_content(self, problem_text: str, preds: List[NodeOutput]) -> str:
        if not self._uses_predecessors() or not preds:
            return problem_text
        pred_block = self._format_predecessors(preds)
        return (
            f"{problem_text}\n\n"
            f"# Predecessor Node Outputs (N={len(preds)})\n"
            f"{pred_block}\n\n"
            f"Produce your JSON output now."
        )

    async def _acall(self, caller: LLMCaller, messages: List[Dict[str, str]]) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, caller, messages)

    async def run(
        self,
        problem: Any,
        predecessor_outputs: List[NodeOutput],
        caller: LLMCaller,
    ) -> NodeOutput:
        problem_text = getattr(problem, "rendered_text", None) or str(problem)
        system = self._system_prompt()
        user = self._build_user_content(problem_text, predecessor_outputs)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        raw = await self._acall(caller, messages)
        raw_outputs = response_attempts(raw)
        parsed = _parse_node_json(raw)
        retried = False

        if parsed is None:
            retry_messages = list(messages) + [
                {"role": "user", "content": _retry_instruction()},
            ]
            raw = await self._acall(caller, retry_messages)
            raw_outputs.extend(response_attempts(raw))
            parsed = _parse_node_json(raw)
            retried = True

        if parsed is None:
            return NodeOutput(
                node_name=self.name,
                node_type=self.node_type,
                reasoning="",
                answer="",
                raw_output=raw,
                parse_ok=False,
                retried=True,
                n_predecessors=len(predecessor_outputs),
                raw_outputs=raw_outputs,
            )

        return NodeOutput(
            node_name=self.name,
            node_type=self.node_type,
            reasoning=parsed["reasoning"],
            answer=parsed["answer"],
            raw_output=raw,
            parse_ok=True,
            retried=retried,
            n_predecessors=len(predecessor_outputs),
            raw_outputs=raw_outputs,
        )


# ---------------------------------------------------------------------------
# Concrete Node types
# ---------------------------------------------------------------------------


@dataclass
class IONode(Node):
    node_type: str = "IO"

    def _system_prompt(self) -> str:
        return _load_prompt(PROMPT_IO_PATH)

    def _uses_predecessors(self) -> bool:
        # IO nodes only touch the paragraphs (they're the "input layer").
        return False


@dataclass
class CoTNode(Node):
    node_type: str = "CoT"

    def _system_prompt(self) -> str:
        return _load_prompt(PROMPT_COT_PATH)

    def _uses_predecessors(self) -> bool:
        return False


@dataclass
class DebateNode(Node):
    node_type: str = "Debate"

    def _system_prompt(self) -> str:
        return _load_prompt(PROMPT_DEBATE_PATH)

    def _uses_predecessors(self) -> bool:
        return True


@dataclass
class AggregatorNode(Node):
    node_type: str = "Aggregator"

    def _system_prompt(self) -> str:
        return _load_prompt(PROMPT_AGGREGATOR_PATH)

    def _uses_predecessors(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


NODE_CLASSES = {
    "IO": IONode,
    "CoT": CoTNode,
    "Debate": DebateNode,
    "Aggregator": AggregatorNode,
}


def make_node(node_type: str, name: str) -> Node:
    cls = NODE_CLASSES.get(node_type)
    if cls is None:
        raise ValueError(f"Unknown node type: {node_type!r}. "
                         f"Allowed: {sorted(NODE_CLASSES)}")
    return cls(name=name)

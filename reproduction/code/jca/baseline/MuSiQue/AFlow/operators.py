"""AFlow operators (async-style) backed by an OpenAI-compatible caller.

Three whitelisted operators — the only surface a searched workflow is
allowed to touch:

    Ops.solve(problem, instruction="") -> {"reasoning": str, "answer": str}
    Ops.ensemble(candidates, problem)  -> {"chosen_index": int, "reason": str}
    Ops.answer_generate(text)          -> {"answer": str}

`solve` calls are distributed round-robin across the configured solver pool.
`ensemble` and `answer_generate` use the designated judge caller (A3 in the
fair heterogeneous setting).
Operators are `async` so a workflow can trivially fan out with
asyncio.gather(). Under the hood we dispatch each blocking urllib call
into a thread pool.
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
PROMPT_SOLVE_PATH = PROMPT_DIR / "op_solve.md"
PROMPT_ENSEMBLE_PATH = PROMPT_DIR / "op_ensemble.md"
PROMPT_ANSWER_PATH = PROMPT_DIR / "op_answer.md"


def _load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


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
        for opener in ("{", "["):
            start = text.find(opener)
            if start >= 0:
                try:
                    decoder = json.JSONDecoder()
                    obj, _ = decoder.raw_decode(text[start:])
                    return obj
                except json.JSONDecodeError:
                    continue
    return None


@dataclass
class OpCallStat:
    op: str
    ok: bool
    caller_id: str = ""
    retried: bool = False
    raw_outputs: List[str] = field(default_factory=list)


class Ops:
    """Bundle of operators used inside a workflow. One `Ops` per problem call."""

    def __init__(
        self,
        solve_callers: List[LLMCaller] | LLMCaller,
        judge_caller: Optional[LLMCaller] = None,
        solve_caller_ids: Optional[List[str]] = None,
        judge_caller_id: str = "",
    ):
        if callable(solve_callers):
            solve_callers = [solve_callers]
        if not solve_callers:
            raise ValueError("Ops requires at least one solve caller")
        self.solve_callers = list(solve_callers)
        self.judge_caller = judge_caller or self.solve_callers[-1]
        self.solve_caller_ids = solve_caller_ids or [
            f"solve_{index}" for index in range(len(self.solve_callers))
        ]
        if len(self.solve_caller_ids) != len(self.solve_callers):
            raise ValueError("solve_caller_ids must match solve_callers")
        self.judge_caller_id = judge_caller_id or self.solve_caller_ids[-1]
        self._solve_counter = 0
        self.calls: List[OpCallStat] = []
        self._prompt_solve = _load_prompt(PROMPT_SOLVE_PATH)
        self._prompt_ensemble = _load_prompt(PROMPT_ENSEMBLE_PATH)
        self._prompt_answer = _load_prompt(PROMPT_ANSWER_PATH)

    # -------------------- internal --------------------

    async def _acall(self, caller: LLMCaller, messages: List[Dict[str, str]]) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, caller, messages)

    def _retry_msg(self, schema_hint: str) -> str:
        return (
            "Your previous output could not be parsed. Return ONLY a valid "
            f"JSON object of the form {schema_hint}. No prose, no code fences."
        )

    # -------------------- solve --------------------

    async def solve(self, problem: Any, instruction: str = "") -> Dict[str, str]:
        solve_index = self._solve_counter % len(self.solve_callers)
        self._solve_counter += 1
        caller = self.solve_callers[solve_index]
        caller_id = self.solve_caller_ids[solve_index]
        problem_text = getattr(problem, "rendered_text", None) or str(problem)
        system_prompt = self._prompt_solve.replace("{INSTRUCTION}", instruction or "(none)")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem_text},
        ]
        raw = await self._acall(caller, messages)
        raw_outputs = response_attempts(raw)
        parsed = self._parse_solve(raw)
        retried = False
        if parsed is None:
            retry_messages = list(messages) + [
                {"role": "user", "content": self._retry_msg(
                    '{"reasoning": "...", "answer": "..."}'
                )},
            ]
            raw = await self._acall(caller, retry_messages)
            raw_outputs.extend(response_attempts(raw))
            parsed = self._parse_solve(raw)
            retried = True
        if parsed is None:
            self.calls.append(OpCallStat(
                "solve", ok=False, caller_id=caller_id, retried=True, raw_outputs=raw_outputs))
            return {"reasoning": "", "answer": ""}
        self.calls.append(OpCallStat(
            "solve", ok=True, caller_id=caller_id, retried=retried, raw_outputs=raw_outputs))
        return parsed

    def _parse_solve(self, raw: str) -> Optional[Dict[str, str]]:
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

    # -------------------- ensemble --------------------

    async def ensemble(self, candidates: List[Dict[str, str]], problem: Any) -> Dict[str, Any]:
        if not isinstance(candidates, list) or len(candidates) < 2:
            self.calls.append(OpCallStat("ensemble", ok=False, caller_id=self.judge_caller_id, raw_outputs=[]))
            return {"chosen_index": 0, "reason": "invalid candidates"}
        problem_text = getattr(problem, "rendered_text", None) or str(problem)
        cand_blocks = []
        for i, c in enumerate(candidates):
            reasoning = c.get("reasoning", "") if isinstance(c, dict) else ""
            answer = c.get("answer", "") if isinstance(c, dict) else str(c)
            cand_blocks.append(
                f"## Candidate {i}\nReasoning: {reasoning}\nAnswer: {answer}"
            )
        user = (
            f"{problem_text}\n\n"
            f"# {len(candidates)} Candidate Solutions\n"
            + "\n\n".join(cand_blocks)
            + f"\n\nPick the best candidate index in [0, {len(candidates)-1}]."
        )
        messages = [
            {"role": "system", "content": self._prompt_ensemble},
            {"role": "user", "content": user},
        ]
        raw = await self._acall(self.judge_caller, messages)
        raw_outputs = response_attempts(raw)
        parsed = self._parse_ensemble(raw, n=len(candidates))
        retried = False
        if parsed is None:
            retry_messages = list(messages) + [
                {"role": "user", "content": self._retry_msg(
                    '{"chosen_index": <int>, "reason": "..."}'
                )},
            ]
            raw = await self._acall(self.judge_caller, retry_messages)
            raw_outputs.extend(response_attempts(raw))
            parsed = self._parse_ensemble(raw, n=len(candidates))
            retried = True
        if parsed is None:
            self.calls.append(OpCallStat(
                "ensemble", ok=False, caller_id=self.judge_caller_id, retried=True, raw_outputs=raw_outputs))
            # Fallback: majority-vote by normalized answer, break ties by first.
            from jca.src.grader import normalize_answer
            counts: Dict[str, int] = {}
            for i, c in enumerate(candidates):
                key = normalize_answer(c.get("answer", "")) if isinstance(c, dict) else ""
                if key:
                    counts[key] = counts.get(key, 0) + 1
            if counts:
                winning_key = max(counts, key=lambda k: counts[k])
                for i, c in enumerate(candidates):
                    if isinstance(c, dict) and normalize_answer(c.get("answer", "")) == winning_key:
                        return {"chosen_index": i, "reason": "fallback majority"}
            return {"chosen_index": 0, "reason": "fallback first"}
        self.calls.append(OpCallStat(
            "ensemble", ok=True, caller_id=self.judge_caller_id, retried=retried, raw_outputs=raw_outputs))
        return parsed

    def _parse_ensemble(self, raw: str, n: int) -> Optional[Dict[str, Any]]:
        payload = _load_json_object(raw)
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            payload = payload[0]
        if isinstance(payload, dict):
            idx = payload.get("chosen_index")
            reason = payload.get("reason", "")
            if isinstance(idx, (int, float)):
                i = int(idx)
                if 0 <= i < n:
                    return {"chosen_index": i, "reason": str(reason)}
        m = re.search(r'"chosen_index"\s*:\s*(-?\d+)', raw)
        if m:
            i = int(m.group(1))
            if 0 <= i < n:
                return {"chosen_index": i, "reason": ""}
        return None

    # -------------------- answer_generate --------------------

    async def answer_generate(self, text: Any) -> Dict[str, str]:
        input_text = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
        messages = [
            {"role": "system", "content": self._prompt_answer},
            {"role": "user", "content": input_text},
        ]
        raw = await self._acall(self.judge_caller, messages)
        raw_outputs = response_attempts(raw)
        parsed = self._parse_answer(raw)
        retried = False
        if parsed is None:
            retry_messages = list(messages) + [
                {"role": "user", "content": self._retry_msg('{"answer": "..."}')},
            ]
            raw = await self._acall(self.judge_caller, retry_messages)
            raw_outputs.extend(response_attempts(raw))
            parsed = self._parse_answer(raw)
            retried = True
        if parsed is None:
            self.calls.append(OpCallStat(
                "answer_generate", ok=False, caller_id=self.judge_caller_id, retried=True,
                raw_outputs=raw_outputs))
            # Fallback: try to strip \boxed{...} or take last non-empty line.
            m = re.search(r"\\boxed\{([^{}]+)\}", input_text)
            if m:
                return {"answer": m.group(1).strip()}
            for line in reversed(input_text.splitlines()):
                if line.strip():
                    return {"answer": line.strip()}
            return {"answer": ""}
        self.calls.append(OpCallStat(
            "answer_generate", ok=True, caller_id=self.judge_caller_id, retried=retried,
            raw_outputs=raw_outputs))
        return parsed

    def _parse_answer(self, raw: str) -> Optional[Dict[str, str]]:
        payload = _load_json_object(raw)
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            payload = payload[0]
        if isinstance(payload, dict):
            answer = payload.get("answer")
            if isinstance(answer, str) and answer.strip():
                return {"answer": answer.strip()}
        m = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, re.DOTALL)
        if m and m.group(1).strip():
            return {"answer": m.group(1).strip()}
        return None

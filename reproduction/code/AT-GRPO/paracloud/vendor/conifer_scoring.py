#!/usr/bin/env python3
"""Deterministic constraint checks and optional LLM process judging."""
from __future__ import annotations

import json
import io
import os
import re
from threading import Condition, Lock
import time
import urllib.error
import urllib.request
from collections import Counter
from typing import Any, Iterable


WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*")
SENTENCE_RE = re.compile(r"[^.!?\n]*[.!?](?:\s|$)")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "into", "is", "it", "of", "on", "or", "that", "the", "this", "to", "use",
    "using", "with", "within", "you", "your", "must", "should", "answer", "provide",
}

_JUDGE_HTTP_POOL = None
_JUDGE_HTTP_POOL_LOCK = Lock()


def _judge_keepalive_enabled() -> bool:
    return os.environ.get("JCA_HTTP_KEEPALIVE", "0").lower() in {
        "1", "true", "yes", "on"
    }


def _read_judge_request(request: urllib.request.Request, timeout: float) -> bytes:
    if not _judge_keepalive_enabled():
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    try:
        import urllib3
    except ImportError:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    global _JUDGE_HTTP_POOL
    with _JUDGE_HTTP_POOL_LOCK:
        if _JUDGE_HTTP_POOL is None:
            try:
                pool_maxsize = int(os.environ.get("JCA_HTTP_POOL_MAXSIZE", "128"))
            except ValueError:
                pool_maxsize = 64
            if pool_maxsize <= 0:
                pool_maxsize = 64
            _JUDGE_HTTP_POOL = urllib3.PoolManager(
                num_pools=16,
                maxsize=pool_maxsize,
                block=True,
                retries=False,
            )
        pool = _JUDGE_HTTP_POOL
    headers = {name: value for name, value in request.header_items()}
    response = None
    try:
        response = pool.request(
            "POST",
            request.full_url,
            body=request.data,
            headers=headers,
            timeout=urllib3.Timeout(connect=timeout, read=timeout),
            preload_content=True,
        )
        body = bytes(response.data or b"")
        status = int(response.status)
    except Exception as exc:
        raise urllib.error.URLError(exc) from exc
    finally:
        if response is not None:
            response.release_conn()
    if status >= 400:
        raise urllib.error.HTTPError(
            request.full_url,
            status,
            "HTTP error",
            headers,
            io.BytesIO(body),
        )
    return body


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def sentence_count(text: str) -> int:
    matches = SENTENCE_RE.findall(text or "")
    if matches:
        return len(matches)
    return 1 if str(text or "").strip() else 0


def item_count(text: str, kind: str | None = None) -> int:
    lines = str(text or "").splitlines()
    if kind == "bullets":
        return sum(bool(re.match(r"^\s*(?:[-*•]|\u2022)\s+", line)) for line in lines)
    if kind == "ordered_list":
        return sum(bool(re.match(r"^\s*\d+[.)]\s+", line)) for line in lines)
    return sum(bool(re.match(r"^\s*(?:[-*•]|\u2022|\d+[.)])\s+", line)) for line in lines)


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in WORD_RE.findall(text or "") if token.casefold() not in STOPWORDS}


def lexical_f1(candidate: str, reference: str | None) -> float | None:
    if not reference:
        return None
    left = Counter(_tokens(candidate))
    right = Counter(_tokens(reference))
    overlap = sum((left & right).values())
    if not left or not right or overlap == 0:
        return 0.0
    precision = overlap / sum(left.values())
    recall = overlap / sum(right.values())
    return round(2 * precision * recall / (precision + recall), 4)


def _contains_format(text: str, name: str) -> bool:
    lines = str(text or "").splitlines()
    if name == "bullets":
        return item_count(text, "bullets") > 0
    if name == "ordered_list":
        return item_count(text, "ordered_list") > 0
    if name == "list":
        return item_count(text) > 0
    if name == "table":
        return sum("|" in line for line in lines) >= 2 and any("---" in line for line in lines)
    if name == "code":
        return "```" in text or bool(re.search(r"\b(?:def|function|const|SELECT)\b", text or ""))
    if name == "paragraph":
        return bool(str(text or "").strip())
    return True


def _requirement_coverage(answer: str, requirements: Iterable[str]) -> float:
    reqs = [str(req).strip() for req in requirements if str(req).strip()]
    if not reqs:
        return 1.0
    answer_tokens = _tokens(answer)
    scores = []
    for req in reqs:
        req_tokens = _tokens(req)
        if not req_tokens:
            scores.append(1.0)
            continue
        scores.append(len(req_tokens & answer_tokens) / len(req_tokens))
    return sum(scores) / len(scores)


def check_constraints(row: dict[str, Any], answer: str) -> dict[str, Any]:
    """Return auditable checks without making semantic claims.

    A check is ``passed`` only for explicit, mechanically testable constraints;
    semantic requirements are represented as a soft lexical coverage signal and
    are left to the LLM judge.
    """
    answer = str(answer or "").strip()
    constraints = row.get("constraints") or {}
    formats = [str(item) for item in constraints.get("formats") or []]
    limits = constraints.get("limits") or {}
    required_terms = [str(item) for item in constraints.get("required_terms") or []]
    requirements = constraints.get("numbered_requirements") or []
    checks: dict[str, Any] = {
        "answer_present": bool(answer),
        "word_count": word_count(answer),
        "sentence_count": sentence_count(answer),
        "format": {},
        "limits": {},
        "required_terms": {},
        "requirement_coverage": round(_requirement_coverage(answer, requirements), 4),
        "reference_lexical_f1": lexical_f1(answer, row.get("reference_answer")),
    }
    for name in formats:
        checks["format"][name] = _contains_format(answer, name)
    for name, limit in limits.items():
        try:
            n = int(limit)
        except (TypeError, ValueError):
            continue
        if name == "max_words":
            checks["limits"][name] = checks["word_count"] <= n
        elif name == "exact_sentences":
            checks["limits"][name] = checks["sentence_count"] == n
        elif name == "max_sentences":
            checks["limits"][name] = checks["sentence_count"] <= n
        elif name == "min_items":
            checks["limits"][name] = item_count(answer) >= n
        elif name == "max_items":
            checks["limits"][name] = item_count(answer) <= n
        elif name == "exact_items":
            checks["limits"][name] = item_count(answer) == n
    lower_answer = answer.casefold()
    for term in required_terms:
        checks["required_terms"][term] = term.casefold() in lower_answer

    boolean_checks = [checks["answer_present"], *checks["format"].values(), *checks["limits"].values(), *checks["required_terms"].values()]
    explicit_score = sum(boolean_checks) / len(boolean_checks) if boolean_checks else 0.0
    # Keep lexical coverage visible but do not let a heuristic parser dominate.
    hard_score = 0.75 * explicit_score + 0.25 * checks["requirement_coverage"]
    if not checks["answer_present"]:
        hard_score = 0.0
    checks["explicit_score"] = round(explicit_score, 4)
    checks["hard_score"] = round(max(0.0, min(1.0, hard_score)), 4)
    checks["all_explicit_passed"] = all(boolean_checks) if boolean_checks else False
    return checks


def build_judge_prompt(
    row: dict[str, Any],
    steps: list[dict[str, Any]],
    final_answer: str,
    final_checks: dict[str, Any],
) -> list[dict[str, str]]:
    compact_steps = []
    for step in steps:
        compact_steps.append({
            "turn": step.get("turn"),
            "agent_id": step.get("active_agent"),
            "action": step.get("action"),
            "reasoning": str(step.get("reasoning") or "")[:1200],
            "tentative_answer": str(step.get("tentative_answer") or "")[:4000],
            "handoff_note": step.get("handoff_note"),
        })
    system = """You are a strict evaluator of an open-ended multi-agent instruction-following trajectory.
Judge semantic usefulness and collaboration process, not stylistic preference.
The deterministic checks supplied by the caller are authoritative for explicit
format/length/literal constraints. Do not invent a unique answer when several
answers can satisfy the instruction.

Return ONLY one JSON object:
{
  "final_quality": 0.0,
  "turns": [
    {"turn": 0, "agent_id": "A1", "reasoning_score": -1.0,
     "action_score": -1.0, "content_score": -1.0,
     "comment": "specific, concise attribution"}
  ]
}
Scores are in [0,1] for final_quality and [-1,1] for each turn score.
Reasoning_score evaluates correctness and useful constraint analysis.
Content_score evaluates whether the draft improves answer quality.
Action_score evaluates whether handoff/stop was timely and useful; penalize
redundant loops, premature stopping, and regressions.
"""
    user = {
        "instruction": row.get("question") or row.get("seed_prompt"),
        "parsed_constraints": row.get("constraints") or {},
        "reference_answer_for_audit_only": row.get("reference_answer"),
        "final_answer": final_answer,
        "final_deterministic_checks": final_checks,
        "trajectory": compact_steps,
    }
    return [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]


class JudgeEndpointPool:
    def __init__(self, api_bases: str | Iterable[str]) -> None:
        if isinstance(api_bases, str):
            values = api_bases.split(",")
        else:
            values = list(api_bases)
        self.api_bases = tuple(str(value).strip().rstrip("/") for value in values if str(value).strip())
        if not self.api_bases:
            raise ValueError("judge API endpoint pool must not be empty")
        if len(set(self.api_bases)) != len(self.api_bases):
            raise ValueError("judge API endpoint pool contains duplicates")
        self._active = [0] * len(self.api_bases)
        self._failures = [0] * len(self.api_bases)
        self._cooldown_until = [0.0] * len(self.api_bases)
        self._next_index = 0
        self._condition = Condition(Lock())

    def _acquire(self, excluded: set[int] | None = None) -> int:
        excluded = excluded or set()
        with self._condition:
            now = time.monotonic()
            candidates = [
                index for index in range(len(self._active))
                if index not in excluded and self._cooldown_until[index] <= now
            ]
            if not candidates:
                candidates = [index for index in range(len(self._active)) if index not in excluded]
            if not candidates:
                candidates = list(range(len(self._active)))
            minimum = min(self._active[index] for index in candidates)
            for offset in range(len(self._active)):
                index = (self._next_index + offset) % len(self._active)
                if index in candidates and self._active[index] == minimum:
                    self._active[index] += 1
                    self._next_index = (index + 1) % len(self._active)
                    return index
        raise RuntimeError("judge endpoint selection failed")

    def _release(self, index: int) -> None:
        with self._condition:
            if not 0 <= index < len(self._active) or self._active[index] <= 0:
                raise RuntimeError(f"invalid judge endpoint release: {index}")
            self._active[index] -= 1
            self._condition.notify()

    def _mark_success(self, index: int) -> None:
        with self._condition:
            self._failures[index] = 0
            self._cooldown_until[index] = 0.0

    def _mark_failure(self, index: int) -> None:
        with self._condition:
            self._failures[index] += 1
            self._cooldown_until[index] = time.monotonic() + min(10.0, 0.25 * (2 ** min(self._failures[index] - 1, 5)))

    def call(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        try:
            retry_budget = max(0, int(os.environ.get("JCA_ENDPOINT_RETRIES", "2")))
        except ValueError:
            retry_budget = 2
        attempts = min(len(self.api_bases), retry_budget + 1)
        last_error: RuntimeError | None = None
        tried: set[int] = set()
        for _ in range(attempts):
            index = self._acquire(tried)
            tried.add(index)
            try:
                result = _call_judge_once(messages, api_base=self.api_bases[index], **kwargs)
                self._mark_success(index)
                return result
            except RuntimeError as exc:
                last_error = exc
                self._mark_failure(index)
            finally:
                self._release(index)
        if last_error is not None:
            raise last_error
        raise RuntimeError("judge endpoint pool call failed")


def _call_judge_once(
    messages: list[dict[str, str]],
    *,
    model: str,
    api_base: str | None = None,
    api_key: str | None = None,
    timeout: float = 300.0,
    max_tokens: int = 1536,
    length_retries: int = 1,
) -> dict[str, Any]:
    base = (api_base or os.environ.get("JCA_JUDGE_API_BASE") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    key = api_key or os.environ.get("JCA_JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"
    if max_tokens <= 0 or length_retries < 0:
        raise ValueError("judge token budget must be positive and retries non-negative")
    for attempt in range(length_retries + 1):
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "top_p": 0.95,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            f"{base}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            method="POST",
        )
        try:
            data = json.loads(_read_judge_request(request, timeout).decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"judge HTTP {exc.code}: {body[:500]}") from exc
        choice = (data.get("choices") or [{}])[0]
        if choice.get("finish_reason") == "length" and attempt < length_retries:
            max_tokens *= 2
            continue
        content = choice.get("message", {}).get("content", "")
        if isinstance(content, list):
            content = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        text = str(content).strip()
        start, end = text.find("{"), text.rfind("}") + 1
        if start < 0 or end <= start:
            raise ValueError("judge did not return a JSON object")
        parsed = json.loads(text[start:end])
        if not isinstance(parsed, dict):
            raise ValueError("judge JSON must be an object")
        return parsed
    raise RuntimeError("judge generation exhausted retries")


def call_judge(
    messages: list[dict[str, str]],
    *,
    model: str,
    api_base: str | None = None,
    api_key: str | None = None,
    timeout: float = 300.0,
    max_tokens: int = 1536,
    length_retries: int = 1,
    endpoint_pool: JudgeEndpointPool | None = None,
) -> dict[str, Any]:
    if endpoint_pool is not None:
        return endpoint_pool.call(
            messages, model=model, api_key=api_key, timeout=timeout,
            max_tokens=max_tokens, length_retries=length_retries,
        )
    bases = tuple(part.strip().rstrip("/") for part in (api_base or "").split(",") if part.strip())
    if len(bases) > 1:
        return JudgeEndpointPool(bases).call(
            messages, model=model, api_key=api_key, timeout=timeout,
            max_tokens=max_tokens, length_retries=length_retries,
        )
    return _call_judge_once(
        messages, model=model, api_base=(bases[0] if bases else api_base),
        api_key=api_key, timeout=timeout, max_tokens=max_tokens,
        length_retries=length_retries,
    )


def normalize_judge_result(value: dict[str, Any], n_steps: int) -> dict[str, Any]:
    try:
        final_quality = max(0.0, min(1.0, float(value.get("final_quality", 0.0))))
    except (TypeError, ValueError):
        final_quality = 0.0
    by_turn: dict[int, dict[str, Any]] = {}
    malformed_turns: list[int] = []
    for item in value.get("turns") or []:
        if not isinstance(item, dict):
            continue
        try:
            turn = int(item.get("turn"))
        except (TypeError, ValueError):
            continue
        if turn < 0 or turn >= n_steps:
            malformed_turns.append(turn)
            continue
        scores: dict[str, Any] = {}
        for name in ("reasoning_score", "action_score", "content_score"):
            try:
                scores[name] = round(max(-1.0, min(1.0, float(item.get(name, 0.0)))), 4)
            except (TypeError, ValueError):
                malformed_turns.append(turn)
                scores[name] = 0.0
        scores["comment"] = str(item.get("comment") or "")[:1000]
        by_turn[turn] = scores
    missing_turns = [turn for turn in range(n_steps) if turn not in by_turn]
    return {
        "final_quality": round(final_quality, 4),
        "turns": [{"turn": turn, **by_turn.get(turn, {"reasoning_score": 0.0, "action_score": 0.0, "content_score": 0.0, "comment": "missing judge score"})} for turn in range(n_steps)],
        "complete": not missing_turns and not malformed_turns,
        "missing_turns": missing_turns,
        "malformed_turns": sorted(set(malformed_turns)),
    }

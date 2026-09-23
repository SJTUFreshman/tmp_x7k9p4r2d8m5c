"""HTTP transport to the vLLM servers, plus the mock used by the CPU smoke test.

A slim re-implementation of ``src/inference.py:OpenAIChatLLMCaller`` rather than a
vendored copy: the baseline needs only chat completions with per-request seed and
temperature, and the upstream class carries a lot of thinking-fallback and
structured-output machinery this loop does not use.
"""
from __future__ import annotations

import hashlib
import json
import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable, Protocol, Sequence

from .protocol import AGENT_IDS, PROTOCOL_FIELDS


@dataclass
class GenerationOptions:
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 1024
    enable_thinking: bool = False


@dataclass
class Completion:
    text: str
    finish_reason: str | None = None


class Caller(Protocol):
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        seed: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> Completion: ...


class OpenAIChatCaller:
    """One vLLM endpoint. ``model_name`` is reassignable for versioned LoRA names."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        *,
        generation: GenerationOptions | None = None,
        timeout: float = 600.0,
        api_key: str = "EMPTY",
        retries: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.generation = generation or GenerationOptions()
        self.timeout = timeout
        self.api_key = api_key
        self.retries = retries

    def set_model_name(self, name: str) -> None:
        self.model_name = name

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        seed: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.generation.temperature,
            "top_p": self.generation.top_p,
            "max_tokens": self.generation.max_new_tokens,
            "chat_template_kwargs": {"enable_thinking": self.generation.enable_thinking},
        }
        if seed is not None:
            payload["seed"] = int(seed)
        if response_format is not None:
            payload["response_format"] = response_format

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                choice = (data.get("choices") or [{}])[0]
                content = choice.get("message", {}).get("content", "")
                if isinstance(content, list):
                    content = "".join(
                        str(part.get("text", ""))
                        for part in content
                        if isinstance(part, dict)
                    )
                return Completion(str(content), choice.get("finish_reason"))
            except Exception as exc:  # noqa: BLE001 - retried, then surfaced
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(4.0, 0.5 * (2**attempt)))
        raise RuntimeError(f"generation failed at {self.base_url}: {last_error}")


class EndpointPoolCaller:
    """Least-in-flight dispatch across replicas of one agent's model."""

    def __init__(self, callers: list[OpenAIChatCaller], *, retries: int = 2) -> None:
        if not callers:
            raise ValueError("endpoint pool must not be empty")
        if retries < 0:
            raise ValueError("retries must be >= 0")
        self._callers = callers
        self._active = [0] * len(callers)
        self._lock = threading.Lock()
        self._next = 0
        self._max_attempts = max(len(callers), retries + 1)

    def set_model_name(self, name: str) -> None:
        for caller in self._callers:
            caller.set_model_name(name)

    def _acquire(self, excluded: set[int] | None = None) -> int:
        excluded = excluded or set()
        with self._lock:
            eligible = [
                index for index in range(len(self._active)) if index not in excluded
            ]
            if not eligible:
                raise RuntimeError("no eligible replica endpoint")
            minimum = min(self._active[index] for index in eligible)
            for offset in range(len(self._active)):
                index = (self._next + offset) % len(self._active)
                if index in eligible and self._active[index] == minimum:
                    self._active[index] += 1
                    self._next = (index + 1) % len(self._active)
                    return index
            raise RuntimeError("endpoint selection failed")

    def generate(self, messages, *, seed=None, response_format=None) -> Completion:
        attempted_in_cycle: set[int] = set()
        errors: list[tuple[int, Exception]] = []
        for _ in range(self._max_attempts):
            if len(attempted_in_cycle) == len(self._callers):
                attempted_in_cycle.clear()
            index = self._acquire(attempted_in_cycle)
            attempted_in_cycle.add(index)
            try:
                return self._callers[index].generate(
                    messages, seed=seed, response_format=response_format
                )
            except Exception as exc:  # noqa: BLE001 - fail over, then surface all
                errors.append((index, exc))
            finally:
                with self._lock:
                    self._active[index] -= 1

        details = "; ".join(
            f"replica {index} ({getattr(self._callers[index], 'base_url', 'unknown')}): "
            f"{type(exc).__name__}: {exc}"
            for index, exc in errors
        )
        raise RuntimeError(
            f"generation failed across all {len(self._callers)} replicas after "
            f"{len(errors)} attempts: {details}"
        )


class ConcurrencyLimitedCaller:
    """Share one request semaphore across otherwise independent callers."""

    def __init__(self, caller: Caller, semaphore: threading.BoundedSemaphore) -> None:
        self._caller = caller
        self._semaphore = semaphore

    def set_model_name(self, name: str) -> None:
        setter = getattr(self._caller, "set_model_name", None)
        if setter is not None:
            setter(name)

    def generate(self, messages, *, seed=None, response_format=None) -> Completion:
        with self._semaphore:
            return self._caller.generate(
                messages, seed=seed, response_format=response_format
            )


def build_openai_caller(
    base_urls: str | Sequence[str],
    model_name: str,
    *,
    generation: GenerationOptions | None = None,
    timeout: float = 600.0,
    api_key: str = "EMPTY",
    retries: int = 2,
) -> OpenAIChatCaller | EndpointPoolCaller:
    """Build one caller or a least-in-flight pool for replicated endpoints."""
    urls = [base_urls] if isinstance(base_urls, str) else list(base_urls)
    callers = [
        OpenAIChatCaller(
            url,
            model_name,
            generation=generation,
            timeout=timeout,
            api_key=api_key,
            retries=retries if len(urls) == 1 else 0,
        )
        for url in urls
    ]
    if not callers:
        raise ValueError("at least one endpoint is required")
    return (
        callers[0]
        if len(callers) == 1
        else EndpointPoolCaller(callers, retries=retries)
    )


class MockCaller:
    """Deterministic protocol-valid responses for the CPU smoke test.

    ``accuracy`` is tunable on purpose: a mock that is always right (or always
    wrong) produces zero reward variance, so the run would only ever exercise the
    degenerate-group branch and the advantage path would go untested.
    """

    def __init__(
        self,
        agent: str,
        *,
        accuracy: float = 0.5,
        malformed_rate: float = 0.05,
        gold_by_problem: dict[str, str] | None = None,
        answer_field: str = "tentative_answer",
        seed: int = 0,
    ) -> None:
        self.agent = agent
        self.accuracy = accuracy
        self.malformed_rate = malformed_rate
        self.gold_by_problem = gold_by_problem or {}
        self.answer_field = answer_field
        self.seed = seed
        self.calls = 0

    @staticmethod
    def _hash_unit(*parts: Any) -> float:
        digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()
        return int(digest[:8], 16) / 0xFFFFFFFF

    def generate(self, messages, *, seed=None, response_format=None) -> Completion:
        self.calls += 1
        text = "\n".join(m.get("content", "") for m in messages)
        # Recover which problem this is so the mock can be "right" sometimes.
        gold = ""
        for pid, value in self.gold_by_problem.items():
            if pid in text:
                gold = value
                break
        # Keyed on the request, never on the call counter: rollouts within a
        # group run concurrently, so a counter would make output depend on
        # thread scheduling and the resume test could never be byte-exact.
        key = (self.agent, self.seed, seed, len(text))
        tag = self._hash_unit("tag", *key)
        if self._hash_unit("malformed", *key) < self.malformed_rate:
            return Completion("I cannot produce JSON right now.", "stop")

        correct = self._hash_unit("correct", *key) < self.accuracy
        answer = gold if (correct and gold) else f"wrong-{self.agent}-{tag:.9f}"
        obj = {field: None for field in PROTOCOL_FIELDS}
        obj["reasoning"] = f"{self.agent} reasoning {tag:.6f}"
        obj[self.answer_field] = answer
        obj["action"] = "confirm_stop"
        obj["confirmed_answer"] = answer
        if self.answer_field != "tentative_answer":
            obj.pop("tentative_answer", None)
            obj.pop("confirmed_answer", None)
            obj["confirmed_completion"] = answer
        return Completion(json.dumps(obj, ensure_ascii=False), "stop")


def build_mock_callers(
    *,
    accuracy: float = 0.5,
    malformed_rate: float = 0.05,
    gold_by_problem: dict[str, str] | None = None,
    answer_field: str = "tentative_answer",
) -> dict[str, MockCaller]:
    return {
        agent: MockCaller(
            agent,
            accuracy=accuracy,
            malformed_rate=malformed_rate,
            gold_by_problem=gold_by_problem,
            answer_field=answer_field,
            seed=index,
        )
        for index, agent in enumerate(AGENT_IDS)
    }

"""HTTP transport to the vLLM servers, plus the mock used by the CPU smoke test.

The vLLM 0.16 token-ID response extension preserves the actual prompt and sampled
tokens for PPO, including stop tokens that are absent from the displayed text.
Sampler settings and adapter versions travel with every completion.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol

from .protocol import AGENT_IDS, PROTOCOL_FIELDS


@dataclass
class GenerationOptions:
    temperature: float = 1.0
    top_p: float = 1.0
    max_new_tokens: int = 1024
    enable_thinking: bool = False
    top_k: int = 0
    min_tokens: int = 0
    force_fixed_length: bool = False
    guided_decoding: bool = True
    logprobs_mode: str = "raw_logprobs"
    require_token_ids: bool = False


@dataclass
class Completion:
    text: str
    finish_reason: str | None = None
    prompt_token_ids: list[int] | None = None
    response_token_ids: list[int] | None = None
    response_logprobs: list[float] | None = None
    model_name: str | None = None
    adapter_version: str | None = None
    generation_config: dict[str, Any] = field(default_factory=dict)
    guided_decoding: bool = False
    stop_reason: str | int | None = None


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
        retries: int = 8,
        adapter_version: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.generation = generation or GenerationOptions()
        self.timeout = timeout
        self.api_key = api_key
        self.retries = retries
        self.adapter_version = adapter_version

    def set_model_name(self, name: str, adapter_version: str | None = None) -> None:
        self.model_name = name
        self.adapter_version = adapter_version

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
            "top_k": self.generation.top_k or -1,
            "min_p": 0.0,
            "min_tokens": (
                self.generation.max_new_tokens
                if self.generation.force_fixed_length
                else self.generation.min_tokens
            ),
            "max_tokens": self.generation.max_new_tokens,
            "ignore_eos": False,
            "stop_token_ids": [],
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
            "logprobs": True,
            "top_logprobs": 0,
            "chat_template_kwargs": {"enable_thinking": self.generation.enable_thinking},
        }
        if seed is not None:
            payload["seed"] = int(seed)
        if response_format is not None and self.generation.guided_decoding:
            payload["response_format"] = response_format
        adapter_version = self.adapter_version
        generation_config = {
            key: value for key, value in payload.items()
            if key not in {"messages", "model"}
        }
        generation_config["logprobs_mode"] = self.generation.logprobs_mode
        generation_config["enable_thinking"] = self.generation.enable_thinking
        generation_config["force_fixed_length"] = self.generation.force_fixed_length

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
                return self._decode_completion(
                    data, model_name=payload["model"],
                    adapter_version=adapter_version,
                    generation_config=generation_config,
                    require_token_ids=self.generation.require_token_ids,
                )
            except Exception as exc:  # noqa: BLE001 - retried, then surfaced
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(4.0, 0.5 * (2**attempt)))
        raise RuntimeError(f"generation failed at {self.base_url}: {last_error}")

    @staticmethod
    def _decode_completion(
        data: dict[str, Any], *, model_name: str,
        adapter_version: str | None, generation_config: dict[str, Any],
        require_token_ids: bool = True,
    ) -> Completion:
        choices = data.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ValueError("vLLM must return exactly one completion choice")
        choice = choices[0]
        content = choice.get("message", {}).get("content")
        if isinstance(content, list):
            content = "".join(
                str(part.get("text", ""))
                for part in content if isinstance(part, dict)
            )
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise ValueError("vLLM returned nontext message content")
        completion = Completion(
            text=content,
            finish_reason=choice.get("finish_reason"),
            model_name=model_name,
            adapter_version=adapter_version,
            generation_config=generation_config,
            guided_decoding="response_format" in generation_config,
            stop_reason=choice.get("stop_reason"),
        )
        prompt_ids = data.get("prompt_token_ids")
        response_ids = choice.get("token_ids")
        if not require_token_ids and prompt_ids is None and response_ids is None:
            return completion
        for name, token_ids in (
            ("prompt_token_ids", prompt_ids), ("token_ids", response_ids),
        ):
            if not isinstance(token_ids, list) or not token_ids or any(
                type(token_id) is not int or token_id < 0 for token_id in token_ids
            ):
                raise ValueError(f"vLLM did not return valid exact {name}")
        logprob_items = (choice.get("logprobs") or {}).get("content")
        if not isinstance(logprob_items, list) or len(logprob_items) != len(response_ids):
            raise ValueError("vLLM logprobs must match the generated token IDs")
        logprobs = []
        for token_id, item in zip(response_ids, logprob_items):
            if item.get("token") != f"token_id:{token_id}":
                raise ValueError("vLLM logprob token identity differs from token_ids")
            value = item.get("logprob")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("vLLM returned a nonnumeric generated logprob")
            if not math.isfinite(value) or value > 1e-5:
                raise ValueError("vLLM returned an invalid generated logprob")
            logprobs.append(float(value))
        usage = data.get("usage") or {}
        for key, token_ids in (
            ("prompt_tokens", prompt_ids), ("completion_tokens", response_ids),
        ):
            if key in usage and usage[key] != len(token_ids):
                raise ValueError(f"vLLM {key} disagrees with its exact token IDs")
        completion.prompt_token_ids = prompt_ids
        completion.response_token_ids = response_ids
        completion.response_logprobs = logprobs
        return completion


class EndpointPoolCaller:
    """Least-in-flight dispatch across replicas of one agent's model."""

    def __init__(self, callers: list[OpenAIChatCaller]) -> None:
        if not callers:
            raise ValueError("endpoint pool must not be empty")
        self._callers = callers
        self._active = [0] * len(callers)
        self._lock = threading.Lock()
        self._next = 0

    def set_model_name(self, name: str, adapter_version: str | None = None) -> None:
        for caller in self._callers:
            caller.set_model_name(name, adapter_version=adapter_version)

    def _acquire(self) -> int:
        with self._lock:
            minimum = min(self._active)
            for offset in range(len(self._active)):
                index = (self._next + offset) % len(self._active)
                if self._active[index] == minimum:
                    self._active[index] += 1
                    self._next = (index + 1) % len(self._active)
                    return index
            raise RuntimeError("endpoint selection failed")

    def generate(self, messages, *, seed=None, response_format=None) -> Completion:
        index = self._acquire()
        try:
            return self._callers[index].generate(
                messages, seed=seed, response_format=response_format
            )
        finally:
            with self._lock:
                self._active[index] -= 1


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

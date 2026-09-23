"""Shared streaming LLM client for the svip-hosted qwen3-8b endpoint.

OpenAI-compatible Server-Sent Events (SSE) format. Each event is:
    data: {"choices":[{"delta":{...}}]}\n\n
Last event is:
    data: [DONE]\n\n

Deltas accumulate three things:
  - content (string, appended)
  - tool_calls[i].id (set once)
  - tool_calls[i].function.name (set once)
  - tool_calls[i].function.arguments (string, appended chunk-by-chunk)

We reassemble these into a final assistant message dict identical to the
non-streaming response, so downstream code (which expects the OpenAI message
schema) works unchanged.

Used by frozen_baseline_probe.py and multi_agent_scheduler.py.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Optional

import requests


# Keep the hosted endpoint as the default, but allow experiment scripts to
# point the judge at an OpenAI-compatible local server without changing the
# global configuration.  The URL is read at request time so a long-running
# process can still be configured through its environment before startup.
LLM_BASE_URL = "https://svip.xty.app/v1/chat/completions"
LLM_MODEL = "gpt-5"
LLM_TIMEOUT_SEC = 600
LLM_MAX_TOKENS = 4096
LLM_TEMPERATURE = 0.2
LLM_TOP_P = 1.0
LLM_MAX_RETRIES = 2
LLM_API_KEY = ""


def _accumulate_delta(state: dict, delta: dict) -> None:
    """Merge one delta into the assistant-message state."""
    if not isinstance(delta, dict):
        return

    if "content" in delta and delta["content"] is not None:
        state["content"] = (state.get("content") or "") + delta["content"]

    if "role" in delta:
        state["role"] = delta["role"]

    # vLLM can stream Qwen reasoning separately from the answer.  Keep it in
    # the normalized message for callers that need it, while leaving
    # `content` as the final answer used by the JSON judge parser.
    if "reasoning_content" in delta and delta["reasoning_content"] is not None:
        state["reasoning_content"] = (
            (state.get("reasoning_content") or "")
            + delta["reasoning_content"]
        )

    raw_tcs = delta.get("tool_calls")
    if not raw_tcs:
        return

    tc_list = state.setdefault("tool_calls", [])

    for tc_delta in raw_tcs:
        # Each delta carries an "index" identifying which tool_call slot it
        # belongs to. Multiple deltas with the same index merge.
        idx = tc_delta.get("index", 0)
        while len(tc_list) <= idx:
            tc_list.append({"id": None, "type": "function",
                            "function": {"name": "", "arguments": ""}})
        slot = tc_list[idx]

        if tc_delta.get("id") is not None:
            slot["id"] = tc_delta["id"]
        if tc_delta.get("type") is not None:
            slot["type"] = tc_delta["type"]

        fn_delta = tc_delta.get("function") or {}
        if fn_delta.get("name"):
            slot["function"]["name"] = (slot["function"].get("name") or "") + fn_delta["name"]
        if fn_delta.get("arguments"):
            slot["function"]["arguments"] = \
                (slot["function"].get("arguments") or "") + fn_delta["arguments"]


def _stream_once(
    messages: List[dict],
    tools: Optional[List[dict]],
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: float,
    reasoning_effort: Optional[str],
    enable_thinking: Optional[bool] = None,
    response_format: Optional[dict] = None,
) -> dict:
    """Single streaming request. Returns reassembled assistant message dict."""
    api_base = (
        os.environ.get("JCA_JUDGE_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
        or LLM_BASE_URL
    ).rstrip("/")
    if not api_base.endswith("/chat/completions"):
        api_base = f"{api_base}/chat/completions"
    api_key = (
        os.environ.get("JCA_JUDGE_API_KEY")
        or os.environ.get("SVIP_API_KEY")
        or LLM_API_KEY
    )
    if not api_key or api_key == "REPLACE_WITH_YOUR_SVIP_API_KEY":
        sys.exit("[fatal] Set SVIP_API_KEY or replace LLM_API_KEY in llm_client.py")

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        payload["response_format"] = response_format
    # vLLM's OpenAI server does not accept the provider-specific
    # reasoning_effort field.  Hosted GPT calls retain the old behavior.
    if reasoning_effort and os.environ.get("JCA_JUDGE_DISABLE_REASONING_EFFORT") != "1":
        payload["reasoning_effort"] = reasoning_effort
    # Qwen3's thinking switch is an OpenAI-compatible vLLM request field.
    # Keep the old explicit-disable behavior and permit judge launchers to
    # request thinking explicitly instead of relying on a tokenizer default.
    if enable_thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    elif os.environ.get("JCA_JUDGE_DISABLE_THINKING") == "1":
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    elif os.environ.get("JCA_JUDGE_ENABLE_THINKING") == "1":
        payload["chat_template_kwargs"] = {"enable_thinking": True}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    state: dict = {"role": "assistant", "content": None}

    # `stream=True` on the requests side returns the response body as a
    # readable stream; we iterate SSE lines until [DONE].
    with requests.post(
        api_base, headers=headers, json=payload,
        timeout=timeout, stream=True,
    ) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if raw.startswith("data:"):
                data_str = raw[len("data:"):].strip()
            else:
                # Some servers prepend whitespace; tolerate.
                data_str = raw.strip()
            if data_str == "[DONE]":
                break
            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                # Stray keep-alive comment or malformed line — ignore.
                continue
            choices = event.get("choices") or []
            if event.get("usage") is not None:
                state["usage"] = event["usage"]
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            _accumulate_delta(state, delta)
            if choice.get("finish_reason") is not None:
                state["finish_reason"] = choice["finish_reason"]

    # Normalize tool_calls: drop placeholder slots that never got filled.
    if state.get("tool_calls"):
        cleaned = []
        for tc in state["tool_calls"]:
            fn = tc.get("function") or {}
            if fn.get("name"):
                cleaned.append(tc)
        if cleaned:
            state["tool_calls"] = cleaned
        else:
            state.pop("tool_calls", None)

    return state


def call_llm_stream(
    messages: List[dict],
    tools: Optional[List[dict]] = None,
    *,
    model: str = LLM_MODEL,
    max_retries: int = LLM_MAX_RETRIES,
    temperature: float = LLM_TEMPERATURE,
    top_p: float = LLM_TOP_P,
    max_tokens: int = LLM_MAX_TOKENS,
    timeout: float = LLM_TIMEOUT_SEC,
    reasoning_effort: Optional[str] = None,
    enable_thinking: Optional[bool] = None,
    response_format: Optional[dict] = None,
) -> dict:
    """Streaming LLM call with retry on transient errors.

    Returns the same shape as the non-streaming response's
    `choices[0].message`, so callers don't need to change.
    """
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            return _stream_once(
                messages,
                tools,
                model,
                temperature,
                top_p,
                max_tokens,
                timeout,
                reasoning_effort,
                enable_thinking,
                response_format,
            )
        except (requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError) as e:
            last_err = e
            if attempt < max_retries:
                wait = 5 * (2 ** attempt)
                print(f"        [retry {attempt+1}/{max_retries}] "
                      f"{type(e).__name__}; waiting {wait}s")
                time.sleep(wait)
                continue
            raise
        except requests.exceptions.HTTPError as e:
            if e.response is not None and 500 <= e.response.status_code < 600 \
                    and attempt < max_retries:
                last_err = e
                wait = 5 * (2 ** attempt)
                print(f"        [retry {attempt+1}/{max_retries}] HTTP "
                      f"{e.response.status_code}; waiting {wait}s")
                time.sleep(wait)
                continue
            raise
    raise last_err  # unreachable

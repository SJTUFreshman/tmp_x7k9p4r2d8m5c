"""Inference helpers for zero-shot Qwen multi-agent rollouts."""

from __future__ import annotations

from dataclasses import dataclass
import io
import math
import os
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Literal, Optional, Sequence

import json
import re
import urllib.error
import urllib.request

from .agents import AGENT_IDS


DEFAULT_LOCAL_MODEL_PATHS: Dict[str, Path] = {
    "A1": Path("/data/wangyuheng/models/Qwen3-1.7B"),
    "A2": Path("/data/wangyuheng/models/Qwen3-4B"),
    "A3": Path("/data/wangyuheng/models/Qwen3-8B"),
}


class GeneratedText(str):
    """String response carrying every transport-level generation attempt."""

    def __new__(cls, value: str, raw_outputs: Sequence[str] | None = None):
        instance = super().__new__(cls, value)
        instance.raw_outputs = tuple(raw_outputs or (value,))
        return instance


def response_attempts(raw: object) -> List[str]:
    attempts = getattr(raw, "raw_outputs", None)
    if isinstance(attempts, (list, tuple)):
        return [str(text) for text in attempts]
    return [str(raw)]


@dataclass(frozen=True)
class GenerationOptions:
    """Text generation options shared by all zero-shot agents."""

    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 0.95
    enable_thinking: bool = False

    @property
    def do_sample(self) -> bool:
        return self.temperature > 0.0


class QwenLLMCaller:
    """Callable adapter matching scheduler.LLMCaller for one Qwen model."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        generation: GenerationOptions | None = None,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        trust_remote_code: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.generation = generation or GenerationOptions()
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.trust_remote_code = trust_remote_code
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        """Load tokenizer and model weights if they are not already loaded."""
        if self.model is not None and self.tokenizer is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "QwenLLMCaller requires torch and transformers for inference."
            ) from exc

        dtype = _resolve_torch_dtype(torch, self.torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=self.device_map,
            torch_dtype=dtype,
            trust_remote_code=self.trust_remote_code,
        )
        self.model.eval()

    def __call__(self, messages: List[Dict[str, str]]) -> str:
        """Generate one raw assistant response for the scheduler."""
        self.load()
        assert self.model is not None
        assert self.tokenizer is not None

        import torch

        encoded = _apply_chat_template(
            self.tokenizer,
            messages,
            enable_thinking=self.generation.enable_thinking,
        )
        encoded = _move_encoded_to_device(encoded, _first_model_device(self.model))
        input_ids = encoded["input_ids"]
        prompt_len = input_ids.shape[-1]

        generate_kwargs = {
            "max_new_tokens": self.generation.max_new_tokens,
            "do_sample": self.generation.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.generation.do_sample:
            generate_kwargs["temperature"] = self.generation.temperature
            generate_kwargs["top_p"] = self.generation.top_p

        with torch.inference_mode():
            output_ids = self.model.generate(**encoded, **generate_kwargs)

        generated_ids = output_ids[0][prompt_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


class VLLMQwenLLMCaller:
    """vLLM adapter matching scheduler.LLMCaller for one Qwen model."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        generation: GenerationOptions | None = None,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.30,
        max_model_len: int | None = None,
        dtype: str = "auto",
        trust_remote_code: bool = True,
        enforce_eager: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.generation = generation or GenerationOptions()
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.dtype = _normalize_vllm_dtype(dtype)
        self.trust_remote_code = trust_remote_code
        self.enforce_eager = enforce_eager
        self.engine = None
        self.tokenizer = None

    def load(self) -> None:
        """Load vLLM engine and tokenizer if they are not already loaded."""
        if self.engine is not None and self.tokenizer is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")

        try:
            from transformers import AutoTokenizer
            from vllm import LLM
        except ImportError as exc:
            raise ImportError(
                "VLLMQwenLLMCaller requires vllm and transformers for inference."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
        )
        llm_kwargs = {
            "model": str(self.model_path),
            "tokenizer": str(self.model_path),
            "trust_remote_code": self.trust_remote_code,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "dtype": self.dtype,
            "enforce_eager": self.enforce_eager,
        }
        if self.max_model_len is not None:
            llm_kwargs["max_model_len"] = self.max_model_len

        self.engine = LLM(**llm_kwargs)

    def __call__(self, messages: List[Dict[str, str]]) -> str:
        """Generate one raw assistant response through vLLM."""
        self.load()
        assert self.engine is not None
        assert self.tokenizer is not None

        from vllm import SamplingParams

        prompt = _apply_chat_template_text(
            self.tokenizer,
            messages,
            enable_thinking=self.generation.enable_thinking,
        )
        sampling_params = SamplingParams(
            max_tokens=self.generation.max_new_tokens,
            temperature=self.generation.temperature,
            top_p=self.generation.top_p,
        )
        outputs = self.engine.generate([prompt], sampling_params, use_tqdm=False)
        return outputs[0].outputs[0].text.strip()


class OpenAIChatLLMCaller:
    """OpenAI-compatible chat-completions caller for external vLLM servers."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        *,
        generation: GenerationOptions | None = None,
        timeout: float = 600.0,
        api_key: str = "EMPTY",
        max_model_len: int | None = None,
        response_format: Dict[str, Any] | None = None,
        length_retries: int = 1,
        thinking_fallback: bool = False,
        thinking_max_tokens: int | None = None,
        thinking_stop: Sequence[str] | None = None,
        fallback_bad_words: Sequence[str] | None = None,
        fallback_stop: Sequence[str] | None = None,
        connection_pool: bool | None = None,
        connection_pool_maxsize: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.generation = generation or GenerationOptions()
        self.timeout = timeout
        self.api_key = api_key
        self.max_model_len = max_model_len
        self.response_format = response_format
        self.length_retries = length_retries
        self.thinking_fallback = thinking_fallback
        self.thinking_max_tokens = thinking_max_tokens
        self.thinking_stop = tuple(thinking_stop or ())
        self.fallback_bad_words = tuple(fallback_bad_words or ())
        self.fallback_stop = tuple(fallback_stop or ())
        if connection_pool is None:
            connection_pool = os.environ.get("JCA_HTTP_KEEPALIVE", "0").lower() in {
                "1", "true", "yes", "on"
            }
        self.connection_pool = bool(connection_pool)
        configured_pool_size = connection_pool_maxsize
        if configured_pool_size is None:
            configured_pool_size = int(os.environ.get("JCA_HTTP_POOL_MAXSIZE", "128"))
        if configured_pool_size <= 0:
            raise ValueError("connection pool maxsize must be positive")
        self.connection_pool_maxsize = configured_pool_size
        self._http_pool = None
        self._http_pool_lock = Lock()

    def _read_request(self, request: urllib.request.Request) -> bytes:
        if not self.connection_pool:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read()
        try:
            import urllib3
        except ImportError:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return response.read()
        with self._http_pool_lock:
            if self._http_pool is None:
                self._http_pool = urllib3.PoolManager(
                    num_pools=1,
                    maxsize=self.connection_pool_maxsize,
                    block=True,
                    retries=False,
                )
            pool = self._http_pool
        headers = {name: value for name, value in request.header_items()}
        response = None
        try:
            response = pool.request(
                "POST",
                request.full_url,
                body=request.data,
                headers=headers,
                timeout=urllib3.Timeout(connect=self.timeout, read=self.timeout),
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

    def __call__(self, messages: List[Dict[str, str]]) -> str:
        """Send one chat-completions request and return assistant content."""
        return self.generate(messages)

    def generate(
        self,
        messages: List[Dict[str, str]],
        *,
        seed: int | None = None,
        temperature: float | None = None,
        response_format: Dict[str, Any] | None = None,
    ) -> str:
        """Generate with optional per-request sampling overrides."""
        configured_thinking = self.generation.enable_thinking
        request_temperature = (
            self.generation.temperature if temperature is None else float(temperature)
        )
        request_response_format = response_format or self.response_format
        use_thinking = configured_thinking
        structured_thinking_fallback = bool(
            self.thinking_fallback
            and configured_thinking
            and request_response_format is not None
        )
        max_tokens = self.generation.max_new_tokens
        if structured_thinking_fallback and self.thinking_max_tokens is not None:
            max_tokens = min(max_tokens, self.thinking_max_tokens)
        length_attempt = 0
        raw_outputs: List[str] = []
        fallback_reasoning = ""
        while True:
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": request_temperature,
                "top_p": self.generation.top_p,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {
                    "enable_thinking": use_thinking,
                },
            }
            if use_thinking and self.thinking_stop:
                payload["stop"] = list(self.thinking_stop)
            if not use_thinking and self.fallback_bad_words:
                payload["bad_words"] = list(self.fallback_bad_words)
            if not use_thinking and self.fallback_stop:
                payload["stop"] = list(self.fallback_stop)
            if seed is not None:
                payload["seed"] = int(seed)
            if request_response_format is not None:
                payload["response_format"] = request_response_format
            data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            try:
                raw = self._read_request(request).decode("utf-8")
                parsed = json.loads(raw)
                message = parsed["choices"][0]["message"]
                content = str(message.get("content") or "").strip()
                reasoning = str(
                    message.get("reasoning_content")
                    or message.get("reasoning")
                    or ""
                ).strip()
                response_text = (
                    f"<think>\n{reasoning}\n</think>\n{content}".strip()
                    if reasoning else content
                )
                raw_outputs.append(response_text)
                finish_reason = parsed["choices"][0].get("finish_reason")
                if structured_thinking_fallback and use_thinking:
                    reasoning_json = _json_object_text(reasoning)
                    if not content and reasoning_json is not None:
                        content = reasoning_json
                        reasoning = _json_reasoning(reasoning_json) or reasoning
                        response_text = (
                            f"<think>\n{reasoning}\n</think>\n{content}".strip()
                        )
                        break
                    if finish_reason == "length" or not content:
                        fallback_reasoning = reasoning or content
                        use_thinking = False
                        structured_thinking_fallback = False
                        max_tokens = self.generation.max_new_tokens
                        length_attempt = 0
                        continue
                if configured_thinking and not use_thinking:
                    reasoning = fallback_reasoning or _json_reasoning(content)
                    response_text = (
                        f"<think>\n{reasoning}\n</think>\n{content}".strip()
                        if reasoning else content
                    )
                elif configured_thinking and not reasoning:
                    inferred_reasoning = _json_reasoning(content)
                    if inferred_reasoning:
                        response_text = (
                            f"<think>\n{inferred_reasoning}\n</think>\n{content}".strip()
                        )
                if finish_reason == "length" and length_attempt < self.length_retries:
                    length_attempt += 1
                    max_tokens = max(max_tokens + 1, math.ceil(max_tokens * 1.5))
                    continue
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                reduced = _context_safe_max_tokens(
                    body,
                    requested_max_tokens=max_tokens,
                    configured_max_model_len=self.max_model_len,
                )
                if exc.code == 400 and reduced is not None:
                    max_tokens = reduced
                    continue
                raise RuntimeError(
                    f"OpenAI-compatible server returned HTTP {exc.code}: {body}"
                ) from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(
                    f"Could not reach OpenAI-compatible server at {self.base_url}: {exc}"
                ) from exc

        return GeneratedText(response_text, raw_outputs)


def _json_object_text(value: str) -> str | None:
    text = value.strip()
    if not text.startswith("{"):
        return None
    try:
        parsed = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return text if isinstance(parsed, dict) else None


def _json_reasoning(value: str) -> str:
    text = _json_object_text(value)
    if text is None:
        return ""
    parsed = json.loads(text)
    reasoning = parsed.get("reasoning")
    return reasoning.strip() if isinstance(reasoning, str) else ""


def build_qwen_llm_callers(
    model_paths: Dict[str, str | Path],
    *,
    agent_ids: Optional[Sequence[str]] = None,
    backend: Literal["vllm", "transformers", "openai"] = "vllm",
    generation: GenerationOptions | None = None,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.30,
    max_model_len: int | None = None,
    enforce_eager: bool = False,
    trust_remote_code: bool = True,
    api_base_urls: Optional[Dict[str, str]] = None,
    api_model_names: Optional[Dict[str, str]] = None,
    api_timeout: float = 600.0,
    api_key: str = "EMPTY",
) -> Dict[str, object]:
    """Create scheduler-compatible callers for A1/A2/A3."""
    selected_agent_ids = list(agent_ids) if agent_ids is not None else list(AGENT_IDS)
    unknown = [agent_id for agent_id in selected_agent_ids if agent_id not in AGENT_IDS]
    if unknown:
        raise ValueError(f"Unknown agent ID(s): {unknown}")

    missing = [
        agent_id for agent_id in selected_agent_ids if agent_id not in model_paths
    ]
    if missing:
        raise ValueError(f"Missing model path(s) for: {missing}")

    if backend == "openai":
        if api_base_urls is None:
            raise ValueError("api_base_urls is required for openai backend")
        missing_urls = [
            agent_id for agent_id in selected_agent_ids if agent_id not in api_base_urls
        ]
        if missing_urls:
            raise ValueError(f"Missing API base URL(s) for: {missing_urls}")
        api_model_names = api_model_names or {
            agent_id: agent_id for agent_id in selected_agent_ids
        }
        return {
            agent_id: OpenAIChatLLMCaller(
                api_base_urls[agent_id],
                api_model_names.get(agent_id, agent_id),
                generation=generation,
                timeout=api_timeout,
                api_key=api_key,
                max_model_len=max_model_len,
            )
            for agent_id in selected_agent_ids
        }

    if backend == "vllm":
        return {
            agent_id: VLLMQwenLLMCaller(
                model_paths[agent_id],
                generation=generation,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                enforce_eager=enforce_eager,
            )
            for agent_id in selected_agent_ids
        }

    if backend != "transformers":
        raise ValueError(f"Unknown inference backend: {backend}")

    return {
        agent_id: QwenLLMCaller(
            model_paths[agent_id],
            generation=generation,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        for agent_id in selected_agent_ids
    }


def _resolve_torch_dtype(torch_module, dtype_name: str):
    """Convert a CLI dtype string to the value expected by transformers."""
    if dtype_name == "auto":
        return "auto"
    aliases = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }
    attr = aliases.get(dtype_name, dtype_name)
    if not hasattr(torch_module, attr):
        raise ValueError(f"Unknown torch dtype: {dtype_name}")
    return getattr(torch_module, attr)


def _apply_chat_template(tokenizer, messages, *, enable_thinking: bool):
    """Apply a Qwen chat template, disabling thinking when supported."""
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "return_dict": True,
    }
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=enable_thinking,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def _move_encoded_to_device(encoded, device):
    """Move tokenized model inputs to the model's first device."""
    if hasattr(encoded, "to"):
        return encoded.to(device)
    return {key: value.to(device) for key, value in encoded.items()}


def _first_model_device(model):
    """Return the device where input IDs should be placed."""
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def _apply_chat_template_text(tokenizer, messages, *, enable_thinking: bool) -> str:
    """Apply a Qwen chat template and return a prompt string for vLLM."""
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    try:
        return tokenizer.apply_chat_template(
            messages,
            enable_thinking=enable_thinking,
            **kwargs,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def _normalize_vllm_dtype(dtype_name: str) -> str:
    """Convert dtype aliases to vLLM's accepted dtype strings."""
    aliases = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }
    return aliases.get(dtype_name, dtype_name)


def _context_safe_max_tokens(
    error_body: str,
    *,
    requested_max_tokens: int,
    configured_max_model_len: int | None,
) -> int | None:
    """Reduce only the completion budget for a vLLM context-length error."""
    prompt_match = re.search(r"prompt contains at least ([\d,]+) input tokens", error_body)
    prompt_char_match = re.search(r"prompt contains ([\d,]+) characters", error_body)
    context_match = re.search(r"maximum context length is ([\d,]+) tokens", error_body)
    if prompt_match is None and prompt_char_match is None:
        return None
    if prompt_match is not None:
        prompt_tokens = int(prompt_match.group(1).replace(",", ""))
    else:
        # Newer vLLM versions validate max_tokens before tokenization and only
        # report prompt characters. One ASCII character cannot expand to more
        # than one tokenizer token, so this is a conservative retry budget.
        assert prompt_char_match is not None
        prompt_tokens = int(prompt_char_match.group(1).replace(",", ""))
    context_tokens = (
        int(context_match.group(1).replace(",", ""))
        if context_match is not None
        else configured_max_model_len
    )
    if context_tokens is None:
        return None
    available = context_tokens - prompt_tokens - 8
    if available < 1 or available >= requested_max_tokens:
        return None
    return available

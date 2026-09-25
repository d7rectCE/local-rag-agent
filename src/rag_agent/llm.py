"""LLM providers behind one interface (FR7): local Ollama or any OpenAI-compatible API.

Structured output (JSON schema) is passed to the server so that decoding is
constrained by the schema (ТЗ S7).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from rag_agent.config import LLMConfig
from rag_agent.tracing import NULL_TRACER, Tracer


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    content: str
    thinking: str | None = None
    tool_calls: list[dict] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    latency_s: float = 0.0

    def json(self) -> Any:
        text = self.content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[text.find("\n") + 1 :] if "\n" in text else text
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"model returned invalid JSON: {exc}: {self.content[:300]!r}") from exc


class BaseLLM:
    def __init__(self, cfg: LLMConfig, tracer: Tracer = NULL_TRACER):
        self.cfg = cfg
        self.tracer = tracer
        self._http = httpx.Client(timeout=cfg.timeout_s)

    @property
    def name(self) -> str:
        return f"{self.cfg.provider}:{self.cfg.model}"

    def chat(
        self,
        messages: list[dict],
        *,
        json_schema: dict | None = None,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        think: bool | None = None,
        purpose: str = "chat",
    ) -> LLMResponse:
        t0 = time.perf_counter()
        try:
            resp = self._chat(messages, json_schema, tools, temperature, max_tokens, think)
        except httpx.HTTPError as exc:
            self.tracer.log("llm_error", purpose=purpose, model=self.name, error=str(exc))
            raise LLMError(f"{self.name}: {exc}") from exc
        resp.latency_s = time.perf_counter() - t0
        self.tracer.log(
            "llm_call",
            purpose=purpose,
            model=self.name,
            latency_s=round(resp.latency_s, 3),
            usage=resp.usage,
            messages=messages if self.tracer.log_prompts else None,
            response=resp.content if self.tracer.log_prompts else None,
            tool_calls=resp.tool_calls or None,
        )
        return resp

    def _chat(self, messages, json_schema, tools, temperature, max_tokens, think) -> LLMResponse:
        raise NotImplementedError

    def unload(self) -> None:
        """Release the model on the server, if the provider supports it."""

    def close(self) -> None:
        self._http.close()


class OllamaLLM(BaseLLM):
    def _chat(self, messages, json_schema, tools, temperature, max_tokens, think) -> LLMResponse:
        body: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "think": self.cfg.think if think is None else think,
            "options": {
                "temperature": self.cfg.temperature if temperature is None else temperature,
                "num_ctx": self.cfg.num_ctx,
                "num_predict": max_tokens or self.cfg.max_tokens,
            },
        }
        if json_schema is not None:
            body["format"] = json_schema
        if tools:
            body["tools"] = tools
        r = self._http.post(f"{self.cfg.base_url.rstrip('/')}/api/chat", json=body)
        if r.status_code >= 400:
            hint = ""
            if "kernel image is invalid" in r.text or "llama-server process has terminated" in r.text:
                hint = (
                    " — похоже, Ollama запустила модель на встроенной видеокарте (например, когда дискретная занята"
                    " другой моделью). Ограничьте Ollama дискретной картой: HIP_VISIBLE_DEVICES=0 (AMD) или"
                    " CUDA_VISIBLE_DEVICES=0 (NVIDIA), затем перезапустите Ollama."
                )
            raise LLMError(f"{self.name}: HTTP {r.status_code}: {r.text[:500]}{hint}")
        data = r.json()
        msg = data.get("message", {})
        return LLMResponse(
            content=msg.get("content", ""),
            thinking=msg.get("thinking"),
            tool_calls=msg.get("tool_calls") or [],
            usage={"prompt_tokens": data.get("prompt_eval_count", 0), "completion_tokens": data.get("eval_count", 0)},
        )

    def unload(self) -> None:
        """Free the model's GPU memory now instead of after Ollama's keep-alive timeout."""
        try:
            self._http.post(f"{self.cfg.base_url.rstrip('/')}/api/generate", json={"model": self.cfg.model, "keep_alive": 0})
        except httpx.HTTPError:
            pass

    def is_available(self) -> bool:
        try:
            r = self._http.get(f"{self.cfg.base_url.rstrip('/')}/api/tags", timeout=3)
            names = {m.get("name") for m in r.json().get("models", [])}
            return self.cfg.model in names or f"{self.cfg.model}:latest" in names
        except (httpx.HTTPError, ValueError):
            return False


class OpenAICompatLLM(BaseLLM):
    def _chat(self, messages, json_schema, tools, temperature, max_tokens, think) -> LLMResponse:
        headers = {}
        if self.cfg.api_key_env:
            key = os.environ.get(self.cfg.api_key_env)
            if not key:
                raise LLMError(f"environment variable {self.cfg.api_key_env} is not set")
            headers["Authorization"] = f"Bearer {key}"
        body: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.cfg.max_tokens,
        }
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": json_schema, "strict": True},
            }
        if tools:
            body["tools"] = tools
        r = self._http.post(f"{self.cfg.base_url.rstrip('/')}/chat/completions", json=body, headers=headers)
        if r.status_code >= 400:
            raise LLMError(f"{self.name}: HTTP {r.status_code}: {r.text[:500]}")
        data = r.json()
        msg = data["choices"][0]["message"]
        usage = data.get("usage") or {}
        return LLMResponse(
            content=msg.get("content") or "",
            tool_calls=msg.get("tool_calls") or [],
            usage={"prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": usage.get("completion_tokens", 0)},
        )

    def is_available(self) -> bool:
        return True


def make_llm(cfg: LLMConfig, tracer: Tracer = NULL_TRACER) -> BaseLLM:
    if cfg.provider == "ollama":
        return OllamaLLM(cfg, tracer)
    if cfg.provider == "openai":
        return OpenAICompatLLM(cfg, tracer)
    raise ValueError(f"unknown LLM provider {cfg.provider!r}")

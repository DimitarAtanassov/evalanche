"""Ollama provider adapter."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

import httpx

from evalharness.core.enums import ErrorClass, FinishReason
from evalharness.core.models import (
    Capabilities,
    GenerationRequest,
    GenerationResponse,
    ModelVersion,
    ToolCall,
)


class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=120.0)

    async def resolve_version(self, model: str) -> ModelVersion:
        tags_resp = await self._client.get("/api/tags")
        tags_resp.raise_for_status()
        match = next(
            (
                item
                for item in tags_resp.json().get("models", [])
                if item.get("name") == model
                or item.get("model") == model
                or str(item.get("name", "")).split(":", 1)[0] == model
                or str(item.get("model", "")).split(":", 1)[0] == model
            ),
            None,
        )
        resp = await self._client.post("/api/show", json={"name": model})
        resp.raise_for_status()
        data = resp.json()
        details = data.get("details") or {}
        digest = (match or {}).get("digest") or data.get("digest") or details.get("digest")
        if not digest:
            raise ValueError(
                f"Ollama model '{model}' has no digest; cannot pin version. "
                "Pull the model and ensure /api/show returns a digest."
            )
        quantization = details.get("quantization_level") or details.get("family")
        return ModelVersion(
            provider=self.name,
            model=model,
            resolved_version=digest,
            quantization=quantization,
            params_b=_parse_params_b(details.get("parameter_size")),
            context_window=_context_window(data),
            capabilities=self._capabilities_from_show(data),
        )

    def capabilities(self, model: str) -> Capabilities:
        return Capabilities(
            supports_seed=True,
            supports_logprobs=False,
            supports_tools=True,
            supports_json_schema=False,
            supports_streaming=True,
            supports_system_role=True,
            max_context_tokens=8192,
        )

    def _capabilities_from_show(self, data: dict[str, Any]) -> Capabilities:
        advertised = set(data.get("capabilities") or [])
        base = self.capabilities("")
        base["supports_tools"] = "tools" in advertised
        base["supports_json_schema"] = "completion" in advertised
        base["max_context_tokens"] = _context_window(data) or base["max_context_tokens"]
        return base

    async def generate(self, model: str, req: GenerationRequest) -> GenerationResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in req.messages],
            "stream": True,
            "options": {
                "temperature": req.temperature,
            },
        }
        if req.max_tokens is not None:
            payload["options"]["num_predict"] = req.max_tokens
        if req.top_p is not None:
            payload["options"]["top_p"] = req.top_p
        if req.top_k is not None:
            payload["options"]["top_k"] = req.top_k
        if req.seed is not None:
            payload["options"]["seed"] = req.seed
        if req.stop:
            payload["options"]["stop"] = req.stop
        if req.response_format:
            payload["format"] = req.response_format.get("json_schema", req.response_format)
        if req.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in req.tools
            ]

        start = time.perf_counter()
        ttft_ms: float | None = None
        text_parts: list[str] = []
        raw_chunks: list[dict[str, Any]] = []
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        finish_reason = FinishReason.STOP
        tool_calls: list[ToolCall] = []

        async with self._client.stream(
            "POST",
            "/api/chat",
            json=payload,
            timeout=req.timeout_s,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                raw_chunks.append(chunk)
                if ttft_ms is None and chunk.get("message", {}).get("content"):
                    ttft_ms = (time.perf_counter() - start) * 1000
                content = chunk.get("message", {}).get("content", "")
                for call in chunk.get("message", {}).get("tool_calls", []):
                    function = call.get("function") or {}
                    tool_calls.append(
                        ToolCall(
                            id=call.get("id")
                            or hashlib.sha256(
                                json.dumps(call, sort_keys=True).encode()
                            ).hexdigest()[:16],
                            name=function.get("name", ""),
                            arguments=function.get("arguments") or {},
                        )
                    )
                if content:
                    text_parts.append(content)
                if chunk.get("done"):
                    prompt_tokens = chunk.get("prompt_eval_count")
                    completion_tokens = chunk.get("eval_count")
                    if chunk.get("done_reason") == "length":
                        finish_reason = FinishReason.LENGTH
                    break

        total_ms = (time.perf_counter() - start) * 1000
        if ttft_ms is None:
            ttft_ms = total_ms

        if tool_calls:
            finish_reason = FinishReason.TOOL_CALLS
        return GenerationResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            logprobs=None,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            raw={"chunks": raw_chunks},
        )

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = await self._client.post("/api/embed", json={"model": model, "input": texts})
        resp.raise_for_status()
        return [list(map(float, vector)) for vector in resp.json()["embeddings"]]

    def classify_error(self, exc: Exception) -> ErrorClass:
        if isinstance(exc, httpx.TimeoutException):
            return ErrorClass.RETRYABLE_TRANSIENT
        if isinstance(exc, httpx.HTTPStatusError):
            code = exc.response.status_code
            if code == 429:
                return ErrorClass.RETRYABLE_RATE_LIMIT
            if code in (401, 403):
                return ErrorClass.NON_RETRYABLE_AUTH
            if code >= 500:
                return ErrorClass.RETRYABLE_TRANSIENT
            return ErrorClass.NON_RETRYABLE_REQUEST
        if isinstance(exc, (httpx.ConnectError, httpx.ReadError)):
            return ErrorClass.RETRYABLE_TRANSIENT
        return ErrorClass.NON_RETRYABLE_REQUEST

    async def aclose(self) -> None:
        await self._client.aclose()


def _parse_params_b(size: str | None) -> float | None:
    if not size:
        return None
    size = size.upper().replace("B", "").strip()
    try:
        return float(size)
    except ValueError:
        return None


def _context_window(data: dict[str, Any]) -> int | None:
    model_info = data.get("model_info") or {}
    for key, value in model_info.items():
        if key.endswith(".context_length"):
            return int(value)
    return None

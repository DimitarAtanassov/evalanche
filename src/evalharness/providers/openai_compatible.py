"""Generic, version-pinned OpenAI-compatible HTTP adapter."""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from evalharness.domain.enums import ErrorClass, FinishReason
from evalharness.domain.generation import (
    Capabilities,
    GenerationRequest,
    GenerationResponse,
    ModelVersion,
    TokenLogprob,
    ToolCall,
)


class OpenAICompatibleProvider:
    name = "openai_compatible"

    def __init__(
        self,
        base_url: str,
        model_revision: str,
        api_key: str | None = None,
        organization: str | None = None,
    ) -> None:
        if not model_revision.strip():
            raise ValueError("model_revision is required for reproducible runs")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        if organization:
            headers["OpenAI-Organization"] = organization
        self.model_revision = model_revision
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=120.0,
        )

    async def resolve_version(self, model: str) -> ModelVersion:
        return ModelVersion(
            provider=self.name,
            model=model,
            resolved_version=self.model_revision,
            capabilities=self.capabilities(model),
        )

    def capabilities(self, model: str) -> Capabilities:
        return Capabilities(
            supports_seed=True,
            supports_logprobs=True,
            supports_tools=True,
            supports_json_schema=True,
            supports_streaming=True,
            supports_system_role=True,
            max_context_tokens=128_000,
        )

    async def generate(self, model: str, req: GenerationRequest) -> GenerationResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in req.messages],
            "temperature": req.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        for key, value in {
            "max_tokens": req.max_tokens,
            "top_p": req.top_p,
            "seed": req.seed,
            "stop": req.stop or None,
            "response_format": req.response_format,
        }.items():
            if value is not None:
                payload[key] = value
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
        chunks: list[dict[str, Any]] = []
        tool_fragments: dict[int, dict[str, str]] = {}
        token_logprobs: list[TokenLogprob] = []
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        finish_reason = FinishReason.STOP

        async with self._client.stream(
            "POST", "/v1/chat/completions", json=payload, timeout=req.timeout_s
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                body = line[6:]
                if body == "[DONE]":
                    break
                chunk = json.loads(body)
                chunks.append(chunk)
                usage = chunk.get("usage") or {}
                prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                completion_tokens = usage.get("completion_tokens", completion_tokens)
                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                content = delta.get("content") or ""
                if content:
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - start) * 1000
                    text_parts.append(content)
                for call in delta.get("tool_calls") or []:
                    idx = int(call.get("index", 0))
                    fragment = tool_fragments.setdefault(
                        idx, {"id": "", "name": "", "arguments": ""}
                    )
                    fragment["id"] += call.get("id") or ""
                    function = call.get("function") or {}
                    fragment["name"] += function.get("name") or ""
                    fragment["arguments"] += function.get("arguments") or ""
                for entry in (choice.get("logprobs") or {}).get("content") or []:
                    token_logprobs.append(
                        TokenLogprob(token=str(entry["token"]), logprob=float(entry["logprob"]))
                    )
                if choice.get("finish_reason"):
                    finish_reason = _finish_reason(choice["finish_reason"])

        total_ms = (time.perf_counter() - start) * 1000
        tools = [
            ToolCall(
                id=fragment["id"],
                name=fragment["name"],
                arguments=json.loads(fragment["arguments"] or "{}"),
            )
            for _, fragment in sorted(tool_fragments.items())
        ]
        return GenerationResponse(
            text="".join(text_parts),
            tool_calls=tools,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            logprobs=token_logprobs or None,
            ttft_ms=ttft_ms if ttft_ms is not None else total_ms,
            total_ms=total_ms,
            raw={"chunks": chunks, "model_revision": self.model_revision},
        )

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        response = await self._client.post("/v1/embeddings", json={"model": model, "input": texts})
        response.raise_for_status()
        rows = sorted(response.json()["data"], key=lambda row: row["index"])
        return [list(map(float, row["embedding"])) for row in rows]

    def classify_error(self, exc: Exception) -> ErrorClass:
        if isinstance(exc, (httpx.TimeoutException, httpx.ConnectError, httpx.ReadError)):
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

    async def aclose(self) -> None:
        await self._client.aclose()


def _finish_reason(value: str) -> FinishReason:
    return {
        "stop": FinishReason.STOP,
        "length": FinishReason.LENGTH,
        "tool_calls": FinishReason.TOOL_CALLS,
        "content_filter": FinishReason.CONTENT_FILTER,
    }.get(value, FinishReason.ERROR)

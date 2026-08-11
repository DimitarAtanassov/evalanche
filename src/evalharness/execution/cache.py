"""Response-cache key derivation and payload round-tripping for execution."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from evalharness.domain.enums import FinishReason
from evalharness.domain.generation import GenerationResponse, ToolCall
from evalharness.domain.ports import RunStoreFactory
from evalharness.hashing import sha256_canonical
from evalharness.observability import get_logger
from evalharness.db.session import session_scope

logger = get_logger(__name__)


def response_cache_key(
    *,
    provider: str,
    resolved_version: str,
    rendered_prompt: str,
    decode_params: dict[str, Any],
) -> str:
    """Key for the shared response cache; callers that purge must use this same derivation."""
    return sha256_canonical(
        {
            "provider": provider,
            "model_version": resolved_version,
            "prompt": rendered_prompt,
            "decode": decode_params,
            "adapter": f"{provider}-v1",
        }
    )


def cache_enabled_for(decode_params: dict[str, Any]) -> bool:
    """Only greedy decoding is reproducible enough for a stored response to be reusable."""
    return float(decode_params.get("temperature", 0.0)) == 0.0


def response_from_cache(payload: dict[str, Any]) -> GenerationResponse:
    """Rebuild a GenerationResponse from a cached payload dict."""
    return GenerationResponse(
        text=payload["text"],
        tool_calls=[ToolCall(**call) for call in payload.get("tool_calls", [])],
        finish_reason=FinishReason(payload["finish_reason"]),
        prompt_tokens=payload.get("prompt_tokens"),
        completion_tokens=payload.get("completion_tokens"),
        logprobs=None,
        ttft_ms=payload.get("ttft_ms"),
        total_ms=payload["total_ms"],
        raw=payload.get("raw", {}),
    )


def response_to_cache_payload(response: GenerationResponse) -> dict[str, Any]:
    """Flatten a GenerationResponse into the JSON-safe shape the cache stores."""
    return {
        "text": response.text,
        "tool_calls": [asdict(call) for call in response.tool_calls],
        "finish_reason": response.finish_reason.value,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "logprobs": None,
        "ttft_ms": response.ttft_ms,
        "total_ms": response.total_ms,
        "raw": response.raw,
    }


async def load_cached_response(
    run_store: RunStoreFactory, cache_key: str
) -> GenerationResponse | None:
    async with session_scope() as session:
        cached_payload = await run_store(session).get_cache(cache_key)
    if not cached_payload:
        return None
    logger.debug("cache_hit", cache_key=cache_key)
    return response_from_cache(cached_payload)


async def store_cached_response(
    run_store: RunStoreFactory, cache_key: str, response: GenerationResponse
) -> None:
    async with session_scope() as session:
        await run_store(session).put_cache(cache_key, response_to_cache_payload(response))

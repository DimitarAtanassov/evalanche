from __future__ import annotations

import json

import httpx
import pytest
import respx

from evalharness.domain.enums import FinishReason
from evalharness.domain.generation import GenerationRequest, Message
from evalharness.providers.openai_compatible import OpenAICompatibleProvider


@pytest.mark.asyncio
@respx.mock
async def test_openai_stream_and_embeddings_contract() -> None:
    chunks = [
        {"choices": [{"delta": {"content": "hel"}, "finish_reason": None}]},
        {
            "choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1},
        },
    ]
    body = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    respx.post("https://example.test/v1/chat/completions").mock(
        return_value=httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
    )
    respx.post("https://example.test/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [1.0, 2.0]}]},
        )
    )
    provider = OpenAICompatibleProvider("https://example.test", "sha256:abc", "dummy")
    response = await provider.generate(
        "model",
        GenerationRequest([Message("user", "hi")], 5, 0.0, None, None, 1, [], None, None, 10),
    )
    assert response.text == "hello"
    assert response.finish_reason == FinishReason.STOP
    assert await provider.embed("embedding", ["text"]) == [[1.0, 2.0]]
    await provider.aclose()

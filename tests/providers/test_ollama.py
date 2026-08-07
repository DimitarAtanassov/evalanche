from __future__ import annotations

import json

import httpx
import pytest
import respx

from evalharness.core.enums import ErrorClass, FinishReason
from evalharness.core.models import GenerationRequest, Message
from evalharness.providers.ollama import OllamaProvider


def _generation_request() -> GenerationRequest:
    return GenerationRequest(
        [Message("user", "hi")],
        5,
        0.0,
        None,
        None,
        1,
        [],
        None,
        None,
        10,
    )


@pytest.mark.asyncio
@respx.mock
async def test_ollama_generate_stream_happy_path() -> None:
    chunks = [
        {"message": {"role": "assistant", "content": "hel"}, "done": False},
        {
            "message": {"role": "assistant", "content": "lo"},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 2,
            "eval_count": 1,
        },
    ]
    body = "\n".join(json.dumps(chunk) for chunk in chunks) + "\n"
    route = respx.post("http://ollama.test/api/chat").mock(
        return_value=httpx.Response(200, text=body)
    )

    provider = OllamaProvider("http://ollama.test")
    try:
        response = await provider.generate("llama", _generation_request())
    finally:
        await provider.aclose()

    assert response.text == "hello"
    assert response.finish_reason == FinishReason.STOP
    assert response.prompt_tokens == 2
    assert response.completion_tokens == 1
    assert route.called
    sent = json.loads(route.calls.last.request.content.decode())
    assert sent["model"] == "llama"
    assert sent["stream"] is True
    assert sent["messages"] == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
@respx.mock
async def test_ollama_embed_happy_path() -> None:
    route = respx.post("http://ollama.test/api/embed").mock(
        return_value=httpx.Response(
            200,
            json={"embeddings": [[1.0, 2.0], [3.0, 4.0]]},
        )
    )

    provider = OllamaProvider("http://ollama.test")
    try:
        vectors = await provider.embed("nomic", ["a", "b"])
    finally:
        await provider.aclose()

    assert vectors == [[1.0, 2.0], [3.0, 4.0]]
    assert route.called
    sent = json.loads(route.calls.last.request.content.decode())
    assert sent == {"model": "nomic", "input": ["a", "b"]}


@pytest.mark.asyncio
@respx.mock
async def test_ollama_embed_non_2xx_classifies_rate_limit() -> None:
    respx.post("http://ollama.test/api/embed").mock(
        return_value=httpx.Response(429, json={"error": "rate limited"})
    )

    provider = OllamaProvider("http://ollama.test")
    try:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await provider.embed("nomic", ["a"])
        assert provider.classify_error(exc_info.value) == ErrorClass.RETRYABLE_RATE_LIMIT
    finally:
        await provider.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_ollama_generate_non_2xx_classifies_auth() -> None:
    respx.post("http://ollama.test/api/chat").mock(
        return_value=httpx.Response(401, json={"error": "unauthorized"})
    )

    provider = OllamaProvider("http://ollama.test")
    try:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await provider.generate("llama", _generation_request())
        assert provider.classify_error(exc_info.value) == ErrorClass.NON_RETRYABLE_AUTH
    finally:
        await provider.aclose()

"""Mock provider unit tests."""

import pytest

from evalharness.core.models import GenerationRequest, Message
from evalharness.providers.mock import MOCK_DIGEST, MockProvider


@pytest.mark.asyncio
async def test_mock_resolves_fixed_digest() -> None:
    provider = MockProvider()
    version = await provider.resolve_version("mock-qa")
    assert version.resolved_version == MOCK_DIGEST


@pytest.mark.asyncio
async def test_mock_answers_synthetic_qa() -> None:
    provider = MockProvider()
    req = GenerationRequest(
        messages=[
            Message(
                role="user",
                content="Answer the following question concisely.\n\nQuestion: What is 3 plus one?\n\nAnswer:\n",
            )
        ],
        max_tokens=32,
        temperature=0.0,
        top_p=None,
        top_k=None,
        seed=42,
        stop=[],
        response_format=None,
        tools=None,
        timeout_s=10.0,
    )
    resp = await provider.generate("mock-qa", req)
    assert resp.text == "4"
    assert resp.total_ms == 5.0

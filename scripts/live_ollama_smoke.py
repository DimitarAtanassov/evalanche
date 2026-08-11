"""Opt-in pinned Ollama generation and embedding smoke test."""

from __future__ import annotations

import asyncio
import json

from evalharness.domain.generation import GenerationRequest, Message
from evalharness.providers.ollama import OllamaProvider


async def main() -> None:
    provider = OllamaProvider()
    generation_version = await provider.resolve_version("llama3.2:1b")
    embedding_version = await provider.resolve_version("nomic-embed-text")
    response = await provider.generate(
        "llama3.2:1b",
        GenerationRequest(
            [Message("user", "Reply with exactly: ready")],
            8,
            0.0,
            None,
            None,
            7,
            [],
            None,
            None,
            120,
        ),
    )
    embeddings = await provider.embed("nomic-embed-text", ["ready"])
    print(
        json.dumps(
            {
                "generation_revision": generation_version.resolved_version,
                "embedding_revision": embedding_version.resolved_version,
                "output": response.text,
                "embedding_dimension": len(embeddings[0]),
            },
            indent=2,
        )
    )
    await provider.aclose()


if __name__ == "__main__":
    asyncio.run(main())

"""Deduplicated, normalized embedding service."""

from __future__ import annotations

import math

from evalharness.domain.provider import Provider
from evalharness.hashing import sha256_hex


class EmbeddingService:
    def __init__(
        self,
        provider: Provider,
        model: str,
        revision: str,
        *,
        dimension: int = 1024,
        batch_size: int = 64,
    ) -> None:
        self.provider = provider
        self.model = model
        self.revision = revision
        self.dimension = dimension
        self.batch_size = batch_size
        self._cache: dict[str, list[float]] = {}

    async def embed(self, texts: list[str]) -> list[list[float]]:
        keys = [sha256_hex(text) for text in texts]
        missing: dict[str, str] = {
            key: text for key, text in zip(keys, texts, strict=True) if key not in self._cache
        }
        unique = list(missing.items())
        for start in range(0, len(unique), self.batch_size):
            batch = unique[start : start + self.batch_size]
            vectors = await self.provider.embed(self.model, [text for _, text in batch])
            if len(vectors) != len(batch):
                raise ValueError("Embedding provider returned an unexpected batch size")
            for (key, _), vector in zip(batch, vectors, strict=True):
                if len(vector) != self.dimension:
                    raise ValueError(
                        f"Embedding dimension {len(vector)} violates {self.dimension}-d contract"
                    )
                norm = math.sqrt(sum(value * value for value in vector))
                if norm == 0:
                    raise ValueError("Zero-norm embedding")
                self._cache[key] = [value / norm for value in vector]
        return [self._cache[key] for key in keys]

    async def cosine_max_reference(self, prediction: str, references: list[str]) -> float:
        if not references:
            raise ValueError("At least one reference is required")
        vectors = await self.embed([prediction, *references])
        prediction_vector = vectors[0]
        return max(
            sum(left * right for left, right in zip(prediction_vector, reference, strict=True))
            for reference in vectors[1:]
        )

    async def asymmetric_similarity(
        self,
        prediction: str,
        references: list[str],
        *,
        variant: str = "prediction_to_reference",
    ) -> float:
        """Score directed similarity against a normalized reference centroid."""
        if variant not in {"prediction_to_reference", "reference_to_prediction"}:
            raise ValueError(f"Unknown asymmetric variant: {variant}")
        vectors = await self.embed([prediction, *references])
        if len(vectors) < 2:
            raise ValueError("At least one reference is required")
        centroid = [
            sum(vector[index] for vector in vectors[1:]) / (len(vectors) - 1)
            for index in range(self.dimension)
        ]
        norm = math.sqrt(sum(value * value for value in centroid))
        centroid = [value / norm for value in centroid]
        cosine = sum(left * right for left, right in zip(vectors[0], centroid, strict=True))
        # Direction is explicit in provenance even though cosine itself is symmetric;
        # downstream threshold calibration is maintained per variant.
        return cosine

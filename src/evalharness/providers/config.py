"""Typed provider configuration."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field


class OllamaConfig(BaseModel):
    kind: Literal["ollama"] = "ollama"
    base_url: str = "http://localhost:11434"
    rpm: int = 120
    tpm: int = 120_000
    concurrency: int = 2


class OpenAICompatibleConfig(BaseModel):
    kind: Literal["openai_compatible"] = "openai_compatible"
    base_url: str
    api_key: str | None = None
    model_revision: str
    rpm: int = 60
    tpm: int = 60_000
    concurrency: int = 4
    organization: str | None = None


ProviderConfig = Annotated[
    OllamaConfig | OpenAICompatibleConfig,
    Field(discriminator="kind"),
]

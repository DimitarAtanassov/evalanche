"""Provider registry via entry points."""

from __future__ import annotations

from importlib.metadata import entry_points

from evalharness.domain.provider import Provider
from evalharness.providers.config import OllamaConfig, OpenAICompatibleConfig, ProviderConfig
from evalharness.providers.ollama import OllamaProvider
from evalharness.providers.openai_compatible import OpenAICompatibleProvider
from evalharness.providers.runtime import ManagedProvider


def load_provider(name: str, **kwargs: object) -> Provider:
    eps = entry_points(group="evalharness.providers")
    for ep in eps:
        if ep.name == name:
            provider_cls = ep.load()
            return provider_cls(**kwargs)  # type: ignore[no-any-return]
    available = [ep.name for ep in eps]
    raise ValueError(f"Unknown provider '{name}'. Available: {available}")


def create_provider(config: ProviderConfig) -> Provider:
    if isinstance(config, OllamaConfig):
        provider: Provider = OllamaProvider(base_url=config.base_url)
    elif isinstance(config, OpenAICompatibleConfig):
        provider = OpenAICompatibleProvider(
            base_url=config.base_url,
            api_key=config.api_key,
            organization=config.organization,
            model_revision=config.model_revision,
        )
    else:  # pragma: no cover - discriminated union exhaustiveness
        raise TypeError(f"Unsupported provider config: {type(config).__name__}")
    return ManagedProvider(
        provider,
        rpm=config.rpm,
        tpm=config.tpm,
        concurrency=config.concurrency,
    )

"""Provider policy shared by the live judge and rag services."""

from __future__ import annotations

from evalharness.app.settings import Settings
from evalharness.providers.call_policy import ProviderCallPolicy

_LIVE_SCORING_PROVIDERS = frozenset({"ollama", "openai_compatible"})


def require_live_scoring_provider(provider_name: str) -> None:
    """Reject provider names the live scoring paths carry no rate-limit configuration for.

    The run path deliberately falls through to the entry-point registry, so the guard
    lives here rather than in ``build_managed_provider``.
    """
    if provider_name not in _LIVE_SCORING_PROVIDERS:
        raise ValueError(
            f"live scoring provider must be ollama or openai_compatible, got {provider_name!r}"
        )


def provider_call_policy(settings: Settings, request_timeout_s: float | None) -> ProviderCallPolicy:
    """Per-call timeout and retry budget for a live scoring provider."""
    return ProviderCallPolicy(
        request_timeout_s=request_timeout_s or settings.default_request_timeout_s,
        max_retries=settings.default_max_retries,
        retry_base_s=settings.default_retry_base_s,
        retry_cap_s=settings.default_retry_cap_s,
    )

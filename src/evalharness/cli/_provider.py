"""Provider call policy and teardown shared by the judge and rag live command paths."""

from __future__ import annotations

from evalharness.config import get_settings
from evalharness.core.protocols import Provider
from evalharness.providers.call_policy import ProviderCallPolicy

_LIVE_SCORING_PROVIDERS = frozenset({"ollama", "openai_compatible"})


def _require_live_scoring_provider(provider_name: str) -> None:
    """Reject provider names the live paths carry no rate-limit configuration for.

    The run path deliberately falls through to the entry-point registry, so the guard
    lives here rather than in ``build_managed_provider``.
    """
    if provider_name not in _LIVE_SCORING_PROVIDERS:
        raise ValueError(
            f"live scoring provider must be ollama or openai_compatible, got {provider_name!r}"
        )


def _provider_call_policy(request_timeout_s: float | None) -> ProviderCallPolicy:
    settings = get_settings()
    return ProviderCallPolicy(
        request_timeout_s=request_timeout_s or settings.default_request_timeout_s,
        max_retries=settings.default_max_retries,
        retry_base_s=settings.default_retry_base_s,
        retry_cap_s=settings.default_retry_cap_s,
    )


async def _close_provider(provider: Provider) -> None:
    await provider.aclose()

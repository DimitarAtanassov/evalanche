"""Provider call policy and teardown shared by the judge and rag live command paths."""

from __future__ import annotations

from evalharness.config import get_settings
from evalharness.core.protocols import Provider
from evalharness.providers.call_policy import ProviderCallPolicy


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

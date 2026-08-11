"""RAG evidence use cases: offline artifact build and provider-backed NLI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evalharness.app.settings import Settings
from evalharness.providers.factory import ProviderBuilder
from evalharness.rag import build_rag_evidence as _build_rag_evidence
from evalharness.rag.live import build_live_rag_evidence as _rag_live
from evalharness.services._provider import provider_call_policy, require_live_scoring_provider


class RagService:
    """RAG evidence artifacts from a run report plus local evidence JSONL."""

    def __init__(
        self,
        *,
        settings: Settings,
        build_provider: ProviderBuilder,
    ) -> None:
        self._settings = settings
        self._build_provider = build_provider

    def build_rag_evidence(
        self,
        *,
        report_path: Path,
        evidence_path: Path,
        output_path: Path,
        nli_provider: str | None = None,
        nli_model: str | None = None,
        nli_responses_path: Path | None = None,
    ) -> dict[str, Any]:
        """Build evidence with the deterministic mock NLI path."""
        return _build_rag_evidence(
            report_path=report_path,
            evidence_path=evidence_path,
            output_path=output_path,
            nli_provider=nli_provider,
            nli_model=nli_model,
            nli_responses_path=nli_responses_path,
        )

    async def build_live_rag_evidence(
        self,
        *,
        report_path: Path,
        evidence_path: Path,
        output_path: Path,
        provider_name: str,
        nli_model: str,
        concurrency: int,
        request_timeout_s: float | None,
    ) -> dict[str, Any]:
        """Build the evidence artifact with live NLI, closing the provider before returning.

        Raises ``ValueError`` when the provider name carries no live rate-limit
        configuration.
        """
        require_live_scoring_provider(provider_name)
        provider = self._build_provider(
            provider_name,
            concurrency=concurrency,
            rpm=self._settings.nli_provider_rpm,
            tpm=self._settings.nli_provider_tpm,
        )
        try:
            return await _rag_live(
                report_path=report_path,
                evidence_path=evidence_path,
                output_path=output_path,
                provider=provider,
                nli_model=nli_model,
                concurrency=concurrency,
                policy=provider_call_policy(self._settings, request_timeout_s),
            )
        finally:
            await provider.aclose()

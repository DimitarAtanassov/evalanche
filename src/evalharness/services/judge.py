"""Judge use cases: offline judgment, live judgment, and holdout calibration."""

from __future__ import annotations

from pathlib import Path

from evalharness.app.settings import Settings
from evalharness.artifacts.calibration import CalibrationArtifact
from evalharness.judge import attach_calibration as _attach_calibration
from evalharness.judge import run_judgment as _run_judgment
from evalharness.judge import validate_calibration as _validate_calibration
from evalharness.judge.live import run_live_judgment as _judge_live
from evalharness.judge.models import JudgeMode, JudgmentArtifact
from evalharness.providers.factory import ProviderBuilder
from evalharness.services._provider import provider_call_policy, require_live_scoring_provider


class JudgeService:
    """LLM-as-judge: offline fixtures, live providers, and holdout calibration."""

    def __init__(
        self,
        *,
        settings: Settings,
        build_provider: ProviderBuilder,
    ) -> None:
        self._settings = settings
        self._build_provider = build_provider

    def run_judgment(
        self,
        *,
        mode: JudgeMode,
        rubric_path: Path,
        candidates_path: Path | None,
        pairs_path: Path | None,
        provider: str,
        model: str,
        judge_family: str,
        candidate_family: str,
        responses_path: Path | None,
        seed: int,
        output_path: Path,
    ) -> JudgmentArtifact:
        """Run the deterministic mock judge path (no live provider)."""
        return _run_judgment(
            mode=mode,
            rubric_path=rubric_path,
            candidates_path=candidates_path,
            pairs_path=pairs_path,
            provider=provider,
            model=model,
            judge_family=judge_family,
            candidate_family=candidate_family,
            responses_path=responses_path,
            seed=seed,
            output_path=output_path,
        )

    async def run_live_judgment(
        self,
        *,
        mode: JudgeMode,
        rubric_path: Path,
        candidates_path: Path | None,
        pairs_path: Path | None,
        provider_name: str,
        model: str,
        judge_family: str,
        candidate_family: str,
        seed: int,
        output_path: Path,
        concurrency: int,
        request_timeout_s: float | None,
    ) -> JudgmentArtifact:
        """Judge against a live provider, closing the provider before returning.

        Raises ``ValueError`` when the provider name carries no live rate-limit
        configuration.
        """
        require_live_scoring_provider(provider_name)
        provider = self._build_provider(
            provider_name,
            concurrency=concurrency,
            rpm=self._settings.judge_provider_rpm,
            tpm=self._settings.judge_provider_tpm,
        )
        try:
            return await _judge_live(
                mode=mode,
                rubric_path=rubric_path,
                candidates_path=candidates_path,
                pairs_path=pairs_path,
                provider=provider,
                model=model,
                judge_family=judge_family,
                candidate_family=candidate_family,
                seed=seed,
                output_path=output_path,
                concurrency=concurrency,
                policy=provider_call_policy(self._settings, request_timeout_s),
            )
        finally:
            await provider.aclose()

    def validate_calibration(
        self,
        *,
        judgment_path: Path,
        labels_dev_path: Path | None,
        labels_holdout_path: Path,
        rubric_path: Path,
        output_path: Path,
    ) -> CalibrationArtifact:
        """Holdout calibration against human labels."""
        return _validate_calibration(
            judgment_path=judgment_path,
            labels_dev_path=labels_dev_path,
            labels_holdout_path=labels_holdout_path,
            rubric_path=rubric_path,
            output_path=output_path,
        )

    def attach_calibration(
        self,
        *,
        judgment_path: Path,
        calibration_path: Path,
        output_path: Path,
    ) -> JudgmentArtifact:
        """Attach a passing calibration digest onto a judgment artifact."""
        return _attach_calibration(
            judgment_path=judgment_path,
            calibration_path=calibration_path,
            output_path=output_path,
        )

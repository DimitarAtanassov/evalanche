"""File-primary pointwise and pairwise judgment runners."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from evalharness.judge.errors import JudgeError
from evalharness.judge.io import load_jsonl_models, write_json
from evalharness.judge.mock_responses import load_mock_responses
from evalharness.judge.models import (
    JudgeMode,
    JudgeModelIdentity,
    JudgmentArtifact,
    LatencySummary,
    PairwiseOrdering,
    PairwisePair,
    PairwiseSummary,
    PointwiseCandidate,
    Rubric,
)
from evalharness.judge.pairwise import (
    bradley_terry_summary,
    build_pairwise_item,
    position_bias_rate,
    swap_consistency_rate,
)
from evalharness.judge.rubric import load_rubric
from evalharness.judge.text import truncate_evidence, truncate_reasoning

INFORMATIONAL_BLOCK_REASON = (
    "Judgment artifacts are informational until a passing calibration digest is merged"
)
MOCK_DIGEST_PREFIX = "sha256:mock-judge-"


def _mock_resolved_version(model: str) -> str:
    digest = "".join(f"{ord(char):02x}" for char in model)[:64].ljust(64, "0")
    return f"sha256:{digest}"


def _reject_duplicate_case_ids(case_ids: Iterable[str]) -> None:
    """Refuse repeated cases at the input boundary.

    Human labels key on ``case_id``, so a case judged twice has no single judge
    value to pair during calibration and would inflate ``n``.
    """
    seen: set[str] = set()
    for case_id in case_ids:
        if case_id in seen:
            raise JudgeError(
                "DUPLICATE_CASE_ID",
                f"case_id={case_id} appears more than once in the judge input",
            )
        seen.add(case_id)


def run_pointwise(
    *,
    rubric: Rubric,
    candidates: list[PointwiseCandidate],
    responses_path: Path,
    provider: str,
    model: str,
    judge_family: str,
    candidate_family: str,
    seed: int,
) -> JudgmentArtifact:
    """Build a pointwise judgment artifact with gating_allowed always false."""
    if rubric.mode is not JudgeMode.POINTWISE:
        raise JudgeError("INVALID_RUBRIC", "rubric.mode must be pointwise")
    if not judge_family.strip() or not candidate_family.strip():
        raise JudgeError("JUDGE_FAMILY_CONFLICT", "judge and candidate families are required")
    _reject_duplicate_case_ids(candidate.case_id for candidate in candidates)
    pointwise_responses, _ = load_mock_responses(responses_path)
    items: list[dict[str, object]] = []
    for candidate in candidates:
        response = pointwise_responses.get(candidate.generation_id)
        if response is None:
            raise JudgeError(
                "MOCK_RESPONSE_MISSING",
                f"missing mock response for generation_id={candidate.generation_id}",
            )
        if response.score < rubric.scale.min or response.score > rubric.scale.max:
            raise JudgeError(
                "INVALID_MOCK_RESPONSES",
                f"score {response.score} outside rubric scale "
                f"[{rubric.scale.min}, {rubric.scale.max}]",
            )
        if rubric.require_reasoning_before_score and not response.reasoning.strip():
            raise JudgeError(
                "INVALID_MOCK_RESPONSES",
                f"missing required reasoning for generation_id={candidate.generation_id}",
            )
        items.append(
            {
                "case_id": candidate.case_id,
                "generation_id": candidate.generation_id,
                "score": response.score,
                "reasoning": truncate_reasoning(response.reasoning),
                "evidence": {
                    "candidate_text": truncate_evidence(candidate.candidate_text),
                },
                "outcome": None,
            }
        )
    artifact = JudgmentArtifact(
        mode=JudgeMode.POINTWISE,
        rubric_name=rubric.name,
        rubric_version=rubric.version,
        judge_model=JudgeModelIdentity(
            provider=provider,
            model=model,
            resolved_version=_mock_resolved_version(model),
        ),
        candidate_model_family=candidate_family,
        judge_model_family=judge_family,
        gating_allowed=False,
        gating_block_reason=INFORMATIONAL_BLOCK_REASON,
        calibration_digest=None,
        cost_usd_total=0.0,
        latency_ms=LatencySummary(p50=0.0, p95=0.0),
        items=items,
        pairwise_summary=None,
        seed=seed,
    )
    return artifact


def run_pairwise(
    *,
    rubric: Rubric,
    pairs: list[PairwisePair],
    responses_path: Path,
    provider: str,
    model: str,
    judge_family: str,
    candidate_family: str,
    seed: int,
) -> JudgmentArtifact:
    """Build a pairwise judgment artifact with both orderings and BT summary."""
    if rubric.mode is not JudgeMode.PAIRWISE:
        raise JudgeError("INVALID_RUBRIC", "rubric.mode must be pairwise")
    if not judge_family.strip() or not candidate_family.strip():
        raise JudgeError("JUDGE_FAMILY_CONFLICT", "judge and candidate families are required")
    _reject_duplicate_case_ids(pair.case_id for pair in pairs)
    _, pairwise_responses = load_mock_responses(responses_path)
    items: list[dict[str, object]] = []
    pairwise_items = []
    for pair in pairs:
        if pair.a_model_label == pair.b_model_label:
            raise JudgeError("SELF_PAIR", f"case_id={pair.case_id} has identical model labels")
        orderings: list[PairwiseOrdering] = []
        for swap_position in (0, 1):
            response = pairwise_responses.get((pair.case_id, swap_position))
            if response is None:
                raise JudgeError(
                    "MOCK_RESPONSE_MISSING",
                    f"missing mock response for case_id={pair.case_id} "
                    f"swap_position={swap_position}",
                )
            if rubric.require_reasoning_before_score and not response.reasoning.strip():
                raise JudgeError(
                    "INVALID_MOCK_RESPONSES",
                    f"missing required reasoning for case_id={pair.case_id} "
                    f"swap_position={swap_position}",
                )
            orderings.append(
                PairwiseOrdering(
                    swap_position=swap_position,
                    preference=response.preference,
                    reasoning=truncate_reasoning(response.reasoning),
                )
            )
        try:
            item = build_pairwise_item(
                case_id=pair.case_id,
                a_generation_id=pair.a_generation_id,
                b_generation_id=pair.b_generation_id,
                a_model_label=pair.a_model_label,
                b_model_label=pair.b_model_label,
                orderings=orderings,
            )
        except ValueError as exc:
            if str(exc) == "SWAP_INCOMPLETE":
                raise JudgeError("SWAP_INCOMPLETE", f"case_id={pair.case_id}") from exc
            raise
        pairwise_items.append(item)
        items.append(item.model_dump(mode="json"))

    bt = bradley_terry_summary(pairwise_items)
    summary = PairwiseSummary(
        n_pairs=len(pairwise_items),
        swap_consistency=swap_consistency_rate(pairwise_items),
        position_bias=position_bias_rate(pairwise_items),
        bradley_terry=bt,
    )
    artifact = JudgmentArtifact(
        mode=JudgeMode.PAIRWISE,
        rubric_name=rubric.name,
        rubric_version=rubric.version,
        judge_model=JudgeModelIdentity(
            provider=provider,
            model=model,
            resolved_version=_mock_resolved_version(model),
        ),
        candidate_model_family=candidate_family,
        judge_model_family=judge_family,
        gating_allowed=False,
        gating_block_reason=INFORMATIONAL_BLOCK_REASON,
        calibration_digest=None,
        cost_usd_total=0.0,
        latency_ms=LatencySummary(p50=0.0, p95=0.0),
        items=items,
        pairwise_summary=summary,
        seed=seed,
    )
    return artifact


def run_judgment(
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
    """CLI entry for ``evalctl judge run``."""
    if provider != "mock":
        raise JudgeError(
            "PROVIDER_UNSUPPORTED",
            "Phase 6 CI path supports --provider mock only; "
            "live providers are deferred until structured judge parsing ships",
        )
    if responses_path is None:
        raise JudgeError("MISSING_ARTIFACT", "--responses is required when --provider mock")
    rubric = load_rubric(rubric_path)
    if mode is not rubric.mode:
        raise JudgeError(
            "INVALID_RUBRIC",
            f"--mode {mode.value} does not match rubric.mode {rubric.mode.value}",
        )
    if mode is JudgeMode.POINTWISE:
        if candidates_path is None:
            raise JudgeError("MISSING_ARTIFACT", "--candidates is required for pointwise mode")
        candidates = load_jsonl_models(
            candidates_path,
            PointwiseCandidate,
            error_code="INVALID_CANDIDATES",
        )
        artifact = run_pointwise(
            rubric=rubric,
            candidates=candidates,
            responses_path=responses_path,
            provider=provider,
            model=model,
            judge_family=judge_family,
            candidate_family=candidate_family,
            seed=seed,
        )
    else:
        if pairs_path is None:
            raise JudgeError("MISSING_ARTIFACT", "--pairs is required for pairwise mode")
        try:
            pairs = load_jsonl_models(pairs_path, PairwisePair, error_code="INVALID_PAIRS")
        except JudgeError as exc:
            if "SELF_PAIR" in str(exc):
                raise JudgeError("SELF_PAIR", str(exc)) from exc
            raise
        artifact = run_pairwise(
            rubric=rubric,
            pairs=pairs,
            responses_path=responses_path,
            provider=provider,
            model=model,
            judge_family=judge_family,
            candidate_family=candidate_family,
            seed=seed,
        )
    write_json(output_path, artifact)
    return artifact

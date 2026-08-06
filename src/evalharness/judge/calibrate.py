"""Holdout calibration and attach-calibration merge path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from evalharness.hashing import calibration_body_digest, judgment_identity_digest
from evalharness.judge.agreement import compute_agreement
from evalharness.judge.errors import JudgeError
from evalharness.judge.io import read_json, write_json
from evalharness.judge.labels import load_labels
from evalharness.judge.models import (
    AgreementMetric,
    CalibrationArtifact,
    HumanLabel,
    JudgmentArtifact,
    LabelShape,
    LabelSplit,
    RubricCalibration,
    SplitCalibration,
)
from evalharness.judge.rubric import load_rubric

AGREEMENT_UNDEFINED = (
    "AGREEMENT_UNDEFINED: holdout has fewer than two observed categories, so "
    "chance-corrected agreement is undefined"
)


def _load_judgment(path: Path) -> tuple[JudgmentArtifact, dict[str, Any]]:
    payload = read_json(path)
    try:
        artifact = JudgmentArtifact.model_validate(payload)
    except ValidationError as exc:
        raise JudgeError("INVALID_ARTIFACT", f"{path}: {exc}") from exc
    return artifact, payload


def _index_items_by_case(judgment: JudgmentArtifact) -> dict[str, dict[str, Any]]:
    """Index judgment items by ``case_id``, refusing repeats.

    Calibration pairs exactly one human label with one judge value per case. If
    a case appeared many times, item multiplicity would inflate ``n`` and let
    volume alone clear ``min_holdout_n`` (ADR-003 fail-closed gate).
    """
    by_case: dict[str, dict[str, Any]] = {}
    for item in judgment.items:
        case_id = str(item.get("case_id", ""))
        if not case_id:
            raise JudgeError("INVALID_ARTIFACT", "judgment item is missing case_id")
        if case_id in by_case:
            raise JudgeError(
                "DUPLICATE_CASE_ID",
                f"judgment contains more than one item for case_id={case_id}; "
                "calibration requires one judgment per case",
            )
        by_case[case_id] = item
    return by_case


def _extract_judge_values(
    judgment: JudgmentArtifact,
    items_by_case: dict[str, dict[str, Any]],
    labels: list[HumanLabel],
) -> tuple[list[Any], list[Any]]:
    """Pair each labeled case with its single judge value, in label order."""
    human_values: list[Any] = []
    judge_values: list[Any] = []
    for label in labels:
        item = items_by_case.get(label.case_id)
        if item is None:
            continue
        if judgment.mode.value == "pointwise":
            score = item.get("score")
            if score is None:
                continue
            if label.label_shape is not LabelShape.ORDINAL_SCORE:
                raise JudgeError(
                    "RUBRIC_VERSION_MISMATCH",
                    f"case {label.case_id}: expected ordinal_score labels for pointwise judgment",
                )
            human_values.append(label.value)
            judge_values.append(score)
        else:
            preference = item.get("final_preference")
            if preference is None:
                continue
            if label.label_shape is not LabelShape.PREFERENCE:
                raise JudgeError(
                    "RUBRIC_VERSION_MISMATCH",
                    f"case {label.case_id}: expected preference labels for pairwise judgment",
                )
            human_values.append(str(label.value))
            judge_values.append(str(preference))
    return human_values, judge_values


def _validate_label_rubric(labels: list[HumanLabel], judgment: JudgmentArtifact) -> None:
    for label in labels:
        if (
            label.rubric_name != judgment.rubric_name
            or label.rubric_version != judgment.rubric_version
        ):
            raise JudgeError(
                "RUBRIC_VERSION_MISMATCH",
                f"label {label.case_id}: rubric {label.rubric_name}@{label.rubric_version} "
                f"!= judgment {judgment.rubric_name}@{judgment.rubric_version}",
            )


def _family_separation_ok(judgment: JudgmentArtifact) -> bool:
    judge_family = judgment.judge_model_family.strip().lower()
    candidate_family = judgment.candidate_model_family.strip().lower()
    if not judge_family or not candidate_family:
        return False
    return judge_family != candidate_family


def _split_section(
    *,
    labels: list[HumanLabel],
    judgment: JudgmentArtifact,
    items_by_case: dict[str, dict[str, Any]],
    agreement_metric: AgreementMetric,
) -> SplitCalibration:
    human, judge = _extract_judge_values(judgment, items_by_case, labels)
    agreement = compute_agreement(agreement_metric, human, judge)
    return SplitCalibration(
        label_set_id=labels[0].label_set_id,
        n=len(human),
        agreement_metric=agreement_metric,
        agreement=agreement,
        agreement_ci=None,
    )


def _resolve_calibration_config(
    *,
    judgment: JudgmentArtifact,
    rubric_path: Path,
) -> RubricCalibration:
    """Read gate thresholds from the rubric that produced the judgment.

    The rubric is mandatory: defaulting to ``RubricCalibration()`` would clear a
    stricter rubric's threshold at 0.60 whenever the flag was omitted.
    """
    rubric = load_rubric(rubric_path)
    if rubric.name != judgment.rubric_name or rubric.version != judgment.rubric_version:
        raise JudgeError(
            "RUBRIC_VERSION_MISMATCH",
            f"rubric {rubric.name}@{rubric.version} != "
            f"judgment {judgment.rubric_name}@{judgment.rubric_version}",
        )
    return rubric.calibration


def validate_calibration(
    *,
    judgment_path: Path,
    labels_dev_path: Path | None,
    labels_holdout_path: Path,
    rubric_path: Path,
    output_path: Path,
) -> CalibrationArtifact:
    """Compute agreement on holdout only and write ``calibration.json``."""
    judgment, judgment_payload = _load_judgment(judgment_path)
    items_by_case = _index_items_by_case(judgment)
    holdout = load_labels(labels_holdout_path, expected_split=LabelSplit.HOLDOUT)
    _validate_label_rubric(holdout, judgment)

    calibration_cfg = _resolve_calibration_config(judgment=judgment, rubric_path=rubric_path)
    holdout_section = _split_section(
        labels=holdout,
        judgment=judgment,
        items_by_case=items_by_case,
        agreement_metric=calibration_cfg.agreement_metric,
    )

    block_reasons: list[str] = []
    dev_section: SplitCalibration | None = None
    if labels_dev_path is None:
        block_reasons.append("missing --labels-dev")
    else:
        dev_labels = load_labels(labels_dev_path, expected_split=LabelSplit.DEV)
        _validate_label_rubric(dev_labels, judgment)
        if dev_labels[0].label_set_id == holdout[0].label_set_id:
            raise JudgeError(
                "INVALID_LABELS",
                "dev and holdout label_set_id values must be distinct",
            )
        dev_section = _split_section(
            labels=dev_labels,
            judgment=judgment,
            items_by_case=items_by_case,
            agreement_metric=calibration_cfg.agreement_metric,
        )
        if dev_section.n < calibration_cfg.min_dev_n:
            block_reasons.append(
                f"n_dev={dev_section.n} below min_dev_n={calibration_cfg.min_dev_n}"
            )

    family_ok = _family_separation_ok(judgment)
    if not family_ok:
        block_reasons.append("JUDGE_FAMILY_CONFLICT")

    if holdout_section.n < calibration_cfg.min_holdout_n:
        block_reasons.append(
            f"n_holdout={holdout_section.n} below min_holdout_n={calibration_cfg.min_holdout_n}"
        )

    agreement_holdout = holdout_section.agreement
    if agreement_holdout is None:
        block_reasons.append(
            AGREEMENT_UNDEFINED
            if holdout_section.n
            else "holdout agreement unavailable: no labelled case paired with a judgment"
        )
    elif agreement_holdout < calibration_cfg.agreement_threshold:
        block_reasons.append(
            f"agreement_holdout={agreement_holdout:.4f} below "
            f"threshold={calibration_cfg.agreement_threshold}"
        )

    gating_allowed = (
        labels_dev_path is not None
        and family_ok
        and holdout_section.n >= calibration_cfg.min_holdout_n
        and agreement_holdout is not None
        and agreement_holdout >= calibration_cfg.agreement_threshold
        and dev_section is not None
        and dev_section.n >= calibration_cfg.min_dev_n
    )

    if gating_allowed:
        plain = (
            f"Holdout agreement {agreement_holdout:.3f} meets threshold "
            f"{calibration_cfg.agreement_threshold} with n={holdout_section.n}; "
            "family separation ok. Gating allowed for downstream consumers."
        )
        block_reasons = []
    else:
        plain = "Judge remains informational. " + (
            "; ".join(block_reasons)
            if block_reasons
            else "holdout calibration did not clear the gate"
        )

    body: dict[str, Any] = {
        "schema_version": "0.1",
        "judgment_digest": judgment_identity_digest(judgment_payload),
        "rubric_name": judgment.rubric_name,
        "rubric_version": judgment.rubric_version,
        "holdout": holdout_section.model_dump(mode="json"),
        "dev": None if dev_section is None else dev_section.model_dump(mode="json"),
        "threshold": calibration_cfg.agreement_threshold,
        "min_holdout_n": calibration_cfg.min_holdout_n,
        "min_dev_n": calibration_cfg.min_dev_n,
        "family_separation_ok": family_ok,
        "gating_allowed": gating_allowed,
        "plain_language": plain,
        "block_reasons": block_reasons,
    }
    digest = calibration_body_digest(body)
    artifact = CalibrationArtifact.model_validate({**body, "calibration_digest": digest})
    write_json(output_path, artifact)
    return artifact


def attach_calibration(
    *,
    judgment_path: Path,
    calibration_path: Path,
    output_path: Path,
) -> JudgmentArtifact:
    """Merge a passing calibration into a new judgment artifact."""
    judgment, judgment_payload = _load_judgment(judgment_path)
    calibration_payload = read_json(calibration_path)
    try:
        calibration = CalibrationArtifact.model_validate(calibration_payload)
    except ValidationError as exc:
        raise JudgeError("INVALID_ARTIFACT", f"{calibration_path}: {exc}") from exc

    if calibration.calibration_digest != calibration_body_digest(calibration_payload):
        raise JudgeError(
            "INVALID_ARTIFACT",
            f"{calibration_path}: calibration_digest does not match artifact body",
        )

    expected = judgment_identity_digest(judgment_payload)
    if calibration.judgment_digest != expected:
        raise JudgeError(
            "CALIBRATION_JUDGMENT_MISMATCH",
            f"calibration judgment_digest {calibration.judgment_digest} "
            f"!= judgment digest {expected}",
        )
    if not calibration.gating_allowed:
        raise JudgeError(
            "UNCALIBRATED_JUDGE",
            "calibration.gating_allowed is false; refuse attach",
        )

    updated = judgment.model_copy(
        update={
            "gating_allowed": True,
            "calibration_digest": calibration.calibration_digest,
            "gating_block_reason": calibration.plain_language,
        }
    )
    write_json(output_path, updated)
    return updated

"""Dataset contract and harness-correctness tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from evalharness.core.enums import FailureOutcome
from evalharness.core.models import Case, Generation
from evalharness.datasets import DatasetManifestError, DatasetTier, load_dataset, validate_dataset
from evalharness.scoring.engine import ScoringEngine
from tools.datasets import MaterializationError, materialize_dataset

DATASET_ROOT = Path("fixtures/datasets")
SOURCE_ROOT = Path("tools/datasets/sources")
SMOKE_DATASETS = tuple(sorted(DATASET_ROOT.glob("*-smoke")))


def test_all_committed_smokes_validate_and_match_source_digest() -> None:
    assert len(SMOKE_DATASETS) == 8

    for dataset_path in SMOKE_DATASETS:
        bundle = load_dataset(dataset_path)
        report = validate_dataset(bundle)

        assert report.errors == []
        assert bundle.manifest.source is not None
        source_path = SOURCE_ROOT / Path(bundle.manifest.source.canonical_url).name
        source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
        assert bundle.manifest.source.revision_digest == f"sha256:{source_digest}"


def test_legacy_unknown_key_warns_without_breaking_compatibility(tmp_path: Path) -> None:
    source = Path("fixtures/sample_dataset")
    dataset_path = tmp_path / "legacy"
    dataset_path.mkdir()
    (dataset_path / "cases.jsonl").write_bytes((source / "cases.jsonl").read_bytes())
    manifest = yaml.safe_load((source / "manifest.yaml").read_text(encoding="utf-8"))
    manifest["legacy_note"] = "ignored"
    (dataset_path / "manifest.yaml").write_text(
        yaml.safe_dump(manifest),
        encoding="utf-8",
    )

    report = validate_dataset(load_dataset(dataset_path))

    assert report.valid
    assert report.warnings == ["Unknown legacy manifest keys ignored: legacy_note"]


def test_versioned_unknown_key_is_rejected(tmp_path: Path) -> None:
    source = DATASET_ROOT / "synthetic-qa-smoke"
    dataset_path = tmp_path / "versioned"
    dataset_path.mkdir()
    (dataset_path / "cases.jsonl").write_bytes((source / "cases.jsonl").read_bytes())
    manifest = yaml.safe_load((source / "manifest.yaml").read_text(encoding="utf-8"))
    manifest["untyped_extension"] = True
    (dataset_path / "manifest.yaml").write_text(
        yaml.safe_dump(manifest),
        encoding="utf-8",
    )

    with pytest.raises(DatasetManifestError, match="UNKNOWN_MANIFEST_KEY"):
        load_dataset(dataset_path)


def test_synthetic_materialization_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    arguments = {
        "adapter_name": "synthetic_qa",
        "source": SOURCE_ROOT / "synthetic_qa.jsonl",
        "seed": 42,
        "size": 5,
        "tier": DatasetTier.SMOKE,
        "check_deterministic": True,
    }

    materialize_dataset(output=first, **arguments)
    materialize_dataset(output=second, **arguments)

    assert (first / "cases.jsonl").read_bytes() == (second / "cases.jsonl").read_bytes()
    assert (first / "manifest.yaml").read_bytes() == (second / "manifest.yaml").read_bytes()


def test_external_digest_mismatch_writes_nothing(tmp_path: Path) -> None:
    source = tmp_path / "dev-v1.1.json"
    source.write_text('{"data":[]}', encoding="utf-8")
    pin = {
        "revision": "dev-v1.1",
        "revision_digest": f"sha256:{'0' * 64}",
        "canonical_url": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
    }
    source.with_name(f"{source.name}.pin.yaml").write_text(
        yaml.safe_dump(pin),
        encoding="utf-8",
    )
    output = tmp_path / "output"

    with pytest.raises(MaterializationError, match="SOURCE_DIGEST_MISMATCH"):
        materialize_dataset(
            adapter_name="squad_v1_1",
            source=source,
            output=output,
            seed=42,
            size=5,
            tier=DatasetTier.SMOKE,
        )

    assert not output.exists()


def test_cache_only_adapter_cannot_write_to_committed_fixtures() -> None:
    output = DATASET_ROOT / "forbidden-cache-only-test"
    assert not output.exists()

    with pytest.raises(MaterializationError, match="LICENSE_BLOCK"):
        materialize_dataset(
            adapter_name="financial_phrasebank",
            source=SOURCE_ROOT / "synthetic_qa.jsonl",
            output=output,
            seed=42,
            size=5,
            tier=DatasetTier.SMOKE,
        )

    assert not output.exists()


def _perfect_output(case: Case) -> str:
    if case.expected_label is not None:
        return case.expected_label
    if case.expected_json is not None:
        return json.dumps(case.expected_json, sort_keys=True)
    if case.qrels is not None:
        return json.dumps(
            [document_id for document_id, relevance in case.qrels.items() if relevance > 0]
        )
    if case.reference_answer is not None:
        return case.reference_answer
    raise AssertionError(f"No expected output for {case.external_id}")


def _generation(case: Case) -> Generation:
    return Generation(
        id=None,
        run_id="dataset-smoke",
        case_external_id=case.external_id,
        repeat_idx=0,
        output=_perfect_output(case),
        tool_calls=[],
        finish_reason=None,
        outcome=FailureOutcome.PASSED,
        prompt_tokens=None,
        completion_tokens=None,
        cost_usd=0.0,
        ttft_ms=None,
        total_ms=None,
        queue_wait_ms=None,
        attempts=1,
        attempt_log=[],
        cached=False,
        raw_response=None,
        trace_id=None,
    )


@pytest.mark.parametrize("dataset_path", SMOKE_DATASETS, ids=lambda path: path.name)
def test_committed_smoke_has_non_vacuous_task_fit_scores(dataset_path: Path) -> None:
    bundle = load_dataset(dataset_path)
    assert bundle.manifest.task_metrics is not None
    engine = ScoringEngine()

    scores = engine.score_one(
        _generation(bundle.cases[0]),
        bundle.cases[0],
        bundle.manifest.task_metrics,
    )

    assert {score.metric_name for score in scores} == set(bundle.manifest.task_metrics)
    assert {score.metric_name: score.value for score in scores} == {
        name: 1.0 for name in bundle.manifest.task_metrics
    }
    assert all(score.passed is True for score in scores)

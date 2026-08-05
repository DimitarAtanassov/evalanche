"""External adapter pin success and field-limit fail-closed materialization."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from evalharness.datasets import DatasetTier, load_dataset, validate_dataset
from evalharness.datasets.validator import INPUT_TEXT_LIMIT, REFERENCE_TEXT_LIMIT
from tools.datasets import MaterializationError, materialize_dataset
from tools.datasets.adapters import ADAPTERS, AdapterSpec, synthetic_spec


def _write_pin(source: Path, *, revision: str, canonical_url: str) -> None:
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    source.with_name(f"{source.name}.pin.yaml").write_text(
        yaml.safe_dump(
            {
                "revision": revision,
                "revision_digest": f"sha256:{digest}",
                "canonical_url": canonical_url,
            }
        ),
        encoding="utf-8",
    )


def _squad_article(*, context: str, count: int = 5, start: int = 0) -> dict[str, object]:
    """One SQuAD article whose paragraphs all carry the same context."""
    paragraphs = [
        {
            "context": context,
            "qas": [
                {
                    "id": f"q{index}",
                    "question": f"Question {index}?",
                    "answers": [{"text": f"answer-{index}", "answer_start": 0}],
                }
            ],
        }
        for index in range(start, start + count)
    ]
    return {"title": "Fixture", "paragraphs": paragraphs}


def _squad_source(*articles: dict[str, object]) -> str:
    return json.dumps({"data": list(articles)})


def test_squad_pin_materialize_succeeds_outside_fixtures(tmp_path: Path) -> None:
    source = tmp_path / "dev-v1.1.json"
    source.write_text(
        _squad_source(_squad_article(context="Short context for smoke.", count=5)),
        encoding="utf-8",
    )
    _write_pin(
        source,
        revision="dev-v1.1",
        canonical_url="https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
    )
    output = tmp_path / "cache" / "squad-smoke"

    materialize_dataset(
        adapter_name="squad_v1_1",
        source=source,
        output=output,
        seed=42,
        size=5,
        tier=DatasetTier.SMOKE,
        check_deterministic=True,
    )

    bundle = load_dataset(output)
    report = validate_dataset(bundle)
    assert report.valid
    assert bundle.manifest.source is not None
    assert bundle.manifest.source.redistributable_smoke is False
    assert bundle.manifest.pii_scrubbed is False
    assert all("context" in case.inputs for case in bundle.cases)
    assert all(len(str(case.inputs["context"])) <= INPUT_TEXT_LIMIT for case in bundle.cases)


def test_squad_source_of_only_oversize_contexts_fails_closed_without_output(
    tmp_path: Path,
) -> None:
    source = tmp_path / "dev-v1.1.json"
    source.write_text(
        _squad_source(_squad_article(context="C" * (INPUT_TEXT_LIMIT + 1), count=5)),
        encoding="utf-8",
    )
    _write_pin(
        source,
        revision="dev-v1.1",
        canonical_url="https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
    )
    output = tmp_path / "cache" / "squad-too-long"

    with pytest.raises(MaterializationError, match="SOURCE_TOO_SMALL") as exc_info:
        materialize_dataset(
            adapter_name="squad_v1_1",
            source=source,
            output=output,
            seed=42,
            size=5,
            tier=DatasetTier.SMOKE,
        )

    assert "0 of 5 source records" in str(exc_info.value)
    assert not output.exists()


def test_squad_samples_only_in_bound_records_from_mixed_source(tmp_path: Path) -> None:
    """Long-context records leave the sampling pool instead of aborting the pack."""
    source = tmp_path / "dev-v1.1.json"
    source.write_text(
        _squad_source(
            _squad_article(context="C" * (INPUT_TEXT_LIMIT + 1), count=20),
            _squad_article(context="Short context for smoke.", count=5, start=20),
        ),
        encoding="utf-8",
    )
    _write_pin(
        source,
        revision="dev-v1.1",
        canonical_url="https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
    )
    output = tmp_path / "cache" / "squad-mixed"

    materialize_dataset(
        adapter_name="squad_v1_1",
        source=source,
        output=output,
        seed=42,
        size=5,
        tier=DatasetTier.SMOKE,
        check_deterministic=True,
    )

    bundle = load_dataset(output)
    assert validate_dataset(bundle).valid
    assert {case.external_id for case in bundle.cases} == {f"q{index}" for index in range(20, 25)}
    assert all(case.inputs["context"] == "Short context for smoke." for case in bundle.cases)


def test_pubmedqa_source_of_only_oversize_contexts_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "pubmedqa.json"
    records = {
        f"pmqa-{index}": {
            "QUESTION": f"Is claim {index} supported?",
            "CONTEXT": ["X" * (INPUT_TEXT_LIMIT + 1)],
            "final_decision": "yes",
        }
        for index in range(5)
    }
    source.write_text(json.dumps(records), encoding="utf-8")
    _write_pin(
        source,
        revision="operator-pinned",
        canonical_url="https://example.invalid/pubmedqa.json",
    )
    output = tmp_path / "cache" / "pubmedqa-too-long"

    with pytest.raises(MaterializationError, match="SOURCE_TOO_SMALL") as exc_info:
        materialize_dataset(
            adapter_name="pubmedqa",
            source=source,
            output=output,
            seed=42,
            size=5,
            tier=DatasetTier.SMOKE,
        )

    assert "0 of 5 source records" in str(exc_info.value)
    assert not output.exists()


def test_finqa_source_of_only_oversize_contexts_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "finqa.json"
    records = [
        {
            "id": f"finqa-{index}",
            "pre_text": ["Y" * (INPUT_TEXT_LIMIT + 1)],
            "post_text": [],
            "table": [],
            "qa": {"question": f"What is value {index}?", "answer": str(index)},
        }
        for index in range(5)
    ]
    source.write_text(json.dumps(records), encoding="utf-8")
    _write_pin(
        source,
        revision="operator-pinned",
        canonical_url="https://example.invalid/finqa.json",
    )
    output = tmp_path / "cache" / "finqa-too-long"

    with pytest.raises(MaterializationError, match="SOURCE_TOO_SMALL") as exc_info:
        materialize_dataset(
            adapter_name="finqa",
            source=source,
            output=output,
            seed=42,
            size=5,
            tier=DatasetTier.SMOKE,
        )

    assert "0 of 5 source records" in str(exc_info.value)
    assert not output.exists()


def test_summaries_drops_oversize_articles_and_references(tmp_path: Path) -> None:
    source = tmp_path / "cnn.jsonl"
    rows = [
        {
            "id": f"long-doc-{index}",
            "document": "D" * (INPUT_TEXT_LIMIT + 1),
            "summary": "Short summary.",
        }
        for index in range(5)
    ]
    rows += [
        {
            "id": f"long-summary-{index}",
            "document": "Short document.",
            "summary": "S" * (REFERENCE_TEXT_LIMIT + 1),
        }
        for index in range(5)
    ]
    rows += [
        {
            "id": f"bounded-{index}",
            "document": f"Fictional notice {index}.",
            "summary": f"Notice {index}.",
        }
        for index in range(5)
    ]
    source.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    _write_pin(
        source,
        revision="operator-pinned",
        canonical_url="https://example.invalid/cnn.jsonl",
    )
    output = tmp_path / "cache" / "cnn-smoke"

    materialize_dataset(
        adapter_name="cnn_dailymail",
        source=source,
        output=output,
        seed=42,
        size=5,
        tier=DatasetTier.SMOKE,
    )

    bundle = load_dataset(output)
    assert validate_dataset(bundle).valid
    assert {case.external_id for case in bundle.cases} == {f"bounded-{index}" for index in range(5)}


def test_nondeterministic_adapter_check_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "flip.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"flip-{index}",
                    "task_type": "qa_short",
                    "inputs": {"question": f"Q{index}?"},
                    "reference_answer": str(index),
                    "slices": {"domain": "test"},
                },
                sort_keys=True,
            )
            for index in range(5)
        )
        + "\n",
        encoding="utf-8",
    )
    calls = {"n": 0}

    def flaky_parser(path: Path) -> list[dict[str, object]]:
        calls["n"] += 1
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if calls["n"] > 1:
                record["reference_answer"] = f"changed-{record['reference_answer']}"
            records.append(record)
        return records

    base = synthetic_spec("synthetic_qa", ("squad_f1", "exact_match"))
    flaky = AdapterSpec(
        name=base.name,
        version=base.version,
        source_id=base.source_id,
        source_revision=base.source_revision,
        dataset_name=base.dataset_name,
        dataset_version=base.dataset_version,
        license=base.license,
        redistributable_smoke=base.redistributable_smoke,
        attribution=base.attribution,
        pii_scrubbed=base.pii_scrubbed,
        contamination_risk=base.contamination_risk,
        pii_scrub_procedure=base.pii_scrub_procedure,
        task_metrics=base.task_metrics,
        slices=base.slices,
        created_at=base.created_at,
        parser=flaky_parser,
        requires_external_pin=False,
        canonical_url=None,
    )
    monkeypatch.setitem(ADAPTERS, "synthetic_qa", flaky)
    output = tmp_path / "flip-out"

    with pytest.raises(MaterializationError, match="NONDETERMINISTIC_ADAPTER") as exc_info:
        materialize_dataset(
            adapter_name="synthetic_qa",
            source=source,
            output=output,
            seed=42,
            size=5,
            tier=DatasetTier.SMOKE,
            check_deterministic=True,
        )

    assert exc_info.value.code == "NONDETERMINISTIC_ADAPTER"
    assert not output.exists()

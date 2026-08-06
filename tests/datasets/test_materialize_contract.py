"""Materialization contract: digests, determinism, tiers, hermetic imports."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import yaml

from evalharness.datasets import DatasetTier, load_dataset, validate_dataset
from evalharness.hashing import sha256_hex
from tests.datasets._helpers import SOURCE_ROOT
from tools.datasets import MaterializationError, materialize_dataset


def test_matching_digest_emits_valid_pack_with_provenance(tmp_path: Path) -> None:
    output = tmp_path / "synthetic-qa-out"
    materialize_dataset(
        adapter_name="synthetic_qa",
        source=SOURCE_ROOT / "synthetic_qa.jsonl",
        output=output,
        seed=42,
        size=5,
        tier=DatasetTier.SMOKE,
        check_deterministic=True,
    )

    bundle = load_dataset(output)
    report = validate_dataset(bundle)

    assert report.valid
    assert bundle.manifest.schema_version == "0.1"
    assert bundle.manifest.adapter is not None
    assert bundle.manifest.adapter.name == "synthetic_qa"
    assert bundle.manifest.adapter.sample_seed == 42
    assert bundle.manifest.adapter.sample_size == 5
    assert bundle.manifest.source is not None
    assert bundle.manifest.source.revision_digest.startswith("sha256:")
    assert bundle.manifest.task_metrics == ["squad_f1", "exact_match"]
    assert bundle.manifest.tier is DatasetTier.SMOKE
    assert all(
        {
            "source_id",
            "source_record_id",
            "source_revision",
            "adapter_name",
            "adapter_version",
        }
        <= set(case.provenance)
        for case in bundle.cases
    )


def test_byte_deterministic_materialization_matches_content_sha256(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    arguments = {
        "adapter_name": "synthetic_math",
        "source": SOURCE_ROOT / "synthetic_math.jsonl",
        "seed": 7,
        "size": 5,
        "tier": DatasetTier.SMOKE,
        "check_deterministic": True,
    }

    materialize_dataset(output=first, **arguments)
    materialize_dataset(output=second, **arguments)

    first_cases = (first / "cases.jsonl").read_bytes()
    second_cases = (second / "cases.jsonl").read_bytes()
    assert first_cases == second_cases
    assert (first / "manifest.yaml").read_bytes() == (second / "manifest.yaml").read_bytes()

    first_bundle = load_dataset(first)
    expected = sha256_hex(first_cases.rstrip(b"\n"))
    assert first_bundle.content_sha256 == expected
    assert first_bundle.manifest.content_sha256 == expected


def test_smoke_tier_rejects_oversized_request(tmp_path: Path) -> None:
    with pytest.raises(MaterializationError, match="TIER_SIZE_INVALID"):
        materialize_dataset(
            adapter_name="synthetic_qa",
            source=SOURCE_ROOT / "synthetic_qa.jsonl",
            output=tmp_path / "too-big",
            seed=42,
            size=500,
            tier=DatasetTier.SMOKE,
        )


def test_smoke_tier_rejects_undersized_request(tmp_path: Path) -> None:
    with pytest.raises(MaterializationError, match="TIER_SIZE_INVALID"):
        materialize_dataset(
            adapter_name="synthetic_qa",
            source=SOURCE_ROOT / "synthetic_qa.jsonl",
            output=tmp_path / "too-small",
            seed=42,
            size=4,
            tier=DatasetTier.SMOKE,
        )


def test_output_exists_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    materialize_dataset(
        adapter_name="synthetic_qa",
        source=SOURCE_ROOT / "synthetic_qa.jsonl",
        output=output,
        seed=42,
        size=5,
        tier=DatasetTier.SMOKE,
    )

    with pytest.raises(MaterializationError, match="OUTPUT_EXISTS"):
        materialize_dataset(
            adapter_name="synthetic_qa",
            source=SOURCE_ROOT / "synthetic_qa.jsonl",
            output=output,
            seed=42,
            size=5,
            tier=DatasetTier.SMOKE,
        )


def test_external_digest_mismatch_leaves_output_absent(tmp_path: Path) -> None:
    source = tmp_path / "dev-v1.1.json"
    source.write_text('{"data":[]}', encoding="utf-8")
    pin = {
        "revision": "dev-v1.1",
        "revision_digest": f"sha256:{'a' * 64}",
        "canonical_url": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
    }
    source.with_name(f"{source.name}.pin.yaml").write_text(
        yaml.safe_dump(pin),
        encoding="utf-8",
    )
    output = tmp_path / "squad-out"

    with pytest.raises(MaterializationError, match="SOURCE_DIGEST_MISMATCH") as exc_info:
        materialize_dataset(
            adapter_name="squad_v1_1",
            source=source,
            output=output,
            seed=42,
            size=5,
            tier=DatasetTier.SMOKE,
        )

    assert exc_info.value.code == "SOURCE_DIGEST_MISMATCH"
    assert not output.exists()


def test_external_pin_rejects_empty_revision_before_materialization(tmp_path: Path) -> None:
    source = tmp_path / "dev-v1.1.json"
    source.write_text('{"data":[]}', encoding="utf-8")
    pin = {
        "revision": "",
        "revision_digest": f"sha256:{sha256_hex(source.read_bytes())}",
        "canonical_url": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json",
    }
    source.with_name(f"{source.name}.pin.yaml").write_text(
        yaml.safe_dump(pin),
        encoding="utf-8",
    )
    output = tmp_path / "squad-out"

    with pytest.raises(MaterializationError, match="revision must be non-empty") as exc_info:
        materialize_dataset(
            adapter_name="squad_v1_1",
            source=source,
            output=output,
            seed=42,
            size=5,
            tier=DatasetTier.SMOKE,
        )

    assert exc_info.value.code == "SOURCE_PIN_INVALID"
    assert not output.exists()


def test_materialize_path_does_not_import_huggingface() -> None:
    banned = ("datasets", "huggingface_hub", "transformers")
    before = {name for name in banned if name in sys.modules}

    importlib.import_module("tools.datasets.materialize")
    importlib.import_module("tools.datasets.adapters")

    after = {name for name in banned if name in sys.modules}
    assert after == before

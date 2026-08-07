"""Cache-only adapter bans vs allowed off-repo materialization."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml
from evaldatasets import MaterializationError, materialize_dataset

from evalharness.datasets import DatasetTier, load_dataset, validate_dataset
from tests.datasets._helpers import SMOKE_ROOT, SOURCE_ROOT


@pytest.mark.parametrize(
    "adapter_name",
    ["financial_phrasebank", "cnn_dailymail", "xsum", "ag_news"],
)
def test_banned_adapters_fail_closed_on_fixtures_path(adapter_name: str) -> None:
    output = SMOKE_ROOT / f"should-not-exist-{adapter_name}"
    assert not output.exists()

    with pytest.raises(MaterializationError) as exc_info:
        materialize_dataset(
            adapter_name=adapter_name,
            source=SOURCE_ROOT / "synthetic_qa.jsonl",
            output=output,
            seed=42,
            size=5,
            tier=DatasetTier.SMOKE,
        )

    assert exc_info.value.code == "LICENSE_BLOCK"
    assert not output.exists()


def test_phrasebank_cache_write_validates_outside_git_fixtures(tmp_path: Path) -> None:
    source = tmp_path / "phrasebank.txt"
    source.write_text(
        "\n".join(
            [
                "The fictional firm raised guidance.@positive",
                "The fictional firm cut jobs.@negative",
                "The fictional firm held a briefing.@neutral",
                "The fictional firm beat estimates.@positive",
                "The fictional firm missed targets.@negative",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    source.with_name(f"{source.name}.pin.yaml").write_text(
        yaml.safe_dump(
            {
                "revision": "operator-pinned",
                "revision_digest": f"sha256:{digest}",
                "canonical_url": "https://example.invalid/phrasebank.txt",
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "cache" / "phrasebank-smoke"

    materialize_dataset(
        adapter_name="financial_phrasebank",
        source=source,
        output=output,
        seed=42,
        size=5,
        tier=DatasetTier.SMOKE,
    )

    bundle = load_dataset(output)
    report = validate_dataset(bundle)
    assert report.valid
    assert bundle.manifest.license == "CC-BY-NC-SA-3.0"
    assert bundle.manifest.source is not None
    assert bundle.manifest.source.redistributable_smoke is False

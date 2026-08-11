"""Shared helpers for dataset tests (public-seam fixtures only)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from evalharness.datasets import load_dataset
from evalharness.domain.dataset import Case
from evalharness.domain.enums import FailureOutcome
from evalharness.domain.generation import Generation

SMOKE_ROOT = Path("fixtures/datasets")
SOURCE_ROOT = Path("packages/evaldatasets/src/evaldatasets/sources")
TEMPLATE_ROOT = Path("fixtures/templates")


def smoke_paths() -> tuple[Path, ...]:
    return tuple(sorted(SMOKE_ROOT.glob("*-smoke")))


def copy_dataset(source: Path, dest: Path) -> Path:
    shutil.copytree(source, dest)
    return dest


def rewrite_manifest(dataset_dir: Path, **updates: Any) -> None:
    manifest_path = dataset_dir / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    for key, value in updates.items():
        if value is _DELETE:
            manifest.pop(key, None)
        else:
            manifest[key] = value
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=True), encoding="utf-8")


def rewrite_source(dataset_dir: Path, **updates: Any) -> None:
    manifest_path = dataset_dir / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    source = dict(manifest["source"])
    source.update(updates)
    manifest["source"] = source
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=True), encoding="utf-8")


def rewrite_cases(dataset_dir: Path, cases: list[dict[str, Any]]) -> None:
    lines = [
        json.dumps(case, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for case in cases
    ]
    (dataset_dir / "cases.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_case_dicts(dataset_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in (dataset_dir / "cases.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def perfect_output(case: Case) -> str:
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


def perfect_generation(case: Case, *, run_id: str = "dataset-mock") -> Generation:
    return Generation(
        id=None,
        run_id=run_id,
        case_external_id=case.external_id,
        repeat_idx=0,
        output=perfect_output(case),
        tool_calls=[],
        finish_reason=None,
        outcome=FailureOutcome.PASSED,
        prompt_tokens=None,
        completion_tokens=None,
        cost_usd=0.0,
        ttft_ms=None,
        total_ms=12.0,
        queue_wait_ms=None,
        attempts=1,
        attempt_log=[],
        cached=False,
        raw_response={"mock": True, "provider": "test"},
        trace_id=None,
    )


def template_for_dataset(dataset_path: Path) -> Path:
    bundle = load_dataset(dataset_path)
    task = bundle.cases[0].task_type.value
    mapping = {
        "qa_short": "qa_short.jinja",
        "classification": "classification.jinja",
        "extraction": "extraction.jinja",
        "summarization": "summarization.jinja",
        "retrieval": "retrieval.jinja",
    }
    if "math" in dataset_path.name or "finance" in dataset_path.name:
        return TEMPLATE_ROOT / "numeric.jinja"
    return TEMPLATE_ROOT / mapping[task]


_DELETE = object()
DELETE = _DELETE

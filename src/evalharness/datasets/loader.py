"""Dataset manifest and JSONL loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from evalharness.core.enums import TaskType
from evalharness.core.models import Case
from evalharness.hashing import sha256_hex


@dataclass(frozen=True)
class DatasetManifest:
    name: str
    version: str
    split: str
    license: str
    pii_scrubbed: bool
    created_at: str
    slices: list[str]
    content_sha256: str | None = None


@dataclass(frozen=True)
class DatasetBundle:
    manifest: DatasetManifest
    cases: list[Case]
    content_sha256: str
    source_path: Path


TASK_REQUIRED_FIELDS: dict[TaskType, list[str]] = {
    TaskType.GENERATION: [],
    TaskType.CLASSIFICATION: ["expected_label"],
    TaskType.EXTRACTION: ["expected_json"],
    TaskType.SUMMARIZATION: ["reference_answer"],
    TaskType.QA_SHORT: ["reference_answer"],
    TaskType.RETRIEVAL: ["qrels"],
    TaskType.RAG: ["reference_answer", "qrels"],
    TaskType.TOOL_USE: ["expected_json"],
    TaskType.AGENT_TRAJECTORY: [],
    TaskType.SAFETY: [],
    TaskType.PAIRWISE: [],
}


def _normalize_prompt(inputs: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in sorted(inputs.keys()):
        value = inputs[key]
        if isinstance(value, str):
            parts.append(f"{key}:{value.strip().lower()}")
        elif isinstance(value, list):
            parts.append(f"{key}:{'|'.join(str(v).strip().lower() for v in value)}")
        else:
            parts.append(f"{key}:{json.dumps(value, sort_keys=True)}")
    return " ".join(parts)


def _parse_case(raw: dict[str, Any]) -> Case:
    task_type = TaskType(raw["task_type"])
    return Case(
        external_id=raw["id"],
        task_type=task_type,
        inputs=raw["inputs"],
        reference_answer=raw.get("reference_answer"),
        references=raw.get("references", []),
        expected_label=raw.get("expected_label"),
        expected_json=raw.get("expected_json"),
        qrels=raw.get("qrels"),
        slices=raw.get("slices", {}),
        must_contain=raw.get("must_contain", []),
        must_not_contain=raw.get("must_not_contain", []),
        canary=raw.get("canary"),
        weight=float(raw.get("weight", 1.0)),
        provenance=raw.get("provenance", {}),
        normalized_prompt=_normalize_prompt(raw["inputs"]),
    )


def load_dataset(dataset_dir: Path) -> DatasetBundle:
    manifest_path = dataset_dir / "manifest.yaml"
    cases_path = dataset_dir / "cases.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not cases_path.exists():
        raise FileNotFoundError(f"Missing cases: {cases_path}")

    manifest_raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest = DatasetManifest(
        name=manifest_raw["name"],
        version=manifest_raw["version"],
        split=manifest_raw["split"],
        license=manifest_raw.get("license", "unknown"),
        pii_scrubbed=bool(manifest_raw.get("pii_scrubbed", False)),
        created_at=manifest_raw.get("created_at", ""),
        slices=list(manifest_raw.get("slices", [])),
        content_sha256=manifest_raw.get("content_sha256"),
    )

    cases: list[Case] = []
    content_lines: list[str] = []
    with cases_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            content_lines.append(line)
            cases.append(_parse_case(json.loads(line)))

    content_sha256 = sha256_hex("\n".join(content_lines).encode("utf-8"))
    return DatasetBundle(
        manifest=manifest,
        cases=cases,
        content_sha256=content_sha256,
        source_path=dataset_dir,
    )

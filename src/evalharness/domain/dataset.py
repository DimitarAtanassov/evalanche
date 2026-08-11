"""Dataset and case domain types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evalharness.domain.enums import TaskType


@dataclass(frozen=True)
class Case:
    external_id: str
    task_type: TaskType
    inputs: dict[str, Any]
    reference_answer: str | None = None
    references: list[str] = field(default_factory=list)
    expected_label: str | None = None
    expected_json: dict[str, Any] | None = None
    qrels: dict[str, int] | None = None
    slices: dict[str, str] = field(default_factory=dict)
    must_contain: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)
    canary: str | None = None
    weight: float = 1.0
    provenance: dict[str, Any] = field(default_factory=dict)
    normalized_prompt: str | None = None


@dataclass(frozen=True)
class DatasetRef:
    """Persisted dataset identity returned by the store port."""

    id: int
    name: str
    version: str
    content_sha256: str
    split: str
    manifest: dict[str, Any]

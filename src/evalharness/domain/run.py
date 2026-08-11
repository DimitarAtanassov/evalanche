"""Run and definition domain records returned by the store port."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class RunRecord:
    id: uuid.UUID
    dataset_id: int
    prompt_template_id: int
    model_version_id: int
    decode_params: dict[str, Any]
    config_sha256: str
    harness_version: str
    git_sha: str
    repeats: int
    status: str
    tenant_id: str
    started_at: datetime | None
    finished_at: datetime | None
    baseline_run_id: uuid.UUID | None


@dataclass(frozen=True)
class PromptTemplateRef:
    id: int
    name: str
    version: str
    body: str
    content_sha256: str


@dataclass(frozen=True)
class ModelVersionRef:
    id: int
    provider: str
    model: str
    resolved_version: str
    quantization: str | None
    params_b: float | None
    context_window: int | None
    capabilities: dict[str, Any]

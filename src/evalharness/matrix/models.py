"""Typed contracts for matrix.yaml and baseline.yaml."""

from __future__ import annotations

from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from evalharness.suite.models import StrictModel

type JsonValue = Any


class DatasetTier(StrEnum):
    SMOKE = "smoke"
    DEV = "dev"
    HOLDOUT = "holdout"
    RELEASE = "release"


class MatrixModel(StrictModel):
    id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    revision: str | None = None
    digest: str | None = None

    @field_validator("model", "revision")
    @classmethod
    def reject_latest(cls, value: str | None) -> str | None:
        if value is not None and value.casefold() == "latest":
            raise ValueError('model revision and model name must not be "latest"')
        return value


class MatrixPrompt(StrictModel):
    id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    digest: str | None = None


class MatrixDataset(StrictModel):
    id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    digest: str | None = None
    tier: DatasetTier = DatasetTier.SMOKE


class DecodeParams(StrictModel):
    temperature: float
    max_tokens: int | None = None
    top_p: float | None = None


class MatrixCell(StrictModel):
    id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    dataset: str = Field(min_length=1)


class MatrixManifest(StrictModel):
    """Input contract for matrix.yaml schema 0.1."""

    schema_version: str
    name: str = Field(min_length=1)
    models: list[MatrixModel] = Field(min_length=1)
    prompts: list[MatrixPrompt] = Field(min_length=1)
    datasets: list[MatrixDataset] = Field(min_length=1)
    metrics: list[str] = Field(min_length=1)
    repeats: int = Field(ge=1)
    decode: DecodeParams
    cells: list[MatrixCell] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_refs_and_ids(self) -> MatrixManifest:
        model_ids = [item.id for item in self.models]
        prompt_ids = [item.id for item in self.prompts]
        dataset_ids = [item.id for item in self.datasets]
        cell_ids = [item.id for item in self.cells]
        for label, values in (
            ("model", model_ids),
            ("prompt", prompt_ids),
            ("dataset", dataset_ids),
            ("cell", cell_ids),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"{label} ids must be unique")
        model_set = set(model_ids)
        prompt_set = set(prompt_ids)
        dataset_set = set(dataset_ids)
        for cell in self.cells:
            if cell.model not in model_set:
                raise ValueError(f"cell {cell.id!r} references unknown model {cell.model!r}")
            if cell.prompt not in prompt_set:
                raise ValueError(f"cell {cell.id!r} references unknown prompt {cell.prompt!r}")
            if cell.dataset not in dataset_set:
                raise ValueError(f"cell {cell.id!r} references unknown dataset {cell.dataset!r}")
        return self


class PinnedCell(StrictModel):
    cell_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    run_report_path: str = Field(min_length=1)
    run_report_digest: str = Field(min_length=1)
    config_sha256: str = Field(min_length=1)
    model_digest: str = Field(min_length=1)

    @field_validator("run_id")
    @classmethod
    def validate_uuid(cls, value: str) -> str:
        try:
            UUID(value)
        except ValueError as exc:
            raise ValueError(f"run_id must be a UUID string, got {value!r}") from exc
        return value


class BaselineManifest(StrictModel):
    """Pinned promotion baseline; never selects wall-clock latest."""

    schema_version: str
    name: str = Field(min_length=1)
    matrix_name: str = Field(min_length=1)
    matrix_digest: str = Field(min_length=1)
    pinned_cells: list[PinnedCell] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_cells(self) -> BaselineManifest:
        cell_ids = [item.cell_id for item in self.pinned_cells]
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError("pinned cell_id values must be unique")
        return self


class LoadedMatrix(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    manifest_path: str
    manifest: MatrixManifest
    matrix_digest: str


class LoadedBaseline(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    manifest_path: str
    manifest: BaselineManifest

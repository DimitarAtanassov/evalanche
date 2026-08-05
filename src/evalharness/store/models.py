"""SQLAlchemy ORM models."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class DatasetRow(Base):
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[str] = mapped_column(Text, nullable=False)
    content_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    split: Mapped[str] = mapped_column(Text, nullable=False)
    manifest: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    cases: Mapped[list[CaseRow]] = relationship(back_populates="dataset")

    __table_args__ = (UniqueConstraint("name", "version"),)


class CaseRow(Base):
    __tablename__ = "cases"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    external_id: Mapped[str] = mapped_column(Text, nullable=False)
    task_type: Mapped[str] = mapped_column(Text, nullable=False)
    inputs: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    reference: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    qrels: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    slices: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default="{}")
    weight: Mapped[float] = mapped_column(Float, nullable=False, server_default="1.0")

    dataset: Mapped[DatasetRow] = relationship(back_populates="cases")

    __table_args__ = (
        UniqueConstraint("dataset_id", "external_id"),
        Index(
            "ix_cases_slices",
            "slices",
            postgresql_using="gin",
            postgresql_ops={"slices": "jsonb_path_ops"},
        ),
    )


class PromptTemplateRow(Base):
    __tablename__ = "prompt_templates"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[str] = mapped_column(Text, nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    content_sha256: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (UniqueConstraint("name", "version"),)


class ModelVersionRow(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    provider: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(Text, nullable=False)
    resolved_version: Mapped[str] = mapped_column(Text, nullable=False)
    quantization: Mapped[str | None] = mapped_column(Text)
    params_b: Mapped[float | None] = mapped_column(Float)
    context_window: Mapped[int | None] = mapped_column(Integer)
    capabilities: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)

    __table_args__ = (UniqueConstraint("provider", "model", "resolved_version", "quantization"),)


class RunRow(Base):
    __tablename__ = "runs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    prompt_template_id: Mapped[int] = mapped_column(
        ForeignKey("prompt_templates.id"), nullable=False
    )
    model_version_id: Mapped[int] = mapped_column(ForeignKey("model_versions.id"), nullable=False)
    decode_params: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    config_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    harness_version: Mapped[str] = mapped_column(Text, nullable=False)
    git_sha: Mapped[str] = mapped_column(Text, nullable=False)
    repeats: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    status: Mapped[str] = mapped_column(Text, nullable=False)
    tenant_id: Mapped[str] = mapped_column(Text, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    baseline_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("runs.id")
    )


class GenerationRow(Base):
    __tablename__ = "generations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("runs.id"), nullable=False
    )
    case_id: Mapped[int] = mapped_column(ForeignKey("cases.id"), nullable=False)
    repeat_idx: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    output: Mapped[str | None] = mapped_column(Text)
    tool_calls: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    finish_reason: Mapped[str | None] = mapped_column(Text)
    outcome: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer)
    completion_tokens: Mapped[int | None] = mapped_column(Integer)
    cost_usd: Mapped[float | None] = mapped_column(Numeric(12, 6))
    ttft_ms: Mapped[float | None] = mapped_column(Float)
    total_ms: Mapped[float | None] = mapped_column(Float)
    queue_wait_ms: Mapped[float | None] = mapped_column(Float)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    attempt_log: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    cached: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    raw_response: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    trace_id: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (UniqueConstraint("run_id", "case_id", "repeat_idx"),)


class ScoreRow(Base):
    __tablename__ = "scores"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    generation_id: Mapped[int] = mapped_column(ForeignKey("generations.id"), nullable=False)
    metric_name: Mapped[str] = mapped_column(Text, nullable=False)
    metric_version: Mapped[str] = mapped_column(Text, nullable=False)
    metric_config_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    value: Mapped[float | None] = mapped_column(Float)
    passed: Mapped[bool | None] = mapped_column(Boolean)
    detail: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    scored_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("generation_id", "metric_name", "metric_version", "metric_config_sha256"),
    )


class JudgmentRow(Base):
    __tablename__ = "judgments"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    generation_id: Mapped[int | None] = mapped_column(ForeignKey("generations.id"))
    compared_generation_id: Mapped[int | None] = mapped_column(ForeignKey("generations.id"))
    judge_model_version_id: Mapped[int | None] = mapped_column(ForeignKey("model_versions.id"))
    rubric_name: Mapped[str | None] = mapped_column(Text)
    rubric_version: Mapped[str | None] = mapped_column(Text)
    score: Mapped[int | None] = mapped_column(Integer)
    preference: Mapped[str | None] = mapped_column(Text)
    swap_position: Mapped[int | None] = mapped_column(Integer)
    reasoning: Mapped[str | None] = mapped_column(Text)
    evidence: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    cost_usd: Mapped[float | None] = mapped_column(Numeric(12, 6))


class AnnotationRow(Base):
    __tablename__ = "annotations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    case_id: Mapped[int | None] = mapped_column(ForeignKey("cases.id"))
    generation_id: Mapped[int | None] = mapped_column(ForeignKey("generations.id"))
    annotator_id: Mapped[str] = mapped_column(Text, nullable=False)
    label: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    adjudicated: Mapped[bool] = mapped_column(Boolean, server_default="false")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class EmbeddingRow(Base):
    __tablename__ = "embeddings"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    content_sha256: Mapped[str] = mapped_column(Text, nullable=False)
    embedding_model_version_id: Mapped[int] = mapped_column(
        ForeignKey("model_versions.id"), nullable=False
    )
    vec: Mapped[list[float]] = mapped_column(Vector(1024), nullable=False)

    __table_args__ = (UniqueConstraint("content_sha256", "embedding_model_version_id"),)


class MetricAggregateRow(Base):
    __tablename__ = "metric_aggregates"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    run_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("runs.id"))
    metric_name: Mapped[str] = mapped_column(Text)
    metric_version: Mapped[str] = mapped_column(Text)
    slice_key: Mapped[str] = mapped_column(Text, nullable=False, server_default="__overall__")
    n: Mapped[int] = mapped_column(Integer, nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    ci_low: Mapped[float | None] = mapped_column(Float)
    ci_high: Mapped[float | None] = mapped_column(Float)
    stddev: Mapped[float | None] = mapped_column(Float)
    method: Mapped[str | None] = mapped_column(Text)


class ResponseCacheRow(Base):
    __tablename__ = "response_cache"

    cache_key: Mapped[str] = mapped_column(Text, primary_key=True)
    response: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

"""Shared logging, tracing, privacy, and progress primitives.

Core pipeline code emits structured events and dependency-free progress snapshots.
Presentation belongs to adapters (the CLI uses Rich); observability must never become a
reason an evaluation fails.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from hashlib import sha256
from time import perf_counter
from typing import Any, TextIO, cast

import structlog
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Tracer

from evalharness.config import get_settings

_tracer: Tracer | None = None
_span_exporter: InMemorySpanExporter | None = None


class _DynamicStderr:
    """Forward writes to the current stderr.

    Test capture and notebook environments replace ``sys.stderr`` at runtime. Holding
    the old stream in a cached logger can otherwise turn a harmless log call into
    ``ValueError: I/O operation on closed file``.
    """

    def write(self, value: str) -> int:
        return sys.stderr.write(value)

    def flush(self) -> None:
        sys.stderr.flush()


class PipelineStage(StrEnum):
    BOOTSTRAP = "bootstrap"
    VALIDATING = "validating"
    RESOLVING = "resolving"
    PLANNING = "planning"
    GENERATING = "generating"
    SCORING = "scoring"
    AGGREGATING = "aggregating"
    REPORTING = "reporting"
    COMPLETED = "completed"


@dataclass(frozen=True)
class ProgressEvent:
    """Transport-neutral pipeline progress update."""

    stage: PipelineStage
    completed: int
    total: int
    message: str = ""
    counters: dict[str, int | float | str] = field(default_factory=dict)


type ProgressCallback = Callable[[ProgressEvent], None]


class StageTimer:
    """Monotonic stage timer suitable for logs and spans."""

    def __init__(self) -> None:
        self._started = perf_counter()

    @property
    def elapsed_ms(self) -> float:
        return round((perf_counter() - self._started) * 1000, 2)


_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;]+"),
    re.compile(r"(?i)((?:api[_-]?key|token|password|secret)\s*[:=]\s*)[^\s,;]+"),
    re.compile(r"(?i)([?&](?:api[_-]?key|token|key)=)[^&\s]+"),
)


def sanitize_text(value: str, *, max_chars: int = 240) -> str:
    """Single-line, bounded text with common credential forms redacted."""
    cleaned = " ".join(value.split())
    for pattern in _SECRET_PATTERNS:
        cleaned = pattern.sub(r"\1[REDACTED]", cleaned)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars] + "…"


def payload_summary(value: str | None) -> dict[str, Any]:
    """Privacy-safe payload metadata, with preview only under explicit opt-in."""
    if value is None:
        return {"chars": 0}
    settings = get_settings()
    result: dict[str, Any] = {"chars": len(value)}
    if settings.log_payload_hashes:
        result["sha256"] = sha256(value.encode("utf-8")).hexdigest()
    if settings.log_payloads:
        result["preview"] = sanitize_text(value, max_chars=settings.log_payload_max_chars)
    return result


def exception_summary(exc: BaseException) -> dict[str, str]:
    """Safe exception fields; provider bodies and credentials are bounded/redacted."""
    return {
        "error_type": type(exc).__name__,
        "error_message": sanitize_text(str(exc)),
    }


def emit_progress(callback: ProgressCallback | None, event: ProgressEvent) -> None:
    """Notify an adapter without allowing UI failures to stop the pipeline."""
    if callback is None:
        return
    try:
        callback(event)
    except Exception:
        get_logger(__name__).warning(
            "progress_callback_failed",
            stage=event.stage.value,
            exc_info=True,
        )


def setup_logging() -> None:
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    renderer: structlog.types.Processor
    use_console = settings.log_format == "console" or (
        settings.log_format == "auto" and sys.stderr.isatty()
    )
    if use_console:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    else:
        renderer = structlog.processors.JSONRenderer(sort_keys=True)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=cast(TextIO, _DynamicStderr())),
        cache_logger_on_first_use=True,
    )


def setup_otel() -> Tracer:
    global _tracer, _span_exporter
    if _tracer is not None:
        return _tracer
    settings = get_settings()
    resource = Resource.create({"service.name": settings.otel_service_name})
    provider = TracerProvider(resource=resource)
    if settings.otel_enabled:
        exporter: SpanExporter
        if settings.otel_exporter_otlp_endpoint:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint)
            provider.add_span_processor(BatchSpanProcessor(exporter))
        else:
            _span_exporter = InMemorySpanExporter()
            exporter = _span_exporter
            provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("evalanche")
    return _tracer


def get_tracer() -> Tracer:
    global _tracer
    if _tracer is None:
        return setup_otel()
    return _tracer


def bind_context(**kwargs: Any) -> None:
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    structlog.contextvars.clear_contextvars()


@contextmanager
def log_context(**kwargs: Any) -> Iterator[None]:
    """Temporarily bind fields across async-safe structlog context variables."""
    with structlog.contextvars.bound_contextvars(**kwargs):
        yield


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]


def progress_fields(event: ProgressEvent) -> dict[str, Any]:
    """Stable mapping useful when mirroring progress into structured logs."""
    fields = asdict(event)
    fields["stage"] = event.stage.value
    return fields

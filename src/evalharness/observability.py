"""Structured logging and OpenTelemetry setup."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Tracer

from evalharness.config import get_settings

_tracer: Tracer | None = None
_span_exporter: InMemorySpanExporter | None = None


def setup_logging() -> None:
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def setup_otel() -> Tracer:
    global _tracer, _span_exporter
    settings = get_settings()
    resource = Resource.create({"service.name": settings.otel_service_name})
    provider = TracerProvider(resource=resource)
    if settings.otel_enabled:
        _span_exporter = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(_span_exporter))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("evalharness")
    return _tracer


def get_tracer() -> Tracer:
    global _tracer
    if _tracer is None:
        return setup_otel()
    return _tracer


def bind_context(**kwargs: Any) -> None:
    structlog.contextvars.bind_contextvars(**kwargs)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]

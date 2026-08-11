"""Application configuration."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def redacted_database_url(url: str) -> str:
    """Strip userinfo from a database URL for safe logging."""
    parts = urlsplit(url)
    netloc = parts.netloc
    if "@" in netloc:
        netloc = netloc.rsplit("@", 1)[1]
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


class Settings(BaseSettings):
    """Environment-backed settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = Field(
        default="postgresql+asyncpg://evalharness:evalharness@localhost:5432/evalharness",
        repr=False,
    )
    ollama_base_url: str = "http://localhost:11434"
    openai_compatible_base_url: str | None = None
    openai_compatible_api_key: SecretStr | None = None
    openai_compatible_model_revision: str | None = None
    harness_version: str = "0.1.0"
    git_sha: str = "local"
    log_level: str = "INFO"
    log_format: Literal["auto", "json", "console"] = "auto"
    log_payloads: bool = False
    log_payload_hashes: bool = False
    log_payload_max_chars: int = Field(default=240, ge=0, le=4096)
    log_progress_every: int = Field(default=100, ge=1)
    otel_enabled: bool = False
    otel_service_name: str = "evalanche"
    otel_exporter_otlp_endpoint: str | None = None
    # Comma-separated allowlists applied to the metric entry-point group. Unset or empty
    # means every discovered metric whose dependencies are installed.
    metric_families: str | None = None
    metrics_enabled: str | None = None
    default_coverage_floor: float = 0.98
    default_max_retries: int = 5
    default_retry_base_s: float = 0.5
    default_retry_cap_s: float = 30.0
    default_concurrency: int = 2
    default_case_timeout_s: float = 120.0
    default_request_timeout_s: float = 60.0
    default_run_timeout_s: float = 14_400.0
    default_shutdown_drain_timeout_s: float = 30.0
    judge_provider_rpm: int = Field(default=60, ge=1)
    judge_provider_tpm: int = Field(default=60_000, ge=1)
    nli_provider_rpm: int = Field(default=60, ge=1)
    nli_provider_tpm: int = Field(default=60_000, ge=1)

    @property
    def database_url_for_logs(self) -> str:
        return redacted_database_url(self.database_url)


@lru_cache
def get_settings() -> Settings:
    return Settings()

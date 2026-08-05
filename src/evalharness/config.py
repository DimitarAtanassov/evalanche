"""Application configuration."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = Field(
        default="postgresql+asyncpg://evalharness:evalharness@localhost:5432/evalharness"
    )
    ollama_base_url: str = "http://localhost:11434"
    harness_version: str = "0.1.0"
    git_sha: str = "local"
    log_level: str = "INFO"
    otel_enabled: bool = False
    otel_service_name: str = "evalanche"
    default_coverage_floor: float = 0.98
    default_max_retries: int = 5
    default_retry_base_s: float = 0.5
    default_retry_cap_s: float = 30.0
    default_concurrency: int = 2
    default_case_timeout_s: float = 120.0
    default_request_timeout_s: float = 60.0


@lru_cache
def get_settings() -> Settings:
    return Settings()

"""Canonical JSON hashing utilities."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> str:
    """Serialize to canonical JSON for stable hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(data: str | bytes) -> str:
    """Return SHA-256 hex digest."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_canonical(obj: Any) -> str:
    """Hash canonical JSON representation."""
    return sha256_hex(canonical_json(obj))


def config_hash(
    *,
    dataset_sha256: str,
    prompt_template_sha256: str,
    provider: str,
    model: str,
    resolved_version: str,
    decode_params: dict[str, Any],
    harness_version: str,
) -> str:
    """Compute immutable run config hash."""
    payload = {
        "dataset_sha256": dataset_sha256,
        "prompt_template_sha256": prompt_template_sha256,
        "provider": provider,
        "model": model,
        "resolved_version": resolved_version,
        "decode_params": decode_params,
        "harness_version": harness_version,
    }
    return sha256_canonical(payload)

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


JUDGMENT_GATE_FIELDS = ("gating_allowed", "gating_block_reason", "calibration_digest")


def judgment_identity_digest(payload: dict[str, Any]) -> str:
    """Digest the judged content of a judgment artifact.

    Excludes the three fields ``attach-calibration`` rewrites. Hashing the whole
    payload would make the published artifact unverifiable against
    ``calibration.judgment_digest``, so any consumer holding only the attached
    file could not tell a real binding from a stolen digest.
    """
    body = {key: value for key, value in payload.items() if key not in JUDGMENT_GATE_FIELDS}
    return f"sha256:{sha256_canonical(body)}"


def calibration_body_digest(payload: dict[str, Any]) -> str:
    """Digest a calibration artifact body, excluding its own digest field."""
    body = {key: value for key, value in payload.items() if key != "calibration_digest"}
    return f"sha256:{sha256_canonical(body)}"


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

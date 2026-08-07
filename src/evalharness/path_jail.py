"""Shared relative-path jail for manifest-declared artifacts.

Manifests are untrusted input. Relative paths must resolve under a declared
base directory; ``../`` traversal and absolute escapes are refused. Callers
that intentionally allow absolute pins (baseline report paths) or CLI overrides
(gates) keep those exceptions local and do not route them through this helper.
"""

from __future__ import annotations

from pathlib import Path


def resolve_jailed_path(base: Path, declared: str) -> Path:
    """Resolve ``declared`` under ``base``, refusing anything outside the tree.

    Returns the resolved path. Callers map ``ValueError`` to their domain error.
    """
    resolved = (base / declared).resolve()
    try:
        resolved.relative_to(base.resolve())
    except ValueError as exc:
        raise ValueError(
            f"{declared!r} resolves outside the jail directory {base.resolve()}"
        ) from exc
    return resolved

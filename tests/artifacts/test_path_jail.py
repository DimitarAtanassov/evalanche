"""Characterization tests for the shared manifest path jail."""

from __future__ import annotations

from pathlib import Path

import pytest

from evalharness.path_jail import resolve_jailed_path


def test_resolve_jailed_path_accepts_relative_child(tmp_path: Path) -> None:
    child = tmp_path / "nested" / "artifact.json"
    child.parent.mkdir(parents=True)
    child.write_text("{}", encoding="utf-8")

    resolved = resolve_jailed_path(tmp_path, "nested/artifact.json")

    assert resolved == child.resolve()


@pytest.mark.parametrize(
    "declared",
    ["../outside.json", "nested/../../outside.json"],
)
def test_resolve_jailed_path_refuses_relative_traversal(tmp_path: Path, declared: str) -> None:
    outside = tmp_path.parent / "outside.json"
    outside.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="resolves outside the jail directory"):
        resolve_jailed_path(tmp_path, declared)


def test_resolve_jailed_path_refuses_absolute_outside_tree(tmp_path: Path) -> None:
    outside = tmp_path.parent / "abs-outside.json"
    outside.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="resolves outside the jail directory"):
        resolve_jailed_path(tmp_path, str(outside.resolve()))

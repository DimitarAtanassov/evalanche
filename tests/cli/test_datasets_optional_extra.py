"""Optional ``evaldatasets`` extra: harness stays importable without the factory."""

from __future__ import annotations

import ast
import importlib
import importlib.abc
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

from typer.testing import CliRunner

ROOT = Path(__file__).parents[2]
HARNESS_SRC = ROOT / "src" / "evalharness"
SMOKE_DATASET = ROOT / "fixtures" / "sample_dataset"
CLI = CliRunner()

_FACTORY_PREFIXES = ("evaldatasets",)
_INSTALL_HINT_MARKERS = ("evalanche[datasets]", "uv sync")


def _is_factory_module(name: str) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in _FACTORY_PREFIXES)


def _imported_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom) and node.module is not None:
        return [node.module]
    return []


def _module_scope_imports(tree: ast.AST) -> list[ast.AST]:
    """Collect Import/ImportFrom at module scope, including Try/If/With bodies.

    Stops at FunctionDef / AsyncFunctionDef / ClassDef so deferred imports inside
    handlers do not count as module-level loads.
    """
    found: list[ast.AST] = []

    def visit_stmts(stmts: list[ast.stmt]) -> None:
        for node in stmts:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                found.append(node)
                continue
            for attr in ("body", "orelse", "finalbody"):
                nested = getattr(node, attr, None)
                if isinstance(nested, list):
                    visit_stmts(nested)
            for handler in getattr(node, "handlers", []):
                visit_stmts(list(handler.body))

    visit_stmts(list(getattr(tree, "body", [])))
    return found


def _function_def(tree: ast.AST, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found at module body")


@contextmanager
def hide_dataset_factory() -> Iterator[None]:
    """Block ``evaldatasets`` for the duration of the block.

    Removes cached modules, inserts a meta-path finder that raises
    ``ModuleNotFoundError`` with the correct ``name``, then restores prior
    ``sys.modules`` entries so later tests still see a full workspace install.
    """
    removed = {
        name: sys.modules.pop(name) for name in list(sys.modules) if _is_factory_module(name)
    }

    class _HideFactory(importlib.abc.MetaPathFinder):
        def find_spec(
            self,
            fullname: str,
            path: object | None = None,
            target: object | None = None,
        ) -> None:
            if _is_factory_module(fullname):
                raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
            return None

    finder = _HideFactory()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        if finder in sys.meta_path:
            sys.meta_path.remove(finder)
        for name in list(sys.modules):
            if _is_factory_module(name):
                sys.modules.pop(name, None)
        sys.modules.update(removed)


def _reload_cli() -> ModuleType:
    """Re-import ``evalharness.cli`` after factory visibility may have changed."""
    # Drop the package and every submodule. Leaving a cached submodule (e.g.
    # ``evalharness.cli.runs``) while replacing the parent package leaves the
    # new parent without a bound attribute, which breaks dotted monkeypatches.
    for name in list(sys.modules):
        if name == "evalharness.cli" or name.startswith("evalharness.cli."):
            sys.modules.pop(name, None)
    return importlib.import_module("evalharness.cli")


def test_non_cli_harness_never_imports_evaldatasets() -> None:
    forbidden = _FACTORY_PREFIXES
    for path in HARNESS_SRC.rglob("*.py"):
        rel = path.relative_to(HARNESS_SRC)
        if rel.parts and rel.parts[0] == "cli":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            for name in _imported_names(node):
                assert not any(name == item or name.startswith(item + ".") for item in forbidden), (
                    f"{path}: forbidden import {name!r}"
                )


def test_cli_module_body_has_no_factory_import() -> None:
    for relative in ("cli/__init__.py", "cli/dataset.py"):
        path = HARNESS_SRC / relative
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in _module_scope_imports(tree):
            for name in _imported_names(node):
                assert not _is_factory_module(name), f"{path}: module-body import {name!r}"

    dataset_tree = ast.parse((HARNESS_SRC / "cli" / "dataset.py").read_text(encoding="utf-8"))
    materialize = _function_def(dataset_tree, "dataset_materialize")
    deferred = [
        node
        for node in ast.walk(materialize)
        if isinstance(node, ast.ImportFrom) and node.module == "evaldatasets"
    ]
    assert deferred, "dataset_materialize must ImportFrom evaldatasets inside its body"


def test_import_evalharness_cli_succeeds_when_factory_absent() -> None:
    with hide_dataset_factory():
        cli = _reload_cli()

        assert hasattr(cli, "app")
        result = CLI.invoke(cli.app, ["dataset-validate", str(SMOKE_DATASET)])

    assert result.exit_code == 0, result.output
    assert "evalanche[datasets]" not in result.output


def test_dataset_materialize_without_factory_prints_install_hint(
    tmp_path: Path,
) -> None:
    with hide_dataset_factory():
        cli = _reload_cli()
        result = CLI.invoke(
            cli.app,
            [
                "dataset",
                "materialize",
                "--adapter",
                "synthetic_qa",
                "--source",
                str(tmp_path / "source.jsonl"),
                "--out",
                str(tmp_path / "out"),
                "--seed",
                "1",
                "--size",
                "1",
                "--tier",
                "smoke",
            ],
        )

    assert result.exit_code == 1
    assert "evalanche[datasets]" in result.output
    assert "uv sync" in result.output
    assert "--extra datasets" in result.output


def test_materialize_error_does_not_emit_install_hint(tmp_path: Path) -> None:
    from evalharness.cli import app

    result = CLI.invoke(
        app,
        [
            "dataset",
            "materialize",
            "--adapter",
            "not_a_registered_adapter",
            "--source",
            str(tmp_path / "source.jsonl"),
            "--out",
            str(tmp_path / "out"),
            "--seed",
            "1",
            "--size",
            "1",
            "--tier",
            "smoke",
        ],
    )

    assert result.exit_code != 0
    assert "UNKNOWN_ADAPTER" in result.output
    for marker in _INSTALL_HINT_MARKERS:
        assert marker not in result.output

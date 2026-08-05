"""Architectural guards for the benchmark-focused active package."""

import ast
from pathlib import Path


SOURCE_ROOT = Path(__file__).parents[1] / "src" / "jaxwind"


def test_active_package_contains_only_the_benchmark_solver_layers() -> None:
    packages = {
        path.name
        for path in SOURCE_ROOT.iterdir()
        if path.is_dir() and not path.name.startswith("__pycache__")
    }

    assert packages == {"domain", "momentum", "pressure"}


def test_active_solver_never_imports_the_archive() -> None:
    violations = []
    for path in SOURCE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = (node.module,)
            else:
                continue
            if any(
                name == "jaxwind_archiv" or name.startswith("jaxwind_archiv.")
                for name in modules
            ):
                violations.append(str(path.relative_to(SOURCE_ROOT)))

    assert not violations, f"active solver imports archive modules: {violations}"


def test_archive_and_active_packages_are_separate_trees() -> None:
    archive = SOURCE_ROOT.with_name("jaxwind_archiv")

    assert archive.is_dir()
    assert not SOURCE_ROOT.resolve().is_relative_to(archive.resolve())
    assert not archive.resolve().is_relative_to(SOURCE_ROOT.resolve())

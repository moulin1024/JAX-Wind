from __future__ import annotations

import ast
from pathlib import Path


SOURCE_ROOT = Path(__file__).parents[1] / "src" / "jaxwind"


def test_active_package_is_the_minimal_benchmark_solver() -> None:
    packages = {
        path.name
        for path in SOURCE_ROOT.iterdir()
        if path.is_dir() and not path.name.startswith("__pycache__")
    }
    assert packages == {"domain", "momentum", "pressure"}


def test_active_solver_does_not_import_the_archive() -> None:
    violations = []
    for path in SOURCE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.ImportFrom):
                module = node.module
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("jaxwind_archiv"):
                        violations.append(str(path.relative_to(SOURCE_ROOT)))
            if module and module.startswith("jaxwind_archiv"):
                violations.append(str(path.relative_to(SOURCE_ROOT)))
    assert not violations


def test_active_solver_has_no_flow_regime_specializations() -> None:
    assert not (SOURCE_ROOT / "momentum" / "neutral_abl.py").exists()
    assert not (SOURCE_ROOT / "momentum" / "convective_abl.py").exists()
    public_api = (SOURCE_ROOT / "momentum" / "__init__.py").read_text()
    assert "NeutralABL" not in public_api
    assert "AMDBoussinesq" not in public_api

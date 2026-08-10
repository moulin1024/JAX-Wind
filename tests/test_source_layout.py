"""Low-cost architectural guards for the active production package."""

import ast
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src" / "jaxwind"
MAX_PRODUCTION_MODULE_LINES = 1_000
INTERPRETER_ROOT = SOURCE_ROOT / "interpreters"


def test_production_modules_stay_within_maintenance_ceiling() -> None:
    oversized = {}
    for path in SOURCE_ROOT.rglob("*.py"):
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        if line_count > MAX_PRODUCTION_MODULE_LINES:
            oversized[str(path.relative_to(SOURCE_ROOT))] = line_count

    assert not oversized, (
        "split production modules at responsibility boundaries before they "
        f"exceed {MAX_PRODUCTION_MODULE_LINES} lines: {oversized}"
    )


def test_zslab_is_the_only_public_interpreter_module() -> None:
    public_modules = {
        path.name
        for path in INTERPRETER_ROOT.glob("*.py")
        if path.name != "__init__.py" and not path.name.startswith("_")
    }

    assert public_modules == {"jax_zslab.py"}


def test_package_has_no_case_dispatch_layer() -> None:
    assert not (SOURCE_ROOT / "runners").exists()
    assert not (SOURCE_ROOT / "cli.py").exists()
    assert not (SOURCE_ROOT / "__main__.py").exists()


def test_cases_are_data_only_and_have_no_benchmark_namespace() -> None:
    assert (REPOSITORY_ROOT / "cases").is_dir()
    assert not (REPOSITORY_ROOT / "benchmark").exists()
    assert not tuple((REPOSITORY_ROOT / "cases").rglob("*.py"))
    for name in ("PressureDrivenLASD", "Andren1994", "Nieuwstadt1993"):
        case = REPOSITORY_ROOT / "cases" / name
        assert (case / "config.toml").is_file()


def test_abl_application_is_not_partitioned_by_stability() -> None:
    applications = REPOSITORY_ROOT / "applications"
    assert (applications / "abl").is_dir()
    for regime in ("neutral_abl", "stable_abl", "convective_abl"):
        assert not (applications / regime).exists()


def test_production_never_imports_test_support() -> None:
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
            if any(name == "tests" or name.startswith("tests.") for name in modules):
                violations.append(str(path.relative_to(SOURCE_ROOT)))

    assert not violations, f"production imports test-only modules: {violations}"


def test_production_never_imports_cases() -> None:
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
                name == "cases" or name.startswith("cases.")
                for name in modules
            ):
                violations.append(str(path.relative_to(SOURCE_ROOT)))

    assert not violations, f"production imports case code: {violations}"


def test_production_never_imports_applications() -> None:
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
                name == "applications" or name.startswith("applications.")
                for name in modules
            ):
                violations.append(str(path.relative_to(SOURCE_ROOT)))

    assert not violations, f"production imports application code: {violations}"

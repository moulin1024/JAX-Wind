"""Low-cost architectural guards for the active production package."""

from pathlib import Path


SOURCE_ROOT = Path(__file__).parents[1] / "src" / "jaxwind"
MAX_PRODUCTION_MODULE_LINES = 1_000


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

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = ROOT / "benchmark" / "Nieuwstadt1993"
PAPER = BENCHMARK_DIR / "reference" / "Nieuwstadt1993.md"
RUNNER = BENCHMARK_DIR / "run_new.py"

MOENG_TABLE3_TARGETS = {
    "zi_over_zi0": (1.0312, 0.02),
    "wstar_over_wstar0": (1.010, 0.015),
    "entrainment_ratio": (0.106, 0.03),
}

REQUIRED_OUTPUTS = (
    "summary.csv",
    "profiles.csv",
    "time_series.csv",
    "benchmark_stats.npz",
    "fig01_energy_time.png",
    "fig02_heat_flux.png",
    "fig09_dissipation.png",
    "fig09_11_energy_budget_terms.png",
    "fig10_11_energy_budget_terms.png",
)

REQUIRED_PROFILE_COLUMNS = {
    "z_over_zi",
    "heat_flux_over_qs",
    "heat_flux_resolved_over_qs",
    "heat_flux_sgs_over_qs",
    "heat_flux_total_over_qs",
    "epsilon_zi_over_wstar3",
    "theta_var_resolved_over_thetastar_sq",
    "theta_var_sgs_over_thetastar_sq",
}


def _read_summary(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header != ["quantity", "value"]:
            raise AssertionError(f"unexpected summary header in {path}: {header}")
        for key, value in reader:
            values[key] = float(value)
    return values


def _read_profiles(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_PROFILE_COLUMNS - set(reader.fieldnames or ())
        if missing:
            raise AssertionError(f"profiles.csv is missing columns: {sorted(missing)}")
        return [{key: float(value) for key, value in row.items()} for row in reader]


def _assert_reference_markdown() -> None:
    if not PAPER.exists():
        raise AssertionError(f"missing benchmark specification: {PAPER}")
    text = PAPER.read_text(encoding="utf-8")
    required = (
        "Nieuwstadt et al. (1993) CBL benchmark specification",
        "Common physical case",
        "Initial condition",
        "Reference bulk targets",
        "Mapping to the current JAX benchmark",
        "independently written simulation summary",
    )
    missing = [needle for needle in required if needle not in text]
    if missing:
        raise AssertionError(
            f"benchmark specification is missing expected anchors: {missing}"
        )
    forbidden = ("## 1. Introduction", "cdn.mathpix.com", "## Acknowledgement")
    present = [needle for needle in forbidden if needle in text]
    if present:
        raise AssertionError(
            f"benchmark specification still contains publication-text markers: {present}"
        )


def _assert_required_files(output_dir: Path) -> None:
    missing = [name for name in REQUIRED_OUTPUTS if not (output_dir / name).exists()]
    if missing:
        raise AssertionError(f"missing benchmark outputs in {output_dir}: {missing}")


def _assert_summary(summary: dict[str, float], quick: bool) -> None:
    required = {
        "sample_count",
        "zi_over_zi0",
        "wstar_over_wstar0",
        "entrainment_ratio",
        "theta_mixed_layer_mean",
    }
    missing = required - set(summary)
    if missing:
        raise AssertionError(f"summary.csv is missing quantities: {sorted(missing)}")
    for key in required:
        if not math.isfinite(summary[key]):
            raise AssertionError(f"summary quantity {key} is not finite: {summary[key]}")
    if summary["sample_count"] < 1.0:
        raise AssertionError(f"expected at least one averaged sample, got {summary['sample_count']}")
    if quick:
        return
    if summary["sample_count"] < 8.0:
        raise AssertionError(f"full benchmark should average at least 8 samples, got {summary['sample_count']}")
    for key, (target, tolerance) in MOENG_TABLE3_TARGETS.items():
        value = summary[key]
        if abs(value - target) > tolerance:
            raise AssertionError(
                f"{key}={value:.6g} is outside Moeng Table 3 target "
                f"{target:.6g} +/- {tolerance:.3g}"
            )


def _assert_profiles(profiles: list[dict[str, float]]) -> None:
    if not profiles:
        raise AssertionError("profiles.csv contains no rows")
    epsilon = [row["epsilon_zi_over_wstar3"] for row in profiles]
    if not all(math.isfinite(value) for value in epsilon):
        raise AssertionError("epsilon_zi_over_wstar3 contains non-finite values")
    if min(epsilon) < -1.0e-10:
        raise AssertionError(f"epsilon_zi_over_wstar3 should be non-negative, min={min(epsilon):.6g}")
    if max(epsilon) <= 0.0:
        raise AssertionError("epsilon_zi_over_wstar3 is identically zero")
    heat_flux = [row["heat_flux_total_over_qs"] for row in profiles]
    if not all(math.isfinite(value) for value in heat_flux):
        raise AssertionError("heat_flux_total_over_qs contains non-finite values")
    theta_sgs = [row["theta_var_sgs_over_thetastar_sq"] for row in profiles]
    if not all(math.isfinite(value) and value >= 0.0 for value in theta_sgs):
        raise AssertionError(
            "theta_var_sgs_over_thetastar_sq must be finite and non-negative"
        )


def _assert_spectrum_levels(output_dir: Path) -> None:
    import numpy as np

    stats = np.load(output_dir / "benchmark_stats.npz")
    requested = np.asarray(stats["spectrum_level_fraction"], dtype=np.float64)
    actual = (
        np.asarray(stats["spectrum_level_z"], dtype=np.float64)
        / float(stats["zi_mean"])
    )
    np.testing.assert_allclose(requested, (0.2, 0.6, 1.0), atol=0.0)
    # Nearest cell-centre sampling can differ by at most half a vertical cell;
    # derive the limit from the output so the intentionally coarse quick grid
    # is checked by the same geometric rule as the canonical 48-level run.
    z = np.asarray(stats["z"], dtype=np.float64)
    half_cell_over_zi = 0.5 * float(np.median(np.diff(z))) / float(stats["zi_mean"])
    if np.max(np.abs(actual - requested)) > half_cell_over_zi + 1.0e-12:
        raise AssertionError(
            f"spectrum levels {actual} are too far from requested {requested}"
        )


def run_integration(output_dir: Path, quick: bool = False, python: str = sys.executable) -> dict[str, float]:
    _assert_reference_markdown()
    if not RUNNER.exists():
        raise AssertionError(f"missing benchmark runner: {RUNNER}")

    command = [
        python,
        str(RUNNER),
        "--output-dir",
        str(output_dir),
        "--sample-every",
        "40",
    ]
    if quick:
        command.append("--quick")

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    pressure_source = Path(
        env.get("JAXWIND_SPECTRAL_FD_SOURCE", ROOT / "external" / "bw1000_benchmark")
    )
    env["JAXWIND_SPECTRAL_FD_SOURCE"] = str(pressure_source)
    env["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT / "src"), str(pressure_source), env.get("PYTHONPATH", ""))
    )

    result = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
    if result.returncode != 0:
        print(result.stdout, file=sys.stdout)
        print(result.stderr, file=sys.stderr)
        raise RuntimeError(f"benchmark command failed with exit code {result.returncode}: {' '.join(command)}")
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")

    _assert_required_files(output_dir)
    summary = _read_summary(output_dir / "summary.csv")
    profiles = _read_profiles(output_dir / "profiles.csv")
    _assert_summary(summary, quick=quick)
    _assert_profiles(profiles)
    _assert_spectrum_levels(output_dir)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and validate the Moeng-style Largeeddy1993 CBL integration case.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "benchmark_results" / "Nieuwstadt1993_moeng_integration",
        help="Directory for generated diagnostics.",
    )
    parser.add_argument("--quick", action="store_true", help="Run only 80 steps and validate IO/diagnostic structure.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to launch the benchmark runner.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_integration(args.output_dir, quick=args.quick, python=args.python)
    mode = "quick" if args.quick else "full"
    print(f"[integration] {mode} Moeng CBL validation passed")
    for key in ("zi_over_zi0", "wstar_over_wstar0", "entrainment_ratio", "sample_count"):
        print(f"  {key}: {summary[key]:.6g}")


if __name__ == "__main__":
    main()

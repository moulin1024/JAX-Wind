#!/usr/bin/env python3
"""Run the long pressure-driven MGM case and plot its neutral log law."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
from dataclasses import replace
import importlib.util
import math
import os
from pathlib import Path
import sys
from typing import TextIO


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

CONFIG = ROOT / "runners" / "pressure_driven_warmup" / "config_mgm.toml"
DEFAULT_OUTPUT = ROOT / "outputs" / "pressure_driven_mgm_64x64x64_gpu"


class _Tee:
    """Mirror runner output to the terminal and a persistent log."""

    def __init__(self, terminal: TextIO, log: TextIO) -> None:
        self._terminal = terminal
        self._log = log

    def write(self, text: str) -> int:
        self._terminal.write(text)
        self._log.write(text)
        self._log.flush()
        return len(text)

    def flush(self) -> None:
        self._terminal.flush()
        self._log.flush()

    def isatty(self) -> bool:
        return self._terminal.isatty()


def _arguments(
    argv: list[str] | None,
    *,
    program: str = "python -m benchmark.PressureDrivenMGM",
    description: str | None = __doc__,
    default_output: Path = DEFAULT_OUTPUT,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=program,
        description=description,
    )
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument(
        "--dt",
        type=float,
        help="override time.dt_seconds from config_mgm.toml",
    )
    parser.add_argument(
        "--hours",
        type=float,
        help="override time.duration_hours and average over the final 20%%",
    )
    parser.add_argument(
        "--restart",
        type=Path,
        help="restart checkpoint; the default automatically resumes the output directory",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="regenerate the log-law plot from an existing profiles.csv",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="optional development cap; omit it to run the configured duration",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="allow an explicit CPU run; the benchmark requires a GPU by default",
    )
    args = parser.parse_args(argv)
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.dt is not None and args.dt <= 0.0:
        parser.error("--dt must be positive")
    if args.hours is not None and args.hours <= 0.0:
        parser.error("--hours must be positive")
    return args


def _require_gpu(jax, *, allow_cpu: bool) -> tuple[object, ...]:
    devices = tuple(jax.devices())
    accelerators = tuple(
        device
        for device in devices
        if str(device.platform).lower() in {"gpu", "cuda", "rocm", "metal"}
    )
    if not accelerators and not allow_cpu:
        raise RuntimeError(
            "no JAX GPU device was detected; install the accelerator-specific JAX "
            "build or pass --allow-cpu only for a smoke run"
        )
    selected = accelerators or devices
    print("JAX devices:", ", ".join(map(str, selected)), flush=True)
    return selected


def _configure_pressure_solver() -> Path | None:
    """Find an installed, submodule, or sibling spectral-fd checkout."""

    if importlib.util.find_spec("spectral_fd") is not None:
        return None
    configured = os.environ.get("JAXWIND_SPECTRAL_FD_SOURCE")
    candidates = tuple(
        path
        for path in (
            Path(configured).expanduser() if configured else None,
            ROOT / "external" / "bw1000_benchmark",
            ROOT.parent / "bw1000_benchmark",
        )
        if path is not None
    )
    for candidate in candidates:
        if (candidate / "spectral_fd" / "__init__.py").is_file():
            resolved = candidate.resolve()
            sys.path.insert(0, str(resolved))
            os.environ["JAXWIND_SPECTRAL_FD_SOURCE"] = str(resolved)
            return resolved
    raise RuntimeError(
        "spectral_fd is unavailable. From the JAX-Wind repository run: "
        "git submodule update --init --recursive"
    )


def _polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def write_log_law_svg(
    profile_path: Path,
    figure_path: Path,
    *,
    friction_velocity_m_s: float,
    roughness_length_m: float,
    von_karman: float,
    model_label: str = "MGM",
    statistics_label: str = "final 20%",
) -> None:
    """Write a dependency-free normalized velocity/log-law comparison."""

    with profile_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"profile contains no samples: {profile_path}")

    z = [float(row["z_m"]) for row in rows]
    measured = [
        float(row["mean_u_m_s"]) / friction_velocity_m_s for row in rows
    ]
    normalized_z = [height / roughness_length_m for height in z]
    reference = [math.log(height) / von_karman for height in normalized_z]
    if min(normalized_z) <= 0.0 or not all(map(math.isfinite, measured)):
        raise ValueError("profile heights and velocities must be finite and positive")

    width, height = 900, 620
    left, right, top, bottom = 92, 34, 68, 78
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_values = measured + reference
    x_min = math.floor(min(x_values) - 1.0)
    x_max = math.ceil(max(x_values) + 1.0)
    log_y = [math.log10(value) for value in normalized_z]
    y_min, y_max = min(log_y), max(log_y)

    def x_pixel(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y_pixel(value: float) -> float:
        return top + (y_max - math.log10(value)) / (y_max - y_min) * plot_height

    measured_points = [
        (x_pixel(u_value), y_pixel(z_value))
        for u_value, z_value in zip(measured, normalized_z, strict=True)
    ]
    reference_points = [
        (x_pixel(u_value), y_pixel(z_value))
        for u_value, z_value in zip(reference, normalized_z, strict=True)
    ]

    x_ticks = 6
    x_tick_values = [
        x_min + index * (x_max - x_min) / (x_ticks - 1)
        for index in range(x_ticks)
    ]
    y_decades = range(math.ceil(y_min), math.floor(y_max) + 1)
    elements = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" role="img" '
        'aria-labelledby="title description">',
        f'<title id="title">Pressure-driven {model_label} neutral log-law profile</title>',
        '<desc id="description">Horizontally and temporally averaged streamwise '
        'velocity compared with the theoretical neutral logarithmic law.</desc>',
        "<style>",
        "text{font-family:system-ui,sans-serif;fill:#202124}",
        ".title{font-size:22px;font-weight:600}",
        ".axis{font-size:15px}",
        ".tick{font-size:13px;fill:#4b5563}",
        ".grid{stroke:#d1d5db;stroke-width:1}",
        ".frame{fill:#fff;stroke:#4b5563;stroke-width:1.2}",
        ".profile{fill:none;stroke:#1769aa;stroke-width:2.5}",
        ".log{fill:none;stroke:#d1495b;stroke-width:2.5;stroke-dasharray:9 6}",
        ".point{fill:#1769aa}",
        "</style>",
        f'<text class="title" x="{width / 2}" y="32" text-anchor="middle">'
        f"Pressure-driven neutral ABL: {model_label}</text>",
        f'<rect class="frame" x="{left}" y="{top}" width="{plot_width}" '
        f'height="{plot_height}"/>',
    ]
    for tick in x_tick_values:
        pixel = x_pixel(tick)
        elements.extend(
            (
                f'<line class="grid" x1="{pixel:.2f}" y1="{top}" '
                f'x2="{pixel:.2f}" y2="{top + plot_height}"/>',
                f'<text class="tick" x="{pixel:.2f}" y="{top + plot_height + 24}" '
                f'text-anchor="middle">{tick:.1f}</text>',
            )
        )
    for decade in y_decades:
        value = 10.0**decade
        pixel = y_pixel(value)
        elements.extend(
            (
                f'<line class="grid" x1="{left}" y1="{pixel:.2f}" '
                f'x2="{left + plot_width}" y2="{pixel:.2f}"/>',
                f'<text class="tick" x="{left - 12}" y="{pixel + 5:.2f}" '
                f'text-anchor="end">10<tspan dy="-5" font-size="10">{decade}</tspan>'
                '<tspan dy="5"></tspan></text>',
            )
        )
    elements.extend(
        (
            f'<polyline class="log" points="{_polyline(reference_points)}"/>',
            f'<polyline class="profile" points="{_polyline(measured_points)}"/>',
        )
    )
    elements.extend(
        f'<circle class="point" cx="{x:.2f}" cy="{y:.2f}" r="2.8"/>'
        for x, y in measured_points
    )
    elements.extend(
        (
            f'<text class="axis" x="{left + plot_width / 2}" y="{height - 22}" '
            'text-anchor="middle">U/u*</text>',
            f'<text class="axis" transform="translate(25 {top + plot_height / 2}) '
            'rotate(-90)" text-anchor="middle">z/z0 (log scale)</text>',
            f'<line class="profile" x1="{left + 25}" y1="{top + 24}" '
            f'x2="{left + 70}" y2="{top + 24}"/>',
            f'<text class="tick" x="{left + 80}" y="{top + 29}">'
            f"{model_label} mean, {statistics_label}</text>",
            f'<line class="log" x1="{left + 280}" y1="{top + 24}" '
            f'x2="{left + 325}" y2="{top + 24}"/>',
            f'<text class="tick" x="{left + 335}" y="{top + 29}">'
            "U+ = ln(z/z0)/kappa</text>",
            "</svg>",
        )
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.write_text("\n".join(elements) + "\n")


def _select_sgs(case, sgs_model: str):
    if sgs_model not in {"mgm", "lasd", "amd"}:
        raise ValueError("sgs_model must be 'mgm', 'lasd', or 'amd'")
    if case.sgs.model == sgs_model:
        return case
    return replace(
        case,
        name=case.name.replace(case.sgs.model, sgs_model),
        sgs=replace(case.sgs, model=sgs_model),
    )


def run_benchmark(
    argv: list[str] | None,
    *,
    sgs_model: str,
    program: str,
    description: str,
    default_output: Path,
) -> int:
    args = _arguments(
        argv,
        program=program,
        description=description,
        default_output=default_output,
    )
    from jaxwind.runners.pressure_driven_warmup import load_case, run_case

    case = load_case(
        CONFIG,
        dt_seconds=args.dt,
        duration_hours=args.hours,
        statistics_fraction=0.2,
    )
    case = _select_sgs(case, sgs_model)
    output = args.output.resolve()
    profile_path = output / "profiles.csv"
    figure_path = output / "loglaw_velocity_profile.svg"
    if not args.plot_only:
        import jax

        _require_gpu(jax, allow_cpu=args.allow_cpu)
        pressure_source = _configure_pressure_solver()
        if pressure_source is not None:
            print(f"spectral_fd source: {pressure_source}", flush=True)
        output.mkdir(parents=True, exist_ok=True)
        latest = output / "checkpoint_latest.npz"
        restart = args.restart
        if restart is None and latest.exists():
            restart = latest
            print(f"Resuming {restart}", flush=True)
        log_mode = "a" if restart is not None else "w"
        with (output / "run.log").open(log_mode) as log:
            with redirect_stdout(_Tee(sys.stdout, log)):
                with redirect_stderr(_Tee(sys.stderr, log)):
                    run_case(
                        case,
                        output_dir=output,
                        restart=restart,
                        max_steps=args.max_steps,
                        overwrite=False,
                    )

    if profile_path.exists():
        write_log_law_svg(
            profile_path,
            figure_path,
            friction_velocity_m_s=case.flow.friction_velocity_m_s,
            roughness_length_m=case.flow.roughness_length_m,
            von_karman=case.flow.von_karman,
            model_label=sgs_model.upper(),
            statistics_label="final 20%",
        )
        print(f"Log-law profile: {figure_path}", flush=True)
    elif args.plot_only:
        raise FileNotFoundError(profile_path)
    else:
        print(
            "No profile was written yet; it starts after "
            f"{case.output.sample_start_hours:g} simulated hours.",
            flush=True,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_benchmark(
        argv,
        sgs_model="mgm",
        program="python -m benchmark.PressureDrivenMGM",
        description=__doc__ or "Pressure-driven MGM benchmark",
        default_output=DEFAULT_OUTPUT,
    )


if __name__ == "__main__":
    raise SystemExit(main())

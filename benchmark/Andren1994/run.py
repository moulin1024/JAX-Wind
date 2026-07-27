#!/usr/bin/env python3
"""Run the Andrén et al. (1994) neutral Ekman intercomparison case."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.NeutralEkman import run as neutral_runner  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from jaxwind.domain import (  # noqa: E402
    AddressableField,
    Candidate,
    Cell,
    VerticalVelocity,
    XVelocity,
    YVelocity,
    ZFace,
)
from jaxwind.interpreters.jax_zslab import ZFaceFieldContext  # noqa: E402
from jaxwind.operators import VelocityVector  # noqa: E402


HERE = Path(__file__).resolve().parent
INITIAL_PROFILE = HERE / "reference" / "initial_profiles.csv"
REFERENCE_RESULTS = HERE / "reference" / "reference_results.json"
F_CORIOLIS = 1.0e-4
GEOSTROPHIC_SPEED = 10.0
CANONICAL_HOURS = 10.0 / F_CORIOLIS / 3600.0
STATISTICS_START_HOURS = 7.0 / F_CORIOLIS / 3600.0
STATISTICS_WINDOW_HOURS = 3.0 / F_CORIOLIS / 3600.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=40)
    parser.add_argument("--ny", type=int, default=40)
    parser.add_argument("--nz", type=int, default=40)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--hours", type=float)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--smagorinsky", type=float, default=0.17)
    parser.add_argument("--seed", type=int, default=1994)
    parser.add_argument("--sample-every", type=int, default=300)
    parser.add_argument("--log-every", type=int, default=600)
    parser.add_argument("--max-cfl-warning", type=float, default=0.25)
    parser.add_argument("--method", choices=("transpose", "spike"), default="spike")
    parser.add_argument("--restart", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run an 8^3, 8-step end-to-end smoke case instead of the paper case",
    )
    args = parser.parse_args(argv)
    if args.quick:
        args.nx = args.ny = args.nz = 8
        args.dt = 0.25
        args.hours = 8.0 * args.dt / 3600.0
        args.sample_every = 1
        args.log_every = 4
    elif args.hours is None:
        args.hours = CANONICAL_HOURS
    if args.output is None:
        suffix = "quick" if args.quick else "static_smag_40x40x40"
        args.output = HERE / "results" / suffix
    if min(args.nx, args.ny, args.nz) <= 1:
        parser.error("all grid dimensions must exceed one")
    if args.dt <= 0.0 or args.hours <= 0.0:
        parser.error("dt and hours must be positive")
    if args.sample_every <= 0 or args.log_every <= 0:
        parser.error("sampling and logging intervals must be positive")
    return args


def paper_initial_profiles() -> np.ndarray:
    data = np.genfromtxt(INITIAL_PROFILE, delimiter=",", names=True)
    if data.shape != (40,):
        raise ValueError("Andrén Table A.1 must contain exactly 40 levels")
    if not np.allclose(np.diff(data["z_m"]), 37.5):
        raise ValueError("Andrén Table A.1 heights must use the 37.5 m grid")
    return data


def _unit_plane_noise(key, shape, dtype):
    noise = jax.random.uniform(key, shape, dtype, minval=-0.5, maxval=0.5)
    noise -= jnp.mean(noise, axis=(-2, -1), keepdims=True)
    rms = jnp.sqrt(jnp.mean(noise * noise, axis=(-2, -1), keepdims=True))
    return noise / jnp.maximum(rms, jnp.finfo(dtype).tiny)


def andren_initial_velocity(
    physical_grid,
    decomposition,
    scales,
    dtype,
    args,
) -> VelocityVector:
    """Interpret the paper's Table A.1 and uniform random perturbation law."""
    table = paper_initial_profiles()
    z = (jnp.arange(physical_grid.nz, dtype=dtype) + 0.5) * physical_grid.dz
    upper_z = (jnp.arange(physical_grid.nz, dtype=dtype) + 1.0) * physical_grid.dz
    table_z = jnp.asarray(table["z_m"], dtype=dtype)
    table_u = jnp.asarray(table["u_m_s"], dtype=dtype)
    table_v = jnp.asarray(table["v_m_s"], dtype=dtype)
    table_tke = jnp.asarray(table["tke_m2_s2"], dtype=dtype)
    mean_u = jnp.interp(z, table_z, table_u, left=table_u[0], right=table_u[-1])
    mean_v = jnp.interp(z, table_z, table_v, left=table_v[0], right=table_v[-1])
    cell_tke = jnp.interp(z, table_z, table_tke, left=table_tke[0], right=0.0)
    face_tke = jnp.interp(
        upper_z,
        table_z,
        table_tke,
        left=table_tke[0],
        right=0.0,
    )
    shape = (1, physical_grid.nz, physical_grid.ny, physical_grid.nx)
    keys = jax.random.split(jax.random.PRNGKey(args.seed), 3)
    component_rms = jnp.sqrt((2.0 / 3.0) * cell_tke)[None, :, None, None]
    face_rms = jnp.sqrt((2.0 / 3.0) * face_tke)[None, :, None, None]
    u = mean_u[None, :, None, None] + component_rms * _unit_plane_noise(
        keys[0], shape, dtype
    )
    v = mean_v[None, :, None, None] + component_rms * _unit_plane_noise(
        keys[1], shape, dtype
    )
    w = face_rms * _unit_plane_noise(keys[2], shape, dtype)
    w = w.at[:, -1].set(0.0)
    lower = jnp.zeros((physical_grid.ny, physical_grid.nx), dtype=dtype)
    return VelocityVector(
        AddressableField(
            XVelocity,
            Cell,
            decomposition.regions(Cell),
            Candidate,
            scales.to_execution_velocity(u).astype(dtype),
        ),
        AddressableField(
            YVelocity,
            Cell,
            decomposition.regions(Cell),
            Candidate,
            scales.to_execution_velocity(v).astype(dtype),
        ),
        ZFaceFieldContext(
            AddressableField(
                VerticalVelocity,
                ZFace,
                decomposition.regions(ZFace),
                Candidate,
                scales.to_execution_velocity(w).astype(dtype),
            ),
            lower,
        ),
    )


def solver_namespace(args: argparse.Namespace) -> argparse.Namespace:
    statistics_start = 0.0 if args.quick else STATISTICS_START_HOURS
    statistics_window = args.hours if args.quick else STATISTICS_WINDOW_HOURS
    return argparse.Namespace(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        lx=4000.0,
        ly=2000.0,
        lz=1500.0,
        dt=args.dt,
        hours=args.hours,
        dtype=args.dtype,
        geostrophic_u=10.0,
        geostrophic_v=0.0,
        coriolis=F_CORIOLIS,
        horizontal_coriolis=F_CORIOLIS,
        roughness=0.1,
        von_karman=0.4,
        smagorinsky=args.smagorinsky,
        length_scale=1500.0,
        velocity_scale=10.0,
        perturbation=0.0,
        perturbation_depth=0.0,
        seed=args.seed,
        sample_every=args.sample_every,
        log_every=args.log_every,
        average_start_hours=statistics_start,
        average_window_hours=statistics_window,
        convergence_window_hours=max(statistics_window / 3.0, args.dt / 3600.0),
        convergence_tolerance=0.01,
        profile_convergence_tolerance=0.005,
        minimum_development_hours=statistics_start,
        jet_speed_ratio=1.0,
        require_supergeostrophic=False,
        stop_on_convergence=False,
        max_cfl_warning=args.max_cfl_warning,
        method=args.method,
        output=args.output,
        restart=args.restart,
    )


def _read_history(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {name: np.asarray([float(row[name]) for row in rows]) for name in rows[0]}


def write_normalized_profiles(output: Path, statistics_ustar: float) -> None:
    profile = np.genfromtxt(output / "profiles.csv", delimiter=",", names=True)
    ustar2 = statistics_ustar**2
    columns = np.column_stack(
        (
            profile["z_m"] * F_CORIOLIS / statistics_ustar,
            profile["u_m_s"] / GEOSTROPHIC_SPEED,
            profile["v_m_s"] / GEOSTROPHIC_SPEED,
            profile["u_std_m_s"] ** 2 / ustar2,
            profile["v_std_m_s"] ** 2 / ustar2,
            profile["w_std_m_s"] ** 2 / ustar2,
            profile["resolved_tke_m2_s2"] / ustar2,
            profile["resolved_uw_m2_s2"] / ustar2,
            profile["resolved_vw_m2_s2"] / ustar2,
            profile["sgs_uw_m2_s2"] / ustar2,
            profile["sgs_vw_m2_s2"] / ustar2,
            profile["total_uw_m2_s2"] / ustar2,
            profile["total_vw_m2_s2"] / ustar2,
        )
    )
    np.savetxt(
        output / "normalized_profiles.csv",
        columns,
        delimiter=",",
        header=(
            "z_f_over_ustar,u_over_ug,v_over_ug,resolved_u_variance_over_ustar2,"
            "resolved_v_variance_over_ustar2,resolved_w_variance_over_ustar2,"
            "resolved_tke_over_ustar2,resolved_uw_over_ustar2,"
            "resolved_vw_over_ustar2,sgs_uw_over_ustar2,sgs_vw_over_ustar2,"
            "total_uw_over_ustar2,total_vw_over_ustar2"
        ),
        comments="",
    )


def plot_comparison(
    output: Path,
    history: dict[str, np.ndarray],
    statistics_ustar: float,
    reference: dict,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    profile = np.genfromtxt(
        output / "normalized_profiles.csv", delimiter=",", names=True
    )
    tf = history["time_seconds"] * F_CORIOLIS
    ratios = np.asarray(tuple(reference["ustar_over_ug"].values()))
    figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    axes[0, 0].plot(tf, history["ustar"] / GEOSTROPHIC_SPEED, label="JAX-Wind")
    axes[0, 0].axhspan(
        ratios.min(), ratios.max(), color="0.7", alpha=0.35, label="1994 code envelope"
    )
    axes[0, 0].set(xlabel=r"$tf$", ylabel=r"$u_*/U_g$", title="Surface stress")
    axes[0, 0].legend()

    axes[0, 1].plot(
        tf,
        F_CORIOLIS * history["integrated_resolved_tke_m3_s2"] / statistics_ustar**3,
        label="JAX-Wind resolved",
    )
    axes[0, 1].axhline(
        reference["normalized_integrated_total_tke_plateau"],
        color="black",
        linestyle="--",
        label="1994 resolved + SGS",
    )
    axes[0, 1].set(
        xlabel=r"$tf$",
        ylabel=r"$f\int e_{res}\,dz/u_*^3$",
        title="Integrated TKE (different decompositions)",
    )
    axes[0, 1].legend()

    height = profile["z_f_over_ustar"]
    axes[1, 0].plot(profile["resolved_u_variance_over_ustar2"], height, label="u")
    axes[1, 0].plot(profile["resolved_v_variance_over_ustar2"], height, label="v")
    axes[1, 0].plot(profile["resolved_w_variance_over_ustar2"], height, label="w")
    axes[1, 0].set(
        xlabel=r"resolved variance/$u_*^2$",
        ylabel=r"$zf/u_*$",
        title="Resolved velocity variances",
    )
    axes[1, 0].legend()

    axes[1, 1].plot(profile["resolved_uw_over_ustar2"], height, label="uw")
    axes[1, 1].plot(profile["resolved_vw_over_ustar2"], height, label="vw")
    axes[1, 1].axvline(0.0, color="0.7", linewidth=0.8)
    axes[1, 1].set(
        xlabel=r"resolved momentum flux/$u_*^2$",
        ylabel=r"$zf/u_*$",
        title="Resolved momentum fluxes",
    )
    axes[1, 1].legend()
    figure.suptitle(f"Andrén et al. (1994); statistics u*={statistics_ustar:.4f} m/s")
    figure.savefig(output / "andren1994_comparison.png", dpi=180)
    plt.close(figure)


def finalize_comparison(args: argparse.Namespace, summary: dict) -> dict:
    reference = json.loads(REFERENCE_RESULTS.read_text())
    history = _read_history(args.output / "history.csv")
    start_seconds = 0.0 if args.quick else 7.0 / F_CORIOLIS
    selected = history["time_seconds"] >= start_seconds
    if not np.any(selected):
        selected = np.ones_like(history["time_seconds"], dtype=bool)
    statistics_ustar = float(np.mean(history["ustar"][selected]))
    ratio = statistics_ustar / GEOSTROPHIC_SPEED
    normalized_resolved_tke = (
        F_CORIOLIS * history["integrated_resolved_tke_m3_s2"] / statistics_ustar**3
    )
    published = np.asarray(tuple(reference["ustar_over_ug"].values()))
    write_normalized_profiles(args.output, statistics_ustar)
    plot_comparison(args.output, history, statistics_ustar, reference)
    canonical_configuration = (
        not args.quick
        and (args.nx, args.ny, args.nz) == (40, 40, 40)
        and math.isclose(args.dt, 1.0)
        and math.isclose(args.hours, CANONICAL_HOURS)
    )
    summary["reference"] = reference
    summary["comparison"] = {
        "canonical_configuration": canonical_configuration,
        "reference_acceptance_evaluated": canonical_configuration,
        "statistics_ustar_m_s": statistics_ustar,
        "ustar_over_ug": ratio,
        "published_ustar_over_ug_min": float(published.min()),
        "published_ustar_over_ug_max": float(published.max()),
        "mean_normalized_integrated_resolved_tke": float(
            np.mean(normalized_resolved_tke[selected])
        ),
        "ustar_over_ug_inside_published_envelope": (
            bool(published.min() <= ratio <= published.max())
            if canonical_configuration
            else None
        ),
        "integrated_tke_comparison": "JAX-Wind resolved only; paper value includes SGS",
    }
    summary["acceptance"]["canonical_configuration"] = canonical_configuration
    summary["acceptance"]["ustar_over_ug_inside_published_envelope"] = (
        bool(published.min() <= ratio <= published.max())
        if canonical_configuration
        else None
    )
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    solver_args = solver_namespace(args)
    summary = neutral_runner.run_case(
        solver_args,
        initial_velocity_factory=andren_initial_velocity,
        schema="jaxwind.andren1994.static-smag.v1",
        case_metadata={
            "citation": "Andren et al. (1994)",
            "doi": "10.1002/qj.49712052003",
            "latitude_degrees_north": 45.0,
            "statistics_interval_tf": [7.0, 10.0],
            "passive_scalar_included": False,
            "sgs_correspondence": "static Smagorinsky, Mason no-backscatter Cs=0.17",
        },
        emit_summary=False,
    )
    summary = finalize_comparison(args, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Uniform execution facade for declarative ABL workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import CaseConfig


def _write_manifest(
    case: CaseConfig,
    output_dir: Path,
    summary: dict[str, Any],
) -> None:
    manifest = {
        "schema": "jaxwind.abl-warmup-manifest.v1",
        "workflow": case.workflow,
        "stability": case.stability,
        "case": case.name,
        "runner": case.runner,
        "checkpoint": {
            "latest": "checkpoint_latest.npz",
            "final": (
                "checkpoint_final.npz"
                if (
                    summary.get("runtime", {}).get("reached_final_time", False)
                    and (output_dir / "checkpoint_final.npz").is_file()
                )
                else None
            ),
            "layout": "z_slab_boussinesq.v1",
            "state": [
                "velocity",
                "transported_scalar",
                "ab2_history",
                "lasd_memory",
            ],
            "statistics": "statistics_latest.npz",
        },
        "compatible_downstream_workflows": [
            "precursor",
            "wind_farm_main",
            "concurrent_precursor_main",
        ],
        "resolved_configuration": "resolved_config.toml",
        "summary": "summary.json",
        "history": "history.csv",
        "runtime": summary.get("runtime", {}),
    }
    (output_dir / "warmup_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def run_case(
    case: CaseConfig,
    *,
    output_dir: Path,
    restart: Path | None,
    max_steps: int | None,
    overwrite: bool,
) -> dict[str, Any]:
    """Run any supported ABL configuration through one workflow contract."""

    if case.benchmark is not None:
        from .andren1994 import run_case as run_benchmark

        summary = run_benchmark(
            case,
            output_dir=output_dir,
            restart=restart,
            max_steps=max_steps,
            overwrite=overwrite,
        )
    elif case.thermal.boundary_condition == "none":
        from ..pressure_driven_warmup.runner import run_case as run_neutral

        summary = run_neutral(
            case,
            output_dir=output_dir,
            restart=restart,
            max_steps=max_steps,
            overwrite=overwrite,
        )
    else:
        from ..gabls1.runner import run_case as run_stratified

        summary = run_stratified(
            case,
            output_dir=output_dir,
            restart=restart,
            max_steps=max_steps,
            overwrite=overwrite,
        )
    runtime = dict(summary.get("runtime", {}))
    runtime["restart"] = None if restart is None else str(restart)
    if "maximum_cfl" in runtime:
        runtime["cfl"] = runtime.pop("maximum_cfl")
    if "reached_configured_final_time" in runtime:
        runtime["reached_final_time"] = runtime.pop(
            "reached_configured_final_time"
        )
    surface_heat_flux = runtime.get("surface_heat_flux_k_m_s")
    if surface_heat_flux is None:
        surface_buoyancy_flux = (
            case.configured_surface_buoyancy_flux_m2_s3
        )
        if (
            surface_buoyancy_flux is None
            and case.thermal.boundary_condition == "none"
        ):
            surface_buoyancy_flux = 0.0
    else:
        surface_buoyancy_flux = (
            case.thermal.gravity_m_s2
            * float(surface_heat_flux)
            / case.thermal.reference_temperature_k
        )
    runtime["surface_buoyancy_flux_m2_s3"] = surface_buoyancy_flux
    runtime.setdefault("mean_obukhov_length_m", None)
    physics = dict(summary.get("physics", {}))
    physics.setdefault("momentum_sgs", "LASD")
    physics.setdefault("scalar_sgs", "LASD")
    physics.setdefault("stability", case.stability)
    physics.setdefault("surface", case.wall.model)
    physics.setdefault(
        "reference_frame",
        (
            "geostrophic translating"
            if case.thermal.boundary_condition
            == "prescribed_surface_temperature"
            else "stationary"
        ),
    )
    physics.setdefault("scalar_advection", "conservative")
    normalized = {
        "schema": "jaxwind.abl-warmup.v1",
        "case": {
            "name": case.name,
            "runner": case.runner,
            "workflow": case.workflow,
            "stability": case.stability,
        },
        "configuration": case.resolved(),
        "physics": physics,
        "runtime": runtime,
        "workflow": {
            "runner": "abl",
            "kind": case.workflow,
            "stability": case.stability,
            "manifest": "warmup_manifest.json",
        },
    }
    if case.benchmark is not None:
        normalized["benchmark"] = {
            "name": case.benchmark.name,
            "result_schema": summary.get("schema"),
            "comparison": summary.get("comparison", {}),
            "acceptance": summary.get("acceptance", {}),
            "final": summary.get("final", {}),
            "reference": summary.get("reference", {}),
        }
    summary = normalized
    (output_dir / "resolved_config.toml").write_text(case.resolved_toml())
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    _write_manifest(case, output_dir, summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


__all__ = ["run_case"]

"""Separate interior upwind and physical scalar-variance dissipation.

On the benchmark's uniform grid, first-order upwind equals centred advection
plus face diffusivity |u| dx/2. This is a spatial operator identity, not the
modified equation of a full time step. Boundary transport, particle sources,
time integration and phase change are excluded from these instantaneous sums.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxwind import FREE_SLIP, OPEN, Boundaries, StaggeredVelocity, Wall
from jaxwind.domain import UniformGrid
from jaxwind.io.checkpoint import checkpoint_metadata
from jaxwind.sgs import AnisotropicMinimumDissipation, eddy_viscosity


def audit(directory):
    summary = json.loads((directory / "summary.json").read_text())
    if summary["status"] != "complete":
        raise ValueError(f"Incomplete run: {directory}")
    path = directory / "checkpoint.npz"
    doc = checkpoint_metadata(path)["resolved_case"]
    jax.config.update("jax_enable_x64", True)
    grid = UniformGrid(*doc["mesh"]["cells"], *doc["mesh"]["lengths_m"])
    with np.load(path, allow_pickle=False) as data:
        velocity = StaggeredVelocity(
            *(jnp.asarray(data[f"state/velocity/{a}"]) for a in "xyz")
        )
        fields = {
            "temperature_anomaly": np.asarray(data["state/scalar"]),
            "vapor_mixing_ratio": np.asarray(data["state/moisture/vapor"]),
        }
    boundaries = Boundaries(
        Wall(FREE_SLIP), Wall(FREE_SLIP), streamwise=OPEN, spanwise=FREE_SLIP
    )
    nu = np.asarray(
        jax.jit(
            lambda v: eddy_viscosity(
                v, grid, boundaries, AnisotropicMinimumDissipation()
            )
        )(velocity)
    )
    # These are the benchmark's fixed scalar transport parameters.
    from jaxwind.config.moisture import load_moisture

    moist, _ = load_moisture(doc["physics"])
    molecular = moist.thermodynamics.vapor_diffusivity
    turbulent = nu / 0.7
    dimensions = (grid.dx, grid.dy, grid.dz)
    volume = np.prod(dimensions)
    result = {}
    for name, field in fields.items():
        axes = {}
        for label, axis, spacing, face_velocity in zip(
            "xyz", (2, 1, 0), dimensions, velocity
        ):
            left, right, interior = (
                [slice(None)] * 3,
                [slice(None)] * 3,
                [slice(None)] * 3,
            )
            left[axis], right[axis], interior[axis] = (
                slice(None, -1),
                slice(1, None),
                slice(1, -1),
            )
            jump = field[tuple(right)] - field[tuple(left)]
            numerical = (
                0.5 * np.abs(np.asarray(face_velocity)[tuple(interior)]) * spacing
            )
            physical = 0.5 * (turbulent[tuple(left)] + turbulent[tuple(right)])
            weight = (volume / spacing**2) * jump**2
            axes[label] = {
                "upwind_variance_loss": float(np.sum(numerical * weight)),
                "amd_variance_loss": float(np.sum(physical * weight)),
                "molecular_variance_loss": float(np.sum(molecular * weight)),
                "upwind_face_diffusivity_percentiles_0_50_95_100_m2_s": np.percentile(
                    numerical, [0, 50, 95, 100]
                ).tolist(),
            }
        totals = {
            key: sum(a[key] for a in axes.values())
            for key in (
                "upwind_variance_loss",
                "amd_variance_loss",
                "molecular_variance_loss",
            )
        }
        result[name] = {
            "axes": axes,
            "totals": totals,
            "upwind_over_amd_variance_loss": totals["upwind_variance_loss"]
            / totals["amd_variance_loss"],
        }
    return {
        "directory": str(directory),
        "time_s": summary["time_seconds"],
        "fields": result,
        "interpretation": (
            "Positive loss of half the integrated squared scalar, interior faces only. "
            "Units K^2 m^3/s for temperature, (kg/kg)^2 m^3/s for vapor. "
            "Exact uniform-grid spatial upwind-minus-centred identity, not a time-discrete "
            "effective diffusivity or full scalar budget. Does not establish the sign "
            "or magnitude of a sensor-temperature bias."
        ),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [audit(d) for d in args.runs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    for result in results:
        print(
            result["directory"],
            {
                k: v["upwind_over_amd_variance_loss"]
                for k, v in result["fields"].items()
            },
        )


if __name__ == "__main__":
    main()

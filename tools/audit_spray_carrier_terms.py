"""Native instantaneous axial momentum terms from a saved uniform LES field.

No advancement. Source projection is loaded from an independent saved replay.
Stored carrier pressure is lagged; terms are not a closed RK-stage budget.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxwind import FREE_SLIP, OPEN, Boundaries, StaggeredVelocity, Wall
from jaxwind.domain import UniformGrid
from jaxwind.numerics.discretization import (
    _centered_cells_to_faces,
    diffusion,
    pressure_gradient,
)
from jaxwind.numerics.momentum import _muscl_flux, muscl_advection
from jaxwind.sgs import AnisotropicMinimumDissipation, subfilter_tendency
from jaxwind.smooth_wall import smooth_duct_tendency


def make_evaluator(grid, bc, nu):
    @jax.jit
    def evaluate(v, p):
        ux = _muscl_flux(
            v.x, 0.5 * (v.x[..., :-1] + v.x[..., 1:]), 2, False, internal=True
        )
        uy = _muscl_flux(
            v.x,
            _centered_cells_to_faces(v.y, 2, periodic=False, boundary="copy"),
            1,
            False,
        )
        uz = _muscl_flux(
            v.x,
            _centered_cells_to_faces(v.z, 2, periodic=False, boundary="copy"),
            0,
            False,
        )
        ax = -jnp.pad(jnp.diff(ux, axis=2) / grid.dx, ((0, 0), (0, 0), (1, 1)))
        ar = -jnp.diff(uy, axis=1) / grid.dy - jnp.diff(uz, axis=0) / grid.dz
        ar = ar.at[..., 0].set(0.0).at[..., -1].set(0.0)
        error = jnp.max(jnp.abs(ax + ar - muscl_advection(v, grid).x))
        sg, _ = subfilter_tendency(v, grid, bc, AnisotropicMinimumDissipation())
        fields = jnp.stack(
            (
                ax,
                ar,
                sg.x,
                diffusion(v, grid, bc, nu).x,
                -pressure_gradient(p, grid, periodic_x=False, periodic_y=False).x,
                smooth_duct_tendency(v, grid, nu).x,
            )
        )
        return 0.5 * (fields[..., :-1] + fields[..., 1:]), error

    return evaluate


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--source-projection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.default_backend() == "gpu"
    jax.config.update("jax_enable_x64", True)
    checkpoint = np.load(args.run / "checkpoint.npz")
    doc = json.loads(str(checkpoint["metadata"]))["resolved_case"]
    assert doc["case"].get("carrier_turbulence_model", "les") == "les"
    assert doc["case"].get("carrier_sgs_model", "amd") == "amd"
    assert doc["numerics"].get("momentum_advection_scheme", "muscl-mc") == "muscl-mc"
    grid = UniformGrid(*doc["mesh"]["cells"], *doc["mesh"]["lengths_m"])
    bc = Boundaries(
        Wall(FREE_SLIP), Wall(FREE_SLIP), streamwise=OPEN, spanwise=FREE_SLIP
    )
    velocity = StaggeredVelocity(
        *(jnp.asarray(checkpoint["state/velocity/" + k]) for k in "xyz")
    )
    pressure = jnp.asarray(checkpoint["state/pressure"])
    nu = doc["physics"]["flow"]["kinematic_viscosity_m2_s"]

    evaluate = make_evaluator(grid, bc, nu)

    fields, error = jax.device_get(evaluate(velocity, pressure))
    assert float(error) < 1e-9, error
    names = [
        "axial_convection",
        "transverse_convection",
        "AMD",
        "molecular",
        "stored_carrier_pressure",
        "wall",
    ]
    source = np.load(args.source_projection / "fields.npz")
    fields = np.concatenate((fields, source["projected"][0, None]), axis=0)
    names.append("projected_parcel_source")
    fields = np.concatenate((fields, np.sum(fields, axis=0, keepdims=True)), axis=0)
    names.append("snapshot_sum")
    x, y, z = map(np.asarray, (grid.x_centers, grid.y_centers, grid.z_centers))
    j, k = grid.ny // 2, grid.nz // 2
    centre = 0.25 * (
        fields[:, k - 1, j - 1]
        + fields[:, k - 1, j]
        + fields[:, k, j - 1]
        + fields[:, k, j]
    )
    radius = np.hypot(y[None, :] - grid.ly / 2, z[:, None] - grid.lz / 2)
    rho = doc["physics"]["moisture"]["dry_air_density_kg_m3"]
    force = (
        np.sum(fields * (radius < 0.05)[None, :, :, None], axis=(1, 2))
        * rho
        * grid.dy
        * grid.dz
    )
    stations = [0.1, 0.2, 0.4, 0.7, 1.0, 1.3, 1.6, 1.85]
    result = {
        "time_s": float(checkpoint["state/time"]),
        "mesh": doc["mesh"]["cells"],
        "native_advection_reconstruction_error_m_s2": float(error),
        "limitation": "Instantaneous terms evaluated at saved state; stored carrier pressure is lagged. Sum is not the exact split RK timestep derivative. No temporal Reynolds decomposition.",
        "centre_acceleration_m_s2": {
            name: dict(zip(map(str, stations), np.interp(stations, x, line).tolist()))
            for name, line in zip(names, centre)
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(
        args.output / "profiles.npz",
        names=names,
        x_m=x,
        centre_acceleration_m_s2=centre,
        core_force_per_length_N_m=force,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

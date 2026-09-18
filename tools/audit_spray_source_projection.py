"""Replay one saved parcel source and isolate its linear pressure response.

No physical timestep is accepted. Homogeneous boundary data isolate the impulse
from the carrier's existing divergence and time-dependent prescribed inflow.
This measures the source projection, not pressure during subsequent transport.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from audit_spray_radial_momentum import make_sampler

from jaxwind.config.document import load_case
from jaxwind.io.checkpoint import load_checkpoint
from jaxwind.numerics.discretization import cell_velocity, divergence
from jaxwind.numerics.poisson import build_pressure_poisson, project
from jaxwind.open_boundary import InflowPlane, enforce_open_velocity
from jaxwind.simulation.water_spray_benchmark import build_simulation


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert jax.default_backend() == "gpu", "GPU required"
    jax.config.update("jax_enable_x64", True)
    case = load_case(args.run / "resolved_case.toml")
    sim = build_simulation(case)
    state, _, _ = load_checkpoint(
        args.run / "checkpoint.npz", sim.initial_state, fingerprint=case.fingerprint
    )
    grid = sim.grid
    dt = case.document["time"]["dt_seconds"]
    rho = case.document["physics"]["moisture"]["dry_air_density_kg_m3"]
    replay = make_sampler(grid, case.document, source_impulse_only=True)[0]
    raw = replay(state)
    zero = InflowPlane(
        jnp.zeros((grid.nz, grid.ny)),
        jnp.zeros((grid.nz, grid.ny + 1)),
        jnp.zeros((grid.nz + 1, grid.ny)),
        jnp.zeros((grid.nz, grid.ny)),
    )
    constrained = enforce_open_velocity(raw, zero, grid)
    poisson = build_pressure_poisson(
        grid,
        backend="gmg",
        periodic_x=False,
        periodic_y=False,
        dtype="float64",
        config={"tolerance": 1e-7},
    )
    corrected, pressure = jax.jit(lambda v: project(v, poisson, dt))(constrained)
    arrays = {
        name: np.stack(jax.device_get(cell_velocity(v))) / dt
        for name, v in [
            ("raw", raw),
            ("boundary_constrained", constrained),
            ("projected", corrected),
        ]
    }
    arrays["pressure_response"] = arrays["projected"] - arrays["boundary_constrained"]
    y, z, x = map(np.asarray, (grid.y_centers, grid.z_centers, grid.x_centers))
    radius = np.hypot(y[None, :] - grid.ly / 2, z[:, None] - grid.lz / 2)
    volume = np.asarray(grid.cell_volumes)
    regions = {
        "all": np.ones_like(volume, dtype=bool),
        "core": np.broadcast_to(radius[..., None] < 0.05, volume.shape),
        "downstream_core": (radius[..., None] < 0.05) & (x > 0.4),
        "near_nozzle": np.broadcast_to(x < 0.1, volume.shape),
    }
    report = {
        "time_s": float(state.time),
        "dt_s": dt,
        "mesh": list(case.document["mesh"]["cells"]),
        "scope": "Instantaneous replay; pressure response excludes carrier transport pressure.",
        "max_projected_impulse_divergence_s-1": float(
            jnp.max(jnp.abs(divergence(corrected, grid)))
        ),
        "regions_axial_equivalent_force_N": {
            region: {
                name: float(np.sum(a[0] * volume * mask) * rho)
                for name, a in arrays.items()
            }
            for region, mask in regions.items()
        },
    }
    # Linearity check includes a nonzero base field and identical affine BCs.
    plane = InflowPlane(
        state.velocity.x[..., 0],
        state.velocity.y[..., 0],
        state.velocity.z[..., 0],
        state.scalar[..., 0],
    )

    def projected(v):
        return project(enforce_open_velocity(v, plane, grid), poisson, dt)[0]

    base = jax.jit(projected)(state.velocity)
    loaded = jax.jit(projected)(jax.tree.map(lambda a, b: a + b, state.velocity, raw))
    report["projection_linearity_max_velocity_error_m_s"] = max(
        float(jnp.max(jnp.abs(a - b - c))) for a, b, c in zip(loaded, base, corrected)
    )
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "fields.npz",
        **arrays,
        x_m=x,
        y_m=y,
        z_m=z,
        kinematic_impulse_pressure=np.asarray(pressure),
    )
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

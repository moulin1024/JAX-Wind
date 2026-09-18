"""Sample coarse LES axial momentum terms and compare with measured storage.

Endpoint RHS sampling leaves RK/splitting and quadrature residuals explicit.
It must not be described as an exactly closed stage-by-stage discrete budget.
"""

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from audit_spray_carrier_terms import make_evaluator
from audit_spray_radial_momentum import make_sampler

from jaxwind import FREE_SLIP, OPEN, Boundaries, Wall
from jaxwind.config.document import load_case
from jaxwind.io.checkpoint import load_checkpoint
from jaxwind.numerics.discretization import cell_velocity
from jaxwind.numerics.poisson import build_pressure_poisson, project
from jaxwind.open_boundary import InflowPlane, enforce_open_velocity
from jaxwind.simulation.api import RunControls
from jaxwind.simulation.water_spray_benchmark import build_simulation


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=1.0)
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()
    assert jax.default_backend() == "gpu"
    jax.config.update("jax_enable_x64", True)
    case = load_case(args.run / "resolved_case.toml")
    doc = case.document
    assert doc["case"].get("carrier_turbulence_model", "les") == "les"
    sim = build_simulation(case)
    g = sim.grid
    dt = doc["time"]["dt_seconds"]
    state, _, _ = load_checkpoint(
        args.run / "checkpoint.npz", sim.initial_state, fingerprint=case.fingerprint
    )
    bc = Boundaries(
        Wall(FREE_SLIP), Wall(FREE_SLIP), streamwise=OPEN, spanwise=FREE_SLIP
    )
    carrier = make_evaluator(g, bc, doc["physics"]["flow"]["kinematic_viscosity_m2_s"])
    replay = make_sampler(g, doc, source_impulse_only=True)[0]
    poisson = build_pressure_poisson(
        g,
        backend="gmg",
        periodic_x=False,
        periodic_y=False,
        dtype="float64",
        config={"tolerance": 1e-7},
    )
    zero = InflowPlane(
        jnp.zeros((g.nz, g.ny)),
        jnp.zeros((g.nz, g.ny + 1)),
        jnp.zeros((g.nz + 1, g.ny)),
        jnp.zeros((g.nz, g.ny)),
    )
    r = jnp.hypot(
        jnp.asarray(g.y_centers)[None, :] - g.ly / 2,
        jnp.asarray(g.z_centers)[:, None] - g.lz / 2,
    )
    mask = (r < 0.05)[None, :, :, None]
    rho = doc["physics"]["moisture"]["dry_air_density_kg_m3"]
    j, k = g.ny // 2, g.nz // 2

    def reduce(a):
        centre = 0.25 * (
            a[:, k - 1, j - 1] + a[:, k - 1, j] + a[:, k, j - 1] + a[:, k, j]
        )
        core = jnp.sum(a * mask, axis=(1, 2)) * rho * g.dy * g.dz
        return jnp.stack((centre, core))

    @jax.jit
    def sample(s):
        terms, error = carrier(s.velocity, s.pressure)
        impulse, _ = project(enforce_open_velocity(replay(s), zero, g), poisson, dt)
        source = cell_velocity(impulse)[0] / dt
        return (
            reduce(jnp.concatenate((terms, source[None]))),
            reduce(cell_velocity(s.velocity)[0][None]),
            error,
        )

    n = round(args.seconds / dt / args.stride)
    assert n >= 2 and n % 2 == 0
    samples = []
    velocity = []
    times = []
    errors = []
    began = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    for i in range(n + 1):
        t, v, e = jax.device_get(sample(state))
        assert np.isfinite(t).all()
        assert float(e) < 1e-10
        samples.append(t)
        velocity.append(v)
        times.append(float(state.time))
        errors.append(float(e))
        if i % 50 == 0:
            print(
                json.dumps(
                    {
                        "sample": i,
                        "total": n,
                        "time_s": times[-1],
                        "elapsed_s": time.monotonic() - began,
                    }
                ),
                flush=True,
            )
            np.savez_compressed(
                args.output / "samples.npz",
                terms=samples,
                velocity=velocity,
                time_s=times,
                x_m=g.x_centers,
            )
        if i < n:
            state = sim.advance(
                state,
                RunControls(
                    count=args.stride, target_time=times[-1] + args.stride * dt
                ),
            )
    terms = np.asarray(samples)
    vel = np.asarray(velocity)
    times = np.asarray(times)
    mean = np.trapezoid(terms, times, axis=0) / (times[-1] - times[0])
    storage = (vel[-1] - vel[0]) / (times[-1] - times[0])
    residual = storage[:, 0] - mean.sum(axis=1)
    # Coarsening the same sequence estimates sampling sensitivity independently of timestep error.
    sparse = np.trapezoid(terms[::2], times[::2], axis=0) / (times[-1] - times[0])
    names = [
        "axial_convection",
        "transverse_convection",
        "AMD",
        "molecular",
        "stored_carrier_pressure",
        "wall",
        "projected_parcel_source",
    ]
    np.savez_compressed(
        args.output / "statistics.npz",
        names=names,
        mean=mean,
        storage=storage,
        residual=residual,
        quadrature_change=sparse - mean,
        x_m=g.x_centers,
        time_s=times,
    )
    meta = {
        "interval_s": [times[0], times[-1]],
        "samples": n + 1,
        "spacing_s": args.stride * dt,
        "elapsed_s": time.monotonic() - began,
        "maximum_native_reconstruction_error": max(errors),
        "axes": ["centre acceleration m/s2", "core force per length N/m"],
        "limitations": "Endpoint sampled terms, not exact RK-stage ledger; residual includes splitting, lagged-pressure, inflow and quadrature effects. One-second mean is not statistical convergence.",
    }
    (args.output / "summary.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()

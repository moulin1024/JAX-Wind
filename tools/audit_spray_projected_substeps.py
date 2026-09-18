"""Frozen source convergence with end-only versus substep pressure projection.

All trials use the same initial checkpoint, injected parcels and physical dt.
No carrier advancement is accepted. Total deposited force excludes pressure;
velocity-increment comparisons include the incompressibility projection.
"""

import argparse
import copy
import json
from pathlib import Path

import jax
import numpy as np
from audit_spray_radial_momentum import make_sampler

from jaxwind.config.document import load_case
from jaxwind.io.checkpoint import load_checkpoint
from jaxwind.numerics.discretization import cell_velocity
from jaxwind.numerics.poisson import build_pressure_poisson, project
from jaxwind.open_boundary import InflowPlane, enforce_open_velocity
from jaxwind.simulation.water_spray_benchmark import build_simulation


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--boundary-only", action="store_true")
    args = ap.parse_args()
    assert jax.default_backend() == "gpu"
    jax.config.update("jax_enable_x64", True)
    case = load_case(args.run / "resolved_case.toml")
    sim = build_simulation(case)
    g = sim.grid
    state, _, _ = load_checkpoint(
        args.run / "checkpoint.npz", sim.initial_state, fingerprint=case.fingerprint
    )
    dt = case.document["time"]["dt_seconds"]
    plane = InflowPlane(
        state.velocity.x[..., 0],
        state.velocity.y[..., 0],
        state.velocity.z[..., 0],
        state.scalar[..., 0],
    )
    poisson = build_pressure_poisson(
        g,
        backend="gmg",
        periodic_x=False,
        periodic_y=False,
        dtype="float64",
        config={"tolerance": 1e-7},
    )

    @jax.jit
    def projector(v, h):
        return project(enforce_open_velocity(v, plane, g), poisson, h)[0]

    base = projector(state.velocity, dt)
    if args.boundary_only:
        report = {}
        mask = np.asarray(g.x_centers) < 0.1
        for count in [1, 4, 16, 64]:

            @jax.jit
            def repeat(v, count=count):
                return jax.lax.fori_loop(
                    0, count, lambda i, u: projector(u, dt / count), v
                )

            final = repeat(state.velocity)
            difference = np.stack(
                jax.device_get(
                    cell_velocity(jax.tree.map(lambda a, b: a - b, final, base))
                )
            )
            report[count] = {
                "max_abs_velocity_difference_m_s": float(np.max(abs(difference))),
                "L2_velocity_difference": float(np.linalg.norm(difference)),
                "near_nozzle_max_abs_difference_m_s": float(
                    np.max(abs(difference[..., mask]))
                ),
                "near_nozzle_L2_difference": float(
                    np.linalg.norm(difference[..., mask])
                ),
            }
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "boundary_only.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(report, indent=2), flush=True)
        return
    values = {}
    forces = {}
    for mode in ["end_only", "each_substep"]:
        for count in [1, 4, 16, 64]:
            doc = copy.deepcopy(case.document)
            doc["case"]["parcel_substeps"] = count
            delta, force = make_sampler(
                g,
                doc,
                source_impulse_only=True,
                return_source_totals=True,
                substep_projector=projector if mode == "each_substep" else None,
            )[0](state)
            final = jax.tree.map(lambda a, b: a + b, state.velocity, delta)
            if mode == "end_only":
                final = projector(final, dt)
            increment = jax.tree.map(lambda a, b: a - b, final, base)
            values[mode, count] = np.stack(jax.device_get(cell_velocity(increment)))
            forces[mode, count] = np.asarray(force)
            print("Completed", mode, count, flush=True)
    reference = values["each_substep", 64]
    mask = np.asarray(g.x_centers) < 0.1
    results = {}
    for (mode, count), v in values.items():
        error = v - reference
        results[f"{mode}_{count}"] = {
            "max_abs_increment_m_s": float(np.max(abs(v))),
            "max_abs_error_vs_projected64_m_s": float(np.max(abs(error))),
            "L2_error_vs_projected64": float(
                np.linalg.norm(error) / np.linalg.norm(reference)
            ),
            "near_nozzle_L2_error_vs_projected64": float(
                np.linalg.norm(error[..., mask]) / np.linalg.norm(reference[..., mask])
            ),
            "drag_force_N": float(forces[mode, count][0]),
            "evap_momentum_force_N": float(forces[mode, count][1]),
        }
    single = float(np.max(abs(values["end_only", 1] - values["each_substep", 1])))
    assert single < 1e-7, single
    report = {
        "mesh": case.document["mesh"]["cells"],
        "dt_s": dt,
        "time_s": float(state.time),
        "single_step_equivalence_error_m_s": single,
        "scope": __doc__,
        "results": results,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

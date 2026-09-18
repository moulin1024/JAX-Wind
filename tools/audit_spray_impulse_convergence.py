"""Frozen checkpoint source-velocity convergence, including fresh injection.

Replays the same physical timestep with increasing parcel substeps. No state is
accepted and no carrier transport occurs. Tests local coupling, not full runs.
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
from jaxwind.simulation.water_spray_benchmark import build_simulation


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    assert jax.default_backend() == "gpu"
    jax.config.update("jax_enable_x64", True)
    case = load_case(args.run / "resolved_case.toml")
    sim = build_simulation(case)
    state, _, _ = load_checkpoint(
        args.run / "checkpoint.npz", sim.initial_state, fingerprint=case.fingerprint
    )
    values = {}
    force = {}
    dt = case.document["time"]["dt_seconds"]
    rho = case.document["physics"]["moisture"]["dry_air_density_kg_m3"]
    for count in [1, 4, 16, 64]:
        doc = copy.deepcopy(case.document)
        doc["case"]["parcel_substeps"] = count
        delta = make_sampler(sim.grid, doc, source_impulse_only=True)[0](state)
        v = np.stack(jax.device_get(cell_velocity(delta)))
        values[count] = v
        force[count] = (
            np.sum(v * np.asarray(sim.grid.cell_volumes)[None], axis=(1, 2, 3))
            * rho
            / dt
        )
        print("Completed source substeps", count, flush=True)
    reference = values[64]
    mask = np.asarray(sim.grid.x_centers) < 0.1
    results = {}
    for count, v in values.items():
        error = v - reference
        results[count] = {
            "max_abs_velocity_increment_m_s": float(np.max(abs(v))),
            "max_abs_error_vs64_m_s": float(np.max(abs(error))),
            "L2_relative_error_vs64": float(
                np.linalg.norm(error) / np.linalg.norm(reference)
            ),
            "near_nozzle_L2_relative_error_vs64": float(
                np.linalg.norm(error[..., mask]) / np.linalg.norm(reference[..., mask])
            ),
            "total_axial_force_N": float(force[count][0]),
            "axial_force_relative_error_vs64": float(
                (force[count][0] - force[64][0]) / force[64][0]
            ),
        }
    report = {
        "time_s": float(state.time),
        "dt_s": dt,
        "mesh": case.document["mesh"]["cells"],
        "scope": __doc__,
        "results": results,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

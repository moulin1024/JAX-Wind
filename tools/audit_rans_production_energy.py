"""Compare collocated RANS production with native stress-strain power.

Uniform-grid diagnostic only. Native shear powers are averaged after products;
normal powers live at cell centres. Wall-model production is separate.
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxwind import FREE_SLIP, OPEN, Boundaries, StaggeredVelocity, Wall
from jaxwind.domain import UniformGrid
from jaxwind.rans_kepsilon import KEpsilonState, production, turbulent_viscosity
from jaxwind.rans_realizable import turbulent_viscosity as realizable_viscosity
from jaxwind.sgs import (
    _to_cell_from_xy_edge,
    _to_cell_from_xz_edge,
    _to_cell_from_yz_edge,
    _to_xy_edge_from_cell,
    _to_xz_edge_from_cell,
    _to_yz_edge_from_cell,
    cell_gradients,
    edge_gradients,
)


def native_production(velocity, grid, boundaries, viscosity):
    g = edge_gradients(velocity, grid, boundaries)
    value = 2 * viscosity * (g["xx"] ** 2 + g["yy"] ** 2 + g["zz"] ** 2)
    value += _to_cell_from_xy_edge(
        _to_xy_edge_from_cell(viscosity, open_x=True, wall_y=True)
        * (g["xy"] + g["yx"]) ** 2,
        open_x=True,
        wall_y=True,
    )
    value += _to_cell_from_xz_edge(
        _to_xz_edge_from_cell(viscosity, open_x=True) * (g["xz"] + g["zx"]) ** 2,
        open_x=True,
    )
    value += _to_cell_from_yz_edge(
        _to_yz_edge_from_cell(viscosity, wall_y=True) * (g["yz"] + g["zy"]) ** 2,
        wall_y=True,
    )
    return value


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    jax.config.update("jax_enable_x64", True)
    a = np.load(args.run / "checkpoint.npz")
    doc = json.loads(str(a["metadata"]))["resolved_case"]
    g = UniformGrid(*doc["mesh"]["cells"], *doc["mesh"]["lengths_m"])
    bc = Boundaries(
        Wall(FREE_SLIP), Wall(FREE_SLIP), streamwise=OPEN, spanwise=FREE_SLIP
    )
    vel = StaggeredVelocity(*(jnp.asarray(a["state/velocity/" + k]) for k in "xyz"))
    turb = KEpsilonState(
        *(jnp.asarray(a["state/turbulence/" + k]) for k in KEpsilonState._fields)
    )
    if doc["case"]["carrier_turbulence_model"] == "realizable-k-epsilon":
        nut = realizable_viscosity(turb, vel, g, bc)
    else:
        nut = turbulent_viscosity(turb)
    grad = cell_gradients(edge_gradients(vel, g, bc))
    old = np.asarray(
        nut
        * sum(0.5 * (grad[i][j] + grad[j][i]) ** 2 for i in range(3) for j in range(3))
    )
    native = np.asarray(native_production(vel, g, bc, nut))
    np.testing.assert_allclose(
        np.asarray(production(vel, g, bc, nut)), native, rtol=1e-13, atol=1e-13
    )
    rho = doc["physics"]["moisture"]["dry_air_density_kg_m3"]
    x = np.asarray(g.x_centers)
    r = np.hypot(
        np.asarray(g.y_centers)[None, :] - g.ly / 2,
        np.asarray(g.z_centers)[:, None] - g.lz / 2,
    )
    interior = np.zeros_like(old, dtype=bool)
    interior[1:-1, 1:-1, 1:-1] = True
    masks = {
        "interior": interior,
        "near_nozzle": interior & (x < 0.1),
        "near_nozzle_core": interior & (x < 0.1) & (r[..., None] < 0.05),
        "downstream_core": interior & (x > 0.4) & (r[..., None] < 0.05),
    }
    result = {}
    for name, mask in masks.items():
        o = float(np.sum(old * mask) * rho * g.dx * g.dy * g.dz)
        n = float(np.sum(native * mask) * rho * g.dx * g.dy * g.dz)
        result[name] = {
            "collocated_production_W": o,
            "native_stress_strain_power_W": n,
            "relative_difference": (n - o) / n if n else None,
        }
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "time_s": float(a["state/time"]),
        "regions": result,
        "limitation": "Native quadrature versus collocated square. Excludes wall-adjacent cells and wall-function production. Does not itself close the open-boundary kinetic energy flux balance.",
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

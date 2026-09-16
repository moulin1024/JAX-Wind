"""Compare instantaneous momentum energy tendencies in a spray checkpoint.

The limited-minus-central difference includes boundary-discretization effects;
it is a numerical damping diagnostic, not a closed turbulence-energy budget.
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
from jaxwind.numerics.discretization import advection, diffusion
from jaxwind.numerics.momentum import muscl_advection
from jaxwind.sgs import AnisotropicMinimumDissipation, subfilter_tendency
from jaxwind.smooth_wall import smooth_duct_tendency


def audit(directory):
    path = directory / "checkpoint.npz"
    doc = checkpoint_metadata(path)["resolved_case"]
    jax.config.update("jax_enable_x64", True)
    grid = UniformGrid(*doc["mesh"]["cells"], *doc["mesh"]["lengths_m"])
    with np.load(path, allow_pickle=False) as data:
        velocity = StaggeredVelocity(
            *(jnp.asarray(data[f"state/velocity/{a}"]) for a in "xyz")
        )
        time = float(data["state/time"])
    boundaries = Boundaries(
        Wall(FREE_SLIP), Wall(FREE_SLIP), streamwise=OPEN, spanwise=FREE_SLIP
    )
    rho = doc["physics"]["moisture"]["dry_air_density_kg_m3"]
    viscosity = doc["physics"]["flow"]["kinematic_viscosity_m2_s"]
    weights = []
    for component, axis in zip(velocity, (2, 1, 0)):
        shape = [1, 1, 1]
        shape[axis] = component.shape[axis]
        normal_weights = jnp.ones(component.shape[axis]).at[0].set(0.5).at[-1].set(0.5)
        weights.append(normal_weights.reshape(shape) * grid.dx * grid.dy * grid.dz)

    def power(tendency):
        return rho * sum(
            jnp.sum(u * f * w) for u, f, w in zip(velocity, tendency, weights)
        )

    @jax.jit
    def evaluate(v):
        central = advection(v, grid)
        limited = muscl_advection(v, grid)
        sgs, nu = subfilter_tendency(
            v, grid, boundaries, AnisotropicMinimumDissipation()
        )
        molecular = diffusion(v, grid, boundaries, viscosity)
        wall = smooth_duct_tendency(v, grid, viscosity)
        return {
            "central_advection_power_w": power(central),
            "limited_advection_power_w": power(limited),
            "limited_minus_central_advection_power_w": power(
                StaggeredVelocity(*(a - b for a, b in zip(limited, central)))
            ),
            "amd_subfilter_power_w": power(sgs),
            "molecular_diffusion_power_w": power(molecular),
            "smooth_wall_power_w": power(wall),
            "volume_mean_amd_viscosity_m2_s": jnp.mean(nu),
            "fraction_cells_with_zero_amd_viscosity": jnp.mean(nu < 1e-12),
            "outlet_mean_amd_viscosity_m2_s": jnp.mean(nu[..., -1]),
        }

    result = {key: float(value) for key, value in evaluate(velocity).items()}
    return {
        "directory": str(directory),
        "time_seconds": time,
        **result,
        "interpretation": (
            "Instantaneous MAC energy tendencies with half-volume normal boundary weights. "
            "Advection powers include open-boundary transport; their difference includes boundary discretization. "
            "Not a full energy budget or a measurement of resolved turbulent flux. Negative power removes kinetic energy."
        ),
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

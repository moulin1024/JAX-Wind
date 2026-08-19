"""Pressure-driven neutral boundary layer on the finite-volume solver.

A half-channel driven by a constant horizontal pressure gradient, periodic in
x and y, frictionless and impermeable at the top, and closed at the bottom by
the Monin-Obukhov surface stress of a neutral logarithmic layer.  At
equilibrium the driving force balances the surface drag, ``f = u_*^2 / H``, and
the mean wind should follow ``U(z) = (u_* / kappa) ln(z / z_0)``.

The diagnostic is the non-dimensional shear ``Phi = (kappa z / u_*) dU/dz``,
which is one wherever the logarithmic law holds.

    python tools/run_fv_neutral_abl.py --cells 64 64 64 --steps 20000

The AMG backend needs the environment described in tools/run_fv_channel.py.
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from jaxwind.domain import UniformGrid
from jaxwind.fv import (
    CELL_AVERAGE,
    CELL_CENTRE,
    LOCAL,
    PLANAR,
    AnisotropicMinimumDissipation,
    FlowModel,
    MoninObukhovWall,
    StaggeredVelocity,
    build_pressure_poisson,
    build_run,
    build_step,
    courant_number,
    divergence,
    friction_velocity,
    initial_solution,
    logarithmic_profile,
    monin_obukhov_boundaries,
    stable_timestep,
)


def initial_field(
    grid: UniformGrid,
    model: MoninObukhovWall,
    friction: float,
    noise: float,
    seed: int,
    dtype: str,
) -> StaggeredVelocity:
    """The equilibrium logarithmic profile, perturbed to trip turbulence."""
    profile = logarithmic_profile(grid, friction, model).astype(dtype)
    cells = (grid.nz, grid.ny, grid.nx)
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    scale = noise * friction
    # Perturb only the interior so the walls stay impermeable.
    vertical = jnp.zeros((grid.nz + 1, grid.ny, grid.nx), dtype)
    vertical = vertical.at[1:-1].set(
        scale * jax.random.normal(keys[2], (grid.nz - 1, grid.ny, grid.nx), dtype)
    )
    return StaggeredVelocity(
        jnp.broadcast_to(profile[:, None, None], cells)
        + scale * jax.random.normal(keys[0], cells, dtype),
        scale * jax.random.normal(keys[1], cells, dtype),
        vertical,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, nargs=3, default=(64, 64, 64))
    parser.add_argument("--lengths", type=float, nargs=3, default=(2000.0, 2000.0, 1000.0))
    parser.add_argument("--friction", type=float, default=0.4)
    parser.add_argument("--roughness", type=float, default=0.005)
    parser.add_argument("--von-karman", type=float, default=0.4)
    parser.add_argument("--sampling", choices=(CELL_AVERAGE, CELL_CENTRE), default=CELL_AVERAGE)
    parser.add_argument("--averaging", choices=(LOCAL, PLANAR), default=LOCAL)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--spinup", type=float, default=0.5, help="fraction discarded")
    parser.add_argument("--chunk", type=int, default=100)
    parser.add_argument(
        "--profile-every",
        type=int,
        default=0,
        help="print the instantaneous mean profile every N steps (0 disables)",
    )
    parser.add_argument("--courant", type=float, default=0.4)
    parser.add_argument("--viscosity", type=float, default=0.0)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--backend", choices=("amg", "cg"), default="amg")
    parser.add_argument("--scheme", choices=("rk3", "fast-rk3", "ab2"), default="rk3")
    parser.add_argument("--precision", choices=("float32", "float64"), default="float32")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    jax.config.update("jax_enable_x64", arguments.precision == "float64")

    grid = UniformGrid(*arguments.cells, *arguments.lengths)
    surface = MoninObukhovWall(
        roughness=arguments.roughness,
        von_karman=arguments.von_karman,
        sampling=arguments.sampling,
        averaging=arguments.averaging,
    )
    boundaries = monin_obukhov_boundaries()
    forcing = arguments.friction**2 / grid.lz
    model = FlowModel(
        viscosity=arguments.viscosity,
        body_force=(forcing, 0.0, 0.0),
        subfilter=AnisotropicMinimumDissipation(),
        surface=surface,
    )
    poisson = build_pressure_poisson(
        grid,
        backend=arguments.backend,
        dtype=arguments.precision,
    )
    run = build_run(build_step(grid, boundaries, poisson, model, scheme=arguments.scheme))

    solution = initial_solution(
        grid,
        initial_field(
            grid,
            surface,
            arguments.friction,
            arguments.noise,
            arguments.seed,
            arguments.precision,
        ),
        dtype=arguments.precision,
    )
    dt = float(
        stable_timestep(
            solution.velocity,
            grid,
            arguments.viscosity,
            courant=arguments.courant,
        )
    )
    turnover = grid.lz / arguments.friction
    print(
        f"mesh {grid.nx}x{grid.ny}x{grid.nz}  dz {grid.dz:.2f} m  "
        f"z1 {surface.reference_height(grid):.3f} m  z0 {arguments.roughness} m  "
        f"dz/z0 {grid.dz / arguments.roughness:.0f}"
    )
    print(
        f"u* {arguments.friction} m/s  forcing {forcing:.3e} m/s^2  "
        f"dt {dt:.3f} s  eddy turnover {turnover:.0f} s  "
        f"sampling {arguments.sampling}  averaging {arguments.averaging}"
    )

    height = (jnp.arange(grid.nz) + 0.5) * grid.dz
    reference = logarithmic_profile(grid, arguments.friction, surface)

    def report_profile(values, diagnosed, label):
        """Print the mean wind and its non-dimensional shear."""
        shear = (values[1:] - values[:-1]) / grid.dz
        faces = jnp.arange(1, grid.nz) * grid.dz
        phi = arguments.von_karman * faces / max(diagnosed, 1.0e-12) * shear
        print(f"\n{label}")
        print(f"{'z [m]':>9} {'z/H':>7} {'U [m/s]':>9} {'log law':>9} {'Phi':>7}")
        for level in range(grid.nz):
            shear_value = float(phi[level - 1]) if level >= 1 else float("nan")
            print(
                f"{float(height[level]):9.2f} {float(height[level]) / grid.lz:7.3f} "
                f"{float(values[level]):9.4f} {float(reference[level]):9.4f} "
                f"{shear_value:7.3f}"
            )
        lower = max(1, int(0.02 * grid.nz))
        upper = max(lower + 1, int(0.2 * grid.nz))
        window = phi[lower - 1 : upper]
        print(
            f"Phi over {float(faces[lower - 1]) / grid.lz:.3f} < z/H < "
            f"{float(faces[upper - 1]) / grid.lz:.3f}: "
            f"mean {float(jnp.mean(window)):.3f}  min {float(jnp.min(window)):.3f}  "
            f"max {float(jnp.max(window)):.3f}",
            flush=True,
        )

    samples, total = 0, np.zeros(grid.nz)
    spinup_steps = int(arguments.spinup * arguments.steps)
    done = 0
    while done < arguments.steps:
        block = min(arguments.chunk, arguments.steps - done)
        start = time.perf_counter()
        solution = run(solution, dt, block)
        jax.block_until_ready(solution.velocity.x)
        elapsed = time.perf_counter() - start
        done += block
        profile = jnp.mean(solution.velocity.x, axis=(1, 2))
        if done > spinup_steps:
            total += np.asarray(profile, dtype=np.float64)
            samples += 1
        diagnosed = float(friction_velocity(solution.velocity, grid, surface))
        print(
            f"step {done:7d}  t {float(solution.time):8.1f} s "
            f"({float(solution.time) / turnover:5.2f} T)  "
            f"u* {diagnosed:6.4f} ({diagnosed / arguments.friction:5.3f} of target)  "
            f"U1 {float(profile[0]):6.3f}  Utop {float(profile[-1]):6.3f}  "
            f"CFL {float(courant_number(solution.velocity, grid, dt)):5.3f}  "
            f"div {float(jnp.max(jnp.abs(divergence(solution.velocity, grid)))):8.2e}  "
            f"{1000 * elapsed / block:6.2f} ms/step",
            flush=True,
        )
        if arguments.profile_every and done % arguments.profile_every == 0:
            report_profile(profile, diagnosed, f"instantaneous profile at step {done}")

    mean = jnp.asarray(total / max(samples, 1))
    final = float(friction_velocity(solution.velocity, grid, surface))
    # Phi is normalised by the equilibrium friction velocity implied by the
    # forcing, u_* = sqrt(f H), which is the value the balance must reach; the
    # diagnosed one is printed alongside so the profile can be rescaled.
    report_profile(
        mean,
        arguments.friction,
        f"time-averaged over {samples} samples "
        f"(final diagnosed u* {final:.4f}, target {arguments.friction})",
    )


if __name__ == "__main__":
    main()

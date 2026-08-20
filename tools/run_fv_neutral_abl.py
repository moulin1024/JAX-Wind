"""Pressure-driven neutral boundary layer on the finite-volume solver.

A half-channel driven by a constant horizontal pressure gradient, periodic in
x and y, frictionless and impermeable at the top, and closed at the bottom by
the Monin-Obukhov surface stress of a neutral logarithmic layer.  At
equilibrium the driving force balances the surface drag, ``f = u_*^2 / H``, and
the mean wind should follow ``U(z) = (u_* / kappa) ln(z / z_0)``.

The diagnostic is the non-dimensional shear ``Phi = (kappa z / u_*) dU/dz``,
which is one wherever the logarithmic law holds.  The run starts from a
uniform speed, not the equilibrium profile, so that shear and turbulence have
to build up on their own.  A Rayleigh sponge relaxes the top of the domain
toward its own horizontal mean (and w toward zero), absorbing what reaches
the lid instead of letting it reflect off the frictionless top wall; disable
it with ``--no-sponge``.

    python tools/run_fv_neutral_abl.py --cells 64 64 64 --steps 20000

The AMG backend needs the environment described in tools/run_fv_channel.py.
Pass ``--backend gmg`` for a
matrix-free geometric multigrid V-cycle built straight from the mesh, or
``--backend fft`` for a direct, exact solve (valid only because the horizontal
boundary is periodic, which it always is here).
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
    PLANE_MEAN,
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
    rayleigh_sponge_tendency,
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
    """A uniform streamwise speed, perturbed to trip turbulence.

    The logarithmic profile is the equilibrium the run is meant to reach, not
    a starting point; starting from it would hide whether the wall stress and
    the forcing can actually build the shear on their own.  The uniform speed
    is the height average of that equilibrium profile, so the run carries the
    same bulk momentum without imposing its shape.
    """
    reference = logarithmic_profile(grid, friction, model).astype(dtype)
    speed = jnp.mean(reference)
    cells = (grid.nz, grid.ny, grid.nx)
    keys = jax.random.split(jax.random.PRNGKey(seed), 3)
    scale = noise * friction
    # Perturb only the interior so the walls stay impermeable.
    vertical = jnp.zeros((grid.nz + 1, grid.ny, grid.nx), dtype)
    vertical = vertical.at[1:-1].set(
        scale * jax.random.normal(keys[2], (grid.nz - 1, grid.ny, grid.nx), dtype)
    )
    return StaggeredVelocity(
        jnp.full(cells, speed, dtype) + scale * jax.random.normal(keys[0], cells, dtype),
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
    parser.add_argument(
        "--sponge-start-fraction",
        type=float,
        default=0.8,
        help="height, as a fraction of the domain, where the top sponge begins",
    )
    parser.add_argument(
        "--sponge-timescale",
        type=float,
        default=100.0,
        help="Rayleigh relaxation time at the lid, in seconds",
    )
    parser.add_argument("--sponge-power", type=float, default=2.0)
    parser.add_argument(
        "--no-sponge",
        action="store_true",
        help="disable the Rayleigh sponge at the top boundary",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="save the time-averaged profile and Phi to this PNG path",
    )
    parser.add_argument("--backend", choices=("amg", "gmg", "fft"), default="amg")
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
    sponge_start = arguments.sponge_start_fraction * grid.lz
    sponge = (
        None
        if arguments.no_sponge
        else rayleigh_sponge_tendency(
            grid,
            start_height=sponge_start,
            timescale=arguments.sponge_timescale,
            power=arguments.sponge_power,
            target=PLANE_MEAN,
        )
    )
    model = FlowModel(
        viscosity=arguments.viscosity,
        body_force=(forcing, 0.0, 0.0),
        forcing=sponge,
        subfilter=AnisotropicMinimumDissipation(),
        surface=surface,
    )
    print(
        f"assembling {arguments.backend} pressure operator "
        f"({grid.nz}x{grid.ny}x{grid.nx} cells)...",
        end="",
        flush=True,
    )
    setup_start = time.perf_counter()
    poisson = build_pressure_poisson(
        grid,
        backend=arguments.backend,
        dtype=arguments.precision,
    )
    print(f" done in {time.perf_counter() - setup_start:.2f} s", flush=True)
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
    print(
        "sponge off"
        if arguments.no_sponge
        else (
            f"sponge on: z > {sponge_start:.0f} m "
            f"({arguments.sponge_start_fraction:.2f} H)  "
            f"tau {arguments.sponge_timescale:.0f} s  power {arguments.sponge_power:.1f}"
        )
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

    def save_profile_plot(values, diagnosed, path):
        """Save the mean wind against the log law, and Phi against z/H."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        shear = (values[1:] - values[:-1]) / grid.dz
        faces = jnp.arange(1, grid.nz) * grid.dz
        phi = arguments.von_karman * faces / max(diagnosed, 1.0e-12) * shear

        figure, (wind, shear_axis) = plt.subplots(1, 2, figsize=(9.0, 4.5))
        wind.semilogy(np.asarray(values), np.asarray(height), "o-", ms=3, label="simulation")
        wind.semilogy(np.asarray(reference), np.asarray(height), "--", label="log law")
        wind.set_xlabel("U [m/s]")
        wind.set_ylabel("z [m]")
        wind.legend()

        shear_axis.semilogy(np.asarray(phi), np.asarray(faces) / grid.lz, "o-", ms=3)
        shear_axis.axvline(1.0, color="0.5", linestyle="--", linewidth=1)
        shear_axis.set_xlabel(r"$\Phi = \kappa z / u_* \, dU/dz$")
        shear_axis.set_ylabel("z / H")

        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)
        print(f"wrote {path}", flush=True)

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
    if arguments.plot:
        save_profile_plot(mean, arguments.friction, arguments.plot)


if __name__ == "__main__":
    main()

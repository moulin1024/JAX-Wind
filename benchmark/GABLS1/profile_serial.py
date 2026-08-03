#!/usr/bin/env python3
"""Stage-level single-process performance profile for GABLS1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src"
for source in (ROOT, SOURCE):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))


from benchmark.GABLS1 import run  # noqa: E402


def _profile_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile-repeats", type=int, default=10)
    parser.add_argument("--profile-output", type=Path)
    profile, runner_argv = parser.parse_known_args(argv)
    if profile.profile_repeats <= 0:
        parser.error("profile-repeats must be positive")
    runner = run.parse_args(runner_argv)
    return profile, runner


def main(argv: list[str] | None = None) -> None:
    profile, args = _profile_args(argv)
    from jax import config as jax_config

    if args.dtype == "float64":
        jax_config.update("jax_enable_x64", True)
    import jax
    import jax.numpy as jnp
    from jaxwind.momentum.neutral_abl import _cell_velocity

    coupled, case, dtype = run._build_coupled(args)
    state = (
        run._initial_state(args, coupled, case, dtype)
        if args.restart is None
        else run._restore_checkpoint(args, coupled, dtype)["state"]
    )

    def ready(value) -> None:
        jax.block_until_ready(value)

    def measure(function, *, repeats: int | None = None, advance=False):
        count = profile.profile_repeats if repeats is None else repeats
        value = state
        ready(function(value))
        start = time.perf_counter()
        for _ in range(count):
            argument = value if advance else state
            value = function(argument)
            ready(value)
        return (time.perf_counter() - start) / count, value

    midpoint_temperature = coupled._compiled_surface_scalar_step(
        state.potential_temperature,
        state.velocity,
        jnp.asarray(0.25, dtype=dtype),
        jnp.asarray(state.time, dtype=dtype),
    )
    ready(midpoint_temperature)
    cells = _cell_velocity(state.velocity)
    velocity_gradient = coupled.momentum.velocity_gradient(cells)
    surface_fluxes = coupled.surface_layer_fluxes(state)
    wall_stress = coupled._momentum_wall_stress(surface_fluxes)
    ready((cells, velocity_gradient, wall_stress))

    timings: dict[str, float] = {}
    timings["cfl_rates_s"], _ = measure(
        lambda current: coupled.timestep_for_cfl(
            current,
            args.target_cfl,
            args.target_diffusive_cfl,
        )
    )
    timings["pre_step_metrics_s"], _ = measure(
        coupled.pre_step_metrics
    )
    timings["accepted_state_metrics_s"], _ = measure(
        coupled.accepted_state_metrics
    )
    timings["surface_fluxes_s"], _ = measure(
        lambda current: coupled.surface_layer_fluxes(current)
    )
    timings["scalar_half_step_s"], _ = measure(
        lambda current: coupled._compiled_surface_scalar_step(
            current.potential_temperature,
            current.velocity,
            jnp.asarray(0.25, dtype=dtype),
            jnp.asarray(current.time, dtype=dtype),
        )
    )
    timings["momentum_rhs_s"], _ = measure(
        lambda current: coupled._compiled_surface_momentum_tendency(
            current.velocity,
            midpoint_temperature,
            jnp.asarray(current.time, dtype=dtype),
            coupled.buoyancy_tendency(midpoint_temperature),
        )
    )
    timings["coupled_stage_rhs_s"], _ = measure(
        lambda current: coupled._compiled_coupled_surface_tendency(
            current.velocity,
            current.potential_temperature,
            jnp.asarray(current.time, dtype=dtype),
        )
    )
    timings["velocity_gradient_s"], _ = measure(
        jax.jit(lambda _: coupled.momentum.velocity_gradient(cells))
    )
    timings["momentum_centered_advection_s"], _ = measure(
        jax.jit(
            lambda _: coupled.momentum.conservative_advection(
                state.velocity,
                cells,
            )
        )
    )
    timings["momentum_amd_s"], _ = measure(
        jax.jit(
            lambda _: coupled.momentum.sgs_tendency(
                cells,
                gradient=velocity_gradient,
                wall_stress=wall_stress,
            )
        )
    )
    timings["momentum_numerical_dissipation_s"], _ = measure(
        jax.jit(
            lambda _: coupled.momentum.advection_dissipation(
                state.velocity,
                cells,
            )
        )
    )
    timings["scalar_centered_advection_s"], _ = measure(
        jax.jit(
            lambda _: coupled.scalar.centered_advective_tendency(
                state.potential_temperature,
                state.velocity,
            )
        )
    )
    timings["scalar_amd_s"], _ = measure(
        jax.jit(
            lambda _: coupled.scalar.sgs_tendency(
                state.potential_temperature,
                velocity_gradient,
                lower_surface_flux=surface_fluxes.heat_flux,
            )
        )
    )
    timings["scalar_numerical_dissipation_s"], _ = measure(
        jax.jit(
            lambda _: coupled.scalar.advection_dissipation(
                state.potential_temperature,
                state.velocity,
            )
        )
    )

    def pressure_projection(current):
        return coupled.momentum.projector.project_velocity_and_pressure(
            current.velocity,
            timestep=0.5,
            initial_pressure=current.pressure,
        )

    timings["pressure_projection_s"], _ = measure(pressure_projection)
    advanced = state
    coupled.momentum.reset_fpj2()
    for _ in range(2):
        advanced = coupled.step(advanced, timestep=0.5)
        ready(advanced)
    start = time.perf_counter()
    for _ in range(profile.profile_repeats):
        advanced = coupled.step(advanced, timestep=0.5)
        ready(advanced)
    timings["full_step_s"] = (
        time.perf_counter() - start
    ) / profile.profile_repeats
    timings["diagnostic_fields_s"], _ = measure(
        jax.jit(coupled.diagnostic_fields)
    )

    pressure = coupled.momentum.pressure_solver
    operator = pressure.operator
    rhs = operator.prepare_rhs(
        -coupled.momentum.projector.project(
            state.velocity,
            timestep=0.5,
            initial_pressure=state.pressure,
        ).divergence_before
        / 0.5
    )[0]
    ready(rhs)
    timings["poisson_apply_s"], _ = measure(
        lambda _: pressure._apply_kernel(state.pressure)
    )
    timings["gmg_v_cycle_s"], _ = measure(
        lambda _: pressure._preconditioner_kernel(rhs)
    )

    original_projection = coupled.momentum.projector.project_velocity_and_pressure
    projection_stage_times: list[float] = []

    def timed_projection(*projection_args, **projection_kwargs):
        stage_start = time.perf_counter()
        result = original_projection(*projection_args, **projection_kwargs)
        ready(result)
        projection_stage_times.append(time.perf_counter() - stage_start)
        return result

    coupled.momentum.projector.project_velocity_and_pressure = timed_projection
    coupled.momentum.reset_fpj2()
    detailed_state = state
    for _ in range(2):
        detailed_state = coupled.step(detailed_state, timestep=0.5)
        ready(detailed_state)
    projection_stage_times.clear()
    detailed_state = coupled.step(detailed_state, timestep=0.5)
    ready(detailed_state)
    coupled.momentum.projector.project_velocity_and_pressure = original_projection

    payload = {
        "schema": "jaxwind.gabls1.serial-profile.v1",
        "backend": jax.default_backend(),
        "devices": len(jax.devices()),
        "shape_zyx": list(coupled.grid.shape),
        "step": state.step,
        "initial_state": "fresh" if args.restart is None else "checkpoint",
        "profile_repeats": profile.profile_repeats,
        "pressure_levels": [
            list(shape) for shape in pressure.preconditioner.level_shapes
        ],
        "pressure_execution": pressure.krylov.execution,
        "projection_method": args.projection_method,
        "advection_limiter": args.advection_limiter,
        "coupling_integrator": args.coupling_integrator,
        "scalar_rhs_calls_per_step": (
            3 if args.coupling_integrator == "coupled-ssprk3" else 6
        ),
        "steady_step_ppe_calls": len(projection_stage_times),
        "steady_step_ppe_stage_s": projection_stage_times,
        "advanced_step": advanced.step,
        **timings,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded, flush=True)
    if profile.profile_output is not None:
        profile.profile_output.parent.mkdir(parents=True, exist_ok=True)
        profile.profile_output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()

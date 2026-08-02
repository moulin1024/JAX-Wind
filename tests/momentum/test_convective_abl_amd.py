from __future__ import annotations

import jax.numpy as jnp

from jaxwind.momentum import (
    AMDBoussinesq,
    AMDBoussinesqConfig,
    AMDModel,
    AMDPassiveScalar,
    AMDPassiveScalarModel,
    NeutralABLConfig,
    NeutralABLMomentum,
)
from jaxwind.pressure import (
    BoundaryCondition,
    FGMRESConfig,
    MACVelocity,
    MatrixFreePoissonSolver,
    PoissonBoundaryConditions,
    RectilinearGrid,
    mac_divergence,
)


def _coupled_solver(
    *,
    lower_flux: float = 0.01,
    mp5_strength: float = 1.0,
    scalar_advection: str = "mp5",
    coupling_integrator: str = "strang",
) -> AMDBoussinesq:
    grid = RectilinearGrid.uniform(8, 8, 8, lx=2.0, ly=2.0, lz=1.0)
    periodic = BoundaryCondition("periodic")
    neumann = BoundaryCondition("neumann")
    pressure = MatrixFreePoissonSolver(
        grid,
        PoissonBoundaryConditions(
            periodic,
            periodic,
            periodic,
            periodic,
            neumann,
            neumann,
        ),
        dtype=jnp.float32,
        krylov=FGMRESConfig(
            restart=20,
            max_iterations=80,
            relative_tolerance=2.0e-6,
        ),
    )
    momentum = NeutralABLMomentum(
        grid,
        pressure,
        NeutralABLConfig(
            friction_velocity=0.1,
            roughness_length=1.0e-3,
            pressure_acceleration=0.0,
            mp5_dissipation_strength=mp5_strength,
            amd=AMDModel(coefficient=0.212),
            sgs_time_integration="explicit",
            projection_method="full",
        ),
    )
    scalar = AMDPassiveScalar(
        grid,
        AMDPassiveScalarModel(
            coefficient=0.212,
            lower_surface_flux=lower_flux,
            upper_surface_flux=0.0,
            mp5_dissipation_strength=mp5_strength,
            advection_scheme=scalar_advection,
        ),
    )
    return AMDBoussinesq(
        momentum,
        scalar,
        AMDBoussinesqConfig(
            gravity=9.81,
            reference_potential_temperature=300.0,
            coupling_integrator=coupling_integrator,
        ),
    )


def _stable_coupled_solver(
    projection_method: str = "full",
    coupling_integrator: str = "strang",
) -> AMDBoussinesq:
    grid = RectilinearGrid.uniform(8, 8, 8, lx=400.0, ly=400.0, lz=400.0)
    periodic = BoundaryCondition("periodic")
    neumann = BoundaryCondition("neumann")
    pressure = MatrixFreePoissonSolver(
        grid,
        PoissonBoundaryConditions(
            periodic,
            periodic,
            periodic,
            periodic,
            neumann,
            neumann,
        ),
        dtype=jnp.float32,
        krylov=FGMRESConfig(
            restart=20,
            max_iterations=80,
            relative_tolerance=2.0e-6,
        ),
    )
    momentum = NeutralABLMomentum(
        grid,
        pressure,
        NeutralABLConfig(
            friction_velocity=0.3,
            roughness_length=0.1,
            pressure_acceleration=0.0,
            geostrophic_wind=(8.0, 0.0),
            coriolis_vertical=1.39e-4,
            amd=AMDModel(coefficient=0.212),
            sgs_time_integration="explicit",
            projection_method=projection_method,
        ),
    )
    scalar = AMDPassiveScalar(
        grid,
        AMDPassiveScalarModel(
            coefficient=0.212,
            lower_surface_flux=0.0,
            upper_surface_flux=0.0,
            mp5_dissipation_strength=1.0,
        ),
    )
    return AMDBoussinesq(
        momentum,
        scalar,
        AMDBoussinesqConfig(
            gravity=9.81,
            reference_potential_temperature=263.5,
            surface_potential_temperature=265.0,
            surface_temperature_tendency=-0.25 / 3600.0,
            thermal_roughness_length=0.1,
            coupling_integrator=coupling_integrator,
        ),
    )


def _zero_velocity(solver: AMDBoussinesq) -> MACVelocity:
    nz, ny, nx = solver.grid.shape
    return MACVelocity(
        jnp.zeros((nz, ny, nx + 1), dtype=jnp.float32),
        jnp.zeros((nz, ny + 1, nx), dtype=jnp.float32),
        jnp.zeros((nz + 1, ny, nx), dtype=jnp.float32),
    )


def test_explicit_pressure_acceleration_override_removes_neutral_driver() -> None:
    coupled = _coupled_solver()
    cells = jnp.zeros(coupled.grid.shape + (3,), dtype=jnp.float32)

    forcing = coupled.momentum.forcing_tendency(cells)

    assert float(jnp.max(jnp.abs(forcing))) == 0.0


def test_horizontally_uniform_temperature_is_hydrostatic_only() -> None:
    coupled = _coupled_solver()
    z = jnp.arange(coupled.grid.shape[0], dtype=jnp.float32)[:, None, None]
    theta = jnp.broadcast_to(0.03 * z, coupled.grid.shape)

    buoyancy = coupled.buoyancy_tendency(theta)

    assert all(
        float(jnp.max(jnp.abs(component))) < 2.0e-9
        for component in buoyancy
    )


def test_active_scalar_step_is_conservative_and_projection_is_solenoidal() -> None:
    coupled = _coupled_solver(lower_flux=0.01)
    nz, ny, nx = coupled.grid.shape
    x = 2.0 * jnp.pi * jnp.arange(nx, dtype=jnp.float32) / nx
    theta = jnp.broadcast_to(
        0.1 * jnp.sin(x)[None, None, :],
        (nz, ny, nx),
    )
    state = coupled.initial_state(_zero_velocity(coupled), theta)
    timestep = 0.005

    advanced = coupled.step(state, timestep=timestep)

    expected_mean = (
        jnp.mean(theta)
        + timestep
        * coupled.scalar.model.lower_surface_flux
        / (coupled.grid.z_faces[-1] - coupled.grid.z_faces[0])
    )
    divergence = mac_divergence(advanced.velocity, coupled.grid)
    assert jnp.isclose(
        jnp.mean(advanced.potential_temperature),
        expected_mean,
        rtol=2.0e-5,
        atol=2.0e-7,
    )
    assert float(coupled.momentum.pressure_solver.operator.norm(divergence)) < 1.0e-3
    assert float(jnp.max(jnp.abs(advanced.velocity.z))) > 0.0


def test_coupled_ssprk3_shares_three_scalar_stages_and_conserves() -> None:
    coupled = _coupled_solver(
        lower_flux=0.01,
        coupling_integrator="coupled-ssprk3",
    )
    theta = jnp.zeros(coupled.grid.shape, dtype=jnp.float32)
    state = coupled.initial_state(_zero_velocity(coupled), theta)
    timestep = 0.005
    original = coupled._compiled_coupled_surface_tendency
    stage_times = []

    def counted(*args):
        stage_times.append(float(args[2]))
        return original(*args)

    coupled._compiled_coupled_surface_tendency = counted
    advanced = coupled.step(state, timestep=timestep)

    expected_mean = (
        timestep
        * coupled.scalar.model.lower_surface_flux
        / (coupled.grid.z_faces[-1] - coupled.grid.z_faces[0])
    )
    divergence = mac_divergence(advanced.velocity, coupled.grid)
    assert jnp.allclose(
        jnp.asarray(stage_times),
        jnp.asarray((0.0, timestep, 0.5 * timestep)),
    )
    assert jnp.isclose(
        jnp.mean(advanced.potential_temperature),
        expected_mean,
        rtol=2.0e-5,
        atol=2.0e-7,
    )
    assert jnp.isclose(
        coupled.last_surface_heat_flux_quadrature,
        coupled.scalar.model.lower_surface_flux,
    )
    assert float(coupled.momentum.pressure_solver.operator.norm(divergence)) < 1e-3


def test_coupled_scalar_rhs_rejects_stage_divergence_background_mode() -> None:
    coupled = _coupled_solver(
        lower_flux=0.0,
        mp5_strength=0.0,
        coupling_integrator="coupled-ssprk3",
    )
    nz, ny, nx = coupled.grid.shape
    theta = jnp.full((nz, ny, nx), 300.0, dtype=jnp.float32)
    x = 1.0e-3 * jnp.sin(
        2.0
        * jnp.pi
        * jnp.arange(nx + 1, dtype=jnp.float32)
        / nx
    )
    velocity = MACVelocity(
        jnp.broadcast_to(x, (nz, ny, nx + 1)),
        jnp.zeros((nz, ny + 1, nx), dtype=jnp.float32),
        jnp.zeros((nz + 1, ny, nx), dtype=jnp.float32),
    )

    _, scalar_rhs, _ = coupled._compiled_coupled_surface_tendency(
        velocity,
        theta,
        jnp.asarray(0.0, dtype=jnp.float32),
    )

    assert float(jnp.max(jnp.abs(scalar_rhs))) < 2.0e-6
    assert float(jnp.abs(jnp.mean(scalar_rhs))) < 2.0e-7


def test_cbl_sgs_energy_includes_prescribed_surface_buoyancy_flux() -> None:
    coupled = _coupled_solver(lower_flux=0.01, mp5_strength=0.0)
    theta = jnp.zeros(coupled.grid.shape, dtype=jnp.float32)
    state = coupled.initial_state(_zero_velocity(coupled), theta)

    fields = coupled.diagnostic_fields(state)

    assert jnp.all(jnp.isfinite(fields.sgs_tke))
    assert float(jnp.mean(fields.sgs_tke[0])) > 0.0
    assert float(jnp.max(jnp.abs(fields.sgs_tke[1:]))) == 0.0
    assert float(jnp.max(jnp.abs(fields.ko6_energy_dissipation))) == 0.0
    assert float(jnp.max(jnp.abs(fields.mp5_energy_dissipation))) == 0.0
    assert float(jnp.max(jnp.abs(fields.mp5_scalar_dissipation))) == 0.0


def test_scalar_advection_split_exposes_mp5_without_changing_tendency() -> None:
    coupled = _coupled_solver(scalar_advection="centered_mp5")
    velocity = coupled.momentum.initial_log_profile(perturbation_amplitude=0.05)
    x = jnp.arange(coupled.grid.shape[2], dtype=jnp.float32)[None, None, :]
    scalar = jnp.broadcast_to(jnp.sin(0.8 * x), coupled.grid.shape)

    total = coupled.scalar.advective_tendency(scalar, velocity)
    split = coupled.scalar.centered_advective_tendency(
        scalar,
        velocity,
    ) + coupled.scalar.mp5_dissipation(scalar, velocity)

    assert jnp.allclose(total, split)


def test_prescribed_cooling_surface_produces_stable_most_fluxes() -> None:
    coupled = _stable_coupled_solver()
    theta = jnp.full(coupled.grid.shape, 265.0, dtype=jnp.float32)
    velocity = MACVelocity(
        jnp.full((8, 8, 9), 8.0, dtype=jnp.float32),
        jnp.zeros((8, 9, 8), dtype=jnp.float32),
        jnp.zeros((9, 8, 8), dtype=jnp.float32),
    )
    initial = coupled.initial_state(velocity, theta)
    cooled = coupled.initial_state(velocity, theta, time=3600.0)

    neutral_fluxes = coupled.surface_layer_fluxes(initial)
    stable_fluxes = coupled.surface_layer_fluxes(cooled)

    assert jnp.allclose(neutral_fluxes.heat_flux, 0.0)
    assert jnp.all(stable_fluxes.heat_flux < 0.0)
    assert jnp.all(stable_fluxes.obukhov_length > 0.0)
    assert jnp.all(
        stable_fluxes.friction_velocity < neutral_fluxes.friction_velocity
    )


def test_stable_surface_step_cools_conservatively_and_remains_projected() -> None:
    coupled = _stable_coupled_solver()
    nz, ny, nx = coupled.grid.shape
    theta = jnp.full((nz, ny, nx), 265.0, dtype=jnp.float32)
    velocity = MACVelocity(
        jnp.full((nz, ny, nx + 1), 8.0, dtype=jnp.float32),
        jnp.zeros((nz, ny + 1, nx), dtype=jnp.float32),
        jnp.zeros((nz + 1, ny, nx), dtype=jnp.float32),
    )
    state = coupled.initial_state(velocity, theta, time=3600.0)

    advanced = coupled.step(state, timestep=0.5)
    fields = coupled.diagnostic_fields(advanced)
    divergence = mac_divergence(advanced.velocity, coupled.grid)

    assert float(jnp.mean(advanced.potential_temperature[0])) < 265.0
    assert float(jnp.mean(fields.surface_heat_flux)) < 0.0
    assert float(jnp.mean(fields.surface_friction_velocity)) > 0.0
    assert jnp.all(jnp.isfinite(fields.sgs_tke))
    assert float(coupled.momentum.pressure_solver.operator.norm(divergence)) < 1e-3


def test_coupled_ssprk3_uses_rk_surface_heat_flux_quadrature() -> None:
    coupled = _stable_coupled_solver("full", "coupled-ssprk3")
    nz, ny, nx = coupled.grid.shape
    theta = jnp.full((nz, ny, nx), 265.0, dtype=jnp.float32)
    velocity = MACVelocity(
        jnp.full((nz, ny, nx + 1), 8.0, dtype=jnp.float32),
        jnp.zeros((nz, ny + 1, nx), dtype=jnp.float32),
        jnp.zeros((nz + 1, ny, nx), dtype=jnp.float32),
    )
    state = coupled.initial_state(velocity, theta, time=3600.0)
    timestep = 0.5

    advanced = coupled.step(state, timestep=timestep)

    heat = coupled.last_surface_heat_flux_quadrature
    assert heat is not None
    expected_change = timestep * heat / (
        coupled.grid.z_faces[-1] - coupled.grid.z_faces[0]
    )
    assert jnp.isclose(
        jnp.mean(advanced.potential_temperature - theta),
        expected_change,
        atol=2.0e-6,
    )


def test_active_scalar_fpj2_uses_one_ppe_after_two_startup_steps() -> None:
    coupled = _stable_coupled_solver("fpj2")
    nz, ny, nx = coupled.grid.shape
    theta = jnp.full((nz, ny, nx), 265.0, dtype=jnp.float32)
    velocity = MACVelocity(
        jnp.full((nz, ny, nx + 1), 8.0, dtype=jnp.float32),
        jnp.zeros((nz, ny + 1, nx), dtype=jnp.float32),
        jnp.zeros((nz + 1, ny, nx), dtype=jnp.float32),
    )
    state = coupled.initial_state(velocity, theta, time=3600.0)

    first = coupled.step(state, timestep=0.25)
    second = coupled.step(first, timestep=0.25)
    advanced = coupled.step(second, timestep=0.25)

    history = coupled.momentum.fpj2_state
    divergence = mac_divergence(advanced.velocity, coupled.grid)
    assert history is not None
    assert history.history_count == 2
    assert advanced.step == 3
    assert jnp.all(jnp.isfinite(advanced.potential_temperature))
    assert float(coupled.momentum.pressure_solver.operator.norm(divergence)) < 1e-3


def test_coupled_ssprk3_fpj2_builds_pressure_history() -> None:
    coupled = _stable_coupled_solver("fpj2", "coupled-ssprk3")
    nz, ny, nx = coupled.grid.shape
    theta = jnp.full((nz, ny, nx), 265.0, dtype=jnp.float32)
    velocity = MACVelocity(
        jnp.full((nz, ny, nx + 1), 8.0, dtype=jnp.float32),
        jnp.zeros((nz, ny + 1, nx), dtype=jnp.float32),
        jnp.zeros((nz + 1, ny, nx), dtype=jnp.float32),
    )
    state = coupled.initial_state(velocity, theta, time=3600.0)
    original = coupled.momentum.projector.project_velocity_and_pressure
    pressure_solves = 0

    def counted_projection(*args, **kwargs):
        nonlocal pressure_solves
        pressure_solves += 1
        return original(*args, **kwargs)

    coupled.momentum.projector.project_velocity_and_pressure = counted_projection

    first = coupled.step(state, timestep=0.25)
    second = coupled.step(first, timestep=0.25)
    advanced = coupled.step(second, timestep=0.25)

    history = coupled.momentum.fpj2_state
    divergence = mac_divergence(advanced.velocity, coupled.grid)
    assert history is not None
    assert history.history_count == 2
    assert pressure_solves == 7
    assert jnp.all(jnp.isfinite(advanced.potential_temperature))
    assert coupled.last_surface_heat_flux_quadrature is not None
    assert float(coupled.momentum.pressure_solver.operator.norm(divergence)) < 1e-3

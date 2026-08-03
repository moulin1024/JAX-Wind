from __future__ import annotations

import math

import jax
import jax.numpy as jnp

import jaxwind.momentum.neutral_abl as neutral_abl
from jaxwind.momentum import (
    AMDModel,
    AMDPassiveScalar,
    AMDPassiveScalarModel,
    LASDModel,
    NeutralABLConfig,
    NeutralABLMomentum,
    staggered_kinetic_energy_work,
    staggered_momentum,
)
from jaxwind.pressure import (
    BoundaryCondition,
    FGMRESConfig,
    MACVelocity,
    MatrixFreePoissonSolver,
    PoissonBoundaryConditions,
    RectilinearGrid,
    kep4_mac_divergence,
)
from jaxwind.pressure.kep4_operators import (
    neumann_divergence_axis,
    periodic_divergence_axis,
)


def _solver(
    *,
    nx: int = 8,
    ny: int = 8,
    nz: int = 8,
    projection_method: str = "full",
    sgs_time_integration: str = "explicit",
    molecular_viscosity: float = 0.0,
    wall_matching_level: int = 0,
    wall_filter_width: float | None = None,
    wall_temporal_filter_timescale: float | None = None,
    advection_scheme: str = "kep4",
    regularization_scheme: str = "ko6",
    ko6_dissipation_strength: float = 1.0,
) -> NeutralABLMomentum:
    grid = RectilinearGrid.uniform(
        nx,
        ny,
        nz,
        lx=2.0,
        ly=1.0,
        lz=1.0,
    )
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
            max_iterations=60,
            relative_tolerance=2.0e-6,
            execution=("jax" if sgs_time_integration == "imex_ark3" else "python"),
        ),
    )
    return NeutralABLMomentum(
        grid,
        pressure,
        NeutralABLConfig(
            friction_velocity=0.1,
            roughness_length=1.0e-3,
            wall_matching_level=wall_matching_level,
            wall_filter_width=wall_filter_width,
            wall_temporal_filter_timescale=wall_temporal_filter_timescale,
            advection_scheme=advection_scheme,
            regularization_scheme=regularization_scheme,
            ko6_dissipation_strength=ko6_dissipation_strength,
            amd=AMDModel(
                coefficient=0.212,
                molecular_viscosity=molecular_viscosity,
            ),
            sgs_time_integration=sgs_time_integration,
            projection_method=projection_method,
        ),
    )


def test_uniform_spacing_accepts_float32_inexact_domain_ratio() -> None:
    faces = tuple(2.0 * math.pi * index / 64 for index in range(65))

    spacing = neutral_abl._uniform_spacing(faces, "x spacing")

    assert spacing == (2.0 * math.pi) / 64


def test_filter_free_amd_is_nonnegative_and_switches_off_for_uniform_flow() -> None:
    solver = _solver()
    uniform = jnp.ones(solver.grid.shape + (3,), dtype=jnp.float32)
    uniform = uniform.at[..., 1:].set(0.0)
    zero_viscosity = solver.amd_viscosity(uniform)
    x = jnp.arange(solver.grid.shape[2], dtype=jnp.float32)[None, None, :]
    y = jnp.arange(solver.grid.shape[1], dtype=jnp.float32)[None, :, None]
    z = jnp.arange(solver.grid.shape[0], dtype=jnp.float32)[:, None, None]
    turbulent = jnp.stack(
        jnp.broadcast_arrays(
            jnp.sin(0.7 * x + 0.3 * y + 0.2 * z),
            jnp.cos(0.2 * x - 0.5 * y + 0.4 * z),
            jnp.sin(0.4 * x + 0.6 * y - 0.3 * z),
        ),
        axis=-1,
    )
    viscosity = solver.amd_viscosity(turbulent)

    assert float(jnp.max(jnp.abs(zero_viscosity))) == 0.0
    assert float(jnp.min(viscosity)) >= 0.0
    assert float(jnp.max(viscosity)) > 0.0


def test_diagnostic_sgs_tke_is_finite_nonnegative_and_not_prognostic() -> None:
    solver = _solver()
    velocity = solver.initial_log_profile(perturbation_amplitude=0.05)
    cells = solver.cell_centered_velocity(velocity)

    sgs_tke = solver.diagnostic_sgs_tke(cells)
    dissipation = solver.resolved_tke_sgs_dissipation(cells)

    assert sgs_tke.shape == solver.grid.shape
    assert jnp.all(jnp.isfinite(sgs_tke))
    assert float(jnp.min(sgs_tke)) >= 0.0
    assert jnp.all(jnp.isfinite(dissipation))


def test_passive_scalar_surface_flux_has_exact_finite_volume_balance() -> None:
    solver = _solver()
    scalar_solver = AMDPassiveScalar(
        solver.grid,
        AMDPassiveScalarModel(
            coefficient=0.0,
            lower_surface_flux=1.0e-3,
            upper_surface_flux=2.0e-4,
            mp5_dissipation_strength=0.0,
        ),
    )
    scalar = jnp.zeros(solver.grid.shape, dtype=jnp.float32)
    velocity = MACVelocity(
        jnp.zeros((8, 8, 9), dtype=jnp.float32),
        jnp.zeros((8, 9, 8), dtype=jnp.float32),
        jnp.zeros((9, 8, 8), dtype=jnp.float32),
    )

    tendency = scalar_solver.tendency(scalar, velocity)
    volume_integral = (
        jnp.sum(tendency) * scalar_solver.dx * scalar_solver.dy * scalar_solver.dz
    )
    expected = (
        scalar_solver.model.lower_surface_flux - scalar_solver.model.upper_surface_flux
    ) * (2.0 * 1.0)

    assert jnp.isclose(volume_integral, expected, rtol=2.0e-6, atol=1.0e-8)


def test_passive_scalar_step_is_finite_and_increases_domain_mean() -> None:
    solver = _solver()
    scalar_solver = AMDPassiveScalar(
        solver.grid,
        AMDPassiveScalarModel(lower_surface_flux=1.0e-3),
    )
    scalar = jnp.zeros(solver.grid.shape, dtype=jnp.float32)
    velocity = solver.initial_log_profile(perturbation_amplitude=0.05)

    advanced = scalar_solver.step(scalar, velocity, timestep=0.01)

    assert jnp.all(jnp.isfinite(advanced))
    assert float(jnp.mean(advanced)) > 0.0


def test_full_mp5_scalar_flux_is_conservative_and_constant_consistent() -> None:
    solver = _solver(nx=8, ny=8, nz=8)
    scalar_solver = AMDPassiveScalar(
        solver.grid,
        AMDPassiveScalarModel(
            coefficient=0.0,
            lower_surface_flux=0.0,
            upper_surface_flux=0.0,
            advection_scheme="mp5",
        ),
    )
    velocity = solver.initial_log_profile(perturbation_amplitude=0.1)
    scalar = jnp.full(solver.grid.shape, 3.25, dtype=jnp.float32)

    tendency = scalar_solver.mp5_advective_tendency(scalar, velocity)
    divergence = solver.projector.divergence(velocity)

    assert jnp.allclose(tendency, -scalar * divergence, atol=2.0e-6)
    assert float(jnp.abs(jnp.sum(tendency))) < 2.0e-5


def test_horizontal_skew_advection_has_zero_resolved_energy_work() -> None:
    solver = _solver(nz=2)
    nz, ny, nx = solver.grid.shape
    x = 2.0 * jnp.pi * jnp.arange(nx, dtype=jnp.float32) / nx
    y = 2.0 * jnp.pi * jnp.arange(ny, dtype=jnp.float32) / ny
    cells = jnp.zeros((nz, ny, nx, 3), dtype=jnp.float32)
    cells = cells.at[..., 0].set(jnp.sin(x)[None, None, :] * jnp.cos(y)[None, :, None])
    cells = cells.at[..., 1].set(-jnp.cos(x)[None, None, :] * jnp.sin(y)[None, :, None])
    tendency = solver.skew_advection(cells)
    work = jnp.sum(cells * tendency)

    assert float(jnp.abs(work)) < 2.0e-5


def test_morinishi_s4_horizontal_transport_conserves_momentum_and_energy() -> None:
    solver = _solver(nx=8, ny=8, nz=8)
    x_faces = jnp.asarray(solver.grid.x_faces, dtype=jnp.float32)
    y_faces = jnp.asarray(solver.grid.y_faces, dtype=jnp.float32)
    streamfunction = (
        jnp.sin(2.0 * jnp.pi * y_faces / y_faces[-1])[:, None]
        * jnp.sin(2.0 * jnp.pi * x_faces / x_faces[-1])[None, :]
    )
    u_plane = periodic_divergence_axis(streamfunction, solver.dy, 0)
    v_plane = -periodic_divergence_axis(streamfunction, solver.dx, 1)
    velocity = MACVelocity(
        jnp.broadcast_to(u_plane[None], (8, 8, 9)),
        jnp.broadcast_to(v_plane[None], (8, 9, 8)),
        jnp.zeros((9, 8, 8), dtype=jnp.float32),
    )
    tendency = solver.kep4_advection(velocity)
    divergence = kep4_mac_divergence(
        velocity,
        solver.grid,
        solver.pressure_solver.operator.boundaries,
    )
    work = staggered_kinetic_energy_work(velocity, tendency)
    scale = sum(
        jnp.sum(jnp.abs(component * rate))
        for component, rate in zip(velocity, tendency, strict=True)
    )
    momentum = staggered_momentum(tendency)
    momentum_scale = jnp.stack(
        (
            jnp.sum(jnp.abs(tendency.x[..., :-1])),
            jnp.sum(jnp.abs(tendency.y[:, :-1, :])),
        )
    )

    assert float(jnp.max(jnp.abs(divergence))) < 1.0e-5
    assert float(jnp.abs(work)) < 2.0e-6 * float(scale)
    assert float(jnp.max(jnp.abs(momentum[:2]) / momentum_scale)) < 2.0e-6


def test_morinishi_s4_wall_transport_conserves_discrete_energy() -> None:
    solver = _solver(nx=8, ny=8, nz=8)
    x_faces = jnp.asarray(solver.grid.x_faces, dtype=jnp.float32)
    z_faces = jnp.asarray(solver.grid.z_faces, dtype=jnp.float32)
    streamfunction = (
        jnp.sin(jnp.pi * z_faces / z_faces[-1])[:, None]
        * jnp.sin(2.0 * jnp.pi * x_faces / x_faces[-1])[None, :]
    )
    u_plane = neumann_divergence_axis(streamfunction, solver.dz, 0)
    w_plane = -periodic_divergence_axis(streamfunction, solver.dx, 1)
    velocity = MACVelocity(
        jnp.broadcast_to(u_plane[:, None], (8, 8, 9)),
        jnp.zeros((8, 9, 8), dtype=jnp.float32),
        jnp.broadcast_to(w_plane[:, None], (9, 8, 8)),
    )
    tendency = solver.kep4_advection(velocity)
    divergence = kep4_mac_divergence(
        velocity,
        solver.grid,
        solver.pressure_solver.operator.boundaries,
    )
    work = staggered_kinetic_energy_work(velocity, tendency)
    scale = sum(
        jnp.sum(jnp.abs(component * rate))
        for component, rate in zip(velocity, tendency, strict=True)
    )

    assert float(jnp.max(jnp.abs(divergence))) < 1.0e-5
    assert float(jnp.abs(work)) < 2.0e-6 * float(scale)


def test_morinishi_s4_has_fourth_order_accuracy() -> None:
    errors = []
    for count in (8, 16):
        length = 2.0 * jnp.pi
        spacing = float(length / count)
        x_faces = jnp.arange(count, dtype=jnp.float32) * spacing
        x_cells = (jnp.arange(count, dtype=jnp.float32) + 0.5) * spacing
        y_faces = x_faces
        y_cells = x_cells
        u = jnp.sin(x_faces)[None, None, :] * jnp.cos(y_cells)[None, :, None]
        u = jnp.broadcast_to(u, (2, count, count))
        v = -jnp.cos(x_cells)[None, None, :] * jnp.sin(y_faces)[None, :, None]
        v = jnp.broadcast_to(v, (2, count, count))
        velocity = MACVelocity(
            jnp.concatenate((u, u[..., :1]), axis=-1),
            jnp.concatenate((v, v[:, :1, :]), axis=1),
            jnp.zeros((3, count, count), dtype=jnp.float32),
        )
        tendency = neutral_abl.morinishi_s4_advection(
            velocity,
            dx=spacing,
            dy=spacing,
            dz=0.5,
        )
        exact_u = -jnp.sin(x_faces) * jnp.cos(x_faces)
        exact_v = -jnp.sin(y_faces) * jnp.cos(y_faces)
        error = jnp.maximum(
            jnp.max(jnp.abs(tendency.x[0, :, :-1] - exact_u[None, :])),
            jnp.max(jnp.abs(tendency.y[0, :-1, :] - exact_v[:, None])),
        )
        errors.append(float(error))

    assert errors[0] / errors[1] > 12.0


def test_morinishi_s4_wall_closure_does_not_amplify_grid_noise() -> None:
    solver = _solver(nx=8, ny=8, nz=8)
    key_u, key_v, key_w = jax.random.split(jax.random.PRNGKey(151), 3)
    u = jax.random.normal(key_u, (8, 8, 8), dtype=jnp.float32)
    v = jax.random.normal(key_v, (8, 8, 8), dtype=jnp.float32)
    w = jax.random.normal(key_w, (9, 8, 8), dtype=jnp.float32)
    velocity = MACVelocity(
        jnp.concatenate((u, u[..., :1]), axis=-1),
        jnp.concatenate((v, v[:, :1, :]), axis=1),
        w.at[0].set(0.0).at[-1].set(0.0),
    )

    tendency = solver.kep4_advection(velocity)
    velocity_scale = max(float(jnp.max(jnp.abs(field))) for field in velocity)
    tendency_scale = max(float(jnp.max(jnp.abs(field))) for field in tendency)

    assert all(bool(jnp.all(jnp.isfinite(field))) for field in tendency)
    assert tendency_scale < 1.5 * velocity_scale**2 / min(
        solver.dx,
        solver.dy,
        solver.dz,
    )


def test_ko6_is_conservative_and_energy_dissipative() -> None:
    solver = _solver(nx=8, ny=8, nz=8)
    key_x, key_y, key_z = jax.random.split(jax.random.PRNGKey(6), 3)
    velocity = MACVelocity(
        jax.random.normal(key_x, (8, 8, 9)),
        jax.random.normal(key_y, (8, 9, 8)),
        jax.random.normal(key_z, (9, 8, 8)),
    )
    velocity = solver.enforce_boundaries(velocity)
    cells = solver.cell_centered_velocity(velocity)

    tendency = solver.ko6_dissipation(velocity, cells)
    work = jnp.sum(cells * tendency)
    mean = jnp.mean(tendency, axis=(0, 1, 2))

    assert float(work) <= 1.0e-6
    assert float(jnp.max(jnp.abs(mean))) < 2.0e-7


def test_ko6_annihilates_constant_fields() -> None:
    solver = _solver(nx=8, ny=8, nz=8)
    velocity = MACVelocity(
        jnp.ones((8, 8, 9)),
        jnp.zeros((8, 9, 8)),
        jnp.zeros((9, 8, 8)),
    )
    cells = solver.cell_centered_velocity(velocity)

    tendency = solver.ko6_dissipation(velocity, cells)

    assert float(jnp.max(jnp.abs(tendency))) == 0.0


def test_mac_conservative_advection_preserves_momentum_and_energy() -> None:
    solver = _solver(nx=8, ny=8, nz=8)
    velocity = solver.initial_log_profile(perturbation_amplitude=0.1)
    cells = solver.cell_centered_velocity(velocity)
    tendency = solver.conservative_advection(velocity, cells)
    mean_tendency = jnp.mean(tendency, axis=(0, 1, 2))
    energy_work = jnp.mean(cells * tendency)

    assert jnp.max(jnp.abs(mean_tendency)) < 1.0e-7
    assert jnp.abs(energy_work) < 1.0e-6


def test_shared_gradient_paths_match_standalone_sgs_and_advection() -> None:
    solver = _solver()
    velocity = solver.initial_log_profile(perturbation_amplitude=0.05)
    cells = solver.cell_centered_velocity(velocity)
    gradient = solver.velocity_gradient(cells)
    viscosity = solver.sgs_viscosity(cells)

    assert jnp.allclose(
        solver.skew_advection(cells, gradient=gradient),
        solver.skew_advection(cells),
    )
    assert jnp.allclose(
        solver.sgs_viscosity(cells, gradient=gradient),
        viscosity,
    )
    assert jnp.allclose(
        solver.variational_sgs_tendency(
            cells,
            viscosity,
            gradient=gradient,
        ),
        solver.variational_sgs_tendency(cells, viscosity),
    )
    assert jnp.allclose(
        solver.sgs_tendency(cells, gradient=gradient),
        solver.sgs_tendency(cells),
    )


def test_cell_tendency_constructs_velocity_gradient_once_per_sgs_model() -> None:
    amd = _solver()
    lasd = NeutralABLMomentum(
        amd.grid,
        amd.pressure_solver,
        NeutralABLConfig(
            friction_velocity=0.1,
            roughness_length=1.0e-3,
            lasd=LASDModel(),
        ),
    )
    for solver in (amd, lasd):
        velocity = solver.initial_log_profile(perturbation_amplitude=0.05)
        solver.reset_lasd(velocity)
        original = solver.velocity_gradient
        calls = 0

        def counted_gradient(cells):
            nonlocal calls
            calls += 1
            return original(cells)

        solver.velocity_gradient = counted_gradient
        solver.cell_tendency(
            velocity,
            None if solver.lasd_state is None else solver.lasd_state.coefficient,
        )

        assert calls == 1


def test_imex_initial_tendency_reuses_gradient_for_frozen_viscosity() -> None:
    solver = _solver(sgs_time_integration="imex_ark3")
    velocity = solver.initial_log_profile(perturbation_amplitude=0.05)
    coefficient = solver._active_lasd_coefficient(velocity)
    original = solver.velocity_gradient
    calls = 0

    def counted_gradient(cells):
        nonlocal calls
        calls += 1
        return original(cells)

    solver.velocity_gradient = counted_gradient
    explicit, implicit, frozen = solver._compiled_imex_initial_tendencies(
        velocity,
        coefficient,
        solver.active_wall_velocity(velocity),
    )
    jnp.asarray(explicit.x).block_until_ready()

    assert calls == 1
    assert jnp.all(jnp.isfinite(frozen))
    assert jnp.all(jnp.isfinite(implicit.x))


def test_segmented_lasd_update_matches_independent_scale_reference() -> None:
    base = _solver()
    solver = NeutralABLMomentum(
        base.grid,
        base.pressure_solver,
        NeutralABLConfig(
            friction_velocity=0.1,
            roughness_length=1.0e-3,
            lasd=LASDModel(update_interval=1),
        ),
    )
    velocity = solver.initial_log_profile(perturbation_amplitude=0.05)
    initial = solver.reset_lasd(velocity)
    assert initial is not None
    cells = solver.cell_centered_velocity(velocity)
    accumulated = solver.lasd_closure.accumulate(initial, cells)
    gradient = solver.velocity_gradient(cells)
    ratio = solver.config.lasd.test_filter_ratio
    lm, mm = solver.lasd_closure._contractions(cells, gradient, ratio)
    qn, nn = solver.lasd_closure._contractions(cells, gradient, ratio**2)
    expected = solver.lasd_closure.update_from_contractions(
        accumulated,
        lm,
        mm,
        qn,
        nn,
        interval_dt=0.1,
        first_update=True,
    )

    solver._advance_lasd(velocity, 0.1)
    actual = solver.lasd_state
    assert actual is not None
    for actual_field, expected_field in zip(actual, expected, strict=True):
        assert jnp.allclose(
            actual_field,
            expected_field,
            rtol=2.0e-5,
            atol=2.0e-6,
        )


def test_pairwise_minmod_matches_stacked_reduction() -> None:
    coordinate = jnp.arange(35, dtype=jnp.float32).reshape(5, 7)
    values = (
        jnp.sin(0.37 * coordinate),
        jnp.cos(0.23 * coordinate),
        jnp.sin(0.11 * coordinate - 0.4),
        jnp.cos(0.29 * coordinate + 0.7),
    )
    stacked = jnp.stack(values)
    magnitude = jnp.min(jnp.abs(stacked), axis=0)
    expected = jnp.where(
        jnp.all(stacked > 0.0, axis=0),
        magnitude,
        jnp.where(jnp.all(stacked < 0.0, axis=0), -magnitude, 0.0),
    )

    assert jnp.array_equal(neutral_abl._minmod(*values), expected)


def test_local_mp5_dissipation_preserves_constant_momentum() -> None:
    solver = _solver()
    velocity = solver.initial_log_profile(perturbation_amplitude=0.0)
    constant = type(velocity)(
        jnp.ones_like(velocity.x),
        0.25 * jnp.ones_like(velocity.y),
        jnp.zeros_like(velocity.z),
    )
    dissipation = solver.mp5_dissipation(constant)

    assert float(jnp.max(jnp.abs(dissipation))) < 1.0e-7


def test_local_mp5_dissipation_acts_only_near_a_jump() -> None:
    solver = _solver(nx=16, ny=4, nz=2)
    nz, ny, nx = solver.grid.shape
    cells = jnp.zeros((nz, ny, nx, 3), dtype=jnp.float32)
    cells = cells.at[..., 0].set(
        (jnp.arange(nx, dtype=jnp.float32) < nx // 2)[None, None, :]
    )
    x_faces = jnp.ones((nz, ny, nx + 1), dtype=jnp.float32)
    x_faces = x_faces.at[..., -1].set(x_faces[..., 0])
    velocity = MACVelocity(
        x_faces,
        jnp.zeros((nz, ny + 1, nx), dtype=jnp.float32),
        jnp.zeros((nz + 1, ny, nx), dtype=jnp.float32),
    )
    dissipation = solver.mp5_dissipation(velocity, cells)
    active_columns = jnp.any(jnp.abs(dissipation[..., 0]) > 1.0e-6, axis=(0, 1))

    assert int(jnp.sum(active_columns)) <= 8
    assert int(jnp.sum(active_columns)) >= 2
    assert jnp.all(jnp.isfinite(dissipation))


def test_log_wall_and_pressure_force_balance_initial_bulk_momentum() -> None:
    solver = _solver()
    velocity = solver.initial_log_profile(perturbation_amplitude=0.0)
    cells = jnp.stack(
        (
            0.5 * (velocity.x[..., 1:] + velocity.x[..., :-1]),
            0.5 * (velocity.y[:, 1:, :] + velocity.y[:, :-1, :]),
            0.5 * (velocity.z[1:] + velocity.z[:-1]),
        ),
        axis=-1,
    )
    forcing = solver.forcing_tendency(cells) + solver.sgs_tendency(cells)

    assert float(jnp.abs(jnp.mean(forcing[..., 0]))) < 2.0e-6
    assert float(jnp.max(jnp.abs(forcing[..., 1:]))) < 1.0e-7


def test_filtered_log_wall_uses_periodic_physical_top_hat_velocity() -> None:
    solver = _solver(wall_filter_width=2.0)
    cells = jnp.zeros(solver.grid.shape + (3,), dtype=jnp.float32)
    alternating = jnp.where(
        jnp.arange(solver.grid.shape[2]) % 2 == 0,
        1.0,
        3.0,
    )
    cells = cells.at[0, ..., 0].set(alternating[None, :])

    filtered = solver.wall_velocity(cells)
    ustar = solver.wall_ustar(cells)
    expected_factor = solver.config.von_karman / math.log(
        (0.5 * solver.dz) / solver.config.roughness_length
    )

    assert jnp.allclose(filtered[..., 0], 2.0)
    assert jnp.allclose(filtered[..., 1], 0.0)
    assert jnp.allclose(ustar, 2.0 * expected_factor)


def test_log_wall_uses_configured_matching_level_and_height() -> None:
    solver = _solver(wall_matching_level=2)
    cells = jnp.zeros(solver.grid.shape + (3,), dtype=jnp.float32)
    cells = cells.at[0, ..., 0].set(1.0)
    cells = cells.at[2, ..., 0].set(3.0)

    sampled = solver.wall_velocity(cells)
    ustar = solver.wall_ustar(cells)
    expected_factor = solver.config.von_karman / math.log(
        (2.5 * solver.dz) / solver.config.roughness_length
    )

    assert solver.wall_matching_height == 2.5 * solver.dz
    assert jnp.allclose(sampled[..., 0], 3.0)
    assert jnp.allclose(ustar, 3.0 * expected_factor)


def test_temporal_wall_filter_advances_once_per_accepted_state() -> None:
    solver = _solver(wall_temporal_filter_timescale=2.0)
    velocity = solver.initial_log_profile(perturbation_amplitude=0.0)
    initial = solver.reset_wall_model(velocity)
    assert initial is not None
    cells = solver.cell_centered_velocity(velocity)
    changed = cells.at[0, ..., 0].add(2.0)
    changed_velocity = solver.enforce_boundaries(neutral_abl._cells_to_faces(changed))
    instantaneous = solver.instantaneous_wall_velocity(changed)

    solver._advance_wall_model(changed_velocity, 0.5)
    state = solver.wall_model_state

    assert state is not None
    assert jnp.allclose(
        state.filtered_velocity,
        0.75 * initial.filtered_velocity + 0.25 * instantaneous,
    )


def test_vertical_sgs_face_flux_is_exact_telescope_with_wall_boundary() -> None:
    solver = _solver()
    velocity = solver.initial_log_profile(perturbation_amplitude=0.05)
    cells = solver.cell_centered_velocity(velocity)
    wall_velocity = solver.active_wall_velocity(velocity)
    tendency = solver.sgs_tendency(
        cells,
        wall_velocity=wall_velocity,
    )
    faces = solver.vertical_sgs_stress_flux(
        cells,
        wall_velocity=wall_velocity,
    )
    plane_tendency = jnp.mean(tendency, axis=(1, 2))
    plane_face_divergence = jnp.mean(
        (faces[1:] - faces[:-1]) / solver.dz,
        axis=(1, 2),
    )
    expected_wall = jnp.mean(
        solver.wall_stress(cells, wall_velocity=wall_velocity),
        axis=(0, 1),
    )

    assert jnp.allclose(plane_tendency, plane_face_divergence, atol=2.0e-6)
    assert jnp.allclose(jnp.mean(faces[0], axis=(0, 1)), expected_wall)
    assert jnp.max(jnp.abs(faces[-1])) < 2.0e-6


def test_wall_filter_width_must_be_positive_and_finite() -> None:
    for invalid in (0.0, -1.0, math.inf, math.nan):
        try:
            NeutralABLConfig(wall_filter_width=invalid)
        except ValueError as error:
            assert "wall filter width" in str(error)
        else:
            raise AssertionError(f"accepted invalid wall filter width {invalid}")


def test_wall_matching_and_temporal_filter_controls_are_validated() -> None:
    for invalid in (-1, 0.5, True):
        try:
            NeutralABLConfig(wall_matching_level=invalid)
        except ValueError as error:
            assert "wall matching level" in str(error)
        else:
            raise AssertionError(f"accepted invalid matching level {invalid}")
    for invalid in (0.0, -1.0, math.inf, math.nan):
        try:
            NeutralABLConfig(wall_temporal_filter_timescale=invalid)
        except ValueError as error:
            assert "temporal filter timescale" in str(error)
        else:
            raise AssertionError(f"accepted invalid timescale {invalid}")


def test_short_neutral_abl_run_remains_projected_and_finite() -> None:
    solver = _solver(nx=6, ny=4, nz=4)
    velocity = solver.initial_log_profile(perturbation_amplitude=0.05)
    velocity = solver.step(
        velocity,
        timestep=1.0e-3,
        time=0.0,
    )
    diagnostic = solver.diagnostic(
        velocity,
        timestep=1.0e-3,
        time=1.0e-3,
    )

    assert diagnostic.divergence_norm < 5.0e-4
    assert diagnostic.maximum_cfl < 0.1
    assert diagnostic.kinetic_energy > 0.0
    assert jnp.all(jnp.isfinite(velocity.x))
    assert jnp.all(jnp.isfinite(velocity.y))
    assert jnp.all(jnp.isfinite(velocity.z))


def test_fpj2_builds_pressure_history_and_projects_each_accepted_step() -> None:
    solver = _solver(
        nx=6,
        ny=4,
        nz=4,
        projection_method="fpj2",
    )
    velocity = solver.initial_log_profile(perturbation_amplitude=0.05)
    for step in range(3):
        velocity = solver.step(
            velocity,
            timestep=1.0e-3,
            time=step * 1.0e-3,
        )

    state = solver.fpj2_state
    diagnostic = solver.diagnostic(
        velocity,
        timestep=1.0e-3,
        time=3.0e-3,
    )
    assert state is not None
    assert state.history_count == 2
    assert state.current_timestep == 1.0e-3
    assert diagnostic.divergence_norm < 8.0e-4


def test_adaptive_timestep_hits_requested_cfl() -> None:
    solver = _solver()
    velocity = solver.initial_log_profile(perturbation_amplitude=0.0)
    timestep = solver.timestep_for_cfl(velocity, 0.9)
    diagnostic = solver.diagnostic(
        velocity,
        timestep=timestep,
        time=0.0,
    )

    assert abs(diagnostic.maximum_cfl - 0.9) < 1.0e-6


def test_cfl_rate_uses_cell_local_face_envelopes() -> None:
    solver = _solver(nx=4, ny=4, nz=4)
    nz, ny, nx = solver.grid.shape
    velocity = MACVelocity(
        jnp.zeros((nz, ny, nx + 1), dtype=jnp.float32),
        jnp.zeros((nz, ny + 1, nx), dtype=jnp.float32),
        jnp.zeros((nz + 1, ny, nx), dtype=jnp.float32),
    )
    velocity = MACVelocity(
        velocity.x.at[0, 0, 0].set(4.0),
        velocity.y.at[1, 1, 1].set(3.0),
        velocity.z.at[2, 2, 2].set(2.0),
    )
    expected = max(4.0 / solver.dx, 3.0 / solver.dy, 2.0 / solver.dz)
    old_global_sum = 4.0 / solver.dx + 3.0 / solver.dy + 2.0 / solver.dz

    actual = float(solver.cfl_rate(velocity))

    assert actual == expected
    assert actual < old_global_sum


def test_variational_principal_sgs_operator_is_dissipative() -> None:
    solver = _solver()
    nz, ny, nx = solver.grid.shape
    cells = jnp.sin(
        0.17
        * jnp.arange(nz * ny * nx * 3, dtype=jnp.float32).reshape(
            nz,
            ny,
            nx,
            3,
        )
    )
    viscosity = 0.3 + 0.1 * jnp.cos(
        0.11
        * jnp.arange(nz * ny * nx, dtype=jnp.float32).reshape(
            nz,
            ny,
            nx,
        )
    )
    tendency = solver.principal_sgs_tendency(cells, viscosity)

    assert float(jnp.sum(cells * tendency)) < 0.0


def test_implicit_sgs_diffusion_damps_beyond_explicit_cfl_limit() -> None:
    solver = _solver(
        nx=8,
        ny=8,
        nz=8,
        sgs_time_integration="imex_ark3",
        molecular_viscosity=5.0,
    )
    velocity = solver.initial_log_profile(perturbation_amplitude=0.1)
    cells_before = solver.cell_centered_velocity(velocity)
    viscosity = jnp.full(solver.grid.shape, 5.0, dtype=jnp.float32)
    timestep = 0.02
    diffusive_cfl = (
        timestep
        * 2.0
        * 5.0
        * (1.0 / solver.dx**2 + 1.0 / solver.dy**2 + 1.0 / solver.dz**2)
    )
    advanced = solver.implicit_diffusion_solve(
        velocity,
        viscosity,
        timestep,
    )
    cells_after = solver.cell_centered_velocity(advanced)

    assert diffusive_cfl > 10.0
    assert float(jnp.sum(cells_after * cells_after)) < float(
        jnp.sum(cells_before * cells_before)
    )
    assert jnp.all(jnp.isfinite(cells_after))


def test_imex_timestep_removes_vertical_sgs_diffusion_limit() -> None:
    explicit = _solver(nz=32, molecular_viscosity=10.0)
    imex = _solver(
        nz=32,
        sgs_time_integration="imex_ark3",
        molecular_viscosity=10.0,
    )
    velocity = explicit.initial_log_profile(perturbation_amplitude=0.0)
    explicit_dt = explicit.timestep_for_cfl(velocity, 0.8, 0.5)
    imex_dt = imex.timestep_for_cfl(velocity, 0.8, 0.5)
    diagnostic = imex.diagnostic(
        velocity,
        timestep=imex_dt,
        time=0.0,
    )

    assert imex_dt > 10.0 * explicit_dt
    assert diagnostic.maximum_diffusive_cfl > 0.5
    horizontal_diffusive_cfl = (
        imex_dt * 2.0 * 10.0 * (1.0 / imex.dx**2 + 1.0 / imex.dy**2)
    )
    assert abs(horizontal_diffusive_cfl - 0.5) < 2.0e-6


def test_imex_ark3_fpj2_remains_projected_at_large_diffusive_cfl() -> None:
    solver = _solver(
        nx=6,
        ny=4,
        nz=4,
        projection_method="fpj2",
        sgs_time_integration="imex_ark3",
        molecular_viscosity=2.0,
    )
    velocity = solver.initial_log_profile(perturbation_amplitude=0.05)
    timestep = min(solver.timestep_for_cfl(velocity, 0.5), 0.02)
    for step in range(3):
        velocity = solver.step(
            velocity,
            timestep=timestep,
            time=step * timestep,
        )
    diagnostic = solver.diagnostic(
        velocity,
        timestep=timestep,
        time=3.0 * timestep,
    )

    assert diagnostic.maximum_diffusive_cfl > 0.5
    assert diagnostic.divergence_norm < 2.0e-3
    assert jnp.all(jnp.isfinite(velocity.x))
    assert solver.fpj2_state is not None


def test_ars233_has_third_order_for_general_additive_split() -> None:
    explicit_rate = -0.7
    implicit_rate = -3.1
    final_time = 0.4

    def integrate(steps: int) -> float:
        timestep = final_time / steps
        value = 1.0
        for _ in range(steps):
            explicit = []
            implicit = []
            stages = []
            for stage_index in range(len(neutral_abl._ARK3_C)):
                rhs = value
                for previous in range(stage_index):
                    rhs += (
                        timestep
                        * neutral_abl._ARK3_EXPLICIT_A[stage_index][previous]
                        * explicit[previous]
                    )
                    rhs += (
                        timestep
                        * neutral_abl._ARK3_IMPLICIT_A[stage_index][previous]
                        * implicit[previous]
                    )
                diagonal = neutral_abl._ARK3_IMPLICIT_A[stage_index][stage_index]
                stage = rhs / (1.0 - timestep * diagonal * implicit_rate)
                stages.append(stage)
                explicit.append(explicit_rate * stage)
                implicit.append(implicit_rate * stage)
            value += timestep * sum(
                neutral_abl._ARK3_EXPLICIT_B[index] * explicit[index]
                + neutral_abl._ARK3_IMPLICIT_B[index] * implicit[index]
                for index in range(len(stages))
            )
        return value

    exact = math.exp((explicit_rate + implicit_rate) * final_time)
    errors = [abs(integrate(steps) - exact) for steps in (10, 20, 40)]
    orders = [math.log(errors[index] / errors[index + 1], 2.0) for index in range(2)]

    assert min(orders) > 2.8


def test_geostrophic_wind_is_horizontal_coriolis_equilibrium() -> None:
    base = _solver()
    solver = NeutralABLMomentum(
        base.grid,
        base.pressure_solver,
        NeutralABLConfig(
            friction_velocity=0.4,
            roughness_length=1.0e-3,
            geostrophic_wind=(10.0, 0.0),
            coriolis_vertical=1.0e-4,
            amd=AMDModel(coefficient=0.212),
        ),
    )
    cells = jnp.zeros(solver.grid.shape + (3,), dtype=jnp.float32)
    cells = cells.at[..., 0].set(10.0)
    forcing = solver.forcing_tendency(cells)

    assert float(jnp.max(jnp.abs(forcing[1:, ..., :2]))) == 0.0


def test_tabulated_initial_profile_preserves_plane_means() -> None:
    solver = _solver(nz=4)
    mean_u = jnp.asarray((1.0, 2.0, 3.0, 4.0), dtype=jnp.float32)
    mean_v = jnp.asarray((0.4, 0.3, 0.2, 0.1), dtype=jnp.float32)
    velocity = solver.initial_profile(mean_u, mean_v, seed=7)
    cells = solver.cell_centered_velocity(velocity)
    recovered = jnp.mean(cells, axis=(1, 2))

    assert jnp.allclose(recovered[:, 0], mean_u, atol=2.0e-5)
    assert jnp.allclose(recovered[:, 1], mean_v, atol=2.0e-5)

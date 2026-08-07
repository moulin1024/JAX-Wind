from __future__ import annotations

import math

import jax
import numpy as np
import jax.numpy as jnp

from jaxwind._linalg import pcr_tridiagonal_solve
from jaxwind.momentum import (
    LASDModel,
    MeanMomentumConstraintConfig,
    MomentumConfig,
    MomentumOperators,
    MoninObukhovWallLaw,
    ScalarConfig,
    ScalarOperators,
)
from jaxwind.momentum.operators import (
    _cells_to_faces,
    _interpolate_to_vertical_faces,
)
from jaxwind.pressure import (
    GMGConfig,
    MatrixFreePoissonSolver,
    PCGConfig,
    PoissonBoundaryConditions,
    RectilinearGrid,
)


def _momentum(grid: RectilinearGrid) -> MomentumOperators:
    pressure = MatrixFreePoissonSolver(
        grid,
        PoissonBoundaryConditions.homogeneous_neumann(),
        dtype=jnp.float32,
        gmg=GMGConfig(min_coarse_cells=1),
        krylov=PCGConfig(max_iterations=5),
    )
    return MomentumOperators(
        grid,
        pressure,
        MomentumConfig(
            roughness_length=0.01,
            pressure_acceleration=0.0,
            mp5_dissipation_strength=0.0,
        ),
    )


def test_neutral_wall_law_filters_the_actual_first_control_volume() -> None:
    grid = RectilinearGrid(
        (0.0, 1.0, 2.0, 3.0, 4.0),
        (0.0, 1.0, 2.0, 3.0, 4.0),
        (0.0, 1.0, 3.0, 6.0, 10.0),
    )
    momentum = _momentum(grid)
    height = grid.z_widths[0]
    roughness = momentum.config.roughness_length
    denominator = math.log(height / roughness) - 1.0 + roughness / height
    target_ustar = 0.35
    first_cell_speed = target_ustar * denominator / momentum.config.von_karman
    velocity = jnp.zeros((*grid.shape, 3), dtype=jnp.float32)
    velocity = velocity.at[0, ..., 0].set(first_cell_speed)
    velocity = velocity.at[1:, ..., 0].set(100.0)

    wall_velocity = momentum.instantaneous_wall_velocity(velocity)
    recovered_ustar = momentum.wall_ustar(velocity)

    assert momentum.wall_cell_height == height
    assert np.allclose(wall_velocity[..., 0], first_cell_speed)
    assert np.allclose(recovered_ustar, target_ustar, rtol=2.0e-6)


def test_filtered_most_recovers_a_stable_first_cell_mean_profile() -> None:
    height = 2.0
    roughness = 0.1
    reference_temperature = 263.5
    law = MoninObukhovWallLaw(
        momentum_roughness_length=roughness,
        thermal_roughness_length=roughness,
        reference_potential_temperature=reference_temperature,
    )
    target_ustar = 0.5
    inverse_obukhov = 0.02
    neutral = math.log(height / roughness) - 1.0 + roughness / height
    momentum_slope = (
        law.stable_momentum_beta * (height - roughness) ** 2 / (2.0 * height)
    )
    heat_slope = law.stable_heat_beta * (height - roughness) ** 2 / (2.0 * height)
    momentum_transfer = neutral + momentum_slope * inverse_obukhov
    heat_transfer = neutral + heat_slope * inverse_obukhov
    target_temperature_scale = (
        inverse_obukhov
        * target_ustar**2
        * reference_temperature
        / (law.von_karman * law.gravity)
    )
    speed = target_ustar * momentum_transfer / law.von_karman
    temperature_difference = (
        target_temperature_scale * heat_transfer / law.von_karman
    )

    fluxes = law.surface_fluxes(
        jnp.asarray([speed, 0.0], dtype=jnp.float32),
        jnp.asarray(265.0 + temperature_difference, dtype=jnp.float32),
        jnp.asarray(265.0, dtype=jnp.float32),
        height,
    )

    assert np.isclose(float(fluxes.friction_velocity), target_ustar, rtol=2.0e-6)
    assert np.isclose(
        float(fluxes.temperature_scale),
        target_temperature_scale,
        rtol=1.0e-5,
    )
    assert np.isclose(float(fluxes.obukhov_length), 1.0 / inverse_obukhov, rtol=1.0e-5)

    face_velocity = law.first_internal_face_velocity(
        jnp.asarray([speed, 0.0], dtype=jnp.float32),
        fluxes,
        height,
    )
    face_transfer = math.log(height / roughness) + law.stable_momentum_beta * (
        height - roughness
    ) * inverse_obukhov
    expected_face_speed = target_ustar * face_transfer / law.von_karman
    assert np.isclose(float(face_velocity[0]), expected_face_speed, rtol=2.0e-6)


def test_most_prescribed_heat_flux_closes_unstable_surface() -> None:
    law = MoninObukhovWallLaw(
        momentum_roughness_length=0.1,
        thermal_roughness_length=0.1,
        reference_potential_temperature=300.0,
    )
    fluxes = law.surface_fluxes_from_heat_flux(
        jnp.asarray([8.0, 0.0], dtype=jnp.float32),
        jnp.asarray(0.06, dtype=jnp.float32),
        10.0,
    )

    assert float(fluxes.friction_velocity) > 0.0
    assert np.isclose(float(fluxes.heat_flux), 0.06, rtol=1.0e-6)
    assert float(fluxes.temperature_scale) < 0.0
    assert float(fluxes.obukhov_length) < 0.0


def test_shared_most_multicell_layer_closes_stable_and_unstable_profiles() -> None:
    law = MoninObukhovWallLaw(
        momentum_roughness_length=0.05,
        thermal_roughness_length=0.02,
        reference_potential_temperature=300.0,
    )
    lower, upper = 18.0, 31.0
    horizontal = jnp.asarray([7.0, 2.0], dtype=jnp.float32)
    for matching_temperature in (302.0, 298.0):
        fluxes = law.surface_fluxes_from_layer(
            horizontal,
            jnp.asarray(matching_temperature, dtype=jnp.float32),
            jnp.asarray(300.0, dtype=jnp.float32),
            lower,
            upper,
        )
        inverse_obukhov = jnp.where(
            jnp.isfinite(fluxes.obukhov_length),
            1.0 / fluxes.obukhov_length,
            0.0,
        )
        momentum_transfer = law._layer_average_transfer_denominator(
            inverse_obukhov,
            lower,
            upper,
            law.momentum_roughness_length,
            law.momentum_stability_correction,
        )
        heat_transfer = law._layer_average_transfer_denominator(
            inverse_obukhov,
            lower,
            upper,
            law.thermal_roughness_length,
            law.heat_stability_correction,
        )
        assert np.isclose(
            float(jnp.linalg.norm(horizontal)),
            float(fluxes.friction_velocity * momentum_transfer / law.von_karman),
            rtol=2.0e-5,
        )
        assert np.isclose(
            matching_temperature - 300.0,
            float(fluxes.temperature_scale * heat_transfer / law.von_karman),
            rtol=2.0e-5,
            atol=2.0e-5,
        )
        face_velocity, face_temperature = law.internal_face_profiles(
            horizontal,
            jnp.asarray(matching_temperature, dtype=jnp.float32),
            fluxes,
            lower,
            upper,
            jnp.asarray([4.0, 11.0, 20.0, 31.0], dtype=jnp.float32),
            surface_temperature=jnp.asarray(300.0, dtype=jnp.float32),
        )
        assert face_velocity.shape == (4, 2)
        assert face_temperature.shape == (4,)
        assert np.all(np.isfinite(np.asarray(face_velocity)))
        assert np.all(np.isfinite(np.asarray(face_temperature)))
        momentum_diffusivity, heat_diffusivity = (
            law.wall_layer_eddy_diffusivities(
                fluxes,
                jnp.asarray([4.0, 11.0, 20.0], dtype=jnp.float32),
            )
        )
        assert np.all(np.asarray(momentum_diffusivity) > 0.0)
        assert np.all(np.asarray(heat_diffusivity) > 0.0)


def test_wall_layer_uses_automatic_matching_cell_and_shared_diffusivity() -> None:
    grid = RectilinearGrid(
        tuple(np.linspace(0.0, 800.0, 9)),
        tuple(np.linspace(0.0, 800.0, 9)),
        (0.0, 5.0, 12.0, 22.0, 36.0, 56.0, 84.0, 123.0, 180.0),
    )
    pressure = MatrixFreePoissonSolver(
        grid,
        PoissonBoundaryConditions.homogeneous_neumann(),
        dtype=jnp.float32,
        gmg=GMGConfig(min_coarse_cells=1),
        krylov=PCGConfig(max_iterations=5),
    )
    momentum = MomentumOperators(
        grid,
        pressure,
        MomentumConfig(
            roughness_length=0.01,
            pressure_acceleration=0.0,
            mp5_dissipation_strength=0.0,
            wall_layer_matching_filter_ratio=1.0,
        ),
    )
    assert momentum.wall_layer_top_face is not None
    matching_cell = momentum.wall_input_cell_index
    cells = jnp.zeros((*grid.shape, 3), dtype=jnp.float32)
    cells = cells.at[0, ..., 0].set(100.0)
    cells = cells.at[matching_cell, ..., 0].set(8.0)
    wall_velocity = momentum.instantaneous_wall_velocity(cells)
    assert np.allclose(wall_velocity[..., 0], 8.0)

    wall_stress = momentum.wall_stress(cells)
    native = jnp.zeros((grid.shape[0] + 1, *grid.shape[1:], 3), dtype=jnp.float32)
    native = native.at[1:, ..., 0].set(0.25)
    coupled = momentum._couple_wall_layer_momentum_stress(native, wall_stress)
    assert np.allclose(coupled[0], wall_stress)
    assert np.allclose(coupled[1:], native[1:])

    wall_viscosity = momentum.neutral_wall_layer_viscosity(cells)
    assert wall_viscosity is not None
    fluxes = momentum.wall_fluxes(cells)
    expected_viscosity = (
        momentum.config.von_karman
        * np.asarray(momentum.z_centers[: wall_viscosity.shape[0]])[:, None, None]
        * np.asarray(fluxes.friction_velocity)[None, ...]
    )
    np.testing.assert_allclose(
        wall_viscosity,
        expected_viscosity,
        rtol=2.0e-6,
    )

    rhs = jnp.asarray(
        np.random.default_rng(4).normal(size=(*grid.shape, 3)),
        dtype=jnp.float32,
    )
    viscosity = jnp.full(grid.shape, 0.2, dtype=jnp.float32)
    implicit_dt = jnp.asarray(0.3, dtype=jnp.float32)
    solution = momentum.solve_vertical_sgs_diffusion(
        rhs,
        viscosity,
        implicit_dt,
    )
    residual = solution - implicit_dt * momentum.principal_sgs_tendency(
        solution,
        viscosity,
    )
    np.testing.assert_allclose(residual, rhs, rtol=2.0e-5, atol=2.0e-5)


def test_mean_momentum_constraint_closes_the_discrete_face_stress() -> None:
    grid = RectilinearGrid.uniform(8, 8, 8, lx=800.0, ly=800.0, lz=400.0)
    pressure = MatrixFreePoissonSolver(
        grid,
        PoissonBoundaryConditions.homogeneous_neumann(),
        dtype=jnp.float32,
        gmg=GMGConfig(min_coarse_cells=1),
        krylov=PCGConfig(max_iterations=5),
    )
    momentum = MomentumOperators(
        grid,
        pressure,
        MomentumConfig(
            friction_velocity=0.4,
            roughness_length=0.1,
            pressure_acceleration=0.4**2 / 400.0,
            mp5_dissipation_strength=1.0,
            mean_momentum_constraint=MeanMomentumConstraintConfig(
                timescale=100.0
            ),
        ),
    )
    velocity = momentum.initial_log_profile(perturbation_amplitude=0.2, project=False)
    mean_state = momentum.reset_mean_momentum(velocity)
    assert mean_state is not None
    momentum.restore_mean_momentum(
        mean_state._replace(previous_timestep=100.0)
    )
    coefficient = momentum._active_lasd_coefficient(velocity)
    wall_velocity = momentum.active_wall_velocity(velocity)
    correction = momentum.mean_stress_correction(
        velocity,
        coefficient,
        wall_velocity=wall_velocity,
    )

    cells = momentum.cell_centered_velocity(velocity)
    first_face = momentum.wall_law.first_internal_face_velocity(
        wall_velocity,
        momentum.wall_cell_height,
    )
    current = (
        -momentum.horizontal_mean(
            momentum.vertical_advective_flux(
                velocity,
                cells,
                first_internal_face_horizontal_velocity=first_face,
            )[..., :2]
        )
        - momentum.horizontal_mean(
            momentum.vertical_advection_dissipation_flux(velocity, cells)[..., :2]
        )
        + momentum.horizontal_mean(
            momentum.vertical_sgs_stress_flux(
                cells,
                coefficient,
                wall_velocity=wall_velocity,
            )[..., :2]
        )
    )
    wall_mean = momentum.surface_mean(
        momentum.wall_stress(cells, wall_velocity=wall_velocity)[..., :2]
    )
    z_faces = np.asarray(grid.z_faces, dtype=float)
    face_fraction = (z_faces - z_faces[0]) / (z_faces[-1] - z_faces[0])
    target = np.asarray(wall_mean)[None, :] * (1.0 - face_fraction[:, None])
    matching_height = momentum.mean_momentum_matching_height
    assert matching_height is not None
    blend = np.clip(1.0 - (z_faces - z_faces[0]) / matching_height, 0.0, 1.0)
    expected = np.asarray(current) + blend[:, None] * (
        target - np.asarray(current)
    )
    np.testing.assert_allclose(
        np.asarray(current + correction),
        expected,
        rtol=2.0e-5,
        atol=2.0e-6,
    )
    np.testing.assert_array_equal(np.asarray(correction[0]), np.zeros(2))
    np.testing.assert_array_equal(np.asarray(correction[-1]), np.zeros(2))


def test_fv_dynamic_scalar_uses_multigrid_filter_and_is_bounded() -> None:
    grid = RectilinearGrid.uniform(8, 8, 8, lx=8.0, ly=8.0, lz=8.0)
    pressure = MatrixFreePoissonSolver(
        grid,
        PoissonBoundaryConditions.homogeneous_neumann(),
        dtype=jnp.float32,
        gmg=GMGConfig(min_coarse_cells=1),
        krylov=PCGConfig(max_iterations=5),
    )
    momentum = MomentumOperators(
        grid,
        pressure,
        MomentumConfig(
            roughness_length=0.01,
            pressure_acceleration=0.0,
            lasd=LASDModel(),
        ),
    )
    scalar_operator = ScalarOperators(
        grid,
        ScalarConfig(
            model="fv_dynamic",
            minimum_dynamic_coefficient=0.0,
            maximum_dynamic_coefficient=0.5,
        ),
        multilevel_filter=momentum.lasd_closure,
    )
    z, y, x = jnp.meshgrid(
        jnp.arange(8, dtype=jnp.float32),
        jnp.arange(8, dtype=jnp.float32),
        jnp.arange(8, dtype=jnp.float32),
        indexing="ij",
    )
    cells = jnp.stack(
        (
            jnp.sin(0.3 * x) + 0.1 * z,
            jnp.cos(0.4 * y) - 0.05 * z,
            0.1 * jnp.sin(0.2 * x + 0.3 * y),
        ),
        axis=-1,
    )
    scalar = jnp.sin(0.2 * x + 0.1 * z) + 0.2 * jnp.cos(0.3 * y)
    gradient = momentum.velocity_gradient(cells)
    diffusivity = scalar_operator.sgs_diffusivity(
        scalar,
        gradient,
        cell_velocity=cells,
    )

    assert diffusivity.shape == grid.shape
    assert np.all(np.isfinite(np.asarray(diffusivity)))
    assert np.all(np.asarray(diffusivity) >= 0.0)


def test_wall_stress_uses_metric_aware_tangential_face_interpolation() -> None:
    grid = RectilinearGrid(
        (0.0, 1.0, 3.0, 6.0, 10.0),
        (0.0, 2.0, 3.0, 7.0, 8.0),
        (0.0, 0.5, 1.5, 4.0, 8.0),
    )
    momentum = _momentum(grid)
    x = np.asarray(grid.x_centers)[None, :]
    y = np.asarray(grid.y_centers)[:, None]
    stress = np.stack(
        (
            x + 2.0 * y,
            3.0 * x - y,
            np.zeros_like(x + y),
        ),
        axis=-1,
    ).astype(np.float32)

    tendency = momentum.wall_stress_face_tendency(jnp.asarray(stress))
    scale = -1.0 / grid.z_widths[0]

    expected_x = np.asarray(grid.x_faces)[None, 1:-1] + 2.0 * y
    expected_y = 3.0 * x - np.asarray(grid.y_faces)[1:-1, None]
    assert np.allclose(np.asarray(tendency.x[0, :, 1:-1]), scale * expected_x)
    assert np.allclose(np.asarray(tendency.y[0, 1:-1, :]), scale * expected_y)
    assert np.array_equal(np.asarray(tendency.x[:, :, 0]), np.asarray(tendency.x[:, :, -1]))
    assert np.array_equal(np.asarray(tendency.y[:, 0, :]), np.asarray(tendency.y[:, -1, :]))
    assert np.count_nonzero(np.asarray(tendency.x[1:])) == 0
    assert np.count_nonzero(np.asarray(tendency.y[1:])) == 0
    assert np.count_nonzero(np.asarray(tendency.z)) == 0


def test_uniform_wall_face_mapping_matches_original_arithmetic_path() -> None:
    grid = RectilinearGrid.uniform(4, 4, 4, lx=4.0, ly=4.0, lz=4.0)
    momentum = _momentum(grid)
    y, x = np.meshgrid(
        np.arange(4, dtype=np.float32),
        np.arange(4, dtype=np.float32),
        indexing="ij",
    )
    stress = np.stack((1.0 + x + y, 2.0 - x + y, np.zeros_like(x)), axis=-1)
    wall_cells = np.zeros((*grid.shape, 3), dtype=np.float32)
    wall_cells[0] = -stress

    original = _cells_to_faces(jnp.asarray(wall_cells))
    metric = momentum.wall_stress_face_tendency(jnp.asarray(stress))

    for original_component, metric_component in zip(original, metric, strict=True):
        assert np.array_equal(
            np.asarray(original_component),
            np.asarray(metric_component),
        )


def test_batched_imex_vertical_solve_satisfies_every_component_system() -> None:
    grid = RectilinearGrid(
        (0.0, 1.0, 2.0, 3.0, 4.0),
        (0.0, 1.0, 2.0, 3.0, 4.0),
        (0.0, 0.1, 0.3, 0.7, 1.5, 3.0, 5.0, 8.0, 12.0, 18.0),
    )
    momentum = _momentum(grid)
    nz, ny, nx = grid.shape
    z, y, x = jnp.meshgrid(
        jnp.arange(nz, dtype=jnp.float32),
        jnp.arange(ny, dtype=jnp.float32),
        jnp.arange(nx, dtype=jnp.float32),
        indexing="ij",
    )
    frozen_viscosity = 0.03 + 0.002 * z + 0.001 * x + 0.0005 * y
    rhs = jnp.stack(
        (
            jnp.sin(0.2 * x + 0.1 * z),
            jnp.cos(0.3 * y - 0.2 * z),
            0.1 * x - 0.2 * y + 0.3 * z,
        ),
        axis=-1,
    )
    timestep = jnp.asarray(0.4, dtype=jnp.float32)

    solution = momentum.solve_vertical_sgs_diffusion(
        rhs,
        frozen_viscosity,
        timestep,
    )

    face_viscosity = _interpolate_to_vertical_faces(
        frozen_viscosity,
        momentum.z_centers,
        momentum.z_faces,
    )
    lower = jnp.zeros_like(frozen_viscosity).at[1:].set(
        -timestep
        * face_viscosity
        / (momentum.dz_cell[1:] * momentum.dz_center)[:, None, None]
    )
    upper = jnp.zeros_like(frozen_viscosity).at[:-1].set(
        -timestep
        * face_viscosity
        / (momentum.dz_cell[:-1] * momentum.dz_center)[:, None, None]
    )
    diagonal = 1.0 - lower - upper
    reconstructed = diagonal[..., None] * solution
    reconstructed = reconstructed.at[1:].add(
        lower[1:, ..., None] * solution[:-1]
    )
    reconstructed = reconstructed.at[:-1].add(
        upper[:-1, ..., None] * solution[1:]
    )

    assert jnp.allclose(reconstructed, rhs, rtol=2.0e-5, atol=2.0e-5)


def test_pcr_matches_reference_solve_for_non_power_of_two_systems() -> None:
    batch, size, components = 5, 9, 3
    lower = jnp.zeros((batch, size), dtype=jnp.float32)
    upper = jnp.zeros((batch, size), dtype=jnp.float32)
    phase = jnp.arange(batch * size, dtype=jnp.float32).reshape(batch, size)
    lower = lower.at[..., 1:].set(-0.1 - 0.01 * jnp.sin(phase[..., 1:]))
    upper = upper.at[..., :-1].set(-0.08 - 0.01 * jnp.cos(phase[..., :-1]))
    diagonal = 1.0 - lower - upper
    rhs = jnp.sin(
        0.13
        * jnp.arange(batch * size * components, dtype=jnp.float32).reshape(
            batch,
            size,
            components,
        )
    )

    pcr = pcr_tridiagonal_solve(lower, diagonal, upper, rhs)
    reference = jax.lax.linalg.tridiagonal_solve(lower, diagonal, upper, rhs)

    assert jnp.allclose(pcr, reference, rtol=2.0e-5, atol=2.0e-5)

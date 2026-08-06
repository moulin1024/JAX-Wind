from __future__ import annotations

import math

import jax
import numpy as np
import jax.numpy as jnp

from jaxwind.momentum import (
    MomentumConfig,
    MomentumOperators,
    MoninObukhovWallLaw,
)
from jaxwind.momentum.operators import (
    _cells_to_faces,
    _interpolate_to_vertical_faces,
    _pcr_tridiagonal_solve,
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

    pcr = _pcr_tridiagonal_solve(lower, diagonal, upper, rhs)
    reference = jax.lax.linalg.tridiagonal_solve(lower, diagonal, upper, rhs)

    assert jnp.allclose(pcr, reference, rtol=2.0e-5, atol=2.0e-5)

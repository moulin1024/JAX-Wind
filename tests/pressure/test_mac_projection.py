from __future__ import annotations

import jax.numpy as jnp

from jaxwind.pressure import (
    BoundaryCondition,
    FGMRESConfig,
    MACStageProjector,
    MACVelocity,
    MatrixFreePoissonSolver,
    PCGConfig,
    PoissonBoundaryConditions,
    RectilinearGrid,
    fpj2_pressure_prediction,
    fpj2_ssprk3_velocity_step,
    kep4_mac_divergence,
    kep4_mac_pressure_gradient,
    mac_divergence,
    mac_pressure_gradient,
    projected_ssprk3_step,
    projected_ssprk3_velocity_pressure_step,
)
from jaxwind.pressure.kep4_operators import (
    neumann_divergence_axis,
    neumann_gradient_axis,
    periodic_gradient_axis,
)


def _wall_solver(grid: RectilinearGrid) -> MatrixFreePoissonSolver:
    return MatrixFreePoissonSolver(
        grid,
        PoissonBoundaryConditions.homogeneous_neumann(),
        dtype=jnp.float32,
        krylov=FGMRESConfig(
            restart=20,
            max_iterations=80,
            relative_tolerance=2.0e-6,
        ),
    )


def _kep4_solver(grid: RectilinearGrid) -> MatrixFreePoissonSolver:
    periodic = BoundaryCondition("periodic")
    neumann = BoundaryCondition("neumann")
    return MatrixFreePoissonSolver(
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
        krylov=PCGConfig(
            max_iterations=100,
            relative_tolerance=1.0e-6,
        ),
        discretization="kep4",
    )


def test_kep4_periodic_gradient_has_fourth_order_accuracy() -> None:
    errors = []
    for count in (8, 16):
        spacing = 1.0 / count
        centers = (jnp.arange(count, dtype=jnp.float32) + 0.5) * spacing
        pressure = jnp.sin(2.0 * jnp.pi * centers)
        gradient = periodic_gradient_axis(pressure, spacing, -1)[:-1]
        exact = (
            2.0
            * jnp.pi
            * jnp.cos(2.0 * jnp.pi * jnp.arange(count, dtype=jnp.float32) * spacing)
        )
        errors.append(float(jnp.max(jnp.abs(gradient - exact))))

    assert errors[0] / errors[1] > 12.0


def test_kep4_neumann_wall_closures_have_fourth_order_accuracy() -> None:
    gradient_errors = []
    divergence_errors = []
    for count in (8, 16):
        spacing = 1.0 / count
        centers = (jnp.arange(count, dtype=jnp.float32) + 0.5) * spacing
        faces = jnp.arange(count + 1, dtype=jnp.float32) * spacing

        pressure = jnp.cos(jnp.pi * centers)
        gradient = neumann_gradient_axis(pressure, spacing, -1)
        exact_gradient = -jnp.pi * jnp.sin(jnp.pi * faces)
        gradient_errors.append(float(jnp.max(jnp.abs(gradient - exact_gradient))))

        velocity = jnp.sin(jnp.pi * faces)
        divergence = neumann_divergence_axis(velocity, spacing, -1)
        exact_divergence = jnp.pi * jnp.cos(jnp.pi * centers)
        divergence_errors.append(float(jnp.max(jnp.abs(divergence - exact_divergence))))

    assert gradient_errors[0] / gradient_errors[1] > 12.0
    assert divergence_errors[0] / divergence_errors[1] > 12.0


def test_kep4_gradient_and_divergence_are_negative_transposes() -> None:
    grid = RectilinearGrid.uniform(8, 8, 8)
    solver = _kep4_solver(grid)
    pressure = jnp.sin(0.17 * jnp.arange(grid.cell_count, dtype=jnp.float32)).reshape(
        grid.shape
    )
    velocity = _closed_velocity(grid, phase=0.31)
    gradient = kep4_mac_pressure_gradient(
        pressure,
        grid,
        solver.operator.boundaries,
    )
    divergence = kep4_mac_divergence(
        velocity,
        grid,
        solver.operator.boundaries,
    )

    x_work = jnp.sum(velocity.x[..., 1:-1] * gradient.x[..., 1:-1])
    x_work += 0.5 * jnp.sum(
        velocity.x[..., 0] * gradient.x[..., 0]
        + velocity.x[..., -1] * gradient.x[..., -1]
    )
    y_work = jnp.sum(velocity.y[:, 1:-1, :] * gradient.y[:, 1:-1, :])
    y_work += 0.5 * jnp.sum(
        velocity.y[:, 0, :] * gradient.y[:, 0, :]
        + velocity.y[:, -1, :] * gradient.y[:, -1, :]
    )
    z_work = jnp.sum(velocity.z * gradient.z)
    defect = x_work + y_work + z_work + jnp.sum(pressure * divergence)

    assert abs(float(defect)) < 2.0e-4


def test_kep4_mac_complex_matches_poisson_and_projects() -> None:
    grid = RectilinearGrid.uniform(8, 8, 8)
    solver = _kep4_solver(grid)
    pressure = jnp.cos(0.11 * jnp.arange(grid.cell_count, dtype=jnp.float32)).reshape(
        grid.shape
    )
    gradient = kep4_mac_pressure_gradient(
        pressure,
        grid,
        solver.operator.boundaries,
    )
    composed = -kep4_mac_divergence(
        gradient,
        grid,
        solver.operator.boundaries,
    )

    result = MACStageProjector(solver).project(
        _closed_velocity(grid, phase=0.23),
        timestep=0.01,
    )

    assert float(jnp.max(jnp.abs(composed - solver.operator.apply(pressure)))) < 5.0e-5
    assert result.linear_result.converged
    assert float(solver.operator.norm(result.divergence_after)) < 3.0e-4


def _closed_velocity(grid: RectilinearGrid, phase: float = 0.0) -> MACVelocity:
    nz, ny, nx = grid.shape
    x = jnp.sin(
        phase
        + 0.13
        * jnp.arange(nz * ny * (nx + 1), dtype=jnp.float32).reshape(nz, ny, nx + 1)
    )
    y = jnp.cos(
        phase
        + 0.11
        * jnp.arange(nz * (ny + 1) * nx, dtype=jnp.float32).reshape(nz, ny + 1, nx)
    )
    z = jnp.sin(
        phase
        + 0.07
        * jnp.arange((nz + 1) * ny * nx, dtype=jnp.float32).reshape(nz + 1, ny, nx)
    )
    return MACVelocity(
        x.at[..., 0].set(0.0).at[..., -1].set(0.0),
        y.at[:, 0, :].set(0.0).at[:, -1, :].set(0.0),
        z.at[0, ...].set(0.0).at[-1, ...].set(0.0),
    )


def test_mac_gradient_and_divergence_form_the_poisson_operator() -> None:
    grid = RectilinearGrid.uniform(8, 6, 10)
    solver = _wall_solver(grid)
    pressure = jnp.sin(0.17 * jnp.arange(grid.cell_count, dtype=jnp.float32)).reshape(
        grid.shape
    )

    composed = -mac_divergence(
        mac_pressure_gradient(
            pressure,
            grid,
            solver.operator.boundaries,
        ),
        grid,
    )

    assert float(jnp.max(jnp.abs(composed - solver.operator.apply(pressure)))) < (
        2.0e-4
    )


def test_mac_stage_projection_eliminates_divergence_and_preserves_walls() -> None:
    grid = RectilinearGrid.uniform(8, 6, 10)
    solver = _wall_solver(grid)
    velocity = _closed_velocity(grid)

    result = MACStageProjector(solver).project(
        velocity,
        timestep=0.01,
    )

    assert result.linear_result.converged
    assert float(solver.operator.norm(result.divergence_after)) < 3.0e-4
    assert float(jnp.max(jnp.abs(result.velocity.x[..., 0]))) == 0.0
    assert float(jnp.max(jnp.abs(result.velocity.x[..., -1]))) == 0.0
    assert float(jnp.max(jnp.abs(result.velocity.y[:, 0, :]))) == 0.0
    assert float(jnp.max(jnp.abs(result.velocity.y[:, -1, :]))) == 0.0
    assert float(jnp.max(jnp.abs(result.velocity.z[0, ...]))) == 0.0
    assert float(jnp.max(jnp.abs(result.velocity.z[-1, ...]))) == 0.0


def test_device_projection_path_avoids_diagnostic_synchronization() -> None:
    grid = RectilinearGrid.uniform(6, 4, 8)
    solver = MatrixFreePoissonSolver(
        grid,
        PoissonBoundaryConditions.homogeneous_neumann(),
        dtype=jnp.float32,
        krylov=FGMRESConfig(
            restart=10,
            max_iterations=40,
            relative_tolerance=2.0e-6,
            execution="jax",
        ),
    )
    projected = MACStageProjector(solver).project_velocity(
        _closed_velocity(grid),
        timestep=0.01,
    )
    divergence = mac_divergence(projected, grid)

    assert float(solver.operator.norm(divergence)) < 8.0e-4
    assert float(jnp.max(jnp.abs(projected.z[0, ...]))) == 0.0
    assert float(jnp.max(jnp.abs(projected.z[-1, ...]))) == 0.0


def test_ssprk3_projects_every_stage() -> None:
    grid = RectilinearGrid.uniform(6, 4, 8)
    solver = _wall_solver(grid)
    projector = MACStageProjector(solver)
    initial = _closed_velocity(grid)
    forcing = _closed_velocity(grid, phase=0.4)

    result = projected_ssprk3_step(
        initial,
        tendency=lambda _velocity, _time: forcing,
        projector=projector,
        timestep=0.005,
    )

    assert all(stage.linear_result.converged for stage in result.stages)
    assert all(
        float(solver.operator.norm(stage.divergence_after)) < 4.0e-4
        for stage in result.stages
    )


def test_fpj2_pressure_prediction_matches_constant_and_variable_steps() -> None:
    current = jnp.asarray((3.0,), dtype=jnp.float32)
    previous = jnp.asarray((1.0,), dtype=jnp.float32)

    second = fpj2_pressure_prediction(
        current,
        previous,
        current_timestep=2.0,
        previous_timestep=2.0,
        next_timestep=2.0,
        stage_abscissa=1.0,
    )
    third = fpj2_pressure_prediction(
        current,
        previous,
        current_timestep=2.0,
        previous_timestep=2.0,
        next_timestep=2.0,
        stage_abscissa=0.5,
    )
    variable = fpj2_pressure_prediction(
        current,
        previous,
        current_timestep=2.0,
        previous_timestep=1.0,
        next_timestep=3.0,
        stage_abscissa=0.5,
    )

    assert jnp.allclose(second, 5.0)
    assert jnp.allclose(third, 4.5)
    assert jnp.allclose(variable, 3.0 + (3.5 / 3.0) * 2.0)


def test_fpj2_uses_one_final_projection_and_remains_divergence_free() -> None:
    grid = RectilinearGrid.uniform(6, 4, 8)
    solver = _wall_solver(grid)
    projector = MACStageProjector(solver)
    initial = projector.project(
        _closed_velocity(grid),
        timestep=0.01,
    ).velocity
    forcing = _closed_velocity(grid, phase=0.4)

    def tendency(_velocity, _time):
        return forcing

    timestep = 0.005

    first = projected_ssprk3_velocity_pressure_step(
        initial,
        tendency=tendency,
        projector=projector,
        timestep=timestep,
    )
    second = projected_ssprk3_velocity_pressure_step(
        first.velocity,
        tendency=tendency,
        projector=projector,
        timestep=timestep,
        initial_pressure=first.pressure,
    )
    fast = fpj2_ssprk3_velocity_step(
        second.velocity,
        tendency=tendency,
        projector=projector,
        timestep=timestep,
        current_pressure=second.pressure,
        previous_pressure=first.pressure,
        current_timestep=timestep,
        previous_timestep=timestep,
    )
    reference = projected_ssprk3_velocity_pressure_step(
        second.velocity,
        tendency=tendency,
        projector=projector,
        timestep=timestep,
        initial_pressure=second.pressure,
    )

    fast_divergence = solver.operator.norm(mac_divergence(fast.velocity, grid))
    difference = max(
        float(jnp.max(jnp.abs(left - right)))
        for left, right in zip(fast.velocity, reference.velocity, strict=True)
    )
    assert float(fast_divergence) < 5.0e-4
    assert difference < 2.0e-4

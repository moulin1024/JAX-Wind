from __future__ import annotations

import math
import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jaxwind.domain import UniformGrid
from jaxwind.fv import (
    FREE_SLIP,
    Boundaries,
    FlowModel,
    StaggeredVelocity,
    Wall,
    build_pressure_poisson,
    build_run,
    build_step,
    divergence,
    initial_solution,
    kinetic_energy,
    project,
    stable_timestep,
)


FREE_SLIP_WALLS = Boundaries(Wall(FREE_SLIP), Wall(FREE_SLIP))


def taylor_green(grid: UniformGrid, viscosity: float, time: float) -> StaggeredVelocity:
    """The exact two-dimensional Taylor-Green solution on the staggered mesh."""
    decay = math.exp(-2.0 * viscosity * time)
    x_face = jnp.arange(grid.nx) * grid.dx
    y_face = jnp.arange(grid.ny) * grid.dy
    x_cell = x_face + 0.5 * grid.dx
    y_cell = y_face + 0.5 * grid.dy
    x_velocity = -jnp.cos(x_face)[None, None, :] * jnp.sin(y_cell)[None, :, None]
    y_velocity = jnp.sin(x_cell)[None, None, :] * jnp.cos(y_face)[None, :, None]
    plane = jnp.ones((grid.nz, 1, 1))
    return StaggeredVelocity(
        decay * plane * x_velocity,
        decay * plane * y_velocity,
        jnp.zeros((grid.nz + 1, grid.ny, grid.nx)),
    )


def taylor_green_grid(cells: int) -> UniformGrid:
    """A mesh with equal horizontal spacing, which the exact solution needs."""
    return UniformGrid(cells, cells, 2, 2.0 * math.pi, 2.0 * math.pi, 1.0)


def run_taylor_green(
    cells: int,
    *,
    viscosity: float,
    end_time: float,
    steps: int,
    scheme: str = "rk3",
) -> tuple[jnp.ndarray, StaggeredVelocity, StaggeredVelocity]:
    grid = taylor_green_grid(cells)
    poisson = build_pressure_poisson(grid, backend="cg")
    model = FlowModel(viscosity=viscosity)
    step = build_step(grid, FREE_SLIP_WALLS, poisson, model, scheme=scheme)
    run = build_run(step)
    start = initial_solution(grid, taylor_green(grid, viscosity, 0.0))
    final = run(start, end_time / steps, steps)
    return final, final.velocity, taylor_green(grid, viscosity, end_time)


class TaylorGreenTest(unittest.TestCase):
    viscosity = 0.05
    end_time = 0.5

    def test_the_initial_field_is_discretely_solenoidal(self) -> None:
        grid = taylor_green_grid(16)
        velocity = taylor_green(grid, self.viscosity, 0.0)
        self.assertLess(float(jnp.max(jnp.abs(divergence(velocity, grid)))), 1.0e-13)

    def test_the_solution_tracks_the_analytic_decay(self) -> None:
        final, velocity, exact = run_taylor_green(
            32,
            viscosity=self.viscosity,
            end_time=self.end_time,
            steps=16,
        )
        error = max(
            float(jnp.max(jnp.abs(velocity.x - exact.x))),
            float(jnp.max(jnp.abs(velocity.y - exact.y))),
        )
        self.assertLess(error, 2.0e-3)
        self.assertLess(float(jnp.max(jnp.abs(velocity.z))), 1.0e-11)
        self.assertAlmostEqual(float(final.time), self.end_time, places=12)

    def test_the_error_is_second_order_in_the_mesh_spacing(self) -> None:
        errors = []
        for cells, steps in ((16, 16), (32, 32)):
            _, velocity, exact = run_taylor_green(
                cells,
                viscosity=self.viscosity,
                end_time=self.end_time,
                steps=steps,
            )
            errors.append(float(jnp.max(jnp.abs(velocity.x - exact.x))))
        self.assertGreater(errors[0] / errors[1], 3.5)

    def test_the_velocity_stays_solenoidal_through_the_run(self) -> None:
        grid = taylor_green_grid(16)
        final, velocity, _ = run_taylor_green(
            16,
            viscosity=self.viscosity,
            end_time=self.end_time,
            steps=16,
        )
        residual = float(jnp.max(jnp.abs(divergence(velocity, grid))))
        self.assertLess(residual, 1.0e-9)

    def test_energy_decays_at_the_analytic_rate(self) -> None:
        grid = taylor_green_grid(32)
        _, velocity, _ = run_taylor_green(
            32,
            viscosity=self.viscosity,
            end_time=self.end_time,
            steps=16,
        )
        start = kinetic_energy(taylor_green(grid, self.viscosity, 0.0), grid)
        ratio = float(kinetic_energy(velocity, grid) / start)
        expected = math.exp(-4.0 * self.viscosity * self.end_time)
        self.assertAlmostEqual(ratio, expected, delta=2.0e-3)

    def test_the_two_time_schemes_agree(self) -> None:
        _, rk3, exact = run_taylor_green(
            16,
            viscosity=self.viscosity,
            end_time=self.end_time,
            steps=32,
            scheme="rk3",
        )
        _, ab2, _ = run_taylor_green(
            16,
            viscosity=self.viscosity,
            end_time=self.end_time,
            steps=32,
            scheme="ab2",
        )
        difference = float(jnp.max(jnp.abs(rk3.x - ab2.x)))
        error = float(jnp.max(jnp.abs(rk3.x - exact.x)))
        self.assertLess(difference, 2.0 * error)


class LaminarChannelTest(unittest.TestCase):
    """Plane Poiseuille flow, the reference case for the wall closure."""

    viscosity = 0.1
    forcing = 1.0

    end_time = 20.0

    def run_to_steady_state(self, cells: int) -> tuple[UniformGrid, jnp.ndarray]:
        grid = UniformGrid(4, 4, cells, 1.0, 1.0, 1.0)
        poisson = build_pressure_poisson(grid, backend="cg")
        model = FlowModel(viscosity=self.viscosity, body_force=(self.forcing, 0.0, 0.0))
        step = build_step(grid, Boundaries(), poisson, model, scheme="ab2")
        run = build_run(step)
        start = initial_solution(grid)
        limit = float(stable_timestep(start.velocity, grid, self.viscosity))
        steps = int(math.ceil(self.end_time / (0.9 * limit)))
        final = run(start, self.end_time / steps, steps)
        return grid, final.velocity.x[:, 0, 0]

    def exact_profile(self, grid: UniformGrid) -> jnp.ndarray:
        height = (jnp.arange(grid.nz) + 0.5) * grid.dz
        return (
            self.forcing
            / (2.0 * self.viscosity)
            * height
            * (grid.lz - height)
        )

    def test_the_steady_profile_is_the_exact_parabola(self) -> None:
        """The quadratic wall closure must reproduce Poiseuille flow exactly."""
        grid, profile = self.run_to_steady_state(8)
        exact = self.exact_profile(grid)
        error = float(jnp.max(jnp.abs(profile - exact)))
        self.assertLess(error, 1.0e-6 * float(jnp.max(exact)))

    def test_the_profile_stays_exact_under_refinement(self) -> None:
        for cells in (16,):
            grid, profile = self.run_to_steady_state(cells)
            exact = self.exact_profile(grid)
            error = float(jnp.max(jnp.abs(profile - exact)))
            self.assertLess(error, 1.0e-5 * float(jnp.max(exact)))

    def test_the_flow_stays_one_dimensional(self) -> None:
        grid = UniformGrid(4, 4, 8, 1.0, 1.0, 1.0)
        poisson = build_pressure_poisson(grid, backend="cg")
        model = FlowModel(viscosity=self.viscosity, body_force=(self.forcing, 0.0, 0.0))
        step = build_step(grid, Boundaries(), poisson, model, scheme="ab2")
        final = build_run(step)(initial_solution(grid), 0.01, 200)
        self.assertLess(float(jnp.max(jnp.abs(final.velocity.y))), 1.0e-14)
        self.assertLess(float(jnp.max(jnp.abs(final.velocity.z))), 1.0e-14)
        spread = float(jnp.max(jnp.std(final.velocity.x, axis=(1, 2))))
        self.assertLess(spread, 1.0e-14)


class FastRungeKuttaTest(unittest.TestCase):
    """One projection per step instead of three."""

    viscosity = 0.05
    end_time = 0.5

    def counting_poisson(self, grid: UniformGrid):
        """A pressure solver that records how often it is invoked."""
        from jaxwind.fv.poisson import PressurePoisson

        base = build_pressure_poisson(grid, backend="cg")
        calls: list[int] = []

        def counted(right_hand_side):
            calls.append(1)
            return base.linear_solver(right_hand_side)

        return PressurePoisson(base.grid, base.matrix, counted), calls

    def test_it_solves_the_pressure_once_per_step(self) -> None:
        grid = UniformGrid(8, 8, 8, 1.0, 1.0, 1.0)
        model = FlowModel(viscosity=0.01, body_force=(1.0, 0.0, 0.0))
        for scheme, expected in (("rk3", 3), ("fast-rk3", 1), ("ab2", 1)):
            poisson, calls = self.counting_poisson(grid)
            step = build_step(grid, Boundaries(), poisson, model, scheme=scheme)
            step(initial_solution(grid), 0.01)
            self.assertEqual(len(calls), expected, scheme)

    def test_the_step_still_ends_solenoidal(self) -> None:
        grid = taylor_green_grid(16)
        poisson = build_pressure_poisson(grid, backend="cg")
        step = build_step(
            grid,
            FREE_SLIP_WALLS,
            poisson,
            FlowModel(viscosity=self.viscosity),
            scheme="fast-rk3",
        )
        solution = initial_solution(grid, taylor_green(grid, self.viscosity, 0.0))
        for _ in range(4):
            solution = step(solution, 0.02)
            residual = float(jnp.max(jnp.abs(divergence(solution.velocity, grid))))
            self.assertLess(residual, 1.0e-9)

    def test_it_tracks_the_analytic_decay(self) -> None:
        _, velocity, exact = run_taylor_green(
            32,
            viscosity=self.viscosity,
            end_time=self.end_time,
            steps=16,
            scheme="fast-rk3",
        )
        error = float(jnp.max(jnp.abs(velocity.x - exact.x)))
        self.assertLess(error, 2.0e-3)

    def test_the_lagging_error_vanishes_with_the_step_size(self) -> None:
        """The price of the skipped projections must be a small, converging one."""
        differences = []
        for steps in (8, 16, 32):
            _, fast, _ = run_taylor_green(
                16,
                viscosity=self.viscosity,
                end_time=self.end_time,
                steps=steps,
                scheme="fast-rk3",
            )
            _, full, _ = run_taylor_green(
                16,
                viscosity=self.viscosity,
                end_time=self.end_time,
                steps=steps,
                scheme="rk3",
            )
            differences.append(float(jnp.max(jnp.abs(fast.x - full.x))))
        for coarse, fine in zip(differences, differences[1:]):
            self.assertGreater(coarse / fine, 3.5)

    def test_walls_stay_impermeable(self) -> None:
        grid = UniformGrid(8, 8, 8, 1.0, 1.0, 0.5)
        poisson = build_pressure_poisson(grid, backend="cg")
        step = build_step(
            grid,
            Boundaries(),
            poisson,
            FlowModel(viscosity=0.01, body_force=(1.0, 0.0, 0.0)),
            scheme="fast-rk3",
        )
        solution = initial_solution(grid)
        for _ in range(4):
            solution = step(solution, 0.01)
        self.assertLess(float(jnp.max(jnp.abs(solution.velocity.z[0]))), 1.0e-14)
        self.assertLess(float(jnp.max(jnp.abs(solution.velocity.z[-1]))), 1.0e-14)

    def test_a_converged_steady_state_needs_no_correction(self) -> None:
        """With the carried pressure right, the final solve has nothing to do."""
        grid = UniformGrid(4, 4, 8, 1.0, 1.0, 1.0)
        poisson = build_pressure_poisson(grid, backend="cg")
        model = FlowModel(viscosity=0.1, body_force=(1.0, 0.0, 0.0))
        step = build_step(grid, Boundaries(), poisson, model, scheme="fast-rk3")
        solution = build_run(step)(initial_solution(grid), 0.004, 5000)
        height = (jnp.arange(grid.nz) + 0.5) * grid.dz
        exact = 1.0 / 0.2 * height * (grid.lz - height)
        error = float(jnp.max(jnp.abs(solution.velocity.x[:, 0, 0] - exact)))
        self.assertLess(error, 1.0e-5 * float(jnp.max(exact)))


class StepTest(unittest.TestCase):
    grid = UniformGrid(8, 8, 4, 1.0, 1.0, 0.5)

    def build(self, scheme: str):
        poisson = build_pressure_poisson(self.grid, backend="cg")
        model = FlowModel(viscosity=0.01, body_force=(1.0, 0.0, 0.0))
        return build_step(self.grid, Boundaries(), poisson, model, scheme=scheme)

    def test_a_body_force_accelerates_the_fluid(self) -> None:
        step = self.build("ab2")
        start = initial_solution(self.grid)
        moved = step(start, 0.01)
        self.assertGreater(float(jnp.mean(moved.velocity.x)), 0.0)
        self.assertLess(float(jnp.max(jnp.abs(moved.velocity.z))), 1.0e-14)

    def test_walls_stay_impermeable_under_forcing(self) -> None:
        step = self.build("rk3")
        solution = initial_solution(self.grid)
        for _ in range(3):
            solution = step(solution, 0.01)
        self.assertLess(float(jnp.max(jnp.abs(solution.velocity.z[0]))), 1.0e-14)
        self.assertLess(float(jnp.max(jnp.abs(solution.velocity.z[-1]))), 1.0e-14)

    def test_the_compiled_loop_matches_stepping_by_hand(self) -> None:
        step = self.build("rk3")
        run = build_run(step)
        start = initial_solution(self.grid)
        stepped = start
        for _ in range(4):
            stepped = step(stepped, 0.01)
        looped = run(start, 0.01, 4)
        self.assertLess(
            float(jnp.max(jnp.abs(stepped.velocity.x - looped.velocity.x))),
            1.0e-12,
        )
        self.assertEqual(int(looped.step), 4)

    def test_rejects_an_unknown_scheme(self) -> None:
        poisson = build_pressure_poisson(self.grid, backend="cg")
        with self.assertRaises(ValueError):
            build_step(self.grid, Boundaries(), poisson, FlowModel(), scheme="euler")


if __name__ == "__main__":
    unittest.main()

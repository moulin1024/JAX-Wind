from __future__ import annotations

import math
import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from jaxwind.domain import UniformGrid
from jaxwind.fv import (
    CELL_AVERAGE,
    CELL_CENTRE,
    LOCAL,
    PLANAR,
    FlowModel,
    MoninObukhovWall,
    StaggeredVelocity,
    build_tendency,
    friction_velocity,
    logarithmic_profile,
    monin_obukhov_boundaries,
    surface_stress,
    wall_tendency,
)


FRICTION = 0.4
ROUGHNESS = 0.005


def channel(nz: int = 64) -> UniformGrid:
    return UniformGrid(8, 8, nz, 200.0, 200.0, 1000.0)


def uniform_log_flow(
    grid: UniformGrid,
    model: MoninObukhovWall,
    friction: float = FRICTION,
) -> StaggeredVelocity:
    """The cell-averaged logarithmic profile, uniform in x and y."""
    profile = logarithmic_profile(grid, friction, model)
    cells = (grid.nz, grid.ny, grid.nx)
    return StaggeredVelocity(
        jnp.broadcast_to(profile[:, None, None], cells),
        jnp.zeros(cells),
        jnp.zeros((grid.nz + 1, grid.ny, grid.nx)),
    )


class ReferenceHeightTest(unittest.TestCase):
    grid = channel()

    def test_cell_average_sampling_uses_the_integrated_height(self) -> None:
        model = MoninObukhovWall(ROUGHNESS, sampling=CELL_AVERAGE)
        self.assertAlmostEqual(
            model.reference_height(self.grid),
            self.grid.dz / math.e,
            places=12,
        )

    def test_cell_centre_sampling_uses_the_midpoint(self) -> None:
        model = MoninObukhovWall(ROUGHNESS, sampling=CELL_CENTRE)
        self.assertAlmostEqual(
            model.reference_height(self.grid),
            0.5 * self.grid.dz,
            places=12,
        )

    def test_a_first_cell_below_the_roughness_is_rejected(self) -> None:
        model = MoninObukhovWall(roughness=10.0)
        with self.assertRaises(ValueError):
            model.drag_coefficient(UniformGrid(4, 4, 4, 1.0, 1.0, 1.0))

    def test_rejects_unusable_parameters(self) -> None:
        for arguments in (
            {"roughness": 0.0},
            {"roughness": 0.1, "von_karman": 0.0},
            {"roughness": 0.1, "sampling": "surface"},
            {"roughness": 0.1, "averaging": "time"},
        ):
            with self.assertRaises(ValueError):
                MoninObukhovWall(**arguments)


class SurfaceStressTest(unittest.TestCase):
    """The equilibrium check the finite-volume reference height exists for."""

    grid = channel()

    def test_cell_average_sampling_reproduces_the_stress_exactly(self) -> None:
        model = MoninObukhovWall(ROUGHNESS, sampling=CELL_AVERAGE)
        stress_x, stress_y = surface_stress(
            uniform_log_flow(self.grid, model),
            self.grid,
            model,
        )
        self.assertAlmostEqual(
            float(jnp.max(jnp.abs(stress_x))),
            FRICTION**2,
            places=12,
        )
        self.assertLess(float(jnp.max(jnp.abs(stress_y))), 1.0e-14)

    def test_cell_centre_sampling_underestimates_the_stress(self) -> None:
        """The bias the finite-volume reference height removes."""
        model = MoninObukhovWall(ROUGHNESS, sampling=CELL_AVERAGE)
        biased = MoninObukhovWall(ROUGHNESS, sampling=CELL_CENTRE)
        flow = uniform_log_flow(self.grid, model)
        stress, _ = surface_stress(flow, self.grid, biased)
        logarithm = math.log(self.grid.dz / ROUGHNESS)
        expected = FRICTION**2 * (
            (logarithm - 1.0) / (logarithm - math.log(2.0))
        ) ** 2
        self.assertAlmostEqual(float(jnp.max(stress)), expected, places=12)
        self.assertLess(expected, FRICTION**2)

    def test_the_bias_grows_as_the_first_cell_is_refined(self) -> None:
        """Refining the mesh makes the midpoint reference height worse.

        The first cell moves closer to the roughness, where the logarithm is
        most strongly curved, so the gap between the cell average and the value
        at the cell centre widens.  A solver sampling at the midpoint therefore
        does not converge to the right surface stress under refinement, which
        is the grid-convergence failure the finite-volume reference height
        removes.
        """
        deficits = []
        for cells in (32, 128):
            grid = channel(cells)
            model = MoninObukhovWall(ROUGHNESS, sampling=CELL_AVERAGE)
            biased = MoninObukhovWall(ROUGHNESS, sampling=CELL_CENTRE)
            stress, _ = surface_stress(uniform_log_flow(grid, model), grid, biased)
            deficits.append(1.0 - float(jnp.max(stress)) / FRICTION**2)
        coarse, fine = deficits
        self.assertGreater(fine, coarse)
        self.assertGreater(coarse, 0.0)

    def test_the_friction_velocity_is_recovered(self) -> None:
        model = MoninObukhovWall(ROUGHNESS)
        recovered = friction_velocity(
            uniform_log_flow(self.grid, model),
            self.grid,
            model,
        )
        self.assertAlmostEqual(float(recovered), FRICTION, places=10)

    def test_the_stress_opposes_the_wind(self) -> None:
        model = MoninObukhovWall(ROUGHNESS)
        flow = uniform_log_flow(self.grid, model)
        reversed_flow = StaggeredVelocity(-flow.x, flow.y, flow.z)
        forward, _ = surface_stress(flow, self.grid, model)
        backward, _ = surface_stress(reversed_flow, self.grid, model)
        self.assertGreater(float(jnp.max(forward)), 0.0)
        self.assertLess(float(jnp.max(backward)), 0.0)

    def test_planar_averaging_matches_local_for_a_uniform_wind(self) -> None:
        flow = uniform_log_flow(self.grid, MoninObukhovWall(ROUGHNESS))
        local, _ = surface_stress(flow, self.grid, MoninObukhovWall(ROUGHNESS, averaging=LOCAL))
        planar, _ = surface_stress(
            flow,
            self.grid,
            MoninObukhovWall(ROUGHNESS, averaging=PLANAR),
        )
        self.assertLess(float(jnp.max(jnp.abs(local - planar))), 1.0e-12)

    def test_a_local_law_overpredicts_the_stress_of_a_fluctuating_wind(self) -> None:
        """The Schwarz-inequality bias of Bou-Zeid et al. (2005)."""
        model = MoninObukhovWall(ROUGHNESS, averaging=LOCAL)
        planar = MoninObukhovWall(ROUGHNESS, averaging=PLANAR)
        flow = uniform_log_flow(self.grid, model)
        noise = 0.3 * jax.random.normal(jax.random.PRNGKey(0), flow.x.shape)
        gusty = StaggeredVelocity(flow.x + noise, flow.y, flow.z)
        local, _ = surface_stress(gusty, self.grid, model)
        averaged, _ = surface_stress(gusty, self.grid, planar)
        self.assertGreater(float(jnp.mean(local)), float(jnp.mean(averaged)))


class WallTendencyTest(unittest.TestCase):
    grid = channel()
    model = MoninObukhovWall(ROUGHNESS)

    def test_the_drag_reaches_only_the_wall_adjacent_cells(self) -> None:
        tendency = wall_tendency(
            uniform_log_flow(self.grid, self.model),
            self.grid,
            self.model,
        )
        self.assertLess(float(jnp.max(jnp.abs(tendency.x[1:]))), 1.0e-15)
        self.assertLess(float(jnp.max(jnp.abs(tendency.z))), 1.0e-15)
        self.assertLess(float(jnp.max(tendency.x[0])), 0.0)

    def test_the_drag_balances_the_driving_force_at_equilibrium(self) -> None:
        """A neutral half-channel is in balance when the force equals u*^2/H."""
        flow = uniform_log_flow(self.grid, self.model)
        forcing = FRICTION**2 / self.grid.lz
        tendency = build_tendency(
            self.grid,
            monin_obukhov_boundaries(),
            FlowModel(body_force=(forcing, 0.0, 0.0), surface=self.model),
        )(flow, 0.0)
        column = float(jnp.mean(jnp.sum(tendency.x, axis=0)) * self.grid.dz)
        self.assertLess(abs(column), 1.0e-12 * forcing * self.grid.lz)


if __name__ == "__main__":
    unittest.main()

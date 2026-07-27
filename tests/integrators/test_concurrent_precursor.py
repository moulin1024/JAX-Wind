from __future__ import annotations

import unittest

import jax.numpy as jnp

from wireles.domain import (
    AcceptedClock,
    Cell,
    Evaluated,
    Field,
    GlobalTestRegion,
    Projected,
    UniformGrid,
    VerticalBoundary,
    VerticalVelocity,
    VerticalVelocityTendency,
    XVelocity,
    XVelocityTendency,
    YVelocity,
    YVelocityTendency,
    ZFace,
)
from wireles.effects import SideBySideStreamLauncher
from wireles.integrators import (
    AB2Config,
    ConcurrentPrecursorState,
    VectorFieldResult,
    cold_start,
    step_concurrent_precursor,
)
from wireles.interpreters.jax_reference import (
    JaxReferencePressureSolver,
    JaxReferenceProjection,
)
from wireles.operators import VelocityVector
from wireles.physics import ConcurrentPrecursorEnvironment


class ConcurrentPrecursorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = UniformGrid(4, 4, 4, 4.0, 4.0, 4.0)
        self.cells = GlobalTestRegion(self.grid, Cell)
        self.faces = GlobalTestRegion(self.grid, ZFace)
        self.config = AB2Config(0.1)
        self.algebra = JaxReferenceProjection()

    def velocity(self, value: float) -> VelocityVector:
        return VelocityVector(
            Field(
                XVelocity,
                Cell,
                self.cells,
                Projected,
                jnp.full(self.cells.storage_shape, value),
            ),
            Field(
                YVelocity,
                Cell,
                self.cells,
                Projected,
                jnp.zeros(self.cells.storage_shape),
            ),
            Field(
                VerticalVelocity,
                ZFace,
                self.faces,
                Projected,
                jnp.zeros(self.faces.storage_shape),
            ),
        )

    def tendency(self, value) -> VelocityVector:
        return VelocityVector(
            Field(
                XVelocityTendency,
                Cell,
                self.cells,
                Evaluated,
                jnp.full(self.cells.storage_shape, value),
            ),
            Field(
                YVelocityTendency,
                Cell,
                self.cells,
                Evaluated,
                jnp.zeros(self.cells.storage_shape),
            ),
            Field(
                VerticalVelocityTendency,
                ZFace,
                self.faces,
                Evaluated,
                jnp.zeros(self.faces.storage_shape),
            ),
        )

    def test_main_consumes_synchronized_precursor_field(self) -> None:
        state = ConcurrentPrecursorState(
            cold_start(
                self.velocity(1.0),
                clock=AcceptedClock(0.0, 0),
                config=self.config,
            ),
            cold_start(
                self.velocity(0.0),
                clock=AcceptedClock(0.0, 0),
                config=self.config,
            ),
        )

        def precursor(evaluation):
            self.assertIsNone(evaluation.environment)
            return VectorFieldResult(self.tendency(1.0), evaluation.time)

        def main(evaluation):
            self.assertIsInstance(
                evaluation.environment,
                ConcurrentPrecursorEnvironment,
            )
            target = jnp.mean(evaluation.environment.velocity.x.payload)
            return VectorFieldResult(self.tendency(target), evaluation.time)

        kwargs = dict(
            config=self.config,
            precursor_vector_field=precursor,
            main_vector_field=main,
            normal_boundary=lambda _clock, _environment: VerticalBoundary(0.0, 0.0),
            algebra=self.algebra,
            precursor_pressure_solver=JaxReferencePressureSolver(),
            main_pressure_solver=JaxReferencePressureSolver(),
        )
        with SideBySideStreamLauncher(execution_streams=False) as launcher:
            first = step_concurrent_precursor(
                state,
                launch_pair=launcher,
                **kwargs,
            )
            second = step_concurrent_precursor(
                first.state,
                launch_pair=launcher,
                **kwargs,
            )

        self.assertAlmostEqual(
            float(jnp.mean(first.state.precursor.velocity.x.payload)),
            1.1,
        )
        self.assertAlmostEqual(
            float(jnp.mean(first.state.main.velocity.x.payload)),
            0.1,
        )
        self.assertAlmostEqual(
            float(jnp.mean(second.state.main.velocity.x.payload)),
            0.215,
        )
        self.assertEqual(second.state.precursor.clock, second.state.main.clock)


if __name__ == "__main__":
    unittest.main()

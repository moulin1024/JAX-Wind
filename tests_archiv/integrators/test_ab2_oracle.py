"""AB2 laws evaluated with the independent global test oracle."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from jaxwind_archiv.domain import (  # noqa: E402
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
from jaxwind_archiv.effects import (  # noqa: E402
    ReferenceCheckpointLayout,
    load_ab2_checkpoint,
    save_ab2_checkpoint,
)
from jaxwind_archiv.integrators import (  # noqa: E402
    AB2Config,
    ColdStart,
    PreviousTendency,
    VectorFieldResult,
    cold_start,
    step,
)
from tests.support.jax_oracle import (  # noqa: E402
    JaxOraclePressureSolver,
    JaxOracleProjection,
)
from jaxwind_archiv.operators import VelocityVector  # noqa: E402


class ReferenceAB2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = UniformGrid(4, 4, 4, 4.0, 4.0, 4.0)
        self.cells = GlobalTestRegion(self.grid, Cell)
        self.faces = GlobalTestRegion(self.grid, ZFace)
        self.algebra = JaxOracleProjection()
        self.pressure_solver = JaxOraclePressureSolver()

    def velocity(self, *, dtype=jnp.float64) -> VelocityVector:
        return VelocityVector(
            Field(
                XVelocity,
                Cell,
                self.cells,
                Projected,
                jnp.zeros(self.cells.storage_shape, dtype),
            ),
            Field(
                YVelocity,
                Cell,
                self.cells,
                Projected,
                jnp.zeros(self.cells.storage_shape, dtype),
            ),
            Field(
                VerticalVelocity,
                ZFace,
                self.faces,
                Projected,
                jnp.zeros(self.faces.storage_shape, dtype),
            ),
        )

    def tendency(self, x_value, *, dtype=jnp.float64) -> VelocityVector:
        return VelocityVector(
            Field(
                XVelocityTendency,
                Cell,
                self.cells,
                Evaluated,
                jnp.full(self.cells.storage_shape, x_value, dtype),
            ),
            Field(
                YVelocityTendency,
                Cell,
                self.cells,
                Evaluated,
                jnp.zeros(self.cells.storage_shape, dtype),
            ),
            Field(
                VerticalVelocityTendency,
                ZFace,
                self.faces,
                Evaluated,
                jnp.zeros(self.faces.storage_shape, dtype),
            ),
        )

    def advance(self, state, config, vector_field, boundary=None):
        return step(
            state,
            config=config,
            environment={"rate": 2.0},
            vector_field=vector_field,
            normal_boundary=(
                boundary
                if boundary is not None
                else lambda _clock, _environment: VerticalBoundary(0.0, 0.0)
            ),
            algebra=self.algebra,
            pressure_solver=self.pressure_solver,
        )

    def test_euler_startup_then_ab2_and_explicit_times(self) -> None:
        config = AB2Config(0.1)
        state = cold_start(
            self.velocity(),
            clock=AcceptedClock(0.0, 0),
            config=config,
        )

        def vector_field(evaluation):
            value = 1.0 + evaluation.environment["rate"] * evaluation.time.time
            return VectorFieldResult(self.tendency(value), evaluation.time.time)

        expected = (0.1, 0.23, 0.38)
        evaluation_times = []
        for index, expected_value in enumerate(expected):
            result = self.advance(state, config, vector_field)
            state = result.state
            evaluation_times.append(result.diagnostic.vector_field)
            self.assertAlmostEqual(
                float(jnp.mean(state.velocity.x.payload)),
                expected_value,
                places=13,
            )
            self.assertEqual(result.diagnostic.used_euler_startup, index == 0)
            self.assertEqual(result.diagnostic.accepted_clock.step, index + 1)
        self.assertEqual(evaluation_times, [0.0, 0.1, 0.2])
        self.assertIsInstance(state.history, PreviousTendency)

    def test_zero_vector_field_leaves_projected_state_unchanged(self) -> None:
        config = AB2Config(0.25)
        initial = self.velocity(dtype=jnp.float32)
        state = cold_start(initial, clock=AcceptedClock(0.0, 0), config=config)

        result = self.advance(
            state,
            config,
            lambda evaluation: VectorFieldResult(
                self.tendency(0.0, dtype=jnp.float32),
                evaluation.time,
            ),
        )

        for component in ("x", "y", "z"):
            before = getattr(initial, component).payload
            after = getattr(result.state.velocity, component).payload
            self.assertEqual(float(jnp.max(jnp.abs(before - after))), 0.0)
            self.assertEqual(after.dtype, jnp.float32)
        self.assertEqual(
            result.state.history.value.x.payload.dtype,
            jnp.float32,
        )

    def test_normal_boundary_is_sampled_at_the_accepted_time(self) -> None:
        config = AB2Config(0.1)
        state = cold_start(
            self.velocity(),
            clock=AcceptedClock(0.0, 0),
            config=config,
        )
        result = self.advance(
            state,
            config,
            lambda evaluation: VectorFieldResult(
                self.tendency(0.0),
                evaluation.time,
            ),
            boundary=lambda clock, _environment: VerticalBoundary(
                clock.time,
                clock.time,
            ),
        )

        self.assertLess(
            float(jnp.max(jnp.abs(result.state.velocity.z.payload - 0.1))),
            3.0e-13,
        )
        self.assertEqual(result.diagnostic.evaluation_time.time, 0.0)
        self.assertEqual(result.diagnostic.accepted_clock.time, 0.1)

    def test_manufactured_time_forcing_has_second_order_error(self) -> None:
        def error(dt: float) -> float:
            config = AB2Config(dt)
            state = cold_start(
                self.velocity(),
                clock=AcceptedClock(0.0, 0),
                config=config,
            )

            def vector_field(evaluation):
                return VectorFieldResult(
                    self.tendency(evaluation.time.time),
                    evaluation.time.time,
                )

            for _ in range(round(0.4 / dt)):
                state = self.advance(state, config, vector_field).state
            exact = 0.5 * 0.4**2
            return abs(float(jnp.mean(state.velocity.x.payload)) - exact)

        coarse = error(0.1)
        fine = error(0.05)
        self.assertAlmostEqual(coarse / fine, 4.0, places=11)

    def test_serialized_restart_matches_uninterrupted_continuation(self) -> None:
        config = AB2Config(0.1)

        def vector_field(evaluation):
            return VectorFieldResult(
                self.tendency(1.0 + 2.0 * evaluation.time.time),
                evaluation.time.time,
            )

        initial = cold_start(
            self.velocity(),
            clock=AcceptedClock(0.0, 0),
            config=config,
        )
        uninterrupted = initial
        for _ in range(5):
            uninterrupted = self.advance(
                uninterrupted,
                config,
                vector_field,
            ).state

        checkpointed = initial
        for _ in range(2):
            checkpointed = self.advance(
                checkpointed,
                config,
                vector_field,
            ).state
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rank0000.npz"
            save_ab2_checkpoint(path, checkpointed)
            restarted = load_ab2_checkpoint(
                path,
                layout=ReferenceCheckpointLayout(self.grid, jnp.asarray),
                config=config,
            )
        for _ in range(3):
            restarted = self.advance(restarted, config, vector_field).state

        self.assertEqual(restarted.clock, uninterrupted.clock)
        for component in ("x", "y", "z"):
            left = getattr(restarted.velocity, component).payload
            right = getattr(uninterrupted.velocity, component).payload
            self.assertEqual(float(jnp.max(jnp.abs(left - right))), 0.0)
        self.assertIsInstance(restarted.history, PreviousTendency)
        self.assertEqual(
            float(
                jnp.max(
                    jnp.abs(
                        restarted.history.value.x.payload
                        - uninterrupted.history.value.x.payload
                    )
                )
            ),
            0.0,
        )

    def test_checkpoint_preserves_cold_tag_and_rejects_changed_dt(self) -> None:
        config = AB2Config(0.1)
        state = cold_start(
            self.velocity(),
            clock=AcceptedClock(0.0, 0),
            config=config,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cold.npz"
            save_ab2_checkpoint(path, state)
            loaded = load_ab2_checkpoint(
                path,
                layout=ReferenceCheckpointLayout(self.grid, jnp.asarray),
                config=config,
            )
            self.assertIsInstance(loaded.history, ColdStart)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                load_ab2_checkpoint(
                    path,
                    layout=ReferenceCheckpointLayout(self.grid, jnp.asarray),
                    config=AB2Config(0.2),
                )


if __name__ == "__main__":
    unittest.main()

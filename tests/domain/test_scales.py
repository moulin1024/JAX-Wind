from __future__ import annotations

import math
import unittest

from wireles.domain import (
    BoussinesqScaleSystem,
    PassiveScalarScaleSystem,
    ScaleSystem,
    UniformGrid,
)


class ScaleSystemTests(unittest.TestCase):
    def test_passive_scalar_flux_and_tendency_round_trip(self) -> None:
        scales = PassiveScalarScaleSystem(ScaleSystem(1000.0, 10.0), 0.5)
        self.assertAlmostEqual(scales.to_execution_concentration(1.25), 2.5)
        self.assertAlmostEqual(
            scales.from_execution_concentration_tendency(
                scales.to_execution_concentration_tendency(0.012),
            ),
            0.012,
        )
        self.assertAlmostEqual(
            scales.from_execution_concentration_flux(
                scales.to_execution_concentration_flux(0.001),
            ),
            0.001,
        )
        self.assertIn("passive-scalar-scales", scales.fingerprint)

    def test_boussinesq_temperature_and_buoyancy_scales(self) -> None:
        scales = BoussinesqScaleSystem(ScaleSystem(1000.0, 8.0), 10.0)
        theta = 7.5
        self.assertEqual(
            scales.from_execution_potential_temperature(
                scales.to_execution_potential_temperature(theta)
            ),
            theta,
        )
        coefficient = scales.to_execution_buoyancy_coefficient(
            gravity=9.81,
            reference_potential_temperature=300.0,
        )
        recovered_acceleration = (
            coefficient
            * scales.to_execution_potential_temperature(theta)
            * scales.mechanical.acceleration
        )
        self.assertAlmostEqual(recovered_acceleration, 9.81 * theta / 300.0)
        self.assertAlmostEqual(
            scales.from_execution_temperature_flux(
                scales.to_execution_temperature_flux(0.06),
            ),
            0.06,
        )

    def test_mechanical_round_trips_and_derived_scales(self) -> None:
        scales = ScaleSystem(1000.0, 8.0)
        self.assertEqual(scales.time, 125.0)
        self.assertEqual(scales.acceleration, 0.064)
        self.assertEqual(scales.inverse_time, 0.008)
        pairs = (
            (scales.to_execution_length, scales.from_execution_length, 4000.0),
            (scales.to_execution_velocity, scales.from_execution_velocity, -3.25),
            (scales.to_execution_time, scales.from_execution_time, 3600.0),
            (
                scales.to_execution_acceleration,
                scales.from_execution_acceleration,
                0.002,
            ),
            (
                scales.to_execution_inverse_time,
                scales.from_execution_inverse_time,
                1.0e-4,
            ),
        )
        for lower, lift, value in pairs:
            with self.subTest(value=value):
                self.assertTrue(
                    math.isclose(lift(lower(value)), value, rel_tol=1.0e-15)
                )

    def test_grid_lowering_changes_only_lengths(self) -> None:
        grid = UniformGrid(16, 16, 32, 4000.0, 4000.0, 1000.0)
        execution = ScaleSystem(1000.0, 8.0).to_execution_grid(grid)
        self.assertEqual((execution.nx, execution.ny, execution.nz), (16, 16, 32))
        self.assertEqual((execution.lx, execution.ly, execution.lz), (4.0, 4.0, 1.0))


if __name__ == "__main__":
    unittest.main()

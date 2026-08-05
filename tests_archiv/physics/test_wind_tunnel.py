from __future__ import annotations

from types import SimpleNamespace
import unittest

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from jaxwind_archiv.domain import (  # noqa: E402
    AddressableField,
    Cell,
    DistributionSpec,
    EqualZSlab,
    Evaluated,
    Field,
    GlobalTestRegion,
    MeshAxis,
    MeshTopology,
    Projected,
    UniformGrid,
    VerticalVelocity,
    VerticalVelocityTendency,
    XVelocity,
    XVelocityTendency,
    YVelocity,
    YVelocityTendency,
    ZFace,
)
from tests.support.jax_oracle import JaxOracleProjection  # noqa: E402
from jaxwind_archiv.interpreters._jax_fringe import plateau_fringe_mask  # noqa: E402
from jaxwind_archiv.interpreters.jax_zslab import (  # noqa: E402
    ZFaceFieldContext,
    build_zslab_interpreter,
)
from jaxwind_archiv.operators import VelocityVector  # noqa: E402
from jaxwind_archiv.physics import (  # noqa: E402
    BladeElementActuatorLine,
    BoussinesqFields,
    BoussinesqTendency,
    ConcurrentPrecursorEnvironment,
    ConcurrentPrecursorFringe,
    PureThrustActuatorDisk,
    WindTunnelBoussinesqVectorField,
    WindTunnelModel,
)


class WindTunnelForcingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = UniformGrid(8, 6, 4, 8.0, 6.0, 4.0)
        z = jnp.arange(self.grid.nz, dtype=jnp.float64)[:, None, None]
        zf = jnp.arange(self.grid.nz + 1, dtype=jnp.float64)[:, None, None]
        y = jnp.arange(self.grid.ny, dtype=jnp.float64)[None, :, None]
        x = jnp.arange(self.grid.nx, dtype=jnp.float64)[None, None, :]
        self.u = 2.0 + 0.03 * x + 0.02 * y + 0.01 * z
        self.v = -0.2 + 0.01 * x - 0.02 * y + 0.0 * z
        self.w = 0.01 * x + 0.02 * y + 0.03 * zf
        self.target_u = self.u + 0.4
        self.target_v = self.v - 0.1
        self.target_w = self.w + 0.2
        self.model = WindTunnelModel(
            PureThrustActuatorDisk(
                3.5,
                3.0,
                2.0,
                2.5,
                1.1,
                0.6,
                0.5,
            ),
            ConcurrentPrecursorFringe(6.0, 0.5),
        )

    def reference_velocity(self, u, v, w) -> VelocityVector:
        cells = GlobalTestRegion(self.grid, Cell)
        faces = GlobalTestRegion(self.grid, ZFace)
        return VelocityVector(
            Field(XVelocity, Cell, cells, Projected, u),
            Field(YVelocity, Cell, cells, Projected, v),
            Field(VerticalVelocity, ZFace, faces, Projected, w),
        )

    def zslab_velocity(self, u, v, w) -> VelocityVector:
        decomposition = EqualZSlab(
            self.grid,
            MeshTopology((MeshAxis("z", 1),)),
            DistributionSpec.z_slab(),
        )
        return VelocityVector(
            AddressableField(
                XVelocity,
                Cell,
                decomposition.regions(Cell),
                Projected,
                u[None],
            ),
            AddressableField(
                YVelocity,
                Cell,
                decomposition.regions(Cell),
                Projected,
                v[None],
            ),
            ZFaceFieldContext(
                AddressableField(
                    VerticalVelocity,
                    ZFace,
                    decomposition.regions(ZFace),
                    Projected,
                    w[None, 1:],
                ),
                w[0],
            ),
        )

    def test_oracle_and_zslab_forcing_commute(self) -> None:
        reference_velocity = self.reference_velocity(self.u, self.v, self.w)
        reference_target = self.reference_velocity(
            self.target_u, self.target_v, self.target_w
        )
        reference = JaxOracleProjection().wind_tunnel_tendency(
            reference_velocity,
            self.model,
            ConcurrentPrecursorEnvironment(reference_target),
        )

        decomposition = EqualZSlab(
            self.grid,
            MeshTopology((MeshAxis("z", 1),)),
            DistributionSpec.z_slab(),
        )
        production = build_zslab_interpreter(
            decomposition,
            addressable_shards=(0,),
        ).wind_tunnel_tendency(
            self.zslab_velocity(self.u, self.v, self.w),
            self.model,
            ConcurrentPrecursorEnvironment(
                self.zslab_velocity(self.target_u, self.target_v, self.target_w)
            ),
        )
        errors = (
            jnp.max(jnp.abs(reference.x.payload - production.x.payload[0])),
            jnp.max(jnp.abs(reference.y.payload - production.y.payload[0])),
            jnp.max(jnp.abs(reference.z.payload[1:] - production.z.owned.payload[0])),
        )
        self.assertLess(max(float(value) for value in errors), 2.0e-12)

    def test_configured_fringe_has_a_unit_plateau_and_smooth_seam(self) -> None:
        x = jnp.asarray(
            [5.9, 6.0, 6.25, 6.5, 7.0, 7.5, 7.75, 8.0],
            dtype=jnp.float64,
        )
        mask = plateau_fringe_mask(
            x,
            start_x=6.0,
            end_x=8.0,
            rise_width=0.5,
            fall_width=0.5,
        )

        self.assertEqual(float(mask[0]), 0.0)
        self.assertEqual(float(mask[1]), 0.0)
        self.assertAlmostEqual(float(mask[2]), 0.5)
        self.assertTrue(jnp.all(mask[3:6] == 1.0))
        self.assertAlmostEqual(float(mask[6]), 0.5)
        self.assertEqual(float(mask[7]), 0.0)

    def test_fringe_tendency_relaxes_toward_precursor_on_the_plateau(self) -> None:
        velocity = self.reference_velocity(self.u, self.v, self.w)
        target = self.reference_velocity(
            self.target_u,
            self.target_v,
            self.target_w,
        )
        tendency = JaxOracleProjection().wind_tunnel_tendency(
            velocity,
            WindTunnelModel(
                fringe=ConcurrentPrecursorFringe(
                    6.0,
                    0.5,
                    rise_width=0.5,
                    fall_width=0.5,
                )
            ),
            ConcurrentPrecursorEnvironment(target),
        )

        expected_u = (self.target_u - self.u) / 0.5
        self.assertEqual(float(jnp.max(jnp.abs(tendency.x.payload[..., :6]))), 0.0)
        self.assertLess(
            float(jnp.max(jnp.abs(tendency.x.payload[..., 6:] - expected_u[..., 6:]))),
            2.0e-12,
        )

    def test_zero_yaw_disk_removes_streamwise_momentum(self) -> None:
        velocity = self.reference_velocity(
            jnp.full_like(self.u, 2.0),
            jnp.zeros_like(self.v),
            jnp.zeros_like(self.w),
        )
        tendency = JaxOracleProjection().wind_tunnel_tendency(
            velocity,
            WindTunnelModel(actuator_disk=self.model.actuator_disk),
            None,
        )
        self.assertLess(float(jnp.sum(tendency.x.payload)), 0.0)
        self.assertEqual(float(jnp.max(jnp.abs(tendency.y.payload))), 0.0)
        self.assertEqual(float(jnp.max(jnp.abs(tendency.z.payload))), 0.0)

    def test_oracle_and_zslab_actuator_lines_commute(self) -> None:
        line = BladeElementActuatorLine(
            x=3.5,
            y=3.0,
            z=2.0,
            blade_count=3,
            hub_radius=0.25,
            tip_radius=1.0,
            angular_velocity=2.0,
            smoothing_width=0.5,
            element_radii=(0.25, 0.625, 1.0),
            element_widths=(0.1875, 0.375, 0.1875),
            element_chords=(0.2, 0.15, 0.1),
            element_twist_degrees=(8.0, 4.0, 0.0),
            element_airfoil_ids=(0, 0, 0),
            polar_alpha_degrees=(-180.0, 0.0, 180.0),
            polar_lift_coefficients=((0.0, 0.0, 0.0),),
            polar_drag_coefficients=((0.1, 0.1, 0.1),),
            tip_loss=False,
            root_loss=False,
        )
        velocity = self.reference_velocity(
            jnp.full_like(self.u, 2.0),
            jnp.zeros_like(self.v),
            jnp.zeros_like(self.w),
        )
        reference = JaxOracleProjection().wind_tunnel_tendency(
            velocity,
            WindTunnelModel(actuator_line=line),
            None,
        )

        decomposition = EqualZSlab(
            self.grid,
            MeshTopology((MeshAxis("z", 1),)),
            DistributionSpec.z_slab(),
        )
        production = build_zslab_interpreter(
            decomposition,
            addressable_shards=(0,),
        ).wind_tunnel_tendency(
            self.zslab_velocity(
                jnp.full_like(self.u, 2.0),
                jnp.zeros_like(self.v),
                jnp.zeros_like(self.w),
            ),
            WindTunnelModel(actuator_line=line),
            None,
        )

        errors = (
            jnp.max(jnp.abs(reference.x.payload - production.x.payload[0])),
            jnp.max(jnp.abs(reference.y.payload - production.y.payload[0])),
            jnp.max(
                jnp.abs(
                    reference.z.payload[1:]
                    - production.z.owned.payload[0]
                )
            ),
        )
        self.assertLess(max(float(value) for value in errors), 2.0e-12)
        self.assertLess(float(jnp.sum(reference.x.payload)), 0.0)

    def test_disk_projection_conserves_thrust_after_grid_translation(self) -> None:
        velocity = self.reference_velocity(
            jnp.full_like(self.u, 2.0),
            jnp.zeros_like(self.v),
            jnp.zeros_like(self.w),
        )
        totals = []
        for disk_y in (3.0, 3.37):
            disk = PureThrustActuatorDisk(
                3.5,
                disk_y,
                2.0,
                2.5,
                1.1,
                0.6,
                0.5,
                filtered_velocity_correction=False,
            )
            tendency = JaxOracleProjection().wind_tunnel_tendency(
                velocity,
                WindTunnelModel(actuator_disk=disk),
                None,
            )
            totals.append(float(jnp.sum(tendency.x.payload)))

        expected = -0.5 * 1.1 * 2.0**2 * jnp.pi * 2.5**2 / 4.0
        self.assertAlmostEqual(totals[0], float(expected), places=11)
        self.assertAlmostEqual(totals[1], float(expected), places=11)

    def test_fringe_requires_explicit_precursor_environment(self) -> None:
        with self.assertRaisesRegex(TypeError, "ConcurrentPrecursorEnvironment"):
            JaxOracleProjection().wind_tunnel_tendency(
                self.reference_velocity(self.u, self.v, self.w),
                WindTunnelModel(fringe=ConcurrentPrecursorFringe(6.0, 0.5)),
                None,
            )

    def test_boussinesq_wrapper_only_augments_momentum(self) -> None:
        cells = GlobalTestRegion(self.grid, Cell)
        faces = GlobalTestRegion(self.grid, ZFace)
        zero_momentum = VelocityVector(
            Field(
                XVelocityTendency,
                Cell,
                cells,
                Evaluated,
                jnp.zeros_like(self.u),
            ),
            Field(
                YVelocityTendency,
                Cell,
                cells,
                Evaluated,
                jnp.zeros_like(self.v),
            ),
            Field(
                VerticalVelocityTendency,
                ZFace,
                faces,
                Evaluated,
                jnp.zeros_like(self.w),
            ),
        )
        scalar_tendency = object()

        def base(_evaluation):
            return SimpleNamespace(
                tendency=BoussinesqTendency(zero_momentum, scalar_tendency),
                diagnostic="boussinesq",
            )

        velocity = self.reference_velocity(
            jnp.full_like(self.u, 2.0),
            jnp.zeros_like(self.v),
            jnp.zeros_like(self.w),
        )
        wrapper = WindTunnelBoussinesqVectorField(
            JaxOracleProjection(),
            base,
            WindTunnelModel(actuator_disk=self.model.actuator_disk),
        )
        result = wrapper(
            SimpleNamespace(
                velocity=BoussinesqFields(velocity, object()),
                environment=None,
            )
        )

        self.assertIs(result.tendency.potential_temperature, scalar_tendency)
        self.assertLess(float(jnp.sum(result.tendency.velocity.x.payload)), 0.0)
        self.assertTrue(result.diagnostic.actuator_disk_enabled)
        self.assertFalse(result.diagnostic.concurrent_fringe_enabled)


if __name__ == "__main__":
    unittest.main()

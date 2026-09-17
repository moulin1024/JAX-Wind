"""Independent finite-gas analytic Stokes settling, including position error."""

import jax
import jax.numpy as jnp
import numpy as np
from test_spray_cell_step import C, P, budget

from jaxwind.domain import UniformGrid
from jaxwind.physics.moisture import saturation_mixing_ratio
from jaxwind.spray_core import WaterCoreBins
from jaxwind.spray_low_mach import moist_gas_from_primitive
from jaxwind.spray_moving import build_moving_source
from jaxwind.state import StaggeredVelocity


def test_finite_gas_stokes_settling_against_continuous_two_mass_solution():
    grid = UniformGrid(4, 3, 4, 0.4, 0.3, 0.4)
    shape = (4, 3, 4)
    qs = saturation_mixing_ratio(jnp.asarray(300.0), C.pressure, C)
    # Slight supersaturation disables evaporation in the no-condensation water
    # law; this isolates fixed-mass Stokes dynamics without mocking its drag.
    qv = qs + 1e-6
    gas = moist_gas_from_primitive(
        jnp.full(shape, 300.0), jnp.full(shape, qv / (1 + qv)), C
    )
    u = StaggeredVelocity(jnp.zeros((4, 3, 5)), jnp.zeros(shape), jnp.zeros((5, 3, 4)))
    e = jnp.zeros(shape)
    diameter = 20e-6
    liquid = WaterCoreBins(
        jnp.array([C.water_density * np.pi * diameter**3 / 6]),
        jnp.array([1e8]),
        jnp.zeros((3, 1)),
        jnp.array([300.0]),
    )
    pos = jnp.array([[0.15], [0.15], [0.15]])
    step = build_moving_source(
        grid,
        C,
        P,
        periodic_x=False,
        periodic_y=True,
        drag_heat_fraction=0.0,
        particle_acceleration=(0.0, 0.0, -9.81),
        max_segments=2,
    )
    total_time = 0.02
    initial = (
        gas,
        u,
        e,
        liquid,
        pos,
        jnp.asarray(True),
        jnp.zeros(3),
        jnp.asarray(0.0),
    )

    @jax.jit
    def integrate(steps):
        dt = total_time / steps

        def body(_, state):
            gas, u, e, liquid, pos, ok, impulse, work = state
            r = step(gas, u, e, liquid, dt, pos)
            s = r.source
            return (
                s.gas,
                s.velocity,
                s.unresolved_density,
                s.liquid,
                r.position,
                ok & s.accepted,
                impulse + jnp.sum(r.external_impulse, axis=1),
                work + jnp.sum(r.external_work),
            )

        return jax.lax.fori_loop(0, steps, body, initial)

    # Derive the continuous ODE from actual gathered face masses, independently
    # of the implementation's split relaxation and body-force routines.
    m = float(liquid.mass[0] * liquid.multiplicity[0])
    rho = float((gas.dry_density + gas.vapor_density)[1, 1, 1])
    meff = 2 * rho * grid.dx * grid.dy * grid.dz
    rate = 18 * P.air_dynamic_viscosity / (C.water_density * diameter**2)
    decay = rate * (1 + m / meff)
    g = -9.81
    center_acceleration = m * g / (meff + m)
    slip = g / decay * (-np.expm1(-decay * total_time))
    exact_v = center_acceleration * total_time + meff / (meff + m) * slip
    exact_u = center_acceleration * total_time - m / (meff + m) * slip
    exact_z = (
        0.15
        + 0.5 * center_acceleration * total_time**2
        + meff
        / (meff + m)
        * g
        / decay
        * (total_time + np.expm1(-decay * total_time) / decay)
    )
    assert C.dry_air_density * abs(g / decay) * diameter / P.air_dynamic_viscosity < 0.1
    before = budget(grid, gas, u, e, liquid)
    position_errors, velocity_errors = [], []
    for count in (32, 64, 128, 256, 512, 1024):
        r = integrate(jnp.asarray(count))
        gas2, u2, e2, q2, pos2, ok, impulse, work = r
        assert bool(ok), count
        np.testing.assert_array_equal(q2.mass, liquid.mass)
        np.testing.assert_allclose(
            impulse, [0.0, 0.0, m * g * total_time], rtol=3e-13, atol=1e-20
        )
        position_errors.append(abs(float(pos2[2, 0]) - exact_z))
        velocity_errors.append(abs(float(q2.velocity[2, 0]) - exact_v))
        gas_velocity = float((u2.z[1, 1, 1] + u2.z[2, 1, 1]) / 2)
        assert abs(gas_velocity - exact_u) < 1e-4
        expected = list(before)
        expected[2] = expected[2] + np.asarray(impulse)
        expected[3] = expected[3] + float(work)
        for a, b in zip(expected, budget(grid, gas2, u2, e2, q2)):
            np.testing.assert_allclose(a, b, rtol=2e-12, atol=2e-12)
    position_ratios = np.array(position_errors[:-1]) / position_errors[1:]
    velocity_ratios = np.array(velocity_errors[:-1]) / velocity_errors[1:]
    # Keep and report the coarsest errors: at lambda*dt ~= 0.63 the O(dt)
    # position lag and O(dt^2) velocity error partially cancel. Test the same
    # order band after refining into the asymptotic regime, not by loosening it.
    assert np.all(np.diff(position_errors) < 0), position_errors
    fine_position_ratios = position_ratios[-3:]
    assert np.all((fine_position_ratios > 1.7) & (fine_position_ratios < 2.3)), (
        position_errors,
        position_ratios,
    )
    assert np.all((velocity_ratios > 3.8) & (velocity_ratios < 4.2)), (
        velocity_errors,
        velocity_ratios,
    )
    print(
        "analytic_settling_z_v_u",
        exact_z,
        exact_v,
        exact_u,
        "position_errors",
        position_errors,
        "position_ratios",
        position_ratios,
        "velocity_errors",
        velocity_errors,
        "velocity_ratios",
        velocity_ratios,
    )

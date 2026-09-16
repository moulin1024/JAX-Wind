"""Conservation and coupled wake regressions for shared cloud/spray physics."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxwind.physics import cryogenic
from jaxwind.physics.moisture import (
    MoistureConfig,
    MoistureState,
    advance_moisture,
    saturation_adjustment,
    saturation_mixing_ratio,
    virtual_temperature_offset,
)

jax.config.update("jax_enable_x64", True)
CONFIG = MoistureConfig()


def water(vapor=0.005, spray=0.001, diameter=50.0e-6, liquid=0.0, ice=0.0):
    n = spray / (CONFIG.water_density * np.pi * diameter**3 / 6)
    return MoistureState(
        *(jnp.asarray(v, jnp.float64) for v in (vapor, liquid, ice, spray, n))
    )


def enthalpy(t, w):
    return (
        CONFIG.dry_air_heat_capacity * t
        + CONFIG.water_vapor_latent_heat * w.vapor
        - (CONFIG.ice_sublimation_latent_heat - CONFIG.water_vapor_latent_heat)
        * w.cloud_ice
    )


def test_nitrogen_uses_same_water_thermodynamics():
    assert cryogenic.saturation_adjustment is saturation_adjustment


@pytest.mark.parametrize("dt", [0.001, 0.1, 1000.0])
def test_evaporation_conserves_water_and_enthalpy_and_cannot_oversaturate(dt):
    initial = water()
    t, result = jax.jit(lambda t, w: advance_moisture(t, w, dt, CONFIG))(
        jnp.asarray(300.0), initial
    )
    assert t < 300.0
    assert 0 <= result.spray_liquid < initial.spray_liquid
    assert result.vapor > initial.vapor
    assert result.vapor <= saturation_mixing_ratio(t, CONFIG.pressure, CONFIG) + 2.0e-8
    np.testing.assert_allclose(sum(result[:4]), sum(initial[:4]), atol=1.0e-12)
    np.testing.assert_allclose(
        enthalpy(t, result), enthalpy(300.0, initial), atol=0.06, rtol=0
    )
    assert all(np.isfinite(v) and v >= 0 for v in result)


def test_saturated_air_suppresses_evaporation_and_size_controls_rate():
    temperature = jnp.asarray(300.0)
    saturated = water(
        vapor=saturation_mixing_ratio(temperature, CONFIG.pressure, CONFIG)
    )
    t, result = advance_moisture(temperature, saturated, 1.0, CONFIG)
    np.testing.assert_allclose(result.spray_liquid, saturated.spray_liquid, atol=2.0e-8)
    _, fine = advance_moisture(temperature, water(diameter=20.0e-6), 0.01, CONFIG)
    _, coarse = advance_moisture(temperature, water(diameter=100.0e-6), 0.01, CONFIG)
    assert fine.spray_liquid < coarse.spray_liquid


def test_cloud_adjustment_keeps_spray_distinct_and_conserves_ice_enthalpy():
    initial = water(vapor=0.02, spray=0.001, liquid=0.002, ice=0.001)
    t, result = advance_moisture(jnp.asarray(270.0), initial, 0.1, CONFIG)
    np.testing.assert_allclose(sum(result[:4]), sum(initial[:4]), atol=1.0e-12)
    np.testing.assert_allclose(
        enthalpy(t, result), enthalpy(270.0, initial), atol=0.06, rtol=0
    )
    assert result.cloud_liquid + result.cloud_ice > 0
    assert result.spray_liquid > 0


def test_no_spray_and_zero_dt_do_not_create_water():
    initial = water(spray=0)
    t, result = advance_moisture(jnp.asarray(300.0), initial, 10.0, CONFIG)
    np.testing.assert_allclose(result, initial, atol=1.0e-12)
    initial = water()
    _, result = advance_moisture(jnp.asarray(300.0), initial, 0.0, CONFIG)
    assert result.spray_liquid == initial.spray_liquid


def test_moist_buoyancy_includes_vapor_and_liquid_loading():
    vapor = water(vapor=0.01, spray=0)
    assert virtual_temperature_offset(vapor, 300.0, 0.005, CONFIG) > 0
    loaded = vapor._replace(spray_liquid=jnp.asarray(0.01))
    assert virtual_temperature_offset(loaded, 300.0, 0.005, CONFIG) < 0
    t, evaporated = advance_moisture(jnp.asarray(300.0), water(), 1.0, CONFIG)
    assert t - 300 + virtual_temperature_offset(evaporated, 300.0, 0.005, CONFIG) < 0


def test_source_mass_number_and_downstream_support():
    from jaxwind.domain import UniformGrid
    from jaxwind.water_spray import build_water_injection

    grid = UniformGrid(12, 6, 4, 6.0, 3.0, 2.0)
    source = build_water_injection(
        grid,
        (2.0, 1.5, 1.0),
        (0.5, 0.2, 0.2),
        0.01,
        50.0e-6,
        1.0,
        CONFIG,
        dtype="float64",
    )
    mass, number = source(jnp.asarray(1.0))
    np.testing.assert_allclose(
        jnp.sum(mass * grid.cell_volumes) * CONFIG.dry_air_density, 0.01
    )
    np.testing.assert_allclose(
        number, mass / (CONFIG.water_density * np.pi * (50.0e-6) ** 3 / 6)
    )
    assert np.all(np.asarray(mass)[..., :4] == 0)
    assert np.all(np.asarray(source(jnp.asarray(0.0))[0]) == 0)


def test_transport_conserves_mass_with_closed_boundaries_and_subcycles():
    from jaxwind.domain import UniformGrid
    from jaxwind.moist_abl import transport_moisture
    from jaxwind.state import StaggeredVelocity

    grid = UniformGrid(8, 6, 4, 4.0, 3.0, 2.0)
    shape = (grid.nz, grid.ny, grid.nx)
    q = jnp.zeros(shape).at[1, 2, 3].set(0.001)
    zero = jnp.zeros(shape)
    w = MoistureState(zero, zero, zero, q, q * 1.0e9)
    velocity = StaggeredVelocity(
        jnp.zeros((4, 6, 9)), jnp.ones(shape), jnp.zeros((5, 6, 8))
    )
    final = jax.jit(lambda w: transport_moisture(w, velocity, grid, 3.0, 0.0, 0.01))(w)
    assert jnp.min(final.spray_liquid) >= 0
    np.testing.assert_allclose(jnp.sum(final.spray_liquid), jnp.sum(q), atol=1.0e-15)
    np.testing.assert_allclose(final.spray_number, final.spray_liquid * 1.0e9)


def test_coupled_step_cools_wake_and_checkpoint_resume_matches(tmp_path):
    from jaxwind import (
        FREE_SLIP,
        OPEN,
        Boundaries,
        FlowModel,
        InflowPlane,
        LinearBoussinesqBuoyancy,
        PassiveScalar,
        StaggeredVelocity,
        Wall,
        build_open_atmospheric_run,
        build_open_atmospheric_step,
        build_pressure_poisson,
        initial_atmospheric_solution,
    )
    from jaxwind.domain import UniformGrid
    from jaxwind.io.checkpoint import load_checkpoint, save_checkpoint
    from jaxwind.moist_abl import build_moist_atmospheric_step, initialize_moisture
    from jaxwind.water_spray import build_water_injection

    grid = UniformGrid(8, 6, 4, 4.0, 3.0, 2.0)
    shape = (4, 6, 8)
    velocity = StaggeredVelocity(
        jnp.full((4, 6, 9), 1.0), jnp.zeros(shape), jnp.zeros((5, 6, 8))
    )
    plane = InflowPlane(
        velocity.x[..., 0], velocity.y[..., 0], velocity.z[..., 0], jnp.zeros((4, 6))
    )
    flow = initial_atmospheric_solution(grid, velocity, dtype="float64")
    initial, ambient = initialize_moisture(flow, 300.0, 0.4, CONFIG)
    boundaries = Boundaries(Wall(FREE_SLIP), Wall(FREE_SLIP), streamwise=OPEN)
    momentum, scalar = FlowModel(), PassiveScalar(advection_scheme="upwind")
    dry_step = build_open_atmospheric_step(
        grid,
        boundaries,
        build_pressure_poisson(grid, backend="gmg", periodic_x=False, dtype="float64"),
        momentum,
        scalar,
        LinearBoussinesqBuoyancy(9.81 / 300.0),
        scheme="fast-rk3",
    )
    source = build_water_injection(
        grid,
        (1.5, 1.5, 1.0),
        (0.3, 0.3, 0.3),
        0.001,
        50.0e-6,
        0.0,
        CONFIG,
        dtype="float64",
    )
    step = jax.jit(
        build_moist_atmospheric_step(
            dry_step,
            grid,
            boundaries,
            momentum,
            scalar,
            CONFIG,
            300.0,
            300.0,
            ambient,
            source,
        )
    )
    final = step(initial, 0.02, plane)
    assert jnp.min(final.scalar) < 0
    assert jnp.max(final.moisture.vapor - initial.moisture.vapor) > 0
    assert jnp.min(final.velocity.z) < 0
    assert jnp.all(jnp.isfinite(final.scalar))
    path = tmp_path / "moist.npz"
    save_checkpoint(path, final, metadata={"fingerprint": "test"})
    restored, _, _ = load_checkpoint(path, initial, fingerprint="test")
    continued = step(restored, 0.02, plane)
    expected = step(final, 0.02, plane)
    for a, b in zip(jax.tree.leaves(continued), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(a, b)
    planes = jax.tree.map(lambda a: jnp.stack([a, a]), plane)
    scanned = build_open_atmospheric_run(step)(initial, 0.02, planes)
    for a, b in zip(jax.tree.leaves(scanned), jax.tree.leaves(expected)):
        np.testing.assert_allclose(a, b, atol=1.0e-12)


def test_workflow_builder_connects_water_to_turbine_and_frame_output():
    from copy import deepcopy
    from pathlib import Path

    from jaxwind import InflowPlane, StaggeredVelocity, initial_atmospheric_solution
    from jaxwind.config.document import ResolvedCase, load_case
    from jaxwind.config.stages import load_workflow
    from jaxwind.runtime.frames import capture_frame
    from jaxwind.simulation.open_atmospheric import build_open_components

    path = (
        Path(__file__).resolve().parents[2]
        / "cases/HITSZWindTunnel/fv_far_wake_water.toml"
    )
    base = load_case(path)
    doc = deepcopy(base.document)
    doc["mesh"]["cells"] = [16, 8, 8]
    doc["workflow"]["record_plane"] = 2
    doc["physics"]["water_spray"]["ramp_time_s"] = 0.0
    workflow = load_workflow(ResolvedCase(base.source, doc))
    grid = workflow.case.physical.physical_grid
    shape = (grid.nz, grid.ny, grid.nx)
    velocity = StaggeredVelocity(
        jnp.full(shape, 3.0, jnp.float32),
        jnp.zeros(shape, jnp.float32),
        jnp.zeros((grid.nz + 1, grid.ny, grid.nx), jnp.float32),
    )
    warm = initial_atmospheric_solution(grid, velocity, dtype="float32")
    first = InflowPlane(*(a[..., 0] for a in (*velocity, warm.scalar)))
    initial, step = build_open_components(workflow, warm, first, return_step=True)
    final = jax.jit(step)(initial, 0.001, first)
    assert jnp.min(final.scalar) < 0
    assert jnp.sum(final.moisture.spray_liquid) > 0
    frame = capture_frame(final, grid, y_m=3.0, z_m=0.876)
    assert frame["spray_liquid_center_zx"].shape == (grid.nz, grid.nx)
    assert "vapor_hub_yx" in frame

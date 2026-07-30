from __future__ import annotations

import numpy as np
import pytest


def test_murphy_koop_saturation_pressures_match_reference_values() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax.cryogenic_microphysics import (
        saturation_vapor_pressure_ice,
        saturation_vapor_pressure_water,
    )

    assert float(saturation_vapor_pressure_water(jnp.asarray(300.0))) == pytest.approx(
        3536.8, rel=2.0e-3
    )
    assert float(saturation_vapor_pressure_ice(jnp.asarray(250.0))) == pytest.approx(
        76.0, rel=3.0e-3
    )


def test_saturation_adjustment_conserves_total_water_and_moist_enthalpy() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax.cryogenic_microphysics import (
        CryogenicMicrophysicsConfig,
        saturation_adjustment,
    )

    config = CryogenicMicrophysicsConfig(saturation_iterations=10)
    temperature = jnp.asarray([280.0, 260.0, 290.0])
    qv = jnp.asarray([0.020, 0.004, 0.002])
    ql = jnp.asarray([0.0, 0.0, 0.006])
    qi = jnp.asarray([0.0, 0.0, 0.002])
    initial_water = qv + ql + qi
    initial_enthalpy = (
        config.dry_air_heat_capacity * temperature
        + config.water_vapor_latent_heat * qv
        - config.water_fusion_latent_heat * qi
    )

    adjusted = saturation_adjustment(temperature, qv, ql, qi, config)
    final_temperature, final_qv, final_ql, final_qi = adjusted
    final_water = final_qv + final_ql + final_qi
    final_enthalpy = (
        config.dry_air_heat_capacity * final_temperature
        + config.water_vapor_latent_heat * final_qv
        - config.water_fusion_latent_heat * final_qi
    )

    np.testing.assert_allclose(final_water, initial_water, rtol=2.0e-6)
    np.testing.assert_allclose(final_enthalpy, initial_enthalpy, rtol=3.0e-4)
    assert np.all(np.asarray(final_qv) >= 0.0)
    assert np.all(np.asarray(final_ql) >= 0.0)
    assert np.all(np.asarray(final_qi) >= 0.0)


def test_finite_rate_fog_microphysics_conserves_water_and_enthalpy() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax.cryogenic_microphysics import (
        CryogenicMicrophysicsConfig,
        advance_fog_microphysics,
    )

    config = CryogenicMicrophysicsConfig(
        saturation_relaxation_timescale=0.02,
        freezing_timescale=0.03,
    )
    temperature = jnp.asarray([265.0, 280.0])
    qv = jnp.asarray([0.006, 0.003])
    ql = jnp.asarray([0.002, 0.001])
    qi = jnp.asarray([0.001, 0.002])
    water_before = qv + ql + qi
    enthalpy_before = (
        config.dry_air_heat_capacity * temperature
        + config.water_vapor_latent_heat * qv
        - config.water_fusion_latent_heat * qi
    )

    update = advance_fog_microphysics(
        temperature, qv, ql, qi, 0.005, config
    )
    water_after = update.qv + update.ql + update.qi
    enthalpy_after = (
        config.dry_air_heat_capacity * update.temperature
        + config.water_vapor_latent_heat * update.qv
        - config.water_fusion_latent_heat * update.qi
    )

    np.testing.assert_allclose(water_after, water_before, rtol=2.0e-6)
    np.testing.assert_allclose(
        enthalpy_after, enthalpy_before, rtol=3.0e-6
    )
    assert float(update.frozen[0]) > 0.0
    assert float(update.melted[1]) > 0.0


def test_resolved_fog_classes_have_physical_stokes_settling_speeds() -> None:
    from wireles_jax.cryogenic_microphysics import (
        CryogenicMicrophysicsConfig,
        stokes_terminal_velocity,
    )

    config = CryogenicMicrophysicsConfig()
    liquid = stokes_terminal_velocity(
        config.liquid_fog_diameter,
        config.water_density,
        config,
    )
    ice = stokes_terminal_velocity(
        config.ice_fog_diameter,
        config.ice_density,
        config,
    )

    assert 0.0 < liquid < 0.1
    assert liquid < ice < 0.1


def test_nitrogen_droplet_evaporation_conserves_supplied_heat() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax.cryogenic_microphysics import (
        CryogenicMicrophysicsConfig,
        advance_nitrogen_droplet,
    )

    config = CryogenicMicrophysicsConfig()
    diameter = jnp.asarray(150.0e-6)
    mass = (
        np.pi
        / 6.0
        * config.liquid_nitrogen_density
        * float(diameter) ** 3
    )
    update = advance_nitrogen_droplet(
        jnp.asarray(mass),
        diameter,
        jnp.asarray(config.nitrogen_boiling_temperature),
        jnp.asarray(300.0),
        jnp.asarray(20.0),
        2.0e-3,
        config,
    )

    assert float(update.evaporated_mass) > 0.0
    assert float(update.mass + update.evaporated_mass) == pytest.approx(
        mass, rel=2.0e-6
    )
    assert float(update.gas_energy_loss) == pytest.approx(
        float(update.evaporated_mass) * config.liquid_nitrogen_latent_heat,
        rel=2.0e-6,
    )


def test_outlet_sink_exactly_balances_evaporation_volume() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax.cryogenic_microphysics import (
        CryogenicMicrophysicsConfig,
        balanced_volume_divergence,
        smooth_outlet_window,
    )

    config = CryogenicMicrophysicsConfig(
        outlet_start_x=0.75,
        outlet_end_x=1.0,
    )
    x = (jnp.arange(8, dtype=jnp.float32) + 0.5) / 8.0
    outlet = smooth_outlet_window(x, 0.75, 1.0)[:, None, None]
    source = jnp.zeros((8, 2, 2), dtype=jnp.float32)
    source = source.at[1, 0, 0].set(0.02)
    target, sink = balanced_volume_divergence(
        source,
        jnp.full_like(source, 300.0),
        jnp.broadcast_to(outlet, source.shape),
        1.0 / source.size,
        config,
    )

    assert float(jnp.sum(sink)) > 0.0
    assert float(jnp.sum(target)) == pytest.approx(0.0, abs=2.0e-9)


def test_mass_only_outlet_does_not_define_momentum_tendencies() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax.cryogenic_microphysics import (
        CryogenicMicrophysicsConfig,
        mass_only_outlet_update,
    )

    config = CryogenicMicrophysicsConfig(
        outlet_start_x=1.0,
        outlet_end_x=3.0,
        outlet_scalar_timescale=2.0,
    )
    nitrogen = jnp.asarray([0.0, 0.01, 0.02])
    update = mass_only_outlet_update(
        evaporation_mass_rate=jnp.asarray([0.02, 0.0, 0.0]),
        gas_temperature=jnp.full((3,), 300.0),
        nitrogen_mass_fraction=nitrogen,
        x_coordinates=jnp.asarray([0.0, 2.0, 3.0]),
        cell_volume=1.0,
        config=config,
    )

    np.testing.assert_allclose(
        update.nitrogen_tendency,
        np.asarray([0.0, -0.0025, -0.01]),
        rtol=1.0e-6,
    )
    assert set(update._fields) == {
        "target_divergence",
        "nitrogen_tendency",
        "volume_sink",
    }
    assert float(jnp.sum(update.target_divergence)) == pytest.approx(
        0.0, abs=2.0e-9
    )


def test_sharded_prescribed_mass_outlet_is_globally_compatible() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import Params
    from wireles_jax.cryogenic_microphysics import CryogenicMicrophysicsConfig
    from wireles_jax.cryogenic_sharded import (
        make_prescribed_ln2_mass_outlet_sharded,
    )
    from wireles_jax.sharding import make_single_node_mesh, put_z_slab

    params = Params(
        nx=16,
        ny=8,
        nz=8,
        lx=8.0,
        ly=4.0,
        lz=2.0,
        z_i=1.0,
        dt=1.0e-3,
        momentum_wall_model="free_slip",
        dtype=jnp.float32,
    )
    mesh = make_single_node_mesh(1)
    closure = make_prescribed_ln2_mass_outlet_sharded(
        params,
        mesh,
        mass_flow_rate=0.02,
        source_x=1.0,
        source_y=2.0,
        source_z=1.0,
        source_sigma_x=0.15,
        source_sigma_r=0.15,
        config=CryogenicMicrophysicsConfig(
            outlet_start_x=7.0,
            outlet_end_x=8.0,
        ),
    )
    temperature = put_z_slab(
        jnp.full((params.nx, params.ny, params.nz), 300.0),
        mesh,
    )
    target = jax.block_until_ready(closure(temperature))

    assert float(jnp.max(target)) > 0.0
    assert float(jnp.min(target)) < 0.0
    assert float(jnp.sum(target)) == pytest.approx(0.0, abs=2.0e-7)


def test_one_step_sharded_ln2_droplet_routes_mass_away_from_qv() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import Params, SprayDPMConfig
    from wireles_jax.cryogenic_microphysics import CryogenicMicrophysicsConfig
    from wireles_jax.cryogenic_sharded import (
        ShardedCryogenicState,
        initial_cryogenic_scalar_state,
        make_step_cryogenic_sharded,
    )
    from wireles_jax.sharding import make_single_node_mesh
    from wireles_jax.spray_dpm_sharded import initialize_sharded_spray
    from wireles_jax.timestep_sharded import (
        initial_sharded_state,
        make_sharded_operators,
    )

    params = Params(
        nx=8,
        ny=4,
        nz=8,
        lx=8.0,
        ly=4.0,
        lz=2.0,
        z_i=1.0,
        dt=1.0e-3,
        initial_condition="geostrophic",
        geostrophic_u=0.0,
        momentum_wall_model="free_slip",
        initial_velocity_noise=0.0,
        thermo_enabled=True,
        moisture_enabled=True,
        theta0=300.0,
        qv0=0.011,
        scalar_sgs_model="fixed_prandtl",
        scalar_vertical_scheme="centered",
        sgs_model="smagorinsky",
        pressure_filter_nyquist=True,
        dtype=jnp.float32,
    )
    spray_config = SprayDPMConfig(
        material="nitrogen",
        max_parcels=8,
        parcels_per_step=1,
        mass_flow_rate=0.02,
        injection_end_time=1.0,
        injection_x=1.0,
        injection_y=2.0,
        injection_z=1.0,
        injection_radius=0.15,
        injection_streamwise_thickness=0.2,
        injection_ramp_time=0.001,
        injection_u=8.0,
        initial_diameter=150.0e-6,
        minimum_diameter=50.0e-6,
        maximum_diameter=300.0e-6,
        initial_temperature=77.34,
        boiling_temperature=77.34,
        liquid_density=806.11,
        water_density=806.11,
        liquid_heat_capacity=2040.0,
        latent_heat=199_180.0,
        surface_tension=8.85e-3,
        substeps=2,
    )
    microphysics = CryogenicMicrophysicsConfig(
        outlet_start_x=7.0,
        outlet_end_x=8.0,
    )
    mesh = make_single_node_mesh(1)
    operators = make_sharded_operators(params, mesh)
    flow = initial_sharded_state(params, mesh)
    state = ShardedCryogenicState(
        flow=flow,
        spray=initialize_sharded_spray(
            spray_config, params, mesh
        ),
        scalars=initial_cryogenic_scalar_state(flow),
    )
    step = jax.jit(
        make_step_cryogenic_sharded(
            spray_config,
            microphysics,
            params,
            operators,
            mesh,
        )
    )
    initial_qv = jnp.sum(state.flow.qv)
    advanced, diagnostics = jax.block_until_ready(
        step(state, operators.pressure, operators.pressure_spike)
    )

    assert int(advanced.flow.step) == 1
    assert float(diagnostics.spray.evaporated_mass) > 0.0
    assert float(diagnostics.nitrogen_gas_mass) > 0.0
    assert float(diagnostics.outlet_volume_rate) > 0.0
    assert float(diagnostics.nitrogen_sensible_cooling) > 0.0
    assert np.all(np.isfinite(np.asarray(advanced.flow.theta)))
    # Nitrogen phase mass must not be deposited into the water-vapour scalar.
    assert float(jnp.sum(advanced.flow.qv)) == pytest.approx(
        float(initial_qv), rel=2.0e-5
    )
    assert float(jnp.sum(advanced.scalars.yn2)) > 0.0
    reconstructed_enthalpy = (
        (
            microphysics.dry_air_heat_capacity
            + advanced.scalars.yn2
            * microphysics.nitrogen_gas_heat_capacity
        )
        * advanced.flow.theta
        + microphysics.water_vapor_latent_heat * advanced.flow.qv
        - microphysics.water_fusion_latent_heat * advanced.scalars.qi
    )
    np.testing.assert_allclose(
        np.asarray(advanced.scalars.enthalpy),
        np.asarray(reconstructed_enthalpy),
        rtol=3.0e-5,
        atol=2.0,
    )

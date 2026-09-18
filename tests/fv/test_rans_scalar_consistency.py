"""Coupled shared-scalar transport must preserve source-free moist enthalpy."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from jaxwind.config.document import load_case
from jaxwind.config.moisture import load_moisture
from jaxwind.simulation.api import RunControls
from jaxwind.simulation.water_spray_benchmark import build_simulation


@pytest.mark.parametrize(
    "model, scheme",
    [
        ("standard-k-epsilon", "upwind-ssprk3"),
        ("realizable-k-epsilon", "muscl-mc"),
    ],
)
def test_shared_rans_scalar_transport_preserves_constant_moist_enthalpy(model, scheme):
    root = Path(__file__).resolve().parents[2]
    case = load_case(
        root
        / "cases/WaterSprayMontazeri2015/centre_investigation/standard_kepsilon_case3.toml"
    )
    doc = deepcopy(case.document)
    doc["mesh"]["cells"] = [8, 8, 8]
    doc["case"]["parcel_capacity"] = 32
    doc["case"]["parcels_per_step"] = 4
    doc["case"]["parcel_substeps"] = 1
    doc["physics"]["water_spray"]["mass_flow_rate_kg_s"] = 0.0
    doc["case"]["carrier_turbulence_model"] = model
    doc["numerics"]["scalar_advection_scheme"] = scheme
    if scheme == "muscl-mc":
        doc["numerics"]["turbulence_advection_scheme"] = "muscl-mc"
    simulation = build_simulation(replace(case, document=doc))
    grid = simulation.grid
    moisture, _ = load_moisture(doc["physics"])
    thermo = moisture.thermodynamics
    cp, latent = thermo.dry_air_heat_capacity, thermo.water_vapor_latent_heat
    initial = simulation.initial_state
    ambient = initial.moisture.vapor
    x = jnp.asarray(grid.x_centers)[None, None, :]
    y = jnp.asarray(grid.y_centers)[None, :, None]
    # Inlet and outlet reservoirs have zero anomaly and the same enthalpy.
    # This mild humidity variation remains far below saturation throughout.
    perturbation = jnp.broadcast_to(
        1e-4
        * jnp.sin(jnp.pi * x / grid.lx) ** 2
        * (1 + 0.2 * jnp.cos(2 * jnp.pi * y / grid.ly)),
        initial.scalar.shape,
    )
    initial = initial._replace(
        scalar=-latent / cp * perturbation,
        moisture=initial.moisture._replace(vapor=ambient + perturbation),
    )
    result = simulation.advance(initial, RunControls(count=12, target_time=0.003))
    defect = cp * result.scalar + latent * (result.moisture.vapor - ambient)
    np.testing.assert_allclose(defect, 0.0, atol=1e-6)
    assert (
        float(jnp.max(jnp.abs(result.moisture.vapor - initial.moisture.vapor))) > 1e-8
    )
    assert float(result.parcels.injected_mass) == 0.0
    assert not bool(jnp.any(result.parcels.active))
    assert bool(jnp.all(jnp.stack(result.turbulence) > 0))
    assert bool(jnp.all(jnp.isfinite(jnp.stack(result.turbulence))))
    assert float(jnp.max(jnp.abs(result.velocity.x - initial.velocity.x))) > 1e-8
    assert int(result.step) == 12


def test_moist_wrapper_separates_heat_and_vapor_diffusion():
    """Check actual wrapper wiring against two independent periodic heat modes."""
    import jax
    from types import SimpleNamespace
    from jaxwind import Boundaries, FREE_SLIP, Wall
    from jaxwind.domain import UniformGrid
    from jaxwind.moist_abl import MoistAtmosphericSolution, build_moist_atmospheric_step
    from jaxwind.physics.moisture import MoistureConfig, MoistureState
    from jaxwind.scalar import PassiveScalar
    from jaxwind.sgs import ConstantEddyViscosity
    from jaxwind.state import StaggeredVelocity

    jax.config.update("jax_enable_x64", True)
    grid = UniformGrid(64, 2, 2, 1., 1., 1.)
    zero = jnp.zeros((2, 2, 64))
    velocity = StaggeredVelocity(zero, zero, jnp.zeros((3, 2, 64)))
    mode = jnp.broadcast_to(jnp.cos(2*jnp.pi*jnp.asarray(grid.x_centers)), zero.shape)
    vapor = .0035 + 1e-5*mode
    state = MoistAtmosphericSolution(velocity, zero, velocity, .1*mode, zero,
                                    jnp.asarray(0.), jnp.asarray(0),
                                    MoistureState(vapor, zero, zero, zero, zero))
    cfg = MoistureConfig()
    scalar = PassiveScalar(diffusivity=2e-5, turbulent_prandtl=.85)
    ambient = jnp.full((2, 2), .0035)
    step = build_moist_atmospheric_step(
        lambda flow, dt, inflow, offset: flow, grid,
        Boundaries(Wall(FREE_SLIP), Wall(FREE_SLIP)),
        SimpleNamespace(subfilter=ConstantEddyViscosity(.01)), scalar, cfg,
        300., 300., ambient, scalar_transport_scheme='muscl-mc',
        vapor_turbulent_schmidt=.7)
    result = jax.jit(lambda s: step(s, .1, SimpleNamespace(scalar=jnp.zeros((2,2)))))(state)
    heat_exact = .1*mode*jnp.exp(-(2e-5+.01/.85)*(2*jnp.pi)**2*.1)
    vapor_exact = .0035+1e-5*mode*jnp.exp(-(cfg.vapor_diffusivity+.01/.7)*(2*jnp.pi)**2*.1)
    np.testing.assert_allclose(result.scalar, heat_exact, atol=4e-6, rtol=0)
    np.testing.assert_allclose(result.moisture.vapor, vapor_exact, atol=5e-10, rtol=0)
    delta_h = cfg.dry_air_heat_capacity*(result.scalar-state.scalar)+cfg.water_vapor_latent_heat*(result.moisture.vapor-vapor)
    assert abs(float(jnp.sum(delta_h))) < 1e-7

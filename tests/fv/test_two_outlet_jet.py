"""Pressure-driven two-outlet flow must release mass through both x faces."""
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxwind import StaggeredVelocity, build_pressure_poisson
from jaxwind.domain import UniformGrid, AnalyticalGrid, TanhMapping
from jaxwind.low_mach import continuity_residual, project_low_mach
from jaxwind.open_boundary import enforce_two_outlet_scalar, enforce_two_outlet_velocity
from jaxwind.config.jet import load_case

ROOT = Path(__file__).resolve().parents[2]
CASE = ROOT / "cases/HITSZWindTunnel/fv_1024x256x512_l24_jet_only_two_outlets_1s.toml"


@pytest.mark.parametrize("mapped", [False, True])
def test_source_leaves_both_ends_and_projection_closes_mass(mapped):
    grid = (AnalyticalGrid(16, 4, 8, 2.0, 1.0, 1.0,
                          TanhMapping(1.0), TanhMapping(0.0), TanhMapping(0.0))
            if mapped else UniformGrid(16, 4, 8, 2.0, 1.0, 1.0))
    shape = (grid.nz, grid.ny, grid.nx)
    rho = jnp.ones(shape, dtype=jnp.float32)
    velocity = StaggeredVelocity(jnp.zeros((grid.nz, grid.ny, grid.nx + 1)),
                                 jnp.zeros((grid.nz, grid.ny + 1, grid.nx)),
                                 jnp.zeros((grid.nz + 1, grid.ny, grid.nx)))
    poisson = build_pressure_poisson(grid, backend="gmg", periodic_x=False,
                                     periodic_y=False, open_x_low=True,
                                     dtype="float32", config={"tolerance": 1e-6})
    velocity, pressure = project_low_mach(velocity, rho, rho, poisson, 0.1, mass_source=0.2)
    residual = continuity_residual(velocity, rho, rho, grid, 0.1, 0.2)
    assert float(jnp.max(jnp.abs(residual))) < 2e-5
    assert np.all(np.asarray(velocity.x[..., 0]) < 0)
    assert np.all(np.asarray(velocity.x[..., -1]) > 0)
    np.testing.assert_allclose(velocity.x[..., 0], -velocity.x[..., -1], atol=2e-5)
    area = np.asarray(grid.z_widths)[:, None] * np.asarray(grid.y_widths)[None, :]
    flux = np.sum(area * np.asarray(velocity.x[..., -1] - velocity.x[..., 0]))
    np.testing.assert_allclose(flux, 0.2 * grid.lx * grid.ly * grid.lz, rtol=2e-5)
    assert float(poisson.residual_norm(pressure, -jnp.ones(shape) * 2.0)) < 1e-3


def test_backflow_uses_ambient_without_prescribing_velocity():
    shape = (4, 4, 8)
    velocity = StaggeredVelocity(jnp.ones((4, 4, 9)), jnp.zeros((4, 5, 8)),
                                 jnp.zeros((5, 4, 8)))
    result = enforce_two_outlet_velocity(velocity)
    np.testing.assert_allclose(result.x, 1.0)
    field = jnp.full(shape, 200.0)
    scalar = enforce_two_outlet_scalar(field, result, 300.0)
    np.testing.assert_allclose(scalar[..., 0], 300.0)
    np.testing.assert_allclose(scalar[..., -1], 200.0)
    reverse = result._replace(x=-result.x)
    scalar = enforce_two_outlet_scalar(field, reverse, 300.0)
    np.testing.assert_allclose(scalar[..., 0], 200.0)
    np.testing.assert_allclose(scalar[..., -1], 300.0)


def test_requested_case_and_small_jet_advance():
    from jaxwind.simulation.jet import build_simulation
    case = load_case(CASE)
    assert case.cells == (1024, 256, 512)
    assert case.lengths == (24.0, 6.0, 3.6)
    assert case.flow_formulation == "low-mach"
    assert case.source_mode == "volume"
    assert case.streamwise_boundaries == "outflow-outflow"
    assert case.steps * case.dt == 1.0
    small = replace(case, cells=(16, 8, 8), dt=0.00025, steps=2)
    grid, jet, microphysics, initial, advance, courant = build_simulation(small)
    assert float(jnp.max(jnp.abs(initial.velocity.x))) == 0.0
    result = advance(initial, 2)
    assert np.all(np.isfinite(np.asarray(result.temperature)))
    assert int(result.step) == 2
    assert float(jnp.sum(result.nitrogen_density)) > 0.0
    assert float(result.continuity_error) < 1e-3

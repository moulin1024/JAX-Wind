"""Accuracy, conservation, limiting, and nonperiodic-stencil checks."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxwind import UniformGrid, StaggeredVelocity, FlowModel, Boundaries
from jaxwind.numerics.momentum import _mc_slope, _muscl_flux, muscl_advection
from jaxwind.numerics.discretization import advection
from jaxwind.numerics.integrate import build_tendency


@pytest.fixture(autouse=True)
def precision():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("speed", [-1., 1.])
def test_mc_linear_reconstruction_and_no_opposite_edge_coupling(axis, speed):
    shape = [1, 1, 1]
    shape[axis] = 12
    q = jnp.arange(12., dtype=jnp.float64).reshape(shape)
    mass = jnp.full(tuple(13 if a == axis else 1 for a in range(3)), speed)
    flux = _muscl_flux(q, mass, axis, False).reshape(-1)
    np.testing.assert_allclose(np.asarray(flux)[2:-2] / speed, np.arange(1.5, 10.))
    # Perturb either edge: neither reconstruction nor slopes may wrap.
    changed = q.at[tuple(-1 if a == axis else 0 for a in range(3))].set(1.e6)
    result = _muscl_flux(changed, mass, axis, False).reshape(-1)
    np.testing.assert_array_equal(result[:3], flux[:3])
    changed = q.at[tuple(0 for _ in range(3))].set(-1.e6)
    result = _muscl_flux(changed, mass, axis, False).reshape(-1)
    np.testing.assert_array_equal(result[-3:], flux[-3:])


@pytest.mark.parametrize("speed", [-1., 1.])
def test_limiter_no_new_extrema_and_flux_conservation(speed):
    q = jnp.asarray([0., 0., 0., 1., 1., 1., 0., 0.])
    flux = _muscl_flux(q, jnp.full_like(q, speed), 0, True)
    tendency = -(jnp.roll(flux, -1) - flux)
    advanced = q + .25 * tendency
    assert float(advanced.min()) >= 0.
    assert float(advanced.max()) <= 1.
    np.testing.assert_allclose(tendency.sum(), 0., atol=1.e-14)
    np.testing.assert_array_equal(_mc_slope(q, 0, True), 0.)


@pytest.mark.parametrize("periodic", [True, False])
def test_uniform_field_and_shapes(periodic):
    g = UniformGrid(12, 8, 6, 192., 128., 24.)
    v = StaggeredVelocity(jnp.full((6, 8, 12 if periodic else 13), 10.),
                          jnp.zeros((6, 8 if periodic else 9, 12)),
                          jnp.zeros((7, 8, 12)))
    result = jax.jit(lambda u: muscl_advection(u, g))(v)
    for tendency, component in zip(result, v):
        assert tendency.shape == component.shape
        np.testing.assert_allclose(tendency, 0., atol=1.e-14)


@pytest.mark.parametrize("speed", [-1., 1.])
def test_smooth_shear_second_order_and_conservative(speed):
    errors = []
    for n in (32, 64, 128):
        g = UniformGrid(n, 4, 4, 2. * np.pi, 4., 4.)
        x = (np.arange(n) + .5) * g.dx
        q = jnp.broadcast_to(jnp.asarray(np.sin(x)), (4, 4, n))
        v = StaggeredVelocity(jnp.full_like(q, speed), q, jnp.zeros((5, 4, n)))
        tendency = muscl_advection(v, g)
        errors.append(np.mean(abs(np.asarray(tendency.y[0, 0]) + speed * np.cos(x))))
        np.testing.assert_allclose(tendency.y.sum(), 0., atol=1.e-11)
        np.testing.assert_allclose(tendency.x, 0., atol=1.e-12)
    assert min(np.log2(np.array(errors[:-1]) / errors[1:])) > 1.8


def test_model_dispatch_and_default_preservation():
    g = UniformGrid(12, 8, 6, 192., 128., 24.)
    q = jnp.broadcast_to(jnp.sin(jnp.arange(12.)), (6, 8, 12))
    v = StaggeredVelocity(jnp.ones_like(q), q, jnp.zeros((7, 8, 12)))
    default = build_tendency(g, Boundaries(), FlowModel(momentum_advection_scheme="central"))(v, 0.)
    limited = build_tendency(g, Boundaries(), FlowModel())(v, 0.)
    for observed, expected in zip(default, advection(v, g)):
        np.testing.assert_array_equal(observed, expected)
    for observed, expected in zip(limited, muscl_advection(v, g)):
        np.testing.assert_array_equal(observed, expected)
    assert not np.allclose(limited.y, default.y)
    with pytest.raises(ValueError, match="momentum_advection_scheme"):
        build_tendency(g, Boundaries(), FlowModel(momentum_advection_scheme="typo"))


def test_three_dimensional_horizontal_momentum_conservation():
    g = UniformGrid(12, 8, 6, 192., 128., 24.)
    keys = jax.random.split(jax.random.PRNGKey(7), 3)
    v = StaggeredVelocity(
        jax.random.normal(keys[0], (6, 8, 12)),
        jax.random.normal(keys[1], (6, 8, 12)),
        jax.random.normal(keys[2], (7, 8, 12)).at[0].set(0.).at[-1].set(0.),
    )
    result = muscl_advection(v, g)
    for field in (result.x, result.y):
        np.testing.assert_allclose(field.sum(), 0., atol=1.e-12)
    np.testing.assert_array_equal(result.z[0], 0.)
    np.testing.assert_array_equal(result.z[-1], 0.)


def test_mapped_grid_rejected():
    from jaxwind.domain import AnalyticalGrid, TanhMapping
    grid = AnalyticalGrid(12, 8, 6, 192., 128., 24., TanhMapping(1.2))
    with pytest.raises(ValueError, match="uniform grid"):
        build_tendency(grid, Boundaries(), FlowModel(momentum_advection_scheme="muscl-mc"))


def test_case_changes_only_advection(monkeypatch):
    from jaxwind.config.document import load_case
    from jaxwind.config.abl import load_fv_abl
    from jaxwind.simulation.abl import build_models
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("JAXWIND_V80_FAST", str(root / "cases/HornsRev1/turbines/V80/CustomRotor.fst"))
    base = load_case(root / "cases/HornsRev1/fv_v80_single_open_debug.toml")
    new = load_case(root / "cases/HornsRev1/fv_v80_single_open_debug_muscl.toml")
    for section in ("mesh", "time", "physics", "diagnostics"):
        assert base.document[section] == new.document[section]
    assert new.document["numerics"] == {**base.document["numerics"], "momentum_advection_scheme": "muscl-mc"}
    configured = load_fv_abl(new)
    assert configured.options.momentum_advection_scheme == "muscl-mc"
    assert build_models(configured, periodic_x=False)[1].momentum_advection_scheme == "muscl-mc"

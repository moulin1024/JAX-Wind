"""Pressure work, projection, and integration contracts for an open outlet."""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from jaxwind import (
    Boundaries, FlowModel, InflowPlane, OPEN, StaggeredVelocity,
    build_open_atmospheric_step, build_pressure_poisson, divergence,
    initial_atmospheric_solution, pressure_gradient, project,
)
from jaxwind.domain import AnalyticalGrid, TanhMapping, UniformGrid
from jaxwind.open_boundary import backflow_outlet_pressure


@pytest.fixture(autouse=True)
def restore_jax_precision():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def uniform_velocity(grid, speed=1.):
    return StaggeredVelocity(
        jnp.full((grid.nz, grid.ny, grid.nx + 1), speed, dtype=jnp.float64),
        jnp.zeros((grid.nz, grid.ny, grid.nx), dtype=jnp.float64),
        jnp.zeros((grid.nz + 1, grid.ny, grid.nx), dtype=jnp.float64),
    )


def plane_for(velocity):
    return InflowPlane(*(f[..., 0] for f in velocity), jnp.zeros_like(velocity.x[..., 0]))


@pytest.mark.parametrize("mapped", [False, True])
def test_nonzero_outlet_pressure_recovers_manufactured_gradient(mapped):
    grid = (AnalyticalGrid(8, 6, 4, 2., 1.5, 1., TanhMapping(.8), TanhMapping(.6), TanhMapping(.5))
            if mapped else UniformGrid(8, 6, 4, 2., 1.5, 1.))
    solver = build_pressure_poisson(grid, backend="gmg", periodic_x=False, dtype="float64")
    phi = jax.random.normal(jax.random.PRNGKey(78), (grid.nz, grid.ny, grid.nx))
    boundary = jax.random.normal(jax.random.PRNGKey(79), (grid.nz, grid.ny))
    gradient = pressure_gradient(phi, grid, periodic_x=False)
    gradient = gradient._replace(x=gradient.x.at[..., -1].add(boundary / (.5 * grid.x_widths[-1])))
    base = uniform_velocity(grid)
    dt = .01
    candidate = StaggeredVelocity(*(a + dt*b for a,b in zip(base, gradient)))
    result, pressure = project(candidate, solver, dt, outlet_pressure=boundary)
    for actual, expected in zip(result, base):
        np.testing.assert_allclose(actual, expected, atol=2e-8)
    np.testing.assert_allclose(pressure, phi, atol=2e-7)
    assert float(jnp.max(jnp.abs(divergence(result, grid)))) < 2e-8


def test_backflow_pressure_has_nonpositive_boundary_power_and_is_local():
    grid = UniformGrid(8, 6, 4, 2., 1.5, 1.)
    velocity = uniform_velocity(grid)
    normal = jnp.broadcast_to(jnp.array([-2., -.1, 0., .1, 1., 3.]), (grid.nz, grid.ny))
    velocity = velocity._replace(x=velocity.x.at[..., -1].set(normal), y=velocity.y + 2.)
    pressure = backflow_outlet_pressure(velocity, grid)
    boundary_power = -(pressure + .5 * (normal**2 + 4.)) * normal
    assert bool(jnp.all(boundary_power <= 0.))
    assert bool(jnp.all(pressure[normal >= 0.] == 0.))
    changed = velocity._replace(x=velocity.x.at[..., 0].set(1e6))
    np.testing.assert_array_equal(backflow_outlet_pressure(changed, grid), pressure)
    # Normal and tangential velocity both contribute, not only reversed u^2.
    np.testing.assert_allclose(pressure[0, :2], [-8., -4.01])


def test_rk3_forward_flow_is_unchanged_and_clock_supports_variable_steps():
    grid = UniformGrid(8, 6, 4, 2., 1.5, 1.)
    velocity = uniform_velocity(grid)
    plane = plane_for(velocity)
    solver = build_pressure_poisson(grid, backend="gmg", periodic_x=False, dtype="float64")
    model = FlowModel(momentum_advection_scheme="central", outlet_backflow="energy")
    advance = jax.jit(build_open_atmospheric_step(grid, Boundaries(streamwise=OPEN), solver, model, None, scheme="rk3"))
    state = initial_atmospheric_solution(grid, velocity, dtype="float64")
    state = state._replace(time=jnp.asarray(7.), step=jnp.asarray(12))
    for dt in (.01, .02, .005):
        state = advance(state, dt, plane)
    assert float(state.time) == pytest.approx(7.035)
    assert int(state.step) == 15
    for actual, expected in zip(state.velocity, velocity):
        np.testing.assert_allclose(actual, expected, atol=1e-12)
    for bad_scheme in ("ab2", "fast-rk3"):
        with pytest.raises(ValueError, match="requires rk3"):
            build_open_atmospheric_step(grid, Boundaries(streamwise=OPEN), solver, model, None, scheme=bad_scheme)


def outlet_vortex(grid):
    # Discrete streamfunction gives divergence-free u,v with strong outlet backflow.
    x = jnp.asarray(grid.x_faces)
    y = jnp.asarray(grid.y_faces[:-1])
    psi = 1.2 * jnp.exp(-((x[None, :] - .88 * grid.lx) / (.2 * grid.lx))**2) * jnp.sin(2*jnp.pi*y[:, None]/grid.ly)
    u = 1. + (jnp.roll(psi, -1, axis=0) - psi) / grid.dy
    v = -(psi[:, 1:] - psi[:, :-1]) / grid.dx
    return StaggeredVelocity(jnp.broadcast_to(u, (grid.nz, *u.shape)), jnp.broadcast_to(v, (grid.nz, *v.shape)), jnp.zeros((grid.nz+1, grid.ny, grid.nx)))


def test_central_rk3_backflow_stays_finite_and_conserves_mass():
    grid = UniformGrid(24, 16, 4, 6., 4., 1.)
    velocity = outlet_vortex(grid)
    assert float(jnp.min(velocity.x[..., -1])) < -.2
    assert float(jnp.max(jnp.abs(divergence(velocity, grid)))) < 1e-12
    plane = plane_for(velocity)
    solver = build_pressure_poisson(grid, backend="gmg", periodic_x=False, dtype="float64")
    advance = build_open_atmospheric_step(grid, Boundaries(streamwise=OPEN), solver,
        FlowModel(momentum_advection_scheme="central", outlet_backflow="energy"), None, scheme="rk3")
    state = initial_atmospheric_solution(grid, velocity, dtype="float64")
    @jax.jit
    def run(state):
        def step(state, _):
            state = advance(state, .02, plane)
            return state, (jnp.max(jnp.abs(divergence(state.velocity, grid))), jnp.min(state.velocity.x[..., -1]))
        return jax.lax.scan(step, state, None, length=200)
    final, (errors, outlet_min) = run(state)
    assert np.isfinite(np.asarray(final.pressure)).all()
    assert float(jnp.max(errors)) < 1e-7
    assert float(jnp.min(outlet_min)) < 0.  # Backflow is permitted, not clipped away.
    assert max(float(jnp.max(jnp.abs(f))) for f in final.velocity) < 5.
    np.testing.assert_array_equal(final.velocity.x[..., 0], plane.x_velocity)
    np.testing.assert_allclose(jnp.sum(final.velocity.x[..., -1]), jnp.sum(plane.x_velocity), atol=1e-8)


def test_dtu_configuration_selects_central_rk3_and_backflow():
    from pathlib import Path
    from jaxwind.config.abl import load_fv_abl
    from jaxwind.simulation.abl import build_models
    path = Path(__file__).resolve().parents[2] / "cases/DTU10MWPrecursor/fv_central_backflow.toml"
    config = load_fv_abl(path)
    assert config.options.time_integration == "rk3"
    assert config.options.wall_gradient_correction
    model = build_models(config, periodic_x=False)[1]
    assert model.momentum_advection_scheme == "central"
    assert model.outlet_backflow == "energy"


def test_central_backflow_open_workflow_records_outlet_diagnostics(tmp_path, monkeypatch):
    """Exercise recorded replay and the actual single-turbine stage adapter."""
    import importlib.util
    from pathlib import Path
    from jaxwind.config.document import load_case
    from jaxwind.config.toml import dumps
    from jaxwind.workflows.engine import execute
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("backflow_case_generator", root / "tools/create_dtu10mw_case.py")
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    # NREL fixture checks adapter plumbing only, not DTU aerodynamic accuracy.
    monkeypatch.setenv("JAXWIND_DTU10MW_FAST", str(root / "tests/fixtures/openfast/nrel5mw/NREL5MW_Rigid_Smoke.fst"))
    prepare, main = tmp_path / "prepare", tmp_path / "main"
    for mode, output in (("prepare", prepare), ("main", main)):
        workflow, _ = generator.generate(mode, output, smoke=True, reference=prepare / "run/reference")
        path = output / "cases/case.toml"
        case = load_case(path).document
        case["mesh"]["cells"] = [8, 6, 8]
        case["case"]["profile_resampling"] = "linear"
        case["numerics"].update(momentum_advection_scheme="central", time_integration="rk3", outlet_backflow="energy", wall_gradient_correction=True)
        path.write_text(dumps(case))
        execute(workflow)
    rows = np.atleast_1d(np.genfromtxt(main / "run/main/history.csv", delimiter=",", names=True))
    assert "outlet_backflow_area_fraction" in rows.dtype.names
    assert np.isfinite(rows["outlet_backflow_pressure_min_m2_s2"]).all()
    assert np.all((rows["outlet_backflow_area_fraction"] >= 0) & (rows["outlet_backflow_area_fraction"] <= 1))
    assert (main / "run/main/checkpoint.npz").is_file()


def test_reversing_outlet_projection_preserves_float32_and_mass():
    grid = UniformGrid(12, 8, 4, 6., 4., 1.)
    velocity = StaggeredVelocity(*(f.astype(jnp.float32) for f in outlet_vortex(grid)))
    solver = build_pressure_poisson(grid, backend="gmg", periodic_x=False, dtype="float32")
    plane = plane_for(velocity)
    advance = build_open_atmospheric_step(grid, Boundaries(streamwise=OPEN), solver,
        FlowModel(momentum_advection_scheme="central", outlet_backflow="energy"), None, scheme="rk3")
    initial = initial_atmospheric_solution(grid, velocity, dtype="float32")
    @jax.jit
    def run(state):
        return jax.lax.scan(lambda state, _: (advance(state, .02, plane), None), state, None, length=40)[0]
    final = run(initial)
    assert final.pressure.dtype == jnp.float32
    assert all(f.dtype == jnp.float32 for f in final.velocity)
    assert np.isfinite(np.asarray(final.pressure)).all()
    assert float(jnp.max(jnp.abs(divergence(final.velocity, grid)))) < 2e-5
    assert float(jnp.min(final.velocity.x[..., -1])) < 0.
    np.testing.assert_array_equal(final.velocity.x[..., 0], plane.x_velocity)

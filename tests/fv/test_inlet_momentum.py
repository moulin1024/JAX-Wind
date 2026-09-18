"""Physical-face momentum flux, curvature, and modeled-stress checks."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxwind.domain import UniformGrid
from jaxwind.inlet_momentum import (
    build_physical_inlet_momentum_correction,
    physical_inlet_advection_correction,
    physical_inlet_diffusion_correction,
    physical_inlet_eddy_viscosity,
    physical_inlet_gradients,
    physical_inlet_subfilter_correction,
)
from jaxwind.numerics.discretization import diffusion
from jaxwind.numerics.integrate import FlowModel, build_tendency
from jaxwind.numerics.momentum import muscl_advection
from jaxwind.open_boundary import InflowPlane
from jaxwind.sgs import AnisotropicMinimumDissipation, StaticSmagorinsky, TransportedEddyViscosity, eddy_viscosity, stress_divergence
from jaxwind.state import FREE_SLIP, OPEN, PERIODIC, Boundaries, StaggeredVelocity, Wall


@pytest.fixture(autouse=True)
def precision():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


def _setup(speed=2.0, a=0.3, b=0.7, c=0.0, nx=10):
    grid = UniformGrid(nx, 6, 4, 2.0, 1.2, 0.8)
    boundaries = Boundaries(Wall(FREE_SLIP), Wall(FREE_SLIP), streamwise=OPEN, spanwise=PERIODIC)
    x = jnp.asarray(grid.x_centers)
    y = jnp.broadcast_to(a + b*x + c*x*x, (grid.nz, grid.ny, grid.nx))
    velocity = StaggeredVelocity(jnp.full((grid.nz, grid.ny, grid.nx+1), speed), y, jnp.zeros((grid.nz+1, grid.ny, grid.nx)))
    plane = InflowPlane(velocity.x[..., 0], jnp.full(y.shape[:2], a), jnp.zeros(velocity.z.shape[:2]), jnp.zeros((grid.nz, grid.ny)))
    return grid, boundaries, velocity, plane


def test_uniform_flow_correction_is_zero_and_jittable():
    grid, boundaries, velocity, plane = _setup(a=0.5, b=0.0)
    model = FlowModel(viscosity=0.03, subfilter=AnisotropicMinimumDissipation())
    correct = build_physical_inlet_momentum_correction(grid, boundaries, model)
    result = jax.jit(correct)(velocity, plane)
    for value in result:
        np.testing.assert_allclose(value, 0.0, atol=1e-14)
    np.testing.assert_allclose(physical_inlet_eddy_viscosity(velocity, plane, grid, boundaries, model.subfilter), 0.0, atol=1e-14)


@pytest.mark.parametrize("speed", [2.0, -2.0])
def test_linear_transverse_advection_and_exact_boundary_flux_balance(speed):
    grid, boundaries, velocity, plane = _setup(speed=speed)
    native = muscl_advection(velocity, grid)
    delta = physical_inlet_advection_correction(velocity, plane, grid)
    observed = native.y + delta.y
    # q=a+b*x transported by a constant U: -div(U*q)=-U*b,
    # including both the first cell and its shared neighboring flux.
    np.testing.assert_allclose(observed[..., :3], -speed*0.7, atol=3e-14)
    low_boundary_flux_change = speed*(0.3-np.asarray(velocity.y[..., 0]))
    np.testing.assert_allclose(np.sum(delta.y, axis=-1)*grid.dx, low_boundary_flux_change, atol=1e-14)
    np.testing.assert_array_equal(delta.y[..., 2:], 0.0)
    np.testing.assert_array_equal(delta.x, 0.0)
    np.testing.assert_array_equal(delta.z, 0.0)
    model = FlowModel(viscosity=0.03, subfilter=AnisotropicMinimumDissipation())
    correction = build_physical_inlet_momentum_correction(grid, boundaries, model)
    full = build_tendency(grid, boundaries, model)(velocity, 0.0)
    np.testing.assert_allclose((full.y+correction(velocity, plane).y)[..., :3], -speed*0.7, atol=5e-14)


def test_quadratic_profile_has_documented_one_sided_boundary_error():
    errors = []
    for nx in (10, 20, 40):
        grid, boundaries, velocity, plane = _setup(a=0.4, b=0.2, c=0.3, nx=nx)
        nu = 0.07
        native = diffusion(velocity, grid, boundaries, nu)
        delta = physical_inlet_diffusion_correction(velocity, plane, grid, nu)
        gradients = physical_inlet_gradients(velocity, plane, grid, boundaries)
        # Two-point Dirichlet shear of the sampled quadratic is b+c*dx/2.
        # Its error halves with dx; the first-cell curvature is 3*c/2,
        # not the exact pointwise 2*c. Interior cells retain exact curvature.
        np.testing.assert_allclose(gradients["yx"][..., 0], 0.2+0.15*grid.dx, atol=3e-14)
        np.testing.assert_allclose((native.y+delta.y)[..., 0], nu*0.45, atol=1e-13)
        np.testing.assert_allclose((native.y+delta.y)[..., 1:4], nu*0.6, atol=1e-13)
        errors.append(float(jnp.max(jnp.abs(gradients["yx"][..., 0]-0.2))))
        np.testing.assert_array_equal(delta.y[..., 1:], 0.0)
        np.testing.assert_array_equal(delta.z, 0.0)
    np.testing.assert_allclose(np.asarray(errors[:-1])/errors[1:], 2.0, rtol=1e-11)


@pytest.mark.parametrize("closure", ["molecular", "amd", "transported"])
def test_physical_inlet_traction_and_positive_discrete_dissipation(closure):
    grid, boundaries, velocity, plane = _setup(a=0.25, b=0.0)
    profile = jnp.asarray([0.4, 0.8, 0.2, -0.1, 0., 0., 0., 0., 0., 0.])
    a = -0.4
    u = jnp.broadcast_to(3+a*jnp.asarray(grid.x_faces), velocity.x.shape)
    v = jnp.broadcast_to(profile, velocity.y.shape)
    velocity = StaggeredVelocity(u, v, velocity.z)
    plane = plane._replace(x_velocity=u[..., 0])
    if closure == "molecular":
        nu = 0.07
        old = diffusion(velocity, grid, boundaries, nu)
        change = physical_inlet_diffusion_correction(velocity, plane, grid, nu)
    else:
        if closure == "amd":
            model = AnisotropicMinimumDissipation()
            # For tensor [[a,0,0],[dv/dx,0,0],[0,0,0]], AMD is constant
            # -C*dx^2*a independently of the arbitrary transverse profile.
            nu = -model.poincare_constant*grid.dx**2*a
        else:
            nu = 0.023
            model = TransportedEddyViscosity(jnp.full((grid.nz, grid.ny, grid.nx), nu))
        observed_nu = physical_inlet_eddy_viscosity(velocity, plane, grid, boundaries, model)
        np.testing.assert_allclose(observed_nu, nu, rtol=2e-13, atol=1e-15)
        old = stress_divergence(velocity, eddy_viscosity(velocity, grid, boundaries, model), grid, boundaries)
        change = physical_inlet_subfilter_correction(velocity, plane, grid, boundaries, model)
    tendency = StaggeredVelocity(*(x+y for x,y in zip(old, change)))
    work = float(jnp.sum(profile*tendency.y[1, 2])*grid.dx)
    qb = 0.25
    # The inlet outward normal is -x. The compact profile has no outlet
    # traction or constrained-outlet work, isolating the physical inlet.
    traction_power = -qb*nu*(float(profile[0])-qb)/(0.5*grid.dx)
    dissipation = nu*(float(jnp.sum(jnp.diff(profile)**2))/grid.dx
                      +(float(profile[0])-qb)**2/(0.5*grid.dx))
    assert dissipation > 0
    np.testing.assert_allclose(work, traction_power-dissipation, rtol=2e-13, atol=1e-14)
    np.testing.assert_allclose(tendency.x, 0.0, atol=1e-13)
    np.testing.assert_allclose(tendency.z, 0.0, atol=1e-13)


def test_affine_amd_recomputes_viscosity_and_neighboring_stresses():
    grid = UniformGrid(10, 10, 4, 2.0, 2.0, 0.8)
    boundaries = Boundaries(Wall(FREE_SLIP), Wall(FREE_SLIP), streamwise=OPEN, spanwise=FREE_SLIP)
    a, b, c = -0.4, 0.9, 0.7
    xface = jnp.asarray(grid.x_faces)[None, None, :]
    xcell = jnp.asarray(grid.x_centers)[None, None, :]
    ycell = jnp.asarray(grid.y_centers)[None, :, None]
    u = jnp.broadcast_to(3+a*xface+c*ycell, (grid.nz, grid.ny, grid.nx+1))
    v = jnp.broadcast_to(b*xcell, (grid.nz, grid.ny+1, grid.nx)).at[:, 0].set(0.).at[:, -1].set(0.)
    velocity = StaggeredVelocity(u, v, jnp.zeros((grid.nz+1, grid.ny, grid.nx)))
    plane = InflowPlane(u[..., 0], jnp.zeros(v.shape[:2]), jnp.zeros(velocity.z.shape[:2]), jnp.zeros((grid.nz, grid.ny)))
    model = AnisotropicMinimumDissipation()
    old_nu = eddy_viscosity(velocity, grid, boundaries, model)
    new_nu = physical_inlet_eddy_viscosity(velocity, plane, grid, boundaries, model)
    # Interior affine tensor [[a,c,0],[b,0,0],[0,0,0]] has constant AMD
    # viscosity. The two-point physical boundary derivative recovers exactly b.
    expected_nu = -model.poincare_constant*(grid.dx**2*(a**3+a*b*(b+c))+grid.dy**2*a*c**2)/(a*a+b*b+c*c)
    assert expected_nu > 0
    np.testing.assert_allclose(new_nu[:, 2:-2, :-1], expected_nu, rtol=5e-13, atol=1e-14)
    assert np.max(np.abs(np.asarray(new_nu[:, 2:-2, 0]-old_nu[:, 2:-2, 0]))) > 1e-5
    original = stress_divergence(velocity, old_nu, grid, boundaries)
    delta = physical_inlet_subfilter_correction(velocity, plane, grid, boundaries, model)
    corrected = StaggeredVelocity(*(x+y for x,y in zip(original, delta)))
    # Constant physical stress has zero divergence away from side corners.
    # This includes cell one: updating only the inlet traction cannot pass.
    np.testing.assert_allclose(corrected.y[:, 3:-3, :2], 0.0, atol=2e-14)
    np.testing.assert_allclose(corrected.x[:, 3:-3, 1], 0.0, atol=2e-14)
    assert np.max(np.abs(np.asarray(delta.y[:, 3:-3, 1]))) > 1e-5
    np.testing.assert_array_equal(delta.y[:, 0], 0.0)
    np.testing.assert_array_equal(delta.y[:, -1], 0.0)
    np.testing.assert_array_equal(delta.z[0], 0.0)
    np.testing.assert_array_equal(delta.z[-1], 0.0)


@pytest.mark.parametrize("model,match", [
    (FlowModel(subfilter=AnisotropicMinimumDissipation(), momentum_advection_scheme="central"), "MUSCL-MC"),
    (FlowModel(subfilter=StaticSmagorinsky()), "AMD"),
    (FlowModel(), "AMD"),
])
def test_unsupported_momentum_models_fail_at_builder(model, match):
    grid, boundaries, _, _ = _setup()
    with pytest.raises(ValueError, match=match):
        build_physical_inlet_momentum_correction(grid, boundaries, model)


def test_unsupported_topology_and_short_mesh_are_rejected():
    grid, boundaries, _, _ = _setup()
    model = FlowModel(subfilter=AnisotropicMinimumDissipation())
    with pytest.raises(ValueError, match="open x"):
        build_physical_inlet_momentum_correction(grid, replace(boundaries, streamwise=PERIODIC), model)
    with pytest.raises(ValueError, match="closed or periodic y"):
        build_physical_inlet_momentum_correction(grid, replace(boundaries, spanwise=OPEN), model)
    with pytest.raises(ValueError, match="nx >= 3"):
        build_physical_inlet_momentum_correction(UniformGrid(2, 6, 4, 2., 1.2, .8), boundaries, model)


def test_transported_viscosity_is_preserved_under_changed_inlet_gradients():
    grid, boundaries, velocity, plane = _setup()
    coefficient = jnp.arange(grid.ny*grid.nx, dtype=jnp.float64).reshape(1, grid.ny, grid.nx)/10000.0 + 0.01
    model = TransportedEddyViscosity(coefficient)
    expected = jnp.broadcast_to(coefficient, (grid.nz, grid.ny, grid.nx))
    observed = physical_inlet_eddy_viscosity(velocity, plane, grid, boundaries, model)
    changed = physical_inlet_eddy_viscosity(
        velocity, plane._replace(y_velocity=plane.y_velocity+2.0), grid, boundaries, model
    )
    np.testing.assert_array_equal(observed, expected)
    np.testing.assert_array_equal(changed, expected)
    momentum = FlowModel(viscosity=0.03, subfilter=model)
    correction = build_physical_inlet_momentum_correction(grid, boundaries, momentum)
    # Uniform velocity has no modeled traction, even with nonuniform nu_t.
    uniform = velocity._replace(y=jnp.full_like(velocity.y, 0.3))
    result = jax.jit(correction)(uniform, plane)
    for field in result:
        np.testing.assert_allclose(field, 0.0, atol=1e-14)

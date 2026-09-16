"""Limited second-order momentum transport on uniform MAC control volumes.

The advecting velocity is the arithmetic interpolation of the projected MAC
mass flux, not a reconstructed velocity. Only the transported component is
MUSCL reconstructed, using the monotonized-central (MC) limiter. Shared fluxes
telescope across interior dual faces. The limiter reduces to first order at
extrema and at nonperiodic endpoints. It does not make the complete pressure-
coupled RK scheme TVD, or impose a nonreflecting outlet condition.
"""
import jax.numpy as jnp

from .discretization import _axis_slice, _centered_cells_to_faces
from jaxwind.state import StaggeredVelocity, streamwise_is_periodic, spanwise_is_periodic


def _mc_slope(values, axis, periodic):
    if periodic:
        backward = values - jnp.roll(values, 1, axis=axis)
        forward = jnp.roll(values, -1, axis=axis) - values
    else:
        differences = jnp.diff(values, axis=axis)
        zero = jnp.zeros_like(values[_axis_slice(values.ndim, axis, 0, 1)])
        backward = jnp.concatenate((zero, differences), axis=axis)
        forward = jnp.concatenate((differences, zero), axis=axis)
    centered = .5 * (backward + forward)
    magnitude = jnp.minimum(jnp.abs(centered),
                            2. * jnp.minimum(jnp.abs(backward), jnp.abs(forward)))
    same_sign = ((backward > 0.) & (forward > 0.)) | ((backward < 0.) & (forward < 0.))
    return jnp.where(same_sign, jnp.sign(centered) * magnitude, 0.)


def _muscl_flux(values, mass_velocity, axis, periodic, *, internal=False, upwind=True):
    """Flux at lower dual faces, or between nodes for open normal components.

    For a nonperiodic transverse axis the two boundary states are copied from
    the boundary-adjacent velocity. The existing BC enforcement supplies inlet
    values and outlet extrapolation; no opposite-edge data are referenced.
    """
    slope = _mc_slope(values, axis, periodic) if upwind else jnp.zeros_like(values)
    plus, minus = values + .5 * slope, values - .5 * slope
    if periodic:
        left, right = jnp.roll(plus, 1, axis=axis), minus
    else:
        left = plus[_axis_slice(values.ndim, axis, 0, -1)]
        right = minus[_axis_slice(values.ndim, axis, 1, None)]
        if not internal:
            low = values[_axis_slice(values.ndim, axis, 0, 1)]
            high = values[_axis_slice(values.ndim, axis, -1, None)]
            left = jnp.concatenate((low, left, high), axis=axis)
            right = jnp.concatenate((low, right, high), axis=axis)
    transported = jnp.where(mass_velocity >= 0., left, right) if upwind else .5 * (left + right)
    return mass_velocity * transported


def muscl_advection(velocity: StaggeredVelocity, grid, *, upwind_axes=(0, 1, 2)) -> StaggeredVelocity:
    """Conservative MC-limited upwind advection; uniform grids only.

    Endpoint tendencies retain the centered operator's boundary policy:
    zero normal momentum tendency at nonperiodic normal faces. Open boundary
    enforcement and pressure projection subsequently set their velocities.
    """
    if not grid.is_uniform:
        raise ValueError("muscl-mc momentum advection requires a uniform grid")
    periodic = (False, spanwise_is_periodic(velocity, grid),
                streamwise_is_periodic(velocity, grid))
    spacing = (grid.dz, grid.dy, grid.dx)
    # Coordinate order for arrays is z,y,x; component order is u,v,w.
    normal_velocity = (velocity.z, velocity.y, velocity.x)
    results = []
    for component, values in zip((2, 1, 0), velocity):
        total = jnp.zeros_like(values)
        for axis in (0, 1, 2):
            if axis == component:
                if periodic[axis]:
                    mass = .5 * (values + jnp.roll(values, 1, axis=axis))
                else:
                    mass = .5 * (values[_axis_slice(3, axis, 0, -1)]
                                 + values[_axis_slice(3, axis, 1, None)])
                flux = _muscl_flux(values, mass, axis, periodic[axis], internal=True, upwind=axis in upwind_axes)
            else:
                # Interpolated normal MAC flux through this component's dual
                # faces: its divergence is the corresponding average of the
                # primal-cell divergence on a uniform mesh.
                mass = _centered_cells_to_faces(
                    normal_velocity[axis], component, periodic=periodic[component],
                    boundary="zero" if component == 0 else "copy",
                )
                flux = _muscl_flux(values, mass, axis, periodic[axis], upwind=axis in upwind_axes)
            if periodic[axis]:
                difference = jnp.roll(flux, -1, axis=axis) - flux
            else:
                difference = jnp.diff(flux, axis=axis)
                if axis == component:
                    zero = jnp.zeros_like(values[_axis_slice(3, axis, 0, 1)])
                    difference = jnp.concatenate((zero, difference, zero), axis=axis)
            total = total - difference / spacing[axis]
        if not periodic[component]:
            total = total.at[_axis_slice(3, component, 0, 1)].set(0.)
            total = total.at[_axis_slice(3, component, -1, None)].set(0.)
        results.append(total)
    return StaggeredVelocity(*results)


def central_open_upwind_advection(velocity: StaggeredVelocity, grid) -> StaggeredVelocity:
    """Central precursor; horizontal MUSCL and central vertical fluxes in open x.

    Periodic precursor transport is exactly the existing central operator.
    Open-domain vertical momentum fluxes remain centered: no upwind diffusion
    is applied across the wall-normal mean shear. Horizontal upwinding still
    changes resolved turbulence, so main-domain profile validation is required.
    """
    from .discretization import advection
    if streamwise_is_periodic(velocity, grid):
        return advection(velocity, grid)
    return muscl_advection(velocity, grid, upwind_axes=(1, 2))


def _fourth_difference_damping(values, axis, periodic):
    """Return -D2.T D2 q, with no opposite-edge coupling for open axes.

    D2 is the dimensionless second difference. Its negative Gram operator
    conserves the sum and dissipates squared amplitude, including at open
    endpoints. Open affine fields are in its null space. This is a weak
    high-frequency filter, not a monotonicity or positivity limiter.
    """
    if periodic:
        curvature = jnp.roll(values, -1, axis=axis) - 2*values + jnp.roll(values, 1, axis=axis)
        return -(jnp.roll(curvature, -1, axis=axis) - 2*curvature + jnp.roll(curvature, 1, axis=axis))
    curvature = jnp.diff(values, n=2, axis=axis)
    def pad(low, high):
        widths = [(0, 0)]*values.ndim
        widths[axis] = (low, high)
        return jnp.pad(curvature, widths)
    return -pad(0, 2) + 2*pad(1, 1) - pad(2, 0)


def central_open_filter_advection(velocity: StaggeredVelocity, grid) -> StaggeredVelocity:
    """Central transport with weak horizontal fourth-difference dissipation.

    The periodic precursor is bitwise central. In open domains the damping
    coefficient is 1/64 times the directional maximum advecting speed divided
    by cell width. A Fourier mode has decay factor proportional to sin(kh/2)^4,
    concentrating dissipation at short horizontal wavelengths. Vertical fluxes
    and horizontally uniform mean shear are unchanged. Uniform grids only.
    """
    from .discretization import advection
    central = advection(velocity, grid)
    if streamwise_is_periodic(velocity, grid):
        return central
    if not grid.is_uniform:
        raise ValueError('central-open-filter requires a uniform grid')
    periodic_y = spanwise_is_periodic(velocity, grid)
    rates = ((2, False, jnp.max(jnp.abs(velocity.x))/(64*grid.dx)),
             (1, periodic_y, jnp.max(jnp.abs(velocity.y))/(64*grid.dy)))
    result = []
    for component, values, base in zip((2, 1, 0), velocity, central):
        damping = sum(rate*_fourth_difference_damping(values, axis, periodic)
                      for axis, periodic, rate in rates)
        # Prescribed normal endpoints retain the existing boundary policy.
        if component == 0 or (component == 1 and not periodic_y):
            damping = damping.at[_axis_slice(3, component, 0, 1)].set(0.)
            damping = damping.at[_axis_slice(3, component, -1, None)].set(0.)
        if component == 2:
            damping = damping.at[..., 0].set(0.).at[..., -1].set(0.)
        result.append(base+damping)
    return StaggeredVelocity(*result)

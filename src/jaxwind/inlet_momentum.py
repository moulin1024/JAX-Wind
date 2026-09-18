"""Opt-in physical-face transverse inlet corrections for AMD or transported viscosity.

Add the returned correction to the ordinary momentum RHS only when its inlet
v/w cell overwrite has been disabled and the pressure operator releases those
DOFs. Normal inlet velocity and the legacy outlet policy remain caller-owned.
The scalar closure must use ``physical_inlet_eddy_viscosity`` too: changing an
inlet derivative changes AMD viscosity and neighboring stresses, not just the
inlet traction. A transported coefficient is retained without recomputation.
No defaults or existing operators are changed by this module.
"""

import jax.numpy as jnp

from .numerics.discretization import _centered_cells_to_faces
from .open_boundary import validate_inflow_plane
from .sgs import AnisotropicMinimumDissipation, TransportedEddyViscosity, edge_gradients, eddy_viscosity, stress_divergence
from .state import OPEN, StaggeredVelocity, spanwise_is_periodic, validate


def _require_grid(grid):
    if not grid.is_uniform or grid.nx < 3:
        raise ValueError("physical transverse inlet requires a uniform grid with nx >= 3")


def _check(velocity, inflow, grid, boundaries=None):
    _require_grid(grid)
    if velocity.x.shape[-1] != grid.nx + 1:
        raise ValueError("physical transverse inlet requires open x velocity")
    validate_inflow_plane(inflow, grid, wall_y=not spanwise_is_periodic(velocity, grid))
    if boundaries is not None:
        if boundaries.streamwise != OPEN or boundaries.spanwise == OPEN:
            raise ValueError("physical transverse inlet requires open x and closed or periodic y")
        validate(velocity, grid, boundaries)


def _inlet_derivative(values, boundary, dx):
    """Two-point face gradient preserving physical traction-work balance.

    This has first-order boundary-gradient accuracy for a curved profile.
    With the half-cell distance it pairs exactly with native FV divergence:
    boundary work uses the prescribed face value and dissipation is positive.
    A quadratic one-sided derivative would add a boundary energy remainder.
    """
    return (values[..., 0] - boundary) / (0.5 * dx)


def _first_slope(values, boundary):
    # The reflected linear ghost supplies the missing MC backward difference.
    # The boundary flux itself uses the prescribed face value, not a ghost donor.
    backward = 2.0 * (values[..., 0] - boundary)
    forward = values[..., 1] - values[..., 0]
    centered = 0.5 * (backward + forward)
    magnitude = jnp.minimum(jnp.abs(centered), 2.0 * jnp.minimum(jnp.abs(backward), jnp.abs(forward)))
    same_sign = ((backward > 0) & (forward > 0)) | ((backward < 0) & (forward < 0))
    return jnp.where(same_sign, jnp.sign(centered) * magnitude, 0.0)


def _transverse_result(velocity, y, z, grid):
    # Match the unchanged impermeable normal-face RHS constraints.
    if not spanwise_is_periodic(velocity, grid):
        y = y.at[:, 0].set(0.0).at[:, -1].set(0.0)
    z = z.at[0].set(0.0).at[-1].set(0.0)
    return StaggeredVelocity(jnp.zeros_like(velocity.x), y, z)


def physical_inlet_advection_correction(velocity, inflow, grid):
    """Exact delta to native MUSCL-MC fluxes for the first v/w x cell.

    The new first-cell slope also changes its shared outflow face. Applying
    that flux difference to both neighboring control volumes preserves the
    telescoping momentum balance. The inlet uses reservoir data on inflow and
    the reconstructed interior donor on reversal; normal MAC mass fluxes are
    never reconstructed or changed here.
    """
    _check(velocity, inflow, grid)
    result = []
    for component, values, boundary in ((1, velocity.y, inflow.y_velocity), (0, velocity.z, inflow.z_velocity)):
        mass = _centered_cells_to_faces(
            velocity.x, component,
            periodic=spanwise_is_periodic(velocity, grid) if component == 1 else False,
            boundary="copy" if component == 1 else "zero",
        )
        slope = _first_slope(values, boundary)
        first = values[..., 0]
        inlet_donor = jnp.where(mass[..., 0] >= 0, boundary, first - 0.5 * slope)
        delta_low = mass[..., 0] * (inlet_donor - first)
        # Native MC has zero slope in cell zero. The donor from cell one is
        # unchanged when this interior face reverses.
        delta_high = jnp.where(mass[..., 1] >= 0, 0.5 * mass[..., 1] * slope, 0.0)
        delta = jnp.zeros_like(values)
        delta = delta.at[..., 0].set((delta_low - delta_high) / grid.dx)
        delta = delta.at[..., 1].set(delta_high / grid.dx)
        result.append(delta)
    return _transverse_result(velocity, *result, grid)


def physical_inlet_diffusion_correction(velocity, inflow, grid, viscosity):
    """Restore physical inlet x-curvature omitted by native molecular diffusion."""
    _check(velocity, inflow, grid)
    result = []
    for values, boundary in ((velocity.y, inflow.y_velocity), (velocity.z, inflow.z_velocity)):
        interior = (values[..., 1] - values[..., 0]) / grid.dx
        inlet = _inlet_derivative(values, boundary, grid.dx)
        # Native diffusion zeroes the entire first-column x-curvature; merely
        # replacing its zero boundary gradient would therefore be insufficient.
        delta = jnp.zeros_like(values).at[..., 0].set(viscosity * (interior - inlet) / grid.dx)
        result.append(delta)
    return _transverse_result(velocity, *result, grid)


def physical_inlet_gradients(velocity, inflow, grid, boundaries):
    """Native gradients with physical Dirichlet inlet dv/dx and dw/dx.

    RANS production and realizable invariants must use this same gradient set.
    The momentum correction adds no separate production or boundary TKE source.
    """
    _check(velocity, inflow, grid, boundaries)
    gradients = edge_gradients(velocity, grid, boundaries)
    gradients["yx"] = gradients["yx"].at[..., 0].set(_inlet_derivative(velocity.y, inflow.y_velocity, grid.dx))
    gradients["zx"] = gradients["zx"].at[..., 0].set(_inlet_derivative(velocity.z, inflow.z_velocity, grid.dx))
    return gradients


def physical_inlet_eddy_viscosity(velocity, inflow, grid, boundaries, model):
    """Consistent AMD viscosity, or the unchanged prescribed transported field."""
    if isinstance(model, TransportedEddyViscosity):
        _check(velocity, inflow, grid, boundaries)
        return eddy_viscosity(velocity, grid, boundaries, model)
    if not isinstance(model, AnisotropicMinimumDissipation):
        raise ValueError("physical transverse inlet supports AMD or transported viscosity only")
    gradients = physical_inlet_gradients(velocity, inflow, grid, boundaries)
    return eddy_viscosity(velocity, grid, boundaries, model, gradients=gradients)


def physical_inlet_subfilter_correction(velocity, inflow, grid, boundaries, model):
    """Replace full native stress divergence with the same coefficient policy.

    AMD recomputes its gradient-dependent coefficient. Transported viscosity is
    shared by both stress evaluations; its evolution belongs to the RANS step.
    """
    if not isinstance(model, (AnisotropicMinimumDissipation, TransportedEddyViscosity)):
        raise ValueError("physical transverse inlet supports AMD or transported viscosity only")
    corrected = physical_inlet_gradients(velocity, inflow, grid, boundaries)
    original = edge_gradients(velocity, grid, boundaries)
    old_nu = eddy_viscosity(velocity, grid, boundaries, model, gradients=original)
    new_nu = (
        old_nu if isinstance(model, TransportedEddyViscosity)
        else eddy_viscosity(velocity, grid, boundaries, model, gradients=corrected)
    )
    old = stress_divergence(velocity, old_nu, grid, boundaries, gradients=original)
    new = stress_divergence(velocity, new_nu, grid, boundaries, gradients=corrected)
    return StaggeredVelocity(*(a - b for a, b in zip(new, old)))


def build_physical_inlet_momentum_correction(grid, boundaries, model):
    """Build correction(velocity, inflow) for the existing momentum RHS.

    Caller must opt in to compatible enforcement/projection and use the same
    inlet-aware viscosity helper for scalar transport. Prescribed body/wall forces
    remain in the original RHS. A surface gradient correction is unsupported.
    """
    _require_grid(grid)
    if boundaries.streamwise != OPEN or boundaries.spanwise == OPEN:
        raise ValueError("physical transverse inlet requires open x and closed or periodic y")
    if model.momentum_advection_scheme != "muscl-mc":
        raise ValueError("physical transverse inlet requires MUSCL-MC momentum")
    if not isinstance(model.subfilter, (AnisotropicMinimumDissipation, TransportedEddyViscosity)):
        raise ValueError("physical transverse inlet supports AMD or transported viscosity only")
    if model.surface is not None:
        raise ValueError("physical transverse inlet does not support surface gradient corrections")

    def correction(velocity, inflow):
        _check(velocity, inflow, grid, boundaries)
        advective = physical_inlet_advection_correction(velocity, inflow, grid)
        molecular = physical_inlet_diffusion_correction(velocity, inflow, grid, model.viscosity)
        subfilter = physical_inlet_subfilter_correction(velocity, inflow, grid, boundaries, model.subfilter)
        return StaggeredVelocity(*(a + b + c for a, b, c in zip(advective, molecular, subfilter)))

    return correction

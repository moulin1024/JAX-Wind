"""Well-mixed tracer building block for prescribed inhomogeneous turbulence.

Scope: constant-density tracers, stationary isotropic Gaussian velocity PDF,
zero mean flow, smooth strictly positive variance q(x) per component. This is
one 3D drift satisfying the Fokker--Planck well-mixed condition, not a unique
closure or a finite-inertia fluid-seen-velocity model. No LES energy, shear,
variable density, walls or turbulent/non-turbulent interface is modeled.
The derivation and numerical verification are in the spray validation case.
"""

import jax.numpy as jnp


def isotropic_well_mixed_drift(velocity, variance, variance_gradient, time_scale):
    """Physical-velocity Itô drift, for diffusion sqrt(2*q/T)*I.

    Vector fields have leading dimension 3. q>0, T>0 are prescribed; grad(q)
    must be the gradient of that same variance. No numerical floor replaces a
    zero-turbulence boundary. The stationary target is uniform concentration
    with a local N(0,q I) velocity PDF. Not applicable to inertial droplets.
    """
    directional = jnp.sum(velocity * variance_gradient, axis=0)
    return (
        -velocity / time_scale
        + 0.5 * variance_gradient
        + velocity * directional / (2 * variance)
    )


def normalized_transport_midpoint(position, normalized_velocity, dt, fields):
    """RK2 transport part: dx=sigma*w dt, dw=grad(sigma) dt.

    fields(x) returns (q, grad(q), T). w=u/sqrt(q(x)) is dimensionless.
    This deterministic subflow preserves the target density in continuous time;
    this explicit midpoint approximation requires timestep convergence checks.
    Boundary handling belongs to the caller, and field evaluation must remain
    valid at intermediate positions (e.g. a periodic analytic field).
    """
    q, gradient, _ = fields(position)
    sigma = jnp.sqrt(q)
    middle_x = position + 0.5 * dt * sigma * normalized_velocity
    middle_w = normalized_velocity + 0.25 * dt * gradient / sigma
    middle_q, middle_gradient, _ = fields(middle_x)
    middle_sigma = jnp.sqrt(middle_q)
    return (
        position + dt * middle_sigma * middle_w,
        normalized_velocity + 0.5 * dt * middle_gradient / middle_sigma,
    )


def well_mixed_tracer_step(position, normalized_velocity, dt, normal_sample, fields):
    """Symmetric OU/transport/OU step in normalized velocity w=u/sqrt(q).

    normal_sample has shape (2,3,...) and independent unit normal entries for
    the two OU half-steps. The caller owns PRNG and particle positions. Each OU
    step preserves N(0,I) exactly at frozen position; deterministic transport
    uses explicit midpoint. No mass exchange or resampling enforces uniformity.
    q and T must stay strictly positive and finite throughout the step.
    Returns (position, normalized_velocity); physical velocity is sqrt(q(x))*w.
    """

    def ou(w, x, sample):
        _, _, tau = fields(x)
        a = jnp.exp(-0.5 * dt / tau)
        b = jnp.sqrt(-jnp.expm1(-dt / tau))
        return a * w + b * sample

    w = ou(normalized_velocity, position, normal_sample[0])
    x, w = normalized_transport_midpoint(position, w, dt, fields)
    return x, ou(w, x, normal_sample[1])

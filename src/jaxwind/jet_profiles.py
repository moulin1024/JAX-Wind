"""Alternative flux-preserving similarity reconstruction, pending assessment."""

import jax.numpy as jnp

from .spray_closure import RoundJetFlux


def round_jet_squared_lorentzian(flux: RoundJetFlux, density):
    """Return Uc,b for U(r)=Uc/(1+(r/b)^2)^2.

    This is the constant-radial-eddy-viscosity boundary-layer similarity shape
    (Basset et al. 2022, doi:10.1017/jfm.2022.638, equations 3.4 and 5.10).
    It is an alternative to the Gaussian, NOT an accuracy correction fitted to
    Hussein. Preconditions: mass, momentum and density positive and finite.

    Integrals give m=pi*rho*Uc*b^2 and J=pi*rho*Uc^2*b^2/3. The half-width is
    b*sqrt(sqrt(2)-1). Holding m and J fixes BOTH Uc and b; independently assigning
    a measured width would break conservation. Flux momentum remains mean-flow
    only, without pressure or Reynolds-normal-stress contributions.
    """
    return (
        3 * flux.momentum / flux.mass,
        flux.mass / jnp.sqrt(3 * jnp.pi * density * flux.momentum),
    )


def gaussian_mixture_moments(weights, coefficients):
    """Radial integrals I1=int s*f ds and I2=int s*f^2 ds, s in [0,infinity).

    f(s)=sum_i weights_i*exp(-coefficients_i*s^2). Preconditions: one-dimensional
    nonnegative weights summing to one, strictly positive finite coefficients.
    The mixture is an empirical shape family, not a turbulent stress closure.
    """
    weights, coefficients = jnp.asarray(weights), jnp.asarray(coefficients)
    i1 = jnp.sum(weights / (2 * coefficients))
    i2 = jnp.sum(
        weights[:, None]
        * weights[None, :]
        / (2 * (coefficients[:, None] + coefficients[None, :]))
    )
    return i1, i2


def round_jet_gaussian_mixture(flux: RoundJetFlux, density, weights, coefficients):
    """Return Uc,L for U(r)=Uc*sum w_i exp(-c_i*(r/L)^2).

    Preserves m=2*pi*rho*Uc*L^2*I1 and J=2*pi*rho*Uc^2*L^2*I2.
    Like the other candidates, J omits stress and pressure contributions.
    Coefficients and weights must be supplied from independent calibration;
    there is no default calibrated profile or automatic production enablement.
    """
    i1, i2 = gaussian_mixture_moments(weights, coefficients)
    return (
        flux.momentum * i1 / (flux.mass * i2),
        flux.mass
        * jnp.sqrt(i2)
        / (i1 * jnp.sqrt(2 * jnp.pi * density * flux.momentum)),
    )


def gaussian_mixture_plane_fluxes(
    flux, density, weights, coefficients, y_edges, z_edges, *, center=(0.0, 0.0)
):
    """Analytic rectangular-face integrals, retaining loss outside finite planes.

    Momentum includes every pair cross term of f^2. Specific scalar quantities
    are uniform, as in the original Gaussian reconstruction. These are separate
    flux integrals, not a completed LES handoff or scalar-mixing model.
    """
    from jax.scipy.special import erf

    weights, coefficients = jnp.asarray(weights), jnp.asarray(coefficients)
    _, length = round_jet_gaussian_mixture(flux, density, weights, coefficients)
    i1, i2 = gaussian_mixture_moments(weights, coefficients)

    def fraction(c):
        scale = jnp.sqrt(c)[:, None] / length
        fy = jnp.diff(erf(scale * (jnp.asarray(y_edges) - center[0])), axis=-1) / 2
        fz = jnp.diff(erf(scale * (jnp.asarray(z_edges) - center[1])), axis=-1) / 2
        return fz[:, :, None] * fy[:, None, :]

    mass_fraction = jnp.sum(
        (weights / (2 * coefficients * i1))[:, None, None] * fraction(coefficients),
        axis=0,
    )
    pair_c = (coefficients[:, None] + coefficients[None, :]).ravel()
    pair_w = (weights[:, None] * weights[None, :]).ravel()
    momentum_fraction = jnp.sum(
        (pair_w / (2 * pair_c * i2))[:, None, None] * fraction(pair_c), axis=0
    )
    return RoundJetFlux(
        flux.mass * mass_fraction,
        flux.momentum * momentum_fraction,
        flux.enthalpy * mass_fraction,
        jnp.asarray(flux.species)[:, None, None] * mass_fraction,
    )

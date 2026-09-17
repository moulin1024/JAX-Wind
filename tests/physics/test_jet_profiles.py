"""Independent quadrature and differential-equation checks of reconstruction."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from jaxwind.jet_profiles import round_jet_squared_lorentzian
from jaxwind.spray_closure import RoundJetFlux

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("mass,momentum,density", [(0.2, 1.5, 1.2), (1.7, 0.09, 998.0)])
def test_reconstruction_preserves_flux_integrals_and_half_width(
    mass, momentum, density
):
    flux = RoundJetFlux(mass, momentum, 0.0, jnp.array([mass]))
    uc, b = map(float, jax.jit(round_jet_squared_lorentzian)(flux, density))
    # Independent adaptive quadrature over the infinite radial domain.
    m = quad(
        lambda r: 2 * np.pi * density * r * uc / (1 + (r / b) ** 2) ** 2,
        0,
        np.inf,
        epsabs=1e-12,
    )[0]
    j = quad(
        lambda r: 2 * np.pi * density * r * uc**2 / (1 + (r / b) ** 2) ** 4,
        0,
        np.inf,
        epsabs=1e-12,
    )[0]
    np.testing.assert_allclose([m, j], [mass, momentum], rtol=1e-10)
    half = b * np.sqrt(np.sqrt(2) - 1)
    np.testing.assert_allclose(uc / (1 + (half / b) ** 2) ** 2, 0.5 * uc, rtol=1e-14)


def test_profile_implies_constant_radial_eddy_viscosity():
    # Boundary-layer identity: nu/(Uc*b) = -f/(eta*f') * integral_0^eta s f ds.
    # Evaluate the integral independently, and derivative by autodiff.
    f = lambda eta: (1 + eta**2) ** -2
    derivative = jax.grad(f)
    for eta in (0.01, 0.1, 0.5, 1.0, 2.0, 5.0):
        integral = quad(lambda s: s * float(f(s)), 0, eta)[0]
        inferred = -float(f(eta)) * integral / (eta * float(derivative(eta)))
        np.testing.assert_allclose(inferred, 1 / 8, rtol=1e-12)


def test_mixture_reconstruction_matches_independent_flux_quadrature():
    from jaxwind.jet_profiles import round_jet_gaussian_mixture

    weights, coefficients = jnp.array([0.7, 0.3]), jnp.array([1.4, 0.2])
    flux = RoundJetFlux(0.17, 0.85, 420.0, jnp.array([0.12, 0.05]))
    uc, length = map(
        float, jax.jit(round_jet_gaussian_mixture)(flux, 1.2, weights, coefficients)
    )

    def velocity(r):
        return uc * sum(
            float(w) * np.exp(-float(c) * (r / length) ** 2)
            for w, c in zip(weights, coefficients)
        )

    for power, expected in [(1, flux.mass), (2, flux.momentum)]:
        value = quad(
            lambda r, power=power: 2 * np.pi * 1.2 * r * velocity(r) ** power,
            0,
            np.inf,
            epsabs=1e-12,
        )[0]
        np.testing.assert_allclose(value, expected, rtol=1e-11)


def test_mixture_plane_integrates_cross_terms_and_retains_cropped_loss():
    from scipy.integrate import dblquad

    from jaxwind.jet_profiles import (
        gaussian_mixture_plane_fluxes,
        round_jet_gaussian_mixture,
    )

    flux = RoundJetFlux(0.17, 0.85, 420.0, jnp.array([0.12, 0.05]))
    w, c = jnp.array([0.7, 0.3]), jnp.array([1.4, 0.2])
    uc, length = map(float, round_jet_gaussian_mixture(flux, 1.2, w, c))
    y, z = jnp.array([-0.02, 0.03, 0.07]), jnp.array([-0.01, 0.015, 0.04])
    center = (0.01, -0.005)
    result = jax.jit(
        lambda: gaussian_mixture_plane_fluxes(flux, 1.2, w, c, y, z, center=center)
    )()
    for iz in range(2):
        for iy in range(2):
            for power, field in [(1, result.mass), (2, result.momentum)]:
                value = dblquad(
                    lambda zz, yy, power=power: (
                        1.2
                        * (
                            uc
                            * sum(
                                float(ww)
                                * np.exp(
                                    -float(cc)
                                    * ((yy - center[0]) ** 2 + (zz - center[1]) ** 2)
                                    / length**2
                                )
                                for ww, cc in zip(w, c)
                            )
                        )
                        ** power
                    ),
                    float(y[iy]),
                    float(y[iy + 1]),
                    lambda _, iz=iz: float(z[iz]),
                    lambda _, iz=iz: float(z[iz + 1]),
                    epsabs=1e-12,
                )[0]
                np.testing.assert_allclose(field[iz, iy], value, rtol=1e-11)
    assert 0 < float(result.mass.sum()) < flux.mass
    np.testing.assert_allclose(result.species.sum(axis=0), result.mass, rtol=1e-14)
    np.testing.assert_allclose(
        result.enthalpy, result.mass * flux.enthalpy / flux.mass, rtol=1e-14
    )


def test_mixture_one_component_and_duplicate_components_recover_original_gaussian():
    from jaxwind.jet_profiles import gaussian_mixture_plane_fluxes
    from jaxwind.spray_closure import round_jet_plane_fluxes

    flux = RoundJetFlux(0.17, 0.85, 420.0, jnp.array([0.12, 0.05]))
    edges = jnp.array([-1.0, -0.015, 0.025, 1.0])
    original = round_jet_plane_fluxes(flux, 1.2, edges, edges)
    for w, c in [([1.0], [75.0]), ([0.3, 0.7], [2.0, 2.0])]:
        result = gaussian_mixture_plane_fluxes(
            flux, 1.2, jnp.array(w), jnp.array(c), edges, edges
        )
        for a, b in zip(result, original):
            np.testing.assert_allclose(a, b, rtol=2e-14, atol=1e-15)

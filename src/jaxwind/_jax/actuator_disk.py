"""Private JAX kernels for force-conserving filtered actuator disks."""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import i0e


_GAUSS_LEGENDRE_NODES = (
    -0.9894009349916499,
    -0.9445750230732326,
    -0.8656312023878318,
    -0.755404408355003,
    -0.6178762444026438,
    -0.45801677765722737,
    -0.2816035507792589,
    -0.09501250983763744,
    0.09501250983763744,
    0.2816035507792589,
    0.45801677765722737,
    0.6178762444026438,
    0.755404408355003,
    0.8656312023878318,
    0.9445750230732326,
    0.9894009349916499,
)
_GAUSS_LEGENDRE_WEIGHTS = (
    0.027152459411754176,
    0.062253523938647456,
    0.0951585116824926,
    0.12462897125553407,
    0.1495959888165767,
    0.16915651939500265,
    0.18260341504492364,
    0.18945061045506864,
    0.18945061045506864,
    0.18260341504492364,
    0.16915651939500265,
    0.1495959888165767,
    0.12462897125553407,
    0.0951585116824926,
    0.062253523938647456,
    0.027152459411754176,
)


def _gaussian_convolved_disk(radius, disk_radius, smoothing_width):
    """Convolve a circular top-hat with a normalized 2-D Gaussian.

    The Gaussian convention is ``exp(-(r / epsilon)**2) / (pi epsilon**2)``.
    A fixed Gauss--Legendre rule integrates the azimuthally reduced
    convolution.  The exponentially scaled Bessel function keeps the
    expression bounded for narrow filters.
    """
    dtype = radius.dtype
    nodes = jnp.asarray(_GAUSS_LEGENDRE_NODES, dtype=dtype)
    weights = jnp.asarray(_GAUSS_LEGENDRE_WEIGHTS, dtype=dtype)
    disk_radius = jnp.asarray(disk_radius, dtype=dtype)
    epsilon = jnp.asarray(smoothing_width, dtype=dtype)
    source_radius = 0.5 * disk_radius * (nodes + 1.0)
    source_weights = 0.5 * disk_radius * weights
    argument = (
        2.0 * radius[..., None] * source_radius / (epsilon * epsilon)
    )
    integrand = (
        2.0
        * source_radius
        / (epsilon * epsilon)
        * jnp.exp(
            -(
                (radius[..., None] - source_radius)
                / epsilon
            )
            ** 2
        )
        * i0e(argument)
    )
    return jnp.sum(source_weights * integrand, axis=-1)


def gaussian_convolved_annulus(
    radius,
    *,
    outer_radius,
    inner_radius,
    smoothing_width,
):
    """Return a Gaussian-convolved circular or annular indicator."""
    outer = _gaussian_convolved_disk(
        radius,
        outer_radius,
        smoothing_width,
    )
    inner = _gaussian_convolved_disk(
        radius,
        inner_radius,
        smoothing_width,
    )
    return jnp.maximum(outer - inner, 0.0)


def filtered_disk_velocity_correction(
    thrust_coefficient_prime,
    *,
    outer_radius,
    inner_radius,
    smoothing_width,
    dtype,
):
    """Evaluate the Shapiro--Gayme--Meneveau filtered-disk correction.

    The overlap integral is evaluated from the actual Gaussian-convolved
    radial indicator instead of using the small-filter asymptotic formula.
    """
    outer_radius = jnp.asarray(outer_radius, dtype=dtype)
    inner_radius = jnp.asarray(inner_radius, dtype=dtype)
    epsilon = jnp.asarray(smoothing_width, dtype=dtype)
    radial_limit = outer_radius + 6.0 * epsilon
    sample_count = 256
    spacing = radial_limit / sample_count
    radius = (jnp.arange(sample_count, dtype=dtype) + 0.5) * spacing
    indicator = gaussian_convolved_annulus(
        radius,
        outer_radius=outer_radius,
        inner_radius=inner_radius,
        smoothing_width=epsilon,
    )
    measure = 2.0 * jnp.pi * radius * spacing
    normalization = jnp.sum(indicator * measure)
    normalized = indicator / jnp.maximum(
        normalization,
        jnp.finfo(dtype).tiny,
    )
    area = jnp.pi * (outer_radius**2 - inner_radius**2)
    overlap = area * jnp.sum(normalized * normalized * measure)
    overlap = jnp.clip(overlap, 0.0, 1.0)
    return 1.0 / (
        1.0
        + 0.25
        * jnp.asarray(thrust_coefficient_prime, dtype=dtype)
        * (1.0 - overlap)
    )

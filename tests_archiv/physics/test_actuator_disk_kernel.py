from __future__ import annotations

import jax
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from jaxwind_archiv.interpreters._jax_actuator_disk import (  # noqa: E402
    filtered_disk_velocity_correction,
    gaussian_convolved_annulus,
)


def test_gaussian_disk_has_exact_center_value() -> None:
    radius = 0.75
    epsilon = 0.2
    value = gaussian_convolved_annulus(
        jnp.asarray(0.0, dtype=jnp.float64),
        outer_radius=radius,
        inner_radius=0.0,
        smoothing_width=epsilon,
    )

    expected = 1.0 - jnp.exp(-(radius / epsilon) ** 2)
    assert float(value) == pytest.approx(float(expected), rel=2.0e-12)


def test_gaussian_disk_convolution_preserves_area() -> None:
    disk_radius = 0.75
    epsilon = 0.2
    spacing = 0.002
    radius = (jnp.arange(800, dtype=jnp.float64) + 0.5) * spacing
    indicator = gaussian_convolved_annulus(
        radius,
        outer_radius=disk_radius,
        inner_radius=0.0,
        smoothing_width=epsilon,
    )
    area = jnp.sum(indicator * 2.0 * jnp.pi * radius * spacing)

    assert float(area) == pytest.approx(
        jnp.pi * disk_radius**2,
        rel=3.0e-5,
    )


def test_filtered_velocity_correction_is_bounded_and_width_dependent() -> None:
    narrow = filtered_disk_velocity_correction(
        1.4,
        outer_radius=0.75,
        inner_radius=0.0,
        smoothing_width=0.05,
        dtype=jnp.float64,
    )
    broad = filtered_disk_velocity_correction(
        1.4,
        outer_radius=0.75,
        inner_radius=0.0,
        smoothing_width=0.2,
        dtype=jnp.float64,
    )

    assert 0.0 < float(broad) < float(narrow) < 1.0

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxwind.momentum.metrics import (
    AxisMetric,
    mp5_interface_states,
    muscl_mc_interface_states,
)


jax.config.update("jax_enable_x64", True)


def _double_sided(count: int, length: float, strength: float) -> np.ndarray:
    """Faces clustered toward the mid plane from both sides."""

    parameter = np.linspace(-1.0, 1.0, count + 1)
    return 0.5 * length * (
        1.0 + np.tanh(strength * parameter) / np.tanh(strength)
    )


def _single_sided(count: int, length: float, strength: float) -> np.ndarray:
    """Faces clustered toward the lower boundary."""

    parameter = np.linspace(0.0, 1.0, count + 1)
    return length * (np.expm1(strength * parameter) / np.expm1(strength))


def _variable(metric: AxisMetric) -> AxisMetric:
    """Force a geometrically uniform axis down the variable-spacing path."""

    metric.uniform = False
    metric._derivative_stencil = metric._build_derivative_stencil()
    metric._reconstruction["left"] = metric._build_reconstruction_stencil(-2)
    metric._reconstruction["right"] = metric._build_reconstruction_stencil(-1)
    return metric


def test_uniform_spacing_recovers_the_classical_stencil_coefficients() -> None:
    metric = _variable(
        AxisMetric(
            np.linspace(0.0, 1.0, 9),
            axis=0,
            periodic=True,
            dtype=jnp.float64,
        )
    )
    interior = 4

    derivative = np.asarray(metric._derivative_stencil[1])[interior]
    left = np.asarray(metric._reconstruction["left"][1])[interior]
    right = np.asarray(metric._reconstruction["right"][1])[interior]

    assert np.allclose(derivative * 12.0 * metric.spacing, (1, -8, 0, 8, -1))
    assert np.allclose(left * 60.0, (2, -13, 47, 27, -3))
    # The mirrored placement of the same primitive-function reconstruction.
    assert np.allclose(right * 60.0, (-3, 27, 47, -13, 2))


def test_stretched_derivative_reproduces_a_constant_and_is_fourth_order() -> None:
    def error(count: int) -> float:
        faces = np.linspace(0.0, 2.0 * np.pi, count + 1)
        faces = faces + 0.35 * np.sin(faces)
        metric = AxisMetric(faces, axis=0, periodic=True, dtype=jnp.float64)
        assert not metric.uniform
        centers = np.asarray(metric.centers)
        field = jnp.asarray(np.sin(centers))[:, None, None]
        derivative = np.asarray(metric.derivative(field))[:, 0, 0]
        return float(np.max(np.abs(derivative - np.cos(centers))))

    coarse, fine = error(32), error(64)
    assert np.log2(coarse / fine) > 3.8

    faces = _single_sided(24, 1.0, 2.5)
    metric = AxisMetric(faces, axis=0, periodic=True, dtype=jnp.float64)
    constant = jnp.ones((24, 3, 2), dtype=jnp.float64)
    assert float(jnp.max(jnp.abs(metric.derivative(constant)))) < 1.0e-12


def test_stretched_adjoint_is_exact_in_the_width_weighted_inner_product() -> None:
    faces = np.linspace(0.0, 2.0 * np.pi, 33)
    faces = faces + 0.35 * np.sin(faces)
    metric = AxisMetric(faces, axis=0, periodic=True, dtype=jnp.float64)
    centers = np.asarray(metric.centers)
    widths = np.asarray(metric.widths)[:, None, None]
    first = jnp.asarray(np.sin(3.1 * centers))[:, None, None]
    second = jnp.asarray(np.cos(2.3 * centers))[:, None, None]

    operator = metric.negative_derivative_transpose(first)
    derivative = metric.derivative(second)

    assert np.isclose(
        float(jnp.sum(widths * second * operator)),
        -float(jnp.sum(widths * derivative * first)),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_wall_normal_axis_keeps_the_three_point_stencil() -> None:
    metric = AxisMetric(
        _single_sided(12, 1.0, 2.0),
        axis=0,
        periodic=False,
        dtype=jnp.float64,
        derivative_width=3,
    )
    centers = np.asarray(metric.centers)[:, None, None]
    quadratic = jnp.asarray(centers**2 + 0.3 * centers + 1.0)

    derivative = metric.derivative(quadratic)

    assert np.allclose(
        np.asarray(derivative)[1:-1],
        2.0 * centers[1:-1] + 0.3,
        atol=1.0e-9,
    )


def test_periodic_three_point_stencil_is_rejected() -> None:
    with pytest.raises(ValueError, match="bounded axes only"):
        AxisMetric(
            np.linspace(0.0, 1.0, 5),
            axis=0,
            periodic=True,
            dtype=jnp.float64,
            derivative_width=3,
        )


@pytest.mark.parametrize("scheme", ["mp5", "muscl-mc"])
@pytest.mark.parametrize("periodic", [True, False])
def test_variable_states_match_the_constant_spacing_kernel(
    scheme: str,
    periodic: bool,
) -> None:
    """The variable-spacing reconstruction is a generalization, not a rewrite."""

    count = 16
    metric = _variable(
        AxisMetric(
            np.linspace(0.0, 1.0, count + 1),
            axis=0,
            periodic=periodic,
            dtype=jnp.float64,
        )
    )
    reference = mp5_interface_states if scheme == "mp5" else muscl_mc_interface_states
    # A bounded axis shifts its stencil inward instead of clamping samples, so
    # only faces whose full stencil is interior can agree term by term.
    window = slice(None) if periodic else slice(2, count - 3)
    generator = np.random.default_rng(20240605)

    for _ in range(25):
        values = jnp.asarray(generator.standard_normal(count))
        expected = reference(values, axis=0, periodic=periodic)
        obtained = (
            metric._variable_mp5_states(values)
            if scheme == "mp5"
            else metric._variable_muscl_states(values)
        )
        for expected_side, obtained_side in zip(expected, obtained, strict=True):
            assert np.allclose(
                np.asarray(obtained_side)[window],
                np.asarray(expected_side)[window],
                rtol=0.0,
                atol=1.0e-13,
            )


@pytest.mark.parametrize("periodic", [True, False])
def test_stretched_muscl_states_stay_inside_the_neighbouring_cell_values(
    periodic: bool,
) -> None:
    faces = (
        _double_sided(24, 1.0, 1.5) if periodic else _single_sided(24, 1.0, 2.5)
    )
    metric = AxisMetric(faces, axis=0, periodic=periodic, dtype=jnp.float64)
    assert not metric.uniform
    centers = np.asarray(metric.centers)
    values = jnp.asarray(
        np.where(centers < 0.4, 1.0, 0.0) + 0.2 * np.sin(11.0 * centers)
    )

    left, right = metric.interface_states(values, "muscl-mc")
    neighbour = (
        jnp.roll(values, -1)
        if periodic
        else jnp.concatenate((values[1:], values[-1:]))
    )
    lower = jnp.minimum(values, neighbour)
    upper = jnp.maximum(values, neighbour)
    # A bounded axis returns one entry per cell, so its last entry is the far
    # wall.  That face has no neighbour and therefore no cell jump, which is
    # asserted separately below: it carries exactly no correction, so the states
    # reported there are never consumed.
    faces = slice(None) if periodic else slice(0, -1)

    assert bool(jnp.all(left[faces] >= lower[faces] - 1.0e-12))
    assert bool(jnp.all(left[faces] <= upper[faces] + 1.0e-12))
    assert bool(jnp.all(right[faces] >= lower[faces] - 1.0e-12))
    assert bool(jnp.all(right[faces] <= upper[faces] + 1.0e-12))
    # The Rusanov correction may not oppose the cell jump.
    assert bool(jnp.all((right - left) * (neighbour - values) >= -1.0e-12))
    if not periodic:
        assert float(jnp.abs(right[-1] - left[-1])) == 0.0


def test_bounded_muscl_uses_a_one_sided_slope_in_the_boundary_cell() -> None:
    """A zero slope there is first-order exactly where the wall model works.

    On a neutral logarithmic profile this reconstruction is exact everywhere, so
    the first interior face is the only place it produces any dissipation at
    all.  Holding the boundary slope at zero makes that one face deliver most of
    first-order upwind's dissipation, which near a wall competes with the
    modeled surface stress rather than correcting an oscillation.
    """

    kappa, ustar, roughness, length = 0.4, 0.425, 0.1, 1500.0
    faces = np.linspace(0.0, length, 41)
    metric = AxisMetric(faces, axis=0, periodic=False, dtype=jnp.float64)
    centers = np.asarray(metric.centers)
    profile = jnp.asarray((ustar / kappa) * np.log(centers / roughness))

    left, right = metric.interface_states(profile, "muscl-mc")
    jump = np.abs(np.asarray(right - left))
    cell_jump = np.abs(np.diff(np.asarray(profile)))
    relative = jump[:-1] / cell_jump

    # First-order upwind would be 1.0 at every face; a zero boundary slope gave
    # 0.63 at the first one.
    assert relative[0] < 0.2
    # Above the boundary the reconstruction is exact on a logarithmic profile.
    assert np.all(relative[1:] < 1.0e-12)


def test_diffusion_diagonal_matches_the_uniform_limit_and_drops_wall_faces() -> None:
    periodic = AxisMetric(
        np.linspace(0.0, 1.0, 9),
        axis=0,
        periodic=True,
        dtype=jnp.float64,
    )
    assert np.allclose(np.asarray(periodic.diffusion_diagonal), 2.0 / 0.125**2)

    bounded = AxisMetric(
        np.linspace(0.0, 1.0, 9),
        axis=0,
        periodic=False,
        dtype=jnp.float64,
    )
    diagonal = np.asarray(bounded.diffusion_diagonal)
    assert np.allclose(diagonal[1:-1], 2.0 / 0.125**2)
    assert np.allclose(diagonal[[0, -1]], 1.0 / 0.125**2)


def test_upper_face_flux_divergence_telescopes_over_cell_volumes() -> None:
    faces = _single_sided(10, 1.0, 2.0)
    metric = AxisMetric(faces, axis=0, periodic=False, dtype=jnp.float64)
    widths = np.asarray(metric.widths)
    flux = jnp.asarray(np.linspace(0.3, -0.2, 10))

    divergence = np.asarray(metric.upper_face_flux_divergence(flux))

    # Integrating the divergence leaves only the flux through the top face.
    assert np.isclose(float(np.sum(widths * divergence)), float(flux[-1]))

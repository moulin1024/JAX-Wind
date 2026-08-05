from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxwind_archiv.momentum.metrics import (
    AxisMetric,
    mp5_interface_states,
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


@pytest.mark.parametrize("periodic", [True, False])
def test_variable_states_match_the_constant_spacing_kernel(
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
    # A bounded axis shifts its stencil inward instead of clamping samples, so
    # only faces whose full stencil is interior can agree term by term.
    window = slice(None) if periodic else slice(2, count - 3)
    generator = np.random.default_rng(20240605)

    for _ in range(25):
        values = jnp.asarray(generator.standard_normal(count))
        expected = mp5_interface_states(values, axis=0, periodic=periodic)
        obtained = metric._variable_mp5_states(values)
        for expected_side, obtained_side in zip(expected, obtained, strict=True):
            assert np.allclose(
                np.asarray(obtained_side)[window],
                np.asarray(expected_side)[window],
                rtol=0.0,
                atol=1.0e-13,
            )


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


@pytest.mark.parametrize("periodic", [True, False])
def test_the_correction_is_dissipative_by_construction(
    periodic: bool,
) -> None:
    """Only the difference of the two states is consumed, as a Rusanov flux.

    A correction that opposes the cell jump injects variance rather than
    removing it, and one that exceeds the jump dissipates harder than first-order
    upwind. Both are excluded for MP5 on smooth data and on data with
    grid-scale structure, so an active scalar cannot pick up spurious buoyancy
    from the advection correction.
    """

    count = 24
    faces = (
        _double_sided(count, 1.0, 1.4)
        if periodic
        else _single_sided(count, 1.0, 2.2)
    )
    metric = AxisMetric(faces, axis=0, periodic=periodic, dtype=jnp.float64)
    centers = np.asarray(metric.centers)
    generator = np.random.default_rng(19940101)

    fields = [
        np.sin(9.0 * centers),                        # smooth
        np.where(centers < 0.5, 1.0, 0.0),            # a front
        263.0 + 4.0 * centers + 0.05 * np.sin(31.0 * centers),  # inversion-like
    ]
    fields.extend(generator.standard_normal(count) for _ in range(20))

    for values in fields:
        field = jnp.asarray(values)
        left, right = metric.interface_states(field)
        neighbour = (
            jnp.roll(field, -1)
            if periodic
            else jnp.concatenate((field[1:], field[-1:]))
        )
        cell_jump = neighbour - field
        correction = right - left
        scale = float(jnp.max(jnp.abs(cell_jump))) + 1.0e-30

        assert bool(jnp.all(correction * cell_jump >= -1.0e-12 * scale**2))
        assert bool(
            jnp.all(jnp.abs(correction) <= jnp.abs(cell_jump) + 1.0e-12 * scale)
        )

"""Conservation and positivity checks for the independent jet marcher."""

import numpy as np
import pytest

from tools.jet_streamfunction import (
    conductance,
    diffusion_rate,
    geometry,
    positive_step,
)


def test_positive_stiff_transaction_closes_integral_with_reaction_and_boundary():
    widths = np.geomspace(1e-4, 1, 51)
    values = np.geomspace(1e-12, 3, 51)[::-1]
    coefficients = np.geomspace(1e-5, 1e3, 51)
    source = np.linspace(0, 0.2, 51)
    sink = np.linspace(1, 10, 51)
    dx = 10.0
    initial = values.copy()
    result, exported = positive_step(
        values, coefficients, widths, 0.01, dx, source, sink
    )
    assert np.all(result > 0)
    np.testing.assert_array_equal(values, initial)
    change = np.sum(widths * (result - values))
    reaction = dx * np.sum(widths * (source - sink * result))
    assert abs(change + exported - reaction) < 5e-11
    np.testing.assert_allclose(
        (result - values) / dx,
        diffusion_rate(result, coefficients, widths, 0.01) + source - sink * result,
        atol=2e-10,
        rtol=2e-9,
    )


def test_uniform_stream_geometry_and_stationary_state():
    faces = np.r_[0, np.geomspace(1e-4, 3, 31)]
    u = np.full(31, 2.0)
    rf, rc = geometry(faces, u)
    np.testing.assert_allclose(rf, faces, atol=1e-15)
    np.testing.assert_allclose(rc, (faces[1:] + faces[:-1]) / 2)
    coefficients = conductance(faces, u, 0.2, 2.0, 0.2)
    result, export = positive_step(u, coefficients, np.diff(faces), 2.0, 0.01)
    np.testing.assert_allclose(result, u, rtol=5e-15)
    assert abs(export) < 1e-15


def test_front_receives_flux_without_clipping_or_mass_creation():
    faces = np.linspace(0, 1, 101)
    u = np.where((faces[:-1] + faces[1:]) / 2 < 0.4, 1.0, 1e-10)
    coefficients = conductance(faces, u, np.where(u > 0.5, 0.1, 1e-12), 1e-10, 1e-12)
    result, export = positive_step(u, coefficients, np.diff(faces), 1e-10, 0.01)
    assert result[40] > u[40] * 1e5
    assert abs(np.sum((result - u) * np.diff(faces)) + export) < 1e-15


@pytest.mark.parametrize("field", ["coefficients", "sink", "source"])
def test_negative_rates_reject(field):
    arguments = {
        "values": np.ones(3),
        "coefficients": np.ones(3),
        "widths": np.ones(3),
        "ambient": 0.0,
        "dx": 0.1,
        "sink": 0.0,
        "source": 0.0,
    }
    arguments[field] = np.full(3, -0.1)
    with pytest.raises(ValueError):
        positive_step(**arguments)

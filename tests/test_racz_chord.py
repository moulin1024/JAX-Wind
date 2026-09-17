"""Independent geometry quadrature and rejection of incompatible calibration."""

import numpy as np
import pytest
from racz_chord import (
    admissible,
    assess_chords,
    chord_cdf,
    fit_radius_squared,
    temporal_training_mask,
)


def geometric_samples(points=2000):
    d = np.repeat(np.geomspace(1e-6, 60e-6, 21), points)
    impact = np.tile((np.arange(points) + 0.5) / points * 2 - 1, 21)
    radius_um = np.sqrt(300 * np.log(d * 1e6) + 1200)
    length_m = 2 * radius_um * np.sqrt(1 - impact**2) * 1e-6
    # Size-dependent speed changes transit time, not recovered geometry.
    u = 5 + d * 1e6
    return d, u, length_m / u


def test_recovers_geometry_without_velocity_or_size_population_bias():
    d, u, t = geometric_samples()
    coefficients = fit_radius_squared(d, u, t)
    np.testing.assert_allclose(coefficients, [300, 1200], rtol=3e-7)
    assert admissible(coefficients, [0.5145e-6, 64.1e-6])
    check = assess_chords(coefficients, *geometric_samples(1700))
    assert check["cdf_distance"] < 0.001
    assert abs(check["relative_first_moment_error"]) < 2e-5
    assert abs(check["relative_second_moment_error"]) < 1e-6
    assert check["support_exceedances"] == 0
    np.testing.assert_allclose(fit_radius_squared(d, -u, t), coefficients)


def test_known_cdf_and_explicit_support_failures():
    np.testing.assert_allclose(chord_cdf([-1, 0, np.sqrt(3), 2, 3]), [0, 0, 0.5, 1, 1])
    check = assess_chords(
        [0, 1], np.ones(3) * 1e-6, np.ones(3), np.array([1, 2, 3]) * 1e-6
    )
    assert check["support_exceedances"] == 1
    assert check["support_exceedance_fraction"] == 1 / 3


def test_nonphysical_fit_is_retained_not_clipped():
    d = np.geomspace(1, 60, 100)
    length = np.sqrt(8 / 3 * (1200 - 100 * np.log(d)))
    coefficients = fit_radius_squared(d * 1e-6, np.ones(100), length * 1e-6)
    np.testing.assert_allclose(coefficients, [-100, 1200], rtol=1e-13)
    assert not admissible(coefficients, [1e-6, 60e-6])
    assert not admissible([300, 0], [0.5145e-6, 64.1e-6])
    with pytest.raises(ValueError, match="nonpositive"):
        assess_chords([300, 0], np.array([0.5e-6]), np.ones(1), np.ones(1))


def test_temporal_split_covers_endpoints_and_preserves_time_units():
    t = np.arange(9, dtype=float)
    expected = [True, True, False, False, True, True, False, False, False]
    np.testing.assert_array_equal(temporal_training_mask(t, 4), expected)
    np.testing.assert_array_equal(temporal_training_mask(t * 0.001 + 2, 4), expected)


@pytest.mark.parametrize("times,blocks", [([0, 0], 4), ([1, 0], 4), ([0, 1], 3)])
def test_bad_temporal_metadata_rejected(times, blocks):
    with pytest.raises(ValueError):
        temporal_training_mask(times, blocks)


def test_missing_or_invalid_geometry_inputs_rejected():
    with pytest.raises(ValueError):
        fit_radius_squared([1e-6, 1e-6], [1, 2], [1e-6, 2e-6])
    with pytest.raises(ValueError):
        fit_radius_squared([1e-6, 2e-6], [1, np.nan], [1e-6, 2e-6])


def test_constrained_fit_uses_exact_boundary_and_preserves_good_solution():
    from racz_chord import fit_nonnegative_radius_squared

    d, u, t = geometric_samples()
    np.testing.assert_allclose(
        fit_nonnegative_radius_squared(d, u, t, 0.5145e-6),
        fit_radius_squared(d, u, t),
        rtol=1e-14,
    )
    d = np.geomspace(1, 60, 100)
    target = 100 * np.log(d) - 10
    # Restrict observations to where the prescribed test radius is positive;
    # require the fitted model to remain nonnegative down to 0.5 micrometres.
    d, target = d[1:], target[1:]
    keep = target > 0
    d, target = d[keep], target[keep]
    x = np.log(d / 0.5)
    coefficients = fit_nonnegative_radius_squared(
        d * 1e-6, np.ones(len(d)), np.sqrt(8 / 3 * target) * 1e-6, 0.5e-6
    )
    expected_a = np.sum(x * target) / np.sum(x**2)
    np.testing.assert_allclose(
        coefficients, [expected_a, -expected_a * np.log(0.5)], rtol=1e-14
    )

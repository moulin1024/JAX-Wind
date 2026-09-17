"""Guard estimator selection semantics without relying on spray targets."""

import numpy as np
import pytest

from tools.racz_gas_proxy import estimate_gas_proxy, scaled_mad_inliers

PARAMETERS = {
    "density": 18.0,
    "viscosity": 1.0,
    "length_scale": 1e-11,
    "stokes_limit": 0.6,
    "minimum_diameter": 0.0,
}


def test_large_accidentally_comoving_droplet_does_not_reenter_filter():
    result = estimate_gas_proxy(
        np.array([1.0, 2.0, 3.0]) * 1e-6, np.array([[10.0], [0.0], [5.0]]), **PARAMETERS
    )
    # Initial mean=5: the largest droplet has Stk=0, but an intermediate
    # diameter violates the threshold. A pointwise Stokes mask would keep it.
    np.testing.assert_array_equal(result.selected, [True, False, False])
    assert result.mean_speed == 10.0
    assert [r["count"] for r in result.history] == [3, 1]


def test_equal_diameter_ties_are_selected_by_stokes_as_well_as_size():
    result = estimate_gas_proxy(
        np.array([1.0, 2.0, 2.0, 3.0]) * 1e-6,
        np.array([[10.0], [0.0], [5.0], [5.0]]),
        **PARAMETERS,
    )
    assert [r["count"] for r in result.history] == [4, 2, 1]
    assert result.history[-1]["maximum_stokes"] <= PARAMETERS["stokes_limit"]


def test_scaled_mad_and_wavelength_filters_preserve_original_record_mapping():
    d = np.array([0.1, 1.0, 2.0, 3.0, 4.0, 5.0, 100.0]) * 1e-6
    result = estimate_gas_proxy(
        d, np.ones((7, 2)), density=997.0, viscosity=1.8691e-5, length_scale=0.012
    )
    np.testing.assert_array_equal(
        result.preprocessed, [False, True, True, True, True, True, False]
    )
    np.testing.assert_array_equal(result.selected, result.preprocessed)
    np.testing.assert_allclose(result.mean_speed, np.sqrt(2))
    assert result.last_cutoff_m is None


def test_zero_speed_and_degenerate_mad_are_finite_limits():
    np.testing.assert_array_equal(
        scaled_mad_inliers(np.array([2.0, 2.0, 2.0, 8.0])), [True, True, True, False]
    )
    result = estimate_gas_proxy(np.ones(10) * 1e-6, np.zeros((10, 2)), **PARAMETERS)
    assert result.mean_speed == 0 and np.all(result.selected)


def test_empty_invalid_and_unphysical_parameters_are_rejected():
    with pytest.raises(ValueError, match="No valid"):
        estimate_gas_proxy(np.array([-1.0, np.nan]), np.zeros((2, 2)), **PARAMETERS)
    with pytest.raises(ValueError, match="positive"):
        estimate_gas_proxy(
            np.ones(2), np.ones((2, 2)), **(PARAMETERS | {"length_scale": 0})
        )

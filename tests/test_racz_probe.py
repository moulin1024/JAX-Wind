"""Axial geometric support conditions and explicit generous bounds."""

import numpy as np
import pytest

from tools.racz_probe import axial_support_ratios


def test_slab_chord_condition_independent_of_unmeasured_velocity():
    # Fixed axial endpoints: transverse motion cannot lengthen axial support.
    r = axial_support_ratios(
        [20, -20, 0], [3e-6, 4e-6, 20e-6], 10e-6, axial_extent=73e-6
    )
    np.testing.assert_allclose(r, [60 / 73, 80 / 73, 0])
    np.testing.assert_array_equal(r > 1, [False, True, False])


def test_finite_particle_overlap_and_generous_uncertainty():
    nominal = axial_support_ratios(
        20, 5e-6, 20e-6, axial_extent=73e-6, finite_size=True
    )
    generous = axial_support_ratios(
        20,
        5e-6,
        20e-6,
        axial_extent=73e-6,
        finite_size=True,
        velocity_allowance=1.92,
        diameter_allowance=0.5e-6,
    )
    assert nominal > 1 and generous < 1
    np.testing.assert_allclose(generous, 90.4 / 93.5)


def test_unit_rescaling_and_reversed_velocity():
    a = axial_support_ratios(-12, 8e-6, 30e-6, axial_extent=73e-6, finite_size=True)
    b = axial_support_ratios(12e3, 8e-6, 30e-3, axial_extent=73e-3, finite_size=True)
    np.testing.assert_allclose(a, b)


@pytest.mark.parametrize(
    "bad",
    [
        {"transit": -1},
        {"diameter": 0},
        {"axial_extent": 0},
        {"velocity_allowance": -1},
        {"axial_velocity": np.nan},
    ],
)
def test_invalid_support_not_silently_clipped(bad):
    kwargs = {
        "axial_velocity": 1.0,
        "transit": 1e-6,
        "diameter": 2e-6,
        "axial_extent": 73e-6,
    }
    kwargs.update(bad)
    with pytest.raises(ValueError):
        axial_support_ratios(**kwargs)

"""Known velocity-bias example and time-correlated sampling diagnostics."""

import numpy as np
import pytest

from tools.racz_sampling import block_summary, event_moments, statistics


def records(u, d=None, transit=None):
    u = np.asarray(u, dtype=float)
    a = np.zeros((len(u), 8))
    a[:, 0] = np.arange(len(u))
    a[:, 1] = np.arange(len(u)) * 0.01
    a[:, 2] = 1e-6 if transit is None else transit
    a[:, 3] = u
    a[:, 4] = -2 * u
    a[:, 7] = 2e-6 if d is None else d
    return a


def test_known_speed_biased_joint_distribution():
    # Equal true concentrations of 1 and 3 m/s; event rates proportional to u.
    # Equal path length gives transit weights proportional to inverse u.
    a = records(
        [1, 3, 3, 3],
        np.array([1, 3, 3, 3]) * 1e-6,
        np.array([1, 1 / 3, 1 / 3, 1 / 3]) * 1e-6,
    )
    raw, _ = event_moments(a, "event")
    residence, _ = event_moments(a, "transit_time")
    np.testing.assert_allclose(
        statistics(raw.sum(axis=0)),
        [2.5, -5, 2.5e-6, 82 / 28 * 1e-6, np.sqrt(0.75)],
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        statistics(residence.sum(axis=0)), [2, -4, 2e-6, 2.8e-6, 1], rtol=1e-14
    )


def test_common_transit_scale_and_time_shift_invariant():
    a = records(np.linspace(1, 10, 50), transit=np.linspace(0.1, 3, 50) * 1e-6)
    shifted = a.copy()
    shifted[:, 1] += 8
    shifted[:, 2] *= 1e6
    x = block_summary(a, "transit_time", 8, 256, 31)
    y = block_summary(shifted, "transit_time", 8, 256, 31)
    for name in x["statistics"]:
        for key in x["statistics"][name]:
            np.testing.assert_allclose(
                x["statistics"][name][key],
                y["statistics"][name][key],
                rtol=1e-12,
                atol=1e-14,
            )
    assert x["block_event_counts"] == y["block_event_counts"]


def test_correlated_bursts_have_wider_block_than_iid_uncertainty():
    u = np.repeat([10.0, 20.0] * 4, 100)
    a = records(u)
    result = block_summary(a, "event", 8, 4096, 123)
    block_std = result["statistics"]["axial_mean_m_s"]["bootstrap_std"]
    naive = np.std(u, ddof=1) / np.sqrt(len(u))
    assert block_std > 8 * naive
    assert result["statistics"]["axial_mean_m_s"]["estimate"] == 15
    assert result["block_event_counts"] == [100] * 8


def test_constant_signal_and_empty_block_reporting():
    a = records(np.full(200, 7.0))
    r = block_summary(a, "event", 16, 256, 14)
    s = r["statistics"]["axial_mean_m_s"]
    assert s["bootstrap_low"] == s["estimate"] == s["bootstrap_high"] == 7
    empty = block_summary(records([1, 2]), "event", 8, 2048, 14)
    assert empty["empty_blocks"] == 6
    assert empty["empty_bootstrap_draws"] > 0
    assert sum(empty["block_event_counts"]) == 2


@pytest.mark.parametrize(
    "fault", ["negative_transit", "negative_diameter", "unsorted", "nonfinite"]
)
def test_invalid_records_are_rejected_without_silent_filtering(fault):
    a = records([1, 2, 3])
    if fault == "negative_transit":
        a[0, 2] = -1
    elif fault == "negative_diameter":
        a[0, 7] = -1
    elif fault == "unsorted":
        a[[0, 1], 1] = a[[1, 0], 1]
    else:
        a[0, 3] = np.nan
    with pytest.raises(ValueError):
        block_summary(a, "event", 8, 128, 13)

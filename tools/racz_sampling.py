"""Joint PDA statistics and conditional time-block bootstrap diagnostics."""

import numpy as np

OBSERVABLES = (
    "axial_mean_m_s",
    "transverse_mean_m_s",
    "D10_m",
    "D32_m",
    "axial_std_m_s",
)


def event_moments(samples, weighting):
    a = np.asarray(samples, dtype=float)
    if a.ndim != 2 or a.shape[1] != 8 or len(a) < 2 or not np.all(np.isfinite(a)):
        raise ValueError("expected finite PDA records")
    if np.any(a[:, 2] <= 0) or np.any(a[:, 7] <= 0):
        raise ValueError("transit time and diameter must be positive")
    if weighting not in ("event", "transit_time"):
        raise ValueError("unknown weighting")
    weights = np.ones(len(a)) if weighting == "event" else a[:, 2] / np.mean(a[:, 2])
    u, v, d = a[:, 3], a[:, 4], a[:, 7]
    # Common weights and block membership retain measured joint dependence.
    moments = weights[:, None] * np.column_stack(
        (np.ones(len(a)), u, v, d, d**2, d**3, u**2)
    )
    return moments, weights


def statistics(moments):
    q = np.asarray(moments)
    if np.any(q[..., 0] <= 0) or np.any(q[..., 4] <= 0):
        raise ValueError("empty moment inventory")
    mean = q[..., 1] / q[..., 0]
    variance = q[..., 6] / q[..., 0] - mean**2
    return np.stack(
        (
            mean,
            q[..., 2] / q[..., 0],
            q[..., 3] / q[..., 0],
            q[..., 5] / q[..., 4],
            np.sqrt(np.maximum(variance, 0)),
        ),
        axis=-1,
    )


def block_summary(
    samples, weighting, blocks, replicates, seed, percentiles=(2.5, 97.5)
):
    """Conditional stationary-block resampling; no instrument-bias correction.

    Block boundaries span observed first/last arrival, including empty blocks.
    The endpoint goes into the final block. Acquisition-window censoring and
    dependence longer than a block are outside this diagnostic.
    """
    moments, weights = event_moments(samples, weighting)
    times = np.asarray(samples)[:, 1]
    if np.any(np.diff(times) < 0) or times[-1] <= times[0] or times[0] < 0:
        raise ValueError("nondecreasing arrival times with positive span required")
    if blocks < 2 or replicates < 2:
        raise ValueError("at least two blocks and replicates required")
    width = (times[-1] - times[0]) / blocks
    index = np.minimum(((times - times[0]) / width).astype(int), blocks - 1)
    sums = np.zeros((blocks, moments.shape[1]))
    np.add.at(sums, index, moments)
    counts = np.bincount(index, minlength=blocks)
    rng = np.random.default_rng(seed)
    selected = rng.integers(blocks, size=(replicates, blocks))
    draws = sums[selected].sum(axis=1)
    nonempty = (draws[:, 0] > 0) & (draws[:, 4] > 0)
    if np.count_nonzero(nonempty) < 2:
        raise ValueError("too few nonempty bootstrap draws")
    estimates = statistics(draws[nonempty])
    original = statistics(sums.sum(axis=0))
    interval = np.percentile(estimates, percentiles, axis=0)
    return {
        "observed_span_s": float(times[-1] - times[0]),
        "block_duration_s": float(width),
        "block_count": blocks,
        "empty_blocks": int(np.sum(counts == 0)),
        "empty_bootstrap_draws": int(np.sum(~nonempty)),
        "block_event_counts": counts.tolist(),
        "weight_concentration_count": float(weights.sum() ** 2 / np.sum(weights**2)),
        "statistics": {
            name: {
                "estimate": float(original[i]),
                "bootstrap_low": float(interval[0, i]),
                "bootstrap_high": float(interval[1, i]),
                "bootstrap_std": float(np.std(estimates[:, i], ddof=1)),
            }
            for i, name in enumerate(OBSERVABLES)
        },
    }

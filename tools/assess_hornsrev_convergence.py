"""Compare two late averaging windows of a neutral Horns Rev warmup."""
from pathlib import Path
import argparse
import json
import numpy as np


def load(root):
    return (
        json.loads((root / "summary.json").read_text()),
        np.atleast_1d(np.genfromtxt(root / "profiles.csv", delimiter=",", names=True)),
        np.atleast_1d(np.genfromtxt(root / "history.csv", delimiter=",", names=True)),
    )


def time_mean(history, key, start, end):
    t = history["time_hours"]
    values = history[key]
    good = np.isfinite(t) & np.isfinite(values)
    t, values = t[good], values[good]
    if not len(t) or t[0] > start + 0.1 or t[-1] < end - 1e-5:
        raise ValueError(f"History does not cover the requested window for {key}")
    inside = (t > start) & (t < end)
    tx = np.r_[start, t[inside], end]
    vx = np.interp(tx, t, values)
    return float(np.trapezoid(vx, tx) / (end - start))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("older", type=Path)
    parser.add_argument("newer", type=Path)
    parser.add_argument("--older-window", type=float, nargs=2, required=True)
    parser.add_argument("--newer-window", type=float, nargs=2, required=True)
    parser.add_argument("--history", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    sa, a, ha = load(args.older)
    sb, b, hb = load(args.newer)
    if not np.array_equal(a["z_m"], b["z_m"]):
        raise ValueError("Profile meshes differ")
    z = b["z_m"]
    height = z[-1] + .5 * (z[1] - z[0])
    us = .4 * 8 / np.log(70 / .0002)
    lower = (z >= 20) & (z <= 100)
    bulk = (z / height >= .05) & (z / height <= .9)
    mean_tke_a = time_mean(ha, "integrated_resolved_tke_m3_s2", *args.older_window)
    mean_tke_b = time_mean(hb, "integrated_resolved_tke_m3_s2", *args.newer_window)
    ua, ub = sa["runtime"]["ustar_m_s"], sb["runtime"]["ustar_m_s"]
    huba, hubb = (float(np.interp(70, v["z_m"], v["mean_u_m_s"])) for v in (a, b))
    stress = -b["total_uw_m2_s2"] / us**2
    ti = hb["time_hours"]
    tail = (ti >= args.newer_window[0]) & (ti <= args.newer_window[1])
    slope = float(np.polyfit(ti[tail], hb["integrated_resolved_tke_m3_s2"][tail], 1)[0])
    result = {
        "older_window_hours": args.older_window,
        "newer_window_hours": args.newer_window,
        "final_window_fraction_of_total_time": (args.newer_window[1] - args.newer_window[0]) / args.newer_window[1],
        "older_mean_integrated_tke_m3_s2": mean_tke_a,
        "newer_mean_integrated_tke_m3_s2": mean_tke_b,
        "mean_tke_change_percent": 100 * (mean_tke_b / mean_tke_a - 1),
        "newer_tke_relative_slope_percent_per_hour": 100 * slope / mean_tke_b,
        "ustar_change_percent": 100 * (ub / ua - 1),
        "ustar_relative_to_target_percent": 100 * (ub / us - 1),
        "hub_velocity_change_percent": 100 * (hubb / huba - 1),
        "hub_velocity_m_s": hubb,
        "profile_change_rmse_20_100m_m_s": float(np.sqrt(np.mean((b["mean_u_m_s"][lower] - a["mean_u_m_s"][lower])**2))),
        "profile_change_rmse_full_height_m_s": float(np.sqrt(np.mean((b["mean_u_m_s"] - a["mean_u_m_s"])**2))),
        "normalized_total_stress_rmse_0p05_0p9H": float(np.sqrt(np.mean((stress[bulk] - (1 - z[bulk] / height))**2))),
        "final_maximum_divergence_s": float(hb["maximum_divergence_s"][-1]),
    }
    # Practical screening thresholds, not a proof of statistical stationarity.
    result["working_stationarity_checks"] = {
        "mean_tke_change_below_3_percent": abs(result["mean_tke_change_percent"]) < 3,
        "tke_trend_below_5_percent_per_hour": abs(result["newer_tke_relative_slope_percent_per_hour"]) < 5,
        "ustar_change_below_1_percent": abs(result["ustar_change_percent"]) < 1,
        "hub_velocity_change_below_1_percent": abs(result["hub_velocity_change_percent"]) < 1,
        "lower_profile_change_below_0p05_m_s": result["profile_change_rmse_20_100m_m_s"] < .05,
        "bulk_stress_rmse_below_0p1": result["normalized_total_stress_rmse_0p05_0p9H"] < .1,
    }
    result["all_working_checks_pass"] = all(result["working_stationarity_checks"].values())
    (args.newer / "convergence.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for root in [*args.history, args.older, args.newer]:
        h = np.atleast_1d(np.genfromtxt(root / "history.csv", delimiter=",", names=True))
        axes[0].plot(h["time_hours"], h["integrated_resolved_tke_m3_s2"], color="C0")
    axes[0].set(xlabel="Simulated time [h]", ylabel="Integrated resolved TKE [m³/s²]")
    for window, color in ((args.older_window, "C1"), (args.newer_window, "C2")):
        axes[0].axvspan(*window, color=color, alpha=.15)
    for data, label in ((a, f"{args.older_window[0]:g}–{args.older_window[1]:g} h"), (b, f"{args.newer_window[0]:g}–{args.newer_window[1]:g} h")):
        axes[1].plot(data["mean_u_m_s"], z, label=label)
        axes[2].plot(-data["total_uw_m2_s2"] / us**2, z / height, label=label)
    axes[1].set(xlabel="Mean U [m/s]", ylabel="Height [m]", ylim=(0, 200))
    axes[2].plot(1 - z / height, z / height, "k--", label="Equilibrium")
    axes[2].set(xlabel="Normalized total stress", ylabel="z/H")
    for ax in axes:
        ax.grid(alpha=.25)
    axes[1].legend()
    axes[2].legend()
    fig.savefig(args.newer / "convergence.png", dpi=160)
    plt.close(fig)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

"""Plot neutral warmup profiles and spin-up history from the shared runtime."""
from pathlib import Path
import argparse
import json
import tomllib

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--averaging-window", type=float, nargs=2, metavar=("START_H", "END_H"))
    args = parser.parse_args()
    root = args.directory
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = json.loads((root / "summary.json").read_text())
    configuration = tomllib.loads((root / "resolved_case.toml").read_text())
    mesh = configuration["mesh"]
    spacing = [length / cells for length, cells in zip(mesh["lengths_m"], mesh["cells"], strict=True)]
    spacing_label = " × ".join(f"{value:g}" for value in spacing)
    profile = np.atleast_1d(np.genfromtxt(root / "profiles.csv", delimiter=",", names=True))
    history = np.atleast_1d(np.genfromtxt(root / "history.csv", delimiter=",", names=True))
    z = profile["z_m"]
    dz = float(z[1] - z[0])
    height = float(z[-1] + dz / 2)
    kappa, z0 = 0.4, 0.0002
    target_ustar = kappa * 8.0 / np.log(70.0 / z0)
    measured_ustar = summary["runtime"]["ustar_m_s"]
    lo, hi = z - dz / 2, z + dz / 2
    def primitive(x):
        return x * (np.log(np.maximum(x, np.finfo(float).tiny) / z0) - 1)
    log_shape = (primitive(hi) - primitive(lo)) / dz / kappa
    reference = target_ustar * log_shape
    diagnosed_reference = measured_ustar * log_shape
    u = profile["mean_u_m_s"]
    faces = 0.5 * (z[1:] + z[:-1])
    shear = kappa * faces / target_ustar * np.diff(u) / dz
    reference_shear = kappa * faces / target_ustar * np.diff(reference) / dz
    window = (z >= 20) & (z <= 100)
    shear_window = (faces >= 20) & (faces <= 100)
    stress = -profile["total_uw_m2_s2"] / target_ustar**2
    expected_stress = 1 - z / height
    times = history["time_hours"]
    tke = history["integrated_resolved_tke_m3_s2"]
    valid = np.isfinite(times) & np.isfinite(tke)
    tail = valid & (times >= times[valid].max() - 1 / 6)
    relative_drift = None
    if np.count_nonzero(tail) >= 2 and np.mean(tke[tail]) > 0:
        relative_drift = float(np.polyfit(times[tail], tke[tail], 1)[0] / np.mean(tke[tail]))
    metrics = {
        "assessment_window_m": [20, 100],
        "target_ustar_m_s": float(target_ustar),
        "sampled_ustar_m_s": float(measured_ustar),
        "hub_velocity_interpolated_m_s": float(np.interp(70, z, u)),
        "loglaw_rmse_20_100m_m_s": float(np.sqrt(np.mean((u[window] - reference[window])**2))),
        "loglaw_rmse_measured_ustar_20_100m_m_s": float(np.sqrt(np.mean((u[window] - diagnosed_reference[window])**2))),
        "shear_ratio_mean_20_100m": float(np.mean(shear[shear_window] / reference_shear[shear_window])),
        "normalized_total_stress_rmse_20_100m": float(np.sqrt(np.mean((stress[window] - expected_stress[window])**2))),
        "last_10min_tke_relative_slope_per_hour": relative_drift,
        "final_time_seconds": summary["time_seconds"],
        "profile_samples": summary["runtime"]["profile_samples"],
        "caution": "A near-logarithmic mean alone does not establish equilibrium; assess stress and time stationarity independently.",
    }
    if args.averaging_window:
        start, end = args.averaging_window
        if not 0 <= start < end or not np.isclose(end * 3600, summary["time_seconds"]):
            raise ValueError("Averaging window must end at the final simulation time")
        metrics["averaging_window_hours"] = [start, end]
        metrics["averaging_fraction_of_total_time"] = (end - start) / end
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(reference, z, "--", label="Cell-average log law: target u*")
    ax.plot(diagnosed_reference, z, ":", label="Cell-average log law: measured u*")
    ax.plot(u, z, "o-", ms=3, label="Time/plane average")
    ax.set(xlabel="Mean U [m/s]", ylabel="Height [m]", ylim=(0, 200))
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    ax.plot(shear, faces, "o-", ms=3, label="Simulation")
    ax.plot(reference_shear, faces, "--", label="Discrete log-law reference")
    ax.set(xlabel="Normalized shear κz/U* · dU/dz (target U*)", ylabel="Height [m]", ylim=(0, 200))
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    ax.plot(stress, z / height, label="Resolved + SGS")
    ax.plot(-profile["resolved_uw_m2_s2"] / target_ustar**2, z / height, ":", label="Resolved")
    ax.plot(expected_stress, z / height, "--", label="Equilibrium 1 − z/H")
    ax.set(xlabel=r"$-\langle u'w'\rangle / u_{*,target}^2$", ylabel="z/H")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    ax.plot(times[valid], tke[valid])
    ax.set(xlabel="Simulated time [h]", ylabel="Integrated resolved TKE [m³/s²]")
    for ax in axes.flat:
        ax.grid(alpha=0.25)
    title = f"Horns Rev 1 neutral warmup — {spacing_label} m cells"
    if args.averaging_window:
        title += f"; average {start:g}–{end:g} h"
    fig.suptitle(title)
    fig.savefig(root / "loglaw_assessment.png", dpi=160)
    plt.close(fig)
    np.savetxt(
        root / "loglaw_profile.csv",
        np.column_stack((z, u, profile["mean_v_m_s"], profile["mean_w_m_s"],
                         reference, diagnosed_reference, u - reference,
                         profile["total_uw_m2_s2"], profile["resolved_tke_m2_s2"])),
        delimiter=",", comments="",
        header=("z_m,mean_u_m_s,mean_v_m_s,mean_w_m_s,"
                "loglaw_target_ustar_m_s,loglaw_measured_ustar_m_s,"
                "u_minus_target_loglaw_m_s,total_uw_m2_s2,resolved_tke_m2_s2"),
    )
    (root / "loglaw_metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

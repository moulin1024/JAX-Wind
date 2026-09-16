"""Plot each turbine's RPM, local wind, TSR, and optional lookup power."""
import argparse
from pathlib import Path
import tomllib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    case = tomllib.loads((args.directory / "resolved_case.toml").read_text())
    farm = case["physics"]["wind_farm"]
    history = np.atleast_1d(np.genfromtxt(args.directory / "history.csv", delimiter=",", names=True, deletechars=""))
    t = history["time_hours"] * 3600.
    lookup = farm["controller"]["model"] == "wind-speed-lookup"
    fig, axes = plt.subplots(4 if lookup else 3, 1, figsize=(10, 12 if lookup else 9), sharex=True, constrained_layout=True)
    for row in farm["layout"]:
        prefix = "turbine_" + row["id"] + "_"
        line, = axes[0].plot(t, history[prefix+"rpm"], label=row["id"])
        color = line.get_color()
        axes[0].plot(t, history[prefix+"target_rpm"], "--", color=color, alpha=.7)
        axes[1].plot(t, history[prefix+"probe_wind_m_s"], color=color, alpha=.4)
        axes[1].plot(t, history[prefix+"filtered_wind_m_s"], color=color, label=row["id"])
        valid = history[prefix+"tsr_valid"] > 0
        axes[2].plot(t, np.where(valid, history[prefix+"tsr"], np.nan), color=color, label=row["id"])
        if lookup:
            axes[3].plot(t, history[prefix+"lookup_power_w"] / 1.e6, color=color, label=row["id"])
    axes[0].set(ylabel="Rotor speed [rpm]", title="Solid: actual RPM; dashed: filtered-wind target")
    axes[1].set(ylabel="Upstream wind [m/s]", title="Solid: filtered; faint: measured rotor-area probe")
    if not lookup:
        axes[2].axhline(farm["controller"]["target_tsr"], color="black", ls="--", label="Target TSR")
    axes[2].set(ylabel="TSR (filtered wind)", xlabel="Simulation time [s]")
    if lookup:
        axes[2].set_xlabel("")
        axes[3].plot(t, history["farm_lookup_power_w"] / 1.e6, "k--", label="Farm total")
        axes[3].set(ylabel="Lookup power [MW]", xlabel="Simulation time [s]")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(loc="best")
    fig.suptitle("Independent turbine control — " + ("RPM/power lookup" if lookup else "idealized TSR speed servos"))
    output = args.directory / "turbine_control.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()

"""Plot the averaged Horns Rev wind profile with logarithmic height."""
import argparse
from pathlib import Path
import json
import tomllib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    configuration = tomllib.loads((args.directory / "resolved_case.toml").read_text())
    metrics = json.loads((args.directory / "loglaw_metrics.json").read_text())
    mesh = configuration["mesh"]
    spacing = [length / cells for length, cells in zip(mesh["lengths_m"], mesh["cells"], strict=True)]
    spacing_label = " × ".join(f"{value:g}" for value in spacing)
    window = metrics.get("averaging_window_hours")
    mean_label = f"LES mean, {window[0]:g}–{window[1]:g} h" if window else "LES mean"
    fraction = metrics.get("averaging_fraction_of_total_time")
    averaging_label = f"final {100 * fraction:g}% average" if fraction is not None else "time average"
    data = np.genfromtxt(args.directory / "loglaw_profile.csv", delimiter=",", names=True)
    z = data["z_m"]
    selected = z <= 200
    fig, ax = plt.subplots(figsize=(6.5, 6), constrained_layout=True)
    ax.semilogy(data["mean_u_m_s"][selected], z[selected], "o-", color="#176b9b",
                lw=2, ms=4, label=mean_label)
    ax.semilogy(data["loglaw_target_ustar_m_s"][selected], z[selected], "--",
                color="#cf6a20", lw=2, label="Cell-averaged log law")
    ax.axhline(70, color="0.6", lw=1, ls=":")
    ax.text(.98, 70, "70 m hub height", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", color="0.4", fontsize=9)
    ax.set(xlabel="Mean streamwise velocity U [m/s]", ylabel="Height z [m] — logarithmic scale",
           ylim=(z[0], 200), title=f"Horns Rev 1: mean wind profile\n{spacing_label} m cells · {averaging_label}")
    ax.set_yticks([tick for tick in [2, 4, 8, 16, 32, 70, 100, 200] if tick >= z[0]])
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.grid(True, which="both", alpha=.2)
    ax.legend(loc="upper left", frameon=False)
    ax.text(.98, .025, r"$z_0=0.0002$ m; target $u_*=0.25067$ m/s",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9)
    for extension in ("png", "pdf"):
        fig.savefig(args.directory / f"log_profile.{extension}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()

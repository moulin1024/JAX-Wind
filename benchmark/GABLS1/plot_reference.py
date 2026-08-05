"""Overlay a unified-runner GABLS1 profile with the official LES ensemble."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

from benchmark.GABLS1.reference import ensemble_on_grid, load_period_sets


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=HERE / "reference" / "official_12p5m",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _ensemble(reference_dir: Path, set_name: str, z: np.ndarray) -> dict:
    return ensemble_on_grid(load_period_sets(reference_dir, set_name, period=9), "z", z)


def main() -> None:
    args = parse_args()
    profile_path = args.result_dir / "profiles.csv"
    profile = np.genfromtxt(profile_path, delimiter=",", names=True)
    z = np.asarray(profile["z_m"])
    reference = {
        set_name: _ensemble(args.reference_dir, set_name, z)
        for set_name in "ABC"
    }

    output = args.output or args.result_dir / "GABLS1_AMD_12p5m_reference_overlay.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 4, figsize=(15, 11), constrained_layout=True)

    panels = (
        ("A", "u_mean", "mean_u_m_s", r"$\langle u\rangle$ (m s$^{-1}$)"),
        ("A", "v_mean", "mean_v_m_s", r"$\langle v\rangle$ (m s$^{-1}$)"),
        ("A", "theta_mean", "mean_scalar", r"$\langle\theta\rangle$ (K)"),
        ("B", "u_var_resolved", "var_u_m2_s2", r"$\langle u'^2\rangle$ (m$^2$ s$^{-2}$)"),
        ("B", "v_var_resolved", "var_v_m2_s2", r"$\langle v'^2\rangle$ (m$^2$ s$^{-2}$)"),
        ("B", "w_var_resolved", "var_w_m2_s2", r"$\langle w'^2\rangle$ (m$^2$ s$^{-2}$)"),
        ("B", "theta_var_resolved", "var_scalar", r"$\langle\theta'^2\rangle$ (K$^2$)"),
        ("C", "uw_resolved", "resolved_uw_m2_s2", r"resolved $\langle u'w'\rangle$"),
        ("C", "vw_resolved", "resolved_vw_m2_s2", r"resolved $\langle v'w'\rangle$"),
        ("C", "wtheta_resolved", "resolved_wscalar", r"resolved $\langle w'\theta'\rangle$"),
    )

    for axis, (set_name, reference_name, model_name, xlabel) in zip(
        axes.flat, panels, strict=False
    ):
        ensemble = reference[set_name].get(reference_name)
        if ensemble is not None:
            axis.fill_betweenx(
                z,
                ensemble["minimum"],
                ensemble["maximum"],
                color="0.84",
                label="official 12.5 m range",
            )
            axis.plot(ensemble["mean"], z, "k--", lw=1.4, label="official mean")
        axis.plot(profile[model_name], z, color="crimson", lw=2, label="JAX-Wind AMD")
        axis.set(xlabel=xlabel, ylabel="z (m)", ylim=(0.0, 400.0))
        axis.grid(alpha=0.25)

    extra = axes.flat[10]
    extra.plot(profile["mean_w_m_s"], z, color="crimson", lw=2)
    extra.axvline(0.0, color="k", ls="--", lw=1)
    extra.set(xlabel=r"$\langle w\rangle$ (m s$^{-1}$)", ylabel="z (m)", ylim=(0.0, 400.0))
    extra.grid(alpha=0.25)

    extra = axes.flat[11]
    extra.plot(profile["sgs_viscosity_m2_s"], z, color="crimson", lw=2)
    extra.set(xlabel=r"$\langle\nu_{sgs}\rangle$ (m$^2$ s$^{-1}$)", ylabel="z (m)", ylim=(0.0, 400.0))
    extra.grid(alpha=0.25)

    axes.flat[0].legend(fontsize=8, loc="best")
    figure.suptitle(
        "GABLS1, 8–9 h mean: JAX-Wind MP5 + AMD, 12.5 m vs official 12.5 m LES"
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)
    print(output)


if __name__ == "__main__":
    main()

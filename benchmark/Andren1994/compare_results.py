#!/usr/bin/env python3
"""Compare a completed MAC/LASD Andrén run with tabulated paper values."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REFERENCE = HERE / "reference" / "reference_results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = args.results / "summary.json"
    profile_path = args.results / "andren1994_profiles.csv"
    summary = json.loads(summary_path.read_text())
    reference = json.loads(REFERENCE.read_text())
    profile = np.genfromtxt(profile_path, delimiter=",", names=True)

    published = reference["ustar_over_ug"]
    labels = (
        "Andrén/Moeng",
        "Mason/Brown\nbackscatter",
        "Mason/Brown\nno backscatter",
        "Nieuwstadt",
        "Schumann/Graf",
    )
    paper_ratios = np.asarray(tuple(published.values()), dtype=float)
    final_jax_ratio = float(summary["friction_velocity_over_geostrophic"])
    domain_height = float(summary["domain_m"][2])
    nz = int(summary["shape_zyx"][0])
    roughness = float(summary["roughness_length_m"])
    geostrophic_speed = float(
        np.linalg.norm(summary["geostrophic_wind_m_s"])
    )
    first_cell_speed = math.hypot(
        float(profile["mean_u_m_s"][0]),
        float(profile["mean_v_m_s"][0]),
    )
    wall_factor = 0.4 / math.log(
        (0.5 * domain_height / nz) / roughness
    )
    window_jax_ratio = wall_factor * first_cell_speed / geostrophic_speed
    low = float(np.min(paper_ratios))
    high = float(np.max(paper_ratios))
    nearest = float(np.clip(window_jax_ratio, low, high))
    range_error = window_jax_ratio - nearest
    center = 0.5 * (low + high)
    center_relative_error = (window_jax_ratio - center) / center

    jax_tke = float(summary["normalized_integrated_resolved_tke"])
    paper_tke = float(reference["normalized_integrated_total_tke_plateau"])
    comparison = {
        "jaxwind_profile_based_window_mean_ustar_over_ug": (
            window_jax_ratio
        ),
        "jaxwind_final_instantaneous_ustar_over_ug": final_jax_ratio,
        "window_mean_ustar_note": (
            "Estimated from the magnitude of the time- and "
            "horizontally-averaged first-cell velocity using the same "
            "log-law wall model; the exact time mean of local ustar was "
            "not recorded."
        ),
        "paper_ustar_over_ug_range": [low, high],
        "distance_to_paper_range": range_error,
        "relative_error_from_paper_range_center": center_relative_error,
        "jaxwind_normalized_integrated_resolved_tke": jax_tke,
        "paper_normalized_integrated_total_tke": paper_tke,
        "tke_comparison_is_like_for_like": False,
        "statistics_window_ft": [
            float(summary["sample_start_ft"]),
            float(summary["end_ft"]),
        ],
    }
    (args.results / "reference_comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n"
    )

    with (args.results / "reference_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("metric", "jaxwind", "reference_low", "reference_high", "note"))
        writer.writerow(
            (
                "ustar_over_ug",
                window_jax_ratio,
                low,
                high,
                "profile-based window estimate; see JSON caveat",
            )
        )
        writer.writerow(
            (
                "final_instantaneous_ustar_over_ug",
                final_jax_ratio,
                low,
                high,
                "instantaneous endpoint; not used for range error",
            )
        )
        writer.writerow(
            (
                "normalized_integrated_tke",
                jax_tke,
                paper_tke,
                paper_tke,
                "JAX-Wind resolved only; paper resolved plus SGS",
            )
        )

    os.environ.setdefault("MPLCONFIGDIR", str(args.results / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15.0, 5.6))
    x = np.arange(len(labels) + 1)
    ratios = np.concatenate(
        (paper_ratios, np.asarray((window_jax_ratio,)))
    )
    colors = ["0.65"] * len(labels) + ["#c2185b"]
    axes[0].bar(x, ratios, color=colors)
    axes[0].axhspan(low, high, color="0.8", alpha=0.25)
    axes[0].set_xticks(
        x,
        (*labels, "JAX-Wind\nwindow estimate"),
        rotation=22,
        ha="right",
    )
    axes[0].set_ylabel(r"$u_*/U_g$")
    axes[0].set_title("Surface friction velocity")
    axes[0].set_ylim(0.0, max(0.05, 1.12 * np.max(ratios)))

    axes[1].bar(
        (0, 1),
        (jax_tke, paper_tke),
        color=("#c2185b", "0.65"),
    )
    axes[1].set_xticks(
        (0, 1),
        ("JAX-Wind\nresolved", "Paper plateau\nresolved + SGS"),
    )
    axes[1].set_ylabel(r"$f\int E\,dz/u_*^3$")
    axes[1].set_title("Integrated TKE\n(different accounting)")

    height = np.asarray(profile["zf_over_ustar"])
    axes[2].plot(profile["mean_u_m_s"] / 10.0, height, label=r"$U/U_g$")
    axes[2].plot(profile["mean_v_m_s"] / 10.0, height, label=r"$V/U_g$")
    axes[2].axhline(
        reference["normalized_top_height_approximate"],
        color="0.4",
        linestyle="--",
        label="paper approximate top",
    )
    axes[2].set_xlabel("mean velocity / $U_g$")
    axes[2].set_ylabel(r"$zf/u_*$")
    axes[2].set_title("Mean Ekman profile")
    axes[2].legend()

    for axis in axes:
        axis.grid(True, alpha=0.25)
    figure.suptitle(
        rf"Andrén 1994 comparison, averaged over "
        rf"${summary['sample_start_ft']:g}\leq ft\leq{summary['end_ft']:g}$"
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    figure.savefig(
        args.results / "andren1994_reference_comparison.png",
        dpi=200,
    )
    plt.close(figure)


if __name__ == "__main__":
    main()

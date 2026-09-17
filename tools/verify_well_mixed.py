#!/usr/bin/env python3
"""Finite-ensemble stationary tracer benchmark with an uncorrected control."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxwind.inhomogeneous_dispersion import well_mixed_tracer_step


def fields(x):
    q = 1 + 0.8 * jnp.cos(2 * jnp.pi * x[0])
    gradient = jnp.zeros_like(x).at[0].set(-1.6 * jnp.pi * jnp.sin(2 * jnp.pi * x[0]))
    return q, gradient, 0.05


def run(count, seed, dt, duration, corrected):
    key = jax.random.key(seed)
    key, kx, kw = jax.random.split(key, 3)
    x = jax.random.uniform(kx, (3, count), dtype=jnp.float64)
    w = jax.random.normal(kw, (3, count), dtype=jnp.float64)
    v = w if corrected else w * jnp.sqrt(fields(x)[0])
    initial = (x, w)

    def step(_, state):
        x, v, key = state
        key, noise_key = jax.random.split(key)
        noise = jax.random.normal(noise_key, (2, 3, count), dtype=jnp.float64)
        if corrected:
            x, v = well_mixed_tracer_step(x, v, dt, noise, fields)
        else:
            # Symmetric free-flight/local-OU/free-flight negative control in
            # physical velocity. No gradient drift and no resampling.
            middle = x + 0.5 * dt * v
            q, _, tau = fields(middle)
            a = jnp.exp(-dt / tau)
            v = a * v + jnp.sqrt(q * -jnp.expm1(-2 * dt / tau)) * noise[0]
            x = middle + 0.5 * dt * v
        return x % 1.0, v, key

    x, v, _ = jax.jit(
        lambda state: jax.lax.fori_loop(0, round(duration / dt), step, state)
    )((x, v, key))
    w = v if corrected else v / jnp.sqrt(fields(x)[0])
    return initial, (np.asarray(x), np.asarray(w))


def diagnostics(x, w, bins, gates):
    x = np.asarray(x)
    w = np.asarray(w)
    indices = np.floor(x[0] * bins).astype(int)
    counts = np.bincount(indices, minlength=bins)
    density = counts / (len(indices) / bins)
    means = np.array(
        [
            np.bincount(indices, weights=component, minlength=bins) / counts
            for component in w
        ]
    )
    variances = (
        np.array(
            [
                np.bincount(indices, weights=component**2, minlength=bins) / counts
                for component in w
            ]
        )
        - means**2
    )
    metrics = {
        "max_relative_bin_density_error": float(np.max(np.abs(density - 1))),
        "max_abs_conditional_normalized_velocity_mean": float(np.max(np.abs(means))),
        "max_conditional_normalized_velocity_variance_error": float(
            np.max(np.abs(variances - 1))
        ),
        "abs_first_position_fourier_moment": float(
            abs(np.mean(np.exp(2j * np.pi * x[0])))
        ),
    }
    return {
        **metrics,
        "pass": all(value <= gates[k] for k, value in metrics.items()),
        "density": density.tolist(),
        "conditional_mean": means.tolist(),
        "conditional_variance": variances.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    protocol_path = root / "cases/SprayClosureValidation/well_mixed_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    jax.config.update("jax_enable_x64", True)
    results = []
    controls = []
    for seed in protocol["seeds"]:
        for dt in protocol["timesteps_s"]:
            initial, final = run(
                protocol["particles"], seed, dt, protocol["duration_s"], True
            )
            result = {
                "seed": seed,
                "dt": dt,
                "initial": diagnostics(*initial, protocol["bins"], protocol["gates"]),
                "final": diagnostics(*final, protocol["bins"], protocol["gates"]),
            }
            results.append(result)
            print(
                f"Corrected seed={seed}, dt={dt}: density error={result['final']['max_relative_bin_density_error']:.5f}, pass={result['final']['pass']}",
                flush=True,
            )
        _, final = run(
            protocol["particles"],
            seed,
            min(protocol["timesteps_s"]),
            protocol["duration_s"],
            False,
        )
        control = {
            "seed": seed,
            "dt": min(protocol["timesteps_s"]),
            "final": diagnostics(*final, protocol["bins"], protocol["gates"]),
        }
        controls.append(control)
        print(
            f"Uncorrected seed={seed}: density error={control['final']['max_relative_bin_density_error']:.5f}",
            flush=True,
        )
    passed = all(r["final"]["pass"] and r["initial"]["pass"] for r in results) and all(
        c["final"]["max_relative_bin_density_error"]
        >= protocol["gates"]["negative_control_min_density_error"]
        for c in controls
    )
    paths = [
        protocol_path,
        Path(__file__),
        root / "src/jaxwind/inhomogeneous_dispersion.py",
        root / "tests/physics/test_inhomogeneous_dispersion.py",
        root / "tools/verify_well_mixed.sbatch",
    ]
    report = {
        "job_id": os.getenv("SLURM_JOB_ID"),
        "backend": jax.default_backend(),
        "protocol": protocol,
        "corrected": results,
        "uncorrected": controls,
        "numerical_screen_pass": passed,
        "physical_spray_validation": False,
        "sha256": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = (np.arange(protocol["bins"]) + 0.5) / protocol["bins"]
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for r in results:
        ax.plot(
            x,
            r["final"]["density"],
            label=f"corrected, seed {r['seed']}, dt {r['dt']}",
            alpha=0.7,
        )
    for r in controls:
        ax.plot(x, r["final"]["density"], "--", label=f"uncorrected, seed {r['seed']}")
    ax.axhline(1, color="k", linewidth=0.7)
    ax.set(xlabel="x (m)", ylabel="Particle density / uniform density")
    ax.legend(fontsize=7)
    fig.savefig(args.output / "density.png", dpi=160)
    plt.close(fig)
    rows = []
    for label, runs in [("corrected", results), ("uncorrected", controls)]:
        for r in runs:
            m = r["final"]
            rows.append(
                f"| {label} | {r['seed']} | {r['dt']} | {100 * m['max_relative_bin_density_error']:.2f}% | {m['max_abs_conditional_normalized_velocity_mean']:.4f} | {m['max_conditional_normalized_velocity_variance_error']:.4f} | {m['pass']} |"
            )
    text = "\n".join(
        [
            "# Inhomogeneous well-mixed tracer verification",
            "",
            f"Job {report['job_id']}; backend {report['backend']}; numerical screens pass: {passed}.",
            "",
            "| Model | Seed | dt | Max density error | Max conditional mean | Max variance error | Stationarity screens |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            *rows,
            "",
            "Constant-density tracers in prescribed stationary isotropic Gaussian turbulence. No resampling.",
            "This tests concentration and local velocity statistics, not finite-inertia droplets or experimental accuracy.",
            "Three timesteps and two independent seeds are reported; sampling error can mask small discretization error.",
            "",
        ]
    )
    (args.output / "report.md").write_text(text)
    print(text)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

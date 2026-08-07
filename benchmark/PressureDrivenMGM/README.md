# Pressure-driven MGM GPU benchmark

This benchmark runs the canonical neutral atmospheric boundary layer with the
modulated gradient model (MGM): a `2000 pi m x 2000 pi m x 1000 m` domain,
`64 x 64 x 64` cells, a `0.1 s` time step, and 10 simulated hours. Statistics
cover the final two hours. Its filtered neutral wall explicitly enables the
Porté-Agel (2000) first-interior-face shear correction.

From the repository root, run everything with one command on a GPU node:

```bash
python -m benchmark.PressureDrivenMGM
```

The repository's `external/bw1000_benchmark` pressure-solver submodule must be
present. A recursive clone already includes it; for an existing non-recursive
clone, initialize it once with `git submodule update --init --recursive`.

The command checks that JAX sees a GPU, automatically resumes
`checkpoint_latest.npz` when present, runs the shared
`runners/pressure_driven_warmup/config_mgm.toml` case, writes `run.log`, and
generates `loglaw_velocity_profile.svg` from `profiles.csv`. Results default to
`outputs/pressure_driven_mgm_64x64x64_gpu/`.

Override the configured timestep and simulated duration directly when needed:

```bash
python -m benchmark.PressureDrivenMGM --dt 0.2 --hours 2
```

An `--hours` override keeps profile sampling over the final 20% of the run.
Values edited directly in `config_mgm.toml` are also accepted.

Use `--restart PATH` to select a different checkpoint or `--plot-only` to
regenerate the figure without running JAX. To start an independent run, pass a
new `--output` directory. `--allow-cpu --max-steps N` is available only for
development smoke tests; the canonical default remains the complete GPU
benchmark.

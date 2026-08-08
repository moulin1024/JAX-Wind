# Pressure-driven MGM GPU benchmark

This benchmark runs the configured neutral atmospheric boundary layer with the
modulated gradient model (MGM). The main solver follows
the shared solver numerics: conservative convection with horizontal
three-halves padding, the Lu--Porté-Agel horizontal-plane dynamic MGM
coefficient, filtered log-law wall gradients, and the Porté-Agel (2000)
first-interior-face shear correction. Domain, timestep, duration, and sampling controls come from
`runners/pressure_driven_warmup/config_mgm.toml`.

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
new `--output` directory. Checkpoints carry the MGM physics fingerprint, so a
checkpoint produced by the former fixed-coefficient implementation is rejected
instead of silently mixing AB2 histories. `--allow-cpu --max-steps N` is
available for development smoke tests.

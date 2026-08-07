# Pressure-driven LASD benchmark

This benchmark runs the same pressure-driven neutral ABL case selected by
`runners/pressure_driven_warmup/config_mgm.toml`, changing only the SGS model
from MGM to the Lagrangian scale-dependent dynamic (LASD) closure. Domain,
resolution, timestep, duration, pressure forcing, wall filtering, and the
configured Porté-Agel correction remain identical to the MGM case.

Run it on a GPU node from the repository root with:

```bash
python -m benchmark.PressureDrivenLASD
```

Results default to `outputs/pressure_driven_lasd_gpu/`. The benchmark checks
for a GPU, automatically resumes its own checkpoint, records `run.log`, and
generates `loglaw_velocity_profile.svg` over the final 20% of simulated time.
The pressure-solver submodule must be initialized once with
`git submodule update --init --recursive`.

For a CPU run, explicitly select the CPU backend and permit it:

```bash
JAX_PLATFORMS=cpu python -m benchmark.PressureDrivenLASD --allow-cpu
```

`--dt`, `--hours`, `--max-steps`, `--restart`, `--output`, and `--plot-only`
match the MGM benchmark options.

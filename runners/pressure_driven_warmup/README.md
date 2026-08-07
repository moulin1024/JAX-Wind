# Pressure-driven neutral ABL

This case develops a full-scale neutral atmospheric boundary layer in a
horizontally periodic `2000 pi m x 2000 pi m x 1000 m` domain (approximately
`6283.185 m x 6283.185 m x 1000 m`). The `64 x 64 x 64` mesh gives
approximately `98.175 m x 98.175 m x 15.625 m` cells.

A constant streamwise kinematic pressure gradient drives the flow. Its
acceleration is diagnosed from the configured reference friction velocity and
forcing height,

```text
u_*^2 / H = (0.4 m/s)^2 / 1000 m = 1.6e-4 m/s^2.
```

The flow starts from the neutral logarithmic profile with small correlated
velocity perturbations. It uses a filtered neutral log-law lower wall and the
Lagrangian scale-dependent dynamic (LASD) closure. Buoyancy, rotation, and
top Rayleigh damping are disabled by this runner, so pressure forcing and wall
stress are the only mean momentum sources and sinks.

## Modulated gradient model option

[`config_mgm.toml`](config_mgm.toml) selects the memoryless modulated gradient
model (MGM) ported from `legacy/fortran_cuda/src/sgs_mgm.cuf` (`model = 4`).
The port preserves the legacy anisotropic gradient tensor, clipped
backscatter, aspect-ratio-corrected dissipation coefficient, molecular
viscosity, and the `Cs = 0.1` Smagorinsky fallback used when `Gkk` is
ill-conditioned. The passive scalar uses a fixed turbulent Prandtl number and
does not allocate LASD closure memory.

Validate or run the MGM variant with:

```bash
jaxwind runners/pressure_driven_warmup \
  --config runners/pressure_driven_warmup/config_mgm.toml \
  --dry-run

jaxwind runners/pressure_driven_warmup \
  --config runners/pressure_driven_warmup/config_mgm.toml
```

The case advances 10 simulated hours with 0.1 s AB2 steps. Horizontal-plane
statistics are sampled every 10 s during the final two hours. Restartable
checkpoints are written hourly, and the completed run writes
`checkpoint_final.npz` for downstream precursor cases.

For the complete MGM GPU run plus the neutral log-law velocity plot, use the
one-command benchmark from the repository root:

```bash
python -m benchmark.PressureDrivenMGM
```

Validate the case without importing JAX:

```bash
jaxwind runners/pressure_driven_warmup --dry-run
```

Run it from the initial condition:

```bash
jaxwind runners/pressure_driven_warmup
```

Use `--restart PATH` to continue from a checkpoint. Use `--overwrite` when an
existing output directory may be replaced. Runtime products are written under
`outputs/pressure_driven_warmup_64x64x64/`.

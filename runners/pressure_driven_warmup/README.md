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
velocity perturbations. It uses a filtered neutral log-law lower wall with the
Porté-Agel (2000) first-interior-face shear correction and the Lagrangian
scale-dependent dynamic (LASD) closure. Buoyancy, rotation, and top Rayleigh
damping are disabled by this runner, so pressure forcing and wall stress are
the only mean momentum sources and sinks.

## Modulated gradient model option

[`config_mgm.toml`](config_mgm.toml) selects the memoryless modulated gradient
model (MGM) ported from `legacy/fortran_cuda/src/sgs_mgm.cuf` (`model = 4`).
The port preserves the anisotropic gradient tensor, clipped backscatter,
molecular viscosity, and the `Cs = 0.1` Smagorinsky fallback used when `Gkk`
is ill-conditioned. Its dissipation coefficient is diagnosed independently on
every horizontal plane from the Lu--Porté-Agel conditional/unconditional cubic
transfer moments; `dissipation_coefficient = 1` is the neutral-ABL demo's
unit multiplier. MGM, LASD, and AMD all use conservative convection with the
same horizontal three-halves padding. Accepted velocity states retain the full
resolved bandwidth apart from the even-grid Nyquist modes.
The benchmark explicitly sets
`wall.porte_agel_correction = true`, matching the legacy correction that adds
`(1/log(3) - 1)` times the horizontal mean shear at the first interior face.
The passive scalar uses a fixed turbulent Prandtl number and does not allocate
LASD closure memory.

Validate or run the MGM variant with:

```bash
jaxwind runners/pressure_driven_warmup \
  --config runners/pressure_driven_warmup/config_mgm.toml \
  --dry-run

jaxwind runners/pressure_driven_warmup \
  --config runners/pressure_driven_warmup/config_mgm.toml
```

The canonical LASD `config.toml` case advances 10 simulated hours with 0.1 s
AB2 steps. The MGM case takes its runtime and sampling controls independently
from `config_mgm.toml`. Both write restartable checkpoints and produce
`checkpoint_final.npz` after reaching their configured final time.

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

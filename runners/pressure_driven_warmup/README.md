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

The canonical LASD `config.toml` case advances 10 simulated hours with 0.1 s
AB2 steps. It writes restartable checkpoints and produces
`checkpoint_final.npz` after reaching the configured final time. For the
complete GPU run plus the neutral log-law velocity plot, use the one-command
benchmark from the repository root:

```bash
jaxwind benchmark/PressureDrivenLASD/config.toml
```

Validate the case without importing JAX:

```bash
jaxwind runners/pressure_driven_warmup/config.toml --dry-run
```

Run it from the initial condition:

```bash
jaxwind runners/pressure_driven_warmup/config.toml
```

Configure `execution.restart_checkpoint` to continue from a checkpoint and
`execution.overwrite = true` to replace existing products. Runtime products
are written under `outputs/pressure_driven_warmup_64x64x64/`.

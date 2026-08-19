# Applications

Applications translate data-only case configurations into generic JAX-Wind
solver components, then own initialization, diagnostics, checkpoints, and
output effects. They are selected explicitly by the command; there is no case
registry or case-name dispatch.

`abl` is regime-agnostic. Neutral, stable, and convective are not application
or solver modes; scalar buoyancy coupling, initial stratification, and surface
heat transfer determine stability.

```bash
python -m applications.pressure_driven_lasd \
  cases/PressureDrivenLASD/config.toml --dry-run
python -m applications.abl \
  cases/Andren1994/config.toml --dry-run
python -m applications.abl \
  cases/Nieuwstadt1993/config.toml --dry-run
python -m applications.windfarm_precursor --dry-run
```

`windfarm_precursor` starts from a developed pressure-driven LASD checkpoint,
records rank-local 11-plane HDF5 inflow/outflow slabs every 10 steps, and runs
a second main domain with the CUDA-Fortran cosine-blend/direct-overwrite inlet.
The main pressure gradient is disabled exactly as in Fortran `sim_flag = 3`.
The complete warmup/precursor/main AD-BEM benchmark is configured by
[`cases/DTU10MWPrecursor/benchmark_adbem.toml`](../cases/DTU10MWPrecursor/benchmark_adbem.toml)
and launched with `python -m applications.windfarm_precursor.benchmark`.

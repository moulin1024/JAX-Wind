# Applications

Applications translate data-only case configurations into generic JAX-Wind
solver components, then own initialization, diagnostics, checkpoints, and
output effects. They are selected explicitly by the command; there is no case
registry or case-name dispatch.

`abl` and `fv_abl` are regime-agnostic solver cores. Neutral, stable, and
convective are not application or solver modes; scalar buoyancy coupling,
initial stratification, and surface heat transfer determine stability. The same
case TOML feeds both cores; `[finite_volume]` contains the FV backend, closure,
chunking, spectrum, and output choices.

```bash
python -m applications.pressure_driven_lasd \
  cases/PressureDrivenLASD/config.toml --dry-run
python -m applications.abl \
  cases/Andren1994/config.toml --dry-run
python -m applications.abl \
  cases/Nieuwstadt1993/config.toml --dry-run
python -m applications.fv_abl \
  cases/Nieuwstadt1993/config.toml --dry-run
python -m applications.windfarm_precursor --dry-run
```

The FV core also owns an offline one-plane precursor workflow. A single
`[finite_volume_workflow]` table configures three fixed stages:

1. a periodic warmup using the FFT pressure solver;
2. a periodic precursor using the same physics and FFT solver, recording one
   `yz` inflow layer at every step; and
3. a nonperiodic main domain using the recorded layer as a direct inlet and
   the GMG pressure solver.

The main outlet applies the three-point second-order zero-gradient condition
to transported tangential velocity and scalar values. Its normal velocity is
then selected by the pressure projection, with pressure Neumann at the inlet
and fixed pressure at the outlet. Inspect or execute all stages with:

```bash
python -m applications.fv_abl.workflow \
  cases/Andren1994/config.toml --dry-run
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m applications.fv_abl.workflow \
  cases/Andren1994/config.toml --overwrite
```

Use `--stage warmup`, `--stage precursor`, or `--stage main` to resume the
workflow from its on-disk checkpoint and inflow recording.

`windfarm_precursor` starts from a developed pressure-driven LASD checkpoint,
records rank-local 11-plane HDF5 inflow/outflow slabs every 10 steps, and runs
a second main domain with the CUDA-Fortran cosine-blend/direct-overwrite inlet.
The main pressure gradient is disabled exactly as in Fortran `sim_flag = 3`.
The complete warmup/precursor/main AD-BEM benchmark is configured by
[`cases/DTU10MWPrecursor/benchmark_adbem.toml`](../cases/DTU10MWPrecursor/benchmark_adbem.toml)
and launched with `python -m applications.windfarm_precursor.benchmark`.

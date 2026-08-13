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
records rank-local HDF5 inflow/outflow planes, and runs a second main domain
whose downstream fringe replays the recorded inflow at the same accepted
clock.

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
```

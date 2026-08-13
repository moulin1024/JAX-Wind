# GABLS1

This is the canonical 32³, 400 m GABLS1 case integrated for nine hours,
with statistics collected over hours 8–9. It is composed by the same uniform
`applications.abl` solver used by the other ABL cases. The physical case data
adds an evolving surface potential temperature (−0.25 K h⁻¹); the generic
Monin–Obukhov component couples that value to momentum and scalar fluxes.
The generic advection-frame input removes the uniform 8 m s⁻¹ translation
from the evolved fields while retaining it in wall exchange and diagnostics.

Run it on one GPU from the repository root:

```bash
PYTHONPATH=src:. JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m applications.abl cases/GABLS1/config.toml
```

Inspect the resolved composition without importing JAX:

```bash
PYTHONPATH=src:. python -m applications.abl cases/GABLS1/config.toml --dry-run
```

After the statistics window has been completed, compare the profiles with the
included official 12.5 m participant ensemble:

```bash
PYTHONPATH=src:. python tools/overlay_gabls1.py
```

The overlay, interpolated ensemble CSV, and checkout metrics are written under
`outputs/gabls1_lasd_32x32x32/overlays/`.

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
The same command also creates `gabls1_complete_overlay.png`, individual
figures for official sets A--E, complete profile and time-series comparison
CSVs, and `complete_overlay_manifest.json`. The complete figure contains all
30 quantities prescribed by the official submission format. A panel is marked
as reference-only when the completed run did not accumulate a numerically
equivalent JAX-Wind observable.
The overlay tool selects the official 12.5 m ensemble for 32³ results and the
official 6.25 m ensemble for 64³ results from the profile spacing. An explicit
`--reference-dir` still overrides this selection.

The refined 64³ case uses a 1/12 s timestep and updates LASD every eight
steps. It preserves the same nine-hour duration and hour-8-to-9 statistics
window:

```bash
PYTHONPATH=src:. JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m applications.abl cases/GABLS1/config_64.toml \
  --advection rotational --dealiasing two_thirds \
  --lasd-filter-backend cufft
```

# Andrén et al. (1994) neutral ABL case

This data-only case is configured by the fixed-schema
[`config.toml`](config.toml) and composed by the
[`abl`](../../applications/abl/config.py) application. The TOML
contains canonical SI inputs: the grid, Coriolis and
geostrophic values, wall roughness, passive-scalar flux, initial profile,
physical times, and numerical controls. The composition owns SI-to-execution
scaling; it performs no execution and never dispatches on the case name.

There is no neutral/stable/convective selector. This case's scalar is
explicitly passive, so it has no buoyancy feedback and the resolved stability
is the neutral limit.

The schema has no solver construction, implementation selector, registry key,
or case-specific solver. Unknown tables and keys are rejected. Its full
resolved composition remains visible through the dry run.

Inspect the resolved declaration without importing JAX:

```bash
python -m applications.abl cases/Andren1994/config.toml --dry-run
```

An alternative file using the same fixed physical schema can be supplied as
the positional configuration path:

```bash
python -m applications.abl /path/to/config.toml --dry-run
```

Run the canonical `40³`, `tf = 10` case, or a short integration:

```bash
python -m applications.abl cases/Andren1994/config.toml
python -m applications.abl \
  cases/Andren1994/config.toml --max-steps 10 --overwrite
```

Use a separate output directory for smoke runs:

```bash
python -m applications.abl cases/Andren1994/config.toml \
  --max-steps 1 \
  --output /tmp/andren1994-smoke \
  --overwrite
```

The reference profile and published comparison envelope live under
[`reference`](reference/). Earlier figure-extraction and detailed budget tools
remain preserved in [`legacy/cases/Andren1994`](../../legacy/cases/Andren1994/).

Clean crops of all 19 published figures live under
[`reference/figure_panels`](reference/figure_panels/). Their source pages,
crop boxes, and active plot-axis registrations are recorded in
[`manifest.json`](reference/figure_panels/manifest.json).

Overlay a completed active run on the directly comparable profile panels:

```bash
python tools/overlay_andren1994.py \
  outputs/andren1994_lasd_40x40x40
```

This writes individual overlays for Figures 2 through 8, 11, 14, and 15, a
compact diagnostic sheet, and a complete 19-figure sheet under the
result directory's `paper_overlays/`. Figures 2, 3, 6, 8, 11, 14, and 15 use
the extended restartable diagnostics: total TKE, component momentum
stationarity, resolved-plus-SGS fluxes, signed SGS TKE transfer, SGS
diffusivities, and streamwise spectra. Reference-only panels are labeled
explicitly when an older run did not record their required observable. The
tool reads `history.csv`, `profiles.csv`, `spectra.csv`, and `summary.json`; it
does not run or reconfigure the solver. To reproduce the checked-in crops from
the DLR article scan:

```bash
python tools/overlay_andren1994.py \
  --extract-from-pdf /path/to/qj-1457-1994.pdf
```

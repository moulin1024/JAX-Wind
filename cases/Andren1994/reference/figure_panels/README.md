# Andrén et al. figure panels

These are clean crops of Figures 1–19 from the article scan identified in
[`manifest.json`](manifest.json). They contain only the published curves; no
JAX-Wind result is baked into the reference images.

The manifest records the rendered page size and crop box for every figure. It
also records plot-frame registrations for Figures 2--8, 11, 14, and 15. The
extended `history.csv`, `profiles.csv`, and `spectra.csv` outputs supply those
ten directly comparable panels; the other nine remain reference-only because
they do not yet have a faithful active observable and registered axis.

Generate result overlays without modifying these reference files:

```bash
python tools/overlay_andren1994.py /path/to/completed/results
```

The article citation and source URL are recorded in the manifest and in
[`../reference_results.json`](../reference_results.json).

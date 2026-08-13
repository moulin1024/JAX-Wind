# Andrén et al. figure panels

These are clean crops of Figures 1–19 from the article scan identified in
[`manifest.json`](manifest.json). They contain only the published curves; no
JAX-Wind result is baked into the reference images.

The manifest records the rendered page size and crop box for every figure. It
also records plot-frame registrations for Figures 4, 5, and 7, which are the
panels directly supported by the active `profiles.csv` output.

Generate result overlays without modifying these reference files:

```bash
python tools/overlay_andren1994.py /path/to/completed/results
```

The article citation and source URL are recorded in the manifest and in
[`../reference_results.json`](../reference_results.json).

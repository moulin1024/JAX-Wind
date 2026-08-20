# Nieuwstadt et al. (1993)

This data-only case reproduces the dry, shear-free boundary-layer comparison
on the paper's 40 x 40 x 48 grid. It uses the same ABL application and the
same configuration schema as the Andrén case. The scalar profile, surface
flux, and buoyancy coefficient determine the resulting dynamics; there is no
case-specific solver mode.

Inspect the fully lowered case without initializing JAX:

```bash
python -m applications.abl cases/Nieuwstadt1993/config.toml --dry-run
```

Run it on one GPU:

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m applications.abl cases/Nieuwstadt1993/config.toml
```

Use `--max-steps 10 --overwrite` for a short end-to-end smoke run.

The same configured case can be run with the finite-volume solver,
hydrostatic-free Boussinesq coupling, AMD, AB2, and FFT pressure projection:

```bash
python -m applications.fv_abl cases/Nieuwstadt1993/config.toml --dry-run
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m applications.fv_abl cases/Nieuwstadt1993/config.toml --overwrite
```

This FV realization writes the same `profiles.csv`, `radial_spectra.csv`, and
`summary.json` comparison contract as the uniform ABL application while
recording its AMD closure distinction. The fixed turbulent Prandtl number and
radial-spectrum policy are declared in the TOML `[finite_volume]` table.

After a completed run, check the selected paper overlays (Figures 2, 3, 6,
8, 11, 14, and 15) with:

```bash
python tools/overlay_nieuwstadt1993.py

# FV result
python tools/overlay_nieuwstadt1993.py \
  --results outputs/nieuwstadt1993_fv_fft_40x40x48 \
  --legend-label "JAX-Wind FV AMD FFT 40×40×48 GPU"
```

Individual panels and the combined checkout are written beneath the selected
result directory; the default LASD location is
`outputs/nieuwstadt1993_lasd_40x40x48/overlays/`.

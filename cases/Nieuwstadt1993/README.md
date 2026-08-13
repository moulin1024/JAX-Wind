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

After a completed run, check the selected paper overlays (Figures 2, 3, 6,
8, 11, 14, and 15) with:

```bash
python tools/overlay_nieuwstadt1993.py
```

Individual panels and the combined checkout are written to
`outputs/nieuwstadt1993_lasd_40x40x48/overlays/`.

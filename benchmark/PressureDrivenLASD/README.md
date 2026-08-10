# Pressure-driven neutral ABL with LASD

This case directly constructs the JAX-Wind solver with:

- conservative, horizontally dealiased momentum transport;
- filtered neutral log-law wall stress;
- constant streamwise pressure-gradient forcing;
- Lagrangian scale-dependent dynamic momentum closure;
- fixed-step AB2 integration; and
- compatible pressure projection through `spectral-fd`.

Inspect the resolved case:

```bash
python -m benchmark.PressureDrivenLASD.case --dry-run
```

Run it, or use a short smoke integration:

```bash
python -m benchmark.PressureDrivenLASD.case
python -m benchmark.PressureDrivenLASD.case --max-steps 10 --overwrite
```

The case owns configuration, initialization, profile statistics, checkpoints,
history, summary output, and the log-law plot. The package owns the solver and
has no knowledge of this benchmark's name or output layout.

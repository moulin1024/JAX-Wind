# Pressure-driven neutral ABL with LASD

The pressure-driven application interprets this data-only case as:

- conservative, horizontally dealiased momentum transport;
- filtered neutral log-law wall stress;
- constant streamwise pressure-gradient forcing;
- Lagrangian scale-dependent dynamic momentum closure;
- fixed-step AB2 integration; and
- compatible pressure projection through `spectral-fd`.

Inspect the resolved case:

```bash
python -m applications.pressure_driven_lasd \
  cases/PressureDrivenLASD/config.toml --dry-run
```

Run it, or use a short smoke integration:

```bash
python -m applications.pressure_driven_lasd \
  cases/PressureDrivenLASD/config.toml
python -m applications.pressure_driven_lasd \
  cases/PressureDrivenLASD/config.toml --max-steps 10 --overwrite
```

The case owns only configuration data. The application owns initialization,
profile statistics, checkpoints, history, summary output, and the log-law plot.
The package owns the solver and has no knowledge of this case's name or output
layout.

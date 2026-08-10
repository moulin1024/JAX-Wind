# Declarative benchmark cases

Executable benchmark cases use the same interface as application cases:

```bash
jaxwind benchmark/PressureDrivenLASD/config.toml --dry-run
jaxwind benchmark/PressureDrivenLASD/config.toml
```

Each runnable case directory owns a `config.toml`. The `[case]` table selects
a package-owned runner under `src/jaxwind/runners`; validation, initialization,
time integration, checkpointing, and online diagnostics do not live in the
benchmark directory.

Python files retained beside literature data are offline analysis utilities.
They consume completed outputs and do not define the simulation case.

The configuration-only cases currently using the uniform runner are:

- `Andren1994/`, using the shared `abl` warmup workflow with neutral Ekman
  physics and LASD-only benchmark diagnostics;
- `PressureDrivenLASD/`, using `pressure_driven_warmup`;
- `GABLS1/`, using `gabls1`.

New executable benchmarks should follow this layout instead of adding a
benchmark-local `run.py`.

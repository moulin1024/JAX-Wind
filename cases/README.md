# Cases

A case is data: physical parameters, numerical controls, input fields, and
optional reference evidence. Case directories contain no Python. Applications
materialize that data through the public JAX-Wind solver and own diagnostics
and output effects.

- [`PressureDrivenLASD`](PressureDrivenLASD/README.md) is a pressure-driven
  neutral ABL case.
- [`Andren1994`](Andren1994/README.md) is a neutral ABL case with literature
  reference data for comparison.

Select the appropriate application explicitly and pass the case TOML:

```bash
python -m applications.pressure_driven_lasd \
  cases/PressureDrivenLASD/config.toml --dry-run
python -m applications.abl \
  cases/Andren1994/config.toml --dry-run
```

JAX-Wind does not contain case names, a registry, or a universal case schema.
The [`applications`](../applications/README.md) layer owns schema-specific
composition and execution without dispatching on case names. Historical case
material lives under [`legacy/cases`](../legacy/cases/README.md).

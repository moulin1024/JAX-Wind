# Cases

A case is data: physical parameters, numerical controls, input fields, and
optional reference evidence. Case directories contain no Python. Applications
materialize that data through the public JAX-Wind solver and own diagnostics
and output effects.

- [`PressureDrivenLASD`](PressureDrivenLASD/README.md) is a pressure-driven
  neutral ABL case.
- [`Andren1994`](Andren1994/README.md) is a neutral ABL case with literature
  reference data for comparison.
- [`Nieuwstadt1993`](Nieuwstadt1993/README.md) is a shear-free, buoyancy-driven
  ABL case using the same application schema and solver transition.
- [`GABLS1`](GABLS1/README.md) is the stable nine-hour ABL
  intercomparison with an evolving surface temperature.
- [`DTU10MWPrecursor`](DTU10MWPrecursor/README.md) contains the three-stage
  strict-inlet DTU 10-MW AD-BEM wake benchmark.
- [`HITSZWindTunnel`](HITSZWindTunnel/README.md) scales the same three-stage
  workflow to the HITSZ R9 wind-tunnel experiment using fitted measured inflow
  and a 480 RPM HITSZ001 AD-BEM rotor.

Select the appropriate application explicitly and pass the case TOML:

```bash
python -m applications.pressure_driven_lasd \
  cases/PressureDrivenLASD/config.toml --dry-run
python -m applications.abl \
  cases/Andren1994/config.toml --dry-run
python -m applications.abl \
  cases/Nieuwstadt1993/config.toml --dry-run
python -m applications.fv_abl \
  cases/GABLS1/config.toml --dry-run
python -m applications.windfarm_precursor.benchmark --dry-run
```

JAX-Wind does not contain case names, a registry, or a universal case schema.
The [`applications`](../applications/README.md) layer owns schema-specific
composition and execution without dispatching on case names. Historical case
material lives under [`legacy/cases`](../legacy/cases/README.md).

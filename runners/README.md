# Runner workflows

Each directory under `runners/` is a declarative case definition consumed by
the `jaxwind` command. A case directory owns:

- a TOML case definition in physical SI units;
- selection of a package-owned runner through `case.runner`;
- physical, numerical, and output choices for that runner; and
- a case README describing those choices.

TOML is the only case-configuration format. Human-authored inputs use
`config.toml`, `jaxwind ... --dry-run` prints TOML, and each run records the
fully derived settings in `resolved_config.toml`. JSON is reserved for
machine-generated results and runtime metadata such as `summary.json` and
checkpoint payload metadata; those files are not accepted as case inputs.

Reusable validation, initialization, runtime orchestration, checkpointing, and
statistics logic lives under `src/jaxwind/runners`. A copied case that uses an
existing runner therefore needs only `config.toml`; it does not need a
`run.py`.

Launch a directory directly:

```bash
jaxwind runners/pressure_driven_warmup
```

Use [`pressure_driven_warmup/`](pressure_driven_warmup/README.md) as the first
template for new cases.

[`dtu10mw_concurrent_precursor_smoke/`](dtu10mw_concurrent_precursor_smoke/README.md)
demonstrates a second package-owned runner: it loads a developed warmup into
synchronized precursor and turbine domains, then applies a pure-thrust ADM and
live precursor fringe in the main domain.

The same runner also accepts `turbine.model =
"openfast_rigid_actuator_line"`. In that mode, `openfast_input_file` points to
the unmodified OpenFAST primary deck and JAX-Wind resolves its ElastoDyn,
AeroDyn, blade, and airfoil references during case validation. See the
[top-level OpenFAST section](../README.md#openfast-rigid-actuator-line) for the
configuration and current compatibility boundary.

[`nrel5mw_direct_alm_smoke/`](nrel5mw_direct_alm_smoke/README.md) is the
cold-start integration gate for the rigid actuator line. It runs without a
warmup checkpoint on a `128 × 128 × 512` mesh in a 512 m cube, with the rotor
over the center of the ground plane.

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

Launches accept no command-line case overrides. The output directory comes
from `[output]`; restart and overwrite behavior come from `[execution]`:

```toml
[execution]
# restart_checkpoint = "outputs/previous/checkpoint_latest.npz"
overwrite = false
```

`--dry-run` is the only optional CLI flag because it validates and displays
the resolved configuration without launching a simulation. Shorter smoke
cases, alternate outputs, and continuations must be separate TOML files.

Reusable validation, initialization, runtime orchestration, checkpointing, and
statistics logic lives under `src/jaxwind/runners`. A copied case that uses an
existing runner therefore needs only `config.toml`; it does not need a
`run.py`.

The same contract applies to executable cases under [`benchmark/`](../benchmark/README.md).
The public `jaxwind.runners.load_case` and `jaxwind.runners.run_case` functions
provide the configuration dispatch used by the CLI, so library callers and
the command line resolve runner selection identically.

## ABL workflow family

The `abl` runner owns a common ABL workflow family. In the currently
implemented `workflow = "warmup"`, the thermal boundary condition and physical
forcing define the case while the invocation and output contract stay
unchanged:

```bash
jaxwind runners/abl_warmup_neutral/config.toml
jaxwind runners/abl_warmup_stable/config.toml
jaxwind runners/abl_warmup_convective/config.toml
```

Do not configure a stability category. The runner derives neutral, stable, or
unstable behavior from the configured surface heat forcing, reports the
derived class as `stability`, and records the surface buoyancy flux and
Obukhov length as continuous runtime diagnostics.

Every warmup writes `checkpoint_latest.npz`, `history.csv`,
`resolved_config.toml`, `summary.json`, and `warmup_manifest.json`. The manifest
identifies the `z_slab_boussinesq.v1` checkpoint layout so later workflow modes
can consume the same developed state. Planned modes are a precursor that dumps
inflow data, a wind-farm main simulation that reads dumped inflow, and a
concurrent precursor/main simulation. Those modes are not selected implicitly
by a warmup configuration.

The three supplied warmups are pure configuration directories: they contain no
case-owned Python. To define another warmup, copy one of them and edit only its
`config.toml`.

Launch any other directory directly:

```bash
jaxwind runners/pressure_driven_warmup/config.toml
```

The older [`pressure_driven_warmup/`](pressure_driven_warmup/README.md) runner
remains available for compatibility; new ABL cases should use `abl` with an
explicit workflow.

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

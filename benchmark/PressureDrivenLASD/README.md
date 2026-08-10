# Pressure-driven LASD benchmark

This directory is a pure declarative case. Its `config.toml` selects the
package-owned pressure-driven runner and the Lagrangian scale-dependent
dynamic (LASD) closure; it contains no benchmark-specific execution code.

Run it on a GPU node from the repository root with:

```bash
jaxwind benchmark/PressureDrivenLASD/config.toml
```

Results default to `outputs/pressure_driven_lasd_64x64x64_gpu/`. The uniform
case runner writes restartable checkpoints, resolved configuration, profiles,
history, summary, and `loglaw_velocity_profile.svg` over the final 20% of
simulated time. The pressure-solver submodule must be initialized once with
`git submodule update --init --recursive`.

Validate the complete resolved configuration without importing JAX:

```bash
jaxwind benchmark/PressureDrivenLASD/config.toml --dry-run
```

Smoke variants and continuations are separate TOML configurations. Set their
`time`, `output.directory`, and optional `execution.restart_checkpoint`
entries in the file before launching it; the CLI does not override case data.

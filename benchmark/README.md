# Benchmark cases

A benchmark is a concrete case over the public JAX-Wind solver. It owns its
physical parameters, initial condition, integration duration, diagnostics,
acceptance criteria, and output effects.

The active benchmark is [`PressureDrivenLASD`](PressureDrivenLASD/README.md):

```bash
python -m benchmark.PressureDrivenLASD.case --dry-run
python -m benchmark.PressureDrivenLASD.case
```

JAX-Wind does not contain benchmark names, runner identifiers, a registry, or a
universal case schema. Literature assets and offline analysis from earlier
studies live in [`legacy/benchmarks`](../legacy/benchmarks/README.md).

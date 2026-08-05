# GABLS1 stable-boundary-layer benchmark

The complete active case is [case.toml](case.toml). It is data consumed by the
same `jaxwind-run` application as the other validation cases; there is no
GABLS-specific runner, distributed adapter, or profiling workflow.

Canonical CUDA run:

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
python -m jaxwind benchmark/GABLS1/case.toml
```

Short end-to-end run:

```bash
python -m jaxwind benchmark/GABLS1/case.toml --quick
```

The configuration declares the `32³`, 400 m domain; geostrophic wind;
Coriolis force; Monin–Obukhov surface coupling and cooling; inversion profile;
AMD momentum/scalar closures; MP5 stabilization; full projection; and the
8–9 hour averaging window.

Official 12.5 m and 6.25 m ensemble archives and plotting scripts remain
reference/postprocessing assets. They are not imported by the solver.

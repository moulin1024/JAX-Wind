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

The analytic `64³` stretched-grid variant is
[case_64_stretched.toml](case_64_stretched.toml). Its periodic x and y axes use
an exponential map focused at the domain centre, while z uses the same family
focused at the ground:

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda PYTHONPATH=src \
python -m jaxwind benchmark/GABLS1/case_64_stretched.toml
```

For each optional `[grid.mapping.<axis>]` table, `focus` is normalized to
`[0, 1]`, `strength=0` is exactly uniform, and positive exponential strength
increases clustering toward the focus. An interior focus joins two analytic
branches at an exact grid face; boundary focuses produce one-sided clustering.

The configuration declares the `32³`, 400 m domain; geostrophic wind;
Coriolis force; Monin–Obukhov surface coupling and cooling; inversion profile;
AMD momentum/scalar closures; MP5 stabilization; full projection; and the
8–9 hour averaging window.

Official 12.5 m and 6.25 m ensemble archives and plotting scripts remain
reference/postprocessing assets. They are not imported by the solver.

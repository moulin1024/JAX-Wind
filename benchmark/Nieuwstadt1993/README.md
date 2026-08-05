# Nieuwstadt et al. (1993) convective benchmark

The benchmark is entirely described by [case.toml](case.toml). It uses the
same generic runner and checkpoint/output schema as Andrén and GABLS1.

Canonical CUDA run:

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
python -m jaxwind benchmark/Nieuwstadt1993/case.toml
```

Short end-to-end run:

```bash
python -m jaxwind benchmark/Nieuwstadt1993/case.toml --quick
```

The TOML configuration declares the `40 × 40 × 48` grid, convective
initial perturbation, prescribed surface heat flux, AMD momentum/scalar
closures, full projection, and the `10 <= t/t* <= 11` averaging window. No
benchmark-specific time loop or solver factory exists.

Paper extraction, comparison, and overlay scripts are offline postprocessing;
they do not participate in numerical execution.

# JAX-Wind

JAX-Wind is a benchmark-focused JAX large-eddy simulation solver for
atmospheric boundary layers. The active `jaxwind` package is intentionally
small: it contains the non-spectral MAC, AMD, MP5, surface-layer, and
Boussinesq implementation needed by three validation cases.

The previous document-first semantic architecture is preserved as the
`jaxwind_archiv` Python package. It remains available for reference and
regression work, but the benchmark runners do not depend on it.

## Active solver

The active numerical path provides:

- face-staggered MAC velocity and cell-centred pressure/scalar fields;
- matrix-free finite-volume Poisson projection;
- geometric-multigrid-preconditioned PCG;
- full pressure projection after every momentum stage;
- conservative momentum transport and MP5 dissipation;
- AMD momentum/scalar closures and pressure-hierarchy multilevel LASD;
- explicit SSPRK3 and IMEX-ARK3 vertical SGS integration;
- neutral-log and Monin-Obukhov surface fluxes; and
- Strang-coupled Boussinesq buoyancy.

Only the canonical uniform-grid, single-process solver belongs to the active
minimum. Distributed solvers, semantic interpreters, AB2, OpenFAST, meshing,
actuator models, and experimental workflows live in `jaxwind_archiv`.
Andrén can additionally run a multilevel LASD closure. Its two Germano test
scales share the pressure solver's first two geometric-multigrid levels;
conservative coarse-grid restriction keeps the bandwidth-heavy statistics and
Lagrangian memory off the LES grid.

## Validation benchmarks

The active solver is exercised by:

- [Andrén et al. (1994)](benchmark/Andren1994/README.md), neutral Ekman ABL;
- [Nieuwstadt et al. (1993)](benchmark/Nieuwstadt1993/README.md), convective
  boundary layer; and
- [GABLS1](benchmark/GABLS1/README.md), stable boundary layer.

Short end-to-end checks:

```bash
python benchmark/Nieuwstadt1993/run_amd.py --quick
python benchmark/GABLS1/run.py --quick
python benchmark/Andren1994/run.py \
  --end-ft 0.0001 --sample-start-ft 0 \
  --sample-every 1 --history-every 1 --single

# Multilevel LASD on the same pressure-GMG hierarchy
python benchmark/Andren1994/run.py --sgs lasd \
  --end-ft 0.0001 --sample-start-ft 0 \
  --sample-every 1 --history-every 1 --single
```

The canonical Andrén statistics window is `7 <= ft <= 10`; its short default
is intended for development. Nieuwstadt uses `10 <= t/t* <= 11`, and GABLS1
uses hours 8--9.

## Installation and tests

JAX-Wind requires Python 3.11 or newer:

```bash
python -m pip install -e .
python -m pytest -q
```

Install the JAX build matching the CUDA runtime for GPU execution. JAX selects
the available GPU automatically.

## Package boundary

```text
benchmark case
    -> jaxwind.momentum
        -> shared MultigridHierarchy <- jaxwind.pressure
            -> jaxwind.domain.RectilinearGrid

jaxwind_archiv    historical architecture; never imported by active benchmarks
```

The distribution remains named `jaxwind`. Python identifiers cannot contain a
hyphen, so the requested `jaxwind-archiv` package is imported as
`jaxwind_archiv`. Its historical mesh command is available as
`jaxwind-archiv-mesh`.

## License

JAX-Wind is released under the [MIT License](LICENSE).

# JAX-Wind

JAX-Wind is a benchmark-focused JAX large-eddy simulation solver for
atmospheric boundary layers. The active `jaxwind` package is intentionally
small: it contains one non-spectral ABL solver composed from MAC, AMD, MP5,
surface-layer, and optional Boussinesq thermodynamics.

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
- AMD closures, pressure-hierarchy FV-native momentum LASD, and an optional
  FV-dynamic scalar Germano closure using the same hierarchy;
- explicit SSPRK3 and IMEX-ARK3 vertical SGS integration;
- finite-volume-filtered neutral-log and Monin-Obukhov surface fluxes with a
  similarity-consistent first-internal-face reconstruction; and
- one `ABLSolver`/`ABLState` path for neutral and thermally stratified flows.

Neutral, convective, and stable are not solver classes. The same solver is
configured with an optional potential-temperature field, buoyancy constants,
and an adiabatic, prescribed-flux, or prescribed-temperature surface. The
initial stratification and surface heat transfer determine the flow regime.

Only the rectilinear single-process solver belongs to the active minimum;
each axis may use an analytic stretching map. Distributed solvers, semantic
interpreters, AB2, OpenFAST, meshing,
actuator models, and experimental workflows live in `jaxwind_archiv`.
Any neutral or thermally coupled case can additionally run the multilevel LASD
closure. Its two Germano test
scales share the pressure solver's first two geometric-multigrid levels;
conservative coarse-grid restriction keeps the bandwidth-heavy statistics and
Lagrangian memory off the LES grid.  The closure is finite-volume native:
velocity is restricted first, then each GMG level recomputes its own metric-
aware strain and model tensor, so the Germano identity uses `D_H(Ru)` rather
than the non-commuting `R(D_hu)` approximation.

The surface closure is boundary-condition driven rather than flow-regime
driven. Prescribed surface temperature and prescribed heat flux can both use
coupled MOST; the neutral limit uses the same interface. Momentum and scalar
values transported through the first internal face are reconstructed from the
same point-to-cell-average similarity relation that diagnoses the wall flux,
including stability functions and the true stretched-cell bounds.

With multilevel LASD momentum, thermodynamic cases may select the FV-dynamic
scalar model:

```toml
[sgs]
model = "multilevel_lasd"

[thermodynamics]
enabled = true
scalar_sgs_model = "fv_dynamic"
```

An optional slow horizontal-zero-mode constraint derives its target total
stress from the filtered discrete mean acceleration, external pressure/
Coriolis/sponge forcing, and the actual MOST traction. It does not prescribe a
logarithmic profile or assume stationary/neutral flow:

```toml
[mean_momentum]
enabled = true
timescale = 600.0
gain = 1.0
```

It is disabled when the table is absent, so benchmark validation does not gain
an implicit mean-profile forcing.

## Validation benchmarks

The active solver is exercised by three declarative TOML cases:

- [Andrén et al. (1994)](benchmark/Andren1994/README.md), neutral Ekman ABL;
- [Nieuwstadt et al. (1993)](benchmark/Nieuwstadt1993/README.md), convective
  boundary layer; and
- [GABLS1](benchmark/GABLS1/README.md), stable boundary layer.

There are no benchmark-specific runners. All cases use the same CLI, solver
factory, time loop, checkpoint schema, and output schema:

```bash
python -m jaxwind benchmark/Nieuwstadt1993/case.toml --quick
python -m jaxwind benchmark/GABLS1/case.toml --quick
python -m jaxwind benchmark/Andren1994/case.toml --quick
```

Any configured value can be changed without adding a runner, using a typed TOML
override such as `--set 'numerics.dtype="float64"'`. The resolved configuration
is stored with every result.

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
    -> jaxwind.momentum.ABLSolver
        -> MomentumOperators + optional ScalarOperators
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

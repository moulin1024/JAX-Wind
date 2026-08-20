# Andrén et al. (1994) neutral ABL case

This data-only case is configured by the fixed-schema
[`config.toml`](config.toml) and composed by the
spectral [`abl`](../../applications/abl/config.py) and finite-volume
[`fv_abl`](../../applications/fv_abl/config.py) applications. The TOML
contains canonical SI inputs: the grid, Coriolis and
geostrophic values, wall roughness, passive-scalar flux, initial profile,
physical times, and numerical controls. The composition owns SI-to-execution
scaling; it performs no execution and never dispatches on the case name.

There is no neutral/stable/convective selector. This case's scalar is
explicitly passive, so it has no buoyancy feedback and the resolved stability
is the neutral limit.

The schema has no solver construction, registry key, or case-specific solver.
The `[finite_volume]` table selects only generic FV numerical and diagnostic
components; unknown tables and keys are rejected. Each core exposes its fully
resolved composition through a dry run.

Inspect the resolved declaration without importing JAX:

```bash
python -m applications.abl cases/Andren1994/config.toml --dry-run
```

An alternative file using the same fixed physical schema can be supplied as
the positional configuration path:

```bash
python -m applications.abl /path/to/config.toml --dry-run
```

Run the canonical `40³`, `tf = 10` case, or a short integration:

```bash
python -m applications.abl cases/Andren1994/config.toml
python -m applications.abl \
  cases/Andren1994/config.toml --max-steps 10 --overwrite
```

Use a separate output directory for smoke runs:

```bash
python -m applications.abl cases/Andren1994/config.toml \
  --max-steps 1 \
  --output /tmp/andren1994-smoke \
  --overwrite
```

Run the same configured physics through the staggered finite-volume path with
AB2 and the FFT pressure solver:

```bash
python -m applications.fv_abl cases/Andren1994/config.toml --dry-run
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m applications.fv_abl cases/Andren1994/config.toml \
  --max-steps 10 --output /tmp/andren1994-fv-smoke --overwrite
```

This FV realization uses AMD for momentum and its eddy viscosity for passive-
scalar diffusion; `resolved_case.json` and `summary.json` preserve that
distinction from the canonical LASD implementation. During the configured
statistics window it writes total momentum and scalar fluxes, signed AMD TKE
transfer, momentum and scalar diffusivities, streamwise spectra, total resolved
TKE history, and momentum-stationarity metrics. AMD has no prognostic SGS
TKE, so
the reported modeled SGS-TKE contribution is explicitly zero rather than an
inferred LASD quantity.

## FV warmup, precursor, and enforced-main workflow

The `[finite_volume_workflow]` table in the same case TOML supplies only stage
lengths, the recorded x-plane, chunking, and output location. The workflow
fixes the pressure and boundary choices required by each stage: warmup and
precursor are periodic and use FFT, while the enforced main run is open in x
and uses GMG. Display the resolved contract without starting JAX:

```bash
python -m applications.fv_abl.workflow \
  cases/Andren1994/config.toml --dry-run
```

Run the complete chain, or run each restartable stage separately:

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python -m applications.fv_abl.workflow \
  cases/Andren1994/config.toml --overwrite

python -m applications.fv_abl.workflow \
  cases/Andren1994/config.toml --stage warmup --overwrite
python -m applications.fv_abl.workflow \
  cases/Andren1994/config.toml --stage precursor
python -m applications.fv_abl.workflow \
  cases/Andren1994/config.toml --stage main
```

The precursor stores exactly one `yz` layer per time step in four memory-
mappable arrays under `precursor_inflow/`: the three staggered velocity
components and scalar. The main domain directly enforces the matching layer at
its inlet. At the outlet, tangential velocity and scalar use the three-point
second-order zero-gradient extrapolation; pressure projection selects the
normal outflow velocity using inlet-Neumann/outlet-Dirichlet pressure
conditions. Because x is not periodic in this stage, attempting to construct
its pressure solve with FFT is rejected.

For a short end-to-end smoke run, `--max-steps 2` caps every stage while
retaining the same boundary and backend choices.

The reference profile and published comparison envelope live under
[`reference`](reference/). Earlier figure-extraction and detailed budget tools
remain preserved in [`legacy/cases/Andren1994`](../../legacy/cases/Andren1994/).

Clean crops of all 19 published figures live under
[`reference/figure_panels`](reference/figure_panels/). Their source pages,
crop boxes, and active plot-axis registrations are recorded in
[`manifest.json`](reference/figure_panels/manifest.json).

Overlay a completed active run on the directly comparable profile panels:

```bash
python tools/overlay_andren1994.py \
  outputs/andren1994_lasd_40x40x40
```

This writes individual overlays for Figures 2 through 8, 11, 14, and 15, a
compact diagnostic sheet, and a complete 19-figure sheet under the
result directory's `paper_overlays/`. Figures 2, 3, 6, 8, 11, 14, and 15 use
the extended restartable diagnostics: total TKE, component momentum
stationarity, resolved-plus-SGS fluxes, signed SGS TKE transfer, SGS
diffusivities, and streamwise spectra. Reference-only panels are labeled
explicitly when an older run did not record their required observable. The
tool reads `history.csv`, `profiles.csv`, `spectra.csv`, and `summary.json`; it
does not run or reconfigure the solver. To reproduce the checked-in crops from
the DLR article scan:

```bash
python tools/overlay_andren1994.py \
  --extract-from-pdf /path/to/qj-1457-1994.pdf
```

# GABLS1 stable-boundary-layer benchmark

This case exercises the non-spectral staggered-MAC solver with the filter-free
AMD momentum and potential-temperature closures. It follows the first GABLS
LES intercomparison: a `400 m × 400 m × 400 m` periodic domain, geostrophic
wind `(8, 0) m/s`, Coriolis parameter `1.39e-4 s^-1`, `265 K` mixed layer to
`100 m`, an overlying `0.01 K/m` inversion, and a prescribed surface cooling
rate of `0.25 K/h`. Momentum and heat use the coupled stable Monin--Obukhov
wall law with `z0 = z0h = 0.1 m`.

The canonical public low-resolution comparison is `32³` (`12.5 m`) for nine
hours, with statistics accumulated over hours 8--9:

```bash
python benchmark/GABLS1/run.py \
  --output-dir benchmark_results/gabls1_amd_32cubed
```

For a true four-process horizontal decomposition, use the MPI y-slab runner:

```bash
PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/opt/homebrew/sbin \
mpiexec -n 4 env \
  PYTHONPATH="$PWD:$PWD/src" \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  XLA_FLAGS=--xla_cpu_multi_thread_eigen=false \
  python benchmark/GABLS1/run_mpi.py \
  --output-dir benchmark_results/gabls1_amd_32cubed
```

The layout is `1 × 4` in `y`: every rank owns `32 × 8 × 32` cells and keeps
complete ground-to-top vertical columns. AMD and MP5 exchange three periodic
halo rows with multi-host `ppermute`; stable MOST and vertical SGS fluxes stay
local. The matrix-free GMG/PCG pressure solve uses the same y slabs on fine
levels and globally replicates only its coarse level. Rank 0 reconstructs the
ordinary full-domain checkpoint and diagnostics, so serial and MPI runs can
restart each other's checkpoints.

Before committing a long run, exercise both short paths:

```bash
python benchmark/GABLS1/run.py --quick \
  --output-dir benchmark_results/gabls1_amd_quick

python benchmark/GABLS1/run.py --smoke \
  --output-dir benchmark_results/gabls1_amd_smoke
```

`--quick` advances four steps on `8³`. `--smoke` advances `16³` for 72 s.
Both retain the same active scalar, MOST wall coupling, AMD closures,
projection, diagnostics, and output path as the canonical case.

The corresponding four-rank quick run uses `16³` so each MP5 slab owns at
least three y cells:

```bash
mpiexec -n 4 python benchmark/GABLS1/run_mpi.py --quick \
  --output-dir benchmark_results/gabls1_amd_mpi_quick
```

Continue an interrupted run without losing accumulated samples:

```bash
python benchmark/GABLS1/run.py \
  --restart benchmark_results/gabls1_amd_32cubed/checkpoint.npz \
  --output-dir benchmark_results/gabls1_amd_32cubed
```

Replace `run.py` by `run_mpi.py` and add `mpiexec -n 4` to continue the same
checkpoint on four y-slab ranks.

The runner writes `checkpoint.npz`, cell-centred `profiles.csv`, face-centred
`flux_profiles.csv`, `time_series.csv`, `benchmark_stats.npz`,
`summary.{csv,json}`, `resolved_config.json`, and
`GABLS1_AMD_12p5m_comparison.png`. Flux profiles contain separate resolved,
SGS, and total contributions on their native vertical-face coordinates; the
comparison plot uses the total momentum and heat flux, rather than comparing
the resolved part alone with the official LES totals. The official total
envelope is formed participant by participant before taking its range. The
plot overlays our 8--9 h mean on the range and mean of the seven official
`12.5 m` submissions.

The archived official text files live under `reference/official_12p5m` with a
pinned SHA-256 and citation in `SOURCE.json`. To reproduce that extraction:

```bash
python benchmark/GABLS1/fetch_reference.py
```

The legacy Met Office host currently presents a mismatched TLS certificate;
the fetcher therefore permits that transport only while enforcing the pinned
archive digest before extracting a restricted set of `.dat` members.

Primary references:

- Beare et al. (2006), *An Intercomparison of Large-Eddy Simulations of the
  Stable Boundary Layer*, Boundary-Layer Meteorology 118, 247--272,
  doi:10.1007/s10546-004-2820-6.
- The original GABLS1 case description and formatted-data specification are
  mirrored by the official archive recorded in `reference/official_12p5m`.

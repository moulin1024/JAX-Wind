# GABLS1 stable-boundary-layer benchmark

This case exercises the non-spectral staggered-MAC solver with the filter-free
AMD momentum and potential-temperature closures. It follows the first GABLS
LES intercomparison: a `400 m × 400 m × 400 m` periodic domain, geostrophic
wind `(8, 0) m/s`, Coriolis parameter `1.39e-4 s^-1`, `265 K` mixed layer to
`100 m`, an overlying `0.01 K/m` inversion, and a prescribed surface cooling
rate of `0.25 K/h`. Momentum and heat use the coupled stable Monin--Obukhov
wall law with `z0 = z0h = 0.1 m`.

The default transport path is fully conservative Morinishi staggered S4
momentum, AMD, conservative KO6 cutoff damping, and a complete MP5
potential-temperature flux. Pressure uses the matching fourth-order
`D4=-G4.T` constraint and
`-D4G4` Poisson operator, with the compact second-order GMG retained only as a
preconditioner. The former second-order centered-plus-MP5 path remains available with
`--momentum-advection centered2 --momentum-regularization mp5
--scalar-advection centered_mp5 --pressure-discretization centered2`. The KEP4
operator identities and conservation qualification are recorded in
`doc/design/decisions/0012-kep4-ko6-mp5-transport.md`.

The canonical public low-resolution comparison is `32³` (`12.5 m`) for nine
hours, with statistics accumulated over hours 8--9:

```bash
python benchmark/GABLS1/run.py \
  --output-dir benchmark_results/gabls1_amd_32cubed
```

The default advective CFL target is `0.9`; the diffusive target remains `0.5`.
Use `--target-cfl 0.7` when reproducing an older, more conservative run.

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
complete ground-to-top vertical columns. KEP4, KO6, AMD, and scalar MP5 share
one packed three-row halo message per neighbor and stage; stable MOST and
vertical SGS fluxes stay local. The matrix-free GMG/PCG pressure solve uses
the same y slabs on fine levels and globally replicates only its coarse level.
Its exact fourth-order apply uses three pressure halo rows; GMG remains a
compact second-order preconditioner.
Rank 0 reconstructs the ordinary full-domain checkpoint and diagnostics, so
serial and MPI runs can restart each other's checkpoints.

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

The default coupled integrator advances momentum and temperature on the same
three SSPRK3 stages, reducing scalar RHS evaluations from six to three. The
legacy `strang` option retains two scalar SSPRK3 half-steps for comparisons:

```bash
python benchmark/GABLS1/run.py \
  --projection-method fpj2 \
  --coupling-integrator coupled-ssprk3 \
  --output-dir benchmark_results/gabls1_amd_coupled_ssprk3
```

Each coupled stage shares the MOST flux, cell-centred velocity, and velocity
gradient between the momentum and scalar closures. FPJ2 intermediate
velocities use a conservative, constant-preserving divergence correction for
scalar transport because their pressure is predicted rather than exactly
projected. The full-PPE path uses the same formulation, where the correction
vanishes to the pressure-solve tolerance.

The corresponding four-rank quick run uses `16³` so each MP5 slab owns at
least three y cells:

```bash
mpiexec -n 4 python benchmark/GABLS1/run_mpi.py --quick \
  --projection-method fpj2 \
  --coupling-integrator coupled-ssprk3 \
  --output-dir benchmark_results/gabls1_amd_mpi_quick
```

The distributed FPJ2 path uses three PPEs for each of its first two accepted
steps, then one PPE per step while the timestep ratio remains within
`--fpj2-timestep-ratio-limit`. Its checkpoint stores both distributed
pressure histories, so a restart continues directly on the one-PPE path.
Momentum and temperature share one packed three-row halo exchange per
direction and explicit stage.

## Performance profiling

Use `64³` when comparing algorithms; `32³` is too small for meaningful CPU
strong-scaling measurements. The stage profiler accepts either a checkpoint
or a freshly initialized grid:

```bash
python benchmark/GABLS1/profile_serial.py \
  --nx 64 --ny 64 --nz 64 \
  --projection-method full \
  --pressure-smooth 2 \
  --profile-repeats 5

python benchmark/GABLS1/profile_serial.py \
  --nx 64 --ny 64 --nz 64 \
  --projection-method fpj2 \
  --pressure-smooth 1 \
  --profile-repeats 5
```

`full` performs a pressure Poisson solve after every SSPRK3 momentum stage.
`fpj2` uses two exact startup steps and then two pressure predictions plus one
final Poisson solve. Choose `coupled-ssprk3` for the default three shared
momentum-temperature stages or `strang` for the original two scalar half-steps.

On the development CPU, the measured `64³` kernel time decreased from about
`0.604 s/step` (`full`, two GMG pre/post smooths) to about
`0.277 s/step` (`fpj2`, one smooth), a roughly 54% reduction. With the same
three-PPE `full` integrator, the optimized MP5 and scalar-stage kernels take
about `0.334 s/step`, so the formula-preserving kernel work accounts for most
of the gain. The one-smooth configuration retains the same PCG tolerance;
zero smooths were tested and rejected because the weaker preconditioner made
the complete solve slower.

The same comparison on a developed `64³` GABLS1 checkpoint (3.13 h) gives
`0.354 s/step` for `full` and `0.269 s/step` for `fpj2`. A 300-step fork from
that checkpoint reduced complete-runner time from 119.6 s to 93.3 s while
retaining an `L2` divergence of `4.31e-8`; total momentum and heat-flux profile
differences were below `4e-4` relative. Over the same 101.3 s of simulated time,
the `0.9` CFL target required 239 instead of 300 steps and reduced loop time
from 93.3 s to 78.3 s; final divergence remained `8.16e-8` and the maximum
scalar-budget residual was unchanged at about `6.5e-6`. For a long `64³` run,
resume the saved full-PPE checkpoint on the validated fast path with:

```bash
python benchmark/GABLS1/run.py \
  --nx 64 --ny 64 --nz 64 \
  --projection-method fpj2 \
  --restart benchmark_results/gabls1_amd_64cubed/checkpoint.npz \
  --output-dir benchmark_results/gabls1_amd_64cubed
```

On the same developed checkpoint, a paired seven-repeat kernel profile reduced
the FPJ2 step from `0.376 s` (`strang`) to `0.340 s`
(`coupled-ssprk3`), about 9%. An equal-time 239-step fork remained stable with
final divergence `8.30e-8`; boundary-layer height, friction velocity, and
surface heat flux agreed with the Strang fork to better than 0.01%. Complete
runner time was within measurement noise on the development CPU. The coupled
path is nevertheless the default here because it halves scalar RHS evaluations
and has the more favorable multi-GPU communication pattern.

The first two accepted steps after switching from a full-PPE checkpoint use
the exact three-PPE startup path to build the variable-step pressure history.

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
`summary.{csv,json}`, `resolved_config.json`, and a resolution-labelled
comparison PNG (`GABLS1_AMD_12p5m_comparison.png` for `32³`, or
`GABLS1_AMD_6p25m_vs_official_12p5m_comparison.png` for `64³`). Flux profiles
contain separate resolved, SGS, and total contributions on their native
vertical-face coordinates; the comparison plot uses the total momentum and
heat flux, rather than comparing the resolved part alone with the official LES
totals. The official total envelope is formed participant by participant
before taking its range. The plot overlays our 8--9 h mean on the range and
mean of the seven official `12.5 m` submissions.

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

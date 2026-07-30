# Yang, Lin & Zhou wind-tunnel benchmark

This case reconstructs the rated R9 operating point from Yang, Lin & Zhou,
[“An ML-based wind turbine blade design method considering multi-objective
aerodynamic similarity and its experimental
validation”](https://doi.org/10.1016/j.renene.2023.119625), *Renewable Energy*
220 (2024), 119625.

## Physical case

The experiment uses a 1:100 NREL 5-MW model. This configuration uses the
supplied facility dimensions: a test section 24 m long, 6 m wide, and 3.6 m
high. The paper extract itself reports the blockage ratio but not these three
dimensions. The model rotor diameter is 1.26 m and the tower/hub height used
here is 0.876 m. The turbine is placed at the reported test-section centre:
`(x, y) = (12, 3) m`.

The 6 m × 3.6 m cross-section gives a rotor blockage ratio of 5.773%, which
reproduces the paper's reported 5.8%. The paper reports uniform flow with
turbulence below 1%. At R9 the measured wind speed is 4.4 m/s, rotor speed is
480 rpm, thrust coefficient is 0.810, and power coefficient is 0.459.

All nine operating points from Tables 4 and 8 are preserved in
`reference/operating_points.csv`. The runnable first milestone is intentionally
limited to R9 because the text only tabulates a measured thrust coefficient
below one at that condition.

## Numerical case

The benchmark grid is a repository choice because the paper does not report a
LES grid or time step:

- domain: `24 × 6 × 3.6 m`;
- grid: `384 × 96 × 64`;
- spacing: `0.0625 × 0.0625 × 0.05625 m`;
- time step: `0.001 s`;
- rotor resolution: 20.2 cells across `y` and 22.4 cells across `z`;
- model: zero-yaw pure-thrust actuator disk with `C_T' = 1.57146`;
- molecular viscosity: `1.5e-5 m²/s`;
- production SGS: Lagrangian scale-dependent dynamic.

The default `paper-uniform` mode uses the paper's uniform 4.4 m/s flow,
free-slip tunnel floor, and a uniform downstream fringe.

The optional `measured-log` mode uses the separately supplied profile in
`reference/inflow_profile_override.csv`. Its unscaled fit is
`u_* = 0.122941 m/s` and `z_0 = 0.0161003 mm` for `κ = 0.4`; the benchmark
rescales `u_*` to 0.161404 m/s so the fitted velocity is 4.4 m/s at the
0.876 m hub. This override is not silently treated as paper data: its measured
8.44–10.9% turbulence contradicts the paper's below-1% uniform-flow statement.
The current case uses its mean profile only and does not synthesize those
turbulence intensities. The static fringe is disabled in this mode because its
current target is uniform.

## Run

Inspect the resolved paper baseline without importing JAX:

```bash
python benchmark/yang2024/run.py --dry-run
```

Run a reduced-grid plumbing check:

```bash
python benchmark/yang2024/run.py --quick
```

Run the production rated case:

```bash
env JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python benchmark/yang2024/run.py
```

Select the measured-profile override:

```bash
python benchmark/yang2024/run.py --inflow measured-log
```

Run the `128 × 32 × 64` measured-log precursor warm-up smoke check on four
CPU processes:

```bash
env PYTHONPATH=src JAX_PLATFORMS=cpu \
  XLA_FLAGS=--xla_force_host_platform_device_count=1 \
  mpiexec -n 4 python benchmark/ConcurrentPrecursor/benchmark_warmup.py \
  --config benchmark/yang2024/configs/warmup_128x32x64.toml \
  --warmup-steps 20 --timed-steps 100 --estimate-steps 4500
```

This warm-up configuration disables the turbine and fringe, uses the
hub-speed-normalized measured log fit, and diagnoses friction velocity with
the dynamic-neutral wall model. Both the initial log profile and pressure
gradient cover the full 3.6 m tunnel height; the fit is therefore extrapolated
above its 2 m measured range. It is a short numerical stability and throughput
check, not a statistically converged turbulent precursor.

Run the refined `256 × 64 × 128` case for 10000 steps at `dt = 0.001 s`,
sampling three flow-field slices every 100 steps and rendering 100-frame GIFs:

```bash
env PYTHONPATH=src:../bw1000_benchmark JAX_PLATFORMS=cpu \
  XLA_FLAGS=--xla_force_host_platform_device_count=1 \
  mpiexec -n 4 \
  python benchmark/ConcurrentPrecursor/run_warmup_diagnostics.py \
  --config benchmark/yang2024/configs/warmup_256x64x128.toml \
  --output-dir benchmark_results/yang2024_warmup_256x64x128_dt0010_10000_fulllog \
  --duration-seconds 10 --frames 100 --flow-height 0.876
```

Results are written to `benchmark_results/yang2024_r9/` and include
`summary.json` and `final_inlet_profile.csv`.

## Offline precursor inflow batches

`benchmark/ConcurrentPrecursor/run_native.py` now has a `precursor-dump` mode.
It advances only the precursor, with the actuator disk, cold source, and
fringe forcing disabled, and saves the five enforced-fringe target fields in
bounded time batches:

```bash
env \
  CUDA_VISIBLE_DEVICES=0 \
  JAX_PLATFORMS=cuda \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH=src:external/bw1000_benchmark \
  python benchmark/ConcurrentPrecursor/run_native.py \
    --mode precursor-dump \
    --config benchmark/yang2024/configs/warmup_256x64x128.toml \
    --output-dir benchmark_results/yang2024_precursor_inflow \
    --warmup-checkpoint \
      benchmark_results/yang2024_warmup_256x64x128_dt0001_10000_gpu/final_checkpoint \
    --allow-unconverged-warmup \
    --precursor-steps 10000 \
    --inflow-batch-steps 10 \
    --inflow-start-x 20 \
    --num-processes 1 \
    --process-id 0 \
    --local-device-id 0
```

The `--allow-unconverged-warmup` flag is needed for a checkpoint produced by
`run_warmup_diagnostics.py`, because that runner does not save the
sliding-window convergence marker. Omit it for a converged checkpoint from
`run_native.py`. A multi-GPU dump uses `mpiexec -n N` and requires a warmup
checkpoint saved with the same z process count.

Files are written to `<output-dir>/inflow_batches` as
`batch_<time-batch>_rank_<z-slab>.npz`. Each `inflow` array has layout
`(time, field, x_fringe, y, z_local)` and fixed field order
`u, v, w, theta, qv`. Samples are the pre-step states used by the existing
concurrent pipeline. `manifest.json` records the time coverage, shapes,
fringe index, dtype, and rank decomposition, and is written only after all
batches and the `precursor_final` restart checkpoint are complete.

`--inflow-batch-steps` bounds the size of each file; `--inflow-compress`
optionally reduces disk use at the cost of host compression time. The
validated readers are `read_inflow_manifest()` and
`load_local_inflow_batch()` in `wireles_jax.inflow_batches`. The current
reader expects the enforced run to use the same z process count as the dump.
For this grid and `--inflow-start-x 20`, a 10-step uncompressed batch is about
67 MiB on one GPU, or about 17 MiB per rank with four z ranks. All 10000
uncompressed steps still require about 66 GiB in total; batching bounds each
file and device-to-host transfer but does not reduce total samples.

## Equivalent liquid-nitrogen nozzle warmup experiment

The warmup diagnostic runner can switch an existing neutral-flow checkpoint
to a localized equivalent LN2 nozzle with temperature transport and ambient
buoyancy:

```bash
env \
  CUDA_VISIBLE_DEVICES=0 \
  JAX_PLATFORMS=cuda \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH=src:external/bw1000_benchmark \
  python benchmark/ConcurrentPrecursor/run_warmup_diagnostics.py \
    --config benchmark/yang2024/configs/warmup_256x64x128.toml \
    --checkpoint \
      benchmark_results/yang2024_warmup_256x64x128_dt0001_10000_gpu/final_checkpoint \
    --output-dir benchmark_results/yang2024_warmup_ln2_20gps \
    --duration-seconds 2 \
    --frames 100 \
    --flow-height 0.876 \
    --liquid-nitrogen-nozzle \
    --num-processes 1 \
    --process-id 0 \
    --local-device-id 0
```

The defaults represent a 20 g/s LN2 flow at 8 m/s from
`(12.15, 3.0, 0.876) m` in the `+x` direction. They apply `0.16 N` of
streamwise momentum and `7673.5 W` of equivalent cooling over
`sigma_x = sigma_r = 0.15 m`. Individual values can be changed with the
`--ln2-*` command-line options.

The runner recognizes a checkpoint made before thermal transport was enabled:
it imports the complete velocity/SGS state, then starts the equivalent cooling
and `ambient` thermal buoyancy from that state. In addition to the standard
ABL outputs it writes:

- `ln2_cold_plume_100frames.gif`;
- `ln2_cold_plume_centerplane_100frames.npz`;
- `ln2_final_centerplane_descent.csv`.

The warmup domain is periodic in `x`; the two-second example is intentionally
short enough to observe the first downstream passage without repeatedly
recirculating the cold plume. This experiment represents completed near-field
vaporization as a momentum source and heat sink. It does not add nitrogen mass
to the carrier phase.

### 8 x 4 x 2 m, 256 x 128 x 256 one-second test

The dedicated configuration
`benchmark/LiquidNitrogenHubJet/configs/warmup_8x4x2_256x128x256.toml.example`
uses an `8 x 4 x 2 m` domain, a `256 x 128 x 256` grid,
`dt = 0.0004 s`, and 2500 steps. The suffix is `.toml.example` only because
the repository ignores `*.toml`; the runner still parses it as TOML. On one
node with four GPUs, launch. Scaling the measured CPU reference CFL by the
twofold grid refinement and the `0.4` timestep ratio predicts a maximum CFL
of about `0.069`, leaving margin below `0.1`:

```bash
env \
  JAX_PLATFORMS=cuda \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  PYTHONPATH=src:external/bw1000_benchmark \
  mpiexec -n 4 \
  python benchmark/ConcurrentPrecursor/run_warmup_diagnostics.py \
    --config \
      benchmark/LiquidNitrogenHubJet/configs/warmup_8x4x2_256x128x256.toml.example \
    --output-dir benchmark_results/ln2_8x4x2_256x128x256_1s \
    --duration-seconds 1 \
    --frames 100 \
    --flow-height 1.0 \
    --liquid-nitrogen-nozzle \
    --ln2-x 1.0 \
    --ln2-y 2.0 \
    --ln2-z 1.0 \
    --ln2-sigma-x 0.15 \
    --ln2-sigma-r 0.15
```

This is a pure-jet case in quiescent air: the initial velocity is zero and
there is no pressure-gradient forcing. The only streamwise momentum comes from
the equivalent LN2 nozzle. The fringe zone is disabled for this initial
one-second test, so the streamwise boundary remains periodic. Each MPI process
automatically selects the GPU matching its MPI or Slurm local rank.

## Validation boundary

This milestone validates tunnel geometry, inflow, rotor thrust, and wake
plumbing. It cannot validate the paper's power coefficient or torque because
the HITSZ001 lift/drag polar tables needed by an actuator-line model are not
published. The CST airfoil coefficients, chord law, and twist law are reported,
but those alone are insufficient to reproduce the blade-element forces.

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
hub-speed-normalized measured log fit, and prescribes its fitted friction
velocity at the wall. Both the initial log profile and pressure gradient cover
the full 3.6 m tunnel height; the fit is therefore extrapolated above its 2 m
measured range. It is a short numerical stability and throughput check, not a
statistically converged turbulent precursor.

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

## Validation boundary

This milestone validates tunnel geometry, inflow, rotor thrust, and wake
plumbing. It cannot validate the paper's power coefficient or torque because
the HITSZ001 lift/drag polar tables needed by an actuator-line model are not
published. The CST airfoil coefficients, chord law, and twist law are reported,
but those alone are insufficient to reproduce the blade-element forces.

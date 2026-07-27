# Archived WiRE-LES JAX Prototype

> This implementation was moved from top-level `jax/` to `legacy/jax/` when
> the document-first reconstruction began. It is frozen as a regression and
> benchmark reference, not the architecture of the new solver. Commands below
> preserve their historical spelling; when running them from the repository
> root, replace a leading `jax/` with `legacy/jax/` and use
> `PYTHONPATH=legacy/jax` where an explicit import path is required.

This directory contains the frozen functional JAX prototype of the WiRE-LES
core solver. The C++ solver is archived under `legacy/cpp/`, and the original
CUDA Fortran solver remains under `legacy/fortran_cuda/`.

Run the commands below from the repository root.

Current scope:

- the default production time-step runner is single-process / single-device
- experimental single- and multi-process JAX z sharding for AB2 timestepping
  with classic Smagorinsky or LASD SGS; communication uses JAX collectives
- optional annular actuator-disk wind-tunnel model; blade-resolved/ALM turbine
  aerodynamics are not yet implemented
- functional state transition: `state_next, diagnostics = step(state, params, ops)`
- spectral derivatives in periodic `x/y`
- finite differences in `z`
- a single persistent-state layout for both local and sharded execution:
  all center fields have shape `(nx, ny, nz)`, while `w[k]` is the upper
  z-face owned by center cell `k`; the bottom face is an implicit boundary
  value and no global ghost planes are stored
- log-law / neutral Monin-Obukhov initial condition
- wall stress using the Fortran-style filtered wall velocity, Porte-Agel wall
  gradient correction, and neutral Monin-Obukhov log profile; the default
  `[physics].wall_stress_model = "dynamic_neutral"` computes `u_*` from the
  first off-wall filtered velocity, while `"prescribed_ustar"` fixes the stress
  magnitude from `u_fric`
- SGS models: classic Smagorinsky and LASD dynamic Smagorinsky
- potential-temperature and moisture transport with fixed-Prandtl or dynamic
  scalar LASD diffusivity, coupled to resolved Boussinesq buoyancy
- localized hub momentum/cooling sources and a periodic fringe for dry
  liquid-nitrogen wind-tunnel wake studies
- optional `spray_dpm` Lagrangian parcels with drag, gravity, finite-rate
  evaporation, convective heat transfer, prescribed shortwave/longwave
  radiation, and conservative two-way mass/momentum/energy coupling
- Fortran-style AB2/Euler time stepping by default; RK4 remains available as an
  explicit experimental option
- pressure-gradient body forcing computed as `u_fric^2 / (bl_height / z_i)`,
  with `--pressure-force` available as an explicit override
- optional Coriolis/geostrophic forcing
  `du/dt += f(v - V_g)`, `dv/dt += -f(u - U_g)` from physical
  `[physics].coriolis_f`, `geostrophic_u`, and `geostrophic_v`
- pressure projection with horizontal FFTs and a Fortran-style batched
  tridiagonal `z` solve

Run a smoke test:

```bash
python jax/run_single.py --nx 16 --ny 16 --nz 16 --steps 3 --log-every 1
```

Run a minimal continuously injected spray case:

```bash
python jax/run_spray_dpm.py \
  --nx 16 --ny 16 --nz 16 --steps 10 --dt 0.1 \
  --mass-flow-rate 0.1 --diameter 1e-4 \
  --diameter-distribution rosin-rammler \
  --minimum-diameter 2e-5 --maximum-diameter 3e-4 \
  --rosin-rammler-spread 3.0 \
  --turbulent-dispersion \
  --shortwave-flux 800 --shortwave-absorption 0.2 \
  --output benchmark_results/spray_dpm_smoke
```

Run the 1 m actuator-disk liquid-nitrogen hub-cooling baseline/control LES:

```bash
python benchmark/LiquidNitrogenHubJet/run.py
```

This benchmark treats unresolved flashing and near-nozzle entrainment through
measurable total momentum flux and effective cooling power.  It sets
`horizontal_homogeneous=false`, so buoyancy uses a fixed ambient reference,
LASD remains local/Lagrangian, and plane-averaged SGS closures are rejected.

The fixed-capacity parcel buffer is JIT-compatible. Each parcel carries
position, velocity, liquid mass, diameter, temperature, multiplicity, and an
active mask. Injection supports monodisperse, truncated Rosin-Rammler,
bounded lognormal, and API-configured tabulated mass distributions. With a
specified mass-flow rate, parcel multiplicity is computed separately for each
sampled diameter so every injection step represents exactly `mdot * dt` while
retaining the requested mass distribution. Radiation heats the droplet energy
equation; only convective droplet/air heat exchange is deposited as a
carrier-temperature source, so solar energy is not incorrectly counted as
air-derived latent cooling.

`--turbulent-dispersion` enables a correlated subgrid fluid-velocity-seen
model for parcels. Each parcel carries a persistent three-component
Ornstein-Uhlenbeck state and a persistent unsigned ID. Component variances are
estimated from the local test-filter Leonard energy and extrapolated below the
LES cutoff with inertial-range scaling; the correlation time is the physical
filter length divided by the diagnosed SGS velocity scale. The reconstructed
velocity is used in drag, Reynolds, Nusselt, and Sherwood numbers, so it affects
both motion and evaporation without adding white-noise position kicks. Random
keys depend on parcel ID, carrier step, and DPM substep rather than buffer slot,
making trajectories reproducible after parcel compaction or z-shard migration.
The option is disabled by default for deterministic regression compatibility.

Run the structured 128^3 neutral ABL case:

```bash
python jax/run_single.py --config jax/configs/neutral_abl_128.toml
```

Run the minimal pressure-driven neutral ABL with LASD and compare its developed
mean profile directly with the neutral log law:

```bash
python jax/run_pressure_driven_neutral_abl.py
```

Add `--gif` to write `profile_evolution.gif`, showing the horizontal-mean
velocity profile at every logging interval as it develops toward the log law.
Add `--field-gif` to write `velocity_three_sections.gif`, showing the speed
magnitude simultaneously on an `x-y` section near the wall and centered `x-z`
and `y-z` sections. Use `--field-gif-z-over-h` to move the horizontal section.
The default case uses a `2*pi*1000 m` square horizontal domain and a `1000 m`
vertical domain on a `64x64x64` grid. It uses `dt = 0.625 s` for 32000 steps
(20000 s physical duration). The runner aborts if any logged directional CFL
exceeds the default `--cfl-limit 0.1`.

The case starts from a uniform plug profile with the same bulk momentum as the
target log-law profile. The constant pressure gradient is parameterized by a
reference `u_*`, but the actual wall `u_*` is diagnosed every step from the
filtered first-level velocity by the same dynamic neutral wall model used in
the C++ implementation. LASD then redistributes momentum and develops the
vertical profile. The final quarter of the run is averaged. Results are written to
`outputs/pressure_driven_neutral_abl_lasd/` as `profiles.csv`, `summary.csv`,
and `profile_vs_loglaw.png`; the reported error uses
`0.05 <= z/H <= 0.30`. For a quick plumbing check, override the grid and time:

```bash
python jax/run_pressure_driven_neutral_abl.py \
  --nx 8 --ny 8 --nz 12 --steps 20 --log-every 10 \
  --average-start-step 10 --output-dir /tmp/pressure_abl_smoke --no-plot
```

Run a geostrophic-wind driven neutral Ekman case with dynamic neutral wall
stress:

```bash
python jax/run_single.py --config jax/configs/neutral_ekman_128.toml
```

The Ekman configs use `initial_condition = "geostrophic"` so the resolved
velocity starts from `(U_g, V_g)` instead of spinning up from the neutral log-law
profile.

Run the Nieuwstadt/Mason/Moeng/Schumann 1993 CBL benchmark with LASD for both
momentum stress and potential-temperature transport:

```bash
python benchmark/Nieuwstadt1993/run.py
```

The benchmark directory contains the solver driver, all case configurations,
digitized-data comparison tools, paper-figure overlays, and benchmark-specific
tests. A short complete workflow is:

```bash
python benchmark/Nieuwstadt1993/run.py --quick
```

The scalar LASD path dynamically computes the potential-temperature
diffusivity, applies the surface heat flux, and advances temperature before its
resolved fluctuation enters the vertical-momentum buoyancy term. Diagnostics
write resolved, SGS, and total heat-flux profiles separately. Every LASD
coefficient update enforces the one-cell departure-point condition
`cs_count * max(CFL_x, CFL_y, CFL_z) < 1` and reports a warning when it is
exceeded. The overall directional CFL warning threshold is `0.2`.

`configs/moeng.toml` remains the fixed-Smagorinsky regression case.

Run the high-resolution 256^3 Moeng-style dry CBL case on a single GPU:

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
python benchmark/Nieuwstadt1993/run.py \
  --config benchmark/Nieuwstadt1993/configs/moeng_256.toml \
  --output-dir benchmark_results/Nieuwstadt1993_moeng_256
```

The 256^3 config is intended for the benchmark driver and keeps the comparison
outputs together under the selected benchmark result directory.

The reference paper markdown is stored in
`benchmark/Nieuwstadt1993/reference/Nieuwstadt1993.md`. The integration
test validates that markdown, runs the benchmark runner, checks the diagnostic
outputs including Fig. 9 dissipation
`<epsilon> z_i / w_*^3`, and compares the full run against Moeng Table 3
targets:

```bash
python benchmark/Nieuwstadt1993/tests/integration_moeng.py --quick
python benchmark/Nieuwstadt1993/tests/integration_moeng.py
WIRELES_JAX_FULL_INTEGRATION=1 \
  pytest benchmark/Nieuwstadt1993/tests/test_moeng.py
```

Run the single-node sharding pressure smoke test:

```bash
python jax/run_sharded_pressure_smoke.py --nx 128 --ny 128 --nz 128 --devices 4
```

The sharded pressure path uses the same z-slab to y-slab redistribution as the
Fortran solver: the horizontal FFT uses a Fortran-compatible
`(nx/2+1, ny, nz)` spectral layout, each device initially owns full horizontal
planes for a local z slab, `lax.all_to_all` redistributes to local y slabs with
complete z lines, and the tridiagonal pressure solve runs locally on those
complete z lines. The smoke test validates FFT roundtrip, z/y all-to-all
roundtrip, and sharded pressure solve agreement with the local reference.

Run the z-sharded timestep path on all devices visible to one process:

```bash
python jax/run_sharded.py \
  --config jax/configs/neutral_abl_128.toml \
  --devices 4
```

The sharded timestep path stores physical interior arrays with shape
`(nx, ny, nz)` as global metadata and shards them along `z`. No process
constructs or owns the global state: initialization callbacks create only
addressable z slabs, and pressure coefficients are created only for the local
y pencils. Each device builds one-plane local halos with JAX collectives for
vertical finite differences, convection, scalar transport, buoyancy,
stress-divergence, diagnostics, and pressure correction. The pressure solve
uses a distributed z-slab to y-pencil all-to-all, so each process holds full z
lines only for its local subset of y modes. Slab interfaces need one exchanged
plane per side, and no device owns a duplicated persistent face or ghost cell.
Fields needed at the same stage are packed into one lower and one upper halo
collective. Momentum and scalar transport share the filtered velocity,
velocity-gradient, potential-temperature, moisture, and strain bundle.
Momentum LASD is halo-aware. The distributed thermo path supports centered
conservative transport of potential temperature and water-vapor mixing ratio,
fixed turbulent Prandtl/Schmidt numbers, globally conservative moisture
positivity correction, and virtual-potential-temperature buoyancy. AB2 is the
supported distributed time integrator.

Spray parcel storage also has a distributed z-slab implementation. Its
`max_parcels` setting is capacity per shard; the global shape is metadata only,
and initialization creates only the addressable local buffer on each process.
After parcel motion, `make_migrate_sharded_spray` uses packed nearest-neighbor
`ppermute` exchanges and stable local compaction to transfer ownership, remove
physical-domain exits, and report capacity overflow. Repeated neighbor passes
support a parcel crossing more than one slab in a step without gathering a
global parcel array. Distributed parcel ownership/migration is available now;
evaporation, drag, sensible heat exchange, radiation, and two-way carrier
coupling are evaluated locally on the owning slab. CIC source weights that
cross a slab interface are packed and returned to the neighboring grid owner,
so liquid loss equals the globally deposited vapor mass without gathering a
carrier or parcel field.

Run the coupled path on the devices visible to one process:

```bash
python jax/run_spray_dpm_sharded.py \
  --nx 32 --ny 32 --nz 64 --devices 4 \
  --steps 20 --dt 0.1 --mass-flow-rate 0.1 \
  --diameter-distribution rosin-rammler \
  --turbulent-dispersion
```

For multiple processes, launch the same command on every process with a shared
`--coordinator-address`, the global `--devices` count, `--num-processes`, and a
unique `--process-id`. Pressure operators remain explicit distributed runtime
arguments; they are never captured as global JIT constants.

The default pressure backend is the full-field y-pencil transpose. A compact
SPIKE backend keeps pressure in the z-slab layout and exchanges only block
interface values:

```bash
python jax/run_sharded.py --config case.toml --devices 4 \
  --sharded-pressure-solver spike
```

SPIKE uses the same eliminated `nz`-row Neumann pressure matrix as the
transpose solver and is checked against it mode by mode. It stores local block
factors plus a mode-sharded `2P x 2P` interface inverse.

For two processes, initialize the same global mesh from both launch contexts:

```bash
# process 0
python jax/run_sharded.py --config case.toml --devices 2 \
  --coordinator-address node0:12345 --num-processes 2 --process-id 0 \
  --local-device-ids 0

# process 1
python jax/run_sharded.py --config case.toml --devices 2 \
  --coordinator-address node0:12345 --num-processes 2 --process-id 1 \
  --local-device-ids 0
```

Run the short 128^3 version with `steps=10` and `log_every=1`:

```bash
python jax/run_single.py --config jax/configs/neutral_abl_128_short.toml
```

Command-line options still override values loaded from `--config` for ad hoc
experiments, but standard cases should keep those values in TOML files.

The runner uses double precision for the resolved velocity/pressure path by
default. SGS uses mixed precision by default (`[sgs].precision = "float32"` or
`--sgs-precision float32`), storing LASD history and evaluating SGS stress in
float32 before it is combined back into the resolved RHS. Set
`[sgs].precision = "float64"` for an all-double SGS path, or `"default"` to
follow the main `[runtime].precision`. Use `--single` only for quick full-float32
smoke tests.
Use physical `--bl-height` and physical `--zo`; the vertical forcing cap is
applied at `bl_height / z_i`.
The user-facing `lx`, `ly`, and `lz` values are physical lengths, and
user-facing `dt` is physical seconds. The runner normalizes them internally by
`z_i`; if `z_i` is omitted, it defaults to physical `lz`. Pass `--z-i` only when
the inversion height differs from the domain height.
The neutral wall law requires the first off-wall reference height
`0.5 * dz * z_i` to be larger than `zo`; otherwise `log(z/zo)` is non-positive.
Passing `--dt` explicitly overrides the scheme-specific default with a physical
time step. Internally the solver uses `dt / z_i`; omitted `dt` keeps the
scheme-specific internal defaults.
Runtime logs include elapsed, estimated remaining, and estimated total wall-clock
seconds; each log point synchronizes JAX work before computing the estimate.
Diagnostics are computed only at those log points, not at every time step.
Pass `--profile` or set `[profiling].enabled = true` to run AB2 split-step
profiling. Profiling mode runs the configured calculation, then prints one final
average per-step timing report. The report breaks RHS into
`velocity_xy_derivatives`, `wall_z_derivatives`, `convection`, `sgs_strain`,
`lasd_coefficients`, `sgs_stress`, `sgs_total`, `stress_divergence`, and
`rhs_assembly`; pressure projection is split into
`projection_divergence`, `pressure_solve`, `pressure_gradient_ifft`, and
`projection_update`. It also reports `ab_update`, `state_pack`, `solver_total`,
and `diagnostics`. `--profile-warmup` /
`[profiling].warmup` excludes the first completed steps from the averages.
`--profile-steps` / `[profiling].steps` is optional and limits the profiled run
length; when omitted, profiling uses `[time].steps`. It synchronizes after each
component, so use it to identify bottlenecks. Normal AB2+LASD runs compile
separate skip/update step kernels so non-update steps do not carry the full LASD
coefficient update path.
Set `[postprocess].dump_fields = true` or pass `--dump-fields` to write velocity
snapshots at the same log points. Each snapshot is an HDF5 file named
`<field_output_prefix>_step_<step>.h5` under `field_output_dir`, with datasets
`fields/u`, `fields/v`, and cell-centred `fields/w`, all with shape
`(nx, ny, nz)`. `fields/w_face` contains the reconstructed `nz+1` physical
faces, including the implicit bottom face. `coords/z_center` and
`coords/z_face` identify the two vertical locations. Set
`[postprocess].field_dump_start_step = N` to skip field dumps before step `N`
while keeping the normal log output.
HDF5 attributes `dt` and `time` are physical seconds; `dt_scaled` and
`time_scaled` record the internal normalized values.
Use `jax/postprocess_profile.py` to compute the horizontally averaged `u`
profile and compare it against the neutral log law:

```bash
python jax/postprocess_profile.py \
  --input-dir jax_fields \
  --pattern 'fields_step_*.h5' \
  --config jax/configs/neutral_abl_128.toml \
  --average last \
  --output u_profile.png
```

The profile script drops the top physical boundary plane by default, matching
the Fortran log/profile convention. Pass `--include-top-boundary` if you need to
inspect the raw top plane written in the HDF5 dump.

Use `jax/postprocess_cross_section.py` to render an `x-z` cross-section
animation at the middle `y` index:

```bash
python jax/postprocess_cross_section.py \
  --input-dir jax_fields \
  --pattern 'fields_step_*.h5' \
  --config jax/configs/neutral_abl_128.toml \
  --component u \
  --frames-dir xz_frames \
  --gif u_xz.gif \
  --fps 12
```

Pass `--component v`, `--component w`, or `--component speed` to render another
field. Use `--y-index` to choose a different `y` plane, `--start-step` /
`--end-step` to restrict the time range, and `--max-frames` to uniformly
subsample long runs before writing the PNG frames and GIF.

JIT runs also report explicit precompile lower/compile/done status before the
time-stepping table starts.
For RK4, `[numerics].projection_mode = "stage"` projects every intermediate
stage, while `"final"` skips intermediate projections and only projects the
full-step velocity as a faster experimental path.
Horizontal derivative and wall-filter FFTs are batched where the same spectral
transform can serve multiple components.
The `[sgs]` table accepts `model = "smagorinsky"` or `model = "lasd"`.
`cs_count` controls the LASD coefficient update interval and defaults to the
Fortran case value of 10. The structured ABL configs use LASD by default; plain
CLI defaults keep Smagorinsky for cheap smoke tests.
Momentum velocity derivatives follow the Fortran `ddxy_filter` data path when
`[numerics].horizontal_dealias = true`: the velocity field is first filtered and
the same filtered spectrum is used for `ddx/ddy`. The cutoff uses the Fortran
`fgr` box rule (`nint(nx/(2*fgr))`, `nint(ny/(2*fgr))`), so the default
`fgr=1.5` is the usual 2/3-rule grid filter. Wall-velocity filtering follows the
Fortran wall path and uses `fgr*tfr`; `tfr=2` is the test-filter ratio used by
the original dynamic SGS path. Stress divergence, pressure projection, and
diagnostic divergence keep the ordinary unfiltered derivative path. The pressure
solve leaves Nyquist pressure modes enabled by default via
`[numerics].pressure_filter_nyquist = false`, so the projection and diagnostic
divergence operators remain consistent to roundoff. Enable it only when you need
to reproduce the Fortran pressure Nyquist zeroing; that compatibility mode can
leave a small diagnostic divergence in the filtered horizontal Nyquist modes.

The archived `legacy/jax/` directory intentionally has no `__init__.py` so it
does not shadow the installed Google JAX package when invoked by path.

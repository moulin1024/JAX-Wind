# Independently controlled periodic wind farms

Use `jaxwind run` with `[physics.wind_farm.controller]` and one
`[[physics.wind_farm.layout]]` entry per turbine. `[physics.turbine]` supplies
the shared blade/polar geometry and operating pitch; its position/RPM are
template fields, not an extra turbine. Layout entries supply unique IDs,
positions, hub heights, and initial RPM. All turbines currently share one
rotor type and controller settings, but have independent dynamic states.

Examples:

- `cases/HornsRev1/fv_v80_two_turbine_tsr.toml`: two V80s separated by 7D,
  in the 512 x 512 x 256 domain, 16 x 16 x 4 m cells, one simulated hour.
- `cases/HornsRev1/fv_v80_two_turbine_tsr_smoke.toml`: inexpensive 120-second
  two-turbine validation on 64 x 16 x 256 cells at the same cell size.

Run commands **on a compute node**:

```bash
export JAXWIND_V80_FAST="$PWD/cases/HornsRev1/turbines/V80/CustomRotor.fst"
jaxwind check cases/HornsRev1/fv_v80_two_turbine_tsr_smoke.toml
jaxwind run cases/HornsRev1/fv_v80_two_turbine_tsr_smoke.toml
python tools/plot_wind_farm_control.py outputs/two_v80_ideal_tsr_gpu_smoke
# Exact continuation of a paused run (preserves every controller state):
jaxwind resume outputs/two_v80_ideal_tsr_gpu_smoke
```

## Controller model: `ideal-tsr`

Alternatively, `model = "wind-speed-lookup"` prescribes numerical RPM and
reported electrical power from the same filtered upstream wind. See the
[V80 lookup case](../cases/HornsRev1/v80_lookup.md) for table provenance,
one-diameter plane sampling, cut-in/out behavior, outputs, and limitations.
Lookup mode has no TSR/speed-servo parameters; it requires equal-length
`wind_speed_m_s`, `rpm`, `power_w` arrays, `cut_in_m_s`, `cut_out_m_s`,
`rpm_source`, and `power_source`, alongside `wind_filter_seconds` and
`probe_distance_diameters`. The complete operating interval must be covered
by strictly increasing wind-speed knots. These tables are embedded in the
resolved case, so changing a value invalidates exact resume fingerprints.
The remainder of this section describes only `ideal-tsr`.

The supplied V80 deck contains no drivetrain inertia, generator-torque law,
or optimal-TSR calibration. This implementation is an idealized speed servo,
**not** a physical rotor/generator torque-balance simulation. It does not
derive an optimum from the blade data: supply `target_tsr` from an appropriate
rotor performance curve. The example value 7 and response/filter/rate settings
are demonstration parameters, not a validated manufacturer controller.

For each turbine independently:

1. Sample positive streamwise velocity over an upstream circular rotor-area
   probe (`probe_distance_diameters`, default is not implicit). The probe is
   cell-area weighted in y/z and Gaussian weighted in x, using a width of at
   least one x cell. It follows the horizontal periodic topology. This is a
   local upstream estimate, potentially affected by neighboring wakes/induction;
   it is not an induction-corrected free-stream estimator.
2. Filter the measurement with time constant `wind_filter_seconds`:
   `U_filtered += (1 - exp(-dt/tau_w)) * (max(U_probe, 0) - U_filtered)`.
3. Set `omega_target = clip(target_tsr * U_filtered / R, omega_min, omega_max)`.
4. Apply a first-order speed response with `response_seconds` and limit the
   change to `maximum_acceleration_rpm_s * dt` (converted to rad/s).

Filtering/control advances once per **actual adaptive flow timestep**, not
per output frame or RK stage. The average of old/new rotor speed is held
across the flow RK substages while blade forces use each substage velocity.
The servo/flow coupling is first-order partitioned, although the flow uses
RK3. There is no instantaneous RPM jump. Zero/reverse wind drives the target
toward minimum RPM after filtering. TSR is reported as zero with `tsr_valid=0`
when the filtered wind is effectively zero. RPM saturation means the target
TSR may be unattainable; there is no above-rated pitch/power regulation,
generator torque, drivetrain inertia, cut-out logic, or yaw controller.

## Output and restart

`history.csv` includes `turbine_ID_rpm`, `target_rpm`, `probe_wind_m_s`,
`filtered_wind_m_s`, `tsr`, and `tsr_valid` per turbine, plus
`controller_dt_seconds`. Diagnostics are taken at the runtime observation
boundaries, not every RK substage. Farm CFL diagnostics use the actual last
step duration. `checkpoint.npz` includes the complete RPM/filter arrays;
exact resume validates the case fingerprint and retains turbine ordering.
An ordinary same-grid `AtmosphericSolution` warmup can be supplied with
`initial_conditions.checkpoint`; the flow is retained and controller RPMs
start from the layout values, with filters initialized from the loaded flow.
Initialization from a controlled-farm checkpoint requires unchanged layout
and ordering; exact `resume` also requires unchanged controller parameters.
The standard flow-plane outputs still use the template turbine x/y/z view;
they are not separate per-turbine movies.

Forces are summed sequentially with a JAX scan, avoiding an N-turbine stack
of full-domain source fields. Computational cost still grows with turbine
count. The 80-turbine Horns Rev layout and its performance have not been
validated by the two-turbine test.

## Current supported scope

The [80-turbine uniform-inflow main case](../cases/HornsRev1/main_uniform10.md)
adds direct neutral `wind-speed-lookup` farms with a uniform inlet, lateral
and downstream pressure outlets, and GMG on a uniform mesh. It uses fixed
timesteps and fast-RK3, no precursor/checkpoint initialization, and the same
shared rotor geometry/pitch. Its probes and force patches are nonperiodic.
The following original scope statement applies to the periodic/TSR path.

Direct periodic Boussinesq runs, upright streamwise AD-BEM rotors, stationary
advection frame, shared rotor geometry/pitch, and disabled nacelle/tower drag.
Unsupported inflow/cooling configurations and legacy workflow stages are
rejected explicitly. Single fixed-RPM turbine cases remain unchanged.
This is a controller/farm foundation, not a complete V80 drivetrain model.

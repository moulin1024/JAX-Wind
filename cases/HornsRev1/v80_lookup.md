# V80 wind-speed lookup mode

Use `fv_v80_two_turbine_lookup.toml` for the 512 x 512 x 256 domain
(8192 x 8192 x 1024 m), or `fv_v80_two_turbine_lookup_smoke.toml` for a
20-second implementation smoke test. These are two-turbine examples, not
the complete 80-turbine Horns Rev layout. Existing TSR cases are unchanged.

Each rotor samples streamwise velocity over a rotor-sized circle centered
80 m upstream (1D). Cell-area weights approximate the circle; linear
interpolation between neighboring x cell centers locates the plane, including
across the periodic boundary. A separate 5-second first-order wind filter
is advanced with each actual flow timestep. This filter constant is a
modeling choice, not a manufacturer setting.

The same filtered wind drives piecewise-linear interpolation of both the
RPM and electrical-power tables. RPM is prescribed directly to AD-BEM at
each step, with no TSR target, rotor inertia, or additional speed servo.
The new RPM is held across RK substages. Initialization uses the initial
local wind, ignoring layout `initial_rpm` in this mode. Resume preserves
the filter state; inline tables and provenance are part of the case fingerprint.

## Data provenance and accuracy

- RPM: approximate manual readings from **Hansen et al. (2012), Figure 2b**,
  PDF page 21, DOI [10.1002/we.512](https://doi.org/10.1002/we.512),
  [DTU manuscript](https://backend.orbit.dtu.dk/ws/files/6499345/paper0149_R2.pdf#page=21).
  The plotted solid RPM line and its m/s axis were visually inspected.
  Readings are rounded to 0.1 rpm; that precision is not a measurement-error
  claim. These are approximate averaged operating values, not verified OEM
  controller setpoints or an explicitly separated unwaked lookup table.
  Values from 23 to 25 m/s hold the last plotted value, 18.1 rpm, as an
  explicit extension assumption. This replaces the earlier example 16.7-rpm
  cap only in the new lookup cases.
- Power: numerical V80 `power_curve` in
  [DTU PyWake hornsrev1.py](https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake/-/raw/master/py_wake/examples/data/hornsrev1.py),
  retrieved 2026-09-15. The 4-25 m/s entries are converted from kW to W and
  embedded verbatim numerically in the case. No density correction is applied.
- Operating limits are explicit modeling choices: output is zero below
  4 m/s and at/above 25 m/s; both quantities jump at these boundaries.
  The nonzero table endpoint at 25 m/s defines interpolation just below
  cut-out, but is masked to zero at cut-out. No startup/shutdown hysteresis
  or extrapolation outside the operating interval is used.

## Outputs and limitations

`history.csv` contains per-turbine `rpm`, `target_rpm`, `probe_wind_m_s`,
`filtered_wind_m_s`, and `lookup_power_w`, plus `farm_lookup_power_w` (sum
of turbine lookup powers). Power is the **reported lookup electrical output**,
not aerodynamic shaft power, an AD-BEM validation result, or energy integrated
over time. The AD-BEM forces remain computed from blade geometry, fixed pitch,
local rotor flow, and prescribed RPM. Lookup power does not rescale the forces
or guarantee mechanical/electrical energy consistency. A stopped rotor can
still exert forces; zero RPM/power is not a feathered shutdown model.

The 1D probe can contain rotor induction and upstream wakes. It is deliberately
used as requested without a free-stream correction; that approximation and the
coarse rotor-area sampling should be checked before quantitative farm studies.
Above-rated fixed-pitch AD-BEM wakes are not validated by imposing rated lookup
power. Wind direction/yaw support is unchanged: streamwise upright rotors only.

On a compute node:

```bash
export JAXWIND_V80_FAST="$PWD/cases/HornsRev1/turbines/V80/CustomRotor.fst"
python -m jaxwind check cases/HornsRev1/fv_v80_two_turbine_lookup_smoke.toml
python -m jaxwind run cases/HornsRev1/fv_v80_two_turbine_lookup_smoke.toml
python tools/plot_wind_farm_control.py outputs/two_v80_lookup_gpu_smoke
```

## Implementation verification (2026-09-15)

- All 25 tests in `tests/test_wind_farm_control.py` passed on an A100,
  including the existing TSR tests, lookup endpoints/interpolation, malformed
  tables, independent probes, periodic plane placement, filter timestep
  consistency, runtime power history, and checkpoint/resume equivalence.
- The full 512 x 512 x 256 lookup case passed `python -m jaxwind check`.
  The full-size one-hour simulation was not launched.
- The small two-turbine smoke run completed 100 steps (20 simulated seconds)
  in 34.40 wall seconds including startup/compilation; final CFL was 0.2025.
  This is an implementation smoke test, not developed-wake validation.
- At the last sample: T01 had filtered wind 7.81206 m/s, 15.96171 rpm,
  and 651646.56 W; T02 had 7.81652 m/s, 15.96974 rpm, and 652699.88 W.
  Farm power was 1304346.44 W. Both RPM and power matched independent NumPy
  interpolation at all 10 recorded samples, and numeric checkpoint arrays
  were finite.
- Output: `outputs/two_v80_lookup_gpu_smoke/`; the generated and visually
  checked `turbine_control.png` includes RPM, wind, TSR, and lookup power.

# Two-V80 ideal-TSR controller validation

GPU smoke run completed on 2026-09-15 (A100, allocation 30244593).
Configuration: `fv_v80_two_turbine_tsr_smoke.toml`.

- 64 x 16 x 256 cells at 16 x 16 x 4 m; two V80s separated by 560 m (7D).
- 120 simulated seconds, 620 actual adaptive steps, 20 saved flow frames.
- Initial rotor speeds: T01 = 8 rpm, T02 = 14 rpm.
- Example controller target TSR = 7; wind filter = 5 s; response = 10 s;
  speed range 0–16.7 rpm; acceleration limit 0.5 rpm/s.
- Final T01: 8.7316 rpm; filtered wind 5.2398 m/s; TSR 6.9802.
- Final T02: 9.2372 rpm; filtered wind 5.3898 m/s; TSR 7.1790.
- Full final state and all saved frames are finite.
- Maximum sampled divergence: 6.71e-8 1/s; maximum sampled actual-dt CFL: 0.236.

Outputs are in `outputs/two_v80_ideal_tsr_gpu_smoke/` relative to the repository
root. `turbine_control.png` plots each turbine against its own time-varying
target, local filtered wind, and achieved TSR. `history.csv` preserves those
measurements and `checkpoint.npz` includes the rotor/filter states.

This short, narrow, periodic domain is for software validation only. Wakes
wrap around and affect both upstream probes. It is not a converged two-turbine
performance result or the full Horns Rev farm. The larger two-turbine example
passed `jaxwind check` but was not run as part of this implementation.

The first GPU attempt exposed a pre-existing float32 observation-boundary
stall at 109.2 s. The scheduler now uses the same tolerance as sampling to
avoid revisiting a completed boundary. The run was replayed from its saved
initial checkpoint and completed after that correction.

## Regression scope

Eight controller tests passed, along with 18 relevant existing turbine/runtime/
workflow tests (26 total). The three unrelated failures below were excluded
from the final broader rerun. The added stopped-rotor force check also passed.

Controller tests cover independent TSR convergence, timestep scaling, wind
filtering, RPM/slew limits, zero/reverse wind, dynamic aerodynamic force,
multi-turbine force sums, schema round trips/rejection, exact checkpoint and
runtime resume, warmup initialization, and the float32 scheduler boundary.

Broader testing also encountered three unrelated legacy/configuration failures
which are not repaired here:

- `test_low_mach_continuation_consumes_new_stage_checkpoint`: configured record
  plane lies outside the test mesh.
- `test_hitsz_main_is_fixed_fast_rk3_with_native_turbine_and_frames`: fixture
  duration is 180 s while the assertion expects 90 s.
- `test_fixed_warmup_blocks_anchor_float32_time_to_step_count`: legacy
  `runtime/periodic.py` lacks a `dataclass` import.

See `doc/wind-farm-control.md` for implementation assumptions and limitations.

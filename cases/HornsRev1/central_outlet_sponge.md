# Central advection with a downstream damping zone

This experiment follows the request to stop the rotor-local stabilization trial
and test a damping zone instead. Use
`fv_v80_small_central_sponge.toml` for the small single-turbine case.

```toml
[numerics]
momentum_advection_scheme = "central"
time_integration = "rk3"
outlet_backflow = "energy"
outlet_sponge_start_fraction = 0.75
outlet_sponge_timescale_seconds = 5.0
```

The original AD-BEM force projection is retained. The experimental widened
force projection and rotor-local momentum damping are disabled. The periodic
precursor has no outlet sponge. Porté-Agel remains enabled in the DTU log-law
comparison.

For x > 0.75 Lx the extra momentum tendency is

`-sigma(x) * (velocity - target)`,

with `sigma = ((x - 0.75 Lx)/(0.25 Lx))^2 / 5 s`. It is zero upstream.
The target is the current inlet's spanwise-mean u(z), v(z), with w=0.
Thus the buffer follows the incoming vertical shear instead of imposing a
height-independent wind. The tendency is included before each RK-stage pressure
projection. The sponge changes momentum and attenuates turbulence inside its
buffer; that region should not be used as a physical wake-recovery measurement.

The zone starts at x=1536 m in the 2048 m single-turbine domain and at x=3072 m
in the 4096 m turbulent ABL domain. The validation records all four streamwise
quarters separately, so the first three are outside the buffer and the last is
inside it. Near-wall profile and wall-stress checks are included.

Run the full paired validation:

```bash
bash tools/submit_backflow_validation.sh \
  --corrected-scheme central \
  --sponge-start-fraction 0.75 --sponge-timescale 5
```

Inside an existing GPU allocation, run the Python helper directly with the same
options plus a fresh `--output` directory. Defaults are 1800 s spin-up and 3600 s
averaging for the log-law comparison, followed by two 300 s single-V80 runs.
The source/input snapshot and detailed reports are saved under that output.

A downstream sponge targets disturbances reaching the outlet. Earlier small-case
oscillations were also observed before the wake reached the outlet, so a sponge
must be tested rather than assumed to remove every upstream grid-scale mode.
Seven focused tests pass: equilibrium to float32 roundoff, compact upstream
support, perturbation damping, stage projection/inlet preservation, and invalid
parameter rejection. Two initially overstrict equality tolerances were changed
to account for float32 spanwise-reduction roundoff (less than 3e-7 m/s²).

## Completed result, 2026-09-16

The interactive A100 run completed both the 5400 s ABL experiment (1800 s
spin-up + 3600 s averaging) and matched 300 s single-turbine cases.
**Log-law checks pass; the upstream-oscillation check fails.**

| Metric | Central baseline | Central + outlet sponge |
|---|---:|---:|
| ABL 20–100 m log-law RMSE [m/s] | 0.611299 | 0.613041 |
| ABL wall u* [m/s] | 0.430758 | 0.428006 |
| Final wake upstream second-difference RMS [m/s] | 0.0397812 | 0.0398294 |
| Final maximum upstream adjacent-face jump [m/s] | 0.182971 | 0.184551 |
| Minimum sampled downstream outlet u [m/s] | 3.32188 | 9.52201 |
| Peak lateral inward volume flux [m³/s] | 5369.49 | 21230.23 |

Corrected-minus-central profile RMSE in the four streamwise quarters is
0.000970, 0.001427, 0.003035, and 0.013236 m/s. The first three quarters are
outside the sponge. The maximum near-wall quarter-profile change is 0.13435 m/s,
and wall-u* changes by 0.639%; both satisfy the declared regression tolerances.
The existing central log-law bias is retained, not eliminated.

The sponge attenuates the wake before the outlet, but upstream oscillation RMS
increases by 0.12% instead of meeting the required 90% reduction. Both wake cases
have zero sampled downstream reverse-flow area, so this is not evidence that
pre-existing downstream backflow was eliminated. Restoring outlet velocity also
increases lateral entrainment; lateral inward flow is not streamwise wake
reentry. Maximum sponge-wake CFL is 0.3393, sampled divergence 1.49e-7 /s,
and inlet error zero. The damping zone is therefore available and preserves the
central ABL profile in this test, but does not solve the upstream oscillation.

- [Log-law report](../../outputs/backflow_sponge_validation_30270853/analysis/README.md)
- [Log-law plot](../../outputs/backflow_sponge_validation_30270853/analysis/log_law_comparison.png)
- [Wake comparison](../../outputs/backflow_sponge_validation_30270853/wake_corrected/backflow_analysis/upstream_comparison.png)
- [Wake movie](../../outputs/backflow_sponge_validation_30270853/wake_corrected/v80_single_hub_u.mp4)
- [Numerical metrics](../../outputs/backflow_sponge_validation_30270853/analysis/report.json)

Postprocessing labels were corrected after integration to distinguish the
passing log-law checks from the failed combined wake screen. Numerical results
and acceptance thresholds were not changed.

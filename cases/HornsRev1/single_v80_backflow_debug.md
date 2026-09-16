# Small single-V80 outlet debugging

Use `fv_v80_small_central_outlet_debug.toml` and
`fv_v80_small_central_backflow_debug.toml` for the ongoing outlet investigation.

- One V80 at (512, 512, 70) m.
- 128 × 64 × 64 cells (524,288): 128 times fewer than the full farm.
- Domain 2048 × 1024 × 256 m; retained spacing 16 × 16 × 4 m.
- Uniform initial/inlet u=10 m/s, v=w=0; AMD and the inherited rough wall.
- Central advection, full RK3, GMG, float32; dt=0.25 s, 1200 steps / 300 s.
- Open lateral boundaries; variants change only `outlet_backflow`, names,
  and output directory.
- 100 history samples and hub-height frames, every 3 s; final checkpoint.

The reduced top height and closer lateral boundaries make this a qualitative
reproducer, not a quantitative replacement for the farm. All variants in this
comparison use the same reduced domain. The auxiliary uniform profile CSV
matches the reduced mesh; the uniform-inflow builder directly initializes the
velocity field without using precursor fluctuations.

Results and source provenance are in
`outputs/v80_small_backflow_debug_20260916/`. Numerical source is archived and
verified before each run. The earlier full-farm outputs are retained, but new
debugging runs use this small setup.

## Completed 300 s comparison

Both runs completed, with 100 finite saved frames and 100 diagnostic samples.
The archived numerical source remained unchanged across both runs. The inlet
was exactly prescribed in both cases.

| Metric | Ordinary outlets | Backflow pressure |
|---|---:|---:|
| Minimum sampled domain u [m/s] | 1.2276443 | 1.2284904 |
| Minimum sampled downstream u [m/s] | 3.3302972 | 3.3007698 |
| Downstream backflow area fraction | 0 | 0 |
| Final upstream second-difference RMS [m/s] | 0.03978103 | 0.03979788 |
| Final upstream adjacent face jump [m/s] | 0.18297291 | 0.18352032 |
| Most negative low-y outward velocity [m/s] | -0.18815054 | -0.51091695 |
| Most negative high-y outward velocity [m/s] | -0.23042735 | -0.35906601 |
| Maximum CFL | 0.33852828 | 0.33851212 |
| Maximum divergence [1/s] | 1.4528632e-07 | 2.3283064e-07 |

The upstream grid-scale pattern is effectively unchanged by outlet treatment.
The small baseline has upstream second-difference RMS 0.03518 m/s at t=30 s,
when the hub-height outlet cell minimum is still 9.99916 m/s. Both runs retain
positive streamwise velocity throughout the sampled domain. Thus the upstream
artifact is reproduced without streamwise reversal, and before substantial
wake deficit reaches the outlet. Together with the earlier single-V80 MUSCL
comparison, this points toward central transport/actuator forcing as the next
place to investigate; it does not uniquely separate pressure induction from
numerical dispersion.

The pressure treatment introduces strong near-wall lateral oscillations. Its
maximum sampled inward speed is 0.51092 m/s versus 0.23043 m/s in the control.
The near-wall boundary plot at z=2 m shows alternating inward/outward spikes.
Total inward lateral volume flux is slightly lower with treatment, so the
adverse change is in local extrema, not an increase in every bulk measure.
The sharp pressure switch and its fixed inlet reference in a developing
rough-wall layer remain a likely mechanism to investigate on this small case.
No new solver correction was applied during this controlled comparison.

Sampling every 3 s cannot exclude shorter events. The experiment does not
validate a log-law profile or steady wake recovery, and it does not demonstrate
elimination of downstream backflow because neither control nor treated run
exhibited it at the sample times.

## Artifacts

- [Matched boundary histories](../../outputs/v80_small_backflow_debug_20260916/energy/run/backflow_analysis/comparison_with_baseline.png)
- [Raw upstream comparison](../../outputs/v80_small_backflow_debug_20260916/energy/run/backflow_analysis/upstream_comparison.png)
- [Near-wall lateral comparison](../../outputs/v80_small_backflow_debug_20260916/energy/run/backflow_analysis/lateral_near_wall_comparison.png)
- [Treated wake movie](../../outputs/v80_small_backflow_debug_20260916/energy/run/v80_single_hub_u.mp4)
- [Baseline wake movie](../../outputs/v80_small_backflow_debug_20260916/ordinary/run/v80_single_hub_u.mp4)
- [Treated numerical metrics](../../outputs/v80_small_backflow_debug_20260916/energy/run/backflow_analysis/metrics.json)

## Subsequent correction

See [the corrected small-case result](single_v80_backflow_fix.md). It removes
the upstream grid pattern while leaving the periodic precursor central.

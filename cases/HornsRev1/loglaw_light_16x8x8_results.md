# Light log-law check at 16 x 8 x 8 m

Tested 2026-09-15 on one A100-SXM4-40GB in allocation 30244593.

## Full-grid feasibility

The 512 x 1024 x 128 case (8192 x 8192 x 1024 m) completed 100 steps,
reaching 136.826447 simulated seconds. Last-50-step advancement speed was
5.28315 steps/s (0.18928 s/step). Observed allocation during advancement
was 25,427 MiB (24.83 GiB), sampled every 10 s. Later samples include the
separate light run and must not be attributed to this benchmark. The full
command took 375.23 s including startup and initial/final checkpoint I/O.
Output: `outputs/hornsrev1_512x1024x128_16x8x8_100steps_30244593/warmup`.

## Light physical check

To keep this preliminary check inexpensive, the light run retained cell
spacing and height but used 256 x 256 x 128 cells: 4096 x 2048 x 1024 m.
It used the ordinary `jaxwind run` path with FFT pressure projection so
the shared runtime collected profiles, stresses, and kinetic-energy history.
Use `fv_loglaw_light_256x256x128_16x8x8_1h.toml` with `run`, not `workflow`.

The run completed 3600 simulated seconds in 3797 adaptive steps. Command
wall time was 126.17 s; steady advancement was 52.8145 steps/s. It averaged
16 profiles sampled from 1800 to 3600 s at 120 s intervals. The 6 s cap is
used to define schedule durations; actual steps use the CFL ceiling 0.9.
The smoke run was compressing its checkpoint during early light-run startup.

| Diagnostic | Result |
|---|---|
| Target friction velocity | 0.250672 m/s |
| Averaged friction velocity | 0.243872 m/s |
| Mean wind interpolated to 70 m | 7.57242 m/s |
| Velocity RMSE against target cell-average log law, 20–100 m | 0.41414 m/s |
| Velocity RMSE using measured friction velocity, 20–100 m | 0.20174 m/s |
| Mean shear / discrete log-law shear, 20–100 m | 1.02270 |
| Normalized total-stress RMSE vs equilibrium, 20–100 m | 0.05537 |
| Resolved TKE growth in final 10 min | 13.78% |
| Final maximum divergence | 6.706e-8 s^-1 |

The lower-layer mean shear is promising, but this is not an equilibrated
log-law boundary layer: the mean speed remains low, TKE is growing, and
upper-layer stress differs substantially from the equilibrium linear profile.
The average shear ratio hides variation with height; inspect the figure.
The initial mean was already logarithmic, so mean-profile agreement alone
cannot establish developed turbulence. One hour is only 0.88 H/u* turnovers.
The smaller horizontal domain also limits what this says about the full farm
domain. A longer spin-up and comparison of consecutive averaging windows
are needed before accepting the inflow.

Outputs: `outputs/hornsrev1_loglaw_light_16x8x8_1h_30244593/`, including
`profiles.csv`, `history.csv`, `checkpoint.npz`, `loglaw_assessment.png`, and
`loglaw_metrics.json`.

Recreate the plot and metrics on a compute node:

```bash
python tools/analyze_hornsrev_loglaw.py \
  outputs/hornsrev1_loglaw_light_16x8x8_1h_30244593
```

The reference velocity integrates the log law over each vertical cell. Its
shear uses the same adjacent-cell finite difference as the simulation profile.

# Single-V80 periodic smoke run

Completed on 2026-09-15 using one A100-SXM4-40GB GPU (allocation 30244593).

- Configuration: `fv_v80_periodic_smoke_512x512x256_1h.toml`.
- Mesh: 512 x 512 x 256; domain: 8192 x 8192 x 1024 m; cells: 16 x 16 x 4 m.
- V80 AD-BEM center: (4096, 4096, 70) m; diameter 80 m; 16.7 rpm; zero pitch.
- Periodic horizontal boundaries; prescribed initial log profile plus noise.
- Completed 3600 simulated seconds in 5046 adaptive RK3 steps.
- Total process wall time: 1780.41 s (29.67 min), including startup and saving.
- Timed advance blocks: 1166.25 s, including compilation; steady rate:
  4.485 steps/s (0.223 s/step), excluding host diagnostics and checkpoint I/O.
- Exactly 100 finite flow frames at t = 36, 72, ..., 3600 s.
- Maximum sampled velocity divergence: 8.57e-8 1/s.
- Eleven profile samples over t = 3240 to 3600 s.
- Five periodic/open-boundary turbine regression tests passed.

Outputs: `outputs/hornsrev1_v80_periodic_smoke_1h_30244593/`, relative to the
repository root. `flow_frames.npz` preserves full-domain hub-height and
centerline planes. `v80_smoke.mp4` shows a near-wake crop at 10 fps (10 s).

The reported `final_cfl = 7.565` uses the configured 6 s step cap, not the
actual adaptive timestep. The adaptive integrator targets pre-step CFL 0.9;
the average actual timestep over this run was 0.7134 s.

This is a startup smoke test, not a statistically developed wind-farm result.
The single turbine is active from t = 0, and its wake can recirculate through
the periodic domain. Supplied custom V80 blade/polar data are used unchanged;
nacelle and tower drag are disabled because body dimensions were not supplied.

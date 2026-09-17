# Refined single-V80 water-spray rerun

The user requested **128 × 128 × 256 cells** and another **60 s** run.
The domain remains **1024 × 1024 × 512 m**, so cells are **8 × 8 × 2 m**
and the mesh contains 4,194,304 cells. [Configuration](v80_uniform10_20kgs_200um_refined.toml).

All physical settings match the [coarse run](v80_uniform10_20kgs_200um.md):
uniform initial/inlet wind 10 m/s, 300 K air, RH 80%, 20 kg/s of 200 µm
water droplets at 290 K; turbine hub (256,512,70) m; spray source
(272,512,70) m with injection velocity (12,0,0) m/s. Both runs begin from
uniform flow with the turbine and spray switched on together. The refined
run is a fresh startup, not interpolation of the coarse final state.

Numerical/closure controls are held fixed: 300 fluid steps at 0.2 s, four
tracking substeps, one parcel per injection step, capacity 512, fixed seed,
MUSCL-MC, carrier AMD with the same uncalibrated 1 m inferred dispersion
length, and 16 m actuator smoothing. Resolved velocity gradients and AMD
viscosity respond to the changed grid. No condensation onto cold droplets
is enabled. The unresolved point source deposits into smaller cells, which
can materially change local maxima and evaporation; two meshes alone do
not establish mesh convergence or measured physical accuracy.

Submit/resume on gpudev:

```bash
sbatch --output=outputs/fluent_dpm_verification/v80-refined-%j.log tools/run_v80_dpm_uniform_refined.sbatch
```

Outputs and exact-state checkpoints are under ignored
`outputs/v80_uniform10_20kgs_200um_290K_refined/`. The driver stops voluntarily
after 660 wall seconds when needed, allowing checkpoint overhead within the
15-minute gpudev limit. Resubmit only after the previous writer has finished.
The code/configuration fingerprint prevents incompatible restarts.

Initial gpudev job **30274558** checkpointed successfully at step 180,
**36 s**. Dependent job **30274589** resumed that exact state and completed step 300,
**60 s**, with exit code 0. Plot job **30274611** completed successfully;
**30274644** completed the comparison with exact disk/cell intersection areas.


## Completed 60-second result

All 300 fluid steps completed. No timestep reduction or change to physical
settings was needed. Both allocation manifests match the evolution source,
runner and case used for the run. The coarse/refined runtime configurations
match in inflow, thermodynamics, turbine, spray, timestep, duration and evolution
source hash. Only mesh/run identity differs.

| Quantity | Coarse: 64 × 64 × 128 | Refined: 128 × 128 × 256 |
| --- | ---: | ---: |
| Cell dimensions (m) | 16 × 16 × 4 | 8 × 8 × 2 |
| Injected water (kg) | 1,200 | 1,200 |
| Evaporated water (kg) | 242.674491 | 195.526569 |
| Evaporated fraction | 20.2229% | 16.2939% |
| Airborne liquid (kg) | 957.325509 | 1,004.473431 |
| Trapped / escaped liquid (kg) | 0 / 0 | 0 / 0 |
| Minimum temperature (K) | 297.683422 | 297.389305 |
| Maximum local cooling (K below 300 K) | 2.316578 | 2.610695 |
| Maximum bulk RH | 96.5253% | 99.0099% |
| Final maximum CFL | 0.206897 | 0.501898 |

Refined total parcel water residual: 3.87e-12 kg. Carrier vapor residual after
net transport: -1.03e-7 kg. Maximum local phase-source energy residual:
2.33e-10 J. The refined run ends with 300 active parcels.

Evaporation is **19.4% lower** than the coarse prediction, while local peak
cooling is **12.7% larger**. This is material grid sensitivity, not proof of
convergence or improved agreement with measurements. The point source enters a
smaller gas inventory on refinement, and resolved flow, AMD viscosity and DRW
histories also change; this pair of runs does not isolate those mechanisms.
Both remain transient startups with the existing Boussinesq carrier and no
condensation onto initially cold droplets.

Artifacts in `outputs/v80_uniform10_20kgs_200um_290K_refined/`: `fields.npz`,
`fields.png`, `history.png`, `grid_comparison.json`, `grid_comparison.png`,
`configuration.json`, `status.json`, complete-state `checkpoint.npz`, and
per-job source manifests. All generated outputs remain ignored by Git.


## Comparison on common physical sampling disks

Both fields are linearly interpolated in x to the same planes. Disk means
use circle/cell intersection areas, including partial cells at the edge, so
each mesh samples the same physical area (5,026.548246 m²). This differs from
the runtime diagnostics, whose native center masks sampled 4,736 m² on the
coarse mesh and 5,056 m² on the refined mesh. Field values are treated as
piecewise constant within each y/z cell for this area integration.

| x/D | x (m) | Coarse cooling (K) | Refined cooling (K) | Coarse u (m/s) | Refined u (m/s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 336 | 0.119430 | 0.067320 | 6.539601 | 6.268975 |
| 2 | 416 | 0.205444 | 0.157448 | 6.458358 | 6.085183 |
| 4 | 576 | 0.288865 | 0.224257 | 7.129779 | 6.667330 |
| 6 | 736 | 0.088650 | 0.032548 | 8.896481 | 9.158805 |
| 8 | 896 | 0.003740 | 0.002657 | 9.952164 | 9.979963 |

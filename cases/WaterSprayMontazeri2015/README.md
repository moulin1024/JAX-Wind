# Water-spray cooling benchmark: Montazeri (2015), case 3

Current per-sensor acceptance is **20%**. The four recent inertial transport
runs pass 9/9 temperature sensors under this revised criterion; full validation
remains open. See the [re-assessment](../SprayClosureValidation/acceptance-20-percent.md).
Earlier 10% results below remain historical.

This reproducible benchmark challenges the current subgrid, entrained-mist model against measured evaporative cooling. **The initial entrained-mist implementation fails this comparison and exceeds its dilute-loading assumptions.** See [results](comparison/report.md) and [comparison plot](comparison/comparison.svg).

The subsequent [inertial-droplet investigation](investigation/README.md) preserves this baseline and adds finite slip, droplet temperature and conservative carrier feedback. Its validation is ongoing.

## Reference and scope

Montazeri, Blocken and Hensen, *Evaporative cooling by water spray systems: CFD simulation, experimental validation and sensitivity analysis*, Building and Environment 83 (2015), 129–141, [DOI](https://doi.org/10.1016/j.buildenv.2014.03.022), [author manuscript](https://pure.tue.nl/ws/portalfiles/portal/32337686/15_bae_si_cpc_montazeri.pdf). The experiments originate from Sureshkumar, Kale and Dhar (2008), [DOI](https://doi.org/10.1016/j.applthermaleng.2007.09.010).

This is an established evaporation/cooling reference with measurable outlet temperatures. It is not an air-assisted snowmaking experiment and cannot validate compressed-air entrainment, atomization, or freezing. Its measured downstream response is useful when the nozzle cannot be resolved, but representing its injected droplet momentum and dispersion remains necessary.

Case 3 uses a 1.9 × 0.585 × 0.585 m tunnel, 3 m/s inlet air, 39.2 °C dry bulb and 18.7 °C wet bulb. Water flow is 12.5 L/min through a 4 mm nozzle at 3 bar gauge, with water entering at 35.2 °C and an 18° cone half-angle. The reported mass-based Rosin–Rammler distribution has scale 369 μm, spread 3.67 and diameter limits 74–518 μm. The builder preserves its initial area per liquid mass using D32 = 292.98 μm; replacing a distribution by one diameter does not preserve its subsequent evaporation history.

## Current model and declared approximations

The benchmark uses the shared moisture/cloud microphysics, moisture transport, open atmospheric solver, canonical runtime, and checkpoint machinery. There is no separate evaporation implementation.

- The unresolved liquid source is a normalized Gaussian centered 0.08 m downstream, with standard deviations (0.025, 0.035, 0.035) m. These are modeling assumptions held fixed in physical units on both meshes, without fitting the temperature observations.
- The current model immediately entrains liquid at gas velocity, shares the gas temperature, and uses diffusion-limited evaporation. It has no droplet slip, spray momentum, cone velocity distribution, droplet thermal lag, settling, wall deposition, or compressed-air source. Reported water temperature, pressure and cone angle are retained as reference metadata but are not applied as missing physics.
- Humidity is reconstructed from dry/wet bulb temperatures using the ventilated-psychrometer relation documented in the builder; atmospheric pressure is assumed 101325 Pa and dry-air density 1.125 kg/m³. See [NIST TN 1994, Appendix A](https://itl.nist.gov/div898/winds/pdf_files/TN1994.pdf) for the psychrometric relation. The builder overrides the configuration's placeholder relative humidity and diameter from these reference inputs.
- Walls are impermeable, free-slip and adiabatic, unlike the reference no-slip treatment. There is no wall-film model. Uniform inlet flow supplies no imposed turbulent fluctuations; AMD models subgrid stresses. These are additional differences from the literature simulation.
- Gas flow uses fast RK3, MUSCL-MC and a geometric multigrid pressure solve; moisture uses conservative upwind transport. All calculations use float64.

The 4 mm nozzle is unresolved on both meshes. Resolving the source kernel more finely does not resolve atomization or establish a valid subgrid handoff model.

## Reproduce

Run on a compute node with JAX and the project's dependencies installed:

```bash
python -m jaxwind run cases/WaterSprayMontazeri2015/coarse.toml
python -m jaxwind run cases/WaterSprayMontazeri2015/fine.toml
python tools/compare_water_spray_benchmark.py \
  outputs/water_spray_montazeri2015/coarse \
  outputs/water_spray_montazeri2015/fine
```

Runs write checkpoints, resolved case metadata and diagnostic histories below `outputs/water_spray_montazeri2015/`. Resume interrupted runs with `python -m jaxwind resume outputs/water_spray_montazeri2015/coarse` (or `fine`). The comparison script requires completed runs and writes JSON, SVG and PNG; the JSON and SVG are retained here.

Both cases simulate 4 s (about 6.3 nominal tunnel transit times), with outlet comparisons over 3–4 s. Coarse: 32 × 12 × 12, dt = 0.001 s; fine: 64 × 24 × 24, dt = 0.0005 s. This is combined mesh/time-step refinement, not an isolated spatial convergence study.

## Experimental target

`reference.json` records nine manually digitized experimental dry-bulb values from Figure 7c, including pixel coordinates, axis calibration, and provenance. The estimated ±0.1 K is digitization uncertainty, not experimental uncertainty. Three touching markers were separated by visual inspection; obtaining the original table would improve this reference.

Nine outlet sensors lie on the 3 × 3 grid y,z = 0.0975, 0.2925, 0.4875 m. The published scatter plot does not label marker-to-sensor identities. We therefore compare the mean, range and sorted temperature distribution; sorted RMSE is not a spatially paired error. Model temperatures are bilinearly interpolated across the last cell-center plane, an approximation to the outlet face.

The ±1 K mean-bias criterion is a declared engineering screening threshold, not a measurement confidence interval. The approximately 8 kW cooling reported in the paper is a literature CFD result, not an independent experimental acceptance target. Neither cooling power nor mean temperature alone establishes agreement.

# Comparison with the literature

**Outcome: failed model-adequacy test. The current spray model is not validated by this case.**

Both runs completed 4 s on an NVIDIA A100 GPU. Six diagnostic samples over 3–4 s give:

| Quantity | Experiment, digitized | Coarse 32×12×12 | Fine 64×24×24 |
| --- | ---: | ---: | ---: |
| Nine-sensor mean dry-bulb temperature, °C | 31.59 | 30.02 | 30.42 |
| Sensor temperature range, °C | 30.68–32.19 | 20.36–39.20 | 21.48–39.20 |
| Mean bias, K | — | −1.56 | −1.16 |
| Sorted-distribution RMSE, K | — | 7.19 | 7.04 |
| Outlet sensible cooling, kW | Not an experimental target | 10.82 | 9.27 |
| Maximum sampled liquid/dry-air mass ratio, kg/kg | — | 4.55 | 6.89 |
| Maximum sampled CFL | — | 0.138 | 0.179 |
| Maximum sampled divergence, 1/s | — | 1.15e−9 | 1.64e−9 |

![Sorted outlet temperatures](comparison.svg)

The experimental points are from Montazeri et al. (2015), Figure 7c, using Sureshkumar et al.'s measurements; see [reference and assumptions](../README.md). The ±0.1 K plot bars describe digitization only. Sensor identities are unavailable, so the error compares sorted distributions, not corresponding spatial points.

Both meshes fail the declared 1 K mean-bias screening threshold. More decisively, the model predicts roughly an 18 K sensor spread versus 1.50 K measured. A relatively close mean therefore conceals a poor cooling distribution. The near-ambient temperatures at several sensors and very cold temperatures at others indicate inadequate modeled spray dispersion.

The maximum local spray mass ratios of 4.55 and 6.89 kg/kg violate the dilute-loading basis of the Boussinesq mist model by a large margin. This loading arises when the measured water flow is injected into the assumed narrow kernel and transported at gas velocity without nozzle momentum. These results cannot be interpreted as quantitative physical predictions. Their purpose is to expose the missing physics; no source width, flow rate or droplet diameter was adjusted to match the data.

Halving mesh spacing and time step changes the mean by 0.397 K and cooling power by about 14%. The distribution error persists. This two-level test does not establish mesh independence or separate spatial from temporal error. Source widths remain fixed in meters. Late-window sensor means vary by less than 3e−11 K in these deterministic runs, indicating a stationary numerical state, not physical accuracy. Sampled water phase mass fractions remained nonnegative.

The literature CFD reports approximately 8 kW cooling. Our fine-grid value is about 16% higher, but that CFD value is not an independent experimental target. Different wall treatment, turbulence, injection momentum, droplet distributions and thermal assumptions preclude attributing the discrepancy to evaporation alone.

## What this establishes

The benchmark runs the implemented moisture-coupled source through the normal simulation/runtime path and provides a reproducible failing reference for further development. It does not validate a snowmaking-style air-assisted spray.

Before claiming that capability, introduce a conservative unresolved injection/handoff model with water and compressed-air mass, momentum and enthalpy; finite droplet slip and temperature; size-dependent evaporation and dispersion; and appropriate loading limits. The reported distribution should be represented by size classes or parcels rather than only its initial area-equivalent diameter. Validate air-jet entrainment and droplet transport separately, then revisit this downstream cooling case and an air-assisted reference. Model changes must also address the presently different wall and inlet-turbulence conditions.

## Artifacts and execution

- [Machine-readable results](comparison.json), [plot](comparison.svg), and [input/reference documentation](../README.md).
- Full local run histories and checkpoints: `outputs/water_spray_montazeri2015/{coarse,fine}`.
- Coarse used 4000 steps; fine used 8000. Fine elapsed time was 74.4 s. Coarse was resumed after a 200-step pilot: the reported 54.5 s covers only the resumed invocation, with an additional 24.2 s for the pilot. These include compilation and overlapping GPU usage and are not performance benchmarks.

## Code verification

The benchmark, buoyancy, moisture and runtime-contract test selection produced 28 passes and one failure. The failure, `test_low_mach_continuation_consumes_new_stage_checkpoint`, also occurs on the unmodified HEAD checkout: its record-plane index lies outside the reduced test mesh. The new reference-reconstruction and sidewall-buoyancy tests passed. New benchmark Python files pass Ruff import/undefined-name checks, and `git diff --check` passes. These code checks do not change the failed physical validation above.

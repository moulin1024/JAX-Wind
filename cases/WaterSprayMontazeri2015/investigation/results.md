# First coupled inertial-spray investigation

**The discrepancy is substantially reduced but remains unresolved.** All four inertial cases completed 4 s on the A100. The original entrained-mist results remain unchanged.

## Measured target and model comparison

| Quantity | Digitized experiment | Entrained coarse | Entrained fine | Inertial coarse | Inertial fine |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nine-sensor mean DBT, °C | 31.587 | 30.025 | 30.422 | 33.592 | 33.689 |
| Sorted-distribution RMSE, K | — | 7.191 | 7.040 | 2.081 | 2.269 |
| Minimum sensor DBT, °C | 30.684 | 20.355 | 21.480 | 31.995 | 32.189 |
| Maximum sensor DBT, °C | 32.186 | 39.196 | 39.200 | 34.819 | 36.517 |
| Outlet sensible cooling, kW | No measured acceptance target | 10.817 | 9.270 | 7.217 | 7.098 |

![Baseline versus inertial closure](comparison/comparison.svg)

The sorted error falls by about 68% on the fine mesh. The updated model replaces excessive localized cooling with a more distributed response, but predicts an approximately 2.1 K warm mean bias. Neither inertial case passes the declared 1 K mean or distribution screens. The spatial pairing of the experimental markers remains unknown. Digitization uncertainty is approximately 0.1 K and is not an experimental confidence interval.

## Controlled numerical refinements

| Case | Mesh | dt, s | Parcels/step | Parcels/s | Mean DBT, °C | Sorted RMSE, K | Cooling, kW |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| inertial_coarse | 32×12×12 | 0.001 | 16 | 16000 | 33.5923 | 2.0807 | 7.2169 |
| inertial_more_parcels | 32×12×12 | 0.001 | 32 | 32000 | 33.5639 | 2.0473 | 7.2438 |
| inertial_half_dt | 32×12×12 | 0.0005 | 16 | 32000 | 33.5690 | 2.0533 | 7.2456 |
| inertial_fine | 64×24×24 | 0.0005 | 16 | 32000 | 33.6893 | 2.2691 | 7.0982 |

- **Sampling:** doubling injection sampling on the same mesh and dt changes the mean by −0.0283 K. The particle-size and azimuth sequences use fixed, independent irrational increments; their distribution is not fitted to temperature observations.
- **Time step:** at the same mesh and parcel injection rate, halving dt changes the mean by +0.00506 K and cooling by 0.025%. This comparison includes the associated halving of the exchange substep. Grouping of the same deterministic injection sequence changes with dt.
- **Mesh:** at fixed dt and parcel injection rate, refining the mesh changes the mean by +0.1203 K and cooling by −2.0%. Crucially, the center sensor rises from 34.598 to 36.517 °C, a 1.919 K change. The mean alone therefore does not establish mesh independence. The maximum of the sampled late-window sensor means varies by less than 0.070 K across that window, which is smaller than the remaining physical disagreement.

See [refinement data](refinement/comparison.json) and [refinement plot](refinement/comparison.svg). Additional cases reproduce with the ordinary CLI:

```bash
python -m jaxwind run cases/WaterSprayMontazeri2015/inertial_more_parcels.toml
python -m jaxwind run cases/WaterSprayMontazeri2015/inertial_half_dt.toml
python tools/compare_water_spray_benchmark.py \
  outputs/water_spray_montazeri2015/inertial_coarse \
  outputs/water_spray_montazeri2015/inertial_more_parcels \
  outputs/water_spray_montazeri2015/inertial_half_dt \
  outputs/water_spray_montazeri2015/inertial_fine \
  --output cases/WaterSprayMontazeri2015/investigation/refinement
```

## Conservation and physical validity

The maximum sampled parcel mass-ledger residual is below 8.5×10⁻¹⁴ kg for the coarse/fine runs, on approximately 0.82045 kg injected water. Source exchange tests also verify total carrier-plus-liquid thermal enthalpy to floating-point accuracy. These do not establish full-domain momentum/energy budgets including open boundaries, nor do they establish physical validity.

The maximum sampled liquid volume fractions reach 0.00284 and 0.01031 on coarse/fine meshes. The [final-state audit](state_audit.json) locates the strongest final accumulation near downstream bottom corners, **not at the nozzle**: approximately (1.870, 0.561, 0.024) m coarse and (1.796, 0.573, 0.012) m fine. At the fine final state, 15.6% of liquid mass lies in cells with liquid volume fraction above 0.001, occupying only 0.91% of domain volume. A volume-fraction estimate of a wall film depends on cell size; treating this liquid as freely suspended evaporating droplets is an unresolved modeling weakness. The reference also idealizes wet-wall behavior, which does not make the approximation physically validated.

The current inlet provides no resolved turbulent fluctuations. The reference assumes 10% turbulence intensity and a 0.04095 m length scale. Its printed inlet equations give k = 0.09 m²/s² and a k–epsilon viscosity estimate of 0.00673 m²/s. The final AMD subgrid-viscosity volume means are 0.000243 and 0.000125 m²/s. **RANS total turbulent viscosity and LES subgrid viscosity are not interchangeable**; these numbers document a substantial model/setup mismatch, not a prescription to insert or tune a constant diffusivity.

Reproduce the audit with:

```bash
python tools/audit_water_spray_state.py \
  outputs/water_spray_montazeri2015/inertial_coarse \
  outputs/water_spray_montazeri2015/inertial_fine
```

## Next actions toward the persistent goal

1. Represent the reference inlet turbulence and assess unresolved turbulent dispersion, with independently specified intensity and length scale. Include gas wall treatment in this study; the present free-slip boundary differs from the reference wall-function calculation.
2. Separate deposited wall liquid from suspended droplets or justify an independently specified film/impact closure. Track wall water and energy explicitly; quantify its effect without tuning against outlet temperatures.
3. Recheck mesh sensitivity of the full sensor distribution, especially the hot center, once those model differences are addressed. Continue to check the unresolved nozzle handoff and distribution uncertainty (the reported mean-size uncertainty is approximately 22%).
4. Extend the verified closure to turbine configuration and implement/validate compressed-air injection and entrainment separately. The present benchmark contains no compressed-air source and cannot validate atomization.

Verification after integration: 30 focused moisture, droplet, parcel, buoyancy and benchmark tests passed. New Python files pass Ruff import/undefined-name checks and the patch passes `git diff --check`.

The original goal remains active. No arbitrary nozzle/source parameter was adjusted to match the measured temperatures, and these results do not warrant a validation claim.

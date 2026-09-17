# Single V80: uniform 10 m/s, 300 K, RH 80%, 20 kg/s water at 290 K

User-prescribed inputs: one V80, domain **1024 × 1024 × 512 m**, uniform
10 m/s inflow, air temperature 300 K, relative humidity 80%, water flow 20 kg/s,
physical droplet diameter 200 µm and water temperature **290 K**, injected at
hub height. [Case settings](v80_uniform10_20kgs_200um.toml).

Additional declared choices inherited from the verified DPM baseline:

- 64 × 64 × 128 cells, each **16 × 16 × 4 m**; float64.
- Turbine at (256, 512, 70) m; V80 AD-BEM at fixed 16.7 rpm and zero pitch.
- Post-atomization point source at (272, 512, 70) m; injection velocity
  (12, 0, 0) m/s. One weighted computational parcel is injected each 0.2 s;
  its constituent droplets each have diameter 200 µm. No resolved nozzle jet.
- Initial carrier velocity is uniform 10 m/s; rotor and spray start together.
  The first run covers **60 s**, with 0.2 s fluid steps and four tracking
  substeps. This is a transient startup, not a statistically stationary wake.
- Pressure 101325 Pa. The driver derives the reference dry-air density from
  the configured T/RH and the same saturation law used for initialization.
- Carrier AMD plus DRW, with the explicit, uncalibrated 1 m dispersion length
  from the preceding integration check. Momentum uses MUSCL-MC.
- Prescribed inlet, open outlet, periodic lateral boundaries, rough ground
  with the inherited neutral momentum wall law, and an impermeable symmetry
  lid. Liquid traps at ground and escapes at the top/open x boundaries.
- Boussinesq carrier; conservative species-enthalpy/vapor transport and local
  phase exchange. No initial turbulence fluctuations are prescribed.

Cold-water limitation: condensation onto droplets is disabled in the selected
Fluent-reference baseline. At 290 K in this warm humid air, the initial modeled
response is droplet heating with suppressed evaporation until conditions allow
vaporization. Possible physical initial condensation is not represented.
This matters when interpreting liquid survival and heat exchange.

Run on the requested queue:

```bash
sbatch --output=outputs/fluent_dpm_verification/v80-20kgs-%j.log tools/run_v80_dpm_uniform.sbatch
```

The driver checkpoints the complete carrier, parcels, RNG state and budgets.
A job voluntarily stops after its wall allowance, within gpudev's 15-minute
limit. Repeating the same command resumes the exact configuration. A source
code/configuration fingerprint prevents incompatible restarts. Do not submit
two simultaneous writers to this output directory.

Outputs are ignored under `outputs/v80_uniform10_20kgs_200um_290K/`:
`configuration.json`, `history.jsonl`, `status.json`, `checkpoint.npz`, final
`fields.npz`, and per-job source manifests. Diagnostics include parcel and
carrier water budgets, cooling extrema, RH, CFL, and disk-area means on several
downstream planes. Actual cell-plane x coordinates are recorded, because an
intended exact x/D location may lie between cell centers. The area is an 80 m
diameter disk centered at hub height; these downstream planes are not additional
turbines.

Initial submission **30274456** stopped before simulation on an inherited
diagnostic start index beyond this shorter run. That configuration was corrected;
**30274457** completed all 300 steps (60 s) successfully. Plot job: **30274477**.
No measured-validation or 20% acceptance claim is implied by this case.


## Result at 60 seconds

Gpudev job **30274457** completed the requested configuration without rejected
steps. The actual runtime manifest was checked against every user-prescribed
input. Results below describe this transient startup; no matched dry run or
statistical averaging was performed for this case.

| Quantity | Result |
| --- | ---: |
| Injected water | 1,200.000 kg |
| Evaporated water | 242.674491 kg (20.22%) |
| Airborne liquid | 957.325509 kg |
| Liquid trapped / escaped | 0 / 0 kg |
| Active computational parcels | 300 |
| Minimum gas temperature | 297.683422 K |
| Maximum local cooling relative to inlet | 2.316578 K |
| Maximum relative humidity | 96.5253% |
| Final maximum CFL | 0.206897 |
| Parcel water-budget residual | -3.87e-12 kg |
| Carrier vapor-budget residual after transport | -1.03e-7 kg |
| Maximum phase-source energy residual | 2.41e-10 J |

Downstream means use an 80 m diameter disk centered at (y,z)=(512,70) m.
These are instantaneous disk-area means, not flow-weighted means. Temperature
deficits are relative to the 300 K inlet; velocity deficits include the turbine
wake and cannot establish a spray-induced wake change without a matched control.

| Actual downstream x/D | x (m) | Mean temperature (K) | Cooling (K) | Mean u (m/s) |
| ---: | ---: | ---: | ---: | ---: |
| 0.9 | 328 | 299.895427 | 0.104573 | 6.355105 |
| 1.9 | 408 | 299.795441 | 0.204559 | 6.252197 |
| 3.9 | 568 | 299.679249 | 0.320751 | 6.921338 |
| 5.9 | 728 | 299.900006 | 0.099994 | 8.744434 |
| 7.9 | 888 | 299.994622 | 0.005378 | 9.929436 |

Explicit unresolved stochastic work was 31.4863 kJ, and gravity work was
185.519 kJ. These are accounted external mechanical contributions in the DPM
ledger; the carrier still has the documented Boussinesq/SGS-energy limitations.
The full domain remains below bulk saturation, although droplet-surface
condensation onto initially cold water is still excluded by the baseline.

Artifacts: `outputs/v80_uniform10_20kgs_200um_290K/status.json`, `fields.npz`,
`checkpoint.npz`, `history.jsonl`, and `configuration.json`. The field and history
plots are generated as `fields.png` and `history.png` in the same ignored folder.

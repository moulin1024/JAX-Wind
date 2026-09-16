# Conservative temperature boundary correction

Completed 2026-09-16. **A real energy-conservation defect is corrected in the
new flux-boundary case; experimental validation still fails at the centre.**

The legacy carrier calls `enforce_open_scalar` on physical inlet/outlet cells,
including after parcel heat exchange. This overwrites deposited heat. The new
`scalar_boundary="flux"` option evolves all scalar cells and prescribes ambient
incoming advective flux, interior outgoing flux and zero open-face diffusive
flux. It is consistent with the existing moisture flux boundary. The legacy
solver default is retained to reproduce archived cases.

Six tests establish integral flux conservation for both flow directions and
central/upwind advection, and retention of boundary-cell sources under AB2 and
fast RK3. Together with existing open-flow, scalar, parcel and benchmark checks,
29 tests passed. The new tests are included in default collection.

Two full 4 s GPU runs use identical physical inputs, smooth walls, 64×24×24
cells, dt=0.0005 s, 16 parcels/step and four exchange substeps. Source, cone,
droplet distribution and turbulence are unchanged. The cell-boundary control
reproduces the archived wall-audit sensor histories within 4.9e-11 K.

| Quantity, 3–4 s | Cell-boundary control | Flux boundary |
| --- | ---: | ---: |
| Mean sensor DBT °C | 33.6933 | 33.6185 |
| Centre DBT °C (measured 31.4) | 35.6933 | 35.4981 |
| Centre relative error, Celsius | 13.6729% | 13.0513% |
| Passing DBT sensors | 8/9 | 8/9 |
| Spatially paired RMSE K | 2.3651 | 2.2893 |
| Sampled gas sensible-budget residual W | 106.655 | 4.047 |
| Reconstructed centre WBT °C (measured 20.2) | 25.2082 | 25.1451 |
| Sampled water-budget residual kg/s | 9.889e-6 | 9.892e-6 |

The sampled sensible budget is `gas storage rate + parcel heat sink - outlet
sensible cooling`. Boundary fluxes use trapezoidal quadrature of 0.025 s saved
diagnostics, not the actual RK-stage fluxes. Its remaining 4 W therefore cannot
be treated as an exact conservation error. Final cloud liquid/ice fields are
zero; the diagnostic does not prove absence of cloud phase heating throughout
the interval. The control-to-flux reduction supports the diagnosed cell-overwrite
mechanism without attributing the entire physical discrepancy to it.

The corrected parcel budgets close to 6.57e-14 kg and 1.03e-9 J maximum cumulative
mass/enthalpy residuals. Escaped water is 22.408°C at the simulation outlet,
which remains a different observable from collected water after the apparatus.

Sensor vapor histories now allow wet-bulb reconstruction with the same stated
psychrometric relation as the inlet. Each sample is inverted before averaging.
The concentrated centre humidity remains discrepant; next inspect carrier
transport/mixing and projection timing, rather than altering droplet size or
cone angle toward the outlet targets. The physical measurement-plane mismatch
at the drift eliminator remains unresolved.

Data: [paired comparison](comparison.json), [carrier budgets](carrier_budgets.json),
[parcel budget](parcel_budget.json). The comparison SVG is explicitly sorted;
spatial pairing is retained in JSON. Source manifests and test output are under
`outputs/waterjet_reference/`. The only source change between the two run launches
was a docstring/line-wrapping edit in `open_abl.py`; both manifests are retained.

Reproduce on a compute node:

```bash
PYTHONPATH=src python -m jaxwind run cases/WaterSprayMontazeri2015/inertial_cell_boundary_control.toml
PYTHONPATH=src python -m jaxwind run cases/WaterSprayMontazeri2015/inertial_flux_boundary.toml
PYTHONPATH=src python tools/compare_water_spray_benchmark.py \
  outputs/water_spray_montazeri2015/inertial_cell_boundary_control \
  outputs/water_spray_montazeri2015/inertial_flux_boundary \
  --reference cases/WaterSprayMontazeri2015/reference_table.json \
  --output cases/WaterSprayMontazeri2015/investigation/carrier_boundary
PYTHONPATH=src python tools/audit_water_spray_carrier.py \
  outputs/water_spray_montazeri2015/inertial_cell_boundary_control \
  outputs/water_spray_montazeri2015/inertial_flux_boundary \
  --output cases/WaterSprayMontazeri2015/investigation/carrier_boundary/carrier_budgets.json
```

Run commands require new empty output directories. Preserve the completed
baseline outputs; the ordinary runtime refuses to overwrite them.

## Projection-order defect and next controlled case

A separate manufactured diagnostic disables evaporation and starts with uniform
vapor q=0.005 kg/kg. One interior parcel impulse produces maximum gas divergence
1.462 s^-1 before pressure projection. Transporting uniform humidity with that
velocity changes it by 7.31e-6 kg/kg (0.146%) in 0.001 s despite no vapor source.
Projecting before transport reduces the change to 4.08e-17 kg/kg. See
[projection-order evidence](projection_order_diagnostic.json).

The optional `project_parcel_feedback=true` benchmark path now enforces the
prescribed velocity boundary and projects the parcel impulse before calling the
moist carrier step. This makes moisture transport see a solenoidal carrier. The
impulse projection pressure is not reused as the carrier's lagged pressure,
which would apply its gradient twice. The legacy path remains available as the
controlled reference. This operator split still requires timestep checks;
intermediate parcel substeps are not separately projected.

`tests/fv/test_water_projection.py` checks preservation of uniform humidity,
solenoidality and prescribed inlet velocity for interior and inlet injection,
with unchanged parcel mass/velocity and zero evaporation/heat exchange. The
regression suite passed 32 tests. A final isolated rerun of the two projection
tests gates the full `inertial_projected_feedback.toml` run. Its outcome must be
compared with `inertial_flux_boundary`, not with a changed source or turbulence
case. Source hashes and logs are in `outputs/waterjet_reference/`.

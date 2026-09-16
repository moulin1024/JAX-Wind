# Centred momentum advection control

Changing only momentum advection from MUSCL-MC to the existing centred operator
does not resolve the discrepancy. Both cases retain AMD, projected parcel
feedback, conservative scalar boundaries, smooth walls and identical physical
source/inflow parameters. Sixteen existing momentum regression tests passed
before the run. The default remains MUSCL-MC.

| 3–4 s result | MUSCL-MC | Centred |
| --- | ---: | ---: |
| Centre DBT °C | 35.47883 | 35.33465 |
| Worst paired error | 12.9899% | 12.5307% |
| Passing DBT sensors | 8/9 | 8/9 |
| Reconstructed centre WBT °C | 25.1303 | 24.2949 |
| Sampled sensible residual W | 3.945 | 2.955 |

The centre DBT reduction is only 0.144 K against a remaining 3.935 K error.
Limiter damping alone therefore does not explain the discrepancy on this mesh.
These short runs do not establish statistical or spatial convergence of either
operator. Between 2–3 s and 3–4 s blocks, the centred case changes by up to
0.329 K across sensors. Its maximum CFL is 0.2814 and final cloud fields are zero.
Parcel cumulative mass and enthalpy residuals are 6.46e-14 kg and 2.74e-9 J.

[Nine-sensor comparison](comparison.json), [carrier budget](carrier_budget.json),
[parcel budget](parcel_budget.json), [temporal blocks](statistics.json).

The single-operator comparison is diagnostic. A modest improvement against the
outlet targets is not a basis for selecting a new numerical or physical model.

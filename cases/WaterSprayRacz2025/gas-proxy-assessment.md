# Source-only carrier-velocity proxy assessment

**Result: the nominal proxy is too sparsely supported to qualify the complete
inlet carrier field.** Gpudev job **30272161** passed 12 software tests and
processed all 34 source stations, but 32 stations retained fewer than the
predeclared preliminary 1,000-event screen. Selected populations ranged from
3 to 1,238. This is a data-qualification finding, not a transport validation.

## Method and provenance

The method is [Rácz et al. (2022), Appendix A](https://doi.org/10.1016/j.ijmultiphaseflow.2022.104260).
We inspected the public `filtering_v2.m` from the
[author software page](https://crg.energia.bme.hu/software/) to resolve details
that are ambiguous in the paper's prose. The archive SHA-256 is
`4af650c1bc37d58a25992a329c0a8a04b82bb990d0e9347d69c09d0867a8f772`.
The author software page states CC-BY-NC-4.0. The downloaded code remains in
ignored research outputs; `tools/racz_gas_proxy.py` is a separate implementation
of the published algorithm, with no author code copied into the repository.
We have not run MATLAB to establish cross-language numerical equivalence.

The input is the two-component speed magnitude and diameter of individual
source droplets. The procedure is:

1. Exclude invalid or sub-wavelength diameters, then apply a three-scaled-MAD
   diameter filter. The author uses MATLAB `isoutlier` with its default
   [scaled **median** absolute deviation](https://www.mathworks.com/help/matlab/ref/isoutlier.html),
   rather than the mean absolute deviation suggested by Appendix A's wording.
2. Estimate gas speed by the mean of the current droplet population and compute
   `Stk = rho_l D^2 |v_mean-v_drop| / (18 mu_air l)`.
3. Find the smallest diameter violating the Stokes threshold. Retain samples
   satisfying both the Stokes threshold and the diameter bound, recompute the
   mean, and repeat until no retained sample violates the threshold. Removed
   samples do not re-enter. Equal-size ties also retain the Stokes test.

A large droplet with a small instantaneous speed difference can have low Stk;
that alone does not make it a tracer. The diameter-prefix restriction prevents
such droplets from remaining simply because they are momentarily co-moving.
All iteration counts, means and maximum Stokes numbers are saved.

The author code explicitly uses `dh=0.4 mm`, `l=30 dh=12 mm`. This resolves the
length convention for **this estimator**: the liquid bore is used. It does not
resolve the differing air-annulus dimensions reported for the apparatus.
The [frozen protocol](gas_proxy_protocol.json) records nominal water density
997 kg/m3 and the author script's air viscosity 18.691 microPa s, with their
assumption/provenance status. The gas proxy filter does not remove any droplets
from the full measured liquid inlet distribution.

## Source isolation and uncertainty diagnostics

Only the **996,164 records at z=20 mm** were processed. The source-only loader
rejects held-out stations before loading their records. File hashes were checked
against the downloaded provenance; no downstream measurement bodies were read.
Masks retain the original record indices. Signed axial and transverse moments
are reported separately, preserving radial X and tangential Y conventions.

The estimator returns the magnitude of two observed velocity components. It is
not an independently measured carrier field, a full three-component speed, or
a direct SGS-energy measurement. No probe-volume or detection-bias correction
is implied. Sample count alone does not establish independent samples.

Stokes limits 0.05, 0.1 and 0.2 were specified before evaluation as sensitivity
cases. The nominal limit remains 0.1. Deleting each of ten equal-duration time
blocks and repeating the nominal estimate tests sensitivity to acquisition
segments. These ranges are **not confidence intervals**: temporal correlation,
detection bias and model discrepancy still require assessment. Cumulative
speed means and standard deviations over Rice-rule diameter bins are retained
for the diagnostic requested in the paper.

## Results

| Stokes limit | Selected count range | Stations below 1,000 |
| --- | ---: | ---: |
| 0.05 | 1–331 | 34/34 |
| 0.1, nominal | 3–1,238 | 32/34 |
| 0.2, sensitivity only | 5–2,605 | 13/34 |

The separate centre measurements give:

| Traverse | Nominal selected events | Speed proxy | Selected axial mean | Leave-one-block-out speed range |
| --- | ---: | ---: | ---: | ---: |
| X | 366 | 55.810 m/s | 54.806 m/s | 55.077–57.001 m/s |
| Y | 405 | 57.103 m/s | 56.104 m/s | 56.698–57.593 m/s |

At the X=+8 mm source edge, only **three events** remain. Deleting one time block
changes the estimated speed over **3.547–11.108 m/s**, a range equal to 96% of
the nominal estimate. Increasing the Stokes limit does not establish adequate
support everywhere and cannot be justified just to obtain a larger sample.
The peripheral sensitivity makes it inappropriate to treat this proxy as a
known carrier boundary for a strict per-observable 10% validation claim.

The numerical tests cover accidental co-motion, equal-diameter ties, filtering
and original record mapping, degenerate MAD/zero speed, invalid input, and the
existing source-only importer contracts. All 12 passed on gpudev; this NumPy
analysis used CPUs within the required GPU queue allocation.

Local ignored evidence:

- `outputs/spray_closure_validation/racz-gas-30272161/report.json`: full protocol,
  every station, iteration histories, checksums and source hashes.
- `source_proxy.csv`, `cumulative_speed.csv`, `selected_source_indices.npz`:
  source diagnostics and exact selected-record indices.
- `tests.xml`: software verification.

Reproduce:

```bash
sbatch --output=outputs/spray_closure_validation/racz-gas-%j.out \
  tools/qualify_racz_gas_source.sbatch
```

## Next physical decision

The raw measured liquid records remain useful. A transport calculation using a
model-assisted carrier could be exploratory, but an independent carrier field
or defensible carrier-input uncertainty is still needed for physical validation.
Do not fill sparse edge stations with fitted downstream velocities, reclassify
the held-out planes as inputs, impose an unmeasured third joint component, or
report the proxy as a directly measured gas field. Further source qualification
must address flux sampling corrections, asymmetry and breakup/evaporation
applicability; meanwhile the model's conservation and dispersion work can proceed.

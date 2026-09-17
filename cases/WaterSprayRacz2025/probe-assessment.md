# Nominal probe geometry and the missing flux observation operator

The [apparatus manuscript](https://dspace.vut.cz/bitstreams/e8d4e5e6-bf03-4fd0-a137-86870e8ca761/download)
provides nominal probe dimensions **0.60 × 0.072 × 0.073 mm** in its xyz frame.
It also describes a 0.1 mm receiver slit reducing the long dimension, settings
changed with pressure/axial station, diameter uncertainty of 0.5 micrometres,
and velocity uncertainty below 1% of the selected range. These metadata are
useful, but do not supply a size-dependent effective detection volume for the
archived condition. The coordinate diagram and optical description were
visually checked in PDF pages 15–16 (one-based, including repository cover).
The downloaded primary PDF is retained in ignored outputs with SHA-256
`4b8a92ff0b5e4fa1eb5b3ba64e5d1f60c7d2ae3ed14a225d6c2a830c904fe2c3`.

## Necessary support screen

The [frozen protocol](probe_protocol.json) tests a deliberately simple
hypothesis: recorded TT is the residence duration of a constant-velocity
particle inside a hard volume of axial extent Lz=73 micrometres. Necessarily,

```
abs(u_axial) * TT <= Lz.
```

This follows from axial displacement alone, irrespective of the missing third
velocity component, unknown impact parameter or receiver-slit length. A generous
finite-sphere overlap version replaces Lz by Lz+D. Two additional scenarios
reduce absolute speed by 1.92 m/s (1% of the largest reported span) and, for the
finite-sphere case, increase diameter by 0.5 micrometres. Those allowances are
not calibrated event uncertainties; no TT uncertainty has been inferred.

Ratios above one contradict that particular hard-support interpretation. They
do **not** prove corrupt events, incorrect measured diameters or a faulty PDA.
Nominal beam dimensions need not equal an optical detection boundary. Conversely,
passing this necessary condition does not establish detection efficiency,
probe volume, valid sizing or accurate mass flux. No raw record is discarded,
and no geometry parameter is fit to force compatibility.

`tools/audit_racz_probe.sbatch` runs the synthetic support checks and importer
regressions, then audits all source files with checksum verification. It records
every station and fixed size-bin result. Gpudev **30272931** completed with
**15 tests passed** and verified all **34 source stations / 996,164 events**.
Results and exact tested source hashes are under ignored
`outputs/spray_closure_validation/racz-probe-30272931/`.

| Support scenario | Events exceeding bound | Fraction |
| --- | ---: | ---: |
| Nominal point-particle support | 257,486 | 25.848% |
| Nominal finite-sphere overlap | 58,091 | 5.831% |
| Point support with velocity allowance | 185,350 | 18.606% |
| Finite-sphere overlap with both allowances | 21,466 | 2.155% |

Every scenario has incompatible events at all 34 stations. The nominal
point-particle fraction ranges from 19.6% to 62.4% by station. In the most
generous scenario it ranges from 0.029% to 4.020%; the largest fraction occurs
at the X-traverse centre, whose 99th-percentile support ratio is **1.158**.
The finding is not confined to diameters outside the previously reported sizing
range: **10,153 of 192,147** events in the 20–40 micrometre bin exceed even the
generous bound (**5.284%**). These events remain in every dataset.

Thus the nominal hard-volume interpretation is not established by the available
metadata and recorded TT. This does not reject the optical measurements or rule
out a calibrated Gaussian-intensity/detection-threshold model. Neither the
geometric exceedance percentage nor a passed support test replaces the existing
10% physical transport criterion.

## Observation contract needed for a measured inlet

For a dilute, locally stationary population, let n(D,u) be the physical number
concentration density and K(D,u)=eta(D,u) A_eff(D,u) |u| the detection-rate
kernel, with efficiency eta and projected effective area A_eff. The expected
event rate is n*K. Raw event statistics therefore describe a detected crossing
population rather than n itself. Recovering a physical number distribution
requires the inverse kernel; recovering a directional mass flux additionally
requires droplet mass and the corresponding signed velocity.

An occupation-time formulation can avoid a one-directional trajectory
assumption if effective volume and efficiency are known. Under homogeneous
sampling and correct transit durations, an event's contribution to an estimated
axial liquid mass flux is

```
(1 / T_live) * mass(D) * u_axial * TT / (eta * V_eff).
```

This is an observation-model requirement derived from residence occupancy,
not a correction applied to the present data. Forward and reverse contributions
must remain distinct for an inlet. Absolute flux also needs live acquisition
time, and spatial integration needs defensible transverse weights. Using only
TT, or forcing the resulting integral to the nominal nozzle flow, does not
supply the missing size/position dependence of eta and V_eff.

Published PDA concentration/flux work explicitly treats probe-volume and
validation-count corrections separately from single-particle sizing/velocity
([Aísa et al., 2002](https://doi.org/10.1016/S0301-9322(01)00071-4)). A separate
cloud-PDI study describes a size-dependent effective volume and per-measurement
software fitting ([AMT, 2022, section 2.2](https://amt.copernicus.org/articles/15/965/2022/)).
That offers a possible auto-calibration route, but its instrument parameters
cannot simply be transferred to the Dantec spray apparatus. Burst splitting,
coincidence and rejection efficiency also need assessment before a flux claim.

The next admissible step is to qualify the applicable auto-calibration method
and its required optical/processor inputs, or obtain a dataset with independently
qualified carrier and flux data. This screen does not establish that such a
calibration is impossible; it prevents using nominal dimensions as if it had
already been done. The source/holdout split and physical acceptance gates remain
unchanged.

The follow-up [conditional chord calibration assessment](chord-assessment.md)
now tests this route with source-only temporal partitions. Both unconstrained
and nonnegative Gaussian-threshold radius fits are retained. Enforcing positive
radii does not resolve the support/distribution mismatch, so no correction is
applied to the inlet. This tests a stated axial-cylinder specialization, not
every published multidirectional auto-calibration method.

# Free air-assisted water-spray transport: candidate assessment

This records the candidate search for measured transport validation within the
[active coarse-grid spray and waterjet goal](../../doc/water-spray-validation-goal.md).
The later [Rácz public PDA dataset](../WaterSprayRacz2025/README.md) now supplies
90 numerical measurement stations with a frozen upstream/downstream split.
Its data audit has passed; physical source qualification remains open. The
historical candidates and their data gaps below remain useful alternatives.
No measured-transport validation pass has been claimed.

## Initial candidate — not yet qualified

Xia, Alshehhi, Hardalupas and Khezzar (2018), *Spray characteristics of free
air-on-water impinging jets*, International Journal of Multiphase Flow 100,
86–103, [DOI 10.1016/j.ijmultiphaseflow.2017.12.007](https://doi.org/10.1016/j.ijmultiphaseflow.2017.12.007).
[Author-institution record](https://khazna.ku.ac.ae/en/publications/spray-characteristics-of-free-air-on-water-impinging-jets/).

Why it fits: water plus an assisting air jet, free spray rather than a wall-bound
cooling tunnel, and PDA measurements of droplet size and velocity. The published
example uses water flow 100 mL/min, air flow 13.5 g/min and a 45-degree jet angle;
it reports a peak mean droplet velocity around 14 m/s at z=75 mm and spatially
varying SMD around 50–120 micrometres. These are preliminary source facts, not a
complete digitized validation dataset or prescribed injection distribution.

Before selecting the actual case, verify measured planes, joint size/velocity
and spatial source information, gas-jet conditions, breakup completion, numerical
data availability and measurement uncertainty. Specify an upstream handoff plane
and reserve downstream planes for validation. Do not resolve or fit atomization
in order to manufacture a transport pass.

## Alternatives screened

- [Christou/KIT experimental airblast dataset](https://publikationen.bibliothek.kit.edu/1000143659),
  DOI 10.5445/IR/1000143659: openly listed hot-wire airflow and PDA data with
  explanatory files. Strong alternative for data availability; fluid, steady
  operating subset, spatial stations and confinement need verification.
- [Urbán et al. (2017)](https://doi.org/10.1016/j.ijmultiphaseflow.2017.02.001):
  downstream size and two-component droplet velocities in an atmospheric
  airblast rig, but the primary paper identifies the liquid as diesel. Lower
  priority than a water experiment for the intended feature.

## Implementation implications

Reuse inertial drag, parcel transport and conservative gas coupling. The current
benchmark source is x-directed; the new handoff may require a general orientation
and measured spatial size/velocity distributions. Carrier air momentum must be
represented explicitly or supplied from independent measured gas profiles.
Exclude wall-film, collector and drift-plate closures from this benchmark.
Thermal physics remains separate from transport acceptance. Carry forward the
user-revised per-observable 20% requirement with predeclared handling of near-zero
quantities, plus numerical and conservation checks.

## Source audit, 2026-09-16

The [Xia author manuscript](https://www.researchgate.net/publication/321665160_Spray_Characteristics_of_Free_Air-on-Water_Impinging_Jets)
places mean and fluctuating velocity profiles in Figures 19–20 at z=75 mm.
Figure 14 reports size evolution with downstream distance. Our assessment:
these figures do not yet provide a complete independent source/target split.
Do not prescribe the 75 mm velocity measurements and then count agreement at
that same plane as a transport validation. Additional upstream distribution and
carrier-flow information is needed before this candidate can be selected.

The [Christou et al. 2022 primary paper](https://journals.sagepub.com/doi/10.1177/17568277221092987)
confirms an unconfined water spray, with both airflow and droplet measurements
at z=40 mm and radial positions x=0 and -5 mm. This is a good physical match,
but the reported measurements do not establish independent upstream and
downstream transport stations. The associated thesis/raw dataset remains a
possible source of additional measurements, not an already qualified benchmark.

Decision: retain both candidates for source-data qualification. No simulation
was launched from an invented injection distribution, and no 10% pass is claimed.
The next selection step is to establish a documented post-breakup source and
held-out downstream observations, or replace these candidates if unavailable.

### KIT raw-data inspection

Retrieved the public file manifest, both readmes, the 102 Hz spray archive and
[Christou's thesis](https://publikationen.bibliothek.kit.edu/1000162144/v2).
Checksums and archive-member row counts are in [kit_data_audit.json](kit_data_audit.json).
Downloads remain under `outputs/free_water_spray_transport/work`.

The archive has 11 numbered files of individual droplet records; its readme
identifies arrival time, axial velocity, radial velocity and diameter, but does
not map file numbers to coordinates. Do not infer that mapping from file order.
The thesis places the forced-flow study at z=40 mm (printed p.117). Its steady
shadowgraphy maps include spatial velocity and size statistics (pp.99–102),
which may support a separate steady case, but are not an upstream distribution
for the oscillating case. Section 4.4.2 also presents a droplet-motion model;
reproducing that calculation would be model verification, not by itself an
independent experimental transport validation.

Next: assess the steady spatial maps and their source-data availability, or
move to another multi-station experiment. The raw data improve provenance but
do not yet resolve the source-plane and coordinate-mapping requirements.

## Stronger transport candidate: Rüger / Khaled

[Khaled et al. (2026), *Assessment of Injection Modeling Techniques for a Water
Spray Using an Euler/Lagrange Approach*](https://doi.org/10.3390/fluids11060150)
revisits the Rüger et al. experiment. Its Case 1 prescribes measured conditions
at z=25 mm, after primary breakup, and compares downstream profiles at 50,
100 and 200 mm. Observables include droplet size and velocity distributions.
Evaporation is neglected. The authors report negligible wall-impact influence
on mean droplet properties, though the carrier domain is a cylinder and is not
unconfined. This provides a better source/target split than the candidates above.

Qualification still requires numerical inlet distributions and downstream data.
Gas conditions are inferred from small-droplet measurements; turbulent length
scale is uncertain. Neither numerical data availability nor agreement within
10% at every station has been established. This pressure-swirl case would test
transport, not compressed-air atomization. Do not replace its carrier boundaries
with open boundaries without checking the effect.

Original experiment: [Rüger et al. (2000)](https://doi.org/10.1615/AtomizSpr.v10.i1.30).
Further primary sources to inspect include [Laín and Sommerfeld (2020)](https://www.researchgate.net/publication/343016697_Influence_of_droplet_collision_modelling_in_EulerLagrange_calculations_of_spray_evolution)
and the [ERCOFTAC test-case notes](https://www.ercoftac.org/downloads/dmf-sommerfeld_6_test_cases.pdf).

Scope interpretation: the user's exclusion concerns wall-impact physics. A
carrier-flow enclosure is not automatically disqualifying if the selected
measurements test airborne droplet transport and wall-impact effects are
shown negligible. Free air-assisted spray remains preferred, but making it an
absolute prerequisite would unnecessarily restrict the requested validation.

### Data access and transport-only implementation

The publisher full text is accessible via [ResearchGate](https://www.researchgate.net/publication/407064150_Assessment_of_Injection_Modeling_Techniques_for_a_Water_Spray_Using_an_EulerLagrange_Approach).
Its Data Availability Statement specifies author request, rather than a public
numerical supplement. A specific [data request draft](data-request-draft.md) is
prepared but has not been sent. ERCOFTAC's two-page notes are guidance and
references, not the experimental data package.

`exchange_water_parcels` and `build_inertial_water_step` now accept the static
keyword `thermal_exchange=False`. It disables parcel heat and phase exchange
while preserving drag, gravity, two-way momentum exchange and escape ledgers.
The default remains the shared moisture thermal closure. This option does not
change carrier boundary conditions or select a measured injection distribution;
it is a prerequisite, not a runnable literature validation case.

`side_boundary="escape"` is now available on both parcel APIs. It removes
parcels at y/z exits and accounts for escaped mass/enthalpy without a wall-contact
tag or imposed loss of normal velocity. The x boundaries already permit escape.
The default `"wet-wall"` behavior is retained for existing cases. Carrier-flow
boundaries must still be configured separately to match the experiment.

Boundary crossing is detected at the end of each parcel substep, as for the
existing x escape. The timestep/domain convergence checks therefore remain
necessary; this option is not an exact boundary-crossing event integrator.

The original Rüger author-manuscript PDF link was located, but retrieval returned
HTTP 403. Its accessible abstract confirms that droplet coalescence can affect
size evolution, reinforcing the need to assess which measured transport
observables can be compared with a collision-free model. No inlet distributions
have been inferred from downstream profiles or invented to bypass data access.

## Uploaded originals

Both original PDFs are now available at repository root. See the
[uploaded-paper assessment](uploaded-pdf-assessment.md) and the transcribed
[Table 4](khaled2026_table4.csv). The PDF download failure is resolved; the
complete numerical source-distribution gap remains.

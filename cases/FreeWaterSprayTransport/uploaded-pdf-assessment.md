# Assessment of the uploaded papers

The upload resolves access to the two publications. Checksums are recorded in
`uploaded_sources.json`. Rüger is a scanned 35-page PDF; its figures and boundary
condition pages were inspected visually. Khaled is a text-readable 29-page PDF.
Neither paper should be confused with the supporting numerical dataset.

## What can now be verified

- Rüger PDF p.14 (printed p.60), Table 3: case 1 nozzle diameter 0.5 mm,
  nominal spread angle 45 degrees, liquid volume flow 2.251e-6 m3/s,
  air temperature 294.1 K. The table labels volume flow as mass flow; retain
  its printed units rather than interpreting the number as kg/s.
- Same page: droplets enter 25 mm downstream, using measured local size
  distributions and diameter-conditioned velocity means/RMS values. Diameter
  classes are approximately 3–5 micrometres wide. These are required inputs,
  not distributions recoverable from a single mean diameter.
- Rüger PDF p.16 (printed p.62), Figure 3: cylinder diameter 400 mm and the
  downstream numerical region. Published profile figures include source and
  downstream measurements; detailed Figure 22 on PDF p.30 contains correlations
  at 50, 100 and 200 mm, not the 25 mm injection plane.
- Khaled PDF p.8: Case 1 uses 31 radial locations, local number CDFs,
  size-conditioned velocity statistics and measured radial mass-flow allocation.
  Wall removal below 0.005% within z<=200 mm supports transport validation
  without a wall-impact closure in that observation region.
- Khaled PDF p.9: Case 2's effective cone angle is 90–100 degrees, distinct from
  the nominal 45-degree nozzle specification; its mass rate is 2.225 g/s.
  Do not silently substitute that surrogate for measured-plane injection.

## Accuracy reassessment

`khaled2026_table4.csv` transcribes the paper's aggregate error table, not
JAX-Wind predictions or individual experimental sensors. For Case 1a, downstream
mean relative diameter errors are 6%, 9%, 6%; axial velocity errors are 6%, 18%,
31%; radial velocity errors are 22%, 54%, 52%. Thus the published calculation
itself does not establish a 10% pass at every selected measurement. Near-zero
velocities need explicit treatment, and aggregate scores cannot certify a
per-sensor pass. Keep the user's criterion; do not weaken it to match the paper.

## Remaining data gap

The inspected figures provide candidate targets for digitization, but no complete
31-position inlet CDF/size–velocity/mass-flux package has been located in the
papers. Khaled's data statement still directs readers to the authors. A
reconstruction from marginal mean profiles would require additional assumptions
and must be labeled approximate, with source uncertainty assessed separately.
The original Rüger dissipation expression also uses epsilon=k^(3/2)/(0.01*l)
with l described as nozzle diameter; this must be reconciled with the newer
paper's length-scale discussion before copying a turbulence inlet prescription.

No literature simulation or agreement claim follows from this document.

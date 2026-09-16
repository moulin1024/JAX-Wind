# Source weighting audit, 2026-09-16

The current truncated Rosin–Rammler sampling is mass based, as explicitly
specified by Montazeri (2015), equations 4–5 and section 3.3. Equal physical-mass
parcels and multiplicity proportional to inverse droplet mass implement this
weighting consistently. No number/mass interchange was found.

Independent SciPy quadrature of the prescribed distribution (scale 369 µm,
spread 3.67, bounds 74–518 µm) gives D10=206.774 µm, D30=248.057 µm and
D32=292.980 µm. In particular, changing that same distribution to number-based
sampling would change the physical input, not repair a sampling defect.

The original uploaded PDF, page 5 / printed page 353, Figure 6, contains a
number-frequency histogram. Visual inspection confirms that it is concentrated
in the 74–222 µm bins. Its coarse bar heights are consistent with a D30 near
240 µm when represented by bin midpoints, not with the approximately 330 µm D30
stated in the surrounding text. This approximate visual moment is a diagnostic,
not a qualified replacement inlet dataset. The printed mean, coarse binning,
resolution uncertainty and later fitted distribution should not be silently
identified with one another.

The source angle and carrier turbulence remain independently uncertain:
Montazeri notes that the precise angle and measured turbulence statistics were
not supplied by the experiment. We retain the archived 18-degree case and
prescribed turbulence for the boundary control; no source parameter is changed
to improve held-out outlet agreement.

Primary source: [Montazeri author manuscript, sections 3.2–3.4](https://pure.tue.nl/ws/portalfiles/portal/32337686/15_bae_si_cpc_montazeri.pdf).
Original measurement source: [uploaded PDF](../../../waterjet_referece.pdf),
section 2.5.2 and Figure 6.

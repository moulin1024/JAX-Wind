# Andrén et al. (1994) neutral Ekman LES specification

This directory contains a concise, factual benchmark specification and values
extracted from the paper; it does not redistribute the article.

## Canonical case

- horizontally periodic domain: `4000 × 2000 × 1500 m`;
- grid: `40 × 40 × 40`, hence `Δx × Δy × Δz = 100 × 50 × 37.5 m`;
- geostrophic wind: `(Ug, Vg) = (10, 0) m s-1`;
- vertical Coriolis parameter: `f = 1e-4 s-1`;
- latitude: 45 degrees north, so the horizontal Coriolis component is also
  `fh = 1e-4 s-1`;
- aerodynamic roughness: `z0 = 0.1 m` with surface-layer similarity;
- impermeable, stress-free rigid upper boundary;
- neutral momentum dynamics. The paper also transports a passive scalar with a
  prescribed surface flux of `1e-3 kg m-2 s-1` at density `1 kg m-3`; it has no
  momentum feedback. The JAX-Wind LASD comparison advances it with zero upper
  flux;
- duration: `10/f = 100000 s = 27.7778 h`;
- statistics: final `3/f`, from `t f = 7` through `t f = 10`.

The paper uses one common initial mean profile and height-dependent initial TKE
for every code. These 40 table entries are stored in `initial_profiles.csv`.
Random velocity perturbations are uniformly distributed on `[-0.5, 0.5]` and
plane-normalized to the tabulated TKE before the compatible projection.

## Present model correspondence

The static JAX-Wind case uses horizontal pseudo-spectral derivatives, vertical
second-order differences, AB2, a neutral log wall, and a static Smagorinsky
coefficient `Cs = 0.17`. This most closely follows the deterministic
Mason--Brown SGS coefficient while retaining Moeng-like horizontal numerics.
It is therefore compared with the published multi-code envelope, not claimed
to reproduce one named code exactly. The published prognostic-SGS-energy and
passive-scalar diagnostics are not fabricated for a static Smagorinsky model.

The second active comparison uses independent Lagrangian-averaged
scale-dependent dynamic coefficients for momentum and scalar flux. It is an
external fifth SGS model, not one of the paper codes. Since LASD does not
predict the isotropic SGS stress or scalar variance, those contributions are
diagnosed from a labeled local production--dissipation balance; resolved,
diagnostic SGS, and total curves remain separate.

## Numeric reference values

The paper reports `u*/Ug` values from `0.0402` to `0.0448` across the seven
runs, and a normalized vertically integrated **resolved plus SGS** TKE plateau
near `0.7`. JAX-Wind reports its resolved-only integral separately so the
different quantities remain explicit. Additional reported spectral peak
ranges are recorded in `reference_results.json`.

Source: Andrén, A., Brown, A. R., Graf, J., Mason, P. J., Moeng, C.-H.,
Nieuwstadt, F. T. M., and Schumann, U. (1994), *Large-eddy simulation of a
neutrally stratified boundary layer: A comparison of four computer codes*,
Q.J.R. Meteorol. Soc. 120, 1457–1484,
<https://doi.org/10.1002/qj.49712052003>.

# Nieuwstadt et al. (1993) CBL benchmark specification

This is an independently written simulation summary for the dry convective
boundary-layer comparison reported by F. T. M. Nieuwstadt, P. J. Mason,
C.-H. Moeng, and U. Schumann, *Large-Eddy Simulation of the Convective
Boundary Layer: A Comparison of Four Computer Codes* (1993),
[DOI 10.1007/978-3-642-77674-8_24](https://doi.org/10.1007/978-3-642-77674-8_24).
It records the facts needed to run and check this implementation; it is not a
transcription of the publication.

## Common physical case

- Dry, shear-free, horizontally periodic convective boundary layer.
- Domain: 6400 m by 6400 m by 2400 m.
- Reference inversion height: `zi0 = 1600 m`.
- Constant kinematic surface potential-temperature flux: `Qs = 0.06 K m/s`.
- Reference potential temperature: `theta0 = 300 K`.
- Gravity: `g = 9.81 m/s2`.
- Moeng grid used for the direct paper comparison: `40 x 40 x 48`.

The reference convective scales are

```text
wstar0     = (g Qs zi0 / theta0)^(1/3) = 1.46422 m/s
thetastar0 = Qs / wstar0               = 0.0409774 K
tstar0     = zi0 / wstar0              = 1092.73 s
```

Paper profiles are averaged over `10 < t/tstar < 11`. Heights in most profile
figures are normalized by the diagnosed boundary-layer height for that run.

## Initial condition

The benchmark starts from zero horizontal velocity and weak deterministic
random perturbations in vertical velocity and potential temperature below the
initial mixed-layer top. The implementation places that top at
`0.844 zi0 = 1350.4 m`, uses a well-mixed `300 K` layer below it, and a
`0.003 K/m` stable gradient above it. Perturbations taper linearly to zero at
the mixed-layer top and are generated from the configured random seed.

## Reference bulk targets

Table 3 of the paper reports these Moeng-code values for the final averaging
period:

| Quantity | Moeng value | Validation tolerance |
| --- | ---: | ---: |
| `zi/zi0` | 1.0312 | 0.020 |
| `wstar/wstar0` | 1.010 | 0.015 |
| `-<w'theta'>i/Qs` | 0.106 | 0.030 |

The tolerances account for stochastic realization, grid-level sampling of the
inversion, and differences between the historical and present numerical/SGS
formulations. They are regression bounds, not uncertainty estimates for the
published values.

## Mapping to the current JAX benchmark

`run_new.py` keeps the paper's physical domain, grid, forcing, initial
condition, and averaging interval while using the new semantic JAX-Wind
implementation:

- JAX/CUDA execution in FP32;
- accepted-boundary AB2 integration, `dt = 1.25 s`, 9646 steps;
- conservative, horizontally dealiased momentum and scalar transport;
- the external `spectral_fd` cell-centred pressure API with a compatible
  impermeable top boundary;
- locally averaged scale-dependent dynamic (LASD) SGS closure for momentum
  and potential temperature;
- a stable-stratification scalar diffusivity correction with beta 30 and
  power 2, expressed through the execution-scale buoyancy coefficient;
- LASD coefficients updated every eight steps, keeping
  `cs_count * max(CFLx, CFLy, CFLz) < 1` in the reference GPU run.

`run.py` selects `run_new.py` by default and does not put `legacy/jax` on the
module path. `run_new.py` writes the complete diagnostic record: scalar and momentum
profiles, time histories, conditional averages, kinetic-energy budget terms,
spectra, summary statistics, and standalone plots. `overlay_figures.py` maps
those diagnostics onto paper Figs. 1--17. The official scan is registered by
`extract_paper_figures.py` after PDF pages 8--20 have been rendered at 200 dpi.

The primary automated acceptance criteria are the three bulk targets above,
finite/non-negative SGS diagnostics where required, valid spectrum sampling
heights, and production of every required diagnostic file.

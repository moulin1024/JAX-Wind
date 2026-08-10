# Andrén et al. (1994) neutral Ekman benchmark

This benchmark reproduces the neutral, rigid-lid Ekman LES intercomparison
configuration of Andrén et al. (1994) as an independent active case.

Run the canonical `40 × 40 × 40` case from the repository root:

```bash
python benchmark/Andren1994/run.py
```

This advances through `tf=10` (`27.7778 h`) and averages the final `3/f`. The
default one-second step follows the Andrén--Moeng entry in the paper's runtime
table. Float32 and the communication-reducing SPIKE pressure method are used by
default.

Run LASD through the conservative, 3/2-padded path:

```bash
CUDA_VISIBLE_DEVICES=0 python benchmark/Andren1994/run_lasd.py \
  --output benchmark/Andren1994/results/lasd_40x40x40_total_tke
```

This runner transports the paper's passive scalar with the prescribed
`1e-3 kg m-2 s-1` surface flux and zero upper flux. It uses `dt=0.8 s` and a
five-step LASD update interval so the total-CFL warning target (`0.2`) and the
one-halo trajectory target (`CFL × interval < 1`) remain credible. These are
warnings, not solution clips.

Figure 2 uses resolved plus diagnostic SGS kinetic energy. The LASD diagnostic
uses a local-equilibrium dissipation coefficient (`Ce=0.93`) and neutral
log-wall shear correction. It does not feed back into momentum and therefore
does not alter the resolved trajectory.

The focused pre-run check is intentionally small:

```bash
PYTHONPATH=src python -m pytest -q \
  tests/interpreters/test_fused_neutral_sgs.py::FusedNeutralSgsTests::test_lasd_diagnoses_negative_resolved_tke_transfer \
  benchmark/Andren1994/tests/test_case.py::test_lasd_uses_three_halves_padding
python benchmark/Andren1994/run_lasd.py --quick --output /tmp/andren-lasd
```

This checks the diagnostic kernels, common numerics, and serialized nonzero
SGS contribution without running the repository-wide integration suite.

Collect the complete resolved vertical scalar-flux budget for paper Fig. 13 by
continuing the developed `tf=10` state over another `3/f` window:

```bash
python benchmark/Andren1994/run_lasd.py \
  --restart benchmark/Andren1994/results/lasd_40x40x40/checkpoint.npz \
  --output benchmark/Andren1994/results/lasd_40x40x40_fig13_budget \
  --hours 36.1111111111 --fig13-budget
```

This observer writes restartable `fig13_budget_samples.npz` and the normalized
`fig13_budget_profiles.csv`. It forms production, SGS, transport, pressure,
Coriolis, and tendency from the solver's actual discrete tendencies. The
projection Lagrange multiplier is already the modified pressure
`p_r/rho + 2 e_sgs/3`, because isotropic SGS stress is not separately applied
to momentum; adding diagnostic SGS energy to it again would double count that
term. The raw samples retain the implied diagnostic isotropic-SGS split for
auditing.

Run the end-to-end smoke case:

```bash
python benchmark/Andren1994/run.py --quick
```

Continue a canonical checkpoint to the paper's final time:

```bash
python benchmark/Andren1994/run.py \
  --restart benchmark/Andren1994/results/static_smag_40x40x40/checkpoint.npz
```

Outputs are written below `benchmark/Andren1994/results/`:

- `checkpoint.npz`, `history.csv`, `profiles.csv`, and `summary.json` retain the
  common accepted-step/restart contract;
- `normalized_profiles.csv` uses the paper's `zf/u*` and variance/flux scaling;
- `ekman_diagnostics.png` shows dimensional solver diagnostics;
- `andren1994_comparison.png` compares `u*/Ug` with the published multi-code
  envelope and explicitly distinguishes resolved-only JAX-Wind TKE from the
  paper's resolved-plus-SGS quantity.

The LASD run additionally writes all 13 closure-memory fields, the scalar
quantity marker, closure fingerprint, and both AB2 tendency histories in
checkpoint schema v2. `statistics_samples.npz` preserves the averaging history
across restart. `profiles.csv` retains resolved, diagnostic SGS, and total
velocity/scalar variance and flux, while `spectra.csv` contains x spectra at
the level nearest `zf/u*=0.1`. New histories also retain both component surface
stresses required by the paper's Fig. 3 stationarity measures `Cu` and `Cv`.
The LASD model also samples the signed resolved-TKE transfer
`tau_ij*d_j(u_i)` required by Fig. 11. Forward transfer is negative. The
dimensional and `f*u*^2`-normalized profiles are written to `profiles.csv`, and
the paper-overlay command adds `fig11_jaxwind_overlay.png` when that column is
present.

The SGS energy and scalar variance are explicitly diagnostic local-equilibrium
quantities. LASD predicts deviatoric stress and scalar flux; it does not evolve
the stress trace or scalar variance. Plots therefore keep resolved,
`diagnostic SGS`, and total contributions distinguishable. At the first cell,
the equilibrium energy uses the same neutral log-wall shear as the wall
traction, while the scalar gradient is reconstructed from the prescribed wall
flux. This keeps the diagnostic budget finite when the centered resolved shear
momentarily vanishes. For the horizontally homogeneous paper comparison, the
scalar-variance numerator and SGS energy are plane-averaged before applying
the equilibrium closure. This benchmark-only statistic avoids averaging a
singular local ratio. Its wall gradient uses the plane-mean dynamic
diffusivity, so the prescribed homogeneous flux is satisfied in the same
plane-mean sense. Both choices are explicit benchmark observation semantics;
the default local diagnostic and the solver field semantics are unchanged.
Resolved component variances likewise subtract each instantaneous horizontal
plane mean before time averaging. This is essential for the passive scalar:
the imposed net wall flux makes its domain mean drift, and that deterministic
drift is not turbulent variance.

Both paths use the complete 45°N rotation vector (`fh=f`), the tabulated 40-level
initial mean/TKE profiles, an impermeable stress-free top, and no Rayleigh
damping. The dry static path omits the passive scalar because it has no momentum
feedback; the LASD path advances it with `NoBuoyancy`. The static Smagorinsky
closure (`Cs=0.17`) corresponds most
closely to Mason--Brown without backscatter; it is assessed against the
intercomparison envelope rather than labeled as one of the original codes.

The compact public specification and extracted factual data are under
[`reference/`](reference/Andren1994.md). The original article is linked there
and is not stored in this repository.

To overlay a completed run directly on the original paper figures without
redistributing the article itself, download the linked PDF and run:

```bash
python benchmark/Andren1994/overlay_paper_figures.py \
  --paper-pdf /path/to/qj-1457-1994.pdf \
  --results benchmark/Andren1994/results/lasd_40x40x40
```

This produces one sheet, `andren1994_all_figures_jaxwind_overlay.png`, containing
all 19 numbered paper figures in order. For LASD, curves are registered on
Figs. 2, 3(a,b), 4(a,b), 5, 6, 7, 8, 13(a-d), 14(a,b), and 15(a-d) when the
corresponding
diagnostics exist. Fig. 13 overlays the same six colored JAX-Wind budget terms
on all four original code panels; other plots use red for a resolved/total
JAX-Wind curve and blue for a diagnostic SGS contribution. Paper-only tiles
remain explicit rather than receiving a misleading substitute quantity.

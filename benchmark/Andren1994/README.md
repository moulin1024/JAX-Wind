# Andrén et al. (1994) neutral Ekman benchmark

This benchmark reproduces the neutral, rigid-lid Ekman LES intercomparison
configuration of Andrén et al. (1994) as an independent active case.

Run the canonical `40 × 40 × 40` case from the repository root:

```bash
python benchmark/Andren1994/run.py \
  --sgs amd --end-ft 10 --sample-start-ft 7 \
  --output-dir benchmark_results/andren1994_amd_40cubed_complete
```

The current runner uses the MAC finite-volume momentum path, a matrix-free
symmetric-GMG/PCG pressure solver, and filter-free AMD by default. The AMD run
also advances the paper's passive scalar with a conservative AMD flux and a
prescribed lower surface flux of `1e-3`; use `--no-passive-scalar` only for a
momentum-only diagnostic run. `--scalar-amd-coefficient` defaults to the
momentum `--amd-coefficient`. The nonlinear correction is MP5; set its strength
with `--advection-dissipation-strength`. `--mp5-strength` remains a
backward-compatible alias.
The stiff vertical part of the SGS principal diffusion defaults to the
third-order ARS(2,3,3) IMEX path and is solved directly along each vertical
column. `CFLnu` therefore diagnoses the complete SGS operator; timestep
selection retains only the explicit horizontal diffusion limit. Use
`--sgs-time-integration explicit` for the fully explicit projected SSPRK3 path.
Use `--sgs lasd` for the multilevel LASD momentum closure; the passive scalar
is then off by default. LASD consumes the exact first two grids of the pressure
GMG hierarchy. Its packed velocity, product, strain, and model tensors are
conservatively restricted to those grids. Germano contractions and Lagrangian
memory live on level one, level-two contractions are interpolated to level one,
and only the resulting coefficient is interpolated to the LES grid. This
reduces the closure's storage and bandwidth while keeping pressure and LASD
coarsening decisions identical. The grid-filter ratio is consequently fixed at
one; `--lasd-sgs-delta-scale` is available only for an explicit model-length
calibration. LASD memory is updated after every accepted flow step. Accepted-
step LASD work remains split into statistics and history/finalization JIT
executables. The default advances to `ft=0.1`; use
`--end-ft 10 --sample-start-ft 7` for the canonical final averaging window.

Momentum uses full pressure projection at every SSPRK3 or ARK3 stage. The
accepted pressure field is checkpointed and reused as the initial guess for
the next pressure solve.

The AMD runner observes the complete resolved momentum- and scalar-flux
budgets online over the configured averaging window and writes
`fig12_budget_profiles.csv` and `fig13_budget_profiles.csv`. It forms
production, SGS, transport, pressure, Coriolis, and tendency from the solver's
actual discrete tendencies. The
projection Lagrange multiplier is already the modified pressure
`p_r/rho + 2 e_sgs/3`, because isotropic SGS stress is not separately applied
to momentum; adding diagnostic SGS energy to it again would double count that
term. The raw samples retain the implied diagnostic isotropic-SGS split for
auditing.

Run a short AMD smoke case on the canonical grid:

```bash
python benchmark/Andren1994/run.py \
  --sgs amd --end-ft 0.001 --sample-start-ft 0 \
  --output-dir benchmark_results/andren1994_amd_smoke
```

Run the corresponding multilevel-LASD smoke case:

```bash
python benchmark/Andren1994/run.py \
  --sgs lasd --end-ft 0.001 --sample-start-ft 0 \
  --output-dir benchmark_results/andren1994_multilevel_lasd_smoke
```

Continue a canonical checkpoint to the paper's final time:

```bash
python benchmark/Andren1994/run.py \
  --sgs amd --end-ft 10 --sample-start-ft 7 \
  --restart benchmark_results/andren1994_amd_40cubed/checkpoint.npz \
  --output-dir benchmark_results/andren1994_amd_40cubed
```

Outputs are written below `--output-dir`:

- `checkpoint.npz`, `history.csv`, `profiles.csv`, and `summary.json` retain the
  common accepted-step/restart contract;
- `normalized_profiles.csv` uses the paper's `zf/u*` and variance/flux scaling;
- `spectra.csv` contains the four resolved discrete-mode spectra at the level
  nearest `zf/u*=0.1`; `mode E_mode` is the discrete counterpart of the scaled
  ordinate printed in paper Fig. 15 (the caption factor is not applied twice);
- `fig12_budget_profiles.csv` and `fig13_budget_profiles.csv` contain all five
  RHS terms, tendency, and a reported closure residual;
- `andren1994_profiles.png` is a compact run diagnostic.

The AMD checkpoint contains velocity, passive scalar, full-projection pressure,
profile/budget samples, and TKE/non-stationarity history. A checkpoint made by
the earlier resolved-only runner is intentionally rejected for a complete
comparison: the missing scalar history cannot be reconstructed after the fact.

The SGS energy and scalar variance are explicitly diagnostic local-equilibrium
quantities. AMD predicts deviatoric stress and scalar flux; it does not evolve
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
damping. The scalar is passive and therefore has no momentum feedback.

The compact public specification and extracted factual data are under
[`reference/`](reference/Andren1994.md). The original article is linked there
and is not stored in this repository.

To overlay a completed run directly on the original paper figures without
redistributing the article itself, download the linked PDF and run:

```bash
python benchmark/Andren1994/overlay_paper_figures.py \
  --paper-pdf /path/to/qj-1457-1994.pdf \
  --results benchmark_results/andren1994_amd_40cubed_complete
```

This produces one sheet, `andren1994_all_figures_jaxwind_overlay.png`, containing
all 19 numbered paper figures in order. A complete AMD run registers directly
comparable curves on Figs. 2–15. Figs. 12 and 13 overlay the same six colored
JAX-Wind budget terms on each original-code panel; other plots use red for a
resolved/total JAX-Wind curve and blue for the diagnostic SGS contribution.
Figs. 16–19 are from the paper's separate fixed-diffusivity experiment and
remain explicitly paper-only rather than receiving a misleading substitute.

# Andrén et al. (1994) neutral Ekman benchmark

This benchmark reproduces the neutral, rigid-lid Ekman LES intercomparison
configuration of Andrén et al. (1994) as an independent active case.

Run the canonical `40 × 40 × 40` case from the repository root:

```bash
python benchmark/Andren1994/run.py
```

The current runner uses the MAC finite-volume momentum path, a matrix-free
symmetric-GMG/PCG pressure solver, and momentum LASD. The restriction is the
volume-weighted adjoint of prolongation. Use `--linear-solver gmres` for the
restarted GMRES reference path.
The stiff vertical part of LASD principal diffusion defaults to the
third-order ARS(2,3,3) IMEX path and is solved directly along each vertical
column. `CFLnu` therefore diagnoses the complete SGS operator; timestep
selection retains only the explicit horizontal diffusion limit. Use
`--sgs-time-integration explicit` for the fully explicit projected SSPRK3 path.
Its two Germano test filters are three-dimensional compact physical-space
top-hat convolutions; they do not call FFTs. Horizontal filter boundaries are
explicit (`periodic` for this Andrén case and `reflect` for a nonperiodic
homogeneous-Neumann boundary). The rigid lower/upper boundaries use even
reflection for tangential velocity and odd reflection for wall-normal
velocity. LASD memory is updated after every accepted flow step so that its
trajectory CFL follows the enlarged momentum timestep. The width-two and
width-four overlap filters use exact compact `reduce_window` forms rather than
materializing every shifted field. Their first separable pass shares one
radius-two padding and one three-point box sum. Accepted-step LASD work is
split into two JIT executables: local gradient/Germano statistics and
Lagrangian history/coefficient finalization. This prevents the filter graph
from being fused into the complete momentum/projection timestep and avoids a
field-sized 21-component intermediate between executables. The default
advances to `ft=0.1`; use
`--end-ft 10 --sample-start-ft 7` for the canonical final averaging window.

The default projection mode is the third-order FPJ-2 fast projection. The
first two accepted steps use the full three-PPE SSPRK3 startup, after which
intermediate stages use variable-step extrapolated pseudo-pressure and only
the final stage solves a PPE. Pressure history and its two actual timesteps are
checkpointed. Use `--projection-method full` for the three-PPE reference path;
an abrupt timestep change beyond `--fpj2-timestep-ratio-limit` also triggers
one exact fallback step automatically.

`run_lasd.py` is retained as the legacy coupled momentum/scalar z-slab
implementation for reference. New benchmark work should use `run.py`.

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
the level nearest `zf/u*=0.1`.

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
Figs. 2, 4(a,b), 5, 6, 7, 8, 13(a-d), 14(a,b), and 15(a-d) when the corresponding
diagnostics exist. Fig. 13 overlays the same six colored JAX-Wind budget terms
on all four original code panels; other plots use red for a resolved/total
JAX-Wind curve and blue for a diagnostic SGS contribution. Paper-only tiles
remain explicit rather than receiving a misleading substitute quantity.

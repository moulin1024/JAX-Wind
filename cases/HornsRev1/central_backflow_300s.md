# Five-minute Horns Rev outlet comparison

`fv_hornsrev1_80_uniform10_central_backflow_300s.toml` retains all 80 V80
locations, the 8192 × 8192 × 1024 m domain, and the 512 × 512 × 256 grid of
the earlier uniform-inflow case. It initializes u=10 m/s, v=w=0 and enforces
that inlet throughout the run. It uses central momentum transport, full RK3,
GMG, dt=0.25 s, and 1200 steps (300 s). The original rough-wall/AMD settings
are retained. There are 100 frames and three-second diagnostic samples.

The treated run enables `outlet_backflow="energy"`. Its matched control uses
`"none"`; both use full RK3 and identical physical/numerical settings otherwise.
These are pressure outlets, not one-way valves. Physical reversal remains
possible, so zero reversal must be measured rather than assumed.

For the high-x outlet the pressure condition is p=-|u|² where u_n<0, zero
otherwise. At the y sides the condition uses perturbation velocity relative
to the ambient streamwise inlet, q=u-(U_ref,0,0), so uniform throughflow does
not spuriously drive transverse motion. The lateral boundary contribution to
perturbation energy, `-(p+|q|²/2)u_n`, is nonpositive for p=-|q|² on incoming
flow and p=0 on outgoing flow. The pressure operator's fixed x-end corner
constraints are retained. This explicit RK implementation is not a proof of
unconditional discrete energy stability or a nonreflecting boundary.

Diagnostics distinguish:

- Negative streamwise u at the downstream outlet and anywhere in the domain.
- Incoming outward-normal velocity at each lateral boundary; this includes
  physical entrainment and is not streamwise wake wraparound.
- Reverse volume flux and boundary area fraction, including a 0.01 m/s
  inward-speed threshold to separate negligible sign changes from stronger flow.
- Inlet error, global boundary-flux imbalance, and divergence.
- Upstream raw-face second differences at the nearest hub-height level, at
  least 160 m ahead of the first turbine. These include physical gradients;
  they are not a unique numerical-dispersion measure.

The last turbine row is at x=6855 m, leaving 1337 m to the outlet. The nominal
travel time at 10 m/s is 133.7 s; actual wake convection can be slower. Five
minutes exercises the downstream outlet but is not an equilibrated full-farm
wake simulation.

Run and inspect on an allocated GPU compute node:

```bash
export PYTHONPATH=src
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAXWIND_V80_FAST="$PWD/cases/HornsRev1/turbines/V80/CustomRotor.fst"
python -m jaxwind run cases/HornsRev1/fv_hornsrev1_80_uniform10_central_backflow_300s.toml
python tools/render_hornsrev1_farm.py outputs/hornsrev1_80_uniform10_central_backflow_20260916/run
python tools/analyze_hornsrev_backflow.py outputs/hornsrev1_80_uniform10_central_backflow_20260916/run
```

The saved experiment directories contain the resolved cases and source
provenance. The matched control is
`outputs/hornsrev1_80_uniform10_central_outlet_baseline_20260916/`.

## September 16 five-minute result

The treatment did **not** remove the upstream striping. Both 300 s runs had
zero sampled reverse-flow area at the downstream outlet, so this experiment
does not demonstrate elimination of downstream backflow: the control did not
exhibit it either. Upstream raw-face velocities in the treated case remained
positive (9.215–10.188 m/s),
while adjacent faces differed by up to 0.731 m/s.
The central momentum stencil uses its nonperiodic branch for this mesh; it does
not connect opposite x boundaries. These observations support distinguishing
upstream grid-scale oscillations from actual reverse transport. They do not
uniquely identify every contribution to the upstream velocity field.

| Sampled metric | Ordinary pressure outlets | Energy backflow treatment |
|---|---:|---:|
| Minimum downstream outlet u [m/s] | 2.773387 | 2.753145 |
| Maximum downstream reverse-flow area [%] | 0 | 0 |
| Maximum lateral inward speed [m/s] | 0.070491 | 0.377987 |
| Minimum domain u [m/s] | -2.088312 | -2.088337 |
| Final upstream second-difference RMS [m/s] | 0.087930299 | 0.087931566 |
| Maximum divergence [1/s] | 2.682e-07 | 4.622e-07 |

The final oscillation measure changes by only 0.0014%, while the
strongest lateral inward velocity increases by a factor of 5.36.
This lateral extension should remain experimental; this benchmark does not
support adopting it as the cure for the upstream artifact. The high-x
reverse-flow pressure branch was not exercised at the diagnostic sample times.

The strongest treated lateral reversals in the final snapshot are in the
lowest cell, z=2 m. At those locations, streamwise velocity is 3.96 and
4.65 m/s, compared with the fixed 10 m/s reference used by the pressure switch.
Evaluating the incoming-flow pressure formula there gives −36.6 and −28.8 m²/s²,
whereas the formula is zero for outgoing flow. A sharp switch acting on this
near-wall reference mismatch is a plausible source of the lateral oscillations;
a dedicated test is needed to establish the mechanism and validate a revision.
These evaluated pressures are diagnostics from the final state, not a record
of the weighted pressure applied at each RK substage.

Both runs completed 1200 steps, with peak CFL about 0.54, an exactly prescribed
inlet, and 100 diagnostic samples/frames at 3 s intervals. These samples cannot
exclude shorter reversal events between outputs. Five minutes is an early
transient, shorter than one inlet-speed traversal of the full 8192 m domain
(819.2 s), and this uniform-inflow test does not validate a log-law profile.
The new lateral pressure/projection checks and existing uniform-farm tests
passed (7 tests), as did the full-RK3 farm adapter test (1 test).

Artifacts:

- [Treated summary and metrics](../../outputs/hornsrev1_80_uniform10_central_backflow_20260916/run/backflow_analysis/README.md)
- [Treated wake movie](../../outputs/hornsrev1_80_uniform10_central_backflow_20260916/run/hornsrev1_hub_u.mp4)

Further full-farm debugging and queued control postprocessing were stopped at
the user's request. Both 300 s integrations and their checkpoints completed;
the matched numbers above were read from those checkpoint histories. New
debugging uses the [small single-turbine case](single_v80_backflow_debug.md),
with 128 times fewer cells. No further farm runs are part of this debug loop.

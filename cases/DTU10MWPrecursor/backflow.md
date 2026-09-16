# Central momentum transport with outlet backflow stabilization

`fv_central_backflow.toml` derives the historical DTU case and selects central
precursor momentum transport, horizontal upwinding in the open main domain,
full RK3, the Porté-Agel gradient correction, and
`numerics.outlet_backflow = "energy"`. Historical cases and completed comparison
runs remain available. The derived case uses the requested half-resolution
64 × 32 × 128 mesh in the historical domain, with the fixed 0.1 s
recording/replay cadence. The existing recorded-inflow driver does not
support adaptive timesteps; the standalone outlet verification uses CFL 0.9.

The periodic warmup/precursor has no outlet. Its central advection operator and
SGS physics are unaffected by the outlet option. In the main domain, the new
condition supplies a kinematic pressure at the high-x face:

```
p_out = -|u_out|²  when u_normal < 0
p_out = 0         otherwise
```

This is the sharp-switch, zero-normal-viscous-traction specialization of OBC-B
in [Dong & Shen (2015)](https://doi.org/10.1016/j.jcp.2015.03.012). At the continuum
level its advective-plus-pressure boundary power is
`-(p_out + |u_out|²/2) u_normal = -|u_out|² |u_normal|/2`, which is nonpositive.
The implementation evaluates pressure explicitly from each incoming RK stage,
using the projected normal velocity and staggered tangential velocities
averaged to the same outlet locations. This known pressure-gradient forcing
is combined with the same Wray coefficients as the other explicit terms.
It is not the paper's complete
semi-implicit algorithm and carries no unconditional or fully discrete energy
stability guarantee.

The known outlet-pressure gradient enters both the Poisson right-hand side and
the velocity correction. Normal flow is not clipped or reset after projection.
The projected outlet flux is retained when evaluating the next momentum RHS.
Vertical fluxes remain central in the open interior. When scalar transport is active, incoming
scalar flux uses the supplied inflow scalar as the external reservoir value;
physical end cells continue to evolve, including local sources.

The implementation requires full RK3 with a pressure projection at every stage.
It supports a prescribed low-x inlet, high-x outlet, and impermeable top/bottom.
Lateral pressure outlets use ordinary extrapolation and pressure projection;
the experimental tangential-energy pressure switch is no longer called by the
integrator. Periodic/solid lateral boundaries remain supported. A second x
pressure outlet and active scalars with open lateral boundaries remain rejected.

The policy `central-open-upwind` calls the original central operator in the
periodic precursor. In the open main domain it uses MUSCL-MC horizontal fluxes
and central vertical fluxes. See the [small-case correction and exact precursor
check](../HornsRev1/single_v80_backflow_fix.md). Long-time open-domain turbulent
profile preservation still requires validation.

Open-stage histories record outlet backflow area fraction, minimum normal
velocity, the pressure condition evaluated from the sampled state, and net
streamwise volume flux. These distinguish actual reversal from centered-scheme
dispersion or ordinary upstream turbine induction. The treatment does not
remove periodic wraparound or guarantee a nonreflecting outlet.

Run the derived workflow with the appropriate DTU OpenFAST deck configured:

```bash
PYTHONPATH=src python -m jaxwind workflow \
  cases/DTU10MWPrecursor/fv_central_backflow.toml
```

Independent reversing-vortex verification (on a compute node):

```bash
PYTHONPATH=src python tools/check_open_backflow.py \
  outputs/central_backflow_verification --cfl 0.9
```

The September 16 verification uses a divergence-free vortex already intersecting
the outlet, with central/RK3 in both variants and no SGS or wall model. At CFL
0.9 the stabilized run retained real backflow, remained finite for 12 s, and
had maximum sampled divergence 2.13e-11 /s and net-flux error 3.22e-12 m³/s.
The unstabilized variant was also stable in this small case. This demonstrates
activation and mass conservation, not improved turbine-wake accuracy. The
DTU main-domain wake and long-time log-law behavior still require validation.

## Rough-wall lateral-outlet limitation

The [300 s Horns Rev comparison](../HornsRev1/central_backflow_300s.md) found
that the experimental lateral extension left upstream oscillations unchanged
and increased the maximum sampled lateral inward speed from 0.070 to
0.378 m/s, with the strongest treated reversal near the ground. Do not infer
rough-wall lateral-boundary performance from the uniform-throughflow or
streamwise reversing-vortex tests. The lateral reference and sharp pressure
switch need further validation. The periodic precursor has no such boundary.

The lateral limitation above is historical: the faulty lateral pressure switch
was subsequently removed from integration in the small-case correction.

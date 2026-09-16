# HITSZ liquid-nitrogen jet (finite volume)

See the [detailed nitrogen-jet guide](NITROGEN_JET.md) for model selection,
source equations, main-only turbine injection, MP4 rendering, and the completed
LN2/no-jet centerline comparison.

> Run all commands below on a compute node, including configuration checks.
> This case uses schema version 1. Historical outputs cannot be resumed;
> regenerate inputs in the new format. See [verification](../../doc/verification.md).

This case isolates the hub-height LN2 jet in still air at 80% relative humidity: no turbine, no tunnel
pressure forcing, and no background velocity.  The 3 m x 3 m x 1.8 m domain
uses 256 cells in every direction.  The nozzle is at one-quarter of the streamwise domain and centred in y
at `(0.75, 1.5, 0.876) m` and points in positive x.

The inlet at x=0 is quiescent ambient air and x=3 is the pressure outlet.  Both
y sides are impermeable wall-modelled surfaces.  They use the same prescribed
Monin--Obukhov stress source as the floor: zero resolved viscous/SGS shear at
the boundary plus log-law stress in the adjacent control volume.  The ceiling
is impermeable and free-slip.

At 0.0125 kg/s and 8 m/s through a 10 mm bore, homogeneous-equilibrium density
closure gives a vapor quality of 0.217596 (21.76% by mass).  That vapor enters
as 77.34 K nitrogen gas.  The remaining liquid is represented by a fixed JAX
parcel buffer with a 50--300 micrometre Rosin--Rammler distribution.  The
carrier coupling includes drag/reaction momentum, Ranz--Marshall sensible and
latent heat transfer, evaporation, nitrogen transport, low-Mach expansion,
thermal/compositional buoyancy, ambient-water saturation adjustment, fog and
ice, molecular transport, and AMD subfilter transport. Classical static Smagorinsky transport
(coefficient 0.16) is an optional alternative.

Carrier density is now evaluated from the constant-thermodynamic-pressure
air/N2/water-vapour ideal-gas mixture rather than held at its ambient value.
The pressure correction acts on face momentum and balances the EOS density
change, direct nozzle vapour, and parcel evaporation in the discrete
continuity equation. Parcel Reynolds number, buoyancy, reaction acceleration,
and thermal/species source conversion use the local carrier density.

The mapped inlet case replaces the unresolved effective source with a resolved
10 mm inlet. It uses one-sided tanh clustering at x=0 and central tanh
clustering at the nozzle y-z axis. At 256^3, strength 2.2 gives local spacings
of about 1.28 mm x 1.29 mm x 0.76 mm. The minimum x spacing sets the explicit
step, so this case uses 50 microseconds rather than the uniform-mesh 500
microseconds:

```bash
jaxwind run \
  cases/HITSZLiquidNitrogenJet/fv_256_low_mach_rk3_inlet_mapped.toml \
  --max-steps 2
```

Run a short validation first:

```bash
jaxwind run cases/HITSZLiquidNitrogenJet/fv_256.toml --max-steps 2
```

The configured carrier and thermodynamic integration uses three-stage RK3
with a pressure projection at each stage. The optional `fast-rk3` path instead
uses a lagged pressure gradient and a final-stage projection. The configured
production run is 1 second
(2,000 steps):

```bash
jaxwind run cases/HITSZLiquidNitrogenJet/fv_256.toml
```

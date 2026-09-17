# Water spray for 16 m × 16 m × 4 m LES cells

User-confirmed application, 2026-09-17. This is the implementation contract and
priority correction, not a claim that the complete model is already implemented
or physically validated. Each cell is 1,024 m³; these are not domain dimensions.

Subsequent user decision, 2026-09-17: first reproduce the documented Fluent DPM
baseline. See [FluentDPMWater](../../cases/FluentDPMWater/README.md). That path uses
the local CFD gas state and takes precedence for current implementation. The
conditional plume environment proposed below is a separate potential extension,
not a feature to introduce silently into the Fluent reference calculation.

## Intended prediction

Predict where injected water travels, how much liquid survives downstream,
where evaporation removes sensible heat, and how cooling, humidity and liquid
loading affect a turbine wake. Resolve the atmospheric flow and turbine wake
at the chosen LES resolution. Represent the nozzle jet and narrow spray plume
below that resolution. Do not replace the atmospheric SGS closure with a fitted
round-jet viscosity or impose a measured cooling field.

## Calculation contract

1. **Nozzle source.** Specify mass flow, liquid temperature, position/orientation,
   and joint droplet size/velocity distribution at a defined post-atomization
   plane. Preserve source mass, momentum and enthalpy. Laboratory defaults are
   not automatically application inputs.
2. **Subgrid spray environment.** Evolve an unresolved plume/envelope containing
   finite air inventory, temperature, humidity, mean velocity and spatial extent.
   Droplets exchange heat and mass with this local environment. Ambient air
   enters through an explicit entrainment/mixing model with conservative
   accounting. This representation may continue over multiple LES cells.
3. **Downstream droplets.** Transport size-resolved parcels with carrier advection,
   finite slip, gravity/settling and evaporation. Account for droplet temperature
   and remaining mass. Add unresolved turbulent dispersion only with a justified
   turbulence input and an inhomogeneous, finite-inertia treatment. Homogeneous
   OU verification alone does not establish that treatment.
4. **LES feedback.** Deposit water vapor, momentum and energy through normalized
   conservative transfers into intersected LES cells. Keep plume and ambient
   inventories disjoint or use an equivalent conservative conditional-state
   formulation. Never count entrained gas or phase exchange twice. Couple the
   resulting moist thermal state and liquid loading to wake buoyancy.
5. **Loss of subgrid segregation.** Release the conditional plume state when a
   justified mixing/resolution criterion makes it unnecessary, preserving its
   inventories and surviving droplets. Neither nozzle-cell exit nor full
   evaporation is a mandatory handoff condition. A numerical deposition kernel
   is not a physical plume width or a physical mixing timescale.

Local plume saturation and cell-average humidity are distinct. Evaluating every
droplet against the whole-cell mean can erase the locally cooled, humid region;
its effect on evaporation must be assessed rather than assumed negligible.
Conversely, retaining an isolated saturated core without ambient mixing can
suppress evaporation excessively. Finite entrainment and exchange are central
model requirements, even when no fine radial jet profile is retained.

## Implementation status and immediate priorities

The legacy atmospheric workflow injects entrained, common-velocity,
common-temperature mist. An opt-in Fluent-reference spatial DPM path now couples
finite-inertia parcels, local evaporation, full species enthalpy and explicit LES
scales to the single-turbine workflow; see the baseline README for evidence and
remaining Boussinesq/SGS-energy limitations. The waterjet investigation also has finite-inertia parcels.
Separate modules verify conservative phase exchange, moving parcels, injection,
and carrier energy accounting. Those components are useful foundations but are
not yet the complete coarse-cell turbine-spray calculation described here.

The untested round-jet profile-viscosity integration was removed before any run.
Keep the independently assessed profiles as supporting evidence for an integral
entrainment closure; do not continue radial-profile optimization as the main
application task. No waterjet or wake improvement has been demonstrated by
this scope correction.

Under the subsequent Fluent-baseline decision, prioritize matched downstream
transport/cooling validation of the integrated local-cell DPM. A finite-inventory
conditional plume remains a separate extension to assess after that baseline.
Coarse-cell mixing sensitivity and independently specified application nozzle
conditions remain essential; conservation alone does not establish their accuracy.

## Validation at the application scale

Freeze source inputs and downstream observables before comparisons. Measure
remaining liquid flow and size distribution, evaporated fraction, plume centroid
and spread, water/enthalpy fluxes, cooling deficit and humidity across downstream
planes. Add wake velocity/temperature changes and rotor-area diagnostics when
the turbine configuration is defined. Use a matched dry baseline.

Test subcell nozzle translation, plume crossing of cell boundaries, timestep,
parcel count, mixing/handoff treatment and averaging duration. Use auxiliary
mesh changes to measure sensitivity without making nozzle resolution a
production requirement. Check gas-plus-liquid budgets, positivity and exchange
ownership separately from agreement with measurements.

The user-authorized physical acceptance remains 20%. Set each observable's
normalization and near-zero handling before validation; a percentage of absolute
or Celsius temperature does not by itself establish the accuracy of the cooling
deficit. Preserve the existing waterjet sensor screen as its own benchmark.
The waterjet is a component check; it cannot alone validate turbine-wake cooling
on 1,024 m³ cells. The initial scope correction involved no simulation; subsequent Fluent-reference
integration checks are recorded in its case README.

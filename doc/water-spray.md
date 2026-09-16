# Water spray through shared moisture physics

The recorded-inflow, single-turbine Boussinesq workflow supports humidity,
cloud liquid/ice, and injected fine water mist. Nitrogen jets and this workflow
use the same saturation pressures and enthalpy-conserving cloud adjustment in
`jaxwind.physics.moisture`; existing cryogenic imports remain compatible.

Use `cases/HITSZWindTunnel/fv_far_wake_water.toml` as an example. It inherits the
HITSZ baseline and its recorded precursor input directory. After producing those
matching warmup and inflow artifacts, run on a compute node:

```bash
python -m jaxwind workflow cases/HITSZWindTunnel/fv_far_wake_water.toml \
  --stage main --output runs/hitsz-water
```

The example's flow and diameter are exploratory inputs, not measured nozzle
calibration. Set `workflow.input_directory` to your precursor workflow output.
For a new precursor, use the turbine-free baseline workflow first. The moisture
and injection coupling activates only in the open-inflow stage; periodic warmup
and precursor remain dry. Direct `jaxwind run` without an open-inflow stage,
synthetic inflow, and controlled multi-turbine farms are currently unsupported.

## Configuration

```toml
[physics.moisture]
temperature_offset_k = 300.0
reference_temperature_k = 300.0
ambient_relative_humidity = 0.5
pressure_pa = 100000.0
dry_air_density_kg_m3 = 1.225

[physics.water_spray]
mass_flow_rate_kg_s = 0.001
streamwise_offset_m = 0.30
standard_deviation_m = [0.15, 0.15, 0.12]
droplet_diameter_m = 20.0e-6
ramp_time_s = 1.0
```

The transported scalar must be temperature in kelvin or a temperature anomaly:
`T = scalar + temperature_offset_k`. For absolute temperature set the offset to
zero. `reference_temperature_k` is the Boussinesq reference: the main-stage
thermal buoyancy coefficient is set to `9.81/reference_temperature_k` and gains
the moisture correction. A constant temperature offset drops out of the existing
horizontal-mean hydrostatic subtraction. Configure `workflow.evolve_scalar =
true` and `numerics.time_integration = "fast-rk3"`.

Humidity initializes from the resolved temperature and specified relative
humidity. Since recorded precursor planes have no humidity channel, inflowing
vapor retains the initial boundary mixing-ratio profile. Initial cloud and spray
reservoirs are zero. Omitting `physics.water_spray` enables humidity/cloud physics
alone. A zero mass flow disables injection while retaining moisture. Prescribed
LN2 cooling and this atmospheric moisture option cannot be combined in one case.

Injection adds liquid mass and droplet number in a normalized downstream Gaussian
kernel, centered at the turbine's `(x + offset, y, hub_height)`. Widths specify
resolved injection support, not an evaporation distance. Neither vapor nor a
prescribed cooling power is injected. Droplet number and mass are transported;
the local monodisperse diameter shrinks as liquid evaporates.

## Physics and numerical scope

The first closure represents dilute, already entrained fine mist after nozzle
atomization and velocity/thermal relaxation. Droplets share the carrier velocity
and temperature. There is no nozzle momentum source, prescribed injection
water temperature, slip, settling, breakup, collision or deposition model.
Large ballistic droplets require a different transport/thermal closure.

Finite-rate evaporation uses the spherical zero-slip diffusion limit `Sh=2`,
with a Stefan logarithmic humidity driving force. The mass-transfer structure
follows the [FDS particle equations](https://github.com/firemodels/fds/blob/master/Manuals/FDS_Technical_Reference_Guide/Particle_Chapter.tex),
but this is a reduced common-temperature bulk closure, not the FDS droplet model.
A backward-Euler cell solve uses the evolving gas humidity and temperature.
Evaporation is bounded by available liquid, saturation and the sensible heat
available above freezing; it stops in saturated air. Spray liquid remains
separate from equilibrium cloud water and is never passed to instantaneous cloud
adjustment. The shared cloud solve handles vapor/cloud condensation, evaporation,
freezing and melting. Spray freezing itself is not modeled.

The dilute thermodynamic convention conserves local total water and
`cp_d T + L_v qv - (L_s-L_v) qi` during phase exchange. Condensate sensible heat
is neglected. Constant air density and incompressible volume are the Boussinesq
approximation; gas mass expansion is not modeled. All water ratios use kg/kg dry
air, and droplet number uses droplets/kg dry air.

Buoyancy includes thermal cooling, the positive vapor correction, and negative
cloud/spray loading. Negative wake buoyancy therefore follows the evolving
state, not a prescribed downward forcing. The moist virtual-temperature offset
is `T_ref * [(Rv/Rd - 1)*(qv-qv_ambient) - ql - qi - qspray]`.

Moisture has conservative upwind advection, molecular/SGS diffusion, and explicit
subcycling for positivity. Boundaries use incoming ambient advective flux and
outgoing interior concentration, with impermeable vertical water boundaries.
Microphysics and injection use a symmetric split around carrier and moisture
transport; first-order moisture transport limits the coupled temporal accuracy.
The carrier retains its existing temperature boundary treatment and pressure
projection. This is not a validated prediction of turbine performance or spray
trajectory; assess diameter, source width, grid, timestep and humidity sensitivity.

## Output and verification

Full-state checkpoints preserve `moisture.vapor`, `cloud_liquid`, `cloud_ice`,
`spray_liquid`, and `spray_number`. Exact resume restores all five fields. Saved
flow frames include their hub-height and center-plane slices alongside actual
temperature scalar and velocity. Existing dry and LN2 state layouts are unchanged.

Run the focused checks on a compute node:

```bash
JAX_PLATFORMS=cpu python -m pytest -q -o 'python_files=test_*.py' \
  tests/fv/test_moisture.py tests/test_moisture_config.py \
  tests/fv/test_cryogenic.py tests/fv/test_open_atmospheric.py
```

These check local mass/enthalpy conservation, saturation inhibition, size
sensitivity, vapor/loading buoyancy, conservative transport, source normalization,
coupled cooling/downward response, full-state restart, and existing LN2 behavior.

## Literature benchmark

The [Montazeri 2015 case 3 benchmark](../cases/WaterSprayMontazeri2015/README.md) includes two completed GPU runs and a comparison against measured outlet temperatures. The current model fails the temperature-distribution comparison and exceeds dilute-loading assumptions; it is not yet validated for air-assisted spray cooling.

A finite-slip, finite-temperature parcel closure is now available in the
[benchmark investigation](../cases/WaterSprayMontazeri2015/investigation/README.md).
It couples to the same humidity/cloud physics and is being verified against the
literature case. It is not yet exposed in the turbine case configuration, and
compressed-air injection remains to be implemented and validated.

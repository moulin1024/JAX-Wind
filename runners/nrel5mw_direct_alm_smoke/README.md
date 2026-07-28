# NREL 5 MW direct modal-aeroelastic ALM smoke test

This case cold-starts JAX-Wind directly from a neutral logarithmic velocity
profile. It does not load or run a precursor/warmup. A three-blade
modal-aeroelastic actuator line is placed over the center of the ground plane at
`(x, y) = (256 m, 256 m)`, with its 90 m hub height read from the OpenFAST
input deck.

The requested grid is `128 × 128 × 512` over a `512 m × 512 m × 512 m`
domain, giving `4 m × 4 m × 1 m` cells and 8,388,608 total cells. The
animation configuration advances 60 0.05 s AB2 steps using float32, a static
Smagorinsky closure, and the distributed SPIKE pressure solve. It captures
the rotor plane and hub-height plane every two steps.

The included deck follows ordinary OpenFAST/ElastoDyn/AeroDyn file structure.
Its rotor radii, hub height, initial speed, precone, chord, and twist are based
on the NREL 5 MW reference turbine. The ElastoDyn blade file supplies two
flapwise modes, one edgewise mode, structural damping, distributed mass, and
flap/edge stiffness. Modal deformation and velocity affect the aerodynamic
loads, and those loads feed back into the structural state every CFD step.

The two compact airfoil polars are smoke-test approximations and must not be
used for scientific load or wake validation. Rotor speed, hub, nacelle, tower,
drivetrain, pitch, and yaw remain prescribed; this is blade-modal coupling,
not complete OpenFAST/ElastoDyn or BeamDyn parity.

Validate without importing JAX:

```bash
jaxwind runners/nrel5mw_direct_alm_smoke --dry-run
```

Run the smoke animation:

```bash
jaxwind runners/nrel5mw_direct_alm_smoke --overwrite
```

This runner writes resolved configuration, flow and structural history,
summary metadata, and compact two-dimensional flow slices, but deliberately
does not write a full-field checkpoint.

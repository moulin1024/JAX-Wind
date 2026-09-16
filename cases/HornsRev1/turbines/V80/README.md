# V80 rotor data

This directory packages the user-supplied `V80.zip` rotor in the same
OpenFAST/AeroDyn input format used by the DTU10MW AD-BEM case.

```text
CustomRotor.fst
├── CustomRotor_ElastoDyn.dat
└── CustomRotor_AeroDyn.dat
    ├── CustomRotor_AeroDyn_blade.dat
    └── airfoils/CustomAirfoil.dat
```

The five aerodynamic/geometry files retain their archive contents. The
archive placed `CustomAirfoil.dat` at its root, although AeroDyn references
`airfoils/CustomAirfoil.dat`; it is installed at the referenced location here.
macOS metadata and the archive's temporary HITSZ-based workflow, initial
profile, and reference results are not imported into this offshore case.

## Supplied rotor

- Three blades; tip radius 40 m; hub radius 3.21582115219 m.
- 108 blade stations with supplied chord and twist, using one airfoil polar.
- 139 angle-of-attack samples, approximately −4.70 to 19.85 degrees.
- Fixed operating point: 16.7 rpm, zero collective pitch.
- The source deck specifies 90 m hub height. The Horns Rev workflow explicitly
  places the rotor at 70 m using the supported hub-height override.

These are user-supplied custom rotor tables, not an independently verified
manufacturer model. The archive describes conversion from chord, twist,
lift, and drag spreadsheets; those spreadsheets were not included. The
minimal deck is for JAX-Wind's rigid, quasi-steady importer, not a complete
standalone aeroelastic OpenFAST simulation. The limited polar range does
not supply a full-angle stall model.

## Use

From the repository root, on a compute node:

```bash
export JAXWIND_V80_FAST="$PWD/cases/HornsRev1/turbines/V80/CustomRotor.fst"
jaxwind check cases/HornsRev1/fv_workflow_v80.toml
jaxwind workflow cases/HornsRev1/fv_workflow_v80.toml --stage warmup
```

`fv_workflow_v80.toml` uses the same `[physics.turbine]` schema as the DTU10MW
case. It inherits the Horns Rev offshore flow and 8192 x 8192 x 1024 m domain
at 16 x 16 x 4 m spacing. The rotor is centered at (4096, 4096, 70) m.
The warmup is 36000 simulated seconds with adaptive CFL 0.9. The example
precursor/main stages each run 1200 fixed 0.1 s steps (120 s); these are
setup defaults, not a validated production sampling duration. Warmup and
precursor use periodic FFT projection; the main stage uses open inflow,
GMG projection, and AD-BEM forcing with background pressure forcing disabled.

No V80 nacelle or tower dimensions were supplied. Their drag coefficients
are set to zero to avoid applying the generic DTU-sized body geometry to
this rotor. Rotor forcing remains enabled in the main stage.

This is a single-turbine setup, matching the DTU example; an 80-turbine farm
layout is not included. Existing Horns Rev warmup configurations and saved
simulation results are unchanged.

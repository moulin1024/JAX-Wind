# DTU 10 MW OpenFAST input data

The `upstream/` directory is an unmodified snapshot of
[Seager1989/DTU10MW_FAST_LIN](https://github.com/Seager1989/DTU10MW_FAST_LIN)
at commit `103edc88d2352c3a615079539f54eb0fab4a76ba`, downloaded 2026-09-21.
All 57 upstream files are included; Git metadata is excluded.
[provenance.json](provenance.json) records SHA-256 hashes for every file.

The upstream model targets OpenFAST 3.0.0 and combines the DTU 10 MW rotor
with the LIFES50+ NAUTILUS Gulf of Maine floating platform. Both AeroDyn14
and AeroDyn15 primary files, blade geometry, airfoil polars, structural data,
platform data, wind inputs, performance tables, and upstream comparisons
are retained. Use the **A15** primary file with JAX-Wind.

## License and attribution

The upstream files are distributed under the retained
[Apache-2.0 license](upstream/LICENSE). See the original
[README](upstream/README.md) for model context and its requested citation:

Du, X., Liang, J., Muro, J. L., Qian, G., Burlion, L., & Bilgen, O. (2024).
Development of a control co-design optimization framework with
aeroelastic-control coupling for floating offshore wind turbines.
Applied Energy, 372, 123728.

## JAX-Wind use

From the repository root, on a compute node:

```bash
export JAXWIND_DTU10MW_FAST="$PWD/cases/DTU10MWPrecursor/turbines/DTU10MW/upstream/DTU_10MW_NAUTILUS_GoM_A15.fst"
jaxwind check cases/DTU10MWPrecursor/fv_workflow.toml
```

The rigid turbine importer successfully resolves the AeroDyn15 deck and
all referenced aerodynamic inputs: three blades, 38 blade stations,
seven airfoil polars, 2.8 m hub radius, **89.2 m tip radius**, 9.6 rpm,
zero pitch, and 117.76719422649163 m hub height. Import validation does
not establish aerodynamic accuracy or standalone OpenFAST execution.

The existing `tools/run_dtu10mw.py` fixed-step ALM schedule assumes an
89.15 m tip radius. This dataset is slightly larger, so its main-stage
tip-sweep CFL preflight rejects that schedule. Recalculate the preparation
and main timestep together for the actual radius before using that runner;
do not change the supplied geometry to bypass the check. The historical
AD-BEM workflow does not use that runner's tip-sweep preflight.

JAX-Wind's rigid importer uses the rotor geometry and quasi-steady polars;
it does not execute the floating platform, ROSCO, structural flexibility,
or AeroDyn dynamic-wake/unsteady-airfoil models. Curved/swept blade axes
are parsed but not represented by the current rigid actuator lines.

The original ServoDyn input contains an absolute ROSCO library path under
`/home/seager/Downloads/ROSCO/`. That library is not supplied by upstream.
Standalone OpenFAST use needs a compatible controller build and corrected
paths in a working copy; the preserved snapshot is not a turnkey OpenFAST
installation.

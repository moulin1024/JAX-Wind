# Legacy WiRE-LES C++ Solver

This directory contains the former top-level C++ WiRE-LES solver. The active
The former JAX implementation is archived under `legacy/jax/`.
Run the commands below from `legacy/cpp/` so the relative build, config, output,
and tool paths remain valid.

Current C++ scope:

- single-process CPU, CPU MPI, and selected CUDA kernels
- FFTW3 for horizontal periodic `x/y` FFTs
- finite differences in `z`
- production-aligned vertical staggering:
  - `u`, `v`, `p`, `theta`, and passive scalars live at cell centers, shape
    `(nx, ny, nz)`
  - `w` lives at vertical faces, shape `(nx, ny, nz + 1)`
  - horizontal directions remain Fourier/collocated, matching the existing
    Fourier + finite-difference LES path rather than a full 3D MAC grid
- explicit Euler momentum step plus pressure projection
- dynamic neutral wall stress as a bottom momentum-flux source
- Coriolis/geostrophic forcing for neutral Ekman boundary layers
- potential-temperature and passive-scalar transport with prescribed surface
  fluxes
- classic Smagorinsky, buoyancy-aware AMD, and dynamic LASD SGS stress divergence with staggered
  stress placement
- fixed-Prandtl, minimum-dissipation scalar AMD, and dynamic scalar LASD heat-flux closure, including optional
  stable-inversion scalar diffusivity damping
- optional non-blocking MPI z-slab execution path with Fortran-style y-pencil
  pressure all-to-all
- no turbine model in the active C++ path

Module layout:

- `include/wireles/params.hpp`: grid, time-step, and run parameters
- `include/wireles/field.hpp`, `src/field.cpp`: field storage, initialization,
  and basic velocity boundary hooks
- `include/wireles/fft.hpp`, `src/fft.cpp`: FFTW-backed horizontal transforms
  and spectral derivative helpers
- `include/wireles/operators.hpp`, `src/operators.cpp`: finite-difference
  vertical center/face transfer operators, Laplacian, and divergence
- `include/wireles/pressure.hpp`, `src/pressure.cpp`: tridiagonal
  Fourier-mode pressure solve and staggered projection
- `include/wireles/wall.hpp`, `src/wall.cpp`: dynamic neutral wall stress and
  bottom face stress forcing
- `include/wireles/scalar.hpp`, `src/scalar.cpp`: potential-temperature and
  water-vapor transport, molecular scalar diffusivity, fixed-Prandtl/Schmidt
  SGS closure, surface scalar fluxes, and face-centered virtual-temperature
  buoyancy forcing
- `include/wireles/thermodynamics.hpp`, `src/thermodynamics.cpp`: legacy
  thermodynamic helpers retained for historical compatibility
- `include/wireles/sgs.hpp`, `src/sgs.cpp`: velocity-gradient helper and
  classic Smagorinsky or dynamic LASD SGS stress divergence on the staggered
  layout
- `include/wireles/timestep.hpp`, `src/timestep.cpp`: RHS assembly, physics
  coupling, explicit timestep, and diagnostics
- `src/main.cpp`: CLI parsing and run loop only

Build:

```bash
cmake -S . -B build-cpp-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpp-cpu -j 4
```

MPI is detected automatically when available. Disable it explicitly with:

```bash
cmake -S . -B build-cpp-cpu -DCMAKE_BUILD_TYPE=Release -DWIRELES_ENABLE_MPI=OFF
```

Run integration tests:

```bash
ctest --test-dir build-cpp-cpu --output-on-failure
ctest --test-dir build-cpp-cpu -L moeng --output-on-failure
ctest --test-dir build-cpp-cpu -L mpi --output-on-failure
```

The `moeng` integration label runs the full `40x40x48`, `4019`-step
`configs/largeeddy1993_moeng_lasd.toml` case and validates the generated
benchmark diagnostics.

Run a small Taylor-Green smoke case:

```bash
./build-cpp-cpu/wireles \
  --nx 32 --ny 32 --nz 32 \
  --steps 20 --log-every 5 \
  --dt 0.001 --nu 0.001
```

Run the Moeng-style dry CBL case from config:

```bash
./build-cpp-cpu/wireles \
  --config configs/largeeddy1993_moeng.toml
```

Run the same CBL setup with dynamic momentum LASD and dynamic scalar LASD:

```bash
./build-cpp-cpu/wireles \
  --config configs/largeeddy1993_moeng_lasd.toml
```

Run a neutral Ekman boundary-layer case:

```bash
./build-cpp-cpu/wireles \
  --config configs/neutral_ekman_64.toml
```

Run the optional non-blocking MPI z-slab path:

```bash
mpiexec -n 2 ./build-cpp-cpu/wireles \
  --mpi-slab \
  --nx 32 --ny 32 --nz 32 \
  --steps 20 --log-every 5 \
  --dt 0.001 --nu 0.001
```

The first MPI slab path owns state updates by contiguous z slabs and exchanges
only neighboring boundary planes with non-blocking `MPI_Irecv`/`MPI_Isend`.
Momentum RHS derivatives, Smagorinsky/AMD/LASD stress divergence, fixed-Prandtl,
AMD, or LASD scalar transport, wall filtering, horizontal dealiasing, and diagnostics
are evaluated on slab-local `x-y` planes. LASD history fields and Lagrangian
velocity accumulators are exchanged with neighboring z slabs before the dynamic
coefficient update. The pressure projection follows the legacy Fortran layout:
each rank performs local `x-y` FFTs for its z slab, uses non-blocking
`MPI_Ialltoall` to transpose spectral data into y pencils, solves complete z
tridiagonal columns locally, then transposes back to the z-slab layout before
inverse FFT and velocity correction. This first distributed pressure path
requires `ny` to be divisible by the MPI rank count.  The z slabs may be
uneven; the pressure transposes use `MPI_Ialltoallv`.

The CPU MPI path also supports `momentum_advection_form = "rotational"`, the
NCAR-style vector-invariant form.  Horizontal vorticity components are formed
on the `w` faces, vertical vorticity on the `u/v` centers, and paired
center-to-face/face-to-center interpolation is used in `u x omega`.  This
placement makes the inviscid rotational term exactly energy-orthogonal on the
staggered grid and preserves domain-integrated horizontal momentum to roundoff.
Horizontal dealiasing is required.  The run log reports both properties as
`[advection-energy]` and `[advection-momentum]` audits.

For a quick compile/runtime smoke check, override the length:

```bash
./build-cpp-cpu/wireles \
  --config configs/neutral_ekman_64.toml \
  --steps 2 --log-every 1 --frame-start-step 100000 --frame-end-step -1
```

The benchmark comparison reports Moeng Table 3 values using the minimum of the
time-averaged total heat-flux profile for `zi`. It also prints
`instantaneous_zi_mean_over_zi0` as a diagnostic because the instantaneous
minimum can jump by one vertical grid cell on this coarse 48-level case.
The LASD config writes `profiles.csv` and `summary.csv` to
`outputs/largeeddy1993_lasd_diagnostics/`; generate the paper-style figures with:

```bash
python3 tools/plot_largeeddy1993.py \
  --input-dir outputs/largeeddy1993_lasd_diagnostics \
  --title "C++ LASD Moeng CBL"
```

The CUDA-MPI Moeng Smagorinsky config writes transient full-field HDF5 dumps
from step 0. Render a diagnostic mid-y x-z GIF for the vertical velocity `w`
with a diverging red-blue colormap centered at zero:

```bash
python3 tools/plot_cross_section_gif.py \
  --input-dir outputs/largeeddy1993_moeng_fields \
  --pattern 'fields_step_*.h5' \
  --component w \
  --frames-dir outputs/largeeddy1993_moeng_w_xz_png \
  --gif outputs/largeeddy1993_moeng_w_xz.gif \
  --fps 10 \
  --symmetric
```

`fig02_heat_flux.png` uses the face-based heat-flux diagnostic
`heat_flux_faces.csv`; the cell-center heat-flux columns remain in
`profiles.csv` for consistency checks. This avoids mixing the prescribed
surface flux with the first cell-center statistic near the wall.

For a quick smoke run, keep the same case config and override only runtime
length:

```bash
./build-cpp-cpu/wireles \
  --config configs/largeeddy1993_moeng.toml \
  --steps 2 --log-every 1
```

Run with the new physics modules enabled:

```bash
./build-cpp-cpu/wireles \
  --nx 32 --ny 32 --nz 32 \
  --steps 20 --log-every 5 \
  --dt 0.001 --nu 0.001 \
  --wall abl --wall-stress dynamic_neutral --zo 0.005 \
  --thermo --theta-gradient 0.01 --surface-theta-flux 0.001 \
  --scalar-diffusivity 0.001 \
  --sgs smagorinsky --smag-cs 0.16
```

The first milestone is numerical plumbing plus clean module boundaries with the
same vertical staggering used by the production Fourier + finite-difference
path: field layout, FFTW transforms, spectral derivatives, finite-difference
vertical operators, Poisson projection, wall stress, scalar transport,
staggered Smagorinsky SGS, dynamic LASD, scalar transport, and diagnostics. The
next physics milestones are:

1. Add paper-grade CBL profile and figure diagnostics behind config flags, not a
   separate case runner.
2. Implement Moeng 1984 prognostic SGS-TKE closure if exact Moeng-1984 closure
   reproduction is required. The current configs are Moeng-style fixed
   Smagorinsky or dynamic LASD, matching the existing benchmark setup rather
   than the prognostic TKE model.

# Lin & Porté-Agel (2019) wind-turbine wake benchmark

This benchmark reconstructs Case 3 from [Large-Eddy Simulation of Yawed
Wind-Turbine Wakes](https://doi.org/10.3390/en12234574), using the paper's
standard, non-rotating actuator-disk comparison before adding ADMR blade
element and tangential loading.

The paper-defined setup is a 6.4 m × 0.8 m × 0.4 m neutral wind-tunnel domain
on 128 × 64 × 32 cells. The WiRE-01 rotor diameter is 0.15 m, its hub is at
0.125 m, and the disk is 3.2 m from the inlet. The incoming boundary layer has
height 0.4 m, hub velocity 4.88 m/s, hub-height turbulence intensity 7%,
roughness length 0.022 mm, and friction velocity 0.22 m/s. The benchmark runs
the paper's 10°, 20°, and 30° yaw cases and extracts the Figure 7 hub-height
profiles at x/D = 4, 6, 8, and 10.

The present first stage applies only uniform rotor-normal thrust. It does not
yet include blade-element loading, wake rotation, tower drag, or nacelle drag.
The measured wind-tunnel thrust coefficients are converted to disk-local
coefficients with one-dimensional momentum theory. The paper does not report a
time step, so the runners use 0.00025 s from the archived WiRE wind-tunnel
setup in this repository.

The geometric rotor annulus is convolved with a normalized anisotropic
Gaussian, using the convention `exp(-(x/epsilon)^2)`. At the paper resolution,
the normal width is `epsilon_x=2 dx=0.10 m` and the transverse width is
`epsilon_r=2 max(dy,dz)=0.025 m`. The same discretely normalized kernel samples
the disk velocity and deposits the force, so total thrust is conserved under
subcell turbine translations. The filtered disk-velocity correction of
[Shapiro, Gayme & Meneveau (2019)](https://wes.copernicus.org/articles/4/291/2019/)
is evaluated from the overlap of the actual convolved annulus rather than its
small-filter approximation.

## Zero-yaw main-solver milestone

`run_nonyawed.py` exercises the new `src/jaxwind` implementation at zero yaw.
The paper itself reports 10°, 20°, and 30° cases, so zero yaw is an intentional
bring-up extension rather than a paper validation point. It uses `C_T=0.78`
and `C_T'=1.445727`, the first standard-ADM operating coefficient in the case
set.

The main and precursor domains are complete, same-layout states resident side
by side on one device. At each AB2 evaluation the main-domain fringe consumes
the precursor velocity at the same `t_n`; no field is gathered or copied
through the host. On a single GPU the dependency-free advances can be launched
from two persistent host threads with distinct PJRT execution-stream IDs. On
the RTX 3070, a 200-step 128 × 64 × 32 timing check measured 81.6 paired domain
steps/s with the stream launcher versus 67.8 domain steps/s serially.

Run the production zero-yaw case:

```bash
env PYTHONPATH=src JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  JAXWIND_SPECTRAL_FD_SOURCE="$PWD/external/bw1000_benchmark" \
  python benchmark/LinPorteAgel2019/run_nonyawed.py \
  --execution cuda-streams
```

The first completed wake run is rejected as a validation result. Its precursor
used static Smagorinsky, an incorrect wall treatment, cell-independent startup
noise, and an averaging window that included spin-up. Those choices produced
only about 1.5% hub-height streamwise intensity and contaminated every wake
profile. The current paired integrator carries independent Boussinesq/LASD
closure memory for the precursor and main domain and can start from the
validated precursor checkpoint described below.

Run a short full-grid diagnostic from that checkpoint, using separate CUDA
execution streams for the side-by-side domains:

```bash
python benchmark/LinPorteAgel2019/run_nonyawed.py \
  --precursor-restart \
    benchmark_results/LinPorteAgel2019_precursor_validated/precursor_final.npz \
  --precursor-steps 0 --concurrent-steps 2600 \
  --sample-start 0 --sample-every 20 --execution cuda-streams \
  --flow-gif --gif-start 1 --gif-every 50 --gif-frames 52 \
  --output-dir benchmark_results/LinPorteAgel2019_fringe_gaussian
```

Besides the main-domain three-plane flow animation, this writes
`fringe_three_plane.gif`, which shows `main - precursor`, and
`fringe_diagnostic.png`, which plots the smooth fringe mask above the evolving
hub-plane mismatch. This short transient is for coupling inspection, not a
wake-profile validation result. Relative to the former axial-Gaussian/radial-
tanh mask, the final-frame upper-half transverse modal energy at the turbine
plane falls from 26.8% to 0.65%; the corresponding transverse
second-difference RMS falls by 90.6%. `smoothing_comparison.png` shows the
fixed-scale old/new field, turbine-plane cut, and spectrum.

### Doubled-resolution study

The complete precursor and short-wake workflow was repeated on
`256 x 128 x 64` cells while retaining the physical domain and
`dt=0.00025 s`. This gives `dx=0.025 m` and `dy=dz=0.00625 m`; the two-cell
Gaussian widths therefore become `epsilon_x=0.05 m` and
`epsilon_r=0.0125 m`. The filtered-velocity multiplier changes from 0.9134
to 0.9543 as the physical filter narrows toward the geometric disk.

The precursor was regenerated rather than interpolated: 19,000 cold-start
steps were followed by a separate 19,000-step clean validation window. The
validated window gives a hub mean of 4.812 m/s and streamwise intensity of
7.056%. Its maximum directional CFL is 0.0698, LASD CFL is 0.698, and
projected divergence is `2.14e-5`.

Run the doubled-grid wake from that checkpoint:

```bash
python benchmark/LinPorteAgel2019/run_nonyawed.py \
  --nx 256 --ny 128 --nz 64 \
  --precursor-restart \
    benchmark_results/LinPorteAgel2019_precursor_2x_validated/precursor_final.npz \
  --precursor-steps 0 --concurrent-steps 2600 \
  --sample-start 0 --sample-every 20 --execution cuda-streams \
  --flow-gif --gif-start 1 --gif-every 50 --gif-frames 52 \
  --output-dir benchmark_results/LinPorteAgel2019_fringe_gaussian_2x
```

The paired run completes 0.65 s of physical time in 267 s on the RTX 3070.
`resolution_comparison.png` compares the turbine-plane field and spectrum,
while `wake_resolution_comparison.png` overlays velocity-deficit and added-TKE
profiles at all four downstream stations. These wake profiles still describe
a short transient and are not a statistically converged resolution study.

Record a three-plane instantaneous-flow GIF after wake development:

```bash
python benchmark/LinPorteAgel2019/run_nonyawed.py \
  --execution cuda-streams --concurrent-steps 14000 \
  --flow-gif --gif-start 9000 --gif-every 50 --gif-frames 100
```

`flow_three_plane.gif` shows the hub-height x-y plane, turbine-centre x-z
plane, and the y-z plane at x/D=6. All panels use instantaneous `u/u_h`; white
arrows show the corresponding in-plane velocity components. The production
visualisation spans paired times 2.250--3.488 s at 12 frames/s.

## Standalone precursor validation

`run_precursor.py` isolates the periodic, pressure-driven neutral boundary
layer: it contains no turbine, actuator force, or fringe. The paper's
Lagrangian scale-dependent dynamic SGS closure is retained by advancing a
passive neutral scalar product state. The lower wall uses the legacy JAX-Wind
sharp two-dimensional filter (`fgr=1.5`, `tfr=2`) before evaluating the local
log-law stress, together with the Porté-Agel first-interior-face shear
correction. Correlated startup perturbations replace grid-cell white noise.

Run a cold precursor and capture its developed final interval:

```bash
env PYTHONPATH=src JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  JAXWIND_SPECTRAL_FD_SOURCE="$PWD/external/bw1000_benchmark" \
  python benchmark/LinPorteAgel2019/run_precursor.py --flow-gif \
  --output-dir benchmark_results/LinPorteAgel2019_precursor
```

Continue a developed checkpoint for a clean stationary average:

```bash
python benchmark/LinPorteAgel2019/run_precursor.py \
  --restart benchmark_results/LinPorteAgel2019_precursor/precursor_final.npz \
  --steps 19000 --sample-start 0 --flow-gif \
  --output-dir benchmark_results/LinPorteAgel2019_precursor_validated
```

The validated 19,000-step window, after 19,000 cold-start steps, gives a hub
mean of 4.775 m/s and streamwise intensity of 7.06%, compared with the paper's
4.88 m/s and 7%. Digitising the paper's Figure 3 LES curves over
`0.1 <= z/D <= 2` gives RMSE values of 0.0243 for `mean(u)/u_h` and 0.0041 for
`I_u`; at hub height the paper and this run are `(0.9777, 0.0704)` and
`(0.9784, 0.0706)`, respectively. Maximum directional CFL is 0.0312, LASD CFL
is 0.312, and projected divergence is `9.00e-6`. Only 2.34% and 0.35% of the
hub-plane fluctuation energy lie in the upper half of the resolved x and y
wavenumbers.

Outputs include `precursor_profiles.csv`, `precursor_profiles.png`,
`history.csv`, the restart-complete `precursor_final.npz`, spectra/statistics,
and `precursor_three_plane.gif`. TKE is the resolved temporal variance; it
does not include the diagnostic SGS contribution.

## Yawed legacy runner

Run all paper yaw angles:

```bash
env JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python benchmark/LinPorteAgel2019/run.py
```

Run the end-to-end plumbing check:

```bash
python benchmark/LinPorteAgel2019/run.py --quick --yaw 20 \
  --output-dir /tmp/lin-porte-agel-smoke
```

Each yaw directory contains `profiles.csv`, a Figure 7-style `profiles.png`,
`summary.json`, and the time-mean and RMS streamwise fields in
`mean_fields.npz`.

# Directional Horns Rev runs with one shared precursor

## One-time reference and per-direction main runs

The production geometry is 512 x 512 x 256 cells in 8192 x 8192 x 1024 m
(16 x 16 x 4 m), with all 80 V80 turbines at 70 m hub height. MUSCL-MC is now
the default momentum choice on uniform grids; generated cases select it explicitly.
Mapped case configurations retain `central` when unspecified, because MUSCL
currently supports uniform Cartesian grids only. Explicit MUSCL on a mapped
grid is rejected. The old centered farm baseline is explicitly
pinned to `central` for reproducibility.

The reference is a turbine-free, horizontally periodic neutral offshore LES:

1. Warmup: 10 h, adaptive CFL=0.9 with a 6 s timestep cap (6000 × 6 s schedule).
2. Precursor recording: 1 h / 14400 steps, from the warmup checkpoint.
3. Each independent directional main: 1 h / 14400 steps, replaying that same
   recording, with GMG, nonperiodic x/y and lateral/downstream pressure outlets.

Precursor and main use fixed dt=0.25 s. All stages checkpoint every simulated
hour using `checkpoint_every_seconds = 3600`, independent of adaptive step counts.
Each schedules 100 frames. Warmup/precursor periodicity
is intentional; there is no periodic coupling of the main solution. The initial
main state is the warmup checkpoint (the start of the recorded sequence), not
the precursor's end state. The turbine controller uses the existing 1D-upstream
lookup for RPM and reported power.

The 10 h schedule is not an automatic guarantee of developed turbulence.
Inspect the warmup's final profiles, stress, and stationarity before accepting
the reference for production comparisons. No full-duration run was launched
while implementing this feature.

## Commands

Run from the repository root in the project environment on a GPU compute node:

```bash
export JAXWIND_V80_FAST="$PWD/cases/HornsRev1/turbines/V80/CustomRotor.fst"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Construct the selected 270-degree case and shared reference declarations.
python tools/create_hornsrev_case.py --wind-direction 270
python -m jaxwind check cases/HornsRev1/directions/reference/workflow.toml
python -m jaxwind check cases/HornsRev1/directions/wd270p000/main.toml

# Run ONLY ONCE, or continue it with --resume after allocation expiry.
python -m jaxwind workflow cases/HornsRev1/directions/reference/workflow.toml

# Run the selected wind-farm direction after the reference finishes.
python -m jaxwind workflow cases/HornsRev1/directions/wd270p000/workflow.toml

# Another direction: construct and run ONLY its main stage.
python tools/create_hornsrev_case.py --wind-direction 30
python -m jaxwind workflow cases/HornsRev1/directions/wd030p000/workflow.toml
```

Use `--resume` with the same workflow command to continue an interrupted stage;
completed stages are skipped. The main-only workflows bind to the same external
reference artifacts and cannot accidentally execute another warmup/precursor.
Keep that reference output directory unchanged while running/comparing sectors.

Generation is idempotent for identical files and refuses to overwrite changed
files. Use `--directory` and `--run-root` for an independent experiment. A
`--wind-speed 8` override selects an explicit mean speed at 70 m, useful for
isolating direction effects. With no override, the nearest circular wind-rose
sector supplies the mean; `direction.json` records which sector was selected.

For example, render a completed main run using:

```bash
python tools/render_hornsrev1_farm.py outputs/hornsrev1_directional/wd270p000/main
```

Only a small test is launched by running the explicitly generated `--smoke`
workflows: 4 steps per stage, one turbine, 32 x 24 x 64 cells. This tests the
workflow plumbing, not the full farm or its turbulence.

## Scaling and its limitations

The reference log law targets the supplied 270-degree sector mean of
9.691004386 m/s at an assumed 70 m reference height, with z0=0.0002 m,
kappa=0.4, and u*=0.3036578697 m/s. Pressure-gradient acceleration is u*^2/Lz.
Small 0.1 m/s initial perturbations seed development; they are not measured TI.

Recordings include the actual time-weighted mean streamwise velocity profile.
For each direction, the replay scale is computed as:

```text
s = target sector mean at 70 m / recorded mean at 70 m
(u, v, w)_inlet = s * (u, v, w)_recorded
(u, v, w)_initial = s * (u, v, w)_warmup
```

Main initialization is projected onto the open-domain constraints. Initial
pressure/tendency/controller state is rebuilt; no old periodic pressure or
rotor state is copied. Scalar values and recording timestamps are unchanged.
The entire 1 h recording is used once: no looping, clipping or missing coverage.
The recorded file is not rewritten or duplicated for another direction.

This preserves fluctuation-to-mean ratios (TI), normalized profile and the
recorded temporal pattern. It is **amplitude scaling**, not an exact dynamically
similar LES: exact velocity/time scaling would also change the time axis by
1/s, require resampling, and potentially require a longer reference. It assumes
common neutral stability, roughness and turbulence statistics across directions;
the wind rose alone does not justify directional stability/TI differences.
Main CFL is checked during advancement and fails explicitly above 1.

The recording stores four float32 yz fields each step: approximately 28.1 GiB
uncompressed for 1 h on the production grid (compressed size depends on flow).
Allow additional space for checkpoints and per-direction output. Checkpoint
compression/output is CPU work; simulation advancement runs on the GPU.

## Direction and wind-rose conventions

`wind_rose.csv` preserves the supplied 12 rows, including their labels and
frequencies. Frequencies total 100.2%; they are not silently normalized. No
frequency weighting or Weibull sampling is applied to these single-speed runs.
Reference-height interpretation at 70 m is an assumption, not supplied metadata.

Angles are meteorological FROM directions clockwise from north. The original
farm coordinates are East/North. For phi=270 degrees minus wind direction:

```text
x_sim = 4096 + cos(phi)*(E-4096) + sin(phi)*(N-4096)
y_sim = 4096 - sin(phi)*(E-4096) + cos(phi)*(N-4096)
```

Thus 270 degrees leaves the original layout unchanged; 0 degrees gives
x_sim=8192-N, y_sim=E. All pairwise distances, turbine IDs and the farm center
are preserved. The solver always uses +x inflow; movie axes are the rotated
simulation coordinates, not fixed geographic East/North.

## Implementation verification

The direction/scaling and MUSCL tests passed (33 tests). Existing uniform-farm
tests passed (4 tests), and six relevant legacy precursor tests passed. Two
unrelated legacy tests remain failing: the HITSZ example now declares 180 s
where its test expects 90 s, and `runtime/periodic.py` lacks a `dataclass` import.
Those unrelated files were not changed to mask these failures.

A GPU smoke reference completed once (4 warmup + 4 recording steps), then two
main-only smoke cases used the same four recorded samples. The 270-degree main
was paused after two steps and resumed successfully; final CFL was 0.202. The
0-degree-speed main used scale 0.790672 and completed with CFL 0.158. Both use
nonperiodic x/y face shapes. The smoke geometry contains one turbine, so full
farm rotations are checked separately by all-pair-distance and centroid tests
for all 12 rose directions and an intermediate angle.

## ROCm Slurm host memory

The ROCm launcher requests 128 GiB of host RAM per node by default. Override
with `MEMORY=192G` (or `SBATCH_MEM_PER_NODE` when `MEMORY` is unset). Device
memory and Slurm host-memory limits are separate: a `slurmstepd` cgroup
`oom_kill` can terminate initialization even when GPU memory is available.
The request is headroom for initialization/checkpoint processing, not a
measured guarantee for every backend or configuration.

```bash
MEMORY=128G WALLTIME=08:00:00 bash tools/submit_hornsrev_rocm.sh prepare \
  --output outputs/hornsrev_prepare_rocm --resume
```

Use the original output directory when resuming. If initialization was killed
before the first checkpoint, resume starts that unfinished stage from its
initial condition. Inspect the failed job on the ROCm cluster with:

```bash
sacct -j JOBID --format=JobID,State,ReqMem,MaxRSS,AllocCPUS,Elapsed
```

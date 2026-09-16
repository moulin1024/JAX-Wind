# Horns Rev runner on ROCm

Run from the repository root on the ROCm cluster. The existing cluster convention
is partition `hx1hdnormal01`, one `dcu`, DTK 26.04 and conda environment `jax060`.
The launcher refuses a missing partition; it does not fall back to NVIDIA or CPU.
Install the repository's dependencies in that environment, including the renderer
dependencies and FFmpeg. Case declarations are generated on the execution host.

## Submit prepare, then all directions

```bash
prepare_job=$(bash tools/submit_hornsrev_rocm.sh prepare --output outputs/hr_prepare)
prepare_job=${prepare_job%%;*}
DEPENDENCY="afterok:$prepare_job" bash tools/submit_hornsrev_rocm.sh windrose \
  --prepare-run outputs/hr_prepare --output outputs/hr_windrose
```

Prepare runs only 20 h warmup and 2 h precursor, with the 270-degree sector's
9.691004386 m/s hub-speed target. Windrose submits a 12-task array, one 2 h main
per 30-degree sector, including 270 degrees. Default concurrency is one GPU;
set `ARRAY_CONCURRENCY=2` etc. to run more directions simultaneously.
Set `WALLTIME=48:00:00`, `PARTITION=...`, `CONDA_ENV=...` as appropriate for the
target cluster. Walltime is a queue limit, not a throughput estimate.

## Submit just one direction

```bash
bash tools/submit_hornsrev_rocm.sh direction --prepare-run outputs/hr_prepare \
  --output outputs/hr_270 --wind-direction 270
```

Use `DEPENDENCY=afterok:JOBID` if prepare is still queued/running. A prepare-run
path is mandatory; it must identify this runner's prepare output. Main refuses
unfinished or smoke/production-mismatched prepare artifacts. Use distinct output
roots, not paths inside the prepare output. Never change shared prepare artifacts
while main jobs use them. Metadata digests detect changed prepare provenance on
resume; large checkpoint and inflow arrays are not individually hashed.

## Local configuration, smoke and resume

```bash
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python tools/run_hornsrev.py prepare --output outputs/hr_prepare --configure-only
python tools/run_hornsrev.py direction --prepare-run outputs/hr_prepare \
  --output outputs/hr_270 --wind-direction 270 --configure-only
```

Configuration-only performs no simulation and can reference a configured pending
prepare. Add `--smoke` to **both** prepare and main for tiny one-turbine, four-step
tests. The normal runner defaults to ROCm; `--backend cuda` or `--backend cpu`
is available for local tests only. Slurm forces ROCm. Without `--sector-index`,
local windrose mode executes all sectors sequentially; Slurm supplies that index.

Resume with the same arguments plus `--resume`. After a prepare walltime limit,
resume prepare and depend on the new job ID. `--max-steps N` deliberately pauses
work and exits with code 3 so `afterok` will not launch a main on incomplete data.
Outputs include `runner.json`, generated `cases/`, simulation `run/`, and main
MP4 rendering (disable with `--no-render`). Slurm logs are under `logs/hornsrev/`.

Production uses 512 x 512 x 256 cells, 8192 x 8192 x 1024 m, MUSCL-MC, 80 V80s,
open main boundaries and GMG, with 100 main frames. Meteorological wind direction
rotates the farm; the computational inflow always travels in +x. Each main scales
all three recorded velocity components and initial velocities by its sector's
mean speed divided by the measured reference hub speed. Recording cadence is
unchanged: this is neutral-ABL amplitude reuse, not exact time-rescaled similarity.
The windrose uses one mean speed per sector, not a Weibull speed-bin/AEP ensemble.
See [directional workflow](../cases/HornsRev1/directional_workflow.md) for details.

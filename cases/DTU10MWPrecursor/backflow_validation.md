# Submit the backflow correction and log-law check

From the repository root:

```bash
bash tools/submit_backflow_validation.sh
```

The wrapper prepares resolved cases, records source/input hashes, and submits one
A100 job to `gpu1` with account `rzg_gpu`, 8 CPUs, 24 GB host memory, and a two-hour
wall-time limit. It prints the output directory and Slurm job ID. No full Horns
Rev farm is used. To inspect the command without submitting:

```bash
bash tools/submit_backflow_validation.sh --dry-run --output outputs/backflow_check
```

Submit the same command without `--dry-run` to run that prepared directory.
An output directory can be reused only before execution and while its inputs and
source files are unchanged. Changes require a fresh output directory. The archive
`source_snapshot.tar.gz` and `source_manifest.json` record the implementation used;
the large input checkpoint is hashed but is not copied into the archive.

## What runs

1. Load the developed periodic DTU checkpoint at 39,600 s from
   `outputs/dtu10mw_central_cfl09_20260916/run/precursor/checkpoint.npz`.
   Its grid is 64 × 32 × 128 over 4096 × 1024 × 1024 m.
2. Advance three turbine-free domains synchronously: the central periodic
   precursor, a central open-domain baseline, and an open domain using
   `central-open-upwind` plus the high-x energy backflow pressure condition.
   The corrected scheme uses MUSCL-MC horizontally and central vertical fluxes.
   The two open branches receive the same turbulent precursor plane at every
   step; lateral boundaries are periodic. All three retain pressure forcing,
   full RK3, AMD, and the Porté-Agel wall-gradient correction. Open-domain pressure
   solves use GMG; the periodic pressure solve uses FFT.
3. Discard 1800 s of spin-up, then average 3600 s, sampling every 10 s. The common
   fixed timestep is 0.5 s. **CFL 0.9 is a hard ceiling checked every step**, not
   an adaptive target. Exceeding it aborts the run; reduce `--dt` for a fresh run.
4. Run matched central and corrected **single-V80** uniform-inflow cases for
   300 s each (128 × 64 × 64 cells). Compare upstream oscillations, divergence,
   inlet error, and boundary histories, and render wake movies.

The log-law experiment explicitly tests the open-domain profile, which was not
established by the earlier bitwise periodic-operator comparison. Both open
branches retain pressure forcing because this is a pressure-driven neutral ABL
regression experiment. It does not validate all wind-farm forcing configurations.

## Reading the result

Under the printed output directory:

- `analysis/README.md`: verdict, mean-profile errors, and links.
- `analysis/log_law_comparison.png` and `.pdf`: mean profile, log-law error,
  and corrected-minus-central profiles in four streamwise quarters.
- `analysis/profiles.csv`: time- and volume-averaged velocity profiles.
- `analysis/report.json`: numerical metrics, thresholds, and individual checks.
- `loglaw/samples.npz`: regularly sampled spanwise means at every x-z cell,
  wall stress, resolved stress, divergence, and maximum per-step CFL.
- `loglaw/{precursor,central,corrected}_checkpoint.npz`: fields saved every
  600 s and at completion for inspection or explicit reuse. This runner does
  not automatically resume interrupted experiments.
- `wake_corrected/backflow_analysis/`: boundary and baseline-comparison plots.
- `wake_corrected/v80_single_hub_u.mp4`: corrected wake movie.
- `status.json` and `loglaw.log`: execution state and progress.

The report averages cell-centered velocity over equal-volume horizontal cells
and equally spaced time samples, and compares it with the vertically integrated
log-law reference. The default surface-layer comparison is 20–100 m, using
u* = sqrt(a H) = 0.4 m/s, z0 = 0.001 m, and kappa = 0.4.

`PASS` requires, in each streamwise quarter:

- Corrected log-law RMSE ≤ 1.0 m/s.
- Corrected-minus-central profile RMSE ≤ 0.2 m/s.
- Increase in log-law RMSE over central ≤ 0.1 m/s.

It also requires the profile change between the two halves of the averaging
window to be ≤ 0.2 m/s for every branch, maximum sampled divergence ≤ 1e-5 /s,
and per-step CFL ≤ 0.9. If wake checks run, upstream second-difference RMS must
fall by at least 90%, with divergence and inlet error ≤ 1e-5 in their respective
units. These are declared engineering regression thresholds, not proof of
statistical convergence. The previous periodic central profile had RMSE about
0.708 m/s; horizontal upwinding can still change turbulent transport and wake
recovery. Read the failed checks before interpreting a `FAIL` as a single cause.

A completed failing screen exits with code 2 and retains all reports. An execution
failure records `failed` in `status.json`; inspect the stage log. Slurm timeout or
node failure can leave status at `running`, so also check Slurm job state.

## Controls and a tiny execution test

```bash
# Six-second synthetic test of submission, advancement, averaging, and plotting.
# Its verdict is always SMOKE_ONLY, never evidence that the log law is retained.
PARTITION=gpudev WALLTIME=00:10:00 MEMORY=8G CPUS=4 \
  bash tools/submit_backflow_validation.sh --smoke

# Run just the turbulent log-law comparison, with longer averaging.
WALLTIME=04:00:00 bash tools/submit_backflow_validation.sh \
  --skip-wake --average-seconds 7200 --output outputs/backflow_loglaw_long

# Override the developed checkpoint and matching physical/grid configuration.
bash tools/submit_backflow_validation.sh \
  --checkpoint /path/to/periodic/checkpoint.npz \
  --reference-case /path/to/matching_neutral_case.toml

bash tools/submit_backflow_validation.sh --help
squeue -u "$USER"
```

Other environment overrides are `ACCOUNT`, `GRES`, `BACKEND`, `JAXWIND_PYTHON`,
`ENV_SETUP`, `CUDA_MODULE`, and `DEPENDENCY`. By default the batch job loads
`cuda/13.0` and uses `$HOME/venvs/numba_cuda_waterboa/bin/python`. Set
`ENV_SETUP=/path/to/environment.sh` to supply another environment. The Slurm log
is `logs/backflow_validation/backflow-loglaw-JOBID.out`.

## Completed interactive validation, 2026-09-16

The full log-law-only experiment completed on the A100 in allocation 30270853:
1800 s spin-up plus 3600 s averaging, with 360 averaged samples. The numerical
run completed, but the declared preservation screen **failed**. Log-law RMSE was
0.6390 m/s for the periodic precursor, 0.6408 m/s for central open flow, and
0.7297 m/s for corrected open flow. The last two streamwise quarters increased
the RMSE by 0.1166 and 0.1304 m/s, exceeding the 0.1 m/s limit. All other declared
checks passed. Maximum CFL was 0.32055 and sampled divergence was 4.00e-8 /s.

The wall-stress diagnostic also changed substantially: u* was 0.4317 m/s for
central open flow and 0.3385 m/s for corrected open flow (forcing reference
0.4 m/s). The plotted first-cell velocity drops strongly with the correction;
the 20–100 m RMSE alone understates this near-wall difference. Thus retaining
central vertical fluxes and an unchanged periodic precursor is insufficient to
claim preservation of the turbulent open-domain wall layer. Wake runs were
skipped in this experiment.

[Full report](../../outputs/backflow_loglaw_interactive_30270853/analysis/README.md)
· [Profile plot](../../outputs/backflow_loglaw_interactive_30270853/analysis/log_law_comparison.png)
· [Metrics](../../outputs/backflow_loglaw_interactive_30270853/analysis/report.json)

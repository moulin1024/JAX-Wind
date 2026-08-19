# DTU 10-MW strict-inlet AD-BEM benchmark

[`benchmark_adbem.toml`](benchmark_adbem.toml) is the reproducible benchmark
configuration for the validated DTU 10-MW wake case. It defines one complete
workflow:

| Stage | Physical time | Steps | Driving and turbine configuration |
| --- | ---: | ---: | --- |
| Warmup | 10 h | 360,000 | pressure-driven neutral LASD |
| Precursor | 1 h | 36,000 | pressure-driven LASD; 11-plane HDF5 slabs every 10 steps |
| Main | 1 h | 36,000 | strict CUDA-Fortran inlet; no pressure gradient or fringe; DTU 10-MW AD-BEM |

The turbine is fixed at `x = 1000 m`, uses the OpenFAST rotor geometry and
polars, and has a prescribed operating point of `9.6 RPM` and `0 degrees`
pitch. It includes the legacy element-size AD-BEM smearing, tower, and nacelle.
This benchmark deliberately prescribes speed; it does not exercise a turbine
controller.

Set the path to the DTU 10-MW OpenFAST deck, then inspect or run the workflow:

```bash
export JAXWIND_DTU10MW_FAST=/path/to/DTU_10MW.fst
python -m applications.windfarm_precursor.benchmark --dry-run
python -m applications.windfarm_precursor.benchmark
```

An interrupted warmup resumes from `checkpoint_latest.npz`. A completed
warmup is reused. Pass `--overwrite` to restart the warmup and replace the
precursor/main result deliberately.

The benchmark automatically measures rotor-area incoming turbulence intensity
from the HDF5 precursor, plots the hub-height centerline deficit against the
TI-consistent Gaussian model, and writes `benchmark_validation.json`. It passes
when the Gaussian deficit RMSE over `4D`--`10D` is at most `0.03` and the fitted
wake-expansion rate is within 15% of the TI prediction. The establishing run
gave `I_u = 0.06523`, `k_TI = 0.02879`, `k_fit = 0.02850`, and RMSE `0.01710`.

With uncompressed output, expect the 1 h precursor recording to occupy about
20 GB at this resolution.

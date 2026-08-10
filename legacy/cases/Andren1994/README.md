# Andrén et al. (1994) neutral Ekman benchmark

This directory is a pure configuration of the shared `abl` warmup runner. It
reproduces the neutral, rigid-lid Ekman LES intercomparison case using LASD;
there is no benchmark-owned simulation script or alternative SGS model.

Validate the fully resolved configuration without importing JAX:

```bash
jaxwind benchmark/Andren1994/config.toml --dry-run
```

Run the canonical `40 × 40 × 40`, `tf = 10` (`27.7778 h`) case from the
repository root:

```bash
export JAXWIND_SPECTRAL_FD_SOURCE="$PWD/external/bw1000_benchmark"
jaxwind benchmark/Andren1994/config.toml
```

The complete case specification is [`config.toml`](config.toml). It declares:

- the `4000 × 2000 × 1500 m`, `40³` domain and `0.8 s` AB2 step;
- the complete 45°N rotation vector (`f_h = f = 10⁻⁴ s⁻¹`) and `10 m/s`
  geostrophic wind;
- the tabulated Table A.1 velocity/TKE initialization;
- neutral log-wall stress with `z₀ = 0.1 m`;
- LASD momentum and passive-scalar closures with five-step updates;
- the prescribed `10⁻³ kg m⁻² s⁻¹` passive-scalar surface flux; and
- averaging over the final `3/f` interval.

The generic runner writes `checkpoint_latest.npz`, `history.csv`,
`statistics_latest.npz`, `profiles.csv`, `normalized_profiles.csv`,
`spectra.csv`, `summary.json`, `resolved_config.toml`, and
`warmup_manifest.json`. A completed canonical run also writes
`checkpoint_final.npz`.

For a continuation, create another TOML case and set its output and execution
tables explicitly:

```toml
[output]
directory = "benchmark/Andren1994/results/lasd_continued"

[execution]
restart_checkpoint = "benchmark/Andren1994/results/lasd_40x40x40/checkpoint_latest.npz"
overwrite = false
```

To collect the complete scalar-flux budget for paper Fig. 13, copy the TOML,
set `benchmark.fig13_budget = true`, extend `time.duration_hours` beyond the
checkpoint time, configure `execution.restart_checkpoint`, and launch that
TOML file. Budget sampling writes restartable `fig13_budget_samples.npz` and
`fig13_budget_profiles.csv`.

The LASD diagnostic reports resolved, diagnostic-SGS, and total velocity and
scalar statistics separately. Diagnostic SGS energy and scalar variance do not
feed back into the resolved trajectory. The checkpoint retains velocity,
passive scalar, both AB2 histories, and all LASD closure memory.

Reference data and the compact public specification live under `reference/`.
To overlay a completed run on a locally downloaded copy of the original paper:

```bash
python benchmark/Andren1994/overlay_paper_figures.py \
  --paper-pdf /path/to/qj-1457-1994.pdf \
  --results benchmark/Andren1994/results/lasd_40x40x40
```

The paper PDF is not redistributed by this repository.

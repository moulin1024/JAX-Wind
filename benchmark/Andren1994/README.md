# Andrén et al. (1994) neutral Ekman benchmark

The complete executable definition is [case.toml](case.toml). This directory
contains no solver runner: the repository-wide `jaxwind-run` application loads
the same schema used by Nieuwstadt and GABLS1.

The configured `40 × 40 × 40` case uses full MAC projection,
GMG-preconditioned PCG, MP5 stabilization, IMEX-ARK3, and multilevel LASD. Its
two Germano scales share the first two pressure-GMG levels.

Canonical CUDA run:

```bash
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
python -m jaxwind benchmark/Andren1994/case.toml
```

Short smoke run:

```bash
python -m jaxwind benchmark/Andren1994/case.toml --quick \
  --output-dir benchmark_results/andren1994_config_smoke
```

Configuration changes use typed TOML overrides, for example:

```bash
python -m jaxwind benchmark/Andren1994/case.toml \
  --set 'numerics.dtype="float64"' \
  --set time.end=1000.0
```

The canonical averaging window is encoded directly as `70000 <= t <= 100000`
seconds, equivalent to `7 <= ft <= 10` for `f = 1e-4 s^-1`. Initial mean wind
and TKE tables are also data inside the TOML file rather than Python constants.

Every run writes `resolved_config.json`, `checkpoint.npz`, `history.csv`,
`profiles.csv`, and `summary.json`. Paper reference material and overlay tools
remain offline consumers of those results.

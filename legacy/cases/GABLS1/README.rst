GABLS1 stable-boundary-layer benchmark
======================================

The canonical case is a 32 x 32 x 32, 400 m cubed GABLS1 domain integrated
for nine hours with statistics over hours 8--9.  It uses the current semantic
JAX-Wind solver: conservative momentum and potential-temperature transport,
horizontal three-halves padding, LASD momentum/scalar SGS closure, linear
Boussinesq buoyancy, spectral/FD pressure projection, and a
plane-mean coupled
Businger--Dyer Monin--Obukhov surface law with a 0.25 K/h cooling rate.

Run on one GPU from the repository root::

  CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda PYTHONPATH=src \
  python -m jaxwind benchmark/GABLS1/config.toml

Validate the resolved configuration without importing JAX::

  PYTHONPATH=src python -m jaxwind benchmark/GABLS1/config.toml --dry-run

Short smoke runs use a separate TOML file with their duration and output
directory declared in the configuration.  The CLI does not override them.

The runner writes restartable checkpoints and statistics, ``profiles.csv``,
``flux_profiles.csv``, ``time_series.csv``, ``summary.json``, the resolved
configuration, and ``gabls1_profiles.png``.

Official reference data
-----------------------

The raw GABLS1 participant submissions used by the literature comparison are
included under ``reference/official_12p5m`` and ``reference/official_6p25m``.
Each directory contains a ``SOURCE.json`` recording the original Met Office
archive URL, SHA-256 checksum, file count, and Beare et al. (2006) citation.

Regenerate a run comparison directly from the participant ``A9`` and ``C9``
records (the 8--9 h means)::

  python -m benchmark.GABLS1.compare_reference gabls1_lasd_32cubed

This writes a six-panel PNG, the interpolated raw ensemble values as two CSV
files, and a JSON summary into the result directory.

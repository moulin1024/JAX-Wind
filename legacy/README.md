# Legacy Code

This directory contains implementations and generated outputs that are no
longer part of the active top-level JAX WiRE-LES project.

- `cpp/`: former top-level C++ solver, including its CMake build, configs,
  tests, benchmarks, and postprocessing tools.
- `fortran_cuda/`: original CUDA Fortran solver, Python `wl` workflow, cases,
  and related scripts.
- `artifacts/`: generated plots, GIFs, CSV diagnostics, caches, and old local
  build directories.

The maintained project entry points are now the top-level `README.md` and
`jax/` directory.

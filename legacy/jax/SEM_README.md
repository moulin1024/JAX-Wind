# P=4 Spectral-Element ABL Prototype

This is a separate redesign from `wireles_jax`: a fourth-order hexahedral
spectral-element prototype for a uniform Cartesian atmospheric-boundary-layer
solver.

Run the commands below from the repository's `experimental/` directory.

Current scope:

- polynomial order `P=4` only
- GLL nodes and quadrature weights
- continuous nodal Cartesian mesh with shared element-boundary nodes
- periodic horizontal `x/y` topology and wall-normal `z` boundaries
- strong-form SEM derivatives assembled by averaging shared-node element
  contributions
- weak-form divergence diagnostics and pressure projection
- single process only
- classic Smagorinsky SGS model
- pressure-gradient body forcing computed from the target `u_fric` balance,
  with `--pressure-force` available only as an explicit override
- roughness length controlled by `--zo`, defaulting to `5.0e-3`
- bottom log-law wall-stress tendency applied to the first dynamic GLL node
- pressure correction constrained to preserve the wall-normal velocity boundary
  condition
- pressure projection uses the mass-normalized weak SEM divergence that matches
  the separable fast-diagonalization operator
- tensor-product fast diagonalization is the default pressure solver for the
  uniform Cartesian single-process path
- matrix-free SPD `G^T M G` and low-order refined GMG code are kept as
  experimental fallback paths, but are not on the default pressure path
- no turbine model
- no sharding / distributed mesh

Smoke test:

```bash
python legacy/jax/run_sem.py --nelx 2 --nely 2 --nelz 2 --steps 3
```

The runner uses double precision by default. Use `--single` only for quick
smoke tests. If you use the Python API directly, enable `jax_enable_x64` before
importing `sem_jax`.

```bash
python legacy/jax/run_sem.py --nelx 16 --nely 16 --nelz 16 --steps 1
python legacy/jax/run_sem.py --single --nelx 16 --nely 16 --nelz 16 --steps 1
python legacy/jax/run_sem.py --nelx 16 --nely 16 --nelz 16 --steps 1000 --log-every 200 --u-fric 0.4 --zo 5e-3
```

The printed `div_max` is the mass-normalized weak divergence residual used by
the default fast-diagonalization projection, not a pointwise finite-difference
divergence.

This is a design prototype. The default pressure path currently uses fast
diagonalization because the single-process uniform Cartesian case is separable.
The experimental LOR/GMG implementation remains available for later work on
nonseparable or distributed preconditioning.

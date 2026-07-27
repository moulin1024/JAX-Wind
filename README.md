# WiRE-LES reconstruction

WiRE-LES is a document-first reconstruction with an active, law-tested core
under `src/wireles`. New solver code is admitted only after its semantic
objects, morphisms, laws, effects, and acceptance tests are specified under
[`doc/design/`](doc/design/README.md).

The previous JAX implementation is frozen under
[`legacy/jax/`](legacy/jax/README.md). The C++ and Fortran/CUDA implementations
remain under `legacy/` as implementation evidence and regression references;
they are not normative specifications for the reconstruction.

The design starts from three commitments:

1. The pure mathematical model is independent of JAX, storage layout, and
   process topology.
2. Composition and law preservation are the primary API criteria; categorical
   language is used only where it yields a concrete interface or executable
   law.
3. I/O, configuration, randomness, logging, checkpointing, and distributed
   communication are explicit effects or interpretations around a pure core.

Read the design documents in order. Open decisions are deliberately recorded
instead of being hidden in placeholder code.

The first implemented vertical slice contains:

- array-independent grid, location, mesh, ownership, and field types;
- the accepted equal z-slab `Cell` and `ZFace` ownership mapping;
- an independent bounded JAX reference interpretation;
- local and JAX-native z-slab interpretations of the compatible three-dimensional
  gradient/divergence complex;
- transient packed `ppermute` halo contexts with one/two/four-device commuting
  tests;
- one higher-order projection program shared by the reference and production
  interpretations;
- an owned-cell adapter for `spectral-fd` transpose, exact SPIKE, and adaptive
  SPIKE pressure solves;
- a fixed-step AB2 higher-order integrator with explicit Euler startup,
  accepted-time diagnostics, one terminal projection, and persistent tendency
  history;
- accepted-boundary reference and per-rank z-slab checkpoints with exact
  continuation tests, including AB2 tendency history and complete LASD
  coefficient/contraction/trajectory memory;
- the first real dry-flow vector field: conservative two-thirds-truncated advection,
  constant kinematic pressure-gradient driving, a local neutral log-law wall,
  static Smagorinsky SGS stress divergence, and tagged
  Coriolis--geostrophic rotation with optional non-traditional horizontal
  Coriolis component;
- a shared velocity/gradient bundle, packed velocity and stress halos, and
  term-by-term reference versus one/two/four-slab commuting tests;
- momentum and passive-scalar Lagrangian scale-dependent dynamic closures,
  prescribed conservative scalar wall flux, and separately labeled diagnostic
  SGS energy/scalar variance.

Run the active core tests with:

```bash
pytest -q
```

Run the true multi-process CPU projection gate with one local CPU device and
one owned z-slab per process:

```bash
python tools/run_distributed_projection_cpu.py --processes 4
```

The runner discovers a sibling `../bw1000_benchmark` checkout by default. Set
`WIRELES_SPECTRAL_FD_SOURCE` when the editable `spectral-fd` source lives
elsewhere. The complete automated 1/2/4-process gate is opt-in because it binds
loopback coordinator sockets:

```bash
WIRELES_RUN_MULTIPROCESS_CPU_TESTS=1 \
  pytest -q tests/interpreters/test_projection_multiprocess_cpu.py
```

Run six complete AB2 steps, including a per-rank checkpoint/restart comparison:

```bash
python tools/run_distributed_ab2_cpu.py \
  --processes 4 --dtype float32 --method spike --steps 6
```

Use the real dry-flow vector field instead of the manufactured AB2 forcing:

```bash
python tools/run_distributed_ab2_cpu.py \
  --processes 2 --dtype float32 --method spike --steps 4 \
  --vector-field dry
```

Run the first SI-configured, nondimensional-float32 static-Smagorinsky neutral
Ekman development benchmark:

```bash
python benchmark/NeutralEkman/run.py \
  --nx 16 --ny 16 --nz 32 --dt 1.0 --hours 1.0 \
  --dtype float32 \
  --output benchmark/NeutralEkman/results/static_smag_16x16x32_1h
```

The case writes averaged profiles, an Ekman hodograph, history, summary, and an
accepted-boundary checkpoint. See
[`benchmark/NeutralEkman/README.md`](benchmark/NeutralEkman/README.md) for the
canonical 64³ command and the distinction between development and stationary
validation runs.

Run the paper-matched Andrén et al. (1994) neutral Ekman intercomparison case:

```bash
python benchmark/Andren1994/run.py
```

It uses the published `40³`, `4000 × 2000 × 1500 m`, 45°N configuration and
Table A.1 initial profiles, then writes paper-normalized profiles and a
multi-code-envelope comparison. See
[`benchmark/Andren1994/README.md`](benchmark/Andren1994/README.md); use
`--quick` for an eight-step smoke run.

Run momentum/scalar LASD as an external fifth SGS closure family, including
the passive scalar and restart-continuous diagnostic history:

```bash
python benchmark/Andren1994/run_lasd.py
python benchmark/Andren1994/overlay_paper_figures.py \
  --paper-pdf tmp/pdfs/andren1994.pdf
```

The production pressure adapter is available through the optional `pressure`
dependency extra. True multi-process CPU execution, deterministic AB2, and the
first dry-flow vector field are validated; GPU execution and physical ABL
stationarity/grid-convergence benchmarks remain later implementation
milestones. Archived runners are not silently reused as active solver code.

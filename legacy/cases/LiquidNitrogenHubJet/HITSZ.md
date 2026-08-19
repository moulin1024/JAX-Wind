# HITSZ liquid-nitrogen hub jet

The full-physics case injects liquid nitrogen in the positive streamwise
direction at `(x, y, z) = (12.15, 3.0, 0.876) m`. The nozzle radius is
`0.005 m` and the mass flow is `0.0125 kg/s`. With the model's liquid density
of `806.11 kg/m3`, these values imply a bulk liquid speed of
`0.1974357632 m/s`; the runner derives it rather than using the older 8 m/s
effective far-field source value.

Note that this is the *liquid* exit speed, so the injected momentum flux is
only `2.47e-3 N`. A real nozzle discharging into warm ambient air flash-boils,
and the vapour leaves far faster (order `36 m/s` at the `77 K` boiling point,
rising as it warms), so this configuration models the cold/dense source
faithfully but understates the jet's momentum. Pass an explicit
`--ln2-injection-speed` to restore vapour-like momentum.

The enabled coupling consists of:

- Rosin-Rammler LN2 parcels with drag, turbulent dispersion, evaporation,
  sensible/latent heat exchange, and ambient longwave radiation;
- conservative two-way momentum, heat, and nitrogen-mass deposition;
- transported potential temperature, water vapour, nitrogen vapour, liquid
  fog, ice fog, and mixture enthalpy;
- nitrogen-composition buoyancy and suspended-fog gravitational loading;
- water saturation adjustment, condensation/deposition, evaporation/
  sublimation, freezing/melting, and fog settling;
- a mass-only low-Mach outlet closure in the last 1.5 m of the domain.

Run 90 s from a new pressure-driven initial condition with:

```bash
tools/run_hitsz_ln2_jet.sh
```

The completed one-second plumbing test uses the memory-bounded grid while
retaining every physics coupling:

```bash
JAXWIND_LN2_CONFIG=legacy/cases/LiquidNitrogenHubJet/configs/hitsz_128x32x64_full_physics_smoke.toml \
JAXWIND_LN2_OUTPUT=outputs/hitsz_ln2_hub_jet/full_physics_smoke_1s \
tools/run_hitsz_ln2_jet.sh 1
```

The production grid requires more than the roughly 5.27 GiB of free device
memory available on the local 8 GiB RTX 3070 when every cryogenic coupling and
its diagnostics are compiled together. The full-grid one-second attempt was
therefore stopped by allocator/NCCL initialization failure before a completed
step; this coarse smoke configuration is a plumbing check, not production
physics data.

An optional second argument is a checkpoint directory written by this legacy
distributed runner. Unified `jaxwind` NPZ checkpoints are not binary-compatible
with its sharded checkpoint format.

The active `256 x 64 x 128` HITSZ grid has `dx = dy = 0.09375 m` and
`dz = 0.028125 m`. The 10 mm nozzle diameter is therefore subgrid. Parcel
positions retain the physical radius, while all carrier-phase exchange is
conservatively deposited over the cloud-in-cell stencil.

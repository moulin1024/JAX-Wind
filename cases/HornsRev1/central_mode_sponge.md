# Central advection with a nonperiodic upstream mode buffer

Diagnostic only; rejected as a solution for the wake-study objective. Do not extend this buffer toward the disk. Preserve AD-BEM sampling and
force kernels, central/RK3, and the existing downstream sponge. The new option
acts on raw face velocity before each RK pressure projection. No periodic
extension or Fourier transform is used.

The tendency is `-D4.T W D4 / (256 tau)`, with `D4` the fourth forward difference
in x and a nonnegative sine-squared window. All five-point rows lie inside the
buffer and avoid the prescribed inlet layers. It has zero integrated component
force, nonpositive energy work, and vanishes for an x-independent vertical
profile. Its interior alternating-mode decay rate is at most `1/tau`; response
at longer wavelengths scales as `sin(k dx/2)^8`. These algebraic properties do
not prove full-solver stability or turbulent log-law retention.

The buffer extends to 0.2 Lx (409.6 m in the V80 case). The rotor is at 512 m;
the upstream probe center is at 432 m. The probe's Gaussian tails and the global
pressure solution can still communicate changes to the rotor. Actual forces
must be checked even though the actuator model is unchanged.

## Small-case results, 300 s, 128×64×64

| Treatment | Upstream raw second-difference RMS [m/s] | Reduction |
|---|---:|---:|
| Central reference | 0.0397812 | — |
| Outlet sponge only | 0.0398294 | none |
| Additional mode buffer, tau=1 s | 0.00640515 | 83.9% |
| Additional mode buffer, tau=0.25 s | 0.00186001 | 95.3% |

The last case passes the existing fixed 90% gate for faces through x=336 m.
It does NOT eliminate the oscillations throughout the upstream region. Extending
the measurement through x=416 m gives RMS 0.0174943 m/s (central 0.0434371).
Raw-face plots reveal residual modes near the downstream edge of the buffer.

At the final checkpoint, tau=0.25 s changes applied thrust by -0.348% and applied
torque by -0.633%. These are instantaneous integrated loads, not time-averaged
accuracy or an independent aerodynamic validation. The prescribed electrical
power lookup is not used as aerodynamic evidence.

Artifacts:
- `outputs/backflow_mode_sponge_30270853`
- `outputs/backflow_mode_sponge_tau025_30270853`
- [Full upstream comparison](../../outputs/backflow_mode_sponge_tau025_30270853/full_upstream_comparison.png)
- Paired log-law validation: `outputs/backflow_mode_sponge_loglaw_30270853` (stopped before completing averaging; no log-law conclusion).

## Reproduce

In an allocated GPU session, with CUDA and the project Python environment:

```bash
export PYTHONPATH="$PWD/src:$PWD/tools"
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAXWIND_V80_FAST="$PWD/cases/HornsRev1/turbines/V80/CustomRotor.fst"
python -m jaxwind run cases/HornsRev1/fv_v80_small_central_mode_sponge_fast.toml
python tools/run_backflow_validation.py \
  --corrected-scheme central --sponge-start-fraction .75 --sponge-timescale 5 \
  --upstream-sponge-end .2 --upstream-sponge-timescale .25 \
  --skip-wake --output outputs/NEW_mode_sponge_loglaw
```

Use a fresh output directory. The runner freezes source/input hashes. The
log-law comparison uses 1800 s spin-up and 3600 s averaging with Porté-Agel,
RK3, fixed dt=0.5 s and a checked CFL ceiling of 0.9.

Targeted tests: 26 passed (`test_mode_sponge.py`, `test_outlet_sponge.py`, and
`test_backflow_validation.py`).

Research basis: [Revaz & Porté-Agel (2021)](https://www.mdpi.com/1996-1073/14/13/3745)
reports sensitivity of rotor predictions to force smoothing;
[Lundquist & Nordström (2020)](https://link.springer.com/article/10.1007/s10915-019-01116-9)
addresses energy-stable filtering and boundary closures. This particular
buffer remains an engineering experiment, not a method validated by those papers.

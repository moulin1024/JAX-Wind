# Conservative higher-order temperature and moisture transport

This experiment changes scalar numerics without fitting physical parameters to
the experimental outlet temperatures. It retains central momentum advection,
AMD, the projected parcel feedback, conservative incoming scalar fluxes, and
the existing grid, timestep, source distribution and inlet realization.

Two optional benchmark settings are provided under `[numerics]`:

- `scalar_advection_scheme = "muscl-mc"`: second-order MUSCL reconstruction
  with the monotonized-central limiter and subcycled SSP-RK3 for temperature
  and all five moisture fields.
- `scalar_advection_scheme = "upwind-ssprk3"`: the same transport and coupling
  path with donor-cell face values, to isolate spatial reconstruction.

The default `"upwind"` preserves the existing benchmark implementation.
The new cases are `inertial_projected_muscl_scalars.toml` and
`inertial_projected_upwind_ssprk3.toml` in the case directory. Both extend
`inertial_projected_central.toml` and require `case.scalar_boundary = "flux"`.

## Conservation and positivity

Every interior face has one shared flux. The open streamwise boundary uses
ambient values only for incoming flow and interior values for outgoing flow.
Lateral and vertical walls are impermeable; periodic lateral transport is also
supported. This implementation requires a uniform mesh.

MC face reconstruction stays between neighboring cell averages and reduces to
first order at extrema and nonperiodic endpoints. A reconstructed outgoing
face value is at most twice its nonnegative cell average. Each forward-Euler
stage uses a sufficient multidimensional bound:

`h * (2 * maximum_outgoing_rate + maximum_diffusion_rate) <= 0.8`.

Incoming advective and diffusive contributions are nonnegative. SSP-RK3 is a
convex combination of these Euler stages, so it preserves nonnegativity.
Uniform upper bounds also require solenoidal face velocities and compatible
reservoir data. No post-transport clipping is used. Temperature is transported
in kelvin; the temperature anomaly is restored afterward. Negative or
nonfinite transported water/temperature causes the benchmark to fail before
microphysics can mask the violation.

## Coupling limits

The new path transports temperature and moisture together on the incoming
projected velocity, with frozen molecular plus AMD diffusivity over that
substep. The carrier then advances momentum, retaining the transported
scalar for buoyancy while disabling its own scalar transport. Microphysical
exchange remains split around this sequence. SSP-RK3 describes the scalar
substep; it does not establish third-order accuracy for the full coupled
parcel/carrier solver. The matched upwind-SSP case is necessary because both
scalar time integration and temperature coupling differ from the legacy path.

The tests cover smooth translation convergence, discontinuous multidimensional
transport, positivity, open-boundary flux balance in either flow direction,
closed variable diffusion, coupled water/enthalpy conservation, and preventing
duplicate scalar advancement by the carrier.

## Comparison protocol

Use completed 4 s simulations and the same 3–4 s diagnostic averaging window.
Compare each of the nine spatially paired dry-bulb temperatures with the
original experimental table; all must meet the recorded 10% Celsius criterion.
Also inspect temporal block changes and carrier/parcel conservation. Reduced
numerical diffusion is not a physical closure for unresolved spray mixing and
need not improve agreement. These short controls do not establish statistical
convergence or resolve the missing drift-eliminator observation model.

## Verification record (2026-09-16)

There are 72 distinct passing targeted/regression checks across the initial
suite and the final 20-test transport/coupling suite (overlapping tests are
counted once). One existing runtime test,
`test_low_mach_continuation_consumes_new_stage_checkpoint`, fails because the
workflow recording plane is outside its mesh. The same failure was reproduced
in an isolated checkout of commit `1f963b2`; no unrelated workflow fix is part
of this experiment.

The new transport module and its tests pass Ruff. The broader modified-file
lint check also reports pre-existing import ordering and loop-lambda findings
in the carrier/benchmark files. Generated logs, XML results, source hashes and
the implementation snapshot are stored locally under
`outputs/waterjet_scalar_transport_20260916_193259/` and remain ignored by Git.

The fresh central-momentum/legacy-scalar control (gpudev job `30271517`)
reproduced the prior result to about 5e-12 K at the centre: 35.33465°C versus
31.4°C measured, worst relative error 12.53074%, 8/9 sensors passing.

## Completed matched experiment

Both new cases completed 8000 steps (4 s) on gpudev, with no positivity failure.
MUSCL job `30271605` and upwind-SSP job `30271606` exited successfully. Their
source hashes match the verified implementation snapshot.

| Scalar transport, central momentum | Centre DBT °C | Worst paired error | Passing sensors | Paired RMSE K |
| --- | ---: | ---: | ---: | ---: |
| Legacy upwind coupling | 35.33465 | 12.53074% | 8/9 | 2.13402 |
| Upwind SSP-RK3 control | 35.33891 | 12.54431% | 8/9 | 2.13636 |
| MUSCL-MC SSP-RK3 | 35.38607 | 12.69448% | 8/9 | 2.13794 |

The measured centre temperature is 31.4°C. Higher-order scalar reconstruction
changes the centre by +0.04715 K relative to the matched upwind control and
changes any sensor by at most 0.11040 K. The nine-sensor mean improves by only
0.01598 K while the paired RMSE and worst-sensor error become slightly worse.
**It does not close the discrepancy.** These small changes do not justify a
claim of statistically significant deterioration either.

The MUSCL run's maximum CFL is 0.2816; minimum saved water mixing ratio is zero.
Every transport step checks positivity before microphysics without clipping.
Maximum cumulative parcel mass/enthalpy residuals are 6.49e-14 kg and 1.78e-9 J.
The sampled carrier sensible residual is 2.783 W, versus 3.146 W for the matched
upwind control. This sampled residual is not an exact discrete stage budget.
Changes between 2–3 s and 3–4 s block means reach 0.360 K for MUSCL and 0.330 K
for upwind, larger than the reconstruction effect; longer statistical
validation would be needed to rank such small differences confidently.

The method remains opt-in. It removes an avoidable first-order transport
limitation but does not supply the missing coarse-grid spray mixing physics.
Full numerical results and the nine-sensor plot are in
`outputs/waterjet_scalar_transport_20260916_193259/summary.md`.

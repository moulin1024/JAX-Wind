"""Online resolved scalar-flux budget diagnostics for Andrén et al. Fig. 13.

The observer is deliberately outside the vector field: it reads accepted fields and
the pressure returned by projection, but cannot alter either the solver state or its
restart law.  Terms are formed from the same discrete tendencies used by the solver.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from benchmark.Andren1994 import run as andren
from jaxwind.integrators import Evaluation
from jaxwind.physics import DiagnosticLasdConstants


TERMS = ("production", "subgrid", "transport", "pressure", "coriolis")


def _plane(values):
    return jnp.mean(values, axis=(0, 2, 3))


def _fluctuation(values):
    return values - _plane(values)[None, :, None, None]


def _faces_to_cells(faces):
    """Interpolate stored upper-face values and the explicit lower face to cells."""
    upper = faces.owned.payload
    lower_plane = jnp.broadcast_to(
        jnp.asarray(faces.lower_boundary, dtype=upper.dtype), upper.shape[2:]
    )
    lower = jnp.concatenate((lower_plane[None, None], upper[:, :-1]), axis=1)
    return 0.5 * (lower + upper)


def _to_numpy(values) -> np.ndarray:
    return np.asarray(jax.device_get(values))


def observe(
    state,
    projection_pressure,
    *,
    vector_field,
    algebra,
    model,
    physical_grid,
    mechanical_scales,
    scalar_scales,
    diagnostic_constants: DiagnosticLasdConstants,
) -> dict[str, np.ndarray]:
    """Observe one instantaneous Fig. 13 budget without mutating the integration."""
    fields = state.fields
    context = algebra.boussinesq_context(fields)
    contributions = vector_field.evaluate_contributions(
        Evaluation(fields, state.clock, None)
    )

    scalar = fields.potential_temperature.payload
    w = context.momentum.arrays.w_at_cells
    scalar_fluctuation = _fluctuation(scalar)
    w_fluctuation = _fluctuation(w)

    scalar_advection = contributions.scalar_advection.payload
    vertical_advection = _faces_to_cells(contributions.advection.z)
    advective_budget = _plane(
        scalar_fluctuation * vertical_advection
        + w_fluctuation * scalar_advection
    )

    # Use the physical mean gradient here because this is the exact term printed in
    # Andrén et al.  Transport is the remainder of the solver's conservative
    # advection contribution, preserving the actual discrete product rule.
    physical_w_variance = _plane(
        (w_fluctuation * mechanical_scales.velocity) ** 2
    )
    physical_scalar_mean = (
        _plane(scalar) * scalar_scales.concentration
    )
    production = -physical_w_variance * jnp.gradient(
        physical_scalar_mean, physical_grid.dz
    )

    budget_scale = (
        scalar_scales.concentration
        * mechanical_scales.velocity
        / mechanical_scales.time
    )
    advective_budget = advective_budget * budget_scale
    transport = advective_budget - production

    scalar_sgs = contributions.scalar_sgs.payload
    vertical_sgs = _faces_to_cells(contributions.momentum_sgs.z)
    subgrid = _plane(
        scalar_fluctuation * vertical_sgs + w_fluctuation * scalar_sgs
    ) * budget_scale

    vertical_coriolis = _faces_to_cells(contributions.coriolis_geostrophic.z)
    coriolis = _plane(scalar_fluctuation * vertical_coriolis) * budget_scale

    # The paper plots p_r/rho + 2e_sgs/3.  In this solver the isotropic SGS stress is
    # not applied as a separate momentum tendency, so the projection Lagrange
    # multiplier is already that *modified* pressure.  Adding diagnostic e_sgs once
    # more here would double count it.  We nevertheless export the implied
    # diagnostic split below so this semantic mapping remains auditable.
    diagnostic = algebra.lasd_diagnostic_fields(
        context,
        model.momentum.sgs,
        model.scalar_sgs,
        model.scalar_boundary,
        constants=diagnostic_constants,
        wall=model.momentum.wall,
    )
    modified_pressure_gradient = _faces_to_cells(
        algebra.pressure_gradient(projection_pressure).z
    )
    # The projection pressure has the compatible zero-normal-gradient boundary
    # already encoded by the projection.  e_sgs is a cell diagnostic, not a
    # projection unknown: differentiate it with one-sided endpoint stencils instead
    # of incorrectly imposing the pressure boundary law on it.
    sgs_energy_gradient = jnp.gradient(
        diagnostic.sgs_tke,
        algebra.decomposition.grid.dz,
        axis=1,
    )
    isotropic_sgs_gradient = (2.0 / 3.0) * sgs_energy_gradient
    vertical_pressure_gradient = modified_pressure_gradient
    pressure = -_plane(
        scalar_fluctuation * vertical_pressure_gradient
    ) * budget_scale
    isotropic_sgs_pressure = -_plane(
        scalar_fluctuation * isotropic_sgs_gradient
    ) * budget_scale

    resolved_flux = _plane(w_fluctuation * scalar_fluctuation) * (
        mechanical_scales.velocity * scalar_scales.concentration
    )
    return {
        "production": _to_numpy(production),
        "subgrid": _to_numpy(subgrid),
        "transport": _to_numpy(transport),
        "pressure": _to_numpy(pressure),
        "diagnostic_isotropic_sgs_pressure": _to_numpy(isotropic_sgs_pressure),
        "coriolis": _to_numpy(coriolis),
        "resolved_flux": _to_numpy(resolved_flux),
    }


def write_samples(path: Path, times: list[float], samples: list[dict]) -> None:
    if not samples:
        return
    arrays = {
        name: np.stack([sample[name] for sample in samples]) for name in samples[0]
    }
    np.savez(path, times_seconds=np.asarray(times), **arrays)


def load_samples(path: Path) -> tuple[list[float], list[dict]]:
    if not path.exists():
        return [], []
    with np.load(path, allow_pickle=False) as archive:
        times = np.asarray(archive["times_seconds"]).tolist()
        names = tuple(name for name in archive.files if name != "times_seconds")
        samples = [
            {name: np.array(archive[name][index], copy=True) for name in names}
            for index in range(len(times))
        ]
    return times, samples


def averaged_budget(
    times: list[float],
    samples: list[dict],
    *,
    ustar: float,
    dz: float,
) -> dict[str, np.ndarray]:
    if len(samples) < 2:
        raise ValueError("Fig. 13 tendency requires at least two budget samples")
    elapsed = float(times[-1] - times[0])
    if elapsed <= 0.0:
        raise ValueError("Fig. 13 sample times must be strictly increasing")
    result = {name: np.mean([sample[name] for sample in samples], axis=0) for name in TERMS}
    result["tendency"] = (
        samples[-1]["resolved_flux"] - samples[0]["resolved_flux"]
    ) / elapsed
    rhs = sum((result[name] for name in TERMS), np.zeros_like(result["tendency"]))
    result["closure_residual"] = result["tendency"] - rhs
    scale = andren.F_CORIOLIS * (1.0e-3 / ustar) * ustar
    for name in TERMS + ("tendency", "closure_residual"):
        result[name] = result[name] / scale
    z = (np.arange(result["tendency"].size) + 0.5) * dz
    result["height"] = z * andren.F_CORIOLIS / ustar
    return result


def write_profile(path: Path, budget: dict[str, np.ndarray]) -> None:
    names = ("height",) + TERMS + ("tendency", "closure_residual")
    np.savetxt(
        path,
        np.column_stack([budget[name] for name in names]),
        delimiter=",",
        header=",".join(names),
        comments="",
    )

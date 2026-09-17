"""SGS-only dispersion scales; AMD inference requires a declared extra closure."""

from typing import NamedTuple

import jax.numpy as jnp

from .sgs import AnisotropicMinimumDissipation, FluentSmagorinsky, eddy_viscosity


class DPMLESFields(NamedTuple):
    viscosity: object
    kinetic_energy: object
    dissipation: object
    length: object


def dpm_les_fields(velocity, grid, boundaries, model, *, amd_length=None):
    """Fluent scale mapping v=nu/l, k=v², eps=k^(3/2)/l.

    Static Smagorinsky uses Fluent's documented wall-limited length. AMD can
    use the same dimensional mapping ONLY with an explicitly supplied length
    closure (meters). This is an unvalidated AMD extension, not a quantity AMD
    predicts or a Fluent AMD implementation. Resolved wake variance is excluded.
    """
    if isinstance(model, FluentSmagorinsky):
        if amd_length is not None:
            raise ValueError("AMD length supplied with a different carrier model")
        length = model.length_scale(grid, velocity.x.dtype)
    elif isinstance(model, AnisotropicMinimumDissipation):
        if amd_length is None or not 0 < amd_length < float("inf"):
            raise ValueError("AMD dispersion needs an explicit finite positive length")
        length = jnp.asarray(amd_length, velocity.x.dtype)
    else:
        raise TypeError(
            "DPM LES scales require FluentSmagorinsky or explicit AMD inference"
        )
    viscosity = eddy_viscosity(velocity, grid, boundaries, model)
    v = viscosity / length
    return DPMLESFields(viscosity, v * v, v**3 / length, length)

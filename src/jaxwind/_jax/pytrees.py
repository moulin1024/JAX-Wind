"""JAX tree registrations for solver states crossing compilation boundaries.

The public domain objects remain backend independent.  This private module
teaches JAX which members are arrays and which members are immutable semantic
metadata so an entire accepted step can be compiled as one callable.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def register_solver_pytrees() -> None:
    """Register persistent solver products exactly once per Python process."""

    import jax

    from jaxwind.domain import AcceptedClock, AddressableField, VerticalFaceField
    from jaxwind.integrators import (
        AB2BoussinesqState,
        ColdStart,
        PreviousTendency,
    )
    from jaxwind.operators import VelocityVector
    from jaxwind.physics import (
        BoussinesqFields,
        BoussinesqTendency,
        ConcurrentPrecursorEnvironment,
        LasdClosureMemory,
        MomentumLasdMemory,
        NoClosureMemory,
        ScalarLasdMemory,
    )

    register = jax.tree_util.register_dataclass
    register(
        AddressableField,
        data_fields=("payload",),
        meta_fields=("quantity", "location", "regions", "phase"),
    )
    register(
        VerticalFaceField,
        data_fields=("owned", "lower_boundary"),
        meta_fields=(),
    )
    register(VelocityVector, data_fields=("x", "y", "z"), meta_fields=())
    register(
        BoussinesqFields,
        data_fields=("velocity", "potential_temperature", "closure"),
        meta_fields=(),
    )
    register(
        BoussinesqTendency,
        data_fields=("velocity", "potential_temperature"),
        meta_fields=(),
    )
    register(
        MomentumLasdMemory,
        data_fields=(
            "coefficient",
            "lm",
            "mm",
            "qn",
            "nn",
            "trajectory_x",
            "trajectory_y",
            "trajectory_z",
        ),
        meta_fields=(),
    )
    register(
        ScalarLasdMemory,
        data_fields=("coefficient", "lm", "mm", "qn", "nn"),
        meta_fields=(),
    )
    register(
        LasdClosureMemory,
        data_fields=("momentum", "scalar"),
        meta_fields=("configuration_fingerprint",),
    )
    register(NoClosureMemory, data_fields=(), meta_fields=())
    register(AcceptedClock, data_fields=("time", "step"), meta_fields=())
    register(ColdStart, data_fields=(), meta_fields=())
    register(PreviousTendency, data_fields=("value",), meta_fields=())
    register(
        AB2BoussinesqState,
        data_fields=("fields", "clock", "history"),
        meta_fields=("integrator_fingerprint",),
    )
    register(
        ConcurrentPrecursorEnvironment,
        data_fields=("velocity", "closure"),
        meta_fields=(),
    )


__all__ = ["register_solver_pytrees"]

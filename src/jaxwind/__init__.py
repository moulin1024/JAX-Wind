"""JAX-Wind semantic core and unified JAX solver facade."""

from .domain import UniformGrid, VerticalBoundary
from .solver import Advance, build_solver, solve


def __getattr__(name: str):
    """Load the JAX execution facade only when an application requests it."""

    if name in ("JaxSolver", "build_jax_solver"):
        from .jax_solver import JaxSolver, build_jax_solver

        return {"JaxSolver": JaxSolver, "build_jax_solver": build_jax_solver}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "UniformGrid",
    "VerticalBoundary",
    "Advance",
    "build_solver",
    "build_jax_solver",
    "JaxSolver",
    "solve",
]

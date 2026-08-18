"""Typed diagnostic payloads returned by the JAX discretization."""

from __future__ import annotations

from typing import Any, NamedTuple


class ActuatorLineDiagnostic(NamedTuple):
    """Replicated per-element aerodynamic data from a distributed evaluation."""

    force_on_fluid_per_density: Any
    positions: Any
    tangents: Any
    normals: Any
    span_directions: Any
    blade_velocity: Any
    sampled_velocity: Any
    alpha_degrees: Any
    lift_coefficients: Any
    drag_coefficients: Any
    loss_factors: Any

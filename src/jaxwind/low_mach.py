"""Conservative variable-density low-Mach building blocks.

The thermodynamic pressure is prescribed and density follows an ideal-gas
mixture equation of state.  Mechanical pressure is obtained from a projection
of face momentum rather than velocity.  With the same face density used to
form and recover momentum, the pressure equation remains the existing compact
Poisson problem while the corrected fields satisfy

``(rho_new-rho_old)/dt + div(rho_face*u) = mass_source``.

This density-weighted projection is preferable to forming a variable-
coefficient velocity Poisson equation: it is conservative to solver tolerance
and keeps the FFT/GMG pressure backends available to mesoscale cases.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import jax.numpy as jnp

from jaxwind.domain.grid import Grid

from jaxwind.numerics.discretization import _cells_to_faces, divergence, pressure_gradient
from jaxwind.numerics.poisson import PressurePoisson
from .scalar import PassiveScalar, scalar_tendency
from .state import (
    StaggeredVelocity,
    spanwise_is_periodic,
    streamwise_is_periodic,
)


@dataclass(frozen=True, slots=True)
class IdealGasMixture:
    """Constant-pressure ideal-gas mixture used by the low-Mach solver.

    ``species_gas_constants`` correspond to explicitly transported mass
    fractions.  The unlisted remainder is assigned
    ``background_gas_constant``.  For example, an air/N2/H2O carrier uses
    ``(296.8, 461.5)`` for injected nitrogen and water vapour.
    """

    pressure: float | jnp.ndarray = 101_325.0
    background_gas_constant: float = 287.05
    species_gas_constants: tuple[float, ...] = ()
    temperature_floor: float = 50.0
    density_floor: float = 1.0e-6

    def __post_init__(self) -> None:
        values = (
            self.background_gas_constant,
            self.temperature_floor,
            self.density_floor,
            *self.species_gas_constants,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("ideal-gas constants and floors must be positive")
        if isinstance(self.pressure, (int, float)) and (
            not math.isfinite(self.pressure) or self.pressure <= 0.0
        ):
            raise ValueError("thermodynamic pressure must be positive")

    def pressure_field(self, template: jnp.ndarray) -> jnp.ndarray:
        """Broadcast scalar or hydrostatic base-state pressure to cell shape."""
        pressure = jnp.asarray(self.pressure, template.dtype)
        return jnp.broadcast_to(pressure, template.shape)

    def gas_constant(self, mass_fractions: Sequence[jnp.ndarray] = ()) -> jnp.ndarray:
        """Return the local mixture gas constant from species mass fractions."""
        if len(mass_fractions) != len(self.species_gas_constants):
            raise ValueError("one mass-fraction field is required per species gas constant")
        if not mass_fractions:
            return jnp.asarray(self.background_gas_constant)
        fractions = tuple(jnp.asarray(value) for value in mass_fractions)
        shape = fractions[0].shape
        if any(value.shape != shape for value in fractions):
            raise ValueError("all mixture mass fractions must have the same shape")
        total = sum(fractions, start=jnp.zeros_like(fractions[0]))
        background = jnp.maximum(1.0 - total, 0.0)
        result = background * self.background_gas_constant
        for fraction, gas_constant in zip(fractions, self.species_gas_constants):
            result = result + fraction * gas_constant
        # Renormalise only where clipping/roundoff makes the supplied sum
        # exceed one. This keeps the EOS finite while transport limiters act.
        normalization = jnp.maximum(background + total, 1.0)
        return result / normalization

    def density(
        self,
        temperature: jnp.ndarray,
        mass_fractions: Sequence[jnp.ndarray] = (),
    ) -> jnp.ndarray:
        """Return thermodynamic density ``p0/(R_mix*T)``."""
        temperature = jnp.asarray(temperature)
        gas_constant = self.gas_constant(mass_fractions).astype(temperature.dtype)
        pressure = self.pressure_field(temperature)
        floor = jnp.asarray(self.temperature_floor, temperature.dtype)
        density = pressure / (gas_constant * jnp.maximum(temperature, floor))
        return jnp.maximum(density, jnp.asarray(self.density_floor, density.dtype))

    def density_from_partial_densities(
        self,
        temperature: jnp.ndarray,
        partial_densities: Sequence[jnp.ndarray] = (),
    ) -> jnp.ndarray:
        """Recover total density from conservative species densities.

        For ``rho_k = rho*Y_k``, the ideal-mixture EOS can be rearranged
        analytically, avoiding an iterative conversion after conservative
        species transport::

            rho = [p0/T - sum(rho_k*(R_k-R_bg))] / R_bg.

        The remainder ``rho-sum(rho_k)`` is the background carrier.
        """
        if len(partial_densities) != len(self.species_gas_constants):
            raise ValueError("one partial-density field is required per species gas constant")
        temperature = jnp.asarray(temperature)
        floor = jnp.asarray(self.temperature_floor, temperature.dtype)
        pressure_over_temperature = self.pressure_field(
            temperature
        ) / jnp.maximum(temperature, floor)
        explicit = jnp.zeros_like(temperature)
        transported = jnp.zeros_like(temperature)
        for partial, gas_constant in zip(
            partial_densities, self.species_gas_constants
        ):
            partial = jnp.maximum(jnp.asarray(partial, temperature.dtype), 0.0)
            if partial.shape != temperature.shape:
                raise ValueError("partial densities must be cell centred")
            transported = transported + partial
            explicit = explicit + partial * (
                gas_constant - self.background_gas_constant
            )
        density = (
            pressure_over_temperature - explicit
        ) / self.background_gas_constant
        lower = jnp.maximum(
            transported, jnp.asarray(self.density_floor, temperature.dtype)
        )
        return jnp.maximum(density, lower)


def mixing_ratio_to_mass_fraction(mixing_ratio: jnp.ndarray) -> jnp.ndarray:
    """Convert kg species / kg dry carrier to a gas-phase mass fraction."""
    ratio = jnp.maximum(jnp.asarray(mixing_ratio), 0.0)
    return ratio / (1.0 + ratio)


def cell_to_faces(
    field: jnp.ndarray,
    velocity: StaggeredVelocity,
    grid: Grid,
) -> StaggeredVelocity:
    """Linearly interpolate a cell-centred field to physical MAC faces."""
    density = jnp.asarray(field)
    expected = (grid.nz, grid.ny, grid.nx)
    if density.shape != expected:
        raise ValueError(f"density must have cell shape {expected}")
    return StaggeredVelocity(
        _cells_to_faces(
            density,
            grid,
            2,
            periodic=streamwise_is_periodic(velocity, grid),
            boundary="copy",
        ),
        _cells_to_faces(
            density,
            grid,
            1,
            periodic=spanwise_is_periodic(velocity, grid),
            boundary="copy",
        ),
        _cells_to_faces(
            density, grid, 0, periodic=False, boundary="copy"
        ),
    )


def face_density(
    density: jnp.ndarray,
    velocity: StaggeredVelocity,
    grid: Grid,
) -> StaggeredVelocity:
    """Interpolate cell density to the three MAC face families."""
    return cell_to_faces(density, velocity, grid)


def _multiply(
    left: StaggeredVelocity,
    right: StaggeredVelocity,
) -> StaggeredVelocity:
    return StaggeredVelocity(
        left.x * right.x,
        left.y * right.y,
        left.z * right.z,
    )


def mass_flux(
    density: jnp.ndarray,
    velocity: StaggeredVelocity,
    grid: Grid,
) -> StaggeredVelocity:
    """Return face mass flux per unit area, ``rho_face*u_face``."""
    return _multiply(face_density(density, velocity, grid), velocity)


def dilatation_correction(
    velocity: StaggeredVelocity,
    grid: Grid,
) -> StaggeredVelocity:
    """Correction turning conservative ``-div(u u)`` into ``-u.grad(u)``.

    The two forms coincide in incompressible flow. Low-Mach heat and mass
    release produce nonzero dilatation, for which the velocity equation needs
    the discrete ``u div(u)`` term. Combined with conservative continuity and
    source-relative momentum this is equivalent to advancing momentum.
    """
    face_dilatation = cell_to_faces(divergence(velocity, grid), velocity, grid)
    return _multiply(velocity, face_dilatation)


def continuity_residual(
    velocity: StaggeredVelocity,
    previous_density: jnp.ndarray,
    density: jnp.ndarray,
    grid: Grid,
    dt: float,
    mass_source: jnp.ndarray | float = 0.0,
) -> jnp.ndarray:
    """Discrete residual of total carrier mass conservation."""
    density = jnp.asarray(density)
    previous_density = jnp.asarray(previous_density, density.dtype)
    if previous_density.shape != density.shape:
        raise ValueError("old and new density fields must have the same shape")
    source = jnp.broadcast_to(jnp.asarray(mass_source, density.dtype), density.shape)
    return (
        (density - previous_density) / jnp.asarray(dt, density.dtype)
        + divergence(mass_flux(density, velocity, grid), grid)
        - source
    )


def project_low_mach(
    velocity: StaggeredVelocity,
    previous_density: jnp.ndarray,
    density: jnp.ndarray,
    poisson: PressurePoisson,
    dt: float,
    *,
    mass_source: jnp.ndarray | float = 0.0,
    initial_pressure: jnp.ndarray | None = None,
    continuity_dt: float | None = None,
) -> tuple[StaggeredVelocity, jnp.ndarray]:
    """Project face momentum so the conservative continuity equation holds.

    ``mass_source`` is a volumetric source in kg m-3 s-1.  A closed or fully
    periodic domain requires its integral to equal the domain-integrated
    density change; an open boundary can carry the imbalance out of the box.
    """
    grid = poisson.grid
    density = jnp.asarray(density)
    if jnp.issubdtype(density.dtype, jnp.complexfloating):
        raise ValueError("density must be real")
    mass_interval = dt if continuity_dt is None else continuity_dt
    residual = continuity_residual(
        velocity,
        previous_density,
        density,
        grid,
        mass_interval,
        mass_source,
    )
    pressure = poisson.solve(residual / jnp.asarray(dt, density.dtype), initial_pressure)
    gradient = pressure_gradient(
        pressure,
        grid,
        periodic_x=poisson.periodic_x,
        periodic_y=poisson.periodic_y,
        open_x_low=poisson.open_x_low,
    )
    rho_face = face_density(density, velocity, grid)
    step = jnp.asarray(dt, density.dtype)
    corrected = StaggeredVelocity(
        velocity.x - step * gradient.x / rho_face.x,
        velocity.y - step * gradient.y / rho_face.y,
        velocity.z - step * gradient.z / rho_face.z,
    )
    return corrected, pressure


def conservative_specific_tendency(
    specific: jnp.ndarray,
    density: jnp.ndarray,
    velocity: StaggeredVelocity,
    grid: Grid,
    model: PassiveScalar,
    *,
    eddy_diffusivity: jnp.ndarray | float = 0.0,
) -> jnp.ndarray:
    """Return ``d(rho*phi)/dt`` from conservative advection and diffusion.

    Molecular and eddy diffusivities are kinematic (m2/s); multiplication by
    density converts their scalar flux to the conservative mass basis.
    Volumetric sources are intentionally added by the caller so their units
    remain explicit.
    """
    density = jnp.asarray(density, specific.dtype)
    effective = density * (
        jnp.asarray(model.diffusivity, specific.dtype)
        + jnp.asarray(eddy_diffusivity, specific.dtype)
    )
    conservative_model = PassiveScalar(
        diffusivity=0.0,
        turbulent_prandtl=1.0,
        lower_flux=model.lower_flux,
        upper_flux=model.upper_flux,
    )
    return scalar_tendency(
        specific,
        mass_flux(density, velocity, grid),
        grid,
        conservative_model,
        eddy_viscosity=effective,
    )


__all__ = [
    "IdealGasMixture",
    "cell_to_faces",
    "conservative_specific_tendency",
    "continuity_residual",
    "dilatation_correction",
    "face_density",
    "mass_flux",
    "mixing_ratio_to_mass_fraction",
    "project_low_mach",
]

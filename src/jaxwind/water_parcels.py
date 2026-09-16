"""Unresolved inertial water parcels coupled to shared atmospheric moisture.

The nozzle is a prescribed post-atomization boundary, not a resolved jet.
CIC sampling/deposition reuses the existing parcel/mesh implementation. Water
phase exchange is exclusively provided by physics.moisture.
"""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .cryogenic import (
    _cell_to_velocity_faces,
    _cic_coordinates,
    _cic_deposit_many,
    _cic_sample_many,
)
from .moist_abl import MoistAtmosphericSolution
from .numerics.discretization import cell_velocity
from .physics.moisture import (
    WaterDropletProperties,
    WaterDropletUpdate,
    advance_water_droplet,
)
from .water_spray import advance_water_droplet_motion


@dataclass(frozen=True)
class WaterParcelSource:
    center: tuple[float, float, float]
    radius: float
    speed: float
    temperature: float
    mass_flow: float
    half_angle_degrees: float
    diameter_scale: float
    diameter_spread: float
    diameter_minimum: float
    diameter_maximum: float
    count_per_step: int = 16
    capacity: int = 16384
    substeps: int = 4
    ramp_time: float = 0.1
    inner_outer_radius_ratio: float = 1.0

    def outer_cone_tangent(self):
        """Preserve the prescribed mass-mean angle for a finite annular spray.

        Uniform mass per annular area is an explicit handoff assumption. The
        inner/outer radius ratio comes from spray photographs, not cooling.
        Ratio=1 retains the legacy infinitesimally thin hollow cone exactly.
        """
        import math

        import numpy as np

        if self.inner_outer_radius_ratio == 1:
            return math.tan(math.radians(self.half_angle_degrees))
        nodes, weights = np.polynomial.legendre.leggauss(32)
        r = np.sqrt(
            self.inner_outer_radius_ratio**2
            + (1 - self.inner_outer_radius_ratio**2) * (nodes + 1) / 2
        )
        target = math.radians(self.half_angle_degrees)
        lo, hi = 0.0, math.pi / 2
        for _ in range(60):
            angle = 0.5 * (lo + hi)
            mean = np.sum(weights * np.arctan(np.tan(angle) * r)) / 2
            if mean < target:
                lo = angle
            else:
                hi = angle
        return math.tan(0.5 * (lo + hi))

    def __post_init__(self):
        import math

        positive = (
            self.radius,
            self.speed,
            self.temperature,
            self.mass_flow,
            self.diameter_scale,
            self.diameter_spread,
            self.diameter_minimum,
            self.diameter_maximum,
        )
        if any(not math.isfinite(v) or v <= 0 for v in positive):
            raise ValueError(
                "water parcel source properties must be finite and positive"
            )
        if not 0 < self.inner_outer_radius_ratio <= 1:
            raise ValueError("annular inner/outer radius ratio must be in (0, 1]")
        if self.diameter_minimum >= self.diameter_maximum:
            raise ValueError("water parcel diameter bounds must increase")
        if (
            not 0 <= self.half_angle_degrees < 90
            or not math.isfinite(self.ramp_time)
            or self.ramp_time < 0
            or len(self.center) != 3
            or any(not math.isfinite(v) for v in self.center)
        ):
            raise ValueError("invalid cone angle or ramp time")
        if any(
            type(v) is not int or v < 1
            for v in (self.count_per_step, self.capacity, self.substeps)
        ):
            raise ValueError("parcel counts and substeps must be positive integers")
        if self.capacity < self.count_per_step:
            raise ValueError("parcel capacity must accommodate an injection")


class WaterParcels(NamedTuple):
    position: jax.Array
    velocity: jax.Array
    mass: jax.Array
    temperature: jax.Array
    multiplicity: jax.Array
    active: jax.Array
    injected_mass: jax.Array
    escaped_mass: jax.Array
    evaporated_mass: jax.Array
    overflow_mass: jax.Array
    injected_enthalpy: jax.Array
    escaped_enthalpy: jax.Array
    gas_sensible_energy_loss: jax.Array
    touched_wall: jax.Array
    first_wall_mass: jax.Array
    first_wall_enthalpy: jax.Array
    escaped_wall_mass: jax.Array
    escaped_wall_enthalpy: jax.Array
    wall_gas_sensible_energy_loss: jax.Array
    wall_evaporated_mass: jax.Array


class InertialMoistAtmosphericSolution(NamedTuple):
    velocity: object
    pressure: jax.Array
    momentum_tendency: object
    scalar: jax.Array
    scalar_tendency: jax.Array
    time: jax.Array
    step: jax.Array
    moisture: object
    parcels: WaterParcels


def initial_water_parcels(source, dtype):
    zeros = jnp.zeros(source.capacity, dtype)
    scalar = jnp.asarray(0.0, dtype)
    return WaterParcels(
        jnp.zeros((3, source.capacity), dtype),
        jnp.zeros((3, source.capacity), dtype),
        zeros,
        jnp.full_like(zeros, source.temperature),
        zeros,
        jnp.zeros(source.capacity, bool),
        scalar,
        scalar,
        scalar,
        scalar,
        scalar,
        scalar,
        scalar,
        jnp.zeros(source.capacity, bool),
        scalar,
        scalar,
        scalar,
        scalar,
        scalar,
        scalar,
    )


def inject_water_parcels(
    parcels, time, step, dt, source, config, properties=WaterDropletProperties()
):
    """Equal physical-mass parcels sample the truncated *mass* distribution.

    Independent irrational sequences cover azimuth and size without runtime
    random state. Overflow is recorded and must be rejected by the runtime.
    """
    count = source.count_per_step
    slots = jnp.argsort(parcels.active.astype(jnp.int32), stable=True)[:count]
    valid = jnp.arange(count) < jnp.sum(~parcels.active)
    index = (
        step.astype(parcels.mass.dtype) * count
        + jnp.arange(count, dtype=parcels.mass.dtype)
        + 0.5
    )
    quantile = jnp.mod(index * 0.7548776662466927, 1.0)
    angle = 2 * jnp.pi * jnp.mod(index * 0.5698402909980532, 1.0)
    lo = jnp.exp(
        -((source.diameter_minimum / source.diameter_scale) ** source.diameter_spread)
    )
    hi = jnp.exp(
        -((source.diameter_maximum / source.diameter_scale) ** source.diameter_spread)
    )
    diameter = source.diameter_scale * (-jnp.log(lo - quantile * (lo - hi))) ** (
        1 / source.diameter_spread
    )
    mass = config.water_density * jnp.pi * diameter**3 / 6
    if source.inner_outer_radius_ratio == 1:
        theta = source.half_angle_degrees * jnp.pi / 180
    else:
        radial_quantile = jnp.mod(index * 0.4142135623730951, 1.0)
        radius_fraction = jnp.sqrt(
            source.inner_outer_radius_ratio**2
            + (1 - source.inner_outer_radius_ratio**2) * radial_quantile
        )
        theta = jnp.arctan(source.outer_cone_tangent() * radius_fraction)
    position = jnp.stack(
        (
            jnp.full_like(angle, source.center[0]),
            source.center[1] + source.radius * jnp.cos(angle),
            source.center[2] + source.radius * jnp.sin(angle),
        )
    )
    velocity = source.speed * jnp.stack(
        (
            jnp.full_like(angle, jnp.cos(theta)),
            jnp.sin(theta) * jnp.cos(angle),
            jnp.sin(theta) * jnp.sin(angle),
        )
    )
    phase = jnp.clip((time + 0.5 * dt) / max(source.ramp_time, 1e-30), 0, 1)
    ramp = 1.0 if source.ramp_time == 0 else 0.5 * (1 - jnp.cos(jnp.pi * phase))
    parcel_mass = source.mass_flow * dt * ramp / count

    def assign(old, new):
        return old.at[..., slots].set(jnp.where(valid, new, old[..., slots]))

    return parcels._replace(
        position=assign(parcels.position, position),
        velocity=assign(parcels.velocity, velocity),
        mass=assign(parcels.mass, mass),
        temperature=assign(parcels.temperature, source.temperature),
        multiplicity=assign(parcels.multiplicity, parcel_mass / mass),
        active=parcels.active.at[slots].set(parcels.active[slots] | valid),
        touched_wall=assign(parcels.touched_wall, False),
        injected_mass=parcels.injected_mass + parcel_mass * jnp.sum(valid),
        overflow_mass=parcels.overflow_mass + parcel_mass * jnp.sum(~valid),
        injected_enthalpy=parcels.injected_enthalpy
        + parcel_mass
        * jnp.sum(valid)
        * properties.liquid_heat_capacity
        * (source.temperature - config.freezing_temperature),
    )


def exchange_water_parcels(
    flow,
    parcels,
    grid,
    dt,
    config,
    temperature_offset,
    properties=WaterDropletProperties(),
    *,
    thermal_exchange=True,
    side_boundary="wet-wall",
):
    """Advance parcels and deposit momentum, vapor and sensible-heat exchange.

    Droplets striking side walls keep tangential velocity and lose their normal
    velocity (the reference's idealized wet-wall condition); x boundaries are
    escape boundaries. Carrier source deposition is at the path midpoint.
    Gas projection, cloud adjustment and transport are performed by the caller.
    Set the static option thermal_exchange=False for transport-only validation:
    droplet mass and temperature stay fixed, while drag coupling remains active.
    side_boundary="escape" removes parcels at y/z domain exits without wall
    contact. This static option does not change the carrier boundary conditions.
    """
    if side_boundary not in ("wet-wall", "escape"):
        raise ValueError("side_boundary must be wet-wall or escape")
    coordinates = _cic_coordinates(*parcels.position, grid)
    fields = jnp.stack(
        (
            *cell_velocity(flow.velocity),
            flow.scalar + temperature_offset,
            flow.moisture.vapor,
        )
    )
    sampled = _cic_sample_many(fields, coordinates)
    gas_velocity, gas_temperature, gas_vapor = sampled[:3], sampled[3], sampled[4]
    active = parcels.active
    diameter = jnp.cbrt(
        6 * jnp.maximum(parcels.mass, 1e-30) / (jnp.pi * config.water_density)
    )
    h = dt * active
    velocity, displacement, reaction = advance_water_droplet_motion(
        parcels.velocity, gas_velocity, diameter, h, config, properties
    )
    if thermal_exchange:
        update = advance_water_droplet(
            parcels.mass,
            parcels.temperature,
            gas_temperature,
            gas_vapor,
            jnp.linalg.norm(parcels.velocity - gas_velocity, axis=0),
            h,
            config,
            properties,
        )
    else:
        update = WaterDropletUpdate(
            parcels.mass,
            diameter,
            parcels.temperature,
            jnp.zeros_like(parcels.mass),
            jnp.zeros_like(parcels.mass),
        )
    number = parcels.multiplicity * active
    evaporated = update.evaporated_mass * number
    impulse = parcels.mass * number * reaction + evaporated * (velocity - gas_velocity)
    heat = update.gas_sensible_energy_loss * number
    candidate = parcels.position + displacement
    wall_collision = jnp.zeros_like(active)
    escaped = active & ((candidate[0] < 0) | (candidate[0] >= grid.lx))
    for axis, extent in ((1, grid.ly), (2, grid.lz)):
        if side_boundary == "escape":
            escaped = escaped | (
                active & ((candidate[axis] < 0) | (candidate[axis] >= extent))
            )
        else:
            collision = (candidate[axis] < 0) | (candidate[axis] > extent)
            wall_collision = wall_collision | (active & collision)
            candidate = candidate.at[axis].set(jnp.clip(candidate[axis], 0, extent))
            velocity = velocity.at[axis].set(jnp.where(collision, 0, velocity[axis]))
    alive = active & ~escaped & (update.mass > 0)
    midpoint = 0.5 * (parcels.position + candidate)
    deposit = _cic_deposit_many(
        jnp.concatenate((impulse, evaporated[None], heat[None])),
        _cic_coordinates(*midpoint, grid),
        flow.scalar.shape,
    )
    dry_mass = config.dry_air_density * jnp.asarray(grid.cell_volumes)
    acceleration = _cell_to_velocity_faces(*(deposit[:3] / dry_mass), grid)
    velocity_field = type(flow.velocity)(
        *(a + b for a, b in zip(flow.velocity, acceleration))
    )
    flow = flow._replace(
        velocity=velocity_field,
        scalar=flow.scalar - deposit[4] / (dry_mass * config.dry_air_heat_capacity),
        moisture=flow.moisture._replace(
            vapor=flow.moisture.vapor + deposit[3] / dry_mass
        ),
    )
    # Tag after the thermal substep: transfer during an impact substep belongs
    # to free flight; subsequent transfer belongs to the wall-contact population.
    # This convention makes the population water/enthalpy budgets exact.
    first_wall = wall_collision & ~parcels.touched_wall
    touched_wall = parcels.touched_wall | wall_collision
    physical_mass = update.mass * number
    physical_enthalpy = (
        physical_mass
        * properties.liquid_heat_capacity
        * (update.temperature - config.freezing_temperature)
    )
    parcels = parcels._replace(
        touched_wall=touched_wall,
        first_wall_mass=parcels.first_wall_mass
        + jnp.sum(jnp.where(first_wall, physical_mass, 0)),
        first_wall_enthalpy=parcels.first_wall_enthalpy
        + jnp.sum(jnp.where(first_wall, physical_enthalpy, 0)),
        escaped_wall_mass=parcels.escaped_wall_mass
        + jnp.sum(jnp.where(escaped & touched_wall, physical_mass, 0)),
        escaped_wall_enthalpy=parcels.escaped_wall_enthalpy
        + jnp.sum(jnp.where(escaped & touched_wall, physical_enthalpy, 0)),
        wall_gas_sensible_energy_loss=parcels.wall_gas_sensible_energy_loss
        + jnp.sum(jnp.where(parcels.touched_wall, heat, 0)),
        wall_evaporated_mass=parcels.wall_evaporated_mass
        + jnp.sum(jnp.where(parcels.touched_wall, evaporated, 0)),
        position=candidate,
        velocity=velocity,
        mass=jnp.where(alive, update.mass, 0),
        temperature=update.temperature,
        active=alive,
        escaped_mass=parcels.escaped_mass
        + jnp.sum(jnp.where(escaped, update.mass * number, 0)),
        evaporated_mass=parcels.evaporated_mass + jnp.sum(evaporated),
        escaped_enthalpy=parcels.escaped_enthalpy
        + jnp.sum(
            jnp.where(
                escaped,
                update.mass
                * number
                * properties.liquid_heat_capacity
                * (update.temperature - config.freezing_temperature),
                0,
            )
        ),
        gas_sensible_energy_loss=parcels.gas_sensible_energy_loss + jnp.sum(heat),
    )
    return flow, parcels


def build_inertial_water_step(
    moist_step, grid, source, config, temperature_offset, *,
    thermal_exchange=True, side_boundary="wet-wall", project_velocity=None
):
    """Couple inertial water to the ordinary moist atmosphere integrator.

    Only vapor and cloud condensate contribute to gas virtual temperature.
    Inertial liquid loads the carrier through its drag reaction, never by also
    inserting parcel mass into the entrained-cloud buoyancy correction.
    thermal_exchange controls parcel heat and phase exchange only; the carrier
    retains its ordinary humidity/cloud integrator in transport-only runs.
    An optional project_velocity(velocity, dt, inflow) projects the feedback impulse
    before scalar/moisture transport. Its pressure is an impulsive constraint;
    retain the carrier's lagged pressure for its subsequent transport step.
    Without this projection, parcel feedback can advect uniform humidity with
    a divergent velocity. The default preserves archived integrations.
    """
    if side_boundary not in ("wet-wall", "escape"):
        raise ValueError("side_boundary must be wet-wall or escape")
    if not (
        0 <= source.center[0] < grid.lx
        and source.radius <= source.center[1] <= grid.ly - source.radius
        and source.radius <= source.center[2] <= grid.lz - source.radius
    ):
        raise ValueError("water parcel nozzle lies outside the domain")
    if source.temperature < config.freezing_temperature:
        raise ValueError("inertial water closure requires warm injected water")

    def step(state, dt, inflow):
        flow = MoistAtmosphericSolution(*state[:8])
        parcels = inject_water_parcels(
            state.parcels, state.time, state.step, dt, source, config
        )

        def subcycle(_, values):
            return exchange_water_parcels(
                *values, grid, dt / source.substeps, config, temperature_offset,
                thermal_exchange=thermal_exchange,
                side_boundary=side_boundary,
            )

        flow, parcels = jax.lax.fori_loop(0, source.substeps, subcycle, (flow, parcels))
        if project_velocity is not None:
            flow = flow._replace(velocity=project_velocity(flow.velocity, dt, inflow))
        flow = moist_step(flow, dt, inflow)
        return InertialMoistAtmosphericSolution(*flow, parcels)

    return step

"""Minimal Lagrangian spray model coupled to the Boussinesq LES state.

The carrier phase remains Eulerian.  A fixed-capacity parcel buffer stores
physical positions, velocities, diameter, temperature, multiplicity, and an
active mask.  Parcel-to-grid exchange uses conservative cloud-in-cell (CIC)
deposition.  All parcel calculations use SI units; the returned carrier-phase
increments can be applied directly to the solver fields before projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import NamedTuple

import jax
from jax import lax
import jax.numpy as jnp

from .config import Params
from .diagnostics import diagnostics as flow_diagnostics
from .grid import upper_face_to_center
from .grid import make_operators
from .init import apply_velocity_bc, initial_state
from .pressure import project_velocity
from .scalar import apply_moisture_bounds
from .sgs import _spectral_box_filter
from .state import FlowState, Operators
from .timestep import step as flow_step


@dataclass(frozen=True)
class SprayDPMConfig:
    """Physical and numerical parameters for a parcel-based spray injection."""

    material: str = "water"
    max_parcels: int = 1024
    initial_parcels: int = 0
    parcel_weight: float = 1.0
    parcels_per_step: int = 0
    mass_flow_rate: float = 0.0
    injection_start_time: float = 0.0
    injection_end_time: float = float("inf")
    random_seed: int = 0
    turbulent_dispersion_enabled: bool = False
    injection_x: float = 0.0
    injection_y: float = 0.0
    injection_z: float = 100.0
    injection_radius: float = 0.0
    injection_u: float = 0.0
    injection_v: float = 0.0
    injection_w: float = 0.0
    initial_diameter: float = 100.0e-6
    diameter_distribution: str = "monodisperse"
    minimum_diameter: float = 10.0e-6
    maximum_diameter: float = 500.0e-6
    rosin_rammler_spread: float = 3.0
    lognormal_geometric_stddev: float = 1.5
    tabulated_diameters: tuple[float, ...] = ()
    tabulated_mass_fractions: tuple[float, ...] = ()
    initial_temperature: float = 283.15
    substeps: int = 4
    min_diameter: float = 1.0e-7

    air_density: float = 1.225
    air_dynamic_viscosity: float = 1.81e-5
    air_thermal_conductivity: float = 0.0257
    air_heat_capacity: float = 1005.0
    vapor_diffusivity: float = 2.5e-5
    dry_air_gas_constant: float = 287.05
    water_vapor_gas_constant: float = 461.5
    liquid_density: float = 997.0
    water_density: float = 997.0
    liquid_heat_capacity: float = 4180.0
    latent_heat: float = 2.45e6
    boiling_temperature: float = 373.15
    salinity_mass_fraction: float = 0.0
    salt_molar_mass: float = 58.44e-3
    water_molar_mass: float = 18.01528e-3
    salt_vant_hoff_factor: float = 2.0
    salt_osmotic_coefficient: float = 0.93
    osmotic_coefficient_model: str = "andreas1989"
    thermodynamic_transfer_model: str = "ranz_marshall_spalding"
    surface_tension: float = 0.072
    universal_gas_constant: float = 8.314462618
    dry_air_molar_mass: float = 28.9644e-3
    vapor_jump_length: float = 8.0e-8
    thermal_jump_length: float = 2.16e-7
    vapor_accommodation_coefficient: float = 0.06
    thermal_accommodation_coefficient: float = 0.7
    drag_correction_model: str = "instantaneous_clift_gauvin"
    ventilation_correction_enabled: bool = True

    shortwave_flux: float = 0.0
    shortwave_absorption_efficiency: float = 0.0
    sky_temperature: float = 273.15
    liquid_emissivity: float = 0.96
    stefan_boltzmann: float = 5.670374419e-8

    def __post_init__(self) -> None:
        material = str(self.material).lower().replace("-", "_")
        if material not in {"water", "nitrogen"}:
            raise ValueError("spray material must be 'water' or 'nitrogen'")
        object.__setattr__(self, "material", material)
        if self.max_parcels <= 0:
            raise ValueError("max_parcels must be positive")
        if not 0 <= self.initial_parcels <= self.max_parcels:
            raise ValueError("initial_parcels must lie in [0, max_parcels]")
        if self.parcel_weight <= 0.0:
            raise ValueError("parcel_weight must be positive")
        if self.parcels_per_step < 0:
            raise ValueError("parcels_per_step must be non-negative")
        if self.parcels_per_step > self.max_parcels:
            raise ValueError("parcels_per_step must not exceed max_parcels")
        if self.mass_flow_rate < 0.0:
            raise ValueError("mass_flow_rate must be non-negative")
        if self.injection_start_time < 0.0:
            raise ValueError("injection_start_time must be non-negative")
        if self.injection_end_time <= self.injection_start_time:
            raise ValueError("injection_end_time must exceed injection_start_time")
        if self.initial_diameter <= 0.0 or self.min_diameter <= 0.0:
            raise ValueError("spray diameters must be positive")
        if self.initial_diameter < self.min_diameter:
            raise ValueError("initial_diameter must not be below min_diameter")
        distribution = str(self.diameter_distribution).lower().replace("-", "_")
        aliases = {
            "mono": "monodisperse",
            "monodisperse": "monodisperse",
            "rosin_rammler": "rosin_rammler",
            "rr": "rosin_rammler",
            "lognormal": "lognormal",
            "log_normal": "lognormal",
            "tabulated": "tabulated",
            "discrete": "tabulated",
        }
        if distribution not in aliases:
            raise ValueError(f"unsupported diameter_distribution: {self.diameter_distribution}")
        object.__setattr__(self, "diameter_distribution", aliases[distribution])
        if self.minimum_diameter <= 0.0 or self.maximum_diameter <= self.minimum_diameter:
            raise ValueError("diameter distribution bounds must satisfy 0 < min < max")
        if self.rosin_rammler_spread <= 0.0:
            raise ValueError("rosin_rammler_spread must be positive")
        if self.lognormal_geometric_stddev <= 1.0:
            raise ValueError("lognormal_geometric_stddev must exceed one")
        if self.diameter_distribution == "tabulated":
            if not self.tabulated_diameters:
                raise ValueError("tabulated distribution requires diameters")
            if len(self.tabulated_diameters) != len(self.tabulated_mass_fractions):
                raise ValueError("tabulated diameters and mass fractions must have equal length")
            if any(value <= 0.0 for value in self.tabulated_diameters):
                raise ValueError("tabulated diameters must be positive")
            if any(value < 0.0 for value in self.tabulated_mass_fractions):
                raise ValueError("tabulated mass fractions must be non-negative")
            if sum(self.tabulated_mass_fractions) <= 0.0:
                raise ValueError("tabulated mass fractions must have positive sum")
        if self.initial_temperature <= 0.0 or self.sky_temperature <= 0.0:
            raise ValueError("absolute temperatures must be positive")
        if self.injection_radius < 0.0:
            raise ValueError("injection_radius must be non-negative")
        if self.substeps <= 0:
            raise ValueError("substeps must be positive")
        positive_properties = (
            self.air_density,
            self.air_dynamic_viscosity,
            self.air_thermal_conductivity,
            self.air_heat_capacity,
            self.vapor_diffusivity,
            self.dry_air_gas_constant,
            self.water_vapor_gas_constant,
            self.liquid_density,
            self.water_density,
            self.liquid_heat_capacity,
            self.latent_heat,
            self.boiling_temperature,
            self.salt_molar_mass,
            self.water_molar_mass,
            self.salt_vant_hoff_factor,
            self.salt_osmotic_coefficient,
            self.surface_tension,
            self.universal_gas_constant,
            self.dry_air_molar_mass,
            self.vapor_jump_length,
            self.thermal_jump_length,
            self.vapor_accommodation_coefficient,
            self.thermal_accommodation_coefficient,
        )
        if any(value <= 0.0 for value in positive_properties):
            raise ValueError("spray material properties must be positive")
        if self.shortwave_flux < 0.0:
            raise ValueError("shortwave_flux must be non-negative")
        if not 0.0 <= self.shortwave_absorption_efficiency <= 1.0:
            raise ValueError("shortwave_absorption_efficiency must lie in [0, 1]")
        if not 0.0 <= self.liquid_emissivity <= 1.0:
            raise ValueError("liquid_emissivity must lie in [0, 1]")
        if not 0.0 <= self.salinity_mass_fraction < 1.0:
            raise ValueError("salinity_mass_fraction must lie in [0, 1)")
        osmotic_model = str(self.osmotic_coefficient_model).lower()
        if osmotic_model not in {"constant", "andreas1989"}:
            raise ValueError(
                "osmotic_coefficient_model must be 'constant' or 'andreas1989'"
            )
        object.__setattr__(self, "osmotic_coefficient_model", osmotic_model)
        transfer_model = str(self.thermodynamic_transfer_model).lower()
        if transfer_model not in {
            "ranz_marshall_spalding",
            "veron2020",
        }:
            raise ValueError(
                "thermodynamic_transfer_model must be "
                "'ranz_marshall_spalding' or 'veron2020'"
            )
        object.__setattr__(
            self, "thermodynamic_transfer_model", transfer_model
        )
        drag_model = str(self.drag_correction_model).lower()
        if drag_model not in {
            "instantaneous_clift_gauvin",
            "terminal_settling",
        }:
            raise ValueError(
                "drag_correction_model must be "
                "'instantaneous_clift_gauvin' or 'terminal_settling'"
            )
        object.__setattr__(self, "drag_correction_model", drag_model)


class SprayState(NamedTuple):
    x: jax.Array
    y: jax.Array
    z: jax.Array
    u: jax.Array
    v: jax.Array
    w: jax.Array
    mass: jax.Array
    solute_mass: jax.Array
    residual_volume: jax.Array
    diameter: jax.Array
    temperature: jax.Array
    weight: jax.Array
    sgs_u: jax.Array
    sgs_v: jax.Array
    sgs_w: jax.Array
    parcel_id: jax.Array
    active: jax.Array


class SprayGasIncrements(NamedTuple):
    u: jax.Array
    v: jax.Array
    w: jax.Array
    theta: jax.Array
    qv: jax.Array


class SprayDiagnostics(NamedTuple):
    active_parcels: jax.Array
    liquid_mass: jax.Array
    evaporated_mass: jax.Array
    air_energy_loss: jax.Array
    net_radiative_energy: jax.Array


class SprayCoupledState(NamedTuple):
    flow: FlowState
    spray: SprayState


class _ExchangeAccumulator(NamedTuple):
    spray: SprayState
    vapor_mass: jax.Array
    gas_energy: jax.Array
    impulse_u: jax.Array
    impulse_v: jax.Array
    impulse_w: jax.Array
    evaporated_mass: jax.Array
    air_energy_loss: jax.Array
    net_radiative_energy: jax.Array


class _ParcelRates(NamedTuple):
    drag_rate: jax.Array
    mass_transfer_coefficient: jax.Array
    pressure: jax.Array
    ambient_mass_fraction: jax.Array
    heat_conductance: jax.Array
    gas_temperature: jax.Array
    radiative_power: jax.Array
    reynolds: jax.Array
    relative_humidity: jax.Array


class _ParcelAdvance(NamedTuple):
    u: jax.Array
    v: jax.Array
    w: jax.Array
    mass: jax.Array
    diameter: jax.Array
    temperature: jax.Array
    drag_delta_u: jax.Array
    drag_delta_v: jax.Array
    drag_delta_w: jax.Array
    convective_energy: jax.Array
    radiative_energy: jax.Array


def sample_diameters(
    config: SprayDPMConfig,
    key: jax.Array,
    count: int,
    dtype: jnp.dtype,
) -> jax.Array:
    """Sample the mass-based injection diameter distribution."""
    if config.diameter_distribution == "monodisperse":
        return jnp.full((count,), config.initial_diameter, dtype=dtype)
    if config.diameter_distribution == "rosin_rammler":
        uniform = jax.random.uniform(key, (count,), dtype=dtype)
        scale = jnp.asarray(config.initial_diameter, dtype=dtype)
        spread = jnp.asarray(config.rosin_rammler_spread, dtype=dtype)
        dmin = jnp.asarray(config.minimum_diameter, dtype=dtype)
        dmax = jnp.asarray(config.maximum_diameter, dtype=dtype)
        cdf_min = 1.0 - jnp.exp(-((dmin / scale) ** spread))
        cdf_max = 1.0 - jnp.exp(-((dmax / scale) ** spread))
        probability = cdf_min + uniform * (cdf_max - cdf_min)
        return scale * (-jnp.log1p(-probability)) ** (1.0 / spread)
    if config.diameter_distribution == "lognormal":
        normal = jax.random.normal(key, (count,), dtype=dtype)
        diameter = config.initial_diameter * jnp.exp(
            jnp.log(config.lognormal_geometric_stddev) * normal
        )
        return jnp.clip(
            diameter, config.minimum_diameter, config.maximum_diameter
        )
    fractions = jnp.asarray(config.tabulated_mass_fractions, dtype=dtype)
    cumulative = jnp.cumsum(fractions / jnp.sum(fractions))
    uniform = jax.random.uniform(key, (count,), dtype=dtype)
    indices = jnp.searchsorted(cumulative, uniform, side="right")
    diameters = jnp.asarray(config.tabulated_diameters, dtype=dtype)
    return diameters[jnp.minimum(indices, diameters.size - 1)]


def initialize_spray(
    config: SprayDPMConfig,
    *,
    dtype: jnp.dtype = jnp.float32,
    seed: int = 0,
) -> SprayState:
    """Create a fixed-capacity parcel buffer with a circular initial injection."""
    count = config.max_parcels
    active = jnp.arange(count) < config.initial_parcels
    key_radius, key_angle, key_diameter = jax.random.split(
        jax.random.PRNGKey(seed), 3
    )
    radial = config.injection_radius * jnp.sqrt(
        jax.random.uniform(key_radius, (count,), dtype=dtype)
    )
    angle = 2.0 * jnp.pi * jax.random.uniform(key_angle, (count,), dtype=dtype)
    x = jnp.asarray(config.injection_x, dtype=dtype) + radial * jnp.cos(angle)
    y = jnp.asarray(config.injection_y, dtype=dtype) + radial * jnp.sin(angle)
    def fill(value: float) -> jax.Array:
        return jnp.full((count,), value, dtype=dtype)
    diameter = sample_diameters(config, key_diameter, count, dtype)
    mass = (jnp.pi / 6.0) * config.liquid_density * diameter**3
    solute_mass = config.salinity_mass_fraction * mass
    residual_volume = jnp.maximum(
        (jnp.pi / 6.0) * diameter**3
        - (mass - solute_mass) / config.water_density,
        0.0,
    )
    return SprayState(
        x=x,
        y=y,
        z=fill(config.injection_z),
        u=fill(config.injection_u),
        v=fill(config.injection_v),
        w=fill(config.injection_w),
        mass=mass,
        solute_mass=solute_mass,
        residual_volume=residual_volume,
        diameter=diameter,
        temperature=fill(config.initial_temperature),
        weight=fill(config.parcel_weight),
        sgs_u=fill(0.0),
        sgs_v=fill(0.0),
        sgs_w=fill(0.0),
        parcel_id=jnp.arange(count, dtype=jnp.uint32),
        active=active,
    )


def inject_spray(
    spray: SprayState,
    step: jax.Array,
    dt_physical: float,
    config: SprayDPMConfig,
) -> SprayState:
    """Fill inactive slots with a fixed number of continuously injected parcels."""
    count = config.parcels_per_step
    if count == 0:
        return spray
    available = jnp.sum(~spray.active)
    slots = jnp.argsort(spray.active.astype(jnp.int32))[:count]
    slot_valid = jnp.arange(count) < available
    time = step.astype(spray.x.dtype) * dt_physical
    time_active = (time >= config.injection_start_time) & (
        time < config.injection_end_time
    )
    slot_valid = slot_valid & time_active

    key = jax.random.fold_in(jax.random.PRNGKey(config.random_seed), step)
    key_radius, key_angle, key_diameter = jax.random.split(key, 3)
    radial = config.injection_radius * jnp.sqrt(
        jax.random.uniform(key_radius, (count,), dtype=spray.x.dtype)
    )
    angle = 2.0 * jnp.pi * jax.random.uniform(
        key_angle, (count,), dtype=spray.x.dtype
    )
    diameter = sample_diameters(config, key_diameter, count, spray.x.dtype)
    mass = (jnp.pi / 6.0) * config.liquid_density * diameter**3
    solute_mass = config.salinity_mass_fraction * mass
    residual_volume = jnp.maximum(
        (jnp.pi / 6.0) * diameter**3
        - (mass - solute_mass) / config.water_density,
        0.0,
    )
    if config.mass_flow_rate > 0.0:
        parcel_weight = config.mass_flow_rate * dt_physical / (count * mass)
    else:
        parcel_weight = jnp.full_like(mass, config.parcel_weight)
    parcel_id = (
        jnp.asarray(config.initial_parcels, dtype=jnp.uint32)
        + step.astype(jnp.uint32) * jnp.asarray(count, dtype=jnp.uint32)
        + jnp.arange(count, dtype=jnp.uint32)
    )

    def assign(field: jax.Array, values: jax.Array) -> jax.Array:
        old = field[slots]
        return field.at[slots].set(jnp.where(slot_valid, values, old))

    return SprayState(
        x=assign(spray.x, config.injection_x + radial * jnp.cos(angle)),
        y=assign(spray.y, config.injection_y + radial * jnp.sin(angle)),
        z=assign(spray.z, jnp.full((count,), config.injection_z, dtype=spray.z.dtype)),
        u=assign(spray.u, jnp.full((count,), config.injection_u, dtype=spray.u.dtype)),
        v=assign(spray.v, jnp.full((count,), config.injection_v, dtype=spray.v.dtype)),
        w=assign(spray.w, jnp.full((count,), config.injection_w, dtype=spray.w.dtype)),
        mass=assign(spray.mass, mass.astype(spray.mass.dtype)),
        solute_mass=assign(
            spray.solute_mass, solute_mass.astype(spray.solute_mass.dtype)
        ),
        residual_volume=assign(
            spray.residual_volume,
            residual_volume.astype(spray.residual_volume.dtype),
        ),
        diameter=assign(spray.diameter, diameter.astype(spray.diameter.dtype)),
        temperature=assign(
            spray.temperature,
            jnp.full((count,), config.initial_temperature, dtype=spray.temperature.dtype),
        ),
        weight=assign(
            spray.weight,
            jnp.asarray(parcel_weight, dtype=spray.weight.dtype),
        ),
        sgs_u=assign(spray.sgs_u, jnp.zeros((count,), dtype=spray.sgs_u.dtype)),
        sgs_v=assign(spray.sgs_v, jnp.zeros((count,), dtype=spray.sgs_v.dtype)),
        sgs_w=assign(spray.sgs_w, jnp.zeros((count,), dtype=spray.sgs_w.dtype)),
        parcel_id=assign(spray.parcel_id, parcel_id),
        active=spray.active.at[slots].set(spray.active[slots] | slot_valid),
    )


def saturation_vapor_pressure(temperature: jax.Array) -> jax.Array:
    """Bolton warm-water saturation pressure in Pa."""
    temperature_c = temperature - 273.15
    return 611.2 * jnp.exp(17.67 * temperature_c / (temperature_c + 243.5))


def _hydrostatic_exner_and_pressure(
    z: jax.Array,
    params: Params,
    config: SprayDPMConfig,
) -> tuple[jax.Array, jax.Array]:
    kappa = config.dry_air_gas_constant / config.air_heat_capacity
    surface_exner = (params.surface_pressure / 100000.0) ** kappa
    exner = surface_exner - params.g * z / (
        config.air_heat_capacity * params.theta0
    )
    exner = jnp.maximum(exner, 0.05)
    pressure = 100000.0 * exner ** (1.0 / kappa)
    return exner, pressure


def _cic_coordinates(
    x: jax.Array,
    y: jax.Array,
    z: jax.Array,
    params: Params,
    *,
    z_offset: float = 0.5,
) -> tuple[jax.Array, ...]:
    dx = params.dx * params.z_i
    dy = params.dy * params.z_i
    dz = params.dz * params.z_i
    lx = params.lx * params.z_i
    ly = params.ly * params.z_i
    x_index = jnp.mod(x, lx) / dx
    y_index = jnp.mod(y, ly) / dy
    z_index = jnp.clip(z / dz - z_offset, 0.0, params.nz - 1.0)
    ix0_raw = jnp.floor(x_index).astype(jnp.int32)
    iy0_raw = jnp.floor(y_index).astype(jnp.int32)
    iz0 = jnp.floor(z_index).astype(jnp.int32)
    ix0 = jnp.mod(ix0_raw, params.nx)
    iy0 = jnp.mod(iy0_raw, params.ny)
    ix1 = jnp.mod(ix0 + 1, params.nx)
    iy1 = jnp.mod(iy0 + 1, params.ny)
    iz1 = jnp.minimum(iz0 + 1, params.nz - 1)
    fx = x_index - ix0_raw
    fy = y_index - iy0_raw
    fz = jnp.where(iz1 > iz0, z_index - iz0, 0.0)
    return ix0, ix1, iy0, iy1, iz0, iz1, fx, fy, fz


def _cic_sample(field: jax.Array, coords: tuple[jax.Array, ...]) -> jax.Array:
    ix0, ix1, iy0, iy1, iz0, iz1, fx, fy, fz = coords
    result = jnp.zeros_like(fx, dtype=field.dtype)
    for ix, wx in ((ix0, 1.0 - fx), (ix1, fx)):
        for iy, wy in ((iy0, 1.0 - fy), (iy1, fy)):
            for iz, wz in ((iz0, 1.0 - fz), (iz1, fz)):
                result = result + wx * wy * wz * field[ix, iy, iz]
    return result


def _cic_min_sample(
    field: jax.Array, coords: tuple[jax.Array, ...]
) -> jax.Array:
    """Return the minimum over CIC cells with nonzero parcel support."""
    ix0, ix1, iy0, iy1, iz0, iz1, fx, fy, fz = coords
    result = jnp.full_like(fx, jnp.inf, dtype=field.dtype)
    for ix, wx in ((ix0, 1.0 - fx), (ix1, fx)):
        for iy, wy in ((iy0, 1.0 - fy), (iy1, fy)):
            for iz, wz in ((iz0, 1.0 - fz), (iz1, fz)):
                weight = wx * wy * wz
                candidate = jnp.where(weight > 0.0, field[ix, iy, iz], jnp.inf)
                result = jnp.minimum(result, candidate)
    return result


def _cic_deposit(
    values: jax.Array,
    coords: tuple[jax.Array, ...],
    shape: tuple[int, int, int],
    dtype: jnp.dtype,
) -> jax.Array:
    ix0, ix1, iy0, iy1, iz0, iz1, fx, fy, fz = coords
    result = jnp.zeros(shape, dtype=dtype)
    for ix, wx in ((ix0, 1.0 - fx), (ix1, fx)):
        for iy, wy in ((iy0, 1.0 - fy), (iy1, fy)):
            for iz, wz in ((iz0, 1.0 - fz), (iz1, fz)):
                result = result.at[ix, iy, iz].add(values * wx * wy * wz)
    return result


def _vertical_test_filter(q: jax.Array) -> jax.Array:
    lower = jnp.concatenate((q[:, :, :1], q[:, :, :-1]), axis=2)
    upper = jnp.concatenate((q[:, :, 1:], q[:, :, -1:]), axis=2)
    return 0.25 * lower + 0.5 * q + 0.25 * upper


def _sgs_velocity_statistics(
    u: jax.Array,
    v: jax.Array,
    w_center: jax.Array,
    params: Params,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Estimate unresolved component variances and correlation time.

    The resolved band between the LES and test filters is extrapolated below
    the LES cutoff with inertial-range ``k^(-5/3)`` scaling. No fitted spray
    dispersion coefficient is introduced.
    """
    filter_width = params.fgr * params.tfr
    variance_scale = 1.0 / (params.tfr ** (2.0 / 3.0) - 1.0)

    def component_variance(q: jax.Array) -> jax.Array:
        mean = _vertical_test_filter(
            _spectral_box_filter(q, params, filter_width)
        )
        mean_square = _vertical_test_filter(
            _spectral_box_filter(q * q, params, filter_width)
        )
        return jnp.maximum(
            variance_scale * (mean_square - mean * mean), 0.0
        ).astype(params.dtype)

    variance_u = component_variance(u)
    variance_v = component_variance(v)
    variance_w = component_variance(w_center)
    k_sgs = 0.5 * (variance_u + variance_v + variance_w)
    velocity_scale = jnp.sqrt(jnp.maximum(2.0 * k_sgs / 3.0, 0.0))
    delta_physical = jnp.asarray(
        params.sgs_delta * params.z_i, dtype=params.dtype
    )
    time_scale = delta_physical / jnp.maximum(
        velocity_scale, jnp.asarray(1.0e-6, dtype=params.dtype)
    )
    return variance_u, variance_v, variance_w, time_scale


def _eddy_crossing_limited_time_scale(
    spray: SprayState,
    gas_u: jax.Array,
    gas_v: jax.Array,
    gas_w: jax.Array,
    time_scale: jax.Array,
    params: Params,
) -> jax.Array:
    """Limit the SGS correlation time by the parcel eddy-crossing time.

    The slip uses the correlated SGS velocity already seen by the parcel.  It
    therefore represents the carrier velocity at the start of the OU update
    and does not introduce an implicit dependence on the newly drawn noise.
    """
    slip_u = gas_u + spray.sgs_u - spray.u
    slip_v = gas_v + spray.sgs_v - spray.v
    slip_w = gas_w + spray.sgs_w - spray.w
    slip_speed = jnp.sqrt(slip_u**2 + slip_v**2 + slip_w**2)
    delta_physical = jnp.asarray(
        params.sgs_delta * params.z_i, dtype=spray.x.dtype
    )
    crossing_time = jnp.where(
        slip_speed > 0.0,
        delta_physical / slip_speed,
        jnp.asarray(jnp.inf, dtype=spray.x.dtype),
    )
    return jnp.minimum(time_scale, crossing_time)


def _advance_sgs_velocity_seen(
    spray: SprayState,
    variance_u: jax.Array,
    variance_v: jax.Array,
    variance_w: jax.Array,
    time_scale: jax.Array,
    dt: float,
    counter: jax.Array,
    config: SprayDPMConfig,
) -> SprayState:
    """Advance a correlated OU reconstruction of SGS velocity seen."""
    time_scale = jnp.maximum(
        time_scale, jnp.asarray(dt * 1.0e-6, dtype=spray.x.dtype)
    )
    alpha = jnp.exp(-jnp.asarray(dt, spray.x.dtype) / time_scale)
    noise_scale = jnp.sqrt(jnp.maximum(1.0 - alpha * alpha, 0.0))
    base_key = jax.random.fold_in(
        jax.random.PRNGKey(config.random_seed), counter.astype(jnp.uint32)
    )

    def parcel_normal(parcel_id: jax.Array) -> jax.Array:
        key = jax.random.fold_in(base_key, parcel_id.astype(jnp.uint32))
        return jax.random.normal(key, (3,), dtype=spray.x.dtype)

    normal = jax.vmap(parcel_normal)(spray.parcel_id)
    active = spray.active.astype(spray.x.dtype)
    sgs_u = active * (
        alpha * spray.sgs_u
        + noise_scale * jnp.sqrt(jnp.maximum(variance_u, 0.0)) * normal[:, 0]
    )
    sgs_v = active * (
        alpha * spray.sgs_v
        + noise_scale * jnp.sqrt(jnp.maximum(variance_v, 0.0)) * normal[:, 1]
    )
    sgs_w = active * (
        alpha * spray.sgs_w
        + noise_scale * jnp.sqrt(jnp.maximum(variance_w, 0.0)) * normal[:, 2]
    )
    return spray._replace(sgs_u=sgs_u, sgs_v=sgs_v, sgs_w=sgs_w)


def _parcel_rates_from_samples(
    spray: SprayState,
    gas_u: jax.Array,
    gas_v: jax.Array,
    gas_w: jax.Array,
    theta: jax.Array,
    qv: jax.Array,
    params: Params,
    config: SprayDPMConfig,
) -> _ParcelRates:
    """Evaluate common parcel physics from carrier samples at parcel locations."""
    exner, pressure = _hydrostatic_exner_and_pressure(spray.z, params, config)
    gas_temperature = theta * exner

    rel_u = gas_u - spray.u
    rel_v = gas_v - spray.v
    rel_w = gas_w - spray.w
    rel_speed = jnp.sqrt(rel_u**2 + rel_v**2 + rel_w**2)
    diameter = jnp.maximum(spray.diameter, config.min_diameter)
    drop_volume = jnp.maximum(
        (jnp.pi / 6.0) * diameter**3,
        jnp.asarray(1.0e-30, dtype=diameter.dtype),
    )
    particle_density = jnp.maximum(
        spray.mass / drop_volume,
        jnp.asarray(config.air_density, dtype=diameter.dtype),
    )
    reynolds = (
        config.air_density * rel_speed * diameter / config.air_dynamic_viscosity
    )
    def clift_gauvin(reynolds_number: jax.Array) -> jax.Array:
        safe_reynolds = jnp.maximum(reynolds_number, 1.0e-12)
        turbulent_drag = 0.42 / (
            1.0 + 42500.0 / safe_reynolds**1.16
        )
        return (
            1.0
            + 0.15 * safe_reynolds**0.687
            + turbulent_drag * safe_reynolds / 24.0
        )

    if config.drag_correction_model == "terminal_settling":
        stokes_time = (
            particle_density
            * diameter**2
            / (18.0 * config.air_dynamic_viscosity)
        )
        gravity = jnp.asarray(params.g, dtype=diameter.dtype) * (
            1.0 - config.air_density / particle_density
        )

        def terminal_iteration(_, settling_speed):
            terminal_reynolds = (
                config.air_density
                * settling_speed
                * diameter
                / config.air_dynamic_viscosity
            )
            return gravity * stokes_time / clift_gauvin(terminal_reynolds)

        settling_speed = lax.fori_loop(
            0, 8, terminal_iteration, gravity * stokes_time
        )
        terminal_reynolds = (
            config.air_density
            * settling_speed
            * diameter
            / config.air_dynamic_viscosity
        )
        drag_factor = clift_gauvin(terminal_reynolds)
    else:
        drag_factor = clift_gauvin(reynolds)
    drag_rate = (
        18.0
        * config.air_dynamic_viscosity
        * drag_factor
        / (particle_density * diameter**2)
    )
    area = jnp.pi * diameter**2
    ambient_mass_fraction = qv / (1.0 + qv)
    epsilon = config.dry_air_gas_constant / config.water_vapor_gas_constant
    ambient_vapor_pressure = pressure * qv / jnp.maximum(epsilon + qv, 1.0e-12)
    relative_humidity = ambient_vapor_pressure / jnp.maximum(
        saturation_vapor_pressure(gas_temperature), 1.0e-12
    )

    if config.thermodynamic_transfer_model == "veron2020":
        radius = 0.5 * diameter
        ventilation = jnp.where(
            config.ventilation_correction_enabled,
            1.0 + 0.25 * jnp.sqrt(reynolds),
            1.0,
        )
        vapor_denominator = (
            radius / (radius + config.vapor_jump_length)
            + config.vapor_diffusivity
            / (radius * config.vapor_accommodation_coefficient)
            * jnp.sqrt(
                2.0
                * jnp.pi
                * config.water_molar_mass
                / (config.universal_gas_constant * gas_temperature)
            )
        )
        modified_vapor_diffusivity = (
            ventilation * config.vapor_diffusivity / vapor_denominator
        )
        thermal_denominator = (
            radius / (radius + config.thermal_jump_length)
            + config.air_thermal_conductivity
            / (
                radius
                * config.thermal_accommodation_coefficient
                * config.air_density
                * config.air_heat_capacity
            )
            * jnp.sqrt(
                2.0
                * jnp.pi
                * config.dry_air_molar_mass
                / (config.universal_gas_constant * gas_temperature)
            )
        )
        modified_thermal_conductivity = (
            ventilation * config.air_thermal_conductivity / thermal_denominator
        )
        heat_conductance = (
            4.0 * jnp.pi * radius * modified_thermal_conductivity
        )
        mass_transfer_coefficient = (
            4.0
            * jnp.pi
            * radius
            * modified_vapor_diffusivity
            * config.water_molar_mass
            * saturation_vapor_pressure(gas_temperature)
            / (config.universal_gas_constant * gas_temperature)
        )
    else:
        prandtl = (
            config.air_heat_capacity
            * config.air_dynamic_viscosity
            / config.air_thermal_conductivity
        )
        schmidt = config.air_dynamic_viscosity / (
            config.air_density * config.vapor_diffusivity
        )
        nusselt = jnp.where(
            config.ventilation_correction_enabled,
            2.0 + 0.6 * jnp.sqrt(reynolds) * prandtl ** (1.0 / 3.0),
            2.0,
        )
        sherwood = jnp.where(
            config.ventilation_correction_enabled,
            2.0 + 0.6 * jnp.sqrt(reynolds) * schmidt ** (1.0 / 3.0),
            2.0,
        )
        heat_transfer = nusselt * config.air_thermal_conductivity / diameter
        heat_conductance = heat_transfer * area
        mass_transfer_coefficient = (
            jnp.pi
            * diameter
            * config.air_density
            * config.vapor_diffusivity
            * sherwood
        )

    shortwave_power = (
        0.25
        * area
        * config.shortwave_absorption_efficiency
        * config.shortwave_flux
    )
    longwave_power = (
        area
        * config.liquid_emissivity
        * config.stefan_boltzmann
        * (
            jnp.asarray(config.sky_temperature, dtype=spray.temperature.dtype)
            ** 4
            - spray.temperature**4
        )
    )
    radiative_power = shortwave_power + longwave_power
    active = spray.active.astype(spray.x.dtype)
    return _ParcelRates(
        drag_rate=active * drag_rate,
        mass_transfer_coefficient=active * mass_transfer_coefficient,
        pressure=pressure,
        ambient_mass_fraction=ambient_mass_fraction,
        heat_conductance=active * heat_conductance,
        gas_temperature=gas_temperature,
        radiative_power=active * radiative_power,
        reynolds=reynolds,
        relative_humidity=relative_humidity,
    )


def _salt_osmotic_coefficient(
    molality: jax.Array, config: SprayDPMConfig
) -> jax.Array:
    """Return the practical NaCl osmotic coefficient.

    Andreas (1989), equation (27), fits Low's measured water activities for
    0 <= molality <= 6 mol/kg.  Clipping at the validity boundary avoids
    extrapolating the polynomial into supersaturated solutions, which require
    an explicit crystallization model.
    """
    if config.osmotic_coefficient_model == "constant":
        return jnp.full_like(molality, config.salt_osmotic_coefficient)
    m = jnp.clip(molality, 0.0, 6.0)
    return (
        0.9270
        - 2.164e-2 * m
        + 3.486e-2 * m**2
        - 5.956e-3 * m**3
        + 3.911e-4 * m**4
    )


def _phase_change_rate_at_temperature(
    spray: SprayState,
    temperature: jax.Array,
    rates: _ParcelRates,
    config: SprayDPMConfig,
) -> jax.Array:
    """Return signed liquid-to-vapour transfer rate at a trial temperature."""
    water_mass = jnp.maximum(
        spray.mass - spray.solute_mass,
        jnp.asarray(1.0e-30, dtype=temperature.dtype),
    )
    molality = spray.solute_mass / (config.salt_molar_mass * water_mass)
    osmotic_coefficient = _salt_osmotic_coefficient(molality, config)
    solute_exponent = (
        config.salt_vant_hoff_factor
        * osmotic_coefficient
        * molality
        * config.water_molar_mass
    )
    water_activity = jnp.exp(-jnp.minimum(solute_exponent, 80.0))
    radius = jnp.maximum(
        0.5 * spray.diameter,
        jnp.asarray(0.5 * config.min_diameter, dtype=temperature.dtype),
    )
    kelvin_exponent = (
        2.0
        * config.water_molar_mass
        * config.surface_tension
        / (
            config.universal_gas_constant
            * config.water_density
            * radius
            * temperature
        )
    )
    if config.thermodynamic_transfer_model == "veron2020":
        temperature_exponent = (
            config.latent_heat
            * config.water_molar_mass
            / config.universal_gas_constant
            * (1.0 / rates.gas_temperature - 1.0 / temperature)
        )
        surface_relative_humidity = (
            rates.gas_temperature
            / temperature
            * jnp.exp(
                jnp.clip(
                    temperature_exponent
                    + kelvin_exponent
                    - solute_exponent,
                    -80.0,
                    20.0,
                )
            )
        )
        return rates.mass_transfer_coefficient * (
            surface_relative_humidity - rates.relative_humidity
        )

    vapor_pressure_surface = jnp.minimum(
        saturation_vapor_pressure(temperature)
        * water_activity
        * jnp.exp(jnp.minimum(kelvin_exponent, 20.0)),
        0.99 * rates.pressure,
    )
    epsilon = config.dry_air_gas_constant / config.water_vapor_gas_constant
    surface_mixing_ratio = (
        epsilon
        * vapor_pressure_surface
        / (rates.pressure - vapor_pressure_surface)
    )
    surface_mass_fraction = surface_mixing_ratio / (
        1.0 + surface_mixing_ratio
    )
    spalding = (
        surface_mass_fraction - rates.ambient_mass_fraction
    ) / jnp.maximum(1.0 - surface_mass_fraction, 1.0e-8)
    return rates.mass_transfer_coefficient * jnp.log1p(
        jnp.maximum(
            spalding,
            jnp.asarray(-1.0 + 1.0e-7, dtype=spalding.dtype),
        )
    )


def _phase_change_at_temperature(
    spray: SprayState,
    rates: _ParcelRates,
    temperature: jax.Array,
    dt: float,
    config: SprayDPMConfig,
) -> jax.Array:
    unconstrained = _phase_change_rate_at_temperature(
        spray, temperature, rates, config
    ) * jnp.asarray(dt, spray.mass.dtype)
    available_water = jnp.maximum(spray.mass - spray.solute_mass, 0.0)
    evaporation = jnp.minimum(
        jnp.maximum(unconstrained, 0.0), available_water
    )
    condensation = jnp.minimum(unconstrained, 0.0)
    return jnp.where(spray.active, evaporation + condensation, 0.0)


def _proposed_phase_change(
    spray: SprayState,
    rates: _ParcelRates,
    dt: float,
    config: SprayDPMConfig,
) -> jax.Array:
    """Solve the local implicit temperature/phase-change balance.

    A fixed-count vectorized bisection avoids explicit latent-heat oscillations
    for droplets whose thermal time scale is much shorter than the LES step.
    """
    if config.material == "nitrogen":
        # Nitrogen is injected close to its normal boiling point.  In this
        # reduced model it cannot condense from the ambient carrier: sensible
        # heating first brings a subcooled parcel to the boiling point, then
        # all remaining convective/radiative heat drives vaporization.
        dtype = spray.mass.dtype
        dt_value = jnp.asarray(dt, dtype=dtype)
        boiling = jnp.asarray(config.boiling_temperature, dtype=dtype)
        heat_available = jnp.maximum(
            dt_value
            * (
                rates.heat_conductance
                * (rates.gas_temperature - spray.temperature)
                + rates.radiative_power
            ),
            0.0,
        )
        sensible = (
            spray.mass
            * config.liquid_heat_capacity
            * jnp.maximum(boiling - spray.temperature, 0.0)
        )
        latent_energy = jnp.maximum(heat_available - sensible, 0.0)
        evaporation = jnp.minimum(
            spray.mass,
            latent_energy
            / jnp.asarray(config.latent_heat, dtype=dtype),
        )
        return jnp.where(spray.active, evaporation, 0.0)

    dt_value = jnp.asarray(dt, spray.mass.dtype)
    conductance_dt = dt_value * rates.heat_conductance
    radiative_energy = dt_value * rates.radiative_power

    def residual(temperature: jax.Array) -> jax.Array:
        phase_change = _phase_change_at_temperature(
            spray, rates, temperature, dt, config
        )
        new_mass = jnp.maximum(spray.mass - phase_change, 0.0)
        heat_capacity = jnp.maximum(
            0.5
            * (spray.mass + new_mass)
            * config.liquid_heat_capacity,
            jnp.asarray(1.0e-30, dtype=spray.mass.dtype),
        )
        return (
            (heat_capacity + conductance_dt) * temperature
            - heat_capacity * spray.temperature
            - conductance_dt * rates.gas_temperature
            - radiative_energy
            + config.latent_heat * phase_change
        )

    lower = jnp.maximum(
        jnp.minimum(spray.temperature, rates.gas_temperature) - 100.0,
        jnp.asarray(150.0, dtype=spray.temperature.dtype),
    )
    upper = jnp.minimum(
        jnp.maximum(spray.temperature, rates.gas_temperature) + 100.0,
        jnp.asarray(450.0, dtype=spray.temperature.dtype),
    )
    lower_residual = residual(lower)
    upper_residual = residual(upper)

    def bisect(_, bounds):
        low, high = bounds
        midpoint = 0.5 * (low + high)
        value = residual(midpoint)
        return (
            jnp.where(value <= 0.0, midpoint, low),
            jnp.where(value <= 0.0, high, midpoint),
        )

    lower, upper = lax.fori_loop(0, 16, bisect, (lower, upper))
    root = 0.5 * (lower + upper)
    bracketed = (lower_residual <= 0.0) & (upper_residual >= 0.0)
    endpoint = jnp.where(
        jnp.abs(lower_residual) <= jnp.abs(upper_residual),
        lower,
        upper,
    )
    temperature = jnp.where(bracketed, root, endpoint)
    return _phase_change_at_temperature(
        spray, rates, temperature, dt, config
    )


def _advance_parcel_implicit(
    spray: SprayState,
    gas_u: jax.Array,
    gas_v: jax.Array,
    gas_w: jax.Array,
    rates: _ParcelRates,
    phase_change: jax.Array,
    dt: float,
    params: Params,
    config: SprayDPMConfig,
) -> _ParcelAdvance:
    """Advance stiff drag and heat exchange with frozen substep coefficients.

    Drag is integrated analytically. Heat transfer is backward Euler in drop
    temperature, while radiation and the already limited phase change are
    integrated explicitly. ``phase_change`` is positive for evaporation and
    negative for condensation.
    """
    dtype = spray.mass.dtype
    dt_value = jnp.asarray(dt, dtype=dtype)
    drag_rate = jnp.maximum(rates.drag_rate, 0.0)
    relaxation = -jnp.expm1(-drag_rate * dt_value)
    response_integral = jnp.where(
        drag_rate > 0.0,
        relaxation / drag_rate,
        dt_value,
    )
    old_volume = jnp.maximum(
        (jnp.pi / 6.0) * spray.diameter**3,
        jnp.asarray(1.0e-30, dtype=dtype),
    )
    particle_density = jnp.maximum(
        spray.mass / old_volume,
        jnp.asarray(config.air_density, dtype=dtype),
    )
    gravity = jnp.asarray(params.g, dtype=dtype) * (
        1.0 - config.air_density / particle_density
    )
    new_u = spray.u + relaxation * (gas_u - spray.u)
    new_v = spray.v + relaxation * (gas_v - spray.v)
    new_w = (
        spray.w
        + relaxation * (gas_w - spray.w)
        - gravity * response_integral
    )

    # Exclude gravity from the reaction impulse returned to the carrier.
    drag_delta_u = new_u - spray.u
    drag_delta_v = new_v - spray.v
    drag_delta_w = new_w - spray.w + gravity * dt_value

    new_mass = jnp.maximum(spray.mass - phase_change, 0.0)
    new_volume = jnp.maximum(
        spray.residual_volume
        + jnp.maximum(new_mass - spray.solute_mass, 0.0)
        / config.water_density,
        jnp.asarray(1.0e-30, dtype=dtype),
    )
    new_diameter = (6.0 * new_volume / jnp.pi) ** (1.0 / 3.0)
    mean_mass = 0.5 * (spray.mass + new_mass)
    heat_capacity = jnp.maximum(
        mean_mass * config.liquid_heat_capacity,
        jnp.asarray(1.0e-30, dtype=dtype),
    )
    conductance_dt = dt_value * rates.heat_conductance
    radiative_energy = dt_value * rates.radiative_power
    if config.material == "nitrogen":
        boiling = jnp.asarray(config.boiling_temperature, dtype=dtype)
        raw_convective_energy = jnp.maximum(
            conductance_dt * (rates.gas_temperature - spray.temperature),
            0.0,
        )
        radiative_heating = jnp.maximum(radiative_energy, 0.0)
        sensible_needed = (
            spray.mass
            * config.liquid_heat_capacity
            * jnp.maximum(boiling - spray.temperature, 0.0)
        )
        sensible_energy = jnp.minimum(
            raw_convective_energy + radiative_heating,
            sensible_needed,
        )
        absorbed_energy = sensible_energy + config.latent_heat * phase_change
        convective_energy = jnp.minimum(
            raw_convective_energy,
            jnp.maximum(absorbed_energy - radiative_heating, 0.0),
        )
        new_temperature = jnp.minimum(
            spray.temperature
            + sensible_energy
            / jnp.maximum(
                spray.mass * config.liquid_heat_capacity,
                jnp.asarray(1.0e-30, dtype=dtype),
            ),
            boiling,
        )
        return _ParcelAdvance(
            u=new_u,
            v=new_v,
            w=new_w,
            mass=new_mass,
            diameter=new_diameter,
            temperature=new_temperature,
            drag_delta_u=drag_delta_u,
            drag_delta_v=drag_delta_v,
            drag_delta_w=drag_delta_w,
            convective_energy=convective_energy,
            radiative_energy=radiative_energy,
        )

    new_temperature = (
        heat_capacity * spray.temperature
        + conductance_dt * rates.gas_temperature
        + radiative_energy
        - config.latent_heat * phase_change
    ) / (heat_capacity + conductance_dt)
    convective_energy = conductance_dt * (
        rates.gas_temperature - new_temperature
    )
    return _ParcelAdvance(
        u=new_u,
        v=new_v,
        w=new_w,
        mass=new_mass,
        diameter=new_diameter,
        temperature=new_temperature,
        drag_delta_u=drag_delta_u,
        drag_delta_v=drag_delta_v,
        drag_delta_w=drag_delta_w,
        convective_energy=convective_energy,
        radiative_energy=radiative_energy,
    )


def _parcel_rates(
    spray: SprayState,
    flow: FlowState,
    params: Params,
    config: SprayDPMConfig,
) -> tuple[jax.Array, ...]:
    coords = _cic_coordinates(spray.x, spray.y, spray.z, params)
    rates = _parcel_rates_from_samples(
        spray,
        _cic_sample(flow.u, coords),
        _cic_sample(flow.v, coords),
        _cic_sample(upper_face_to_center(flow.w), coords),
        _cic_sample(flow.theta, coords),
        jnp.maximum(_cic_sample(flow.qv, coords), 0.0),
        params,
        config,
    )
    return (coords, *rates)


def spray_exchange(
    flow: FlowState,
    spray: SprayState,
    params: Params,
    config: SprayDPMConfig,
) -> tuple[SprayState, SprayGasIncrements, SprayDiagnostics]:
    """Advance parcels one carrier step and return conservative gas increments."""
    if not params.thermo_enabled or not params.moisture_enabled:
        raise ValueError("spray_dpm requires thermo_enabled and moisture_enabled")
    shape = (params.nx, params.ny, params.nz)
    zeros = jnp.zeros(shape, dtype=params.dtype)
    accumulator = _ExchangeAccumulator(
        spray=spray,
        vapor_mass=zeros,
        gas_energy=zeros,
        impulse_u=zeros,
        impulse_v=zeros,
        impulse_w=zeros,
        evaporated_mass=jnp.asarray(0.0, dtype=params.dtype),
        air_energy_loss=jnp.asarray(0.0, dtype=params.dtype),
        net_radiative_energy=jnp.asarray(0.0, dtype=params.dtype),
    )
    dt_sub = params.dt_physical / config.substeps
    domain_x = params.lx * params.z_i
    domain_y = params.ly * params.z_i
    domain_z = params.lz * params.z_i
    cell_mass = (
        config.air_density
        * params.dx
        * params.z_i
        * params.dy
        * params.z_i
        * params.dz
        * params.z_i
    )
    if config.turbulent_dispersion_enabled:
        sgs_statistics = _sgs_velocity_statistics(
            flow.u, flow.v, upper_face_to_center(flow.w), params
        )
    else:
        sgs_statistics = (zeros, zeros, zeros, jnp.ones_like(zeros))

    def substep(index: int, acc: _ExchangeAccumulator) -> _ExchangeAccumulator:
        current = acc.spray
        coords = _cic_coordinates(current.x, current.y, current.z, params)
        gas_u = _cic_sample(flow.u, coords)
        gas_v = _cic_sample(flow.v, coords)
        gas_w = _cic_sample(upper_face_to_center(flow.w), coords)
        effective_qv = jnp.maximum(
            flow.qv + acc.vapor_mass / cell_mass,
            jnp.asarray(params.qv_floor, dtype=params.dtype),
        )
        if config.turbulent_dispersion_enabled:
            sampled_statistics = tuple(
                _cic_sample(field, coords) for field in sgs_statistics
            )
            sampled_statistics = (
                *sampled_statistics[:3],
                _eddy_crossing_limited_time_scale(
                    current,
                    gas_u,
                    gas_v,
                    gas_w,
                    sampled_statistics[3],
                    params,
                ),
            )
            counter = (
                flow.step.astype(jnp.uint32) * config.substeps
                + jnp.asarray(index, dtype=jnp.uint32)
            )
            current = _advance_sgs_velocity_seen(
                current,
                *sampled_statistics,
                dt_sub,
                counter,
                config,
            )
        rates = _parcel_rates_from_samples(
            current,
            gas_u + current.sgs_u,
            gas_v + current.sgs_v,
            gas_w + current.sgs_w,
            _cic_sample(flow.theta, coords),
            _cic_sample(effective_qv, coords),
            params,
            config,
        )
        proposed_phase_change = _proposed_phase_change(
            current, rates, dt_sub, config
        )
        proposed_weighted_change = (
            current.weight
            * proposed_phase_change
            * current.active.astype(params.dtype)
        )
        proposed_vapor_mass = _cic_deposit(
            proposed_weighted_change, coords, shape, params.dtype
        )
        available_vapor_mass = jnp.maximum(
            effective_qv - params.qv_floor, 0.0
        ) * cell_mass
        condensation_ratio = jnp.where(
            proposed_vapor_mass < 0.0,
            available_vapor_mass
            / jnp.maximum(-proposed_vapor_mass, 1.0e-30),
            1.0,
        )
        condensation_scale = _cic_min_sample(
            jnp.clip(condensation_ratio, 0.0, 1.0), coords
        )
        phase_change = jnp.where(
            proposed_phase_change < 0.0,
            condensation_scale * proposed_phase_change,
            proposed_phase_change,
        )
        advanced = _advance_parcel_implicit(
            current,
            gas_u + current.sgs_u,
            gas_v + current.sgs_v,
            gas_w + current.sgs_w,
            rates,
            phase_change,
            dt_sub,
            params,
            config,
        )
        old_mass = current.mass
        new_x = jnp.mod(
            current.x + 0.5 * dt_sub * (current.u + advanced.u), domain_x
        )
        new_y = jnp.mod(
            current.y + 0.5 * dt_sub * (current.v + advanced.v), domain_y
        )
        new_z = current.z + 0.5 * dt_sub * (current.w + advanced.w)
        active = (
            current.active
            & (advanced.diameter >= config.min_diameter)
            & (new_z > 0.0)
            & (new_z < domain_z)
        )
        active_value = current.active.astype(params.dtype)
        weighted_evaporation = current.weight * phase_change * active_value
        weighted_energy = current.weight * advanced.convective_energy * active_value
        weighted_radiation = current.weight * advanced.radiative_energy * active_value
        weighted_mass = current.weight * old_mass * active_value
        impulse_u = -weighted_mass * advanced.drag_delta_u
        impulse_v = -weighted_mass * advanced.drag_delta_v
        impulse_w = -weighted_mass * advanced.drag_delta_w
        face_coords = _cic_coordinates(
            current.x, current.y, current.z, params, z_offset=1.0
        )

        updated = SprayState(
            x=new_x,
            y=new_y,
            z=new_z,
            u=jnp.where(current.active, advanced.u, current.u),
            v=jnp.where(current.active, advanced.v, current.v),
            w=jnp.where(current.active, advanced.w, current.w),
            mass=jnp.where(current.active, advanced.mass, current.mass),
            solute_mass=current.solute_mass,
            residual_volume=current.residual_volume,
            diameter=jnp.where(current.active, advanced.diameter, current.diameter),
            temperature=jnp.where(
                current.active, advanced.temperature, current.temperature
            ),
            weight=current.weight,
            sgs_u=current.sgs_u,
            sgs_v=current.sgs_v,
            sgs_w=current.sgs_w,
            parcel_id=current.parcel_id,
            active=active,
        )
        return _ExchangeAccumulator(
            spray=updated,
            vapor_mass=acc.vapor_mass + _cic_deposit(weighted_evaporation, coords, shape, params.dtype),
            gas_energy=acc.gas_energy - _cic_deposit(weighted_energy, coords, shape, params.dtype),
            impulse_u=acc.impulse_u + _cic_deposit(impulse_u, coords, shape, params.dtype),
            impulse_v=acc.impulse_v + _cic_deposit(impulse_v, coords, shape, params.dtype),
            impulse_w=acc.impulse_w
            + _cic_deposit(impulse_w, face_coords, shape, params.dtype),
            evaporated_mass=acc.evaporated_mass + jnp.sum(weighted_evaporation),
            air_energy_loss=acc.air_energy_loss + jnp.sum(weighted_energy),
            net_radiative_energy=acc.net_radiative_energy
            + jnp.sum(weighted_radiation),
        )

    accumulator = lax.fori_loop(0, config.substeps, substep, accumulator)
    dz = params.dz * params.z_i
    z_centers = (jnp.arange(params.nz, dtype=params.dtype) + 0.5) * dz
    exner, _ = _hydrostatic_exner_and_pressure(z_centers, params, config)
    increments = SprayGasIncrements(
        u=accumulator.impulse_u / cell_mass,
        v=accumulator.impulse_v / cell_mass,
        w=accumulator.impulse_w / cell_mass,
        theta=accumulator.gas_energy / (
            cell_mass * config.air_heat_capacity * exner[None, None, :]
        ),
        qv=accumulator.vapor_mass / cell_mass,
    )
    final_liquid_mass = jnp.sum(
        accumulator.spray.weight
        * accumulator.spray.mass
        * accumulator.spray.active
    )
    diagnostics = SprayDiagnostics(
        active_parcels=jnp.sum(accumulator.spray.active),
        liquid_mass=final_liquid_mass,
        evaporated_mass=accumulator.evaporated_mass,
        air_energy_loss=accumulator.air_energy_loss,
        net_radiative_energy=accumulator.net_radiative_energy,
    )
    return accumulator.spray, increments, diagnostics


def apply_spray_increments(
    flow: FlowState,
    increments: SprayGasIncrements,
    params: Params,
    ops: Operators | None = None,
) -> FlowState:
    """Apply DPM exchange to the carrier state, optionally projecting velocity."""
    u = flow.u + increments.u
    v = flow.v + increments.v
    w = flow.w + increments.w
    u, v, w = apply_velocity_bc(u, v, w, params)
    p = flow.p
    if ops is not None:
        u, v, w, p = project_velocity(u, v, w, params, ops)
        u, v, w = apply_velocity_bc(u, v, w, params)
    return flow._replace(
        u=u,
        v=v,
        w=w,
        p=p,
        theta=flow.theta + increments.theta,
        qv=apply_moisture_bounds(flow.qv + increments.qv, params),
    )


def step_spray_dpm(
    state: SprayCoupledState,
    params: Params,
    ops: Operators,
    config: SprayDPMConfig,
) -> tuple[SprayCoupledState, SprayDiagnostics]:
    """First-order partitioned DPM/LES step with projected two-way coupling."""
    injected = inject_spray(
        state.spray, state.flow.step, params.dt_physical, config
    )
    spray, increments, diagnostics = spray_exchange(
        state.flow, injected, params, config
    )
    forced_flow = apply_spray_increments(state.flow, increments, params, ops=None)
    advanced_flow = flow_step(forced_flow, params, ops)
    return SprayCoupledState(flow=advanced_flow, spray=spray), diagnostics


def run_spray_dpm(
    params: Params,
    config: SprayDPMConfig,
    *,
    seed: int = 0,
    log_every: int | None = None,
    log_callback: Callable[[FlowState, SprayDiagnostics], None] | None = None,
) -> tuple[SprayCoupledState, list[SprayDiagnostics]]:
    """Run the coupled single-process solver with optional JIT compilation."""
    ops = make_operators(params)
    state = SprayCoupledState(
        flow=initial_state(params, seed),
        spray=initialize_spray(config, dtype=params.dtype, seed=seed),
    )
    def step_fn(coupled: SprayCoupledState) -> tuple[SprayCoupledState, SprayDiagnostics]:
        return step_spray_dpm(coupled, params, ops, config)

    if params.use_jit:
        step_fn = jax.jit(step_fn)
    interval = params.c_count if log_every is None else log_every
    logs: list[SprayDiagnostics] = []
    for n in range(params.nsteps):
        state, spray_diag = step_fn(state)
        if (n + 1) % interval == 0:
            state, spray_diag = jax.block_until_ready((state, spray_diag))
            # Force the carrier diagnostic path as part of the coupled run so
            # divergence/CFL failures are visible to callers using callbacks.
            jax.block_until_ready(flow_diagnostics(state.flow, params, ops))
            logs.append(spray_diag)
            if log_callback is not None:
                log_callback(state.flow, spray_diag)
    return state, logs

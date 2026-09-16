"""Direct finite-volume HITSZ liquid-nitrogen jet simulation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import NamedTuple


SMAGORINSKY_COEFFICIENT = 0.16

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


@dataclass(frozen=True, slots=True)
class JetCase:
    cells: tuple[int, int, int]
    lengths: tuple[float, float, float]
    mapping_types: tuple[str, str, str]
    mapping_focus: tuple[float, float, float]
    mapping_strength: tuple[float, float, float]
    dt: float
    cfl: float | None
    steps: int
    chunk_steps: int
    checkpoint_every: int
    ambient_temperature: float
    ambient_streamwise_velocity: float
    ambient_relative_humidity: float
    ambient_water_vapor: float
    pressure: float
    ambient_density: float
    dry_air_density: float
    ambient_heat_capacity: float
    ambient_gas_constant: float
    kinematic_viscosity: float
    scalar_diffusivity: float
    scalar_advection_scheme: str
    nitrogen_buoyancy_coefficient: float
    gravity: tuple[float, float, float]
    roughness: float
    source_mode: str
    streamwise_boundaries: str
    fully_vaporized_within_source_cell: bool
    gas_inlet_radius: float
    gas_inlet_temperature: float
    subgrid_jet_enabled: bool
    subgrid_support_radius_cells: float
    subgrid_transition_width_cells: float
    subgrid_momentum_length_cells: float
    subgrid_turbulence_intensity: float
    subgrid_integral_scale_cells: float
    subgrid_correlation_time: float
    subgrid_transverse_ratio: float
    nozzle: tuple[float, float, float]
    radius: float
    speed: float
    mass_flow_rate: float
    vapor_quality: float
    jet_temperature: float
    liquid_density: float
    liquid_latent_heat: float
    ramp_time: float
    initial_diameter: float
    minimum_diameter: float
    maximum_diameter: float
    rosin_rammler_spread: float
    parcels_per_step: int
    maximum_parcels: int
    parcel_substeps: int
    cone_half_angle_degrees: float
    edge_speed_ratio: float
    edge_diameter_ratio: float
    profile_radius_m: tuple[float, ...]
    profile_axial_velocity_m_s: tuple[float, ...]
    profile_radial_velocity_m_s: tuple[float, ...]
    profile_d10_m: tuple[float, ...]
    output: Path
    gmg_tolerance: float
    gmg_presweeps: int
    gmg_postsweeps: int
    flow_formulation: str
    momentum_closure: str
    time_integration: str


def load_case(path: str | Path) -> JetCase:
    from .document import native_document
    document = native_document(path)
    domain = document["domain"]
    time_table = document["time"]
    ambient = document["ambient"]
    walls = document["walls"]
    jet = document["jet"]
    source = document.get("source", {})
    numerics = document["numerics"]
    output = document["output"]
    cells = tuple(int(value) for value in domain["cells"])
    lengths = tuple(float(value) for value in domain["lengths_m"])
    mapping_table = domain.get("mapping", {})
    mapping_types = tuple(
        str(value).replace("_", "-")
        for value in mapping_table.get("types", ("uniform",) * 3)
    )
    mapping_focus = tuple(
        float(value)
        for value in mapping_table.get(
            "focus_m", tuple(0.5 * length for length in lengths)
        )
    )
    mapping_strength = tuple(
        float(value)
        for value in mapping_table.get("strength", (0.0,) * 3)
    )
    if len(cells) != 3 or len(lengths) != 3 or min(cells) <= 0:
        raise ValueError("domain cells and lengths must contain three positive values")
    if not all(
        len(values) == 3
        for values in (mapping_types, mapping_focus, mapping_strength)
    ):
        raise ValueError("domain mapping entries must contain three values")
    case = JetCase(
        cells=cells,
        lengths=lengths,
        mapping_types=mapping_types,
        mapping_focus=mapping_focus,
        mapping_strength=mapping_strength,
        dt=float(time_table["dt_seconds"]),
        cfl=float(time_table["cfl"]) if "cfl" in time_table else None,
        steps=int(time_table["steps"]),
        chunk_steps=int(time_table["chunk_steps"]),
        checkpoint_every=int(time_table["checkpoint_every_steps"]),
        ambient_temperature=float(ambient["temperature_k"]),
        ambient_relative_humidity=float(ambient["relative_humidity"]),
        ambient_water_vapor=float(ambient["water_vapor_mixing_ratio"]),
        pressure=float(ambient["pressure_pa"]),
        ambient_density=float(ambient.get("density_kg_m3", 1.225)),
        dry_air_density=float(
            ambient.get(
                "dry_air_density_kg_m3",
                ambient.get("density_kg_m3", 1.225),
            )
        ),
        ambient_heat_capacity=float(
            ambient.get("heat_capacity_j_kg_k", 1005.0)
        ),
        ambient_gas_constant=float(
            ambient.get("gas_constant_j_kg_k", 287.05)
        ),
        kinematic_viscosity=float(
            ambient.get("kinematic_viscosity_m2_s", 1.5e-5)
        ),
        scalar_diffusivity=float(
            ambient.get("scalar_diffusivity_m2_s", 2.2e-5)
        ),
        scalar_advection_scheme=str(
            numerics.get("scalar_advection_scheme", "central")
        ).replace("_", "-"),
        nitrogen_buoyancy_coefficient=float(
            ambient.get("nitrogen_buoyancy_coefficient", 0.03398)
        ),
        gravity=tuple(
            float(value)
            for value in ambient.get("gravity_m_s2", (0.0, 0.0, -9.81))
        ),
        roughness=float(walls["roughness_length_m"]),
        source_mode=str(source.get("mode", "volume")),
        streamwise_boundaries=str(source.get("streamwise_boundaries", "inflow-outflow")),
        ambient_streamwise_velocity=float(ambient.get("streamwise_velocity_m_s", 0.0)),
        fully_vaporized_within_source_cell=bool(
            source.get("fully_vaporized_within_source_cell", False)
        ),
        gas_inlet_radius=float(
            source.get("gas_inlet_radius_m", jet["radius_m"])
        ),
        gas_inlet_temperature=float(
            source.get(
                "gas_inlet_temperature_k",
                jet.get("temperature_k", 77.34),
            )
        ),
        subgrid_jet_enabled=bool(source.get("subgrid_jet_enabled", False)),
        subgrid_support_radius_cells=float(
            source.get("subgrid_support_radius_cells", 2.5)
        ),
        subgrid_transition_width_cells=float(
            source.get("subgrid_transition_width_cells", 0.5)
        ),
        subgrid_momentum_length_cells=float(
            source.get("subgrid_momentum_length_cells", 4.0)
        ),
        subgrid_turbulence_intensity=float(
            source.get("subgrid_turbulence_intensity", 0.0)
        ),
        subgrid_integral_scale_cells=float(
            source.get("subgrid_integral_scale_cells", 4.0)
        ),
        subgrid_correlation_time=float(
            source.get("subgrid_correlation_time_seconds", 3.0e-4)
        ),
        subgrid_transverse_ratio=float(
            source.get("subgrid_transverse_ratio", 1.0)
        ),
        nozzle=tuple(float(value) for value in jet["position_m"]),
        radius=float(jet["radius_m"]),
        speed=float(jet["speed_m_s"]),
        mass_flow_rate=float(jet["mass_flow_rate_kg_s"]),
        vapor_quality=float(jet["vapor_quality"]),
        jet_temperature=float(jet.get("temperature_k", 77.34)),
        liquid_density=float(jet.get("liquid_density_kg_m3", 806.11)),
        liquid_latent_heat=float(
            jet.get("liquid_latent_heat_j_kg", 199_180.0)
        ),
        ramp_time=float(jet.get("ramp_time_seconds", 0.05)),
        initial_diameter=float(jet.get("initial_diameter_m", 150.0e-6)),
        minimum_diameter=float(jet.get("minimum_diameter_m", 50.0e-6)),
        maximum_diameter=float(jet.get("maximum_diameter_m", 300.0e-6)),
        rosin_rammler_spread=float(jet.get("rosin_rammler_spread", 3.0)),
        parcels_per_step=int(jet.get("parcels_per_step", 8)),
        maximum_parcels=int(jet.get("maximum_parcels", 16_384)),
        parcel_substeps=int(jet.get("parcel_substeps", 4)),
        cone_half_angle_degrees=float(jet.get("cone_half_angle_degrees", 0.0)),
        edge_speed_ratio=float(jet.get("edge_speed_ratio", 1.0)),
        edge_diameter_ratio=float(jet.get("edge_diameter_ratio", 1.0)),
        profile_radius_m=tuple(
            float(value) for value in jet.get("profile_radius_m", ())
        ),
        profile_axial_velocity_m_s=tuple(
            float(value)
            for value in jet.get("profile_axial_velocity_m_s", ())
        ),
        profile_radial_velocity_m_s=tuple(
            float(value)
            for value in jet.get("profile_radial_velocity_m_s", ())
        ),
        profile_d10_m=tuple(
            float(value) for value in jet.get("profile_d10_m", ())
        ),
        output=Path(output["directory"]),
        gmg_tolerance=float(numerics["gmg_tolerance"]),
        gmg_presweeps=int(numerics["gmg_presweeps"]),
        gmg_postsweeps=int(numerics["gmg_postsweeps"]),
        flow_formulation=str(numerics.get("flow_formulation", "low-mach")),
        momentum_closure=str(
            numerics.get("momentum_closure", "amd")
        ).replace("_", "-"),
        time_integration=str(numerics["time_integration"]),
    )
    finite_positive = (
        *case.lengths,
        case.dt,
        case.ambient_temperature,
        case.pressure,
        case.ambient_density,
        case.dry_air_density,
        case.ambient_heat_capacity,
        case.ambient_gas_constant,
        case.kinematic_viscosity,
        case.scalar_diffusivity,
        case.roughness,
        case.gas_inlet_radius,
        case.gas_inlet_temperature,
        case.subgrid_support_radius_cells,
        case.subgrid_transition_width_cells,
        case.subgrid_momentum_length_cells,
        case.subgrid_integral_scale_cells,
        case.subgrid_correlation_time,
        case.radius,
        case.speed,
        case.mass_flow_rate,
        case.jet_temperature,
        case.liquid_density,
        case.liquid_latent_heat,
        case.initial_diameter,
        case.minimum_diameter,
        case.maximum_diameter,
        case.rosin_rammler_spread,
        case.edge_speed_ratio,
        case.edge_diameter_ratio,
        case.gmg_tolerance,
    )
    if not all(math.isfinite(value) and value > 0.0 for value in finite_positive):
        raise ValueError("physical dimensions, timestep, and material values must be positive")
    if any(kind not in {"uniform", "tanh", "sinh"} for kind in case.mapping_types):
        raise ValueError("mapping types must be uniform, tanh, or sinh")
    if not all(
        math.isfinite(focus) and 0.0 <= focus <= length
        for focus, length in zip(case.mapping_focus, case.lengths)
    ):
        raise ValueError("mapping focus must lie inside each physical axis")
    if not all(
        math.isfinite(strength) and strength >= 0.0
        for strength in case.mapping_strength
    ):
        raise ValueError("mapping strength must be finite and nonnegative")
    if len(case.gravity) != 3 or not all(math.isfinite(value) for value in case.gravity):
        raise ValueError("gravity_m_s2 must contain three finite values")
    if not math.isfinite(case.ramp_time) or case.ramp_time < 0.0:
        raise ValueError("jet ramp time must be finite and nonnegative")
    if case.steps <= 0 or case.chunk_steps <= 0 or case.checkpoint_every <= 0:
        raise ValueError("steps, chunk size, and checkpoint interval must be positive")
    if case.cfl is not None:
        if not math.isfinite(case.cfl) or case.cfl <= 0.0:
            raise ValueError("jet CFL target must be finite and positive")
        if not case.fully_vaporized_within_source_cell or case.time_integration != "rk3":
            raise ValueError("adaptive jets currently require RK3 and the fully vaporized volume source")
    if case.parcels_per_step <= 0 or case.maximum_parcels < case.parcels_per_step:
        raise ValueError("parcel capacity must accommodate one injection")
    if case.parcel_substeps <= 0:
        raise ValueError("parcel substeps must be positive")
    if case.minimum_diameter >= case.maximum_diameter:
        raise ValueError("minimum droplet diameter must be below maximum")
    if not 0.0 <= case.cone_half_angle_degrees < 90.0:
        raise ValueError("cone half-angle must lie in [0, 90) degrees")
    if case.time_integration not in {"rk3", "fast-rk3"}:
        raise ValueError("time_integration must be .rk3. or .fast-rk3.")
    if case.scalar_advection_scheme not in {"central", "upwind"}:
        raise ValueError(
            "scalar_advection_scheme must be .central. or .upwind."
        )
    if case.flow_formulation not in {"low-mach", "incompressible"}:
        raise ValueError(
            "flow_formulation must be .low-mach. or .incompressible."
        )
    if case.momentum_closure not in {"amd", "classical-static-smagorinsky"}:
        raise ValueError(
            "momentum_closure must be .amd. or .classical-static-smagorinsky."
        )
    if case.source_mode not in {"volume", "inflow"}:
        raise ValueError("source mode must be .volume. or .inflow.")
    if not math.isfinite(case.ambient_streamwise_velocity) or case.ambient_streamwise_velocity < 0.0:
        raise ValueError("ambient streamwise_velocity_m_s must be finite and nonnegative")
    if case.ambient_streamwise_velocity and (
        case.source_mode != "volume" or case.streamwise_boundaries != "inflow-outflow"
    ):
        raise ValueError("ambient inflow requires a volume source and inflow-outflow boundaries")
    if case.streamwise_boundaries not in {"inflow-outflow", "outflow-outflow"}:
        raise ValueError("streamwise_boundaries must be inflow-outflow or outflow-outflow")
    if case.streamwise_boundaries == "outflow-outflow" and case.source_mode != "volume":
        raise ValueError("two streamwise outlets require an embedded volume source")
    if case.subgrid_jet_enabled and case.source_mode != "inflow":
        raise ValueError("the subgrid boundary jet requires source mode .inflow.")
    if case.fully_vaporized_within_source_cell and case.source_mode != "volume":
        raise ValueError(
            "fully vaporized within-source-cell closure requires volume source mode"
        )
    if not math.isfinite(case.subgrid_turbulence_intensity) or case.subgrid_turbulence_intensity < 0.0:
        raise ValueError("subgrid turbulence intensity must be finite and nonnegative")
    if not math.isfinite(case.subgrid_transverse_ratio) or case.subgrid_transverse_ratio < 0.0:
        raise ValueError("subgrid transverse ratio must be finite and nonnegative")
    if not 0.0 <= case.ambient_relative_humidity <= 1.0:
        raise ValueError("ambient relative humidity must lie in [0, 1]")
    if not 0.0 <= case.vapor_quality < 1.0:
        raise ValueError("vapor quality must lie in [0, 1)")
    profiles = (
        case.profile_radius_m,
        case.profile_axial_velocity_m_s,
        case.profile_radial_velocity_m_s,
        case.profile_d10_m,
    )
    if any(profiles) and not all(
        len(values) == len(case.profile_radius_m) for values in profiles
    ):
        raise ValueError("empirical LN2 source profiles must have equal lengths")
    inside = all(
        0.0 < coordinate < length
        for coordinate, length in zip(case.nozzle, case.lengths)
    )
    at_inflow = (
        case.source_mode == "inflow"
        and case.nozzle[0] == 0.0
        and 0.0 < case.nozzle[1] < case.lengths[1]
        and 0.0 < case.nozzle[2] < case.lengths[2]
    )
    if not (inside or at_inflow):
        raise ValueError(
            "the nozzle must lie inside the domain or on its inflow"
        )
    return case

"""Measured wind-tunnel challenge for the existing subgrid mist closure.

The entrained baseline and inertial investigation share the same moisture
physics. Physical source inputs are not fitted to the outlet temperatures.
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad

from jaxwind import (
    FREE_SLIP,
    OPEN,
    Boundaries,
    FlowModel,
    InflowPlane,
    LinearBoussinesqBuoyancy,
    PassiveScalar,
    StaggeredVelocity,
    Wall,
    build_open_atmospheric_step,
    build_pressure_poisson,
    courant_number,
    initial_atmospheric_solution,
)
from jaxwind.config.moisture import load_moisture
from jaxwind.config.spray_mesh import build_spray_grid
from jaxwind.moist_abl import build_moist_atmospheric_step, initialize_moisture
from jaxwind.physics.moisture import saturation_vapor_pressure_water
from jaxwind.sgs import AnisotropicMinimumDissipation, StaticSmagorinsky
from jaxwind.water_spray import build_water_injection

from .api import Simulation


def equivalent_diameter(reference):
    """D32 = 1 / integral (1/d) dF_mass for the truncated literature fit."""
    scale = reference["rosin_rammler_scale_m"]
    spread = reference["rosin_rammler_spread"]
    lo = reference["minimum_diameter_m"] / scale
    hi = reference["maximum_diameter_m"] / scale
    normalizer = np.exp(-(lo**spread)) - np.exp(-(hi**spread))
    inverse = quad(lambda x: spread * x ** (spread - 2) * np.exp(-(x**spread)), lo, hi)[
        0
    ]
    return scale * normalizer / inverse


def inlet_mixing_ratio(reference, config):
    """Ventilated psychrometer equation, with measured dry/wet bulbs in C.

    e = es(Tw) - 0.00066*(1+0.00115*Tw)*p*(Td-Tw).
    Its coefficient is an explicit reconstruction assumption, not measured RH.
    """
    td, tw = reference["dry_bulb_c"], reference["wet_bulb_c"]
    e = float(saturation_vapor_pressure_water(jnp.asarray(tw + 273.15)))
    e -= 0.00066 * (1 + 0.00115 * tw) * config.pressure * (td - tw)
    if not 0 < e < config.pressure:
        raise ValueError("invalid reconstructed inlet vapor pressure")
    return (
        config.dry_air_gas_constant
        / config.water_vapor_gas_constant
        * e
        / (config.pressure - e)
    )


def inlet_kinetic_energy(plane, mean_speed, area=None):
    """Cell-centered plane TKE after staggered interpolation."""
    u = plane.x_velocity - mean_speed
    v = 0.5 * (plane.y_velocity[:, :-1] + plane.y_velocity[:, 1:])
    w = 0.5 * (plane.z_velocity[:-1] + plane.z_velocity[1:])

    def mean(values):
        return (
            jnp.mean(values) if area is None else jnp.sum(values * area) / jnp.sum(area)
        )

    return 0.5 * jnp.asarray([mean(u * u), mean(v * v), mean(w * w)])


def build_benchmark_inlet(grid, mean_speed, uniform_plane, turbulence, duration):
    """Frozen isotropic turbulence with prescribed post-interpolation TKE.

    Grid interpolation, wall blocking and instantaneous bulk-flux correction
    remove energy. Normalize the whole vector by one scalar using the full box
    period, never using a temperature target. Two-point Gauss quadrature on
    each piecewise-linear advection interval integrates the squared velocity
    exactly. Tangential components have shifted x knots on the MAC mesh.
    """
    if turbulence is None:
        return lambda time: uniform_plane
    from jaxwind.inflow import build_mann_inflow, generate_mann_box

    intensity = turbulence["intensity"]
    length_scale = turbulence["length_scale_m"]
    if not 0 < intensity < 1 or not length_scale > 0:
        raise ValueError("invalid inlet turbulence intensity or scale")
    box_lengths = tuple(turbulence["box_lengths_m"])
    period = box_lengths[0] / mean_speed
    if period <= duration:
        raise ValueError("turbulence box must not repeat during benchmark")
    box = generate_mann_box(
        shape=tuple(turbulence["box_cells"]),
        lengths=box_lengths,
        length_scale=length_scale,
        gamma=0.0,
        seed=turbulence["seed"],
    )
    normalization = turbulence.get("normalization", "streamwise_rms")
    if normalization not in ("streamwise_rms", "total_energy"):
        raise ValueError("unknown inlet turbulence normalization")
    sampled = build_mann_inflow(
        box,
        grid,
        mean_speed=mean_speed,
        wall_y=True,
        sigma_u_profile=(
            mean_speed * intensity * np.sqrt(2 / 3)
            if normalization == "streamwise_rms"
            else None
        ),
    )

    area = (
        None
        if grid.is_uniform
        else jnp.asarray(grid.z_widths[:, None] * grid.y_widths[None, :])
    )

    def corrected(time):
        plane = sampled(time)
        return plane._replace(
            x_velocity=plane.x_velocity
            - (
                jnp.mean(plane.x_velocity)
                if area is None
                else jnp.sum(plane.x_velocity * area) / jnp.sum(area)
            )
            + mean_speed
        )

    if normalization == "streamwise_rms":
        # Preserve the original completed turbulence experiment for reproduction.
        return corrected
    nx = turbulence["box_cells"][0]
    nodes = jnp.asarray([0.5 - 0.5 / np.sqrt(3), 0.5 + 0.5 / np.sqrt(3)])
    times = ((jnp.arange(nx)[:, None] + nodes) * period / nx).reshape(-1)
    energies = jax.jit(
        jax.vmap(lambda t: inlet_kinetic_energy(corrected(t), mean_speed, area))
    )
    normal_energy = jnp.mean(energies(times)[:, 0])
    tangential_energy = jnp.sum(
        jnp.mean(energies(times + grid.x_centers[0] / mean_speed)[:, 1:], axis=0)
    )
    raw_energy = float(normal_energy + tangential_energy)
    if not np.isfinite(raw_energy) or raw_energy <= 0:
        raise ValueError("inlet turbulence has zero or invalid resolved energy")
    # The reference explicitly states k=(U I)^2, not 3/2*(U I)^2.
    scale = np.sqrt((mean_speed * intensity) ** 2 / raw_energy)

    def inlet(time):
        plane = corrected(time)
        return plane._replace(
            x_velocity=mean_speed + scale * (plane.x_velocity - mean_speed),
            y_velocity=scale * plane.y_velocity,
            z_velocity=scale * plane.z_velocity,
        )

    return inlet


def build_simulation(case):
    doc = case.document
    jax.config.update("jax_enable_x64", doc["numerics"]["dtype"] == "float64")
    grid = build_spray_grid(
        doc["mesh"], spray_model=doc["case"].get("spray_model", "entrained")
    )
    if (
        not grid.is_uniform
        and doc["numerics"].get("momentum_advection_scheme", "muscl-mc") != "central"
    ):
        raise ValueError(
            "stretched spray grids require explicit momentum_advection_scheme=central"
        )
    moist, spray = load_moisture(doc["physics"])
    config = moist.thermodynamics
    model = doc["case"].get("spray_model", "entrained")
    if model not in ("entrained", "inertial", "fluent-dpm"):
        raise ValueError(
            "benchmark spray_model must be entrained, inertial or fluent-dpm"
        )
    if (model == "fluent-dpm") != (spray.model == "fluent-dpm"):
        raise ValueError("case.spray_model and physics.water_spray.model disagree")
    carrier_turbulence = doc["case"].get("carrier_turbulence_model", "les")
    if carrier_turbulence not in ("les", "standard-k-epsilon", "realizable-k-epsilon"):
        raise ValueError(
            "carrier_turbulence_model must be les, standard-k-epsilon, or realizable-k-epsilon"
        )
    rans = carrier_turbulence != "les"
    physical_inlet = doc["case"].get("physical_transverse_inlet", False)
    if not isinstance(physical_inlet, bool):
        raise ValueError("physical_transverse_inlet must be boolean")
    if physical_inlet and (
        model != "inertial" or not grid.is_uniform
        or doc["case"].get("carrier_sgs_model", "amd") != "amd"
        or doc["numerics"].get("momentum_advection_scheme", "muscl-mc") != "muscl-mc"
        or not doc["case"].get("project_parcel_feedback", False)
    ):
        raise ValueError("physical_transverse_inlet requires uniform projected inertial MUSCL with AMD or transported RANS")
    scalar_closure = doc["case"].get("carrier_scalar_transport", "shared-diffusivity")
    if scalar_closure not in ("shared-diffusivity", "fluent-rans-defaults"):
        raise ValueError("unknown carrier_scalar_transport")
    fluent_scalar = scalar_closure == "fluent-rans-defaults"
    if fluent_scalar and (
        not rans or doc["numerics"].get("scalar_advection_scheme")
        not in ("muscl-mc", "upwind-ssprk3")
    ):
        raise ValueError("fluent-rans-defaults requires RANS with shared SSP scalar transport")
    if not rans and "turbulence_advection_scheme" in doc["numerics"]:
        raise ValueError("turbulence_advection_scheme requires a transported RANS model")
    if rans and (
        model != "inertial"
        or doc["case"].get("carrier_sgs_model", "amd") != "amd"
        or doc["case"].get("scalar_boundary") != "flux"
        or doc["case"].get("gas_wall_model") != "smooth-spalding"
        or not doc["case"].get("project_parcel_feedback", False)
        or doc["case"].get("inlet_turbulence") is None
    ):
        raise ValueError(
            "RANS control requires projected inertial flux baseline, wall model, inlet turbulence; no SGS override"
        )
    reference = doc["case"]["reference"]
    dtype = doc["numerics"]["dtype"]
    u = doc["physics"]["flow"]["streamwise_velocity_m_s"]
    shape = (grid.nz, grid.ny, grid.nx)
    velocity = StaggeredVelocity(
        jnp.full((grid.nz, grid.ny, grid.nx + 1), u, dtype),
        jnp.zeros((grid.nz, grid.ny + 1, grid.nx), dtype),
        jnp.zeros((grid.nz + 1, grid.ny, grid.nx), dtype),
    )
    flow = initial_atmospheric_solution(grid, velocity, dtype=dtype)
    initial, _ = initialize_moisture(flow, moist.temperature_offset_k, 0.0, config)
    qv = inlet_mixing_ratio(reference, config)
    initial = initial._replace(
        moisture=initial.moisture._replace(vapor=jnp.full(shape, qv, dtype))
    )
    ambient = jnp.full((grid.nz, grid.ny), qv, dtype)
    plane = InflowPlane(*(a[..., 0] for a in (*velocity, flow.scalar)))
    # Resolved wall shear is zero; an optional smooth-wall stress supplies the
    # unresolved momentum flux without double counting viscous/SGS wall shear.
    # Both options are adiabatic and retain the existing droplet-wall rule.
    boundaries = Boundaries(
        Wall(FREE_SLIP), Wall(FREE_SLIP), streamwise=OPEN, spanwise=FREE_SLIP
    )
    wall_model = doc["case"].get("gas_wall_model", "free-slip")
    if wall_model not in ("free-slip", "smooth-spalding"):
        raise ValueError("gas_wall_model must be free-slip or smooth-spalding")
    viscosity = doc["physics"]["flow"]["kinematic_viscosity_m2_s"]
    wall_force = None
    if wall_model == "smooth-spalding":
        from jaxwind.smooth_wall import smooth_duct_tendency

        if viscosity <= 0:
            raise ValueError("smooth gas walls require positive molecular viscosity")
        wall_force = lambda velocity, time: smooth_duct_tendency(
            velocity, grid, viscosity
        )
    subfilter = AnisotropicMinimumDissipation()
    carrier_sgs = doc["case"].get("carrier_sgs_model", "amd")
    if carrier_sgs not in (
        "amd",
        "smagorinsky",
        "wall-limited-smagorinsky",
        "frozen-inlet-rans",
    ):
        raise ValueError(
            "carrier_sgs_model must be amd, smagorinsky, wall-limited-smagorinsky, or frozen-inlet-rans"
        )
    if carrier_sgs != "amd":
        if model != "inertial":
            raise ValueError(
                "carrier_sgs_model override is only supported for inertial parcels"
            )
        from jaxwind.sgs import FluentSmagorinsky

        # Fixed existing static-model preset, selected before outlet comparison.
        # This is a closure control, not an outlet-calibrated diffusivity.
        subfilter = (
            StaticSmagorinsky()
            if carrier_sgs == "smagorinsky"
            else FluentSmagorinsky(coefficient=0.16, tunnel_walls=True)
        )
    if carrier_sgs == "frozen-inlet-rans":
        from jaxwind.sgs import ConstantEddyViscosity

        # Mechanism test only: paper Eqs. 1–2 and nu_t=C_mu*k^2/epsilon.
        # The inlet-derived value is frozen throughout the domain; no RANS
        # turbulence equations are solved and this is not a production closure.
        turbulence = doc["case"].get("inlet_turbulence")
        if turbulence is None:
            raise ValueError("frozen-inlet-rans requires prescribed inlet turbulence")
        nu_t = 0.09**0.25 * u * turbulence["intensity"] * turbulence["length_scale_m"]
        subfilter = ConstantEddyViscosity(nu_t)
    if model == "fluent-dpm" and spray.dpm.les_model == "fluent-smagorinsky":
        from jaxwind.sgs import FluentSmagorinsky

        subfilter = FluentSmagorinsky(
            coefficient=spray.dpm.smagorinsky_constant,
            von_karman=spray.dpm.von_karman,
            tunnel_walls=True,
        )
    momentum = FlowModel(
        forcing=wall_force,
        viscosity=doc["physics"]["flow"]["kinematic_viscosity_m2_s"],
        subfilter=subfilter,
        momentum_advection_scheme=doc["numerics"].get(
            "momentum_advection_scheme", "muscl-mc"
        ),
    )
    # Fluent12 defaults Pr_t=.85, Sc_t=.7 (Theory Guide 4.4.7).
    # Explicit comparison option: paper does not disclose these settings.
    from jaxwind.physics.moisture import WaterDropletProperties

    heat_diffusivity = WaterDropletProperties().air_thermal_conductivity / (
        config.dry_air_density * config.dry_air_heat_capacity
    )
    scalar = PassiveScalar(
        diffusivity=heat_diffusivity if fluent_scalar else config.vapor_diffusivity,
        turbulent_prandtl=0.85 if fluent_scalar else 0.7,
        advection_scheme="upwind",
    )
    pressure = build_pressure_poisson(
        grid,
        backend="gmg",
        periodic_x=False,
        periodic_y=False,
        dtype=dtype,
        config={"tolerance": 1.0e-7},
        physical_transverse_inlet=physical_inlet,
    )
    inlet_eddy = None
    if physical_inlet:
        from jaxwind.inlet_momentum import physical_inlet_eddy_viscosity
        inlet_eddy = lambda velocity, inflow: physical_inlet_eddy_viscosity(
            velocity, inflow, grid, boundaries, momentum.subfilter
        )
    scalar_scheme = doc["numerics"].get("scalar_advection_scheme", "upwind")
    if scalar_scheme not in {"upwind", "upwind-ssprk3", "muscl-mc"}:
        raise ValueError(
            "benchmark scalar_advection_scheme must be upwind, upwind-ssprk3, or muscl-mc"
        )
    split_scalar_scheme = (
        None
        if scalar_scheme == "upwind"
        else ("upwind" if scalar_scheme == "upwind-ssprk3" else "muscl-mc")
    )
    if (
        split_scalar_scheme is not None
        and doc["case"].get("scalar_boundary", "cell") != "flux"
    ):
        raise ValueError("SSP scalar transport requires case.scalar_boundary=flux")
    carrier = build_open_atmospheric_step(
        grid,
        boundaries,
        pressure,
        momentum,
        scalar,
        LinearBoussinesqBuoyancy(9.81 / moist.reference_temperature_k),
        scheme="fast-rk3",
        scalar_boundary=doc["case"].get("scalar_boundary", "cell"),
        transport_scalar=split_scalar_scheme is None and model != "fluent-dpm",
    )
    injection = build_water_injection(
        grid,
        (spray.streamwise_offset_m, grid.ly / 2, grid.lz / 2),
        spray.standard_deviation_m,
        spray.mass_flow_rate_kg_s,
        equivalent_diameter(reference),
        spray.ramp_time_s,
        config,
        dtype=dtype,
    )
    step = build_moist_atmospheric_step(
        carrier,
        grid,
        boundaries,
        momentum,
        scalar,
        config,
        moist.temperature_offset_k,
        moist.reference_temperature_k,
        ambient,
        injection,
        scalar_transport_scheme=split_scalar_scheme,
        eddy_viscosity_override=inlet_eddy,
    )
    if model == "fluent-dpm":
        from jaxwind.fluent_dpm_atmosphere import (
            build_dpm_atmospheric_step,
            initialize_dpm,
            specific_enthalpy,
        )
        from jaxwind.numerics.poisson import project
        from jaxwind.open_boundary import enforce_open_velocity
        from jaxwind.physics.fluent_dpm import DPMWaterMaterial
        from jaxwind.waterjet_observation import WaterjetObservation

        if scalar_scheme != "muscl-mc" or doc["case"].get("scalar_boundary") != "flux":
            raise ValueError(
                "DPM benchmark requires muscl-mc scalar transport and flux boundaries"
            )
        observation = WaterjetObservation.from_table(
            doc["case"].get("observation"), grid, spray.dpm
        )
        material = DPMWaterMaterial()
        initial = initialize_dpm(flow, moist, spray, material)
        initial = initial._replace(
            moisture=initial.moisture._replace(vapor=jnp.full(shape, qv, dtype)),
            enthalpy=jnp.full(
                shape,
                specific_enthalpy(moist.temperature_offset_k, qv, material),
                dtype,
            ),
        )
        project_velocity = lambda velocity, h, inflow: project(
            enforce_open_velocity(velocity, inflow, grid), pressure, h
        )[0]
        step = build_dpm_atmospheric_step(
            carrier,
            grid,
            boundaries,
            momentum,
            scalar,
            moist,
            spray,
            (spray.streamwise_offset_m, grid.ly / 2, grid.lz / 2),
            ambient,
            project_velocity,
            material=material,
        )
    if model == "inertial":
        from jaxwind.water_parcels import (
            InertialMoistAtmosphericSolution,
            WaterParcelSource,
            build_inertial_water_step,
            initial_water_parcels,
        )

        source = WaterParcelSource(
            center=(0.0, grid.ly / 2, grid.lz / 2),
            radius=reference["nozzle_diameter_m"] / 2,
            speed=0.9
            * np.sqrt(2 * reference["water_gauge_pressure_pa"] / config.water_density),
            temperature=reference["water_inlet_c"] + 273.15,
            mass_flow=spray.mass_flow_rate_kg_s,
            half_angle_degrees=reference["cone_half_angle_degrees"],
            diameter_scale=reference["rosin_rammler_scale_m"],
            diameter_spread=reference["rosin_rammler_spread"],
            diameter_minimum=reference["minimum_diameter_m"],
            diameter_maximum=reference["maximum_diameter_m"],
            count_per_step=doc["case"].get("parcels_per_step", 16),
            capacity=doc["case"].get("parcel_capacity", 16384),
            substeps=doc["case"].get("parcel_substeps", 4),
            ramp_time=spray.ramp_time_s,
            inner_outer_radius_ratio=doc["case"].get("spray_annulus_ratio", 1.0),
        )
        # Finite-temperature particles exchange with the same vapor/cloud fields.
        # No simultaneous entrained liquid source or duplicate loading force.
        moist_step = build_moist_atmospheric_step(
            carrier,
            grid,
            boundaries,
            momentum,
            scalar,
            config,
            moist.temperature_offset_k,
            moist.reference_temperature_k,
            ambient,
            scalar_transport_scheme=split_scalar_scheme,
            eddy_viscosity_override=inlet_eddy,
        )
        project_feedback = doc["case"].get("project_parcel_feedback", False)
        if not isinstance(project_feedback, bool):
            raise ValueError("project_parcel_feedback must be boolean")
        project_velocity = None
        if project_feedback:
            from jaxwind.numerics.poisson import project
            from jaxwind.open_boundary import enforce_open_velocity

            # Split off the divergence-free impulse before transporting scalars.
            # Do not reuse this impulse pressure as the carrier's lagged pressure.
            project_velocity = lambda velocity, h, inflow: project(
                enforce_open_velocity(velocity, inflow, grid, physical_transverse_inlet=physical_inlet), pressure, h
            )[0]
        step = build_inertial_water_step(
            moist_step,
            grid,
            source,
            config,
            moist.temperature_offset_k,
            project_velocity=project_velocity,
        )
        initial = InertialMoistAtmosphericSolution(
            *initial, initial_water_parcels(source, dtype)
        )
    if rans:
        from .water_spray_rans import build_rans_control

        initial, step = build_rans_control(
            initial,
            grid,
            boundaries,
            pressure,
            momentum,
            scalar,
            config,
            moist,
            ambient,
            source,
            project_velocity,
            doc["case"]["inlet_turbulence"],
            u,
            scalar_transport_scheme=split_scalar_scheme,
            vapor_turbulent_schmidt=0.7 if fluent_scalar else None,
            turbulence_model=carrier_turbulence,
            turbulence_transport_scheme=doc["numerics"].get(
                "turbulence_advection_scheme", "upwind"
            ),
        )
    inlet = build_benchmark_inlet(
        grid,
        u,
        plane,
        None if rans else doc["case"].get("inlet_turbulence"),
        doc["time"]["steps"] * doc["time"]["dt_seconds"],
    )
    dt = doc["time"]["dt_seconds"]

    @partial(jax.jit, static_argnums=1)
    def block(state, count):
        return jax.lax.fori_loop(
            0, count, lambda _, s: step(s, dt, inlet(s.time)), state
        )

    cfl = jax.jit(lambda s: courant_number(s.velocity, grid, dt))

    def advance(state, controls):
        result = block(state, controls.count)
        value = float(cfl(result))
        if not np.isfinite(value) or value > 0.9:
            raise RuntimeError(f"benchmark CFL {value:g} exceeds .9; reduce dt")
        if split_scalar_scheme is not None:
            fields = jnp.stack(
                (result.scalar + moist.temperature_offset_k, *result.moisture)
            )
            if not bool(jnp.all(jnp.isfinite(fields)) & jnp.all(fields >= 0)):
                raise RuntimeError(
                    "bounded scalar transport produced nonfinite or negative temperature/water"
                )
        if rans and not bool(jnp.all(jnp.isfinite(jnp.stack(result.turbulence)))):
            raise RuntimeError(
                "RANS state invalid or explicit diffusion number exceeded 0.5; reduce dt"
            )
        if model == "fluent-dpm" and not bool(result.accepted):
            raise RuntimeError(
                "DPM step rejected atomically; inspect capacity, path/event limits and thermal timestep"
            )
        if model == "inertial":
            if float(result.parcels.overflow_mass) > 0:
                raise RuntimeError(
                    "water parcel capacity exhausted; increase parcel_capacity"
                )
            minimum = float(jnp.min(result.scalar)) + moist.temperature_offset_k
            if minimum < config.freezing_temperature:
                raise RuntimeError(
                    "water parcel exchange exceeded the warm-physics range; reduce dt/substep"
                )
        return result

    # Original experimental Table 2(b) identifies the spatial pairing:
    # bottom/middle/top rows, with left/center/right sensors in each row.
    sensor_positions = (0.0975, 0.2925, 0.4875)
    y = jnp.asarray(grid.y_centers)
    z = jnp.asarray(grid.z_centers)
    area = jnp.asarray(grid.z_widths)[:, None] * jnp.asarray(grid.y_widths)[None, :]
    volumes = jnp.asarray(grid.cell_volumes)
    inlet_area = None if grid.is_uniform else area
    diffusion_metric = sum(
        float(np.min(getattr(grid, a + "_widths"))) ** -2 for a in "xyz"
    )

    @jax.jit
    def diagnostic(state):
        temperature = state.scalar[..., -1] + reference["dry_bulb_c"]
        readings = jnp.stack(
            [
                jnp.interp(
                    zp,
                    z,
                    jax.vmap(lambda row, yp=yp: jnp.interp(yp, y, row))(temperature),
                )
                for zp in sensor_positions
                for yp in sensor_positions
            ]
        )
        vapor_readings = jnp.stack(
            [
                jnp.interp(
                    zp,
                    z,
                    jax.vmap(lambda row, yp=yp: jnp.interp(yp, y, row))(
                        state.moisture.vapor[..., -1]
                    ),
                )
                for zp in sensor_positions
                for yp in sensor_positions
            ]
        )
        outflow = state.velocity.x[..., -1]
        vapor_out = config.dry_air_density * jnp.sum(
            outflow * state.moisture.vapor[..., -1] * area
        )
        spray_out = config.dry_air_density * jnp.sum(
            outflow * state.moisture.spray_liquid[..., -1] * area
        )
        current_inlet = inlet(state.time)
        inlet_u = current_inlet.x_velocity - u
        inlet_v = 0.5 * (
            current_inlet.y_velocity[:, :-1] + current_inlet.y_velocity[:, 1:]
        )
        inlet_w = 0.5 * (current_inlet.z_velocity[:-1] + current_inlet.z_velocity[1:])
        parcel_metrics = {
            "inlet_resolved_k_m2_s2": 0.5
            * (
                jnp.mean(inlet_u**2 + inlet_v**2 + inlet_w**2)
                if inlet_area is None
                else jnp.sum((inlet_u**2 + inlet_v**2 + inlet_w**2) * area)
                / jnp.sum(area)
            ),
            "inlet_bulk_u_m_s": (
                jnp.mean(current_inlet.x_velocity)
                if inlet_area is None
                else jnp.sum(current_inlet.x_velocity * area) / jnp.sum(area)
            ),
        }
        if rans:
            from jaxwind.rans_kepsilon import turbulent_viscosity

            if carrier_turbulence == "realizable-k-epsilon":
                from jaxwind.rans_realizable import (
                    turbulent_viscosity as realizable_viscosity,
                )

                diagnostic_gradients = None
                if physical_inlet:
                    from jaxwind.inlet_momentum import physical_inlet_gradients
                    diagnostic_gradients = physical_inlet_gradients(
                        state.velocity, inlet(state.time), grid, boundaries
                    )
                nut = realizable_viscosity(
                    state.turbulence, state.velocity, grid, boundaries,
                    gradients=diagnostic_gradients,
                )
            else:
                nut = turbulent_viscosity(state.turbulence)
            parcel_metrics.update(
                {
                    "rans_k_min": jnp.min(state.turbulence.kinetic_energy),
                    "rans_k_max": jnp.max(state.turbulence.kinetic_energy),
                    "rans_epsilon_min": jnp.min(state.turbulence.dissipation),
                    "rans_epsilon_max": jnp.max(state.turbulence.dissipation),
                    "rans_nut_max": jnp.max(nut),
                    "rans_diffusion_number": 2
                    * dt
                    * jnp.max(nut + viscosity)
                    * diffusion_metric,
                }
            )
        if model == "fluent-dpm":
            from jaxwind.fluent_dpm_atmosphere import dpm_diagnostics
            from jaxwind.waterjet_observation import sample_plane, wet_bulb_c

            readings = sample_plane(
                state.scalar + moist.temperature_offset_k - 273.15, grid, observation
            )
            vapor_readings = sample_plane(state.moisture.vapor, grid, observation)
            wet = wet_bulb_c(
                readings,
                vapor_readings,
                config.pressure,
                config.dry_air_gas_constant / config.water_vapor_gas_constant,
            )
            collected = state.dpm_ledger.collected
            drained = collected + state.dpm_ledger.trapped

            def drain_temperature(inventory):
                return jnp.where(
                    inventory[0] > 0,
                    material.reference_temperature
                    - 273.15
                    + inventory[4]
                    / (
                        material.liquid_cp
                        * jnp.where(inventory[0] > 0, inventory[0], 1)
                    ),
                    0.0,
                )

            parcel_metrics.update(dpm_diagnostics(state))
            parcel_metrics.update(
                {
                    **{f"sensor_{i}_wbt_c": wet[i] for i in range(9)},
                    "sensor_plane_x_m": jnp.asarray(observation.sensor_x_m),
                    "dpm_carrier_vapor_balance_error_kg": config.dry_air_density
                    * jnp.sum((state.moisture.vapor - qv) * volumes)
                    - state.transport_water
                    - state.dpm_ledger.evaporated_mass,
                    "dpm_carrier_enthalpy_J": config.dry_air_density
                    * jnp.sum(state.enthalpy * volumes),
                    "separator_collection_temperature_c": drain_temperature(collected),
                    "separator_collection_temperature_valid": collected[0] > 0,
                    "combined_drain_temperature_c": drain_temperature(drained),
                    "combined_drain_temperature_valid": drained[0] > 0,
                }
            )
        if model == "inertial":
            from jaxwind.cryogenic import _cic_coordinates, _cic_deposit_many

            p = state.parcels
            from jaxwind.physics.moisture import WaterDropletProperties

            inventory = jnp.sum(p.mass * p.multiplicity * p.active)
            liquid_enthalpy = jnp.sum(
                p.mass
                * p.multiplicity
                * p.active
                * WaterDropletProperties().liquid_heat_capacity
                * (p.temperature - config.freezing_temperature)
            )
            wall_inventory = jnp.sum(
                p.mass * p.multiplicity * p.active * p.touched_wall
            )
            wall_inventory_h = jnp.sum(
                p.mass
                * p.multiplicity
                * p.active
                * p.touched_wall
                * WaterDropletProperties().liquid_heat_capacity
                * (p.temperature - config.freezing_temperature)
            )
            deposited_mass = _cic_deposit_many(
                (p.mass * p.multiplicity * p.active)[None],
                _cic_coordinates(*p.position, grid),
                shape,
            )[0]
            parcel_metrics.update(
                {
                    "parcel_count": jnp.sum(p.active),
                    "wall_population_mass_balance_error_kg": wall_inventory
                    + p.escaped_wall_mass
                    + p.wall_evaporated_mass
                    - p.first_wall_mass,
                    "wall_population_enthalpy_balance_error_j": wall_inventory_h
                    + p.escaped_wall_enthalpy
                    + config.water_vapor_latent_heat * p.wall_evaporated_mass
                    - p.first_wall_enthalpy
                    - p.wall_gas_sensible_energy_loss,
                    **{
                        f"parcel_{key}_{unit}": getattr(p, key)
                        for key, unit in (
                            ("first_wall_mass", "kg"),
                            ("first_wall_enthalpy", "j"),
                            ("escaped_wall_mass", "kg"),
                            ("escaped_wall_enthalpy", "j"),
                            ("wall_gas_sensible_energy_loss", "j"),
                            ("wall_evaporated_mass", "kg"),
                        )
                    },
                    "parcel_injected_enthalpy_j": p.injected_enthalpy,
                    "parcel_escaped_enthalpy_j": p.escaped_enthalpy,
                    "parcel_gas_sensible_energy_loss_j": p.gas_sensible_energy_loss,
                    "parcel_inventory_enthalpy_j": liquid_enthalpy,
                    "parcel_enthalpy_balance_error_j": liquid_enthalpy
                    + p.escaped_enthalpy
                    + config.water_vapor_latent_heat * p.evaporated_mass
                    - p.injected_enthalpy
                    - p.gas_sensible_energy_loss,
                    "parcel_injected_mass_kg": p.injected_mass,
                    "parcel_escaped_mass_kg": p.escaped_mass,
                    "parcel_evaporated_mass_kg": p.evaporated_mass,
                    "parcel_inventory_kg": inventory,
                    "parcel_mass_balance_error_kg": inventory
                    + p.escaped_mass
                    + p.evaporated_mass
                    - p.injected_mass,
                    "parcel_maximum_volume_fraction": jnp.max(
                        deposited_mass / (config.water_density * volumes)
                    ),
                    "minimum_gas_temperature_k": jnp.min(state.scalar)
                    + moist.temperature_offset_k,
                }
            )
        return {
            **parcel_metrics,
            **{f"sensor_{i}_dbt_c": readings[i] for i in range(9)},
            **{f"sensor_{i}_vapor_kg_kg": vapor_readings[i] for i in range(9)},
            "sensor_mean_dbt_c": jnp.mean(readings),
            "effective_droplet_diameter_m": jnp.asarray(
                1
                / sum(
                    f / d
                    for f, d in zip(spray.dpm.mass_fractions, spray.dpm.diameters_m)
                )
                if model == "fluent-dpm"
                else equivalent_diameter(reference)
            ),
            "inlet_vapor_mixing_ratio": jnp.asarray(qv),
            "cooling_power_w": -config.dry_air_density
            * config.dry_air_heat_capacity
            * jnp.sum(outflow * state.scalar[..., -1] * area),
            "gas_sensible_anomaly_j": config.dry_air_density
            * config.dry_air_heat_capacity
            * jnp.sum(state.scalar * volumes),
            "water_inventory_kg": config.dry_air_density
            * jnp.sum(sum(state.moisture[:4]) * volumes),
            "maximum_spray_mixing_ratio": jnp.max(state.moisture.spray_liquid),
            "minimum_water_mixing_ratio": jnp.min(jnp.stack(state.moisture[:4])),
            "spray_outflow_kg_s": spray_out,
            "vapor_outflow_kg_s": vapor_out,
        }

    return Simulation(case, grid, initial, advance, cfl, state_diagnostics=diagnostic)

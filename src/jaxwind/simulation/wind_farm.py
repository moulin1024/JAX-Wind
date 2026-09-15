"""Independent lookup/ideal-TSR controllers coupled to periodic AD-BEM flow.

This is a prescribed-response controller, not a drivetrain torque balance.
Wind is measured on an upstream rotor-area probe; no universal Cp-optimal
TSR is assumed. Controller state advances once per actual flow step.
"""
from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp

from jaxwind.abl import AtmosphericSolution
from jaxwind.state import StaggeredVelocity
from jaxwind.numerics.discretization import cell_velocity


class RotorState(NamedTuple):
    omega: jnp.ndarray
    filtered_wind: jnp.ndarray


class FarmSolution(NamedTuple):
    # Preserve the ordinary field paths for diagnostics/checkpoint analysis.
    velocity: StaggeredVelocity
    pressure: jnp.ndarray
    momentum_tendency: StaggeredVelocity
    scalar: jnp.ndarray
    scalar_tendency: jnp.ndarray
    time: jnp.ndarray
    step: jnp.ndarray
    rotors: RotorState
    controller_dt: jnp.ndarray


def lookup_outputs(wind, control):
    """Piecewise-linear RPM and electrical power [W], zero outside operation.

    The cut-in endpoint operates; the cut-out endpoint does not. No
    extrapolation, density correction, or aerodynamic power inference.
    """
    speeds = jnp.asarray(control["wind_speed_m_s"])
    operating = (wind >= control["cut_in_m_s"]) & (wind < control["cut_out_m_s"])
    rpm = jnp.interp(wind, speeds, jnp.asarray(control["rpm"]))
    power = jnp.interp(wind, speeds, jnp.asarray(control["power_w"]))
    return jnp.where(operating, rpm, 0.), jnp.where(operating, power, 0.)


def target_omega(wind, radius, control):
    rpm_to_rad = 2. * jnp.pi / 60.
    if control["model"] == "wind-speed-lookup":
        return lookup_outputs(wind, control)[0] * rpm_to_rad
    return jnp.clip(control["target_tsr"] * jnp.maximum(wind, 0.) / radius,
                    control["minimum_rpm"] * rpm_to_rad,
                    control["maximum_rpm"] * rpm_to_rad)


def advance_rotors(state, wind, dt, radius, control):
    """Filter wind, then prescribe lookup RPM or advance an ideal-TSR servo.

    Uses the newly filtered wind as a held target for this step (first-order
    partitioned coupling). The ideal-TSR servo has a physical-time slew cap;
    lookup mode assigns RPM directly and masks out-of-operation wind speeds.
    """
    wind = jnp.maximum(wind, 0.)
    filtered = state.filtered_wind + (-jnp.expm1(-dt / control["wind_filter_seconds"])) * (wind - state.filtered_wind)
    target = target_omega(filtered, radius, control)
    if control["model"] == "wind-speed-lookup":
        # Prescribed numerical RPM, not a generator/drivetrain speed servo.
        return RotorState(jnp.where(dt > 0., target, state.omega), filtered)
    change = (-jnp.expm1(-dt / control["response_seconds"])) * (target - state.omega)
    limit = control["maximum_acceleration_rpm_s"] * (2. * jnp.pi / 60.) * dt
    omega = state.omega + jnp.clip(change, -limit, limit)
    return RotorState(omega, filtered)


class ControlledFarm:
    def __init__(self, case, *, open_domain=None):
        from jaxwind.config.stages import load_workflow
        from jaxwind.domain import ScaleSystem
        from jaxwind.simulation.turbines import build_turbine_definition
        from jaxwind.turbine import build_adbem_forcing

        if "inflow" in case.document["physics"] or open_domain:
            from jaxwind.config.abl import load_fv_abl
            from jaxwind.config.document import native_document
            from jaxwind.config.stages import FiniteVolumeWorkflow, WorkflowOptions, _load_turbine
            settings = case.document["time"]
            workflow = FiniteVolumeWorkflow(
                load_fv_abl(case),
                WorkflowOptions(0, 0, settings["steps"], 0, settings.get("chunk_steps", 12), case.output),
                turbine=_load_turbine(native_document(case)),
            )
        else:
            workflow = load_workflow(case)
        turbine = build_turbine_definition(workflow)
        self.disk = turbine.to_actuator_disk(scales=ScaleSystem(1., 1.))
        self.grid = workflow.case.physical.physical_grid
        if any(workflow.case.physical.advection_frame_velocity_m_s):
            raise ValueError("controlled wind_farm requires a stationary advection frame")
        farm = case.document["physics"]["wind_farm"]
        self.control = farm["controller"]
        self.open_domain = "inflow" in case.document["physics"] if open_domain is None else open_domain
        self.layout = farm["layout"]
        self.ids = [row["id"] for row in self.layout]
        positions = np.array([(row["x_m"], row["y_m"], row["hub_height_m"]) for row in self.layout])
        radius = self.disk.tip_radius
        if self.open_domain and (np.any(positions[:, 1] - radius <= 0.) or np.any(positions[:, 1] + radius >= self.grid.ly)):
            raise ValueError("farm rotor intersects a lateral boundary")
        if np.any(positions[:, 2] - radius <= 0.) or np.any(positions[:, 2] + radius >= self.grid.lz):
            raise ValueError("farm rotor intersects a vertical boundary")
        self.positions = jnp.asarray(positions, dtype=workflow.case.physical.dtype)
        self.single_force = build_adbem_forcing(self.grid, self.disk,
                                               periodic_x=not self.open_domain, periodic_y=not self.open_domain)
        if self.open_domain:
            # Six Gaussian smoothing widths make omitted tails negligible in
            # float32. Evaluate each rotor on a local patch, not 80 full grids.
            from jaxwind.domain import UniformGrid
            grid = self.grid
            if not grid.is_uniform:
                raise ValueError("open farm force patches require a uniform mesh")
            margin = 6. * max(self.disk.element_smoothing_widths)
            half = np.array([margin, radius + margin, radius + margin])
            spacing = np.array([grid.dx, grid.dy, grid.dz])
            counts = np.array([grid.nx, grid.ny, grid.nz])
            size = np.minimum(np.ceil(2 * half / spacing).astype(int) + 2, counts)
            starts = np.clip(np.floor((positions - half) / spacing).astype(int), 0, counts - size)
            self.patch_starts = jnp.asarray(starts[:, ::-1], jnp.int32)
            self.patch_positions = jnp.asarray(positions - starts * spacing, self.positions.dtype)
            self.patch_shape = tuple(int(n) for n in size[::-1])
            local_grid = UniformGrid(*(int(n) for n in size), *(float(v) for v in size * spacing))
            self.patch_force = build_adbem_forcing(local_grid, self.disk, periodic_x=False, periodic_y=False)
        # Separable volume-weighted probes avoid N full-domain sampling arrays.
        grid = self.grid
        probe_distance = self.control["probe_distance_diameters"] * 2. * radius
        if not self.open_domain and probe_distance >= grid.lx / 2.:
            raise ValueError("upstream probe distance must be less than half the periodic domain length")
        dx = (np.asarray(grid.x_centers)[None, :] - (positions[:, 0, None] - probe_distance) + .5 * grid.lx) % grid.lx - .5 * grid.lx
        wx = np.exp(-(dx / max(float(np.max(grid.x_widths)), float(turbine.smoothing_width_m))) ** 2) * np.asarray(grid.x_widths)
        wx /= wx.sum(axis=1, keepdims=True)
        if self.control["model"] == "wind-speed-lookup":
            # Interpolate to the upstream plane, rather than using a Gaussian
            # slab whose tails include points closer to the rotor.
            centers = np.asarray(grid.x_centers)
            extended = np.r_[centers[-1] - grid.lx, centers, centers[0] + grid.lx]
            probes = positions[:, 0] - probe_distance
            if self.open_domain:
                if np.any(probes < centers[0]) or np.any(probes > centers[-1]):
                    raise ValueError("open-domain upstream probes must lie inside cell-center bounds")
            else:
                probes = probes % grid.lx
            right = np.searchsorted(extended, probes, side="right")
            fraction = (probes - extended[right - 1]) / (extended[right] - extended[right - 1])
            wx = np.zeros_like(wx)
            rows = np.arange(len(probes))
            wx[rows, (right - 2) % len(centers)] = 1. - fraction
            wx[rows, (right - 1) % len(centers)] += fraction
        dy = (np.asarray(grid.y_centers)[None, None, :] - positions[:, 1, None, None] + .5 * grid.ly) % grid.ly - .5 * grid.ly
        if self.open_domain:
            dy = np.asarray(grid.y_centers)[None, None, :] - positions[:, 1, None, None]
        dz = np.asarray(grid.z_centers)[None, :, None] - positions[:, 2, None, None]
        wyz = ((dy * dy + dz * dz) <= radius * radius) * np.asarray(grid.z_widths)[None, :, None] * np.asarray(grid.y_widths)[None, None, :]
        area = wyz.sum(axis=(1, 2), keepdims=True)
        if np.any(area == 0.):
            raise ValueError("rotor-area probe has no mesh cells; refine the grid")
        self.wx = jnp.asarray(wx, self.positions.dtype)
        self.wyz = jnp.asarray(wyz / area, self.positions.dtype)

    def sample_wind(self, velocity):
        u, _, _ = cell_velocity(velocity)
        def probe(weights):
            wx, wyz = weights
            return jnp.einsum("zy,x,zyx->", wyz, wx, u, optimize="optimal")
        return jax.lax.map(probe, (self.wx, self.wyz))

    def initialize(self, flow):
        omega = jnp.asarray([row["initial_rpm"] for row in self.layout], flow.time.dtype) * (2. * jnp.pi / 60.)
        rotors = RotorState(omega, jnp.maximum(self.sample_wind(flow.velocity), 0.))
        if self.control["model"] == "wind-speed-lookup":
            rotors = rotors._replace(omega=target_omega(rotors.filtered_wind, self.disk.tip_radius, self.control))
        return FarmSolution(*flow, rotors, jnp.zeros_like(flow.time))

    def force(self, velocity, time, omega):
        # Accumulate sequentially: no array of N full-domain force fields.
        zero = jax.tree.map(jnp.zeros_like, velocity)
        if self.open_domain:
            nz, ny, nx = self.patch_shape
            shapes = ((nz, ny, nx + 1), (nz, ny + 1, nx), (nz + 1, ny, nx))
            def add_patch(total, inputs):
                start, position, speed = inputs
                local = StaggeredVelocity(*(jax.lax.dynamic_slice(field, start, shape)
                                           for field, shape in zip(velocity, shapes)))
                source = self.patch_force(local, time, position=position, angular_velocity=speed)
                updated = StaggeredVelocity(*(jax.lax.dynamic_update_slice(
                    field, jax.lax.dynamic_slice(field, start, shape) + value, start)
                    for field, value, shape in zip(total, source, shapes)))
                return updated, None
            return jax.lax.scan(add_patch, zero, (self.patch_starts, self.patch_positions, omega))[0]
        def add(total, inputs):
            position, speed = inputs
            value = self.single_force(velocity, time, position=position, angular_velocity=speed)
            return jax.tree.map(jnp.add, total, value), None
        return jax.lax.scan(add, zero, (self.positions, omega))[0]

    def couple_step(self, step_factory):
        def step(state, dt):
            measured = self.sample_wind(state.velocity)
            rotors = advance_rotors(state.rotors, measured, dt, self.disk.tip_radius, self.control)
            # Hold midpoint servo RPM (or new prescribed lookup RPM) across
            # RK substages; forces still sample each substage's velocity.
            # Controller/flow coupling is first order.
            omega = .5 * (state.rotors.omega + rotors.omega)
            if self.control["model"] == "wind-speed-lookup":
                omega = rotors.omega
            flow_step = step_factory(lambda velocity, time: self.force(velocity, time, omega))
            flow = flow_step(AtmosphericSolution(*state[:7]), dt)
            return FarmSolution(*flow, rotors, jnp.asarray(dt, state.time.dtype))
        return step

    def diagnostics(self, state):
        wind = self.sample_wind(state.velocity)
        filtered = state.rotors.filtered_wind
        tsr = jnp.where(filtered > 1.e-6, state.rotors.omega * self.disk.tip_radius / jnp.maximum(filtered, 1.e-6), 0.)
        target = target_omega(filtered, self.disk.tip_radius, self.control)
        result = {}
        for i, name in enumerate(self.ids):
            for field, value in (("rpm", state.rotors.omega * 60. / (2. * jnp.pi)),
                                 ("target_rpm", target * 60. / (2. * jnp.pi)),
                                 ("probe_wind_m_s", wind), ("filtered_wind_m_s", filtered),
                                 ("tsr", tsr), ("tsr_valid", filtered > 1.e-6)):
                result[f"turbine_{name}_{field}"] = value[i]
        result["controller_dt_seconds"] = state.controller_dt
        if self.control["model"] == "wind-speed-lookup":
            _, power = lookup_outputs(filtered, self.control)
            for i, name in enumerate(self.ids):
                result[f"turbine_{name}_lookup_power_w"] = power[i]
            result["farm_lookup_power_w"] = jnp.sum(power)
        return result

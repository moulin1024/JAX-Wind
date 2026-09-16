"""Simulation adapters for periodic development, recording, and open inflow."""
from __future__ import annotations
from dataclasses import replace
from copy import deepcopy
from .api import Simulation, AdvanceResult, build_simulation
from jaxwind.config.document import ResolvedCase


def build_stage(case, operation, inputs, options):
    farm = "wind_farm" in case.document.get("physics", {})
    if farm and operation not in {"simulation", "open-inflow"}:
        raise ValueError("warmup and precursor must be turbine-free")
    if operation == "open-inflow" and not {"checkpoint", "inflow"} <= inputs.keys():
        raise ValueError("open-inflow requires checkpoint and inflow inputs")
    if operation == "simulation":
        if "checkpoint" in inputs:
            from jaxwind.io.checkpoint import checkpoint_metadata
            from jaxwind.config.document import ResolvedCase
            document = deepcopy(case.document)
            initial = document.setdefault("initial_conditions", {})
            if case.formulation == "low-mach-abl":
                for key in ("initial_condition", "incompressible_checkpoint", "low_mach_checkpoint"):
                    initial.pop(key, None)
                source_formulation = checkpoint_metadata(inputs["checkpoint"])["formulation"]
                if source_formulation not in {"boussinesq", "low-mach-abl"}:
                    raise ValueError("unsupported low-Mach initialization conversion")
                key = "incompressible_checkpoint" if source_formulation == "boussinesq" else "low_mach_checkpoint"
                initial[key] = inputs["checkpoint"]
            else:
                initial["checkpoint"] = inputs["checkpoint"]
            case = ResolvedCase(case.source, document)
        return build_simulation(case)
    if case.formulation != "boussinesq":
        raise ValueError("periodic/inflow stage adapters currently require Boussinesq flow")
    from jaxwind.config.abl import load_fv_abl
    from .abl import initialize_periodic, build_periodic_advance
    from jaxwind.io.state_fields import atmospheric_state
    import jax
    import jax.numpy as jnp
    from jaxwind import courant_number, extract_inflow_plane, stable_timestep
    configured = load_fv_abl(case)
    jax.config.update("jax_enable_x64", configured.physical.dtype == "float64")
    grid = configured.physical.physical_grid
    warm = atmospheric_state(inputs["checkpoint"], grid) if "checkpoint" in inputs else initialize_periodic(configured, jax, jnp)
    dt = case.document["time"]["dt_seconds"]
    adaptive = "cfl" in case.document["time"]
    if adaptive and configured.options.time_integration == "ab2":
        raise ValueError("adaptive periodic/inflow stages require rk3 or fast-rk3; AB2 requires fixed timesteps")
    courant = jax.jit(lambda state: courant_number(state.velocity, grid, dt))
    if operation == "open-inflow":
        from jaxwind.config.stages import load_workflow
        from .open_atmospheric import build_open_components
        from jaxwind.io.recording import InflowReader
        if "inflow" not in inputs or "checkpoint" not in inputs:
            raise ValueError("open-inflow requires checkpoint and inflow inputs")
        if adaptive:
            raise ValueError("open-inflow currently supports fixed timesteps only")
        factor = options.get("substeps_per_inflow", 1)
        if type(factor) is not int or factor <= 0:
            raise ValueError("substeps_per_inflow must be a positive integer")
        reader = InflowReader(inputs["inflow"], grid, samples=(case.document["time"]["steps"] + factor - 1) // factor, dt=dt*factor)
        first = reader.read(0, 1)
        first = type(first)(*(item[0] for item in first))
        turbine_diagnostics = None
        if farm:
            if options.get("lateral_boundary") != "outflow":
                raise ValueError("recorded farm requires lateral_boundary = outflow")
            from .recorded_farm import build_recorded_farm
            scale = 1.
            target = options.get("target_hub_wind_speed_m_s")
            if target is not None:
                import numpy as np
                if isinstance(target, bool) or not np.isfinite(target) or target <= 0.:
                    raise ValueError("target_hub_wind_speed_m_s must be finite and positive")
                mean = np.asarray(reader.metadata.get("mean_x_velocity_profile_m_s", []))
                if mean.shape != (grid.nz,) or not np.isfinite(mean).all():
                    raise ValueError("scaled replay requires recorded mean velocity profile metadata")
                height = case.document["physics"]["turbine"]["hub_height_m"]
                reference = float(np.interp(height, np.asarray(grid.z_centers), mean))
                if reference <= 0.:
                    raise ValueError("reference recorded hub-height wind must be positive")
                scale = target / reference
                print(f"Scaled replay: recorded mean Uhub={reference:.6g} m/s, target={target:.6g} m/s, velocity scale={scale:.6g}; timestamps unchanged", flush=True)
            initial, advance, turbine_diagnostics = build_recorded_farm(case, warm, first, velocity_scale=scale)
        else:
            if "target_hub_wind_speed_m_s" in options:
                raise ValueError("scaled replay currently requires a controlled farm")
            if options.get("lateral_boundary", "periodic") != "periodic":
                raise ValueError("legacy single-turbine replay supports periodic y only")
            workflow = load_workflow(case)
            initial, advance = build_open_components(workflow, warm, first)
        def advance_open(state, controls):
            start = int(state.step)
            first_sample, last_sample = start // factor, (start + controls.count + factor - 1) // factor
            planes = reader.read(first_sample, last_sample)
            planes = type(planes)(*(jnp.repeat(item, factor, axis=0)[start % factor:start % factor + controls.count] for item in planes))
            result = advance(state, dt, planes)
            if farm:
                import math
                value = float(courant(result))
                if not math.isfinite(value) or value > 1.:
                    raise RuntimeError(f"recorded farm stopped: CFL={value}; reduce fixed dt")
            return result
        state_diagnostics = None
        if configured.options.outlet_backflow == "energy":
            from jaxwind.open_boundary import backflow_outlet_pressure
            area = jnp.asarray(grid.z_widths)[:, None] * jnp.asarray(grid.y_widths)[None, :]
            @jax.jit
            def state_diagnostics(state):
                normal = state.velocity.x[..., -1]
                return {
                    "outlet_backflow_area_fraction": jnp.sum(area * (normal < 0.)) / jnp.sum(area),
                    "outlet_minimum_u_m_s": jnp.min(normal),
                    "outlet_backflow_pressure_min_m2_s2": jnp.min(backflow_outlet_pressure(state.velocity, grid)),
                    "open_x_net_volume_flux_m3_s": jnp.sum(area * (normal - state.velocity.x[..., 0])),
                }
        return Simulation(case, grid, initial, advance_open, courant,
                          turbine_diagnostics=turbine_diagnostics,
                          state_diagnostics=state_diagnostics)
    if operation not in {"periodic", "record-inflow"}:
        raise ValueError(f"unknown stage operation: {operation}")
    from .abl import build_models
    from .atmospheric import build_diagnostics
    boundaries, momentum, scalar, _, surface = build_models(configured, periodic_x=True)
    diagnostics = build_diagnostics(
        configured, grid, boundaries, momentum.surface, scalar,
        momentum.subfilter, surface,
    )
    step, fixed = build_periodic_advance(configured)
    if operation == "periodic" and not adaptive:
        initial_time, initial_step = float(warm.time), int(warm.step)
        def advance_periodic(state, controls):
            final = fixed(state, dt, controls.count)
            return final._replace(time=jnp.asarray(initial_time + (int(final.step)-initial_step)*dt, final.time.dtype))
        return Simulation(case, grid, warm, advance_periodic, courant, diagnostics=diagnostics)
    recording = operation == "record-inflow"
    plane_index = options.get("record_plane", 0)
    if type(plane_index) is not int or not 0 <= plane_index < grid.nx:
        raise ValueError("record_plane is outside the mesh")
    initial_time, initial_step = float(warm.time), int(warm.step)
    def block(current, count, target_time):
        def advance(state, unused):
            if not adaptive:
                state = state._replace(time=jnp.asarray(initial_time, state.time.dtype) + (state.step-initial_step)*dt)
            plane = extract_inflow_plane(state, grid, plane_index) if recording else ()
            active_dt = jnp.asarray(dt, state.time.dtype)
            if adaptive:
                active_dt = jnp.minimum(jnp.minimum(active_dt, stable_timestep(state.velocity, grid, 0., courant=case.document["time"]["cfl"])), jnp.maximum(target_time-state.time, 0.))
                final = jax.lax.cond(active_dt > 0., lambda value: step(value, active_dt), lambda value: value, state)
            else:
                final = step(state, active_dt)
                final = final._replace(time=jnp.asarray(initial_time, final.time.dtype) + (final.step-initial_step)*dt)
            values = (*plane, state.time, active_dt)
            if adaptive:
                values += (courant_number(state.velocity, grid, active_dt),)
            return final, values
        return jax.lax.scan(advance, current, None, length=count)
    compiled = jax.jit(block, static_argnums=1)
    step_metrics = {"dt_seconds": jnp.asarray(dt, warm.time.dtype),
                    "block_maximum_cfl": jnp.asarray(0., warm.time.dtype)}
    measured_courant = jax.jit(lambda state, actual_dt: courant_number(state.velocity, grid, actual_dt))
    def advance_block(state, controls):
        final, arrays = compiled(state, controls.count, controls.target_time)
        if adaptive:
            offset = 4 if recording else 0
            last = jnp.maximum(final.step - state.step - 1, 0)
            step_metrics["dt_seconds"] = arrays[offset + 1][last]
            step_metrics["block_maximum_cfl"] = jnp.max(arrays[offset + 2])
        if not recording:
            return final
        names = ("x_velocity", "y_velocity", "z_velocity", "scalar", "time_seconds", "dt_seconds")
        if adaptive:
            names += ("maximum_cfl",)
        return AdvanceResult(final, dict(zip(names, arrays)))
    if adaptive:
        courant = lambda state: measured_courant(state, step_metrics["dt_seconds"])
    return Simulation(case, grid, warm, advance_block, courant, adaptive, diagnostics,
                      state_diagnostics=(lambda state: dict(step_metrics)) if adaptive else None)

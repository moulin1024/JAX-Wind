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
        return Simulation(case, grid, initial, advance_open, courant,
                          turbine_diagnostics=turbine_diagnostics)
    if operation not in {"periodic", "record-inflow"}:
        raise ValueError(f"unknown stage operation: {operation}")
    step, fixed = build_periodic_advance(configured)
    if operation == "periodic":
        if adaptive:
            from jaxwind import build_adaptive_atmospheric_run
            advance = build_adaptive_atmospheric_run(step, grid, cfl_ceiling=case.document["time"]["cfl"], maximum_dt=dt)
            return Simulation(case, grid, warm, lambda state, controls: advance(state, controls.target_time, controls.count), courant, True)
        initial_time, initial_step = float(warm.time), int(warm.step)
        def advance_periodic(state, controls):
            final = fixed(state, dt, controls.count)
            return final._replace(time=jnp.asarray(initial_time + (int(final.step)-initial_step)*dt, final.time.dtype))
        return Simulation(case, grid, warm, advance_periodic, courant)
    plane_index = options.get("record_plane", 0)
    if type(plane_index) is not int or not 0 <= plane_index < grid.nx:
        raise ValueError("record_plane is outside the mesh")
    initial_time, initial_step = float(warm.time), int(warm.step)
    def block(current, count, target_time):
        def advance(state, unused):
            if not adaptive:
                state = state._replace(time=jnp.asarray(initial_time, state.time.dtype) + (state.step-initial_step)*dt)
            plane = extract_inflow_plane(state, grid, plane_index)
            active_dt = jnp.asarray(dt, state.time.dtype)
            if adaptive:
                active_dt = jnp.minimum(jnp.minimum(active_dt, stable_timestep(state.velocity, grid, 0., courant=case.document["time"]["cfl"])), jnp.maximum(target_time-state.time, 0.))
                final = jax.lax.cond(active_dt > 0., lambda value: step(value, active_dt), lambda value: value, state)
            else:
                final = step(state, active_dt)
                final = final._replace(time=jnp.asarray(initial_time, final.time.dtype) + (final.step-initial_step)*dt)
            return final, (*plane, state.time, active_dt)
        return jax.lax.scan(advance, current, None, length=count)
    compiled = jax.jit(block, static_argnums=1)
    def record(state, controls):
        final, arrays = compiled(state, controls.count, controls.target_time)
        outputs = dict(zip(("x_velocity", "y_velocity", "z_velocity", "scalar", "time_seconds", "dt_seconds"), arrays))
        return AdvanceResult(final, outputs)
    return Simulation(case, grid, warm, record, courant, adaptive)

"""Small execution contract shared by all coupled formulations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from jaxwind.config.document import ResolvedCase, load_case


@dataclass(frozen=True)
class RunControls:
    count: int
    target_time: float


@dataclass(frozen=True)
class AdvanceResult:
    state: Any
    outputs: dict


@dataclass(frozen=True)
class Simulation:
    case: ResolvedCase
    grid: Any
    initial_state: Any
    advance_block: Callable
    courant: Callable
    adaptive: bool = False
    diagnostics: Any = None
    turbine_diagnostics: Any = None
    state_diagnostics: Any = None
    courant_with_timestep: Callable | None = None

    def initialize(self, inputs=None):
        if inputs:
            raise ValueError("initial inputs must be resolved by the simulation builder")
        return self.initial_state

    def advance(self, state, controls: RunControls):
        return self.advance_block(state, controls)


def _build(case) -> Simulation:
    case = load_case(case)
    import jax
    import jax.numpy as jnp
    from jaxwind import courant_number
    dtype = case.document["numerics"].get("dtype", "float32")
    jax.config.update("jax_enable_x64", dtype == "float64")
    dt = case.document["time"]["dt_seconds"]
    if case.document["case"].get("benchmark") == "montazeri2015-water-spray":
        from .water_spray_benchmark import build_simulation as build_benchmark
        return build_benchmark(case)
    if case.formulation == "boussinesq" and "inflow" in case.document.get("physics", {}):
        if case.document["physics"]["inflow"]["model"] == "uniform":
            from .uniform_farm import build_simulation as build_uniform
            return build_uniform(case)
        from .synthetic_inflow import build_simulation as build_synthetic
        return build_synthetic(case)
    if case.formulation == "boussinesq":
        if "moisture" in case.document.get("physics", {}):
            raise ValueError("moisture requires an open-inflow workflow stage")
        from jaxwind.config.abl import load_fv_abl
        from .atmospheric import build_components
        forcing = None
        farm = None
        if "wind_farm" in case.document.get("physics", {}):
            from .wind_farm import ControlledFarm
            farm = ControlledFarm(case)
        elif "turbine" in case.document.get("physics", {}):
            from jaxwind.config.stages import load_workflow
            from .turbines import build_turbine_forcing
            forcing = build_turbine_forcing(load_workflow(case))
        components = build_components(load_fv_abl(case), forcing=forcing, farm=farm)
        def advance(state, controls):
            target = controls.target_time if components.adaptive else dt
            return components.advance(state, target, controls.count)
        def courant(state, timestep=None):
            active_dt = (state.controller_dt if farm is not None and timestep is None else dt if timestep is None else timestep)
            return courant_number(state.velocity, components.grid, active_dt)
        courant = jax.jit(courant)
        return Simulation(case, components.grid, components.initial, advance, courant, components.adaptive, components,
                          jax.jit(farm.diagnostics) if farm is not None else None,
                          courant if components.adaptive else None)
    if case.formulation == "low-mach-abl":
        from jaxwind.config.low_mach import load_case as load_native
        from jaxwind.simulation.low_mach import build_simulation as build_native
        native = load_native(case)
        workflow, grid, initial, advance, courant = build_native(native)
        return Simulation(case, grid, initial, lambda state, controls: advance(state, controls.count), courant)
    from jaxwind.config.jet import load_case as load_native
    from jaxwind.simulation.jet import build_simulation as build_native
    native = load_native(case)
    grid, jet, microphysics, initial, advance, _ = build_native(native)
    if native.cfl is not None:
        return Simulation(case, grid, initial,
                          lambda state, controls: advance(state, controls.target_time, controls.count),
                          lambda state: state.last_cfl, adaptive=True)
    courant = jax.jit(lambda state: courant_number(state.velocity, grid, dt))
    return Simulation(case, grid, initial, lambda state, controls: advance(state, controls.count), courant)


def build_simulation(case) -> Simulation:
    from dataclasses import replace
    from jaxwind.io.state_fields import initialize_state
    case = load_case(case)
    initial = case.document.get("initial_conditions", {})
    if initial.get("operation") in {"periodic", "record-inflow", "open-inflow"}:
        from .stages import build_stage
        return build_stage(case, initial["operation"], initial.get("artifacts", {}),
                           initial.get("stage_options", {}))
    simulation = _build(case)
    checkpoint = case.document.get("initial_conditions", {}).get("checkpoint")
    if checkpoint is not None:
        template = simulation.initial_state
        if hasattr(template, "rotors"):
            from jaxwind.io.checkpoint import checkpoint_metadata
            header = checkpoint_metadata(checkpoint)
            if header["state"].get("record") == "AtmosphericSolution":
                from jaxwind.abl import AtmosphericSolution
                from .wind_farm import ControlledFarm
                flow = initialize_state(checkpoint, AtmosphericSolution(*template[:7]), simulation.grid, case.formulation)
                state = ControlledFarm(case).initialize(flow)
            else:
                previous = header.get("resolved_case", {}).get("physics", {}).get("wind_farm", {})
                if previous.get("layout") != case.document["physics"]["wind_farm"]["layout"]:
                    raise ValueError("farm checkpoint initialization requires an unchanged turbine layout/order")
                state = initialize_state(checkpoint, template, simulation.grid, case.formulation)
        else:
            state = initialize_state(checkpoint, template, simulation.grid, case.formulation)
        simulation = replace(simulation, initial_state=state)
    return simulation

"""Functional JAX implementation of WiRE-LES."""

from .config import AB2_DEFAULT_DT, RK4_DEFAULT_DT, Params, default_dt_for_time_scheme
from .convergence import UStarSlidingWindow
from .driver import run
from .init import initial_state
from .state import Diagnostics, FlowState
from .spray_dpm import (
    SprayCoupledState,
    SprayDPMConfig,
    SprayDiagnostics,
    SprayGasIncrements,
    SprayState,
    apply_spray_increments,
    inject_spray,
    initialize_spray,
    run_spray_dpm,
    sample_diameters,
    spray_exchange,
    step_spray_dpm,
)
from .timestep import step
from .timestep_sharded import ShardedFlowState, run_sharded
from .adjoint_sharded import (
    duplicate_state_for_adjoint,
    make_adjoint_chunk_step,
    make_empty_fringe_chunk,
    make_exchange_precursor_chunk,
    make_adjoint_pipeline_batch,
    make_adjoint_pipeline_prime,
)
from .spray_dpm_sharded import (
    ShardedSprayCoupledState,
    ShardedSprayDiagnostics,
    SprayMigrationDiagnostics,
    initialize_sharded_spray,
    make_inject_sharded_spray,
    make_migrate_sharded_spray,
    make_spray_exchange_sharded,
    make_step_spray_dpm_sharded,
    spray_sharding,
)

__all__ = [
    "Diagnostics",
    "FlowState",
    "SprayCoupledState",
    "SprayDPMConfig",
    "SprayDiagnostics",
    "SprayGasIncrements",
    "SprayState",
    "SprayMigrationDiagnostics",
    "ShardedSprayCoupledState",
    "ShardedSprayDiagnostics",
    "ShardedFlowState",
    "UStarSlidingWindow",
    "Params",
    "AB2_DEFAULT_DT",
    "RK4_DEFAULT_DT",
    "default_dt_for_time_scheme",
    "duplicate_state_for_adjoint",
    "initial_state",
    "initialize_sharded_spray",
    "initialize_spray",
    "inject_spray",
    "apply_spray_increments",
    "run",
    "run_spray_dpm",
    "sample_diameters",
    "run_sharded",
    "make_inject_sharded_spray",
    "make_adjoint_chunk_step",
    "make_adjoint_pipeline_batch",
    "make_adjoint_pipeline_prime",
    "make_empty_fringe_chunk",
    "make_exchange_precursor_chunk",
    "make_migrate_sharded_spray",
    "make_spray_exchange_sharded",
    "make_step_spray_dpm_sharded",
    "step",
    "spray_exchange",
    "step_spray_dpm",
    "spray_sharding",
]

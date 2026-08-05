from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from benchmark.GABLS1 import run as gabls1
from benchmark.Nieuwstadt1993 import run_amd as nieuwstadt
from jaxwind.pressure import mac_divergence


def test_nieuwstadt_minimal_solver_advances_one_projected_step() -> None:
    args = nieuwstadt.parse_args(["--quick"])
    coupled, case, dtype = _nieuwstadt_solver(args)
    state = nieuwstadt._initial_state(args, coupled, case, dtype)

    advanced = coupled.step(state, timestep=0.25)

    assert advanced.step == 1
    assert np.all(np.isfinite(np.asarray(advanced.potential_temperature)))
    divergence = mac_divergence(advanced.velocity, coupled.grid)
    assert float(coupled.momentum.pressure_solver.operator.norm(divergence)) < 1.0e-4


def _nieuwstadt_solver(args):
    from jaxwind.momentum import (
        AMDBoussinesq,
        AMDBoussinesqConfig,
        AMDModel,
        AMDPassiveScalar,
        AMDPassiveScalarModel,
        NeutralABLConfig,
        NeutralABLMomentum,
    )
    from jaxwind.pressure import (
        BoundaryCondition,
        GMGConfig,
        MatrixFreePoissonSolver,
        PCGConfig,
        PoissonBoundaryConditions,
        RectilinearGrid,
    )

    case = nieuwstadt.amd_diagnostics.NieuwstadtCase()
    dtype = jnp.float32
    grid = RectilinearGrid.uniform(8, 8, 8, lx=6400.0, ly=6400.0, lz=2400.0)
    periodic = BoundaryCondition("periodic")
    neumann = BoundaryCondition("neumann")
    pressure = MatrixFreePoissonSolver(
        grid,
        PoissonBoundaryConditions(
            periodic, periodic, periodic, periodic, neumann, neumann
        ),
        dtype=dtype,
        gmg=GMGConfig(coarse_smooth=20),
        krylov=PCGConfig(max_iterations=40, relative_tolerance=1.0e-5),
    )
    momentum = NeutralABLMomentum(
        grid,
        pressure,
        NeutralABLConfig(
            friction_velocity=case.wstar0,
            roughness_length=nieuwstadt.ROUGHNESS_LENGTH,
            pressure_acceleration=0.0,
            amd=AMDModel(),
        ),
    )
    scalar = AMDPassiveScalar(
        grid,
        AMDPassiveScalarModel(lower_surface_flux=case.surface_theta_flux),
    )
    return (
        AMDBoussinesq(
            momentum,
            scalar,
            AMDBoussinesqConfig(
                gravity=case.gravity,
                reference_potential_temperature=case.theta0,
            ),
        ),
        case,
        dtype,
    )


def test_gabls1_minimal_solver_advances_one_projected_step() -> None:
    args = gabls1.parse_args(["--quick"])
    coupled, case, dtype = gabls1._build_coupled(args)
    state = gabls1._initial_state(args, coupled, case, dtype)

    advanced = coupled.step(state, timestep=0.25)

    assert advanced.step == 1
    assert np.all(np.isfinite(np.asarray(advanced.potential_temperature)))
    divergence = mac_divergence(advanced.velocity, coupled.grid)
    assert float(coupled.momentum.pressure_solver.operator.norm(divergence)) < 1.0e-3

"""A staggered-mesh finite-volume incompressible solver with a pressure solve.

The discretisation is the classical marker-and-cell arrangement on a uniform
Cartesian box: periodic in x and y, impermeable walls in z.  Its discrete
divergence and gradient are exact adjoints, so the pressure Poisson operator
is the compact seven-point Laplacian and can be handed to any of three
backends -- :mod:`jaxamg` on the GPU (needs AmgX and a CUDA build); ``gmg``, a
JAX-native,
matrix-free geometric multigrid V-cycle built straight from the mesh, also
preconditioning conjugate gradients; or ``fft``, a direct diagonalisation
that is exact but only valid because of the periodic horizontal boundary.
"""

from .abl import (
    AtmosphericSolution,
    build_adaptive_atmospheric_run,
    build_atmospheric_run,
    build_atmospheric_step,
    initial_atmospheric_solution,
)
from .buoyancy import LinearBoussinesqBuoyancy, boussinesq_tendency
from .diagnostics import (
    atmospheric_history_diagnostics,
    atmospheric_profile_diagnostics,
)
from .integrate import (
    FlowModel,
    Solution,
    build_adaptive_run,
    build_run,
    build_step,
    build_tendency,
    initial_solution,
)
from .open_abl import build_open_atmospheric_run, build_open_atmospheric_step
from .open_boundary import (
    InflowPlane,
    enforce_open_scalar,
    enforce_open_velocity,
    extract_inflow_plane,
    periodic_to_open_velocity,
    validate_inflow_plane,
)
from .operators import (
    advection,
    cell_velocity,
    courant_number,
    diffusion,
    divergence,
    kinetic_energy,
    pressure_gradient,
    stable_timestep,
    tangential_z_gradient,
)
from .poisson import (
    CLASSICAL_AMG_PCG,
    PressurePoisson,
    SparseMatrix,
    assemble_pressure_matrix,
    build_amg_solver,
    build_fft_solver,
    build_gmg_solver,
    build_pressure_poisson,
    default_tolerance,
    matrix_vector_product,
    project,
)
from .rotation import CoriolisGeostrophic, coriolis_tendency
from .scalar import PassiveScalar, scalar_tendency
from .sgs import (
    AnisotropicMinimumDissipation,
    eddy_viscosity,
    edge_gradients,
    stress_divergence,
    subfilter_tendency,
)
from .surface import (
    MoninObukhovSurface,
    SurfaceExchange,
    coupled_surface_exchange,
    surface_momentum_tendency,
)
from .turbine import build_adbem_forcing
from .sponge import (
    PLANE_MEAN,
    REST,
    rayleigh_sponge_tendency,
)
from .wall import (
    CELL_AVERAGE,
    CELL_CENTRE,
    LOCAL,
    PLANAR,
    MoninObukhovWall,
    friction_velocity,
    logarithmic_profile,
    monin_obukhov_boundaries,
    surface_stress,
    wall_tendency,
)
from .state import (
    FREE_SLIP,
    NO_SLIP,
    OPEN,
    PERIODIC,
    Boundaries,
    StaggeredVelocity,
    Wall,
    cell_coordinates,
    cell_shape,
    enforce_impermeability,
    face_coordinates,
    streamwise_is_periodic,
    validate,
    x_face_shape,
    z_face_shape,
    zeros,
)

__all__ = [
    "CELL_AVERAGE",
    "CELL_CENTRE",
    "CLASSICAL_AMG_PCG",
    "FREE_SLIP",
    "LOCAL",
    "LinearBoussinesqBuoyancy",
    "PLANAR",
    "AnisotropicMinimumDissipation",
    "AtmosphericSolution",
    "CoriolisGeostrophic",
    "MoninObukhovSurface",
    "MoninObukhovWall",
    "NO_SLIP",
    "OPEN",
    "PERIODIC",
    "InflowPlane",
    "PLANE_MEAN",
    "REST",
    "Boundaries",
    "FlowModel",
    "PressurePoisson",
    "PassiveScalar",
    "Solution",
    "SparseMatrix",
    "SurfaceExchange",
    "StaggeredVelocity",
    "Wall",
    "advection",
    "assemble_pressure_matrix",
    "atmospheric_history_diagnostics",
    "atmospheric_profile_diagnostics",
    "build_amg_solver",
    "build_adbem_forcing",
    "build_adaptive_atmospheric_run",
    "build_adaptive_run",
    "build_atmospheric_run",
    "build_atmospheric_step",
    "build_open_atmospheric_run",
    "build_open_atmospheric_step",
    "boussinesq_tendency",
    "build_fft_solver",
    "build_gmg_solver",
    "build_pressure_poisson",
    "build_run",
    "build_step",
    "build_tendency",
    "cell_coordinates",
    "cell_shape",
    "cell_velocity",
    "courant_number",
    "coriolis_tendency",
    "coupled_surface_exchange",
    "default_tolerance",
    "diffusion",
    "divergence",
    "eddy_viscosity",
    "edge_gradients",
    "enforce_impermeability",
    "enforce_open_scalar",
    "enforce_open_velocity",
    "extract_inflow_plane",
    "face_coordinates",
    "friction_velocity",
    "initial_solution",
    "initial_atmospheric_solution",
    "kinetic_energy",
    "logarithmic_profile",
    "matrix_vector_product",
    "monin_obukhov_boundaries",
    "periodic_to_open_velocity",
    "pressure_gradient",
    "project",
    "rayleigh_sponge_tendency",
    "scalar_tendency",
    "stable_timestep",
    "stress_divergence",
    "subfilter_tendency",
    "surface_momentum_tendency",
    "surface_stress",
    "tangential_z_gradient",
    "streamwise_is_periodic",
    "validate",
    "validate_inflow_plane",
    "x_face_shape",
    "wall_tendency",
    "z_face_shape",
    "zeros",
]

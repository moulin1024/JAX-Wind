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

from .integrate import (
    FlowModel,
    Solution,
    build_run,
    build_step,
    build_tendency,
    initial_solution,
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
from .sgs import (
    AnisotropicMinimumDissipation,
    eddy_viscosity,
    edge_gradients,
    stress_divergence,
    subfilter_tendency,
)
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
    Boundaries,
    StaggeredVelocity,
    Wall,
    cell_coordinates,
    cell_shape,
    enforce_impermeability,
    face_coordinates,
    validate,
    z_face_shape,
    zeros,
)

__all__ = [
    "CELL_AVERAGE",
    "CELL_CENTRE",
    "CLASSICAL_AMG_PCG",
    "FREE_SLIP",
    "LOCAL",
    "PLANAR",
    "AnisotropicMinimumDissipation",
    "MoninObukhovWall",
    "NO_SLIP",
    "PLANE_MEAN",
    "REST",
    "Boundaries",
    "FlowModel",
    "PressurePoisson",
    "Solution",
    "SparseMatrix",
    "StaggeredVelocity",
    "Wall",
    "advection",
    "assemble_pressure_matrix",
    "build_amg_solver",
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
    "default_tolerance",
    "diffusion",
    "divergence",
    "eddy_viscosity",
    "edge_gradients",
    "enforce_impermeability",
    "face_coordinates",
    "friction_velocity",
    "initial_solution",
    "kinetic_energy",
    "logarithmic_profile",
    "matrix_vector_product",
    "monin_obukhov_boundaries",
    "pressure_gradient",
    "project",
    "rayleigh_sponge_tendency",
    "stable_timestep",
    "stress_divergence",
    "subfilter_tendency",
    "surface_stress",
    "tangential_z_gradient",
    "validate",
    "wall_tendency",
    "z_face_shape",
    "zeros",
]

"""Monin-Obukhov surface stress for the finite-volume solver.

In a wall-modelled large-eddy simulation the near-wall flow is not resolved,
so the surface exerts a parameterised drag instead of a viscous stress.  Under
neutral stratification Monin-Obukhov similarity reduces to the logarithmic law
and the surface stress follows from the wind at one reference height,

    u_*  = kappa * U(z_1) / ln(z_1 / z_0),
    tau  = u_*^2 * (u, v) / U,

which enters a finite-volume momentum balance directly as the flux through the
bottom face of the wall-adjacent control volume.  That is the natural place
for it: no ghost value is invented, and the stress the model returns is the
stress the discrete equations feel.

The reference height needs care in a finite-volume code, and this is where the
finite-volume treatment departs from the finite-difference one.  A
finite-volume unknown is the average of the profile over its cell, whereas the
logarithmic law is a point value.  Because the logarithm is concave, the cell
average is smaller than the value at the cell centre, so feeding the average
into the law as though it were the centre value underestimates the surface
stress (Clement, Lemarie and Blayo, arXiv:2305.09254).  Integrating the law
over the first cell gives the height at which the two agree exactly,

    (1/dz) * integral_0^dz ln(z/z0) dz = ln(dz/z0) - 1  =>  z_1 = dz / e,

so ``CELL_AVERAGE`` sampling evaluates the law at ``dz / e`` instead of the
``dz / 2`` that a finite-difference code would use.  Both are available:
``CELL_CENTRE`` reproduces the usual finite-difference convention.

The same cell-average-versus-point-value distinction reappears in the *shear*
the subfilter closure reads, and there it is larger.  Differencing cell
averages across the first interior face overestimates the logarithmic gradient
by ``ln 4 = 1.39``, where differencing point values overestimates it by only
``ln 3 = 1.10``; ``log_law_gradient_correction`` removes that bias, and
``log_law_face_ratio`` carries the derivation.  Enable it with
``gradient_correction=True`` -- it is off by default because it changes the
closure's input, not just a boundary value.

References
----------
Schumann (1975); Moeng, J. Atmos. Sci. 41, 2052 (1984) -- the surface stress
parameterisation and its planar-averaged variant.
Bou-Zeid, Meneveau and Parlange, Phys. Fluids 17, 025105 (2005) -- a local law
of the wall overpredicts the stress, which filtering the input wind mitigates.
Kawai and Larsson, Phys. Fluids 24, 015105 (2012) -- log-layer mismatch and the
contamination of the wall-adjacent LES data.
Clement, Lemarie and Blayo, arXiv:2305.09254 (2023) -- the finite-volume
reconstruction of the surface layer inside the first cell.
Porte-Agel, Meneveau and Parlange, J. Fluid Mech. 415, 261 (2000), Appendix --
the near-wall gradient correction, in its finite-difference form.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import jax.numpy as jnp

from jaxwind.domain.grid import Grid

from .state import FREE_SLIP, Boundaries, StaggeredVelocity, Wall
from .surface_layer import face_ratio, face_ratio_array


CELL_AVERAGE = "cell-average"
CELL_CENTRE = "cell-centre"

LOCAL = "local"
PLANAR = "planar"


@dataclass(frozen=True, slots=True)
class MoninObukhovWall:
    """A rough wall whose drag follows the neutral logarithmic law.

    ``sampling`` selects the reference height inside the first cell, and
    ``averaging`` selects between the local law of the wall and the
    planar-averaged wind speed of Moeng (1984), which removes the bias a local
    law carries from the correlation between the speed and its fluctuations.
    """

    roughness: float
    von_karman: float = 0.4
    sampling: str = CELL_AVERAGE
    averaging: str = LOCAL
    gradient_correction: bool = False
    corrected_faces: int = 3

    def __post_init__(self) -> None:
        if self.roughness <= 0.0:
            raise ValueError("the roughness length must be positive")
        if self.von_karman <= 0.0:
            raise ValueError("the von Karman constant must be positive")
        if self.sampling not in (CELL_AVERAGE, CELL_CENTRE):
            raise ValueError(f"unsupported sampling: {self.sampling!r}")
        if self.averaging not in (LOCAL, PLANAR):
            raise ValueError(f"unsupported averaging: {self.averaging!r}")
        if self.corrected_faces < 1:
            raise ValueError("corrected_faces must be at least one")

    def reference_height(self, grid: Grid) -> float:
        """Height inside the first cell at which the law is evaluated."""
        if self.sampling == CELL_AVERAGE:
            return float(grid.z_widths[0]) / math.e
        return 0.5 * float(grid.z_widths[0])

    def drag_coefficient(self, grid: Grid) -> float:
        """The factor relating ``U * (u, v)`` to the surface stress."""
        height = self.reference_height(grid)
        if height <= self.roughness:
            raise ValueError(
                "the first cell is not tall enough for the logarithmic law: "
                f"reference height {height:g} is below the roughness "
                f"{self.roughness:g}"
            )
        return (self.von_karman / math.log(height / self.roughness)) ** 2


def log_law_face_ratio(face: int, mesh_stability: float = 0.0) -> float:
    """Discrete-to-true wall-normal gradient ratio on interior face ``m``.

    Thin wrapper over :func:`jaxwind.surface_layer.face_ratio`, which carries
    the derivation and the closed form.  ``mesh_stability`` is ``s = h / L``,
    the cell height in Obukhov units; the default of zero is the neutral case,
    where the ratio is ``m Lambda(m)`` and the first face gives ``ln 4``.

    This wall model is the neutral one, so callers that know the Obukhov length
    should pass it -- the neutral factor over-corrects in stable air and
    under-corrects in unstable air by roughly ten percent at GABLS1-like
    stability.
    """
    return face_ratio(face, mesh_stability)


def log_law_gradient_correction(
    gradients: dict[str, jnp.ndarray],
    velocity: StaggeredVelocity,
    grid: Grid,
    model: MoninObukhovWall,
    *,
    mesh_stability=0.0,
) -> dict[str, jnp.ndarray]:
    """Impose the logarithmic near-wall variation on the shear the closure sees.

    This adjusts only the wall-normal gradients ``du/dz`` and ``dv/dz`` that the
    subfilter model reads.  The momentum boundary condition is untouched: the
    surface stress still comes from :func:`surface_stress`, so the correction
    changes what the closure believes about the unresolved shear, not what the
    wall does to the flow.

    Two distinct adjustments
    ------------------------
    **Interior faces** are rescaled by ``1 / ratio(m)`` from
    :func:`log_law_face_ratio`.  Following the original appendix the rescaling
    is applied to the **plane mean only** -- the horizontal directions are
    statistically homogeneous here, so the mean is what similarity theory makes
    a statement about, while the fluctuations about it carry the resolved
    turbulence the closure is meant to act on.  Rescaling those too would damp
    the very structures being modelled.

    **The wall face** cannot be rescaled, because the discrete gradient there is
    not a bad estimate of a finite quantity -- the log-law gradient
    ``u*/(kappa z)`` diverges as ``z -> 0``, so there is nothing to rescale
    towards.  It is replaced instead by the logarithmic gradient evaluated at
    the height the first cell average actually represents.  For
    ``CELL_AVERAGE`` sampling that height is ``dz/e``, the same height at which
    :meth:`MoninObukhovWall.drag_coefficient` evaluates the law, which keeps the
    stress and the gradient telling the closure a consistent story.  This is a
    deliberate departure from the finite-difference convention, which uses the
    cell centre ``dz/2``; the two differ by ``e/2 = 1.36``, and which is right
    is a modelling choice worth testing on a case with known statistics.

    The friction velocity comes from the same drag coefficient and the same
    ``LOCAL`` / ``PLANAR`` averaging as the surface stress, and the corrected
    wall gradient is aligned with the wall-adjacent wind, so a turning wind
    keeps the stress and the shear parallel.

    Limitations
    -----------
    The ratios assume a *uniform* vertical mesh and a *neutral* surface layer.
    On a stretched grid the telescoping above no longer holds, and under
    stratification the profile follows ``phi_m(z/L)`` rather than the neutral
    log law, so both the ratios and the wall-face value would need the
    stability correction.  Neither case is detected here.
    """
    corrected = dict(gradients)
    height = model.reference_height(grid)
    centred_x, centred_y, speed = _first_level_speed(velocity.x[0], velocity.y[0])
    if model.averaging == PLANAR:
        speed = jnp.mean(speed)
    tiny = jnp.finfo(speed.dtype).tiny
    safe_speed = jnp.maximum(speed, tiny)
    # u* from the same law the surface stress uses, so the two stay consistent.
    friction = math.sqrt(model.drag_coefficient(grid)) * speed
    wall_slope = friction / (model.von_karman * height)
    for key, component in (("xz", centred_x), ("yz", centred_y)):
        field = corrected[key]
        # Wall face: replace outright, aligned with the wall-adjacent wind.
        value = jnp.where(
            speed > tiny, wall_slope * component / safe_speed, 0.0
        )
        if field.shape[-1] == value.shape[-1] + 1:
            value = _open_x_faces(value)
        field = field.at[0].set(value)
        # Interior faces: shift the plane mean, leave the fluctuations alone.
        # The top face is excluded, hence the ``- 2``.
        faces = min(model.corrected_faces, field.shape[0] - 2)
        for face in range(1, faces + 1):
            # face_ratio_array keeps this valid when the Obukhov length is a
            # traced array; it reduces to log_law_face_ratio at s = 0.
            ratio = face_ratio_array(face, mesh_stability)
            plane = jnp.mean(field[face])
            field = field.at[face].add((1.0 / ratio - 1.0) * plane)
        corrected[key] = field
    return corrected


def _open_x_faces(values: jnp.ndarray) -> jnp.ndarray:
    """Interpolate cell values to the distinct faces of an open x domain."""
    interior = 0.5 * (values[..., :-1] + values[..., 1:])
    return jnp.concatenate(
        (values[..., :1], interior, values[..., -1:]), axis=-1
    )


def _first_level_speed(
    x_velocity: jnp.ndarray,
    y_velocity: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Cell-centred wind components and speed on the wall-adjacent level."""
    centred_x = (
        0.5 * (x_velocity[..., :-1] + x_velocity[..., 1:])
        if x_velocity.shape[-1] == y_velocity.shape[-1] + 1
        else 0.5 * (x_velocity + jnp.roll(x_velocity, -1, axis=1))
    )
    centred_y = (
        0.5 * (y_velocity[:-1] + y_velocity[1:])
        if y_velocity.shape[0] == x_velocity.shape[0] + 1
        else 0.5 * (y_velocity + jnp.roll(y_velocity, -1, axis=0))
    )
    speed = jnp.sqrt(centred_x**2 + centred_y**2 + 1.0e-20)
    return centred_x, centred_y, speed


def surface_stress(
    velocity: StaggeredVelocity,
    grid: Grid,
    model: MoninObukhovWall,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the surface stress on the x-faces and the y-faces.

    The stress is formed at cell centres, where the wind speed is defined
    without bias, and then interpolated to the faces that carry the two
    horizontal momentum components.
    """
    first_x, first_y = velocity.x[0], velocity.y[0]
    centred_x, centred_y, speed = _first_level_speed(first_x, first_y)
    if model.averaging == PLANAR:
        speed = jnp.mean(speed)
    coefficient = model.drag_coefficient(grid)
    stress_x = coefficient * speed * centred_x
    stress_y = coefficient * speed * centred_y
    if velocity.x.shape[-1] == stress_x.shape[-1] + 1:
        stress_x_faces = _open_x_faces(stress_x)
    else:
        stress_x_faces = 0.5 * (stress_x + jnp.roll(stress_x, 1, axis=1))
    if velocity.y.shape[1] == stress_y.shape[0] + 1:
        interior_y = 0.5 * (stress_y[:-1] + stress_y[1:])
        side = jnp.zeros_like(stress_y[:1])
        stress_y_faces = jnp.concatenate((side, interior_y, side), axis=0)
    else:
        stress_y_faces = 0.5 * (stress_y + jnp.roll(stress_y, 1, axis=0))
    return stress_x_faces, stress_y_faces


def _side_plane_stress(
    u: jnp.ndarray,
    w: jnp.ndarray,
    cell_width: float,
    model: MoninObukhovWall,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Tangential stress on one y wall at z-x cell centres."""
    speed = jnp.sqrt(u**2 + w**2 + 1.0e-20)
    if model.averaging == PLANAR:
        speed = jnp.mean(speed)
    height = cell_width / math.e if model.sampling == CELL_AVERAGE else 0.5 * cell_width
    if height <= model.roughness:
        raise ValueError("the first side-wall cell is below the roughness length")
    coefficient = (model.von_karman / math.log(height / model.roughness)) ** 2
    return coefficient * speed * u, coefficient * speed * w


def sidewall_stress(
    velocity: StaggeredVelocity,
    grid: Grid,
    model: MoninObukhovWall,
) -> tuple[tuple[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]:
    """Return lower/upper y-wall stresses for tangential u and w."""
    if velocity.y.shape[1] != grid.ny + 1:
        raise ValueError("side-wall stress requires distinct y boundary faces")
    u_cells = (
        0.5 * (velocity.x[..., :-1] + velocity.x[..., 1:])
        if velocity.x.shape[-1] == grid.nx + 1
        else 0.5 * (velocity.x + jnp.roll(velocity.x, -1, axis=2))
    )
    w_cells = 0.5 * (velocity.z[:-1] + velocity.z[1:])
    return (
        _side_plane_stress(
            u_cells[:, 0], w_cells[:, 0], float(grid.y_widths[0]), model
        ),
        _side_plane_stress(
            u_cells[:, -1], w_cells[:, -1], float(grid.y_widths[-1]), model
        ),
    )


def sidewall_tendency(
    velocity: StaggeredVelocity,
    grid: Grid,
    model: MoninObukhovWall,
) -> StaggeredVelocity:
    """Apply log-law drag in cells adjacent to both y side walls."""
    lower, upper = sidewall_stress(velocity, grid, model)
    lower_u, lower_w = lower
    upper_u, upper_w = upper
    if velocity.x.shape[-1] == grid.nx + 1:
        def x_faces(stress):
            interior = 0.5 * (stress[..., :-1] + stress[..., 1:])
            return jnp.concatenate((stress[..., :1], interior, stress[..., -1:]), axis=1)
    else:
        def x_faces(stress):
            return 0.5 * (stress + jnp.roll(stress, 1, axis=1))
    def z_faces(stress):
        interior = 0.5 * (stress[:-1] + stress[1:])
        wall = jnp.zeros_like(stress[:1])
        return jnp.concatenate((wall, interior, wall), axis=0)
    x = jnp.zeros_like(velocity.x)
    x = x.at[:, 0].add(-x_faces(lower_u) / float(grid.y_widths[0]))
    x = x.at[:, -1].add(-x_faces(upper_u) / float(grid.y_widths[-1]))
    z = jnp.zeros_like(velocity.z)
    z = z.at[:, 0].add(-z_faces(lower_w) / float(grid.y_widths[0]))
    z = z.at[:, -1].add(-z_faces(upper_w) / float(grid.y_widths[-1]))
    return StaggeredVelocity(x, jnp.zeros_like(velocity.y), z)


def friction_velocity(
    velocity: StaggeredVelocity,
    grid: Grid,
    model: MoninObukhovWall,
) -> jnp.ndarray:
    """Planar-averaged friction velocity implied by the surface stress."""
    _, _, speed = _first_level_speed(velocity.x[0], velocity.y[0])
    if model.averaging == PLANAR:
        speed = jnp.mean(speed)
    return jnp.sqrt(jnp.mean(model.drag_coefficient(grid) * speed**2))


def wall_tendency(
    velocity: StaggeredVelocity,
    grid: Grid,
    model: MoninObukhovWall,
) -> StaggeredVelocity:
    """Momentum tendency from the surface drag on the wall-adjacent cells.

    The stress is the flux through the bottom face of the first control
    volume, so it reaches only that cell, divided by the cell height.
    """
    stress_x, stress_y = surface_stress(velocity, grid, model)
    x_tendency = jnp.zeros_like(velocity.x).at[0].set(
        -stress_x / float(grid.z_widths[0])
    )
    y_tendency = jnp.zeros_like(velocity.y).at[0].set(
        -stress_y / float(grid.z_widths[0])
    )
    return StaggeredVelocity(
        x_tendency,
        y_tendency,
        jnp.zeros_like(velocity.z),
    )


def logarithmic_profile(
    grid: Grid,
    friction: float,
    model: MoninObukhovWall,
) -> jnp.ndarray:
    """Cell-averaged logarithmic wind profile, the equilibrium of the model.

    The average over each cell is used rather than the value at the centre, so
    that the profile is the exact discrete equilibrium of a finite-volume
    solver using ``CELL_AVERAGE`` sampling.
    """
    upper = jnp.asarray(grid.z_faces[1:])
    lower = jnp.asarray(grid.z_faces[:-1])
    roughness = model.roughness

    def integral(height):
        # The antiderivative z * (ln(z / z0) - 1), whose limit at z = 0 is
        # zero. The logarithm is evaluated on a dummy height there so that no
        # infinity is formed: a small positive guard would underflow to zero
        # in single precision and produce a NaN.
        positive = height > 0.0
        safe = jnp.where(positive, height, 1.0)
        return jnp.where(positive, safe * (jnp.log(safe / roughness) - 1.0), 0.0)

    averaged = (integral(upper) - integral(lower)) / jnp.asarray(grid.z_widths)
    return friction / model.von_karman * jnp.maximum(averaged, 0.0)


def monin_obukhov_boundaries() -> Boundaries:
    """Boundaries whose resolved viscous flux vanishes on both walls.

    The surface model supplies the entire wall stress, so the resolved viscous
    closure must not add a second one; the top is frictionless by assumption.
    """
    return Boundaries(Wall(FREE_SLIP), Wall(FREE_SLIP))


__all__ = [
    "CELL_AVERAGE",
    "CELL_CENTRE",
    "LOCAL",
    "PLANAR",
    "MoninObukhovWall",
    "friction_velocity",
    "log_law_face_ratio",
    "log_law_gradient_correction",
    "logarithmic_profile",
    "monin_obukhov_boundaries",
    "sidewall_stress",
    "sidewall_tendency",
    "surface_stress",
    "wall_tendency",
]

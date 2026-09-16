"""Frame sampling geometry shared by simulations and postprocessing."""
from __future__ import annotations
from typing import Any
import numpy as np

def frame_steps(steps: int, count: int) -> tuple[int, ...]:
    """Return unique, evenly spaced one-based capture steps."""
    if count == 0:
        return ()
    return tuple(
        int(value)
        for value in ((np.arange(count, dtype=np.int64) + 1) * steps // count)
    )


def build_frame_capture(grid, *, y_m: float, z_m: float):
    """Compile extraction of velocity and scalar streamwise slices."""
    import jax
    import jax.numpy as jnp

    z_centers = np.asarray(grid.z_centers, dtype=np.float64)
    z_upper = int(np.searchsorted(z_centers, z_m, side="right"))
    z_upper = min(max(z_upper, 1), grid.nz - 1)
    z_lower = z_upper - 1
    if z_m <= z_centers[0]:
        z_lower = z_upper = 0
        z_weight = 0.0
    elif z_m >= z_centers[-1]:
        z_lower = z_upper = grid.nz - 1
        z_weight = 0.0
    else:
        z_weight = (z_m - z_centers[z_lower]) / (
            z_centers[z_upper] - z_centers[z_lower]
        )
    y_index = y_m / grid.dy - 0.5
    y_floor = np.floor(y_index)
    y_lower = int(y_floor) % grid.ny
    y_upper = (y_lower + 1) % grid.ny
    y_weight = y_index - y_floor

    def capture(x_faces, scalar_field):
        hub_faces = (
            (1.0 - z_weight) * x_faces[z_lower]
            + z_weight * x_faces[z_upper]
        )
        centre_faces = (
            (1.0 - y_weight) * x_faces[:, y_lower]
            + y_weight * x_faces[:, y_upper]
        )
        scalar_hub = (
            (1.0 - z_weight) * scalar_field[z_lower]
            + z_weight * scalar_field[z_upper]
        )
        scalar_centre = (
            (1.0 - y_weight) * scalar_field[:, y_lower]
            + y_weight * scalar_field[:, y_upper]
        )

        def cell_centered(values):
            if values.shape[-1] == grid.nx + 1:
                return 0.5 * (values[..., :-1] + values[..., 1:])
            return 0.5 * (values + jnp.roll(values, -1, axis=-1))

        return (
            cell_centered(hub_faces),
            cell_centered(centre_faces),
            scalar_hub,
            scalar_centre,
        )

    return jax.jit(capture)


def capture_frame(
    solution,
    grid,
    *,
    y_m: float,
    z_m: float,
    capture=None,
) -> dict[str, Any]:
    """Copy only the four saved streamwise slices from device to host."""
    if capture is None:
        capture = build_frame_capture(grid, y_m=y_m, z_m=z_m)
    velocity_hub, velocity_centre, scalar_hub, scalar_centre = capture(
        solution.velocity.x,
        solution.scalar,
    )
    result = {
        "u_hub_yx": np.asarray(velocity_hub),
        "u_center_zx": np.asarray(velocity_centre),
        "scalar_hub_yx": np.asarray(scalar_hub),
        "scalar_center_zx": np.asarray(scalar_centre),
        "time_seconds": float(solution.time),
        "step": int(solution.step),
    }
    if hasattr(solution, "moisture"):
        for name in solution.moisture._fields:
            _, _, hub, centre = capture(solution.velocity.x, getattr(solution.moisture, name))
            result[f"{name}_hub_yx"] = np.asarray(hub)
            result[f"{name}_center_zx"] = np.asarray(centre)
    return result

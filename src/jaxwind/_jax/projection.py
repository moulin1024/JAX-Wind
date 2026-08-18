"""Base-grid horizontal projection kernels for the JAX solver."""

from __future__ import annotations

import jax.numpy as jnp


def build_projection_kernels(
    *,
    grid,
    kx,
    ky,
    keep,
    state_keep,
    divergence_local,
    enforce_upper_boundary_local,
    pressure_gradient_local,
):
    """Build projection and the distinct legacy pre-RHS state filter."""

    def horizontal_divergence_local(x_velocity, y_velocity):
        x_spectrum = jnp.fft.rfftn(x_velocity, axes=(-2, -1))
        y_spectrum = jnp.fft.rfftn(y_velocity, axes=(-2, -1))
        spectrum = (
            1j * kx[None, None, :] * x_spectrum
            + 1j * ky[None, :, None] * y_spectrum
        ) * keep[None, ...]
        return jnp.fft.irfftn(
            spectrum,
            s=(grid.ny, grid.nx),
            axes=(-2, -1),
        ).astype(x_velocity.dtype)

    def horizontal_gradient_local(pressure):
        spectrum = jnp.fft.rfftn(pressure, axes=(-2, -1)) * keep[None, ...]
        gradients = jnp.fft.irfftn(
            jnp.stack(
                (
                    spectrum * (1j * kx[None, None, :]),
                    spectrum * (1j * ky[None, :, None]),
                ),
                axis=0,
            ),
            s=(grid.ny, grid.nx),
            axes=(-2, -1),
        ).astype(pressure.dtype)
        return gradients[0], gradients[1]

    def filter_state_local(values):
        spectrum = jnp.fft.rfftn(values, axes=(-2, -1))
        return jnp.fft.irfftn(
            spectrum * state_keep[None, ...],
            s=(grid.ny, grid.nx),
            axes=(-2, -1),
        ).astype(values.dtype)

    def filter_state_velocity_local(x, y, z):
        filtered = filter_state_local(jnp.stack((x, y, z), axis=0))
        return filtered[0], filtered[1], filtered[2]

    def filter_projection_local(values):
        spectrum = jnp.fft.rfftn(values, axes=(-2, -1))
        return jnp.fft.irfftn(
            spectrum * keep[None, ...],
            s=(grid.ny, grid.nx),
            axes=(-2, -1),
        ).astype(values.dtype)

    def filter_projection_velocity_local(x, y, z):
        filtered = filter_projection_local(jnp.stack((x, y, z), axis=0))
        return filtered[0], filtered[1], filtered[2]

    def prepare_projection_local(
        x_velocity,
        y_velocity,
        z_velocity,
        lower_boundary,
        upper_boundary,
    ):
        velocity_spectra = jnp.fft.rfftn(
            jnp.stack((x_velocity, y_velocity, z_velocity), axis=0),
            axes=(-2, -1),
        )
        filtered_spectra = velocity_spectra * keep[None, None, ...]
        horizontal_divergence_spectrum = (
            1j * kx[None, None, :] * filtered_spectra[0]
            + 1j * ky[None, :, None] * filtered_spectra[1]
        ) * keep[None, ...]
        transformed = jnp.fft.irfftn(
            jnp.stack(
                (filtered_spectra[2], horizontal_divergence_spectrum),
                axis=0,
            ),
            s=(grid.ny, grid.nx),
            axes=(-2, -1),
        ).astype(x_velocity.dtype)
        filtered_z = enforce_upper_boundary_local(transformed[0], upper_boundary)
        vertical_divergence = divergence_local(filtered_z, lower_boundary)
        return (
            filtered_spectra[0],
            filtered_spectra[1],
            filtered_z,
            transformed[1] + vertical_divergence,
        )

    def finish_projection_local(
        candidate_x_spectrum,
        candidate_y_spectrum,
        candidate_z,
        pressure,
        dt,
    ):
        local_dt = jnp.asarray(dt, dtype=pressure.dtype)
        pressure_spectrum = jnp.fft.rfftn(pressure, axes=(-2, -1)) * keep[None, ...]
        corrected_horizontal = jnp.fft.irfftn(
            jnp.stack(
                (
                    candidate_x_spectrum
                    - local_dt * pressure_spectrum * (1j * kx[None, None, :]),
                    candidate_y_spectrum
                    - local_dt * pressure_spectrum * (1j * ky[None, :, None]),
                ),
                axis=0,
            ),
            s=(grid.ny, grid.nx),
            axes=(-2, -1),
        ).astype(pressure.dtype)
        gradient_z = pressure_gradient_local(pressure, 0.0)
        return (
            corrected_horizontal[0],
            corrected_horizontal[1],
            candidate_z - local_dt * gradient_z,
        )

    def filter_boundary(boundary):
        plane = jnp.broadcast_to(jnp.asarray(boundary), (grid.ny, grid.nx))
        spectrum = jnp.fft.rfftn(plane, axes=(-2, -1))
        return jnp.fft.irfftn(
            spectrum * keep,
            s=(grid.ny, grid.nx),
            axes=(-2, -1),
        ).astype(plane.dtype)

    def correct_local(
        candidate_x,
        candidate_y,
        candidate_z,
        gradient_x,
        gradient_y,
        gradient_z,
        lower_boundary,
        lower_gradient,
        dt,
    ):
        local_dt = jnp.asarray(dt, dtype=candidate_x.dtype)
        return (
            candidate_x - local_dt * gradient_x,
            candidate_y - local_dt * gradient_y,
            candidate_z - local_dt * gradient_z,
            jnp.asarray(lower_boundary, dtype=candidate_x.dtype)
            - local_dt * jnp.asarray(lower_gradient, dtype=candidate_x.dtype),
        )

    return (
        horizontal_divergence_local,
        horizontal_gradient_local,
        filter_state_local,
        filter_state_velocity_local,
        filter_projection_velocity_local,
        prepare_projection_local,
        finish_projection_local,
        filter_boundary,
        correct_local,
    )

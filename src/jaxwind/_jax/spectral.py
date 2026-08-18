"""Legacy WIRE-LES horizontal spectral kernels shared by the JAX solver."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class HorizontalSpectralKernels(NamedTuple):
    kx: object
    ky: object
    keep: object
    state_keep: object
    spectrum: object
    gradient_pair: object
    spectral_flux_divergence: object
    flux_divergence: object
    wall_filter: object


def build_horizontal_spectral_kernels(grid):
    """Build the single legacy base-grid spectral implementation.

    WIRE-LES applies its exact ``nint(N/3)`` box cutoff to prognostic fields
    immediately before the RHS.  Nonlinear products remain on the base grid;
    only the FFT Nyquist modes are suppressed by derivative/projection kernels.
    """

    kx = 2.0 * jnp.pi * jnp.fft.rfftfreq(grid.nx, d=grid.lx / grid.nx)
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(grid.ny, d=grid.ly / grid.ny)
    keep = jnp.ones((grid.ny, grid.nx // 2 + 1))
    if grid.nx % 2 == 0:
        kx = kx.at[-1].set(0.0)
        keep = keep.at[:, -1].set(0.0)
    if grid.ny % 2 == 0:
        ky = ky.at[grid.ny // 2].set(0.0)
        keep = keep.at[grid.ny // 2, :].set(0.0)

    x_mode = jnp.arange(grid.nx // 2 + 1)
    y_mode = jnp.fft.fftfreq(grid.ny) * grid.ny
    legacy_x_cutoff = round(grid.nx / 3.0)
    legacy_y_cutoff = round(grid.ny / 3.0)
    state_keep = (
        (x_mode[None, :] < legacy_x_cutoff)
        & (jnp.abs(y_mode)[:, None] < legacy_y_cutoff)
    )

    def horizontal_spectrum_local(values):
        spectrum = jnp.fft.rfftn(values, axes=(-2, -1))
        return spectrum * keep.astype(values.real.dtype)

    def inverse_spectrum_local(spectrum, dtype):
        return jnp.fft.irfftn(
            spectrum,
            s=(grid.ny, grid.nx),
            axes=(-2, -1),
        ).astype(dtype)

    def horizontal_gradient_pair_local(values):
        spectrum = jnp.fft.rfftn(values, axes=(-2, -1))
        spectra = jnp.stack(
            (
                spectrum * (1j * kx.astype(values.real.dtype)),
                spectrum * (1j * ky.astype(values.real.dtype)[:, None]),
            ),
            axis=0,
        ) * keep.astype(values.real.dtype)
        gradients = jnp.fft.irfftn(
            spectra,
            s=(grid.ny, grid.nx),
            axes=(-2, -1),
        ).astype(values.dtype)
        return gradients[0], gradients[1]

    def flux_divergence_spectrum(x_spectra, y_spectra, dtype):
        return (
            x_spectra * (1j * kx.astype(dtype))
            + y_spectra * (1j * ky.astype(dtype)[:, None])
        ) * keep.astype(dtype)

    def spectral_flux_divergence(x_spectra, y_spectra, dtype):
        return inverse_spectrum_local(
            flux_divergence_spectrum(x_spectra, y_spectra, dtype),
            dtype,
        )

    def horizontal_flux_divergence_local(x_fluxes, y_fluxes):
        count = x_fluxes.shape[0]
        spectra = jnp.fft.rfftn(
            jnp.concatenate((x_fluxes, y_fluxes), axis=0),
            axes=(-2, -1),
        )
        return spectral_flux_divergence(
            spectra[:count], spectra[count:], x_fluxes.dtype
        )

    def wall_filter_local(values, filter_width):
        spectrum = jnp.fft.rfftn(values, axes=(-2, -1))
        cutoff_x = jnp.floor(grid.nx / (2.0 * filter_width))
        cutoff_y = jnp.floor(grid.ny / (2.0 * filter_width))
        wall_keep = (
            (jnp.abs(y_mode)[:, None] < cutoff_y)
            & (x_mode[None, :] < cutoff_x)
        )
        return inverse_spectrum_local(spectrum * wall_keep, values.dtype)

    return HorizontalSpectralKernels(
        kx,
        ky,
        keep,
        state_keep,
        horizontal_spectrum_local,
        horizontal_gradient_pair_local,
        spectral_flux_divergence,
        horizontal_flux_divergence_local,
        wall_filter_local,
    )

"""Horizontal spectral kernels shared by the z-slab interpreter."""

from __future__ import annotations

import math
from typing import NamedTuple

import jax.numpy as jnp


class HorizontalSpectralKernels(NamedTuple):
    kx: object
    ky: object
    keep: object
    state_keep: object
    pad: object
    project_spectrum: object
    truncate: object
    derivative: object
    gradient_pair: object
    padded_gradient_pair: object
    spectral_flux_divergence: object
    flux_divergence: object
    padded_flux_divergence: object
    wall_filter: object


def build_horizontal_spectral_kernels(grid, nonlinear_padding_ratio):
    """Build base-grid and three-halves horizontal FFT operations."""
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
    padded_ny = int(math.ceil(nonlinear_padding_ratio * grid.ny))
    padded_nx = int(math.ceil(nonlinear_padding_ratio * grid.nx))
    padded_kx = 2.0 * jnp.pi * jnp.fft.rfftfreq(padded_nx, d=grid.lx / padded_nx)
    padded_ky = 2.0 * jnp.pi * jnp.fft.fftfreq(padded_ny, d=grid.ly / padded_ny)
    if padded_nx % 2 == 0:
        padded_kx = padded_kx.at[-1].set(0.0)
    if padded_ny % 2 == 0:
        padded_ky = padded_ky.at[padded_ny // 2].set(0.0)
    pad_y_before = padded_ny // 2 - grid.ny // 2
    pad_y_after = padded_ny - grid.ny - pad_y_before
    padded_half_nx = padded_nx // 2 + 1
    half_x_after = padded_half_nx - (grid.nx // 2 + 1)
    base_y_index = jnp.arange(grid.ny)
    base_y_mode = jnp.where(
        base_y_index <= (grid.ny - 1) // 2,
        base_y_index,
        base_y_index - grid.ny,
    )
    base_y_in_padded = base_y_mode % padded_ny
    opposite_base_y_in_padded = (-base_y_mode) % padded_ny

    def pad_horizontal_local(values):
        spectrum = jnp.fft.rfftn(values, axes=(-2, -1))
        spectrum = spectrum * keep.astype(values.real.dtype)
        shifted = jnp.fft.fftshift(spectrum, axes=(-2,))
        padded = jnp.pad(
            shifted,
            ((0, 0),) * (values.ndim - 2)
            + ((pad_y_before, pad_y_after), (0, half_x_after)),
        )
        scale = (padded_ny * padded_nx) / (grid.ny * grid.nx)
        return (
            jnp.fft.irfftn(
                jnp.fft.ifftshift(padded, axes=(-2,)),
                s=(padded_ny, padded_nx),
                axes=(-2, -1),
            )
            * scale
        ).astype(values.dtype)

    def truncate_padded_spectrum_local(values):
        half_spectrum = jnp.fft.rfftn(values, axes=(-2, -1))
        shifted = jnp.fft.fftshift(half_spectrum, axes=(-2,))
        cropped = jnp.fft.ifftshift(
            shifted[
                ...,
                pad_y_before : pad_y_before + grid.ny,
                : grid.nx // 2 + 1,
            ],
            axes=(-2,),
        )
        if grid.ny % 2 == 0:
            cropped = cropped.at[..., grid.ny // 2, :].set(
                0.5
                * (
                    half_spectrum[..., (-grid.ny // 2) % padded_ny, : grid.nx // 2 + 1]
                    + half_spectrum[..., grid.ny // 2, : grid.nx // 2 + 1]
                )
            )
        if grid.nx % 2 == 0:
            x_nyquist = 0.5 * (
                jnp.conj(
                    half_spectrum[
                        ...,
                        opposite_base_y_in_padded,
                        grid.nx // 2,
                    ]
                )
                + half_spectrum[..., base_y_in_padded, grid.nx // 2]
            )
            cropped = cropped.at[..., -1].set(x_nyquist)
            if grid.ny % 2 == 0:
                cropped = cropped.at[..., grid.ny // 2, -1].set(
                    half_spectrum[..., grid.ny // 2, grid.nx // 2].real
                )
        scale = (grid.ny * grid.nx) / (padded_ny * padded_nx)
        return cropped * scale

    def inverse_horizontal_spectrum_local(spectrum, dtype):
        return jnp.fft.irfftn(
            spectrum,
            s=(grid.ny, grid.nx),
            axes=(-2, -1),
        ).astype(dtype)

    def truncate_padded_local(values):
        return inverse_horizontal_spectrum_local(
            truncate_padded_spectrum_local(values), values.dtype
        )

    def horizontal_derivative_local(values, axis):
        spectrum = jnp.fft.rfftn(values, axes=(-2, -1))
        multiplier = (
            1j * kx.astype(values.real.dtype)
            if axis == 0
            else 1j * ky.astype(values.real.dtype)[:, None]
        )
        return jnp.fft.irfftn(
            spectrum * multiplier * keep.astype(values.real.dtype),
            s=(grid.ny, grid.nx),
            axes=(-2, -1),
        ).astype(values.dtype)

    def gradient_pair(values, local_kx, local_ky, shape, local_keep=None):
        spectrum = jnp.fft.rfftn(values, axes=(-2, -1))
        spectra = jnp.stack(
            (
                spectrum * (1j * local_kx.astype(values.real.dtype)),
                spectrum * (1j * local_ky.astype(values.real.dtype)[:, None]),
            ),
            axis=0,
        )
        if local_keep is not None:
            spectra = spectra * local_keep.astype(values.real.dtype)
        gradients = jnp.fft.irfftn(
            spectra,
            s=shape,
            axes=(-2, -1),
        ).astype(values.dtype)
        return gradients[0], gradients[1]

    def horizontal_gradient_pair_local(values):
        return gradient_pair(values, kx, ky, (grid.ny, grid.nx), keep)

    def padded_horizontal_gradient_pair_local(values):
        return gradient_pair(
            values,
            padded_kx,
            padded_ky,
            (padded_ny, padded_nx),
        )

    def flux_divergence_spectrum(x_spectra, y_spectra, dtype):
        return (
            x_spectra * (1j * kx.astype(dtype))
            + y_spectra * (1j * ky.astype(dtype)[:, None])
        ) * keep.astype(dtype)

    def spectral_flux_divergence(x_spectra, y_spectra, dtype):
        return inverse_horizontal_spectrum_local(
            flux_divergence_spectrum(x_spectra, y_spectra, dtype), dtype
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

    def padded_horizontal_flux_divergence_local(x_fluxes, y_fluxes):
        count = x_fluxes.shape[0]
        spectra = truncate_padded_spectrum_local(
            jnp.concatenate((x_fluxes, y_fluxes), axis=0)
        )
        return spectral_flux_divergence(
            spectra[:count], spectra[count:], x_fluxes.dtype
        )

    def wall_filter_local(values, filter_width):
        spectrum = jnp.fft.rfftn(values, axes=(-2, -1))
        cutoff_x = jnp.floor(grid.nx / (2.0 * filter_width))
        cutoff_y = jnp.floor(grid.ny / (2.0 * filter_width))
        wall_keep = (jnp.abs(y_mode)[:, None] < cutoff_y) & (x_mode[None, :] < cutoff_x)
        return jnp.fft.irfftn(
            spectrum * wall_keep,
            s=(grid.ny, grid.nx),
            axes=(-2, -1),
        ).astype(values.dtype)

    return HorizontalSpectralKernels(
        kx,
        ky,
        keep,
        keep,
        pad_horizontal_local,
        truncate_padded_spectrum_local,
        truncate_padded_local,
        horizontal_derivative_local,
        horizontal_gradient_pair_local,
        padded_horizontal_gradient_pair_local,
        spectral_flux_divergence,
        horizontal_flux_divergence_local,
        padded_horizontal_flux_divergence_local,
        wall_filter_local,
    )

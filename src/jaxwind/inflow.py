"""Mann (1994) homogeneous, neutral synthetic turbulence in SI units.

Generation runs once on the host; the frozen-box sampler is JAX compatible.
Equations: Jakob Mann, Atmospheric turbulence (2012), (20)--(25), (31).
https://breeze.colorado.edu/ftp/RSWE/Jakob_Mann.pdf
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import hyp2f1


def _positive(name, value, *, zero=False):
    if not np.isfinite(value) or (value < 0 if zero else value <= 0):
        raise ValueError(
            f"{name} must be finite and {'nonnegative' if zero else 'positive'}"
        )


def _mann_factor(k, length_scale, gamma, alpha_epsilon):
    """Real factor B with Phi = B B^T; k has final axis (kx, ky, kz)."""
    k1, k2, k3 = np.moveaxis(k, -1, 0)
    kk = np.sum(k * k, axis=-1)
    safe_kk = np.where(kk > 0, kk, 1.0)
    kl = np.sqrt(safe_kk) * length_scale
    beta = gamma * kl ** (-2 / 3) / np.sqrt(hyp2f1(1 / 3, 17 / 6, 4 / 3, -(kl**-2)))
    k30 = k3 + beta * k1
    k0 = np.stack((k1, k2, k30), axis=-1)
    kk0 = np.sum(k0 * k0, axis=-1)
    q = k1 * k1 + k2 * k2
    safe_q = np.where(q > 0, q, 1.0)
    safe_k1 = np.where(k1 != 0, k1, 1.0)
    c1 = beta * k1**2 * (kk0 - 2 * k30**2 + beta * k1 * k30) / (safe_kk * safe_q)
    # Difference of angles retains the continuous branch through k3=0.
    angle = np.arctan2(k30, np.sqrt(safe_q)) - np.arctan2(k3, np.sqrt(safe_q))
    c2 = k2 * kk0 / safe_q**1.5 * angle
    zeta1 = np.where(k1 != 0, c1 - k2 / safe_k1 * c2, -beta)
    zeta2 = np.where(k1 != 0, k2 / safe_k1 * c1 + c2, 0.0)
    distortion = np.broadcast_to(np.eye(3), (*kk.shape, 3, 3)).copy()
    distortion[..., 0, 2] = zeta1
    distortion[..., 1, 2] = zeta2
    distortion[..., 2, 2] = kk0 / safe_kk
    # The transverse projector itself is its own square root. Unlike
    # Cholesky this handles the rank-two incompressible tensor exactly.
    safe_kk0 = np.where(kk0 > 0, kk0, 1.0)
    projection = (
        np.eye(3) - k0[..., :, None] * k0[..., None, :] / safe_kk0[..., None, None]
    )
    amplitude = np.sqrt(
        alpha_epsilon
        * length_scale ** (17 / 3)
        * kk0
        / (4 * np.pi * (1 + length_scale**2 * kk0) ** (17 / 6))
    )
    factor = (distortion @ projection) * amplitude[..., None, None]
    return np.where((kk > 0)[..., None, None], factor, 0.0)


@dataclass(frozen=True)
class MannBox:
    """Periodic collocated fluctuations, shape (nz, ny, nx, 3).

    Nodes start at zero and exclude the periodic endpoint. Lengths are
    (lx, ly, lz) in metres, components are (u, v, w) in m/s.
    """

    velocity: np.ndarray
    lengths: tuple[float, float, float]


def generate_mann_box(
    *,
    shape,
    lengths,
    length_scale=29.4,
    gamma=3.9,
    alpha_epsilon=1.0,
    seed=0,
    sigma_u=None,
) -> MannBox:
    """Generate a Gaussian Mann box with a local NumPy random generator.

    ``shape`` is (nx, ny, nz); ``alpha_epsilon`` is alpha*epsilon**(2/3)
    in m**(4/3)/s**2. Optional ``sigma_u`` (m/s) rescales all components
    together to the realized streamwise standard deviation, preserving
    anisotropy. Without it, amplitudes follow the discretized spectral tensor.
    The zero mode and even-grid Nyquist planes are omitted to preserve both
    Hermitian symmetry and spectral incompressibility. No wall blocking or
    unresolved-wavenumber compensation is applied.
    """
    shape, lengths = tuple(shape), tuple(lengths)
    if len(shape) != 3 or any(
        isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 3
        for n in shape
    ):
        raise ValueError("shape must contain three integers >= 3 (nx, ny, nz)")
    if len(lengths) != 3:
        raise ValueError("lengths must contain (lx, ly, lz)")
    for name, value in zip(("lx", "ly", "lz"), lengths):
        _positive(name, value)
    _positive("length_scale", length_scale)
    _positive("gamma", gamma, zero=True)
    _positive("alpha_epsilon", alpha_epsilon)
    if sigma_u is not None:
        _positive("sigma_u", sigma_u, zero=True)
    axes = [
        2 * np.pi * np.fft.fftfreq(n, d=length / n) for n, length in zip(shape, lengths)
    ]
    kz, ky, kx = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    factor = _mann_factor(
        np.stack((kx, ky, kz), axis=-1), length_scale, gamma, alpha_epsilon
    )
    for axis, n in enumerate(shape[::-1]):
        if n % 2 == 0:
            index = [slice(None)] * 5
            index[axis] = n // 2
            factor[tuple(index)] = 0.0
    noise = np.random.default_rng(seed).standard_normal((*shape[::-1], 3))
    noise_hat = np.fft.fftn(noise, axes=(0, 1, 2))
    # E|FFT(white noise)|^2=N; desired Fourier-series covariance is Phi*dk.
    scale = np.sqrt(np.prod(shape) * (2 * np.pi) ** 3 / np.prod(lengths))
    spectrum = np.einsum("...ij,...j->...i", factor, noise_hat) * scale
    velocity = np.fft.ifftn(spectrum, axes=(0, 1, 2)).real
    if sigma_u is not None:
        velocity *= sigma_u / np.std(velocity[..., 0])
    return MannBox(velocity, tuple(float(length) for length in lengths))


def build_mann_inflow(
    box,
    grid,
    *,
    mean_speed,
    mean_profile=None,
    scalar=0.0,
    wall_y=False,
    sigma_u_profile=None,
):
    """Build ``inflow(time_seconds) -> InflowPlane`` for the open FV solver.

    Frozen turbulence travels along +x: u(x,t)=u_box(x-U*t,y,z).
    All axes wrap periodically; recurrence time is lx/mean_speed. The grid
    must fit inside the box transversely. ``mean_profile`` and ``scalar``
    accept a scalar or (nz,) profile. Mean advection speed remains constant.
    Trilinear interpolation samples each component at its MAC location;
    impermeable top/bottom and optional y walls have zero normal velocity.
    ``sigma_u_profile`` optionally calibrates the streamwise RMS at each
    cell height, averaged over y and a continuous full box period, including
    interpolation losses. All components share the local scaling, and the
    streamwise period mean is removed at each height. This introduces
    vertical inhomogeneity.
    Interpolation/wall enforcement do not preserve spectral divergence.
    """
    import jax.numpy as jnp

    from .open_boundary import InflowPlane

    _positive("mean_speed", mean_speed)
    if grid.ly > box.lengths[1] or grid.lz > box.lengths[2]:
        raise ValueError("Mann box must cover the receiving y/z domain")
    values = jnp.asarray(box.velocity)
    nz, ny, nx, _ = values.shape
    lx, ly, lz = box.lengths

    def profile(value, name):
        value = np.asarray(value)
        if value.shape not in ((), (grid.nz,)) or not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite and scalar or shape (nz,)")
        return jnp.broadcast_to(jnp.asarray(value), (grid.nz,))[:, None]

    mean = profile(mean_speed if mean_profile is None else mean_profile, "mean_profile")
    scalar_plane = jnp.broadcast_to(profile(scalar, "scalar"), (grid.nz, grid.ny))

    def sample(x, y, z, component):
        coords = jnp.broadcast_arrays(
            z[:, None] * nz / lz, y[None, :] * ny / ly, x * nx / lx
        )
        lower = [jnp.floor(c).astype(jnp.int32) for c in coords]
        frac = [c - jnp.floor(c) for c in coords]
        result = jnp.zeros_like(coords[0])
        for dz in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    offsets = (dz, dy, dx)
                    weight = 1.0
                    indices = []
                    for i, (offset, size) in enumerate(zip(offsets, (nz, ny, nx))):
                        weight = weight * (frac[i] if offset else 1 - frac[i])
                        indices.append((lower[i] + offset) % size)
                    result = (
                        result
                        + weight * values[indices[0], indices[1], indices[2], component]
                    )
        return result

    yc, zc = jnp.asarray(grid.y_centers), jnp.asarray(grid.z_centers)
    yf = jnp.asarray(grid.y_faces if wall_y else grid.y_faces[:-1])
    zf = jnp.asarray(grid.z_faces)

    # Match the measured streamwise RMS after transverse interpolation and
    # continuous frozen advection. For linear x interpolation the period
    # variance is (2*variance + lag-one covariance)/3.
    shift = np.zeros(grid.nz)
    scaling = np.ones(grid.nz)
    if sigma_u_profile is not None:
        target = np.broadcast_to(np.asarray(sigma_u_profile, dtype=float), (grid.nz,))
        if not np.isfinite(target).all() or (target < 0).any():
            raise ValueError("sigma_u_profile must be finite and nonnegative")
        cy = np.asarray(grid.y_centers) * ny / ly
        iy = np.floor(cy).astype(int)
        fy = (cy - iy)[:, None]
        for level, z in enumerate(grid.z_centers):
            cz = z * nz / lz
            iz = int(np.floor(cz))
            fz = cz - iz
            section = (1 - fz) * box.velocity[iz % nz, :, :, 0] + fz * box.velocity[
                (iz + 1) % nz, :, :, 0
            ]
            section = (1 - fy) * section[iy % ny] + fy * section[(iy + 1) % ny]

            def section_mean(values):
                if grid.is_uniform:
                    return np.mean(values)
                return np.average(np.mean(values, axis=1), weights=grid.y_widths)

            shift[level] = section_mean(section)
            centered = section - shift[level]
            variance = (
                2 * section_mean(centered**2)
                + section_mean(centered * np.roll(centered, 1, axis=1))
            ) / 3
            if variance <= 0:
                raise ValueError("cannot calibrate a zero-variance Mann box")
            scaling[level] = target[level] / np.sqrt(variance)
    shift = jnp.asarray(shift)[:, None]
    scale_cell = jnp.asarray(scaling)[:, None]
    scale_face = jnp.asarray(np.interp(grid.z_faces, grid.z_centers, scaling))[:, None]

    def inflow(time):
        x = jnp.mod(-mean_speed * time, lx)
        u = mean + scale_cell * (sample(x, yc, zc, 0) - shift)
        # Tangential components are at the first cell centre in x.
        v = scale_cell * sample(x + grid.x_centers[0], yf, zc, 1)
        w = scale_face * sample(x + grid.x_centers[0], yc, zf, 2)
        w = w.at[0].set(0.0).at[-1].set(0.0)
        if wall_y:
            v = v.at[:, 0].set(0.0).at[:, -1].set(0.0)
        return InflowPlane(u, v, w, scalar_plane)

    return inflow


__all__ = ["MannBox", "build_mann_inflow", "generate_mann_box"]

#!/usr/bin/env python3
"""Neutral atmospheric-boundary-layer demo using JAX and the MGM SGS model.

Purpose
-------
This file is a compact, single-device port of ``neutral_abl_mgm_demo.cu``.  It
is intended both as an executable demonstration and as a readable reference
for the numerical algorithm.  The implementation follows the CUDA solver's
sharp horizontal filtering, staggered first-order vertical differences,
Porté--Agel wall treatment, rotational convection, clipped mixed-gradient
model (MGM), second-order Adams--Bashforth stepping, and pressure projection.
JAX JIT-compiles the shape-specialized operators for either a CPU or a GPU.

Units and nondimensionalization
-------------------------------
Velocity values remain dimensional (m/s), and ``dtr`` is the dimensional time
step in seconds.  Spatial derivatives use coordinates normalized by
``zi = lx/(2*pi)``.  Consequently the time step used inside the equations is
``dt = dtr/zi``; multiplying a nondimensional-coordinate derivative by ``dt``
still produces a velocity increment in m/s.

Array and staggering conventions
---------------------------------
The leading dimensions carry physical meaning throughout the code:

* ``velocity[component, z, y, x]`` stores ``(u, v, w)``;
* ``gradient[component, direction, z, y, x]`` stores ``du_i/dx_j``;
* ``stress[component, z, y, x]`` stores the packed symmetric components
  ``(xx, xy, xz, yy, yz, zz)``.

Horizontal directions are periodic and differentiated spectrally.  The two
extra z planes in ``velocity`` are vertical boundary/ghost storage.  Horizontal
velocity is carried at cell-center levels; its top value is duplicated to
impose a zero-gradient condition.  Vertical velocity is face-staggered and is
zero at the lower and upper impermeable boundaries.  Helper routines retain
this layout rather than silently converting between arrangements.

Default study and outputs
-------------------------
The defaults reproduce the 64-cubed case of the resolution study: 0.5 s fixed
time step, 57,600 steps (8 h), with statistics accumulated from 4 to 8 h every
20 steps (10 s).  ``mgm_profile.csv`` is the final resolved plane-mean profile.
The ``ta_*.bin`` files are raw float32, C-order, time-averaged 3-D fields for
``u``, ``v``, ``w``, their squares, and ``uv``.

Examples
--------
Quick CPU check::

    python neutral_abl_mgm_jax.py --backend cpu --nx 16 --ny 16 --nz 16 \
        --steps 10

Full default GPU case::

    python neutral_abl_mgm_jax.py --backend gpu
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import NamedTuple

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np


Array = jax.Array
STRESS_NAMES = ("xx", "xy", "xz", "yy", "yz", "zz")
PAIR_I = jnp.asarray((0, 0, 0, 1, 1, 2))
PAIR_J = jnp.asarray((0, 1, 2, 1, 2, 2))


@dataclass(frozen=True)
class Config:
    """Physical, numerical, and sampling controls for one compiled grid.

    Lengths are dimensional meters except for the derived ``dx``, ``dy``, and
    ``dz``, which are normalized by ``zi``.  ``ustar`` is the target friction
    velocity, ``z0`` the aerodynamic roughness length, and ``nu`` the molecular
    kinematic viscosity.  Grid sizes are part of JAX array shapes, so changing
    any of them causes a new compilation.
    """

    # Grid and run length: the defaults advance exactly 8 physical hours.
    nx: int = 64
    ny: int = 64
    nz: int = 64
    steps: int = 57_600

    # Dimensional domain and boundary-layer forcing scales (meters, m/s).
    lx: float = 4000.0
    ly: float = 4000.0
    lz: float = 1000.0
    bl_height: float = 1000.0
    ustar: float = 0.38
    z0: float = 0.005
    nu: float = 1.5e-5

    # Time step and SGS/filter controls.  dtr is dimensional seconds.
    dtr: float = 0.5
    fgr: float = 1.5
    tfr: float = 2.0
    cs: float = 0.1
    kappa: float = 0.4

    # The defaults sample every 10 s over the final four physical hours.
    statistics_start: int = 28_800
    sample_every: int = 20

    @property
    def nz2(self) -> int:
        """Vertical allocation including the two boundary/ghost planes."""
        return self.nz + 2

    @property
    def nkx(self) -> int:
        """Number of stored x wave numbers produced by a real FFT."""
        return self.nx // 2 + 1

    @property
    def lr(self) -> int:
        """Integer horizontal aspect ratio used to scale y wave numbers."""
        return int(self.lx / self.ly + 0.5)

    @property
    def zi(self) -> float:
        """Reference length that maps the x domain to the interval 2*pi."""
        return self.lx / (2.0 * math.pi)

    @property
    def dx(self) -> float:
        """Nondimensional streamwise grid spacing."""
        return 2.0 * math.pi / self.nx

    @property
    def dy(self) -> float:
        """Nondimensional spanwise grid spacing."""
        return 2.0 * math.pi / (self.ny * self.lr)

    @property
    def dz(self) -> float:
        """Nondimensional spacing between vertical velocity faces."""
        return (self.lz / self.zi) / (self.nz - 1)

    @property
    def dt(self) -> float:
        """Time increment compatible with the nondimensional derivatives."""
        return self.dtr / self.zi

    @property
    def pressure_force(self) -> float:
        """Constant streamwise pressure-gradient acceleration."""
        return self.ustar**2 / (self.bl_height / self.zi)


class State(NamedTuple):
    """Device-resident variables carried between time steps.

    ``previous_rhs`` supplies the history term for Adams--Bashforth 2, while
    ``iteration`` selects a forward-Euler startup on the very first step.
    """

    velocity: Array
    previous_rhs: Array
    iteration: Array


class Operators(NamedTuple):
    """JIT-compiled functions specialized to one :class:`Config` shape."""

    step: callable
    clean: callable
    diagnostics: callable
    accumulate: callable


def clipping_coefficient(gs: Array, gkk: Array) -> Array:
    """Evaluate the Lu--Porté-Agel anisotropic correction on horizontal planes.

    Equation (4) in Lu--Porté-Agel 2010. Horizontal x-y means are appropriate here because
    the neutral boundary layer is statistically homogeneous in those
    directions. Otherwise a conditional averaging is needed.
    Because the cubic moment is signed, backscatter reduces the
    unconditional mean.  A plane with an undefined or non-positive denominator
    uses the uncorrected baseline value ``C=1``.
    """
    valid_g = jnp.abs(gkk) > 1.0e-6
    safe_gkk = jnp.where(valid_g, gkk, 1.0)
    transfer = jnp.where(valid_g, -gs / safe_gkk, 0.0)
    transfer_moment = transfer**3

    # Reducing the last two axes produces one coefficient per vertical plane.
    valid_count = jnp.sum(valid_g, axis=(-2, -1))
    forward = valid_g & (gs <= 0.0)
    forward_count = jnp.sum(forward, axis=(-2, -1))
    conditional_sum = jnp.sum(jnp.where(forward, transfer_moment, 0.0), axis=(-2, -1))
    conditional_mean = conditional_sum / jnp.maximum(forward_count, 1)
    unconditional_sum = jnp.sum(jnp.where(valid_g, transfer_moment, 0.0), axis=(-2, -1))
    unconditional_mean = unconditional_sum / jnp.maximum(valid_count, 1)
    absolute_mean = jnp.sum(
        jnp.where(valid_g, jnp.abs(transfer_moment), 0.0), axis=(-2, -1)
    ) / jnp.maximum(valid_count, 1)
    denominator_floor = jnp.finfo(gkk.dtype).eps * jnp.maximum(
        absolute_mean,
        jnp.finfo(gkk.dtype).tiny,
    )

    coefficient_squared = conditional_mean / jnp.where(
        unconditional_mean > 0.0, unconditional_mean, 1.0
    )
    usable = (
        (valid_count > 0)
        & (forward_count > 0)
        & (unconditional_mean > denominator_floor)
        & jnp.isfinite(coefficient_squared)
        & (coefficient_squared > 0.0)
    )
    coefficient = jnp.sqrt(jnp.where(usable, coefficient_squared, 1.0))
    return coefficient[:, None, None]


def select_device(requested: str) -> jax.Device:
    """Choose a JAX device with an actionable error for missing GPU support."""

    if requested == "cpu":
        return jax.devices("cpu")[0]
    try:
        gpu_devices = jax.devices("gpu")
    except RuntimeError:
        gpu_devices = []
    if requested == "gpu" and not gpu_devices:
        raise SystemExit(
            "--backend gpu requested, but JAX has no GPU device. Install the "
            "CUDA-enabled JAX package and verify the NVIDIA driver."
        )
    return gpu_devices[0] if gpu_devices else jax.devices("cpu")[0]


def initial_velocity(cfg: Config, restart: Path | None = None) -> Array:
    """Construct an analytic solenoidal seed or load a CUDA restart.

    Restart files contain only the ``nz * ny * nx`` physical values for each
    component as raw float32 data.  Boundary values are reconstructed after
    loading.  Without a restart, the seed is a log-law profile plus two weak
    streamfunction-derived modes whose horizontal and vertical derivatives
    cancel, giving a discretely divergence-free perturbation.
    """

    # Allocate all three components together, including z boundary storage.
    shape = (3, cfg.nz2, cfg.ny, cfg.nx)
    if restart is not None:
        # Match the CUDA binary convention: one C-order file per component.
        host = np.zeros(shape, dtype=np.float32)
        count = cfg.nz * cfg.ny * cfg.nx
        for component, name in enumerate(("u", "v", "w")):
            path = restart / f"{name}.bin"
            values = np.fromfile(path, dtype=np.float32)
            if values.size != count:
                raise SystemExit(
                    f"invalid restart {path}: expected {count} float32 values, "
                    f"found {values.size}"
                )
            host[component, 1 : cfg.nz + 1] = values.reshape(cfg.nz, cfg.ny, cfg.nx)
        # Recreate the top Neumann condition for u/v and impermeable w faces.
        host[0:2, cfg.nz] = host[0:2, cfg.nz - 1]
        host[2, (1, cfg.nz)] = 0.0
        return jnp.asarray(host)

    # Coordinates below are nondimensional except z, which is converted back
    # to meters before evaluating the dimensional logarithmic wind profile.
    i = jnp.arange(cfg.nx, dtype=jnp.float32)
    j = jnp.arange(cfg.ny, dtype=jnp.float32)
    k = jnp.arange(1, cfg.nz + 1, dtype=jnp.float32)
    kc = jnp.minimum(k, cfg.nz - 1.0)
    x, y = i * cfg.dx, j * cfg.dy
    z = (kc - 0.5) * cfg.dz * cfg.zi

    # Neutral Monin--Obukhov log law.  The lower bound protects the logarithm
    # if a deliberately unusual configuration puts a center at or below z0.
    base = cfg.ustar / cfg.kappa * jnp.log(jnp.maximum(z / cfg.z0, 1.0001))

    # Sine envelopes vanish at the w boundaries.  Their finite difference
    # ``ds`` is paired with the horizontal derivative of each perturbation.
    sb = jnp.sin(math.pi * (kc - 1.0) / (cfg.nz - 1.0))
    st = jnp.sin(math.pi * kc / (cfg.nz - 1.0))
    ds = (st - sb) / cfg.dz

    # The x-z and y-z mode pairs are discrete streamfunctions: du/dx is
    # canceled by dw/dz for the first pair, and dv/dy by dw/dz for the second.
    u = base[:, None, None] + 0.015 * ds[:, None, None] * jnp.cos(
        x[None, None, :] + 0.37
    )
    u = jnp.broadcast_to(u, (cfg.nz, cfg.ny, cfg.nx))
    v = (
        0.010
        / (2.0 * cfg.lr)
        * ds[:, None, None]
        * jnp.cos(2.0 * cfg.lr * y[None, :, None] - 0.23)
    )
    v = jnp.broadcast_to(v, (cfg.nz, cfg.ny, cfg.nx))
    w = 0.015 * sb[:, None, None] * jnp.sin(x[None, None, :] + 0.37)
    w += 0.010 * sb[:, None, None] * jnp.sin(2.0 * cfg.lr * y[None, :, None] - 0.23)
    w = jnp.broadcast_to(w, (cfg.nz, cfg.ny, cfg.nx))
    w = w.at[(0, -1), :, :].set(0.0)

    # Insert physical values into the padded layout, then enforce the top
    # zero-gradient condition on horizontal velocity explicitly.
    velocity = jnp.zeros(shape, dtype=jnp.float32)
    velocity = velocity.at[:, 1 : cfg.nz + 1].set(jnp.stack((u, v, w)))
    return velocity.at[0:2, cfg.nz].set(velocity[0:2, cfg.nz - 1])


def make_operators(cfg: Config) -> Operators:
    """Build all spatial and time-integration operators for ``cfg``.

    The nested functions close over grid-dependent wave numbers, masks, and
    scalar coefficients.  This makes those quantities compile-time constants
    for XLA and avoids rebuilding them on every step.  Arrays passed to these
    functions remain on the selected JAX device.
    """

    nx, ny, nz, nz2, nkx = cfg.nx, cfg.ny, cfg.nz, cfg.nz2, cfg.nkx
    physical = slice(1, nz + 1)

    # rfft2 stores only non-negative x modes.  y retains FFT order, so indices
    # above ny/2 are mapped to negative wave numbers.  The aspect ratio scales
    # y derivatives into the same nondimensional coordinates as x.
    kx = jnp.arange(nkx, dtype=jnp.float32)
    j_index = jnp.arange(ny)
    ky = (
        jnp.where(
            j_index < ny // 2,
            j_index,
            jnp.where(j_index == ny // 2, 0, j_index - ny),
        ).astype(jnp.float32)
        * cfg.lr
    )

    # Sharp LES cutoff.  fgr > 1 removes the highest resolved Fourier modes
    # so nonlinear products are returned to the intended filtered bandwidth.
    fcx = int(nx / (2.0 * cfg.fgr) + 0.5)
    fcy = int(ny / (2.0 * cfg.fgr) + 0.5)
    sharp_mask = (jnp.arange(nkx)[None, :] < fcx) & (
        (j_index[:, None] < fcy) | (j_index[:, None] > ny - fcy)
    )
    # The wall model uses an additional test filter with width tfr times the
    # grid filter.  It is applied only to the first horizontal velocity plane.
    wall_cx = int(nx / (2.0 * cfg.fgr * cfg.tfr))
    wall_cy = int(cfg.lr * ny / (2.0 * cfg.fgr * cfg.tfr))
    wall_ky = jnp.where(j_index > nx // 2, j_index - ny, j_index) * cfg.lr
    wall_mask = (jnp.arange(nkx)[None, :] < wall_cx) & (
        jnp.abs(wall_ky[:, None]) < wall_cy
    )
    # A real-valued spectral derivative cannot represent a standalone
    # Nyquist mode consistently.  Zero it exactly as in the CUDA reference.
    joint_nyquist = (jnp.arange(nkx)[None, :] == nx // 2) | (
        j_index[:, None] == ny // 2
    )
    plain_kx = jnp.where(joint_nyquist, 0.0, kx[None, :])
    plain_ky = jnp.where(joint_nyquist, 0.0, ky[:, None])
    # Pressure uses its own mode convention and mask because each retained
    # horizontal Fourier mode becomes one independent vertical Poisson solve.
    pressure_ky = jnp.where(j_index > ny // 2, j_index - ny, j_index) * cfg.lr
    pressure_mask = (jnp.arange(nkx)[None, :] < int(nx / 2.0 + 0.5)) & (
        jnp.abs(pressure_ky[:, None]) < int(cfg.lr * ny / 2.0 + 0.5)
    )
    pressure_k2 = kx[None, :] ** 2 + pressure_ky[:, None] ** 2
    # Apply the constant pressure-gradient forcing only inside the specified
    # boundary-layer height; with the defaults this includes the full domain.
    forcing_levels = (
        jnp.arange(1, nz + 1, dtype=jnp.float32) - 0.5
    ) * cfg.dz <= cfg.bl_height / cfg.zi

    def pad(values: Array) -> Array:
        """Insert nz physical planes into an nz+2 zero-padded z layout."""
        result = jnp.zeros((*values.shape[:-3], nz2, ny, nx), values.dtype)
        return result.at[..., physical, :, :].set(values)

    def ddz(field: Array, stagger: bool) -> Array:
        """First-order z derivative respecting the variable's staggering.

        ``stagger=False`` differences a center-stored field onto its lower faces;
        ``stagger=True`` differences a face-stored field onto cell centers.
        """
        values = (
            field[2 : nz + 2] - field[1 : nz + 1]
            if stagger
            else field[1 : nz + 1] - field[0:nz]
        ) / cfg.dz
        return pad(values)

    def filtered_velocity_and_xy_gradient(velocity: Array) -> tuple[Array, Array]:
        """Sharp-filter velocity and compute its spectral x/y derivatives."""

        # A single forward transform supplies the filtered field and both
        # horizontal derivatives, reducing FFT traffic in the time step.
        spectrum = jnp.fft.rfft2(velocity[:, physical], axes=(-2, -1))
        spectrum *= sharp_mask[None, None]
        filtered = jnp.fft.irfft2(spectrum, s=(ny, nx), axes=(-2, -1))
        dx_values = jnp.fft.irfft2(
            1j * kx[None, None, None, :] * spectrum,
            s=(ny, nx),
            axes=(-2, -1),
        )
        dy_values = jnp.fft.irfft2(
            1j * ky[None, None, :, None] * spectrum,
            s=(ny, nx),
            axes=(-2, -1),
        )
        velocity = velocity.at[:, physical].set(filtered)
        velocity = velocity.at[0:2, nz].set(velocity[0:2, nz - 1])
        gradient = jnp.zeros((3, 3, nz2, ny, nx), dtype=velocity.dtype)
        gradient = gradient.at[:, 0, physical].set(dx_values)
        return velocity, gradient.at[:, 1, physical].set(dy_values)

    def clean_velocity_impl(velocity: Array) -> Array:
        """Return a filtered, boundary-consistent state for sampling.

        Intermediate AB2 states can contain modes outside the sharp LES mask.
        Diagnostics and statistics must use the resolved state, so this helper
        reapplies the mask and boundary values without modifying time history.
        """

        spectrum = jnp.fft.rfft2(velocity[:, physical], axes=(-2, -1))
        spectrum *= sharp_mask[None, None]
        filtered = jnp.fft.irfft2(spectrum, s=(ny, nx), axes=(-2, -1))
        clean = velocity.at[:, physical].set(filtered)
        clean = clean.at[0:2, nz].set(clean[0:2, nz - 1])
        return clean.at[2, (1, nz), :, :].set(0.0)

    def plain_derivative(field: Array, axis: int) -> Array:
        """Differentiate one padded scalar field spectrally in x or y."""
        spectrum = jnp.fft.rfft2(field[physical], axes=(-2, -1))
        wave = plain_kx if axis == 0 else plain_ky
        values = jnp.fft.irfft2(1j * wave * spectrum, s=(ny, nx), axes=(-2, -1))
        return pad(values)

    def wall_filter(field: Array) -> Array:
        """Test-filter the first off-wall plane for the wall-stress model."""
        spectrum = jnp.fft.rfft2(field[1], axes=(-2, -1)) * wall_mask
        return jnp.fft.irfft2(spectrum, s=(ny, nx), axes=(-2, -1))

    def velocity_gradient_and_wall(velocity: Array) -> tuple[Array, Array, Array]:
        """Assemble all velocity gradients and the lower-wall shear stress."""
        velocity, gradient = filtered_velocity_and_xy_gradient(velocity)
        gradient = gradient.at[0, 2].set(ddz(velocity[0], False))
        gradient = gradient.at[1, 2].set(ddz(velocity[1], False))
        gradient = gradient.at[2, 2].set(ddz(velocity[2], True))

        # Porté--Agel JFM 2000 Appendix: adjust the plane-mean gradient on
        # the second level to the logarithmic near-wall variation.
        correction = 1.0 / math.log(3.0) - 1.0
        gradient = gradient.at[0, 2, 2].add(correction * jnp.mean(gradient[0, 2, 2]))
        gradient = gradient.at[1, 2, 2].add(correction * jnp.mean(gradient[1, 2, 2]))

        # Infer a local friction velocity from the test-filtered speed and
        # the neutral log law at the first cell center.  Stress opposes motion.
        uf, vf = wall_filter(velocity[0]), wall_filter(velocity[1])
        speed = jnp.hypot(uf, vf)
        safe_speed = jnp.maximum(speed, jnp.finfo(jnp.float32).tiny)
        denominator = math.log(0.5 * cfg.dz * cfg.zi / cfg.z0)
        ustar = speed * cfg.kappa / denominator
        wall_stress = -(ustar**2)[None] * jnp.stack((uf, vf)) / safe_speed
        wall_gradient = (
            jnp.stack((uf, vf)) * ustar / (safe_speed * cfg.kappa * 0.5 * cfg.dz)
        )
        moving = speed > jnp.finfo(jnp.float32).tiny
        wall_stress = jnp.where(moving[None], wall_stress, 0.0)
        wall_gradient = jnp.where(moving[None], wall_gradient, 0.0)
        gradient = gradient.at[0:2, 2, 1].set(wall_gradient)
        return velocity, gradient, wall_stress

    def rotational_convection(velocity: Array, gradient: Array) -> Array:
        """Evaluate the rotational convective term on the staggered grid.

        This is ``omega x u`` up to the gradient absorbed by pressure.  Values
        involving w are averaged between adjacent faces to collocate products
        with the horizontal-momentum or vertical-momentum control point.
        """
        u, v, w = velocity
        uy, uz = gradient[0, 1], gradient[0, 2]
        vx, vz = gradient[1, 0], gradient[1, 2]
        wx, wy = gradient[2, 0], gradient[2, 1]
        c = slice(1, nz + 1)
        above, below = slice(2, nz + 2), slice(0, nz)
        cx = v[c] * (uy[c] - vx[c]) + 0.5 * (
            w[above] * (uz[above] - wx[above]) + w[c] * (uz[c] - wx[c])
        )
        cy = u[c] * (vx[c] - uy[c]) + 0.5 * (
            w[above] * (vz[above] - wy[above]) + w[c] * (vz[c] - wy[c])
        )
        cz = 0.5 * (u[c] + u[below]) * (wx[c] - uz[c])
        cz += 0.5 * (v[c] + v[below]) * (wy[c] - vz[c])
        return pad(jnp.stack((cx, cy, cz)))

    def mgm_tensor(gradient: Array) -> Array:
        """Return six packed SGS-stress components from a collocated gradient.

        The mixed-gradient tensor uses grid-anisotropy weights.  Its local SGS
        energy is obtained from the tensor/strain contraction, clipped to the
        dissipative branch used by the reference implementation.  The energy
        coefficient is computed from instantaneous horizontal-plane averages.
        Degenerate points fall back to a Smagorinsky stress to avoid division
        by tiny ``trace(g)`` values.
        """
        strain = 0.5 * (gradient + jnp.swapaxes(gradient, 0, 1))
        weights = jnp.asarray(
            (1.0, (cfg.dy / cfg.dx) ** 2, (cfg.dz / (cfg.dx * cfg.fgr)) ** 2),
            dtype=jnp.float32,
        )
        # g_ij is the anisotropy-aware gradient product; gkk is its trace
        # and gs its contraction with the resolved strain-rate tensor.
        g = jnp.einsum("ik...,jk...,k->ij...", gradient, gradient, weights)
        gkk = jnp.trace(g, axis1=0, axis2=1)
        gs = jnp.sum(g * strain, axis=(0, 1))
        safe_gkk = jnp.where(jnp.abs(gkk) > 1.0e-6, gkk, 1.0)
        delta = (cfg.fgr * cfg.dx * cfg.fgr * cfg.dy * cfg.dz) ** (1.0 / 3.0)
        ce = clipping_coefficient(gs, gkk)
        ratio = jnp.minimum(gs, 0.0) / safe_gkk
        ksgs = (2.0 * delta / ce) ** 2 * ratio**2
        nu = cfg.nu / cfg.zi
        modeled = 2.0 * ksgs[None, None] * g / safe_gkk[None, None]
        modeled -= 2.0 * nu * strain

        # Near gkk=0 the MGM ratio is undefined.  The fallback remains
        # dissipative and includes the same molecular-viscosity contribution.
        strain_magnitude = jnp.sqrt(2.0 * jnp.sum(strain**2, axis=(0, 1)))
        fallback_factor = -2.0 * cfg.cs**2 * delta**2 * strain_magnitude - 2.0 * nu
        fallback = fallback_factor[None, None] * strain
        tensor = jnp.where((jnp.abs(gkk) > 1.0e-6)[None, None], modeled, fallback)
        return tensor[PAIR_I, PAIR_J]

    def mgm_stress(gradient: Array, wall_stress: Array) -> Array:
        """Collocate gradients, evaluate MGM, and impose stress boundaries.

        Normal/horizontal shear stresses live with u and v, whereas xz and yz
        shear stresses live on w faces.  Separate collocations preserve this
        staggering before the six components are packed into one array.
        """
        center = gradient[:, :, physical]
        above, below = gradient[:, :, 2 : nz + 2], gradient[:, :, 0:nz]

        # Gradients collocated for xx, xy, yy, and zz stresses.
        uv_gradient = center
        uv_gradient = uv_gradient.at[0:2, 2].set(0.5 * (center[0:2, 2] + above[0:2, 2]))
        uv_gradient = uv_gradient.at[0:2, 2, 0].set(center[0:2, 2, 0])
        uv_gradient = uv_gradient.at[2, 0:2].set(0.5 * (center[2, 0:2] + above[2, 0:2]))
        uv_gradient = uv_gradient.at[2, 2].set(-(center[0, 0] + center[1, 1]))
        uv_stress = mgm_tensor(uv_gradient)

        # Gradients collocated for the face-centered xz and yz stresses.
        w_gradient = center
        w_gradient = w_gradient.at[0:2, 0:2].set(
            0.5 * (below[0:2, 0:2] + center[0:2, 0:2])
        )
        w_gradient = w_gradient.at[2, 2].set(0.5 * (below[2, 2] + center[2, 2]))
        w_stress = mgm_tensor(w_gradient)

        # The lower xz/yz values come directly from the wall model; their
        # upper values vanish at the stress-free top boundary.
        stress = jnp.zeros((6, nz2, ny, nx), dtype=gradient.dtype)
        stress = stress.at[jnp.asarray((0, 1, 3, 5)), physical].set(
            uv_stress[jnp.asarray((0, 1, 3, 5))]
        )
        stress = stress.at[jnp.asarray((2, 4)), 1].set(wall_stress)
        stress = stress.at[jnp.asarray((2, 4)), 2 : nz + 1].set(
            w_stress[jnp.asarray((2, 4)), 1:]
        )
        stress = stress.at[jnp.asarray((2, 4)), nz].set(0.0)
        return stress.at[5, nz].set(stress[5, nz - 1])

    def stress_divergence(stress: Array) -> Array:
        """Compute one divergence component from each symmetric stress row."""
        component_triplets = ((0, 1, 2, True), (1, 3, 4, True), (2, 4, 5, False))
        values = []
        for x_component, y_component, z_component, stagger in component_triplets:
            values.append(
                plain_derivative(stress[x_component], 0)
                + plain_derivative(stress[y_component], 1)
                + ddz(stress[z_component], stagger)
            )
        return jnp.stack(values)

    def pressure_thomas(rhs_hat: Array) -> Array:
        """Solve the vertical pressure system for every horizontal mode.

        Fourier transformation diagonalizes the horizontal Laplacian, leaving
        one tridiagonal z problem per (kx, ky).  ``lax.scan`` performs the
        forward Thomas sweep for all modes in parallel; ``lax.fori_loop`` then
        back-substitutes.  Boundary rows impose the vertical pressure Neumann
        conditions, while the zero mode is fixed to remove the pressure gauge.
        """

        aa = 1.0 / cfg.dz**2
        zero_mode = pressure_k2 == 0.0
        b0 = jnp.where(zero_mode, 1.0, -1.0)
        c0 = jnp.where(zero_mode, 0.0, 1.0)
        cp0 = c0 / b0
        dp0 = jnp.zeros_like(rhs_hat[0])

        # Modified upper diagonal (cp) and right-hand side (dp) are arrays
        # over all horizontal modes, so no Python loop over modes is needed.
        def forward(carry: tuple[Array, Array], row: Array):
            cp_previous, dp_previous = carry
            top = row == nz
            a = jnp.where(top, -1.0, aa)
            b = jnp.where(top, 1.0, -pressure_k2 - 2.0 * aa)
            c = jnp.where(top, 0.0, aa)
            source = jnp.where(top, 0.0, rhs_hat[row - 1])
            denominator = b - a * cp_previous
            cp = c / denominator
            dp = (source - a * dp_previous) / denominator
            return (cp, dp), (cp, dp)

        _, (cp_tail, dp_tail) = lax.scan(forward, (cp0, dp0), jnp.arange(1, nz + 1))
        cp = jnp.concatenate((cp0[None], cp_tail), axis=0)
        dp = jnp.concatenate((dp0[None], dp_tail), axis=0)
        solution = jnp.zeros_like(dp).at[nz].set(dp[nz])

        def backward(index: int, current: Array) -> Array:
            row = nz - 1 - index
            return current.at[row].set(dp[row] - cp[row] * current[row + 1])

        solution = lax.fori_loop(0, nz, backward, solution)
        return solution[1 : nz + 1]

    def project(velocity: Array) -> Array:
        """Project a provisional velocity onto the discrete solenoidal space."""

        # Solve Laplacian(p) = div(u*)/dt, then apply u = u* - dt*grad(p).
        # Horizontal derivatives are spectral; the z derivative retains the
        # same first-order staggering used by the divergence operator.
        divergence = plain_derivative(velocity[0], 0)
        divergence += plain_derivative(velocity[1], 1)
        divergence += ddz(velocity[2], True)
        rhs_hat = jnp.fft.rfft2(divergence[physical] / cfg.dt, axes=(-2, -1))
        pressure_hat = pressure_thomas(rhs_hat) * pressure_mask[None]
        dpdx = jnp.fft.irfft2(
            1j * kx[None, None, :] * pressure_hat,
            s=(ny, nx),
            axes=(-2, -1),
        )
        dpdy = jnp.fft.irfft2(
            1j * pressure_ky[None, :, None] * pressure_hat,
            s=(ny, nx),
            axes=(-2, -1),
        )
        pressure = jnp.fft.irfft2(pressure_hat, s=(ny, nx), axes=(-2, -1))
        padded_pressure = pad(pressure)
        dpdz = ddz(padded_pressure, False)
        dpdz = dpdz.at[(1, nz), :, :].set(0.0)
        correction = pad(jnp.stack((dpdx, dpdy, dpdz[physical])))
        return velocity - cfg.dt * correction

    def step_impl(state: State) -> State:
        """Advance one filtered momentum step and enforce incompressibility."""

        # Build the explicit momentum right-hand side in the reference sign
        # convention: rotational convection plus divergence of modeled stress.
        velocity, gradient, wall_stress = velocity_gradient_and_wall(state.velocity)
        convection = rotational_convection(velocity, gradient)
        stress = mgm_stress(gradient, wall_stress)
        rhs = -convection - stress_divergence(stress)
        # A constant streamwise pressure gradient maintains the target wall
        # stress in this horizontally homogeneous neutral boundary layer.
        rhs = rhs.at[0, physical].add(
            cfg.pressure_force * forcing_levels[:, None, None]
        )
        # AB2 needs one previous RHS.  Use forward Euler only for iteration
        # zero, then switch to 3/2*rhs(n) - 1/2*rhs(n-1).
        tendency = jnp.where(
            state.iteration == 0,
            rhs,
            1.5 * rhs - 0.5 * state.previous_rhs,
        )
        velocity += cfg.dt * tendency
        velocity = velocity.at[2, (1, nz), :, :].set(0.0)
        velocity = project(velocity)
        return State(velocity, rhs, state.iteration + 1)

    def diagnostic_impl(velocity: Array) -> Array:
        """Return plane/global mean u and the resolved divergence infinity norm."""
        u = velocity[0, 1:nz]
        divergence = plain_derivative(velocity[0], 0)
        divergence += plain_derivative(velocity[1], 1)
        divergence += ddz(velocity[2], True)
        # k=nz is the duplicated top u/v plane and the Neumann pressure
        # boundary, not an interior Poisson row (matching pressure_thomas.cu).
        interior_divergence = divergence[1:nz]
        return jnp.asarray((jnp.mean(u), jnp.max(jnp.abs(interior_divergence))))

    def accumulate_impl(statistics: Array, velocity: Array) -> Array:
        """Add first and second moments needed for mean/turbulence profiles."""
        u, v, w = velocity[:, physical]
        snapshot = jnp.stack((u, v, w, u * u, v * v, w * w, u * v))
        return statistics + snapshot

    # Donating state/statistics buffers lets XLA reuse their device memory;
    # callers must use the returned value rather than access a donated input.
    return Operators(
        step=jax.jit(step_impl, donate_argnums=(0,)),
        clean=jax.jit(clean_velocity_impl),
        diagnostics=jax.jit(diagnostic_impl),
        accumulate=jax.jit(accumulate_impl, donate_argnums=(0,)),
    )


def write_outputs(
    clean_velocity: Array,
    statistics: Array | None,
    statistic_samples: int,
    cfg: Config,
    output_directory: Path,
) -> None:
    """Transfer final fields to the host and write profiles/statistics.

    ``mgm_profile.csv`` describes the final resolved snapshot.  Time averages
    retain their full 3-D shape in ``ta_*.bin`` so downstream tools can form
    arbitrary plane profiles and Reynolds moments without rerunning the LES.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    velocity = np.asarray(jax.device_get(clean_velocity), dtype=np.float32)

    # u/v are cell-centered.  Center w by averaging its bounding face values;
    # omit the duplicated top horizontal-velocity plane from the CSV profile.
    levels = np.arange(1, cfg.nz)
    z = (levels - 0.5) * cfg.dz * cfg.zi
    mean_u = velocity[0, 1 : cfg.nz].mean(axis=(1, 2))
    mean_v = velocity[1, 1 : cfg.nz].mean(axis=(1, 2))
    mean_w = 0.5 * (velocity[2, 1 : cfg.nz] + velocity[2, 2 : cfg.nz + 1]).mean(
        axis=(1, 2)
    )
    profile = np.column_stack((z, mean_u, mean_v, mean_w))
    np.savetxt(
        output_directory / "mgm_profile.csv",
        profile,
        delimiter=",",
        header="z_m,mean_u,mean_v,mean_w",
        comments="",
    )

    if statistics is not None and statistic_samples:
        averaged = np.asarray(jax.device_get(statistics)) / statistic_samples
        # Dividing once at output avoids a division on every sample.  Raw
        # C-order float32 files match the CUDA/JAX comparison tooling.
        for name, values in zip(
            ("u", "v", "w", "u2", "v2", "w2", "uv"), averaged, strict=True
        ):
            values.astype(np.float32).tofile(output_directory / f"ta_{name}.bin")
        print(f"Wrote {statistic_samples} matched statistics samples.")
    else:
        print(f"No statistics: run must reach step {cfg.statistics_start}.")
    print(f"Wrote {output_directory / 'mgm_profile.csv'}")


def run(
    cfg: Config,
    device: jax.Device,
    restart: Path | None,
    output_directory: Path,
    report_every: int,
) -> None:
    """Initialize device state, execute the time loop, and write outputs."""
    print(
        f"JAX MGM ABL: {cfg.nx}x{cfg.ny}x{cfg.nz}, {cfg.steps} steps, "
        f"device={device.platform}:{device.id}",
        flush=True,
    )
    # The context places initialization arrays and all closed-over constants
    # on the requested device before the first JIT compilation.
    with jax.default_device(device):
        velocity = initial_velocity(cfg, restart)
        state = State(
            velocity,
            jnp.zeros_like(velocity),
            jnp.asarray(0, dtype=jnp.int32),
        )
        operators = make_operators(cfg)

        # Allocate moment sums only when this run can actually reach the start
        # step.  Seven fields are u, v, w, u^2, v^2, w^2, and uv.
        statistics = (
            jnp.zeros((7, cfg.nz, cfg.ny, cfg.nx), dtype=jnp.float32)
            if cfg.steps >= cfg.statistics_start
            else None
        )

        started = time.perf_counter()
        statistic_samples = 0
        for iteration in range(1, cfg.steps + 1):
            # JAX dispatch is asynchronous: this Python loop queues compiled
            # kernels, while report points/device_get provide synchronization.
            state = operators.step(state)
            sample_now = (
                statistics is not None
                and iteration >= cfg.statistics_start
                and iteration % cfg.sample_every == 0
            )
            report_now = (
                iteration == 1
                or iteration % report_every == 0
                or iteration == cfg.steps
            )
            # Filtering only at sample/report points avoids extra FFTs on
            # ordinary steps.  Both consumers therefore see the same resolved
            # state, rather than an unfiltered intermediate AB2 field.
            clean_velocity = (
                operators.clean(state.velocity) if sample_now or report_now else None
            )
            if sample_now:
                statistics = operators.accumulate(statistics, clean_velocity)
                statistic_samples += 1
            if report_now:
                mean_u, max_divergence = np.asarray(
                    jax.device_get(operators.diagnostics(clean_velocity))
                )
                print(
                    f"step {iteration:7d}  t={iteration * cfg.dtr:9.1f} s  "
                    f"<u>={mean_u:.6f} m/s  "
                    f"|div_resolved|inf={max_divergence:.2e}",
                    flush=True,
                )

        # Synchronize before timing so elapsed includes all queued GPU work.
        clean_velocity = operators.clean(state.velocity)
        clean_velocity.block_until_ready()
        elapsed = time.perf_counter() - started
        print(f"Finished in {elapsed:.2f} s ({elapsed / cfg.steps:.4f} s/step).")
        write_outputs(
            clean_velocity, statistics, statistic_samples, cfg, output_directory
        )


def positive_int(value: str) -> int:
    """Argparse converter that rejects zero and negative integer controls."""
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def positive_float(value: str) -> float:
    """Argparse converter that rejects zero and negative float controls."""
    result = float(value)
    if result <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def parse_args() -> argparse.Namespace:
    """Expose common grid/run overrides while retaining study defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--nx", type=positive_int, default=Config.nx)
    parser.add_argument("--ny", type=positive_int, default=Config.ny)
    parser.add_argument("--nz", type=positive_int, default=Config.nz)
    parser.add_argument("--steps", type=positive_int, default=Config.steps)
    parser.add_argument(
        "--dtr",
        type=positive_float,
        default=Config.dtr,
        help="dimensional timestep in seconds (default: 0.5)",
    )
    parser.add_argument("--restart", type=Path, help="directory containing u/v/w.bin")
    parser.add_argument("--output-dir", type=Path, default=Path("jax_output"))
    parser.add_argument(
        "--report-every",
        type=positive_int,
        help="report interval (default: one tenth of the run)",
    )
    parser.add_argument(
        "--statistics-start", type=positive_int, default=Config.statistics_start
    )
    parser.add_argument(
        "--sample-every", type=positive_int, default=Config.sample_every
    )
    return parser.parse_args()


def main() -> None:
    """Validate CLI controls, select a device, and launch one simulation."""
    args = parse_args()
    if args.nx % 2 or args.ny % 2 or min(args.nx, args.ny, args.nz) < 4:
        raise SystemExit("nx and ny must be even; all dimensions must be at least 4")
    cfg = Config(
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        steps=args.steps,
        dtr=args.dtr,
        statistics_start=args.statistics_start,
        sample_every=args.sample_every,
    )
    if cfg.lr != round(cfg.lx / cfg.ly):
        raise SystemExit("lx/ly must round to the integer spectral aspect ratio lr")
    device = select_device(args.backend)
    report_every = args.report_every or max(1, cfg.steps // 10)
    run(cfg, device, args.restart, args.output_dir, report_every)


if __name__ == "__main__":
    main()

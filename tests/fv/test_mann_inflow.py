"""Physical and integration contracts for synthetic Mann inflow."""

import jax
import numpy as np
import pytest

from jaxwind import UniformGrid, build_mann_inflow, generate_mann_box
from jaxwind.inflow import _mann_factor
from jaxwind.io.inflow import write_synthetic_inflow
from jaxwind.io.recording import InflowReader
from jaxwind.open_boundary import validate_inflow_plane


def test_isotropic_tensor_and_singular_modes():
    k = np.array([[0.0, 0.0, 0.0], [0.0, 0.2, 0.3], [0.1, 0.2, -0.3], [0.0, 0.0, 0.2]])
    length, amplitude = 10.0, 0.7
    b = _mann_factor(k, length, 0.0, amplitude)
    for vector, factor in zip(k[1:], b[1:]):
        kk = vector @ vector
        expected = (
            amplitude
            * length ** (17 / 3)
            / (4 * np.pi * (1 + length**2 * kk) ** (17 / 6))
            * (kk * np.eye(3) - np.outer(vector, vector))
        )
        np.testing.assert_allclose(factor @ factor.T, expected, atol=1e-13)
    np.testing.assert_array_equal(b[0], 0.0)
    b = _mann_factor(k, length, 3.9, amplitude)
    assert np.isfinite(b).all()
    np.testing.assert_allclose(np.einsum("ni,nij->nj", k, b), 0.0, atol=1e-13)
    np.testing.assert_allclose(_mann_factor(-k, length, 3.9, amplitude), b, atol=1e-13)
    # At kx=0, exact RDT has u=u0-beta*w0 and w=w0.
    assert (b[1] @ b[1].T)[0, 2] < 0


@pytest.mark.parametrize("shape", [(16, 12, 10), (15, 11, 9)])
def test_reproducibility_scaling_and_spectral_divergence(shape):
    kwargs = {"shape": shape, "lengths": (300.0, 120.0, 100.0), "seed": 123}
    box = generate_mann_box(**kwargs)
    np.testing.assert_array_equal(box.velocity, generate_mann_box(**kwargs).velocity)
    assert not np.array_equal(
        box.velocity, generate_mann_box(**(kwargs | {"seed": 124})).velocity
    )
    np.testing.assert_allclose(box.velocity.mean(axis=(0, 1, 2)), 0.0, atol=1e-14)
    scaled = generate_mann_box(**kwargs, sigma_u=1.5)
    np.testing.assert_allclose(scaled.velocity[..., 0].std(), 1.5)
    np.testing.assert_allclose(
        generate_mann_box(**kwargs, alpha_epsilon=4.0).velocity, 2 * box.velocity
    )
    axes = [2 * np.pi * np.fft.fftfreq(n, d=l / n) for n, l in zip(shape, box.lengths)]
    kz, ky, kx = np.meshgrid(axes[2], axes[1], axes[0], indexing="ij")
    spectrum = np.fft.fftn(box.velocity, axes=(0, 1, 2))
    residual = np.sum(spectrum * np.stack((kx, ky, kz), axis=-1), axis=-1)
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)


def test_absolute_spectral_normalization():
    shape = (9, 9, 9)
    lengths = (100.0, 100.0, 100.0)
    k = np.array([2 * np.pi / 100, 2 * np.pi / 100, 0.0])
    factor = _mann_factor(k, 29.4, 3.9, 1.0)
    expected = (factor @ factor.T) * (2 * np.pi / 100) ** 3
    coefficients = []
    for seed in range(160):
        box = generate_mann_box(shape=shape, lengths=lengths, seed=seed)
        coefficients.append(
            np.fft.fftn(box.velocity, axes=(0, 1, 2))[0, 1, 1] / np.prod(shape)
        )
    coefficients = np.asarray(coefficients)
    covariance = (coefficients.T @ coefficients.conj()).real / len(coefficients)
    np.testing.assert_allclose(np.diag(covariance), np.diag(expected), rtol=0.2)
    assert covariance[0, 2] < 0


def test_sampler_jit_periodicity_staggering_and_recording(tmp_path):
    grid = UniformGrid(nx=8, ny=6, nz=5, lx=40.0, ly=30.0, lz=25.0)
    box = generate_mann_box(shape=(16, 12, 10), lengths=(80.0, 60.0, 50.0), seed=4)
    inflow = build_mann_inflow(box, grid, mean_speed=10.0, scalar=np.arange(5.0))
    plane = jax.jit(inflow)(0.25)
    validate_inflow_plane(plane, grid)
    for first, repeated in zip(plane, inflow(8.25)):
        np.testing.assert_allclose(first, repeated, atol=2e-6)
    np.testing.assert_array_equal(plane.z_velocity[[0, -1], :], 0.0)
    # At t=0.5, x=-5 wraps to node 15, while yz centres are half nodes.
    expected = (
        sum(box.velocity[z : z + 5, y : y + 6, 15, 0] for z in (0, 1) for y in (0, 1))
        / 4
        + 10
    )
    np.testing.assert_allclose(inflow(0.5).x_velocity, expected, atol=2e-6)
    walls = build_mann_inflow(box, grid, mean_speed=10.0, wall_y=True)(0.0)
    validate_inflow_plane(walls, grid, wall_y=True)
    np.testing.assert_array_equal(np.asarray(walls.y_velocity)[:, [0, -1]], 0.0)
    write_synthetic_inflow(tmp_path, grid, inflow, samples=5, dt=0.25, chunk_size=2)
    reader = InflowReader(tmp_path, grid, samples=5, dt=0.25)
    recorded = reader.read(1, 4)
    np.testing.assert_allclose(recorded.x_velocity[0], plane.x_velocity)
    with pytest.raises(ValueError, match="empty"):
        write_synthetic_inflow(tmp_path, grid, inflow, samples=5, dt=0.25)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shape": (2, 4, 4)},
        {"lengths": (0, 1, 1)},
        {"gamma": -1},
        {"length_scale": np.nan},
        {"alpha_epsilon": 0},
        {"sigma_u": -1},
    ],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        generate_mann_box(
            **({"shape": (5, 5, 5), "lengths": (10.0, 10.0, 10.0)} | kwargs)
        )


def test_shear_tensor_against_integrated_rapid_distortion():
    """Independent ODE reference checks shear signs and the arctangent branch."""
    from scipy.integrate import solve_ivp
    from scipy.special import hyp2f1

    for k in (
        np.array([0.1, 0.2, -0.3]),
        np.array([0.2, 0.01, -0.4]),
        np.array([0.0, 0.2, 0.3]),
    ):
        length, gamma = 10.0, 3.9
        kl = np.linalg.norm(k) * length
        beta = gamma * kl ** (-2 / 3) / np.sqrt(hyp2f1(1 / 3, 17 / 6, 4 / 3, -(kl**-2)))
        k0 = k + np.array([0.0, 0.0, beta * k[0]])
        initial = _mann_factor(k0, length, 0.0, 1.0)

        def rhs(strain, flattened, k0=k0, k=k):
            wave = k0 - np.array([0.0, 0.0, strain * k[0]])
            matrix = flattened.reshape(3, 3)
            derivative = 2 * wave[:, None] * wave[0] / (wave @ wave) * matrix[2]
            derivative[0] -= matrix[2]
            return derivative.ravel()

        result = solve_ivp(rhs, (0.0, beta), initial.ravel(), rtol=1e-10, atol=1e-12)
        reference = result.y[:, -1].reshape(3, 3)
        actual = _mann_factor(k, length, gamma, 1.0)
        np.testing.assert_allclose(
            actual @ actual.T, reference @ reference.T, rtol=1e-8, atol=1e-10
        )

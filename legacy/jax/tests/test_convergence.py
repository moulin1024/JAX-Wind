from __future__ import annotations

import pytest

from wireles_jax.convergence import UStarSlidingWindow


def test_ustar_window_requires_full_window_and_minimum_step() -> None:
    criterion = UStarSlidingWindow(
        target=0.2,
        relative_tolerance=0.02,
        window_samples=3,
        minimum_step=4,
    )
    assert not criterion.update(1, 0.2)
    assert not criterion.update(2, 0.2)
    assert not criterion.update(3, 0.2)
    assert criterion.update(4, 0.2)
    assert criterion.mean == pytest.approx(0.2)


def test_ustar_window_uses_moving_not_cumulative_mean() -> None:
    criterion = UStarSlidingWindow(
        target=0.2,
        relative_tolerance=0.01,
        window_samples=3,
    )
    assert not criterion.update(1, 0.4)
    assert not criterion.update(2, 0.2)
    assert not criterion.update(3, 0.2)
    assert criterion.update(4, 0.2)
    assert criterion.sample_count == 3


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"target": 0.0}, "target ustar"),
        ({"relative_tolerance": 0.0}, "relative_tolerance"),
        ({"window_samples": 0}, "window_samples"),
        ({"minimum_step": -1}, "minimum_step"),
    ],
)
def test_ustar_window_rejects_invalid_settings(kwargs, message: str) -> None:
    settings = {
        "target": 0.2,
        "relative_tolerance": 0.02,
        "window_samples": 3,
        "minimum_step": 0,
    }
    settings.update(kwargs)
    with pytest.raises(ValueError, match=message):
        UStarSlidingWindow(**settings)


def test_sharded_driver_honors_diagnostic_stop_callback() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import Params
    from wireles_jax.timestep_sharded import run_sharded

    params = Params(
        nx=4,
        ny=4,
        nz=4,
        lx=2.0,
        ly=2.0,
        lz=1.0,
        nsteps=5,
        dt=1.0e-4,
        c_count=1,
        time_scheme="ab2",
        sgs_model="smagorinsky",
        momentum_wall_model="free_slip",
        use_jit=False,
        dtype=jnp.float32,
    )
    sampled_steps: list[int] = []

    def stop(diag) -> bool:
        sampled_steps.append(int(diag.step))
        return int(diag.step) >= 2

    state, _ = run_sharded(
        params,
        num_devices=1,
        log_every=1,
        stop_callback=stop,
    )
    assert int(state.step) == 2
    assert sampled_steps == [0, 1, 2]


def test_sharded_driver_projects_noisy_initial_velocity_before_step_zero() -> None:
    jnp = pytest.importorskip("jax.numpy")

    from wireles_jax import Params
    from wireles_jax.timestep_sharded import run_sharded

    params = Params(
        nx=8,
        ny=8,
        nz=8,
        nsteps=1,
        dt=1.0e-4,
        time_scheme="ab2",
        sgs_model="smagorinsky",
        momentum_wall_model="free_slip",
        initial_velocity_noise=0.1,
        use_jit=False,
        dtype=jnp.float32,
    )
    divergence: list[float] = []

    def stop(diag) -> bool:
        divergence.append(float(diag.div_max))
        return True

    state, _ = run_sharded(
        params,
        num_devices=1,
        log_every=1,
        stop_callback=stop,
    )
    assert int(state.step) == 0
    assert divergence[0] < 2.0e-5

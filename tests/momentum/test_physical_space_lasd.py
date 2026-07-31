from __future__ import annotations

import jax.numpy as jnp

from jaxwind.momentum import (
    LASDModel,
    PhysicalSpaceLASD,
    physical_top_hat_filter,
    physical_top_hat_filter_pair,
    top_hat_stencil,
)


def test_top_hat_stencil_matches_cell_overlap_geometry() -> None:
    assert top_hat_stencil(3.0) == (
        (-1, 1.0 / 3.0),
        (0, 1.0 / 3.0),
        (1, 1.0 / 3.0),
    )
    offsets, weights = zip(*top_hat_stencil(6.0), strict=True)
    assert offsets == (-3, -2, -1, 0, 1, 2, 3)
    assert jnp.allclose(
        jnp.asarray(weights),
        jnp.asarray((0.5, 1, 1, 1, 1, 1, 0.5)) / 6.0,
    )


def test_reflected_filter_preserves_constants_without_periodic_wrap() -> None:
    constant = jnp.ones((2, 8, 8), dtype=jnp.float32)
    assert jnp.allclose(
        physical_top_hat_filter(
            constant,
            3.0,
            boundaries=("reflect", "reflect"),
        ),
        constant,
    )

    impulse = jnp.zeros_like(constant).at[:, :, 0].set(1.0)
    reflected = physical_top_hat_filter(
        impulse,
        3.0,
        boundaries=("reflect", "reflect"),
    )
    periodic = physical_top_hat_filter(
        impulse,
        3.0,
        boundaries=("periodic", "periodic"),
    )
    assert float(jnp.max(jnp.abs(reflected[:, :, -1]))) == 0.0
    assert float(jnp.max(periodic[:, :, -1])) > 0.0


def test_odd_reflection_changes_sign_across_a_physical_wall() -> None:
    constant = jnp.ones((6,), dtype=jnp.float32)
    filtered = physical_top_hat_filter(
        constant,
        3.0,
        axes=(0,),
        boundaries=("reflect_odd",),
    )
    assert float(filtered[0]) == jnp.float32(1.0 / 3.0)
    assert float(filtered[-1]) == jnp.float32(1.0 / 3.0)
    assert jnp.allclose(filtered[1:-1], 1.0)


def test_lasd_filter_is_three_dimensional_and_component_aware() -> None:
    model = LASDModel()
    closure = PhysicalSpaceLASD(dx=1.0, dy=1.0, dz=1.0, model=model)
    alternating = (jnp.arange(8) % 2).astype(jnp.float32)[:, None, None]
    scalar = jnp.broadcast_to(alternating, (8, 4, 4))
    filtered_scalar = closure._filter(scalar, 3.0)
    assert not jnp.allclose(filtered_scalar, scalar)

    vector = jnp.ones((8, 4, 4, 3), dtype=jnp.float32)
    filtered_vector = closure._filter(
        vector,
        3.0,
        odd_z_components=(2,),
    )
    assert jnp.allclose(filtered_vector[0, ..., :2], 1.0)
    assert jnp.allclose(filtered_vector[0, ..., 2], 1.0 / 3.0)


def test_fast_even_width_filters_match_overlap_stencils() -> None:
    values = jnp.arange(8, dtype=jnp.float32)
    for width in (2.0, 4.0):
        stencil = top_hat_stencil(width)
        expected = sum(
            weight * jnp.roll(values, -offset)
            for offset, weight in stencil
        )
        actual = physical_top_hat_filter(
            values,
            width,
            axes=(0,),
            boundaries=("periodic",),
        )
        assert jnp.allclose(actual, expected)


def test_fast_filter_supports_component_aware_odd_reflection() -> None:
    values = jnp.ones((8, 2), dtype=jnp.float32)
    filtered = physical_top_hat_filter(
        values,
        4.0,
        axes=(0,),
        boundaries=("reflect",),
        odd_reflect_components=(1,),
    )
    even = physical_top_hat_filter(
        values[..., 0],
        4.0,
        axes=(0,),
        boundaries=("reflect",),
    )
    odd = physical_top_hat_filter(
        values[..., 1],
        4.0,
        axes=(0,),
        boundaries=("reflect_odd",),
    )
    assert jnp.allclose(filtered[..., 0], even)
    assert jnp.allclose(filtered[..., 1], odd)


def test_joint_even_width_filter_matches_independent_filters() -> None:
    values = jnp.arange(8 * 6 * 4 * 3, dtype=jnp.float32).reshape(
        8,
        6,
        4,
        3,
    )
    joint_two, joint_four = physical_top_hat_filter_pair(
        values,
        axes=(0, 1, 2),
        boundaries=("reflect", "periodic", "periodic"),
        odd_reflect_components=(2,),
    )
    for width, joint in ((2.0, joint_two), (4.0, joint_four)):
        independent = physical_top_hat_filter(
            values,
            width,
            axes=(0, 1, 2),
            boundaries=("reflect", "periodic", "periodic"),
            odd_reflect_components=(2,),
        )
        assert jnp.allclose(joint, independent)


def test_batched_departure_matches_independent_scalar_departures() -> None:
    model = LASDModel()
    closure = PhysicalSpaceLASD(dx=1.0, dy=1.0, dz=1.0, model=model)
    velocity = jnp.ones((4, 5, 6, 3), dtype=jnp.float32)
    state = closure.accumulate(closure.initialize(velocity), velocity)
    fields = jnp.arange(4 * 5 * 6 * 4, dtype=jnp.float32).reshape(
        4, 5, 6, 4
    )
    batched = closure._departure(fields, state, 0.1)
    independent = jnp.stack(
        tuple(
            closure._departure(fields[..., component], state, 0.1)
            for component in range(fields.shape[-1])
        ),
        axis=-1,
    )
    assert jnp.allclose(batched, independent)


def test_lasd_update_is_finite_and_bounded() -> None:
    nz = ny = nx = 8
    z, y, x = jnp.meshgrid(
        jnp.arange(nz, dtype=jnp.float32),
        jnp.arange(ny, dtype=jnp.float32),
        jnp.arange(nx, dtype=jnp.float32),
        indexing="ij",
    )
    velocity = jnp.stack(
        (
            jnp.sin(0.3 * x + 0.2 * y),
            jnp.cos(0.4 * y - 0.1 * z),
            0.2 * jnp.sin(0.5 * x + 0.3 * z),
        ),
        axis=-1,
    )
    gradient = jnp.stack(
        tuple(
            jnp.stack(
                (
                    jnp.gradient(velocity[..., component], axis=2),
                    jnp.gradient(velocity[..., component], axis=1),
                    jnp.gradient(velocity[..., component], axis=0),
                ),
                axis=-1,
            )
            for component in range(3)
        ),
        axis=-2,
    )
    model = LASDModel(x_boundary="reflect", y_boundary="reflect")
    closure = PhysicalSpaceLASD(dx=1.0, dy=1.0, dz=1.0, model=model)
    state = closure.accumulate(closure.initialize(velocity), velocity)
    updated = closure.update(
        state,
        velocity,
        gradient,
        interval_dt=0.1,
        first_update=True,
    )

    assert jnp.all(jnp.isfinite(updated.coefficient))
    assert float(jnp.min(updated.coefficient)) >= model.minimum_coefficient
    assert float(jnp.max(updated.coefficient)) <= model.maximum_coefficient
    assert float(jnp.max(jnp.abs(updated.trajectory_x))) == 0.0


def test_joint_lasd_contractions_match_independent_scale_paths() -> None:
    nz = ny = nx = 8
    z, y, x = jnp.meshgrid(
        jnp.arange(nz, dtype=jnp.float32),
        jnp.arange(ny, dtype=jnp.float32),
        jnp.arange(nx, dtype=jnp.float32),
        indexing="ij",
    )
    velocity = jnp.stack(
        (
            jnp.sin(0.3 * x + 0.2 * y),
            jnp.cos(0.4 * y - 0.1 * z),
            0.2 * jnp.sin(0.5 * x + 0.3 * z),
        ),
        axis=-1,
    )
    gradient = jnp.stack(
        tuple(
            jnp.stack(
                (
                    jnp.gradient(velocity[..., component], axis=2),
                    jnp.gradient(velocity[..., component], axis=1),
                    jnp.gradient(velocity[..., component], axis=0),
                ),
                axis=-1,
            )
            for component in range(3)
        ),
        axis=-2,
    )
    closure = PhysicalSpaceLASD(
        dx=1.0,
        dy=1.0,
        dz=1.0,
        model=LASDModel(),
    )
    ratio = closure.model.test_filter_ratio
    independent = (
        *closure._contractions(velocity, gradient, ratio),
        *closure._contractions(velocity, gradient, ratio**2),
    )
    joint = closure.contractions(velocity, gradient)

    for actual, expected in zip(joint, independent, strict=True):
        assert jnp.allclose(actual, expected, rtol=2.0e-5, atol=2.0e-6)


def test_physical_lasd_defaults_to_unity_for_clipped_beta() -> None:
    model = LASDModel()
    assert model.clipped_beta_fallback
    assert model.update_interval == 1
    assert model.effective_delta_scale == model.filter_grid_ratio

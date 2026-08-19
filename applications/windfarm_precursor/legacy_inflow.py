"""Literal WIRE-LES/CUDA-Fortran precursor-inlet semantics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True, slots=True)
class LegacyInflowContract:
    """One-based inlet parameters used by the production Fortran case."""

    start_plane: int = 10
    end_plane: int = 20
    update_interval_steps: int = 10
    cycle_interval_updates: int = 4

    def __post_init__(self) -> None:
        values = (
            self.start_plane,
            self.end_plane,
            self.update_interval_steps,
            self.cycle_interval_updates,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in values
        ):
            raise TypeError("legacy inlet parameters must be integers")
        if self.start_plane <= 1 or self.end_plane < self.start_plane:
            raise ValueError("legacy inlet plane range is invalid")
        if self.update_interval_steps <= 0 or self.cycle_interval_updates <= 0:
            raise ValueError("legacy inlet cadences must be positive")

    @property
    def width(self) -> int:
        return self.end_plane - self.start_plane + 1

    @property
    def zero_based_start(self) -> int:
        return self.start_plane - 1


STRICT_LEGACY_INFLOW = LegacyInflowContract()


def force_inflow_component(
    payload,
    target_payload,
    shift,
    blend,
    *,
    jnp,
    section_width: int = 11,
):
    """Translate legacy ``force_inflow`` including its k+1-to-k write."""

    source_block = jnp.roll(
        target_payload[..., :section_width],
        shift,
        axis=-2,
    )
    source = source_block[..., 0]
    base = payload[..., 0]
    blend_width = blend.shape[0]
    blended = payload[..., :blend_width]
    shifted = base[:, 1:, :, None] + blend * (
        source[:, 1:, :] - base[:, 1:, :]
    )[..., None]
    blended = blended.at[:, :-1, :, :].set(shifted)
    payload = payload.at[..., :blend_width].set(blended)
    return payload.at[
        ..., blend_width : blend_width + section_width
    ].set(source_block)


def build_accepted_state_transform(
    *,
    contract: LegacyInflowContract,
    jax: Any,
    jnp: Any,
    ny: int,
):
    """Build the post-projection inlet overwrite used by Fortran ``main.cuf``."""

    transforms: dict[int, Any] = {}
    blend_width = contract.start_plane - 1
    blend = jnp.asarray(
        0.5
        * (
            1.0
            - jnp.cos(
                jnp.pi
                * jnp.arange(blend_width, dtype=jnp.float32)
                / float(blend_width - 1)
            )
        )
    )

    def accepted_state_transform(state, environment, completed: int):
        if (completed - 1) % contract.update_interval_steps != 0:
            return state
        shift = (
            (completed - 1)
            // (
                contract.update_interval_steps
                * contract.cycle_interval_updates
            )
        ) % ny + 1
        transform = transforms.get(shift)
        if transform is None:

            def apply(current, target):
                velocity = current.fields.velocity
                target_velocity = target.velocity

                def component(field, target_field):
                    return replace(
                        field,
                        payload=force_inflow_component(
                            field.payload,
                            target_field.payload,
                            shift,
                            blend,
                            jnp=jnp,
                            section_width=contract.width,
                        ),
                    )

                updated_velocity = replace(
                    velocity,
                    x=component(velocity.x, target_velocity.x),
                    y=component(velocity.y, target_velocity.y),
                    z=replace(
                        velocity.z,
                        owned=component(
                            velocity.z.owned,
                            target_velocity.z.owned,
                        ),
                    ),
                )
                return replace(
                    current,
                    fields=replace(current.fields, velocity=updated_velocity),
                )

            transform = jax.jit(apply)
            transforms[shift] = transform
        return transform(state, environment)

    return accepted_state_transform


__all__ = [
    "LegacyInflowContract",
    "STRICT_LEGACY_INFLOW",
    "build_accepted_state_transform",
    "force_inflow_component",
]

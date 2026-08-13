"""Accepted-boundary AB2 checkpoint effects with owned-payload serialization."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from jaxwind.domain import (
    Accepted,
    AcceptedClock,
    AddressableField,
    Cell,
    EqualVerticalPartition,
    Evaluated,
    Field,
    GlobalTestRegion,
    LasdTrajectoryXVelocity,
    LasdTrajectoryYVelocity,
    LasdTrajectoryZVelocity,
    MomentumLasdCoefficient,
    MomentumLasdLm,
    MomentumLasdMm,
    MomentumLasdNn,
    MomentumLasdQn,
    PassiveScalarConcentration,
    PassiveScalarTendency,
    PotentialTemperaturePerturbation,
    PotentialTemperatureTendency,
    Projected,
    ScalarLasdCoefficient,
    ScalarLasdLm,
    ScalarLasdMm,
    ScalarLasdNn,
    ScalarLasdQn,
    UniformGrid,
    VerticalVelocity,
    VerticalVelocityTendency,
    XVelocity,
    XVelocityTendency,
    YVelocity,
    YVelocityTendency,
    ZFace,
)
from jaxwind.integrators import (
    AB2BoussinesqState,
    AB2Config,
    AB2PersistentState,
    ColdStart,
    PreviousTendency,
)
from jaxwind.operators import VelocityVector
from jaxwind.physics import (
    BoussinesqFields,
    BoussinesqTendency,
    LasdClosureMemory,
    MomentumLasdMemory,
    NoClosureMemory,
    ScalarLasdMemory,
)


SCHEMA = "jaxwind.ab2.accepted-checkpoint.v1"
BOUSSINESQ_SCHEMA_V1 = "jaxwind.ab2.boussinesq-accepted-checkpoint.v1"
BOUSSINESQ_SCHEMA = "jaxwind.ab2.boussinesq-accepted-checkpoint.v2"


_CLOSURE_FIELDS = (
    ("momentum_coefficient", "momentum", "coefficient", MomentumLasdCoefficient),
    ("momentum_lm", "momentum", "lm", MomentumLasdLm),
    ("momentum_mm", "momentum", "mm", MomentumLasdMm),
    ("momentum_qn", "momentum", "qn", MomentumLasdQn),
    ("momentum_nn", "momentum", "nn", MomentumLasdNn),
    (
        "trajectory_x",
        "momentum",
        "trajectory_x",
        LasdTrajectoryXVelocity,
    ),
    (
        "trajectory_y",
        "momentum",
        "trajectory_y",
        LasdTrajectoryYVelocity,
    ),
    (
        "trajectory_z",
        "momentum",
        "trajectory_z",
        LasdTrajectoryZVelocity,
    ),
    ("scalar_coefficient", "scalar", "coefficient", ScalarLasdCoefficient),
    ("scalar_lm", "scalar", "lm", ScalarLasdLm),
    ("scalar_mm", "scalar", "mm", ScalarLasdMm),
    ("scalar_qn", "scalar", "qn", ScalarLasdQn),
    ("scalar_nn", "scalar", "nn", ScalarLasdNn),
)


@dataclass(frozen=True, slots=True)
class ReferenceCheckpointLayout:
    grid: UniformGrid
    array_factory: Callable[[np.ndarray], Any]


@dataclass(frozen=True, slots=True)
class DistributedCheckpointLayout:
    decomposition: EqualVerticalPartition
    addressable_partitions: tuple[int, ...]
    array_factory: Callable[[np.ndarray], Any]

    def __post_init__(self) -> None:
        if not self.addressable_partitions:
            raise ValueError("checkpoint layout requires addressable partitions")
        if any(
            not 0 <= partition < self.decomposition.partition_count
            for partition in self.addressable_partitions
        ):
            raise ValueError("checkpoint partition is outside the decomposition")


CheckpointLayout = ReferenceCheckpointLayout | DistributedCheckpointLayout


def _grid_metadata(grid: UniformGrid) -> dict[str, Any]:
    return {
        "nx": grid.nx,
        "ny": grid.ny,
        "nz": grid.nz,
        "lx": grid.lx,
        "ly": grid.ly,
        "lz": grid.lz,
    }


def _representation_and_grid(velocity: VelocityVector) -> tuple[str, UniformGrid]:
    if isinstance(velocity.x, Field):
        return "reference-global-test", velocity.x.ownership.grid
    if isinstance(velocity.x, AddressableField):
        return "owned-z-slab", velocity.x.regions[0].grid
    raise TypeError("unsupported checkpoint velocity representation")


def _state_metadata(
    state: Any,
    velocity: VelocityVector,
    *,
    schema: str,
) -> tuple[dict[str, Any], str]:
    representation, grid = _representation_and_grid(velocity)
    metadata = {
        "schema": schema,
        "representation": representation,
        "grid": _grid_metadata(grid),
        "clock": {
            "time": float(state.clock.time),
            "step": int(state.clock.step),
        },
        "integrator_fingerprint": state.integrator_fingerprint,
        "history": (
            "cold-start"
            if isinstance(state.history, ColdStart)
            else "previous-tendency"
        ),
    }
    if representation == "owned-z-slab":
        # Preserve the v1/v2 serialized key so existing production restart
        # files remain loadable after the public ownership rename.
        metadata["addressable_shards"] = [
            region.coordinate.indices[0] for region in velocity.x.regions
        ]
    return metadata, representation


def _write_archive(
    path: str | Path,
    metadata: dict[str, Any],
    arrays: dict[str, Any],
) -> None:
    target = Path(path)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez(stream, metadata=np.asarray(json.dumps(metadata)), **arrays)
    os.replace(temporary, target)


def _velocity_arrays(velocity: VelocityVector, representation: str) -> dict[str, Any]:
    arrays = {
        "x": np.asarray(velocity.x.payload),
        "y": np.asarray(velocity.y.payload),
    }
    if representation == "reference-global-test":
        arrays["z"] = np.asarray(velocity.z.payload)
    else:
        arrays["z"] = np.asarray(velocity.z.owned.payload)
        arrays["z_lower_boundary"] = np.asarray(velocity.z.lower_boundary)
    return arrays


def save_ab2_checkpoint(path: str | Path, state: AB2PersistentState) -> None:
    """Atomically save one reference state or one process's owned regions."""
    metadata, representation = _state_metadata(
        state,
        state.velocity,
        schema=SCHEMA,
    )
    arrays = {
        f"velocity_{name}": value
        for name, value in _velocity_arrays(state.velocity, representation).items()
    }
    if isinstance(state.history, PreviousTendency):
        arrays.update(
            {
                f"history_{name}": value
                for name, value in _velocity_arrays(
                    state.history.value,
                    representation,
                ).items()
            }
        )
    _write_archive(path, metadata, arrays)


def _load_array(archive, name: str, factory: Callable) -> Any:
    return factory(np.array(archive[name], copy=True))


def _load_distributed_array(
    archive,
    name: str,
    layout: DistributedCheckpointLayout,
    *,
    reshard: bool,
) -> Any:
    values = np.array(archive[name], copy=True)
    if reshard and values.ndim == 4:
        grid = layout.decomposition.grid
        expected = (
            len(layout.addressable_partitions),
            layout.decomposition.cells_per_partition,
            grid.ny,
            grid.nx,
        )
        if values.size != math.prod(expected):
            raise ValueError("checkpoint payload cannot be reshaped to the load layout")
        values = values.reshape(expected)
    return layout.array_factory(values)


def _validate_metadata(
    metadata: dict[str, Any],
    layout: CheckpointLayout,
    config: AB2Config,
    allow_reshard: bool = False,
) -> str:
    if metadata.get("schema") != SCHEMA:
        raise ValueError("unsupported AB2 checkpoint schema")
    expected_representation = (
        "reference-global-test"
        if isinstance(layout, ReferenceCheckpointLayout)
        else "owned-z-slab"
    )
    if metadata.get("representation") != expected_representation:
        raise ValueError("checkpoint representation does not match the load layout")
    grid = (
        layout.grid
        if isinstance(layout, ReferenceCheckpointLayout)
        else layout.decomposition.grid
    )
    if metadata.get("grid") != _grid_metadata(grid):
        raise ValueError("checkpoint grid does not match the load layout")
    if metadata.get("integrator_fingerprint") != config.fingerprint:
        raise ValueError("checkpoint integrator fingerprint does not match")
    if isinstance(layout, DistributedCheckpointLayout):
        stored = metadata.get(
            "addressable_shards",
            metadata.get("addressable_partitions", ()),
        )
        stored = tuple(stored)
        full_destination = tuple(range(layout.decomposition.partition_count))
        can_reshard = (
            allow_reshard
            and stored == (0,)
            and layout.addressable_partitions == full_destination
        )
        if stored != layout.addressable_partitions and not can_reshard:
            raise ValueError("checkpoint partitions do not match the load layout")
    history = metadata.get("history")
    if history not in ("cold-start", "previous-tendency"):
        raise ValueError("checkpoint has an invalid AB2 history tag")
    return history


def _reference_velocity(
    archive, prefix: str, layout: ReferenceCheckpointLayout, *, tendency: bool
):
    grid = layout.grid
    cells = GlobalTestRegion(grid, Cell)
    faces = GlobalTestRegion(grid, ZFace)
    phase = Evaluated if tendency else Projected
    return VelocityVector(
        Field(
            XVelocityTendency if tendency else XVelocity,
            Cell,
            cells,
            phase,
            _load_array(archive, f"{prefix}_x", layout.array_factory),
        ),
        Field(
            YVelocityTendency if tendency else YVelocity,
            Cell,
            cells,
            phase,
            _load_array(archive, f"{prefix}_y", layout.array_factory),
        ),
        Field(
            VerticalVelocityTendency if tendency else VerticalVelocity,
            ZFace,
            faces,
            phase,
            _load_array(archive, f"{prefix}_z", layout.array_factory),
        ),
    )


def _distributed_velocity(
    archive,
    prefix: str,
    layout: DistributedCheckpointLayout,
    *,
    tendency: bool,
    reshard: bool = False,
):
    from jaxwind.domain import VerticalFaceField

    decomposition = layout.decomposition
    cells = decomposition.regions(Cell)
    faces = decomposition.regions(ZFace)
    cell_regions = tuple(cells[index] for index in layout.addressable_partitions)
    face_regions = tuple(faces[index] for index in layout.addressable_partitions)
    phase = Evaluated if tendency else Projected
    return VelocityVector(
        AddressableField(
            XVelocityTendency if tendency else XVelocity,
            Cell,
            cell_regions,
            phase,
            _load_distributed_array(
                archive, f"{prefix}_x", layout, reshard=reshard
            ),
        ),
        AddressableField(
            YVelocityTendency if tendency else YVelocity,
            Cell,
            cell_regions,
            phase,
            _load_distributed_array(
                archive, f"{prefix}_y", layout, reshard=reshard
            ),
        ),
        VerticalFaceField(
            AddressableField(
                VerticalVelocityTendency if tendency else VerticalVelocity,
                ZFace,
                face_regions,
                phase,
                _load_distributed_array(
                    archive, f"{prefix}_z", layout, reshard=reshard
                ),
            ),
            _load_array(
                archive,
                f"{prefix}_z_lower_boundary",
                layout.array_factory,
            ),
        ),
    )


def load_ab2_checkpoint(
    path: str | Path,
    *,
    layout: CheckpointLayout,
    config: AB2Config,
) -> AB2PersistentState:
    """Load an accepted state into an explicitly supplied ownership layout."""
    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        history_tag = _validate_metadata(metadata, layout, config)
        if isinstance(layout, ReferenceCheckpointLayout):
            velocity = _reference_velocity(archive, "velocity", layout, tendency=False)
            tendency_loader = _reference_velocity
        else:
            velocity = _distributed_velocity(
                archive, "velocity", layout, tendency=False
            )
            tendency_loader = _distributed_velocity
        if history_tag == "cold-start":
            history = ColdStart()
        else:
            history = PreviousTendency(
                tendency_loader(archive, "history", layout, tendency=True)
            )
    clock = AcceptedClock(
        float(metadata["clock"]["time"]),
        int(metadata["clock"]["step"]),
    )
    return AB2PersistentState(velocity, clock, history, config.fingerprint)


def _scalar_arrays(scalar: Field | AddressableField) -> np.ndarray:
    return np.asarray(scalar.payload)


def _scalar_quantity_tag(scalar: Field | AddressableField) -> str:
    if scalar.quantity is PotentialTemperaturePerturbation:
        return "potential-temperature-perturbation"
    if scalar.quantity is PassiveScalarConcentration:
        return "passive-scalar-concentration"
    raise TypeError("unsupported Boussinesq checkpoint scalar quantity")


def _closure_metadata(closure: Any) -> dict[str, Any]:
    if isinstance(closure, NoClosureMemory):
        return {"kind": "none"}
    if isinstance(closure, LasdClosureMemory):
        return {
            "kind": "lasd",
            "configuration_fingerprint": closure.configuration_fingerprint,
        }
    raise TypeError("unsupported Boussinesq checkpoint closure memory")


def _closure_arrays(closure: Any) -> dict[str, np.ndarray]:
    if isinstance(closure, NoClosureMemory):
        return {}
    if not isinstance(closure, LasdClosureMemory):
        raise TypeError("unsupported Boussinesq checkpoint closure memory")
    arrays = {}
    for storage_name, group, attribute, _quantity in _CLOSURE_FIELDS:
        memory = getattr(closure, group)
        arrays[f"closure_{storage_name}"] = _scalar_arrays(getattr(memory, attribute))
    return arrays


def save_boussinesq_checkpoint(
    path: str | Path,
    state: AB2BoussinesqState,
    *,
    scale_fingerprint: str | None = None,
    physics_fingerprint: str | None = None,
) -> None:
    """Atomically save velocity, scalar, and both previous AB2 tendencies."""
    velocity = state.fields.velocity
    metadata, representation = _state_metadata(
        state,
        velocity,
        schema=BOUSSINESQ_SCHEMA,
    )
    metadata.update({
        "scalar_quantity": _scalar_quantity_tag(state.fields.potential_temperature),
        "closure": _closure_metadata(state.fields.closure),
    })
    if scale_fingerprint is not None:
        if not scale_fingerprint:
            raise ValueError("scale fingerprint must be non-empty")
        metadata["scale_fingerprint"] = scale_fingerprint
    if physics_fingerprint is not None:
        if not physics_fingerprint:
            raise ValueError("physics fingerprint must be non-empty")
        metadata["physics_fingerprint"] = physics_fingerprint
    arrays = {
        f"velocity_{name}": value
        for name, value in _velocity_arrays(velocity, representation).items()
    }
    arrays["scalar"] = _scalar_arrays(state.fields.potential_temperature)
    arrays.update(_closure_arrays(state.fields.closure))
    if isinstance(state.history, PreviousTendency):
        arrays.update(
            {
                f"history_velocity_{name}": value
                for name, value in _velocity_arrays(
                    state.history.value.velocity,
                    representation,
                ).items()
            }
        )
        arrays["history_scalar"] = _scalar_arrays(
            state.history.value.potential_temperature
        )
    _write_archive(path, metadata, arrays)


def _checkpoint_scalar(
    archive,
    name: str,
    layout: CheckpointLayout,
    *,
    tendency: bool,
    scalar_quantity: str = "potential-temperature-perturbation",
    reshard: bool = False,
):
    quantities = {
        "potential-temperature-perturbation": (
            PotentialTemperaturePerturbation,
            PotentialTemperatureTendency,
        ),
        "passive-scalar-concentration": (
            PassiveScalarConcentration,
            PassiveScalarTendency,
        ),
    }
    try:
        accepted_quantity, tendency_quantity = quantities[scalar_quantity]
    except KeyError as exc:
        raise ValueError("checkpoint has an unsupported scalar quantity") from exc
    quantity = tendency_quantity if tendency else accepted_quantity
    phase = Evaluated if tendency else Accepted
    if isinstance(layout, ReferenceCheckpointLayout):
        return Field(
            quantity,
            Cell,
            GlobalTestRegion(layout.grid, Cell),
            phase,
            _load_array(archive, name, layout.array_factory),
        )
    regions = layout.decomposition.regions(Cell)
    return AddressableField(
        quantity,
        Cell,
        tuple(regions[index] for index in layout.addressable_partitions),
        phase,
        _load_distributed_array(archive, name, layout, reshard=reshard),
    )


def _checkpoint_closure_field(
    archive,
    name: str,
    quantity: type,
    layout: CheckpointLayout,
    *,
    reshard: bool = False,
):
    if isinstance(layout, ReferenceCheckpointLayout):
        return Field(
            quantity,
            Cell,
            GlobalTestRegion(layout.grid, Cell),
            Accepted,
            _load_array(archive, name, layout.array_factory),
        )
    regions = layout.decomposition.regions(Cell)
    return AddressableField(
        quantity,
        Cell,
        tuple(regions[index] for index in layout.addressable_partitions),
        Accepted,
        _load_distributed_array(archive, name, layout, reshard=reshard),
    )


def _checkpoint_closure(
    archive,
    metadata: dict[str, Any],
    layout: CheckpointLayout,
    *,
    reshard: bool = False,
) -> NoClosureMemory | LasdClosureMemory:
    closure_metadata = metadata.get("closure", {"kind": "none"})
    if closure_metadata.get("kind") == "none":
        return NoClosureMemory()
    if closure_metadata.get("kind") != "lasd":
        raise ValueError("checkpoint has an unsupported closure-memory kind")
    fingerprint = closure_metadata.get("configuration_fingerprint", "")
    values = {}
    for storage_name, group, attribute, quantity in _CLOSURE_FIELDS:
        values[(group, attribute)] = _checkpoint_closure_field(
            archive,
            f"closure_{storage_name}",
            quantity,
            layout,
            reshard=reshard,
        )
    momentum = MomentumLasdMemory(
        *(
            values[("momentum", attribute)]
            for attribute in (
                "coefficient",
                "lm",
                "mm",
                "qn",
                "nn",
                "trajectory_x",
                "trajectory_y",
                "trajectory_z",
            )
        )
    )
    scalar = ScalarLasdMemory(
        *(
            values[("scalar", attribute)]
            for attribute in (
                "coefficient",
                "lm",
                "mm",
                "qn",
                "nn",
            )
        )
    )
    return LasdClosureMemory(momentum, scalar, fingerprint)


def load_boussinesq_checkpoint(
    path: str | Path,
    *,
    layout: CheckpointLayout,
    config: AB2Config,
    scale_fingerprint: str | None = None,
    closure_fingerprint: str | None = None,
    physics_fingerprint: str | None = None,
    allow_single_process_reshard: bool = False,
) -> AB2BoussinesqState:
    with np.load(Path(path), allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if metadata.get("schema") not in (BOUSSINESQ_SCHEMA, BOUSSINESQ_SCHEMA_V1):
            raise ValueError("unsupported Boussinesq checkpoint schema")
        if (
            scale_fingerprint is not None
            and metadata.get("scale_fingerprint") != scale_fingerprint
        ):
            raise ValueError("Boussinesq checkpoint scale fingerprint does not match")
        if (
            physics_fingerprint is not None
            and metadata.get("physics_fingerprint") != physics_fingerprint
        ):
            raise ValueError("Boussinesq checkpoint physics fingerprint does not match")
        validation_metadata = dict(metadata)
        validation_metadata["schema"] = SCHEMA
        history_tag = _validate_metadata(
            validation_metadata,
            layout,
            config,
            allow_reshard=allow_single_process_reshard,
        )
        if isinstance(layout, ReferenceCheckpointLayout):
            velocity_loader = _reference_velocity
        else:
            velocity_loader = _distributed_velocity
        velocity = velocity_loader(
            archive,
            "velocity",
            layout,
            tendency=False,
            **(
                {"reshard": allow_single_process_reshard}
                if isinstance(layout, DistributedCheckpointLayout)
                else {}
            ),
        )
        scalar_quantity = metadata.get(
            "scalar_quantity",
            "potential-temperature-perturbation",
        )
        scalar = _checkpoint_scalar(
            archive,
            "scalar",
            layout,
            tendency=False,
            scalar_quantity=scalar_quantity,
            reshard=allow_single_process_reshard,
        )
        closure = _checkpoint_closure(
            archive,
            metadata,
            layout,
            reshard=allow_single_process_reshard,
        )
        if (
            closure_fingerprint is not None
            and getattr(closure, "configuration_fingerprint", None)
            != closure_fingerprint
        ):
            raise ValueError("Boussinesq checkpoint closure fingerprint does not match")
        if history_tag == "cold-start":
            history = ColdStart()
        else:
            history = PreviousTendency(
                BoussinesqTendency(
                    velocity_loader(
                        archive,
                        "history_velocity",
                        layout,
                        tendency=True,
                        **(
                            {"reshard": allow_single_process_reshard}
                            if isinstance(layout, DistributedCheckpointLayout)
                            else {}
                        ),
                    ),
                    _checkpoint_scalar(
                        archive,
                        "history_scalar",
                        layout,
                        tendency=True,
                        scalar_quantity=scalar_quantity,
                        reshard=allow_single_process_reshard,
                    ),
                )
            )
    return AB2BoussinesqState(
        BoussinesqFields(velocity, scalar, closure),
        AcceptedClock(
            float(metadata["clock"]["time"]),
            int(metadata["clock"]["step"]),
        ),
        history,
        config.fingerprint,
    )

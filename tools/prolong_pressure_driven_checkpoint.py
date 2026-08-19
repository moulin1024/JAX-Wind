#!/usr/bin/env python3
"""Prolong a pressure-driven checkpoint and rebuild fine-grid solver memory.

The accepted velocity and passive scalar are linearly interpolated onto an
integer-refined grid.  Horizontal interpolation is periodic, vertical
cell-centred interpolation is clamped at the walls, and vertical velocity is
interpolated at its native faces.  The velocity is then pressure-projected on
the target grid.  AB2 history and LASD memory are deliberately initialized
fresh because neither is a resolved physical field.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from applications.pressure_driven_lasd.config import load_case
from applications.pressure_driven_lasd.evaluate import _configure_source_paths
from applications.pressure_driven_lasd.problem import build_pressure_driven_problem


def _indices_and_weights(
    source_size: int,
    target_size: int,
    *,
    cell_centred: bool,
    periodic: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return source bracketing indices for uniform physical coordinates."""

    if source_size < 2 or target_size < source_size:
        raise ValueError("prolongation requires a larger target axis")
    if target_size % source_size:
        raise ValueError("target axes must be integer refinements of source axes")
    if cell_centred:
        coordinate = (
            (np.arange(target_size, dtype=np.float64) + 0.5)
            * source_size
            / target_size
            - 0.5
        )
    else:
        # Here sizes are cell counts although the returned indices address the
        # corresponding size+1 arrays of boundary-inclusive face values.
        coordinate = (
            np.arange(target_size + 1, dtype=np.float64)
            * source_size
            / target_size
        )
    lower_unbounded = np.floor(coordinate).astype(np.int64)
    weight = coordinate - lower_unbounded
    upper_unbounded = lower_unbounded + 1
    array_size = source_size if cell_centred else source_size + 1
    if periodic:
        lower = lower_unbounded % array_size
        upper = upper_unbounded % array_size
    else:
        lower = np.clip(lower_unbounded, 0, array_size - 1)
        upper = np.clip(upper_unbounded, 0, array_size - 1)
        weight = np.where(lower == upper, 0.0, weight)
    return lower, upper, weight


def _resample_axis(
    values: np.ndarray,
    target_size: int,
    *,
    axis: int,
    cell_centred: bool = True,
    periodic: bool = False,
) -> np.ndarray:
    source_array_size = values.shape[axis]
    source_size = source_array_size if cell_centred else source_array_size - 1
    lower, upper, weight = _indices_and_weights(
        source_size,
        target_size,
        cell_centred=cell_centred,
        periodic=periodic,
    )
    shape = [1] * values.ndim
    shape[axis] = weight.size
    weight = weight.reshape(shape)
    lo = np.take(values, lower, axis=axis)
    hi = np.take(values, upper, axis=axis)
    return lo + (hi - lo) * weight.astype(values.dtype)


def prolong_cell_field(
    values: np.ndarray,
    target_shape: tuple[int, int, int],
) -> np.ndarray:
    """Trilinearly prolong a ``(z, y, x)`` cell-centred field."""

    if values.ndim != 3:
        raise ValueError("cell field must have shape (z, y, x)")
    target_z, target_y, target_x = target_shape
    result = _resample_axis(values, target_x, axis=2, periodic=True)
    result = _resample_axis(result, target_y, axis=1, periodic=True)
    result = _resample_axis(result, target_z, axis=0, periodic=False)
    return result


def prolong_vertical_faces(
    upper_faces: np.ndarray,
    lower_boundary: np.ndarray,
    target_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Prolong staggered vertical velocity and return upper/lower storage."""

    if upper_faces.ndim != 3 or lower_boundary.shape != upper_faces.shape[1:]:
        raise ValueError("vertical-face checkpoint arrays have inconsistent shapes")
    target_z, target_y, target_x = target_shape
    full_faces = np.concatenate((lower_boundary[None, ...], upper_faces), axis=0)
    result = _resample_axis(full_faces, target_x, axis=2, periodic=True)
    result = _resample_axis(result, target_y, axis=1, periodic=True)
    result = _resample_axis(
        result,
        target_z,
        axis=0,
        cell_centred=False,
        periodic=False,
    )
    return result[1:], result[0]


def _global_field(
    archive,
    name: str,
    expected_shape: tuple[int, int, int],
) -> np.ndarray:
    values = np.asarray(archive[name])
    if values.ndim == 4:
        values = values.reshape((-1, values.shape[-2], values.shape[-1]))
    if values.shape != expected_shape:
        raise ValueError(
            f"checkpoint {name} has shape {values.shape}, expected {expected_shape}"
        )
    return values


def prolong_checkpoint(
    source: Path,
    target_config: Path,
    output: Path,
    *,
    overwrite: bool,
) -> dict[str, object]:
    if output.exists() and not overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it")

    case = load_case(target_config)
    _configure_source_paths()
    import jax

    jax.config.update("jax_enable_x64", case.numerics.dtype == "float64")
    import jax.numpy as jnp
    from jaxwind.domain import Accepted, AcceptedClock, PassiveScalarConcentration
    from jaxwind.effects import JaxRuntime, save_boussinesq_checkpoint
    from jaxwind.physics import BoussinesqFields

    runtime = JaxRuntime.from_initialized_jax(jax)
    if runtime.process_count != 1:
        raise RuntimeError(
            "checkpoint prolongation currently requires one JAX process; "
            "the resulting checkpoint may still be used by a distributed run"
        )
    problem = build_pressure_driven_problem(case, runtime=runtime)
    target_shape = (case.domain.nz, case.domain.ny, case.domain.nx)

    with np.load(source, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        source_grid = metadata.get("grid", {})
        source_shape = tuple(
            int(source_grid[name]) for name in ("nz", "ny", "nx")
        )
        if metadata.get("representation") != "owned-z-slab":
            raise ValueError("source must use the owned-z-slab representation")
        if metadata.get("scale_fingerprint") != problem.scales.fingerprint:
            raise ValueError("source and target mechanical scales differ")
        if metadata.get("physics_fingerprint") != problem.physics_fingerprint:
            raise ValueError("source and target physics configurations differ")
        execution_grid = problem.solver.grid
        for name in ("lx", "ly", "lz"):
            if not np.isclose(
                float(source_grid[name]),
                float(getattr(execution_grid, name)),
            ):
                raise ValueError("source and target physical domains differ")
        refinement = tuple(
            target // source
            for target, source in zip(target_shape, source_shape)
        )
        dimensions = zip(target_shape, source_shape, refinement)
        if any(target != factor * source for target, source, factor in dimensions):
            raise ValueError("each target grid dimension must be an integer refinement")
        if any(factor < 2 for factor in refinement):
            raise ValueError("each target grid dimension must be refined")

        u = prolong_cell_field(
            _global_field(archive, "velocity_x", source_shape), target_shape
        )
        v = prolong_cell_field(
            _global_field(archive, "velocity_y", source_shape), target_shape
        )
        scalar_values = prolong_cell_field(
            _global_field(archive, "scalar", source_shape), target_shape
        )
        w, w_lower = prolong_vertical_faces(
            _global_field(archive, "velocity_z", source_shape),
            np.asarray(archive["velocity_z_lower_boundary"]),
            target_shape,
        )

    dtype = getattr(jnp, case.numerics.dtype)
    candidate = problem.solver.candidate_velocity(
        jnp.asarray(u, dtype=dtype),
        jnp.asarray(v, dtype=dtype),
        jnp.asarray(w, dtype=dtype),
        lower_boundary=jnp.asarray(w_lower, dtype=dtype),
    )
    velocity = problem.solver.project_initial_velocity(candidate)
    scalar = problem.solver.cell_field(
        PassiveScalarConcentration,
        Accepted,
        jnp.asarray(scalar_values, dtype=dtype),
    )
    fields = problem.solver.initialize_fields(BoussinesqFields(velocity, scalar))
    state = problem.solver.cold_start(fields, clock=AcceptedClock(0.0, 0))
    state.fields.velocity.x.payload.block_until_ready()

    output.parent.mkdir(parents=True, exist_ok=True)
    save_boussinesq_checkpoint(
        output,
        state,
        scale_fingerprint=problem.scales.fingerprint,
        physics_fingerprint=problem.physics_fingerprint,
    )
    report: dict[str, object] = {
        "source": str(source),
        "target_config": str(target_config),
        "output": str(output),
        "source_shape_zyx": list(source_shape),
        "target_shape_zyx": list(target_shape),
        "refinement_zyx": list(refinement),
        "velocity_transfer": "trilinear-periodic-xy-clamped-z-plus-pressure-projection",
        "vertical_velocity_transfer": "native-face-linear-plus-pressure-projection",
        "integrator_history": "cold-start",
        "lasd_memory": "freshly-initialized-on-target-grid",
    }
    output.with_suffix(output.suffix + ".prolongation.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target_config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = prolong_checkpoint(
        args.source,
        args.target_config,
        args.output,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared host runtime. All numerical advancement is supplied by Simulation."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time

from jaxwind.config.document import ResolvedCase, load_case
from jaxwind.config.toml import dumps
from jaxwind.io.checkpoint import save_checkpoint, load_checkpoint
from jaxwind.simulation.api import AdvanceResult, RunControls, build_simulation


@dataclass(frozen=True)
class RunResult:
    output: Path
    checkpoint: Path
    summary: dict


def _json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(case, *, output=None, max_steps=None, _resume=False, _simulation=None):
    case = load_case(case)
    directory = Path(output or case.output).resolve()
    if max_steps is not None and (type(max_steps) is not int or max_steps <= 0):
        raise ValueError("max_steps must be a positive integer")
    if not _resume and directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"run directory is not empty: {directory}; use resume or a new output")
    simulation = _simulation if _simulation is not None else build_simulation(case)
    import jax
    import numpy as np
    from .observers import Observer

    state = simulation.initialize()
    settings = case.document["time"]
    dt = settings["dt_seconds"]
    steps = settings["steps"]
    chunk = settings.get("chunk_steps", 100)
    observer = Observer(simulation)
    initial_step = int(state.step)
    initial_time = float(state.time)
    metadata = {
        "fingerprint": case.fingerprint,
        "formulation": case.formulation,
        "initial_step": initial_step,
        "initial_time": initial_time,
        "target_step": initial_step + steps,
        "target_time": initial_time + steps * dt,
        "jax_version": jax.__version__,
        "devices": [str(device) for device in jax.devices()],
        "units": "SI",
        "resolved_case": case.document,
        "mesh": {name: np.asarray(getattr(simulation.grid, name)).tolist() for name in ("x_faces", "y_faces", "z_faces")},
    }
    checkpoint = directory / "checkpoint.npz"
    if _resume:
        state, saved_observer, metadata = load_checkpoint(checkpoint, state, fingerprint=case.fingerprint)
        observer.restore(saved_observer)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "resolved_case.toml").write_text(dumps(case.document), encoding="utf-8")
    _json(directory / "run.json", {"schema": "jaxwind.run.v1", "status": "running", "fingerprint": case.fingerprint})
    if not _resume:
        save_checkpoint(checkpoint, state, metadata=metadata, observer=observer.snapshot())
    invocation_start = int(state.step)
    elapsed_blocks = []
    advanced_blocks = []
    tolerance = 8 * np.finfo(np.asarray(state.time).dtype).eps * max(1., metadata["target_time"])

    def finished():
        return float(state.time) >= metadata["target_time"] - tolerance if simulation.adaptive else int(state.step) >= metadata["target_step"]

    try:
        while not finished():
            if max_steps is not None and int(state.step) - invocation_start >= max_steps:
                break
            count = min(chunk, metadata["target_step"] - int(state.step)) if not simulation.adaptive else chunk
            if max_steps is not None:
                count = min(count, max_steps - (int(state.step) - invocation_start))
            count, target = observer.next_block(state, count, metadata)
            before = int(state.step)
            started = time.perf_counter()
            advanced = simulation.advance(state, RunControls(count, target))
            outputs = advanced.outputs if isinstance(advanced, AdvanceResult) else None
            state = advanced.state if isinstance(advanced, AdvanceResult) else advanced
            jax.block_until_ready(state)
            elapsed_blocks.append(time.perf_counter() - started)
            active = int(state.step) - before
            if active <= 0 or not math.isfinite(float(state.time)):
                raise RuntimeError("simulation made no finite forward progress")
            advanced_blocks.append(active)
            if outputs is not None:
                observer.consume(directory, {key: np.asarray(value)[:active] for key, value in outputs.items()})
            observer.sample(state, metadata)
            checkpoint_every = settings.get("checkpoint_every_steps", chunk * 10)
            if int(state.step) - metadata.get("last_checkpoint_step", invocation_start) >= checkpoint_every:
                metadata["last_checkpoint_step"] = int(state.step)
                save_checkpoint(checkpoint, state, metadata=metadata, observer=observer.snapshot())
            timestep = f" dt={float(state.last_dt):.6g}s" if hasattr(state, "last_dt") else ""
            print(f"step={int(state.step)} time={float(state.time):.6g}s CFL={float(simulation.courant(state)):.4g}{timestep}", flush=True)
        complete = finished()
        save_checkpoint(checkpoint, state, metadata=metadata, observer=observer.snapshot())
        observer.write(directory, state)
        steady_seconds = sum(elapsed_blocks[2:])
        summary = {
            **observer.summary(),
            "schema": "jaxwind.run.v1", "status": "complete" if complete else "paused",
            "formulation": case.formulation, "step": int(state.step), "time_seconds": float(state.time),
            "target_step": metadata["target_step"], "target_time_seconds": metadata["target_time"],
            "final_cfl": float(simulation.courant(state)), "elapsed_seconds": sum(elapsed_blocks),
            "steady_steps_per_second": sum(advanced_blocks[2:]) / steady_seconds if steady_seconds else None,
            "checkpoint": str(checkpoint),
        }
        _json(directory / "summary.json", summary)
        _json(directory / "run.json", {**summary, "fingerprint": case.fingerprint})
        return RunResult(directory, checkpoint, summary)
    except BaseException:
        _json(directory / "run.json", {"schema": "jaxwind.run.v1", "status": "interrupted", "fingerprint": case.fingerprint})
        raise


def resume(directory, *, max_steps=None):
    directory = Path(directory).resolve()
    if not (directory / "checkpoint.npz").is_file():
        raise ValueError("run has no complete checkpoint to resume")
    case = load_case(directory / "resolved_case.toml")
    initial = case.document.get("initial_conditions", {})
    simulation = None
    if "operation" in initial:
        from jaxwind.simulation.stages import build_stage
        simulation = build_stage(case, initial["operation"], initial.get("artifacts", {}), initial.get("stage_options", {}))
    return run(case, output=directory, max_steps=max_steps, _resume=True, _simulation=simulation)

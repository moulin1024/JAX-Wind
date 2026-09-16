"""Sequential dependency execution; stages delegate all advancement to runtime."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from jaxwind.config.document import ResolvedCase, load_case, merge, validate, tomllib, _paths


def _recipe(case):
    """Expand the atmospheric convenience recipe into ordinary stage nodes."""
    options = case.document.get("workflow")
    if options is None:
        raise ValueError("case has no workflow recipe; supply a workflow with [stages.NAME] tables")
    dt = case.document["time"]["dt_seconds"]
    base = str(case.source)
    precursor_dt = options.get("precursor_dt_seconds", dt)
    main_dt = options.get("main_dt_seconds", precursor_dt)
    factor = options.get("main_substeps_per_inflow", 1)
    warm_inputs = {}
    if options.get("warmup_restart_checkpoint"):
        warm_inputs["checkpoint"] = options["warmup_restart_checkpoint"]
    input_directory = options.get("input_directory")
    checkpoint_input = str(Path(input_directory) / "warmup/checkpoint.npz") if input_directory else "@warmup/checkpoint"
    inflow_input = str(Path(input_directory) / "precursor/inflow") if input_directory else "@precursor/inflow"
    return {
        "schema_version": 1, "output": {"directory": options["output_directory"]},
        "stages": {
            "warmup": {"case": base, "operation": "periodic", "inputs": warm_inputs,
                       "overrides": {"time": {"steps": options["warmup_steps"], "chunk_steps": options["chunk_steps"]}, "diagnostics": {"sample_start_step": 0}, "numerics": {"pressure_backend": "fft"}}},
            "precursor": {"case": base, "operation": "record-inflow", "inputs": {"checkpoint": checkpoint_input},
                          "fixed_dt": "precursor_dt_seconds" in options,
                          "options": {"record_plane": options["record_plane"]},
                          "overrides": {"time": {"steps": options["precursor_steps"], "dt_seconds": precursor_dt, "chunk_steps": options["chunk_steps"], "frame_count": options.get("precursor_frame_count", 0)}, "diagnostics": {"sample_start_step": 0}, "numerics": {"pressure_backend": "fft"}}},
            "main": {"case": base, "operation": "open-inflow", "inputs": {"checkpoint": checkpoint_input, "inflow": inflow_input},
                     "fixed_dt": True, "options": {"substeps_per_inflow": factor},
                     "overrides": {"time": {"steps": options["main_steps"]*factor, "dt_seconds": main_dt/factor, "chunk_steps": options["chunk_steps"]*factor, "frame_count": options.get("main_frame_count", 0)}, "diagnostics": {"sample_start_step": 0}, "numerics": {"pressure_backend": "gmg"}}},
        },
    }


def load_workflow(path):
    source = Path(path).resolve()
    with source.open("rb") as stream:
        document = tomllib.load(stream)
    if "stages" not in document:
        document = _recipe(load_case(source))
    if document.get("schema_version") != 1 or not isinstance(document.get("stages"), dict) or not document["stages"]:
        raise ValueError("workflow requires schema_version=1 and named stages")
    if document.keys() - {"schema_version", "output", "stages"}:
        raise ValueError("unknown workflow sections")
    if not isinstance(document.get("output", {}).get("directory"), str):
        raise ValueError("workflow requires output.directory")
    for name, node in document["stages"].items():
        unknown = node.keys() - {"case", "operation", "inputs", "overrides", "options", "fixed_dt"}
        if unknown:
            raise ValueError(f"unknown stage settings in {name}: {sorted(unknown)}")
        if node.get("inputs", {}).keys() - {"checkpoint", "inflow"}:
            raise ValueError(f"unknown artifact inputs in stage {name}")
        if not name or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in name):
            raise ValueError("stage names must be simple identifiers")
        if node.get("operation", "simulation") not in {"simulation", "periodic", "record-inflow", "open-inflow"}:
            raise ValueError(f"unknown stage operation: {name}")
        operation = node.get("operation", "simulation")
        allowed_options = {"record_plane"} if operation == "record-inflow" else {"substeps_per_inflow", "lateral_boundary", "target_hub_wind_speed_m_s"} if operation == "open-inflow" else set()
        if node.get("options", {}).keys() - allowed_options:
            raise ValueError(f"unsupported stage options: {name}")
        if operation != "open-inflow" and "inflow" in node.get("inputs", {}):
            raise ValueError(f"stage does not consume inflow: {name}")
        _paths(node.get("overrides", {}), source.parent)
        node["case"] = str((source.parent / node["case"]).resolve())
        for binding, value in node.get("inputs", {}).items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"artifact binding must be a nonempty path/reference: {name}")
            if not value.startswith("@"):
                node["inputs"][binding] = str((source.parent / value).resolve())
    return source, document


def _order(nodes, selected=None):
    order, active = [], set()
    def visit(name):
        if name not in nodes:
            raise ValueError(f"unknown workflow stage: {name}")
        if name in active:
            raise ValueError("workflow dependency cycle")
        if name in order:
            return
        active.add(name)
        for value in nodes[name].get("inputs", {}).values():
            if value.startswith("@"):
                upstream, separator, artifact = value[1:].partition("/")
                if not separator or artifact not in {"checkpoint", "inflow"}:
                    raise ValueError(f"invalid artifact binding: {value}")
                visit(upstream)
        active.remove(name)
        order.append(name)
    for name in ([selected] if selected else nodes):
        visit(name)
    return order


def check_workflow(path):
    """Resolve the dependency graph without requiring future stage artifacts."""
    source, document = load_workflow(path)
    order = _order(document["stages"])
    stages = {}
    for name in order:
        node = document["stages"][name]
        base = load_case(node["case"])
        doc = merge(base.document, node.get("overrides", {}))
        if node.get("fixed_dt"):
            doc["time"].pop("cfl", None)
        validate(doc)
        stages[name] = {"operation": node.get("operation", "simulation"), "formulation": doc["formulation"],
                        "time": doc["time"], "inputs": node.get("inputs", {})}
    return {"schema_version": 1, "workflow": str(source), "order": order, "stages": stages,
            "artifact_validation": "performed at stage construction, after dependencies are produced"}


def execute(path, *, stage=None, resume=False, output=None, max_steps=None):
    from jaxwind.runtime.engine import run, _json
    from jaxwind.simulation.stages import build_stage
    source, document = load_workflow(path)
    nodes = document["stages"]
    order = _order(nodes, stage)
    root = Path(output or document["output"]["directory"]).resolve()
    if root.exists() and any(root.iterdir()) and not resume:
        raise FileExistsError(f"workflow output is not empty: {root}; use --resume")
    manifest_path = root / "workflow.json"
    manifest = json.loads(manifest_path.read_text()) if resume and manifest_path.exists() else {"schema": "jaxwind.workflow.v1", "stages": {}}
    for name in order:
        node = nodes[name]
        base = load_case(node["case"])
        doc = merge(base.document, node.get("overrides", {}))
        if node.get("fixed_dt"):
            doc["time"].pop("cfl", None)
        inputs = {}
        for key, value in node.get("inputs", {}).items():
            if value.startswith("@"):
                upstream, artifact = value[1:].split("/", 1)
                inputs[key] = str(root / upstream / ("checkpoint.npz" if artifact == "checkpoint" else "inflow"))
            else:
                inputs[key] = value
        doc["output"] = {"directory": str(root / name)}
        doc["initial_conditions"] = {**doc.get("initial_conditions", {}), "operation": node.get("operation", "simulation"), "artifacts": inputs, "stage_options": node.get("options", {})}
        validate(doc)
        case = ResolvedCase(base.source, doc)
        previous = manifest["stages"].get(name, {})
        if previous.get("fingerprint") not in (None, case.fingerprint):
            raise ValueError(f"stage configuration changed: {name}; choose a new workflow output")
        if previous.get("status") == "complete":
            if not (root / name / "checkpoint.npz").is_file():
                raise ValueError(f"completed stage artifact is missing: {name}")
            from jaxwind.io.checkpoint import checkpoint_metadata
            if checkpoint_metadata(root / name / "checkpoint.npz").get("fingerprint") != case.fingerprint:
                raise ValueError(f"completed checkpoint does not match stage: {name}")
            if node.get("operation") == "record-inflow":
                from jaxwind.io.recording import InflowReader
                InflowReader(root / name / "inflow")
            continue
        simulation = build_stage(case, node.get("operation", "simulation"), inputs, node.get("options", {}))
        root.mkdir(parents=True, exist_ok=True)
        checkpoint_exists = (root / name / "checkpoint.npz").is_file()
        manifest["stages"][name] = {"status": "running", "fingerprint": case.fingerprint}
        _json(manifest_path, manifest)
        result = run(case, max_steps=max_steps, _resume=resume and checkpoint_exists, _simulation=simulation)
        manifest["stages"][name] = {**result.summary, "fingerprint": case.fingerprint}
        _json(manifest_path, manifest)
        if result.summary["status"] != "complete":
            break
    return manifest

"""Run DTU10MW preparation (10 h + 1 h) or a separate 1 h ALM main."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path

SCHEMA = "jaxwind.dtu10mw-runner.v1"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_subparsers(dest="mode", required=True)
    for mode in ("prepare", "main"):
        sub = modes.add_parser(mode)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--backend", choices=("rocm", "cuda", "cpu"), default="rocm")
        sub.add_argument("--resume", action="store_true")
        sub.add_argument("--configure-only", action="store_true", help="validate declarations without allocating simulation fields")
        sub.add_argument("--smoke", action="store_true", help="small mesh: 4 fixed steps per stage")
        sub.add_argument("--max-steps", type=int, help="pause each active stage after this many additional steps")
        if mode == "main":
            sub.add_argument("--prepare-run", type=Path, required=True)
            sub.add_argument("--openfast-model", type=Path, help="DTU10MW AeroDyn15 .fst deck; otherwise JAXWIND_DTU10MW_FAST")
    return p


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare_binding(path, smoke, require_complete):
    path = path.resolve()
    manifest = read_json(path / "runner.json")
    if manifest.get("schema") != SCHEMA or manifest.get("mode") != "prepare":
        raise ValueError("--prepare-run must identify a DTU10MW prepare-mode output")
    if manifest.get("smoke") != smoke:
        raise ValueError("prepare and main smoke settings must match")
    reference = Path(manifest["reference_output"]).resolve()
    if reference != path / "run/reference":
        raise ValueError("prepare reference artifacts must remain inside the selected run")
    if not require_complete:
        return reference, None
    if manifest.get("status") != "complete":
        raise ValueError("prepare is not complete; finish/resume prepare first")
    workflow = read_json(reference / "workflow.json")
    for stage in ("warmup", "precursor"):
        if workflow.get("stages", {}).get(stage, {}).get("status") != "complete":
            raise ValueError(f"prepare {stage} is not complete")
        if read_json(reference / stage / "run.json").get("status") != "complete":
            raise ValueError(f"prepare {stage} output is not complete")
        if not (reference / stage / "checkpoint.npz").is_file():
            raise ValueError(f"missing prepare {stage} checkpoint")
    metadata = read_json(reference / "precursor/inflow/metadata.json")
    from create_dtu10mw_case import STEPS_PER_HOUR
    if metadata.get("samples") != (4 if smoke else STEPS_PER_HOUR):
        raise ValueError("prepare recording has the wrong sample count")
    for chunk in metadata["chunks"]:
        name = chunk["file"]
        if Path(name).name != name or not (reference / "precursor/inflow" / name).is_file():
            raise ValueError("prepare recording contains a missing or invalid chunk")
    digest = hashlib.sha256()
    for relative in ("workflow.json", "warmup/resolved_case.toml", "precursor/inflow/metadata.json"):
        digest.update(relative.encode())
        digest.update((reference / relative).read_bytes())
    return reference, digest.hexdigest()


def turbine_binding(args, case_path):
    value = args.openfast_model or os.environ.get("JAXWIND_DTU10MW_FAST")
    if not value:
        if args.configure_only:
            return None
        raise ValueError("main requires --openfast-model or JAXWIND_DTU10MW_FAST pointing to a DTU10MW .fst deck")
    path = Path(value).resolve()
    os.environ["JAXWIND_DTU10MW_FAST"] = str(path)
    from jaxwind.config.stages import load_workflow
    from jaxwind.simulation.turbines import build_turbine_definition
    turbine = build_turbine_definition(load_workflow(case_path))
    rotor = turbine.rotor
    from jaxwind.config.document import load_case
    from create_dtu10mw_case import TIP_SWEEP_CFL
    case = load_case(case_path).document
    minimum_width = min(length / cells for length, cells in zip(case["mesh"]["lengths_m"], case["mesh"]["cells"]))
    tip_speed = 2. * math.pi * rotor.tip_radius_m * case["physics"]["turbine"]["rotor_speed_rpm"] / 60.
    actual_sweep_cfl = tip_speed * case["time"]["dt_seconds"] / minimum_width
    if actual_sweep_cfl > TIP_SWEEP_CFL * (1. + 1.e-9):
        raise ValueError("OpenFAST rotor exceeds the configured tip-sweep CFL; regenerate with a smaller timestep")
    sources = (rotor.source, rotor.aerodyn_source, rotor.elastodyn_source,
               rotor.blade_source, *rotor.airfoil_sources)
    digest = hashlib.sha256()
    for source in sources:
        digest.update(str(source).encode())
        digest.update(source.read_bytes())
    return {"path": str(path), "sha256": digest.hexdigest(),
            "tip_sweep_cfl": actual_sweep_cfl, "compatibility_notes": list(rotor.compatibility_notes)}


def preflight(backend):
    import jax
    devices = jax.devices(backend)
    if not devices:
        raise RuntimeError(f"no {backend} devices; refusing CPU fallback")
    print(f"Backend {backend}: {devices}", flush=True)
    return [str(device) for device in devices]


def run(args):
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    os.environ["JAX_PLATFORMS"] = "cpu" if args.configure_only else args.backend
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    from create_dtu10mw_case import generate
    from jaxwind.config.validation import check_case
    from jaxwind.workflows.engine import check_workflow, execute
    output = args.output.resolve()
    if args.mode == "main":
        prepare = args.prepare_run.resolve()
        if output.is_relative_to(prepare) or prepare.is_relative_to(output):
            raise ValueError("main output must be separate from the prepare run")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"another runner owns {output}") from error
        manifest_path = output / "runner.json"
        previous = read_json(manifest_path) if manifest_path.exists() else {}
        identity = {"mode": args.mode, "smoke": args.smoke,
            "prepare_run": str(args.prepare_run.resolve()) if args.mode == "main" else None}
        if previous and (previous.get("schema") != SCHEMA or any(previous.get(k) != v for k, v in identity.items())):
            raise ValueError("output belongs to another mode or prepare run")
        if previous.get("status") not in (None, "configured"):
            if args.configure_only:
                raise ValueError("configure-only cannot overwrite an executed manifest")
            if not args.resume:
                raise ValueError("existing execution requires --resume or a new --output")
        reference, provenance = (None, None) if args.mode == "prepare" else prepare_binding(
            args.prepare_run, args.smoke, require_complete=not args.configure_only)
        if previous.get("prepare_provenance") not in (None, provenance):
            raise ValueError("prepare recording changed since this main was executed")
        workflow, reference = generate(args.mode, output, smoke=args.smoke, reference=reference)
        plan = check_workflow(workflow)
        check_case(workflow.parent / "case.toml")
        expected = ["warmup", "precursor"] if args.mode == "prepare" else ["main"]
        if plan["order"] != expected:
            raise ValueError("runner mode and workflow stages disagree")
        turbine = turbine_binding(args, workflow.parent / "case.toml") if args.mode == "main" else None
        if previous.get("turbine") not in (None, turbine):
            raise ValueError("OpenFAST input deck changed; choose a new main output")
        manifest = {**previous, **identity, "schema": SCHEMA, "workflow": str(workflow),
            "reference_output": str(reference), "prepare_provenance": provenance,
            "turbine": turbine, "backend": args.backend, "status": "configured",
            "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
        if args.configure_only:
            save_json(manifest_path, manifest)
            print(json.dumps(plan, indent=2))
            return "configured"
        manifest["devices"] = preflight(args.backend)
        manifest["status"] = "running"
        save_json(manifest_path, manifest)
        try:
            result = execute(workflow, resume=args.resume, max_steps=args.max_steps)
            manifest["status"] = "complete" if all(result["stages"].get(name, {}).get("status") == "complete" for name in expected) else "paused"
            save_json(manifest_path, manifest)
            print(json.dumps(manifest, indent=2))
            return manifest["status"]
        except BaseException:
            manifest["status"] = "interrupted"
            save_json(manifest_path, manifest)
            raise


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        status = run(args)
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    return 0 if status in {"complete", "configured"} else 3


if __name__ == "__main__":
    raise SystemExit(main())

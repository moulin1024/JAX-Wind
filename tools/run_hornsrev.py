"""Run prepare, one direction, or all wind-rose directions with explicit provenance."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_subparsers(dest="mode", required=True)
    for mode in ("prepare", "direction", "windrose"):
        sub = modes.add_parser(mode)
        sub.add_argument("--output", type=Path, required=True, help="unique runner output root")
        sub.add_argument("--backend", choices=("rocm", "cuda", "cpu"), default="rocm")
        sub.add_argument("--resume", action="store_true")
        sub.add_argument("--configure-only", action="store_true", help="write/validate declarations without running")
        sub.add_argument("--smoke", action="store_true", help="4 steps per stage, one turbine; must match prepare")
        sub.add_argument("--max-steps", type=int, help="pause each active stage after this many steps")
        if mode != "prepare":
            sub.add_argument("--prepare-run", type=Path, required=True, help="output root of a completed prepare-mode run")
            sub.add_argument("--no-render", action="store_true", help="skip main MP4 rendering")
        if mode == "direction":
            sub.add_argument("--wind-direction", type=float, required=True)
            sub.add_argument("--wind-speed", type=float)
        if mode == "windrose":
            sub.add_argument("--sector-index", type=int, help="zero-based Slurm array index; absent runs all sequentially")
    return p


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, data):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare_binding(path, smoke, require_complete):
    path = path.resolve()
    manifest = read_json(path / "runner.json")
    if manifest.get("schema") != "jaxwind.hornsrev-runner.v1" or manifest.get("mode") != "prepare":
        raise ValueError("--prepare-run must identify a prepare-mode output, not a direction run")
    if bool(manifest.get("smoke")) != smoke:
        raise ValueError("prepare and main smoke/production settings must match explicitly")
    reference = Path(manifest["reference_output"])
    if not reference.is_relative_to(path):
        raise ValueError("prepare reference artifacts must remain inside the selected prepare run")
    if require_complete:
        workflow = read_json(reference / "workflow.json")
        for stage in ("warmup", "precursor"):
            if workflow.get("stages", {}).get(stage, {}).get("status") != "complete":
                raise ValueError(f"prepare {stage} is not complete; finish/resume prepare first")
            if read_json(reference / stage / "run.json").get("status") != "complete":
                raise ValueError(f"prepare {stage} output is not complete")
            if not (reference / stage / "checkpoint.npz").is_file():
                raise ValueError(f"missing prepare {stage} checkpoint")
        metadata = read_json(reference / "precursor/inflow/metadata.json")
        if "mean_x_velocity_profile_m_s" not in metadata:
            raise ValueError("prepare recording lacks the velocity statistics needed for scaling")
        for chunk in metadata["chunks"]:
            if Path(chunk["file"]).name != chunk["file"] or not (reference / "precursor/inflow" / chunk["file"]).is_file():
                raise ValueError("prepare inflow contains an invalid or missing chunk")
    # Pin the exact completed recording/initialization metadata, without hashing
    # tens of GB of arrays. Keep shared artifacts immutable during all mains.
    digest = hashlib.sha256()
    for relative in ("workflow.json", "warmup/resolved_case.toml", "precursor/inflow/metadata.json"):
        artifact = reference / relative
        if artifact.is_file():
            digest.update(relative.encode())
            digest.update(artifact.read_bytes())
    return reference, digest.hexdigest() if require_complete else None


def preflight(backend):
    os.environ["JAX_PLATFORMS"] = backend
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAXWIND_V80_FAST", str(ROOT / "cases/HornsRev1/turbines/V80/CustomRotor.fst"))
    import jax
    devices = jax.devices(backend)
    if not devices:
        raise RuntimeError(f"no {backend} devices; refusing to fall back to CPU")
    print(f"Backend {backend}: {devices}", flush=True)
    return [str(d) for d in devices]


def run_one(args):
    # Set the backend before importing any JAX-Wind modules. Configure-only
    # uses CPU metadata construction and never allocates production fields.
    os.environ["JAX_PLATFORMS"] = "cpu" if args.configure_only else args.backend
    os.environ.setdefault("JAXWIND_V80_FAST", str(ROOT / "cases/HornsRev1/turbines/V80/CustomRotor.fst"))
    from create_hornsrev_case import generate
    from jaxwind.workflows.engine import check_workflow, execute
    output = args.output.resolve()
    if args.mode != "prepare" and (output == args.prepare_run.resolve() or output.is_relative_to(args.prepare_run.resolve())):
        raise ValueError("main output must be separate from the prepare run")
    output.mkdir(parents=True, exist_ok=True)
    # Prevent two writers to the same prepare/direction output, including resume.
    with (output / ".runner.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"another runner owns {output}") from error
        manifest_path = output / "runner.json"
        previous = read_json(manifest_path) if manifest_path.exists() else {}
        identity = {"mode": args.mode, "smoke": args.smoke,
                    "prepare_run": str(args.prepare_run.resolve()) if args.mode != "prepare" else None,
                    "wind_direction": args.wind_direction % 360. if args.mode != "prepare" else None,
                    "wind_speed": args.wind_speed if args.mode != "prepare" else None}
        if previous and any(previous.get(k) != v for k, v in identity.items()):
            raise ValueError("output belongs to another mode, direction, or prepare run")
        if previous.get("status") not in (None, "configured") and not args.resume:
            raise ValueError("existing execution requires --resume or a new --output")
        if args.configure_only and previous.get("status") not in (None, "configured"):
            raise ValueError("configure-only cannot overwrite an executed runner manifest")
        reference, provenance = (None, None) if args.mode == "prepare" else prepare_binding(
            args.prepare_run, args.smoke, require_complete=not args.configure_only)
        if previous.get("prepare_provenance") not in (None, provenance) and not args.configure_only:
            raise ValueError("prepare recording changed since the main run was configured/executed")
        data = generate(270. if args.mode == "prepare" else args.wind_direction,
            None if args.mode == "prepare" else args.wind_speed,
            output / "cases", output / "run", args.smoke, reference_run=reference)
        workflow = Path(data["reference_workflow"] if args.mode == "prepare" else data["main_workflow"])
        plan = check_workflow(workflow)
        expected = ["warmup", "precursor"] if args.mode == "prepare" else ["main"]
        if plan["order"] != expected:
            raise ValueError("runner mode and workflow stages disagree")
        reference_output = output / "run" / ("reference_smoke" if args.smoke else "reference")
        manifest = {**previous, **identity, "schema": "jaxwind.hornsrev-runner.v1",
            "workflow": str(workflow), "reference_output": str(reference_output if reference is None else reference),
            "prepare_provenance": provenance, "backend": args.backend,
            "target_hub_speed_m_s": data["target_mean_hub_speed_m_s"],
            "status": "configured", "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
        if args.configure_only:
            save_json(manifest_path, manifest)
            print(json.dumps(plan, indent=2))
            return "configured"
        manifest["devices"] = preflight(args.backend)
        manifest["status"] = "running"
        save_json(manifest_path, manifest)
        try:
            result = execute(workflow, resume=args.resume, max_steps=args.max_steps)
            complete = all(result["stages"].get(name, {}).get("status") == "complete" for name in expected)
            manifest["status"] = "complete" if complete else "paused"
            save_json(manifest_path, manifest)
            if complete and args.mode != "prepare" and not args.no_render:
                from jaxwind.config.document import load_case
                case = load_case(workflow.parent / "main.toml")
                subprocess.run([sys.executable, str(ROOT / "tools/render_hornsrev1_farm.py"), str(case.output)], check=True)
            print(json.dumps(manifest, indent=2))
            return manifest["status"]
        except BaseException:
            manifest["status"] = "interrupted"
            save_json(manifest_path, manifest)
            raise


def run(args):
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.mode != "windrose":
        return run_one(args)
    with (ROOT / "cases/HornsRev1/wind_rose.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    if args.sector_index is not None:
        if not 0 <= args.sector_index < len(rows):
            raise ValueError("--sector-index is outside the wind rose")
        rows = [rows[args.sector_index]]
    # Independent subdirectories allow Slurm array tasks to run without sharing
    # mutable case/runner metadata. No prepare execution occurs in this mode.
    for row in rows:
        direction = float(row["wind_direction_deg"])
        child = argparse.Namespace(**vars(args))
        child.mode, child.wind_direction, child.wind_speed = "direction", direction, None
        child.output = args.output / f"wd{direction:07.3f}".replace(".", "p")
        status = run_one(child)
        if status == "paused":
            return status
    return "configured" if args.configure_only else "complete"


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        status = run(args)
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    # afterok dependencies must not treat a deliberately paused prepare as done.
    return 0 if status in ("complete", "configured") else 3


if __name__ == "__main__":
    raise SystemExit(main())

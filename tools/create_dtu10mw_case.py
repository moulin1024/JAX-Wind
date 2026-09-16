"""Generate separate DTU10MW ALM preparation and main workflows."""
from copy import deepcopy
import math
from pathlib import Path

from jaxwind.config.document import load_case
from jaxwind.config.toml import dumps
from jaxwind.windfarm.actuator_disk import DTU_10MW_ROTOR_DIAMETER_M

ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = ROOT / "cases/DTU10MWPrecursor"
TIP_SWEEP_CFL = .5
ROTOR_SPEED_RPM = 9.6
MIN_CELL_WIDTH_M = 2.
TIP_SPEED_M_S = math.pi * DTU_10MW_ROTOR_DIAMETER_M * ROTOR_SPEED_RPM / 60.
STEPS_PER_HOUR = math.ceil(3600. * TIP_SPEED_M_S / (TIP_SWEEP_CFL * MIN_CELL_WIDTH_M))
# Round down to a timestep that divides one hour exactly.
FIXED_DT = 3600. / STEPS_PER_HOUR


def write_once(path, text):
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f"different configuration already exists: {path}; choose a new --output")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(text)


def profile_text(nz, height):
    text = "z_m,u_m_s,v_m_s,w_upper_m_s,scalar,u_rms_m_s,v_rms_m_s,w_upper_rms_m_s,scalar_rms\n"
    for i in range(nz):
        z = (i + .5) * height / nz
        rms = .1 * math.sin(math.pi * z / height)**.25
        wrms = .1 * math.sin(math.pi * (i + 1) / nz) if i + 1 < nz else 0.
        text += ",".join(map(str, (z, math.log(z / .001), 0, 0, 0, rms, rms, wrms, 0))) + "\n"
    return text


def generate(mode, output, *, smoke=False, reference=None):
    if mode not in {"prepare", "main"}:
        raise ValueError("expected prepare or main")
    if mode == "main" and reference is None:
        raise ValueError("main requires a prepare reference")
    output = Path(output).resolve()
    directory, run_root = output / "cases", output / "run"
    reference = run_root / "reference" if mode == "prepare" else Path(reference).resolve()
    case = deepcopy(load_case(CASE_ROOT / "fv_alm_smoke_512x256x512.toml").document)
    case["case"].update(name="dtu10mw_alm_" + mode + ("_smoke" if smoke else ""),
        initial_profile=str(directory / "initial_profile.csv"))
    if smoke:
        case["mesh"] = {"cells": [32, 24, 64], "lengths_m": [512., 384., 256.]}
        case["physics"]["turbine"].update(x_m=256., y_m=192.)
    warm_steps, precursor_steps = (4, 4) if smoke else (10 * STEPS_PER_HOUR, STEPS_PER_HOUR)
    main_steps = precursor_steps
    frame_count = 2 if smoke else 100
    case["time"] = {"dt_seconds": FIXED_DT, "steps": warm_steps,
        "chunk_steps": 2 if smoke else 100, "frame_count": frame_count,
        "checkpoint_every_steps": 2 if smoke else STEPS_PER_HOUR}
    if not smoke:
        case["time"]["checkpoint_every_seconds"] = 3600.
    case["diagnostics"].update(sample_start_step=0, sample_every_steps=2 if smoke else round(30. / FIXED_DT))
    case["numerics"]["pressure_backend"] = "fft"
    case["output"] = {"directory": str(reference / "warmup")}
    if mode == "prepare":
        case["physics"].pop("turbine")
        case.pop("workflow", None)
        stages = {
            "warmup": {"case": "case.toml", "operation": "periodic", "fixed_dt": True},
            "precursor": {"case": "case.toml", "operation": "record-inflow", "fixed_dt": True,
                "inputs": {"checkpoint": "@warmup/checkpoint"}, "options": {"record_plane": 0},
                "overrides": {"time": {"dt_seconds": FIXED_DT, "steps": precursor_steps,
                    "checkpoint_every_steps": 2 if smoke else STEPS_PER_HOUR},
                    "diagnostics": {"sample_every_steps": 2 if smoke else 100}}}}
        workflow_output = reference
    else:
        case["time"].update(dt_seconds=FIXED_DT, steps=main_steps,
            chunk_steps=2 if smoke else 100,
            checkpoint_every_steps=2 if smoke else STEPS_PER_HOUR)
        case["diagnostics"]["sample_every_steps"] = 2 if smoke else round(30. / FIXED_DT)
        case["numerics"]["pressure_backend"] = "gmg"
        case["output"] = {"directory": str(run_root / "main")}
        # The single-turbine open adapter consumes these physical options.
        case["workflow"].update(warmup_steps=warm_steps, precursor_steps=precursor_steps,
            main_steps=precursor_steps, precursor_dt_seconds=FIXED_DT,
            main_dt_seconds=FIXED_DT, main_substeps_per_inflow=1,
            precursor_frame_count=frame_count, main_frame_count=frame_count,
            chunk_steps=2 if smoke else 100, input_directory=str(reference),
            output_directory=str(run_root), main_pressure_force=False, evolve_scalar=False)
        stages = {"main": {"case": "case.toml", "operation": "open-inflow", "fixed_dt": True,
            "inputs": {"checkpoint": str(reference / "warmup/checkpoint.npz"),
                       "inflow": str(reference / "precursor/inflow")},
            "options": {"substeps_per_inflow": 1}}}
        workflow_output = run_root
    nz, height = case["mesh"]["cells"][2], case["mesh"]["lengths_m"][2]
    write_once(directory / "initial_profile.csv", profile_text(nz, height))
    write_once(directory / "case.toml", dumps(case))
    workflow = directory / "workflow.toml"
    write_once(workflow, dumps({"schema_version": 1,
        "output": {"directory": str(workflow_output)}, "stages": stages}))
    return workflow, reference

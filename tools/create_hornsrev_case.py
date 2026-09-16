"""Generate directional Horns Rev main cases sharing ONE reference precursor.

Meteorological directions are clockwise from north and denote wind FROM.
Only the turbine layout is rotated; the numerical wind always travels in +x.
"""
import argparse
import csv
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np

from jaxwind.config.document import load_case
from jaxwind.config.toml import dumps

ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = ROOT / "cases/HornsRev1"


def rotate_layout(layout, direction, lengths):
    """Geographic East/North -> right-handed downstream/cross-stream frame."""
    angle = math.radians(270. - direction % 360.)
    c, s = math.cos(angle), math.sin(angle)
    c, s = (0. if abs(v) < 1.e-14 else v for v in (c, s))
    xy = np.array([(r["x_m"], r["y_m"]) for r in layout])
    center = np.asarray(lengths[:2]) / 2.
    shifted = xy - center
    rotated = shifted @ np.array([[c, -s], [s, c]]) + center
    result = deepcopy(layout)
    for row, position in zip(result, rotated):
        row.update(x_m=float(position[0]), y_m=float(position[1]))
    # Need room for the rotor, 1D probe and smoothed force support.
    if (rotated < 176.).any() or (rotated > np.asarray(lengths[:2]) - 176.).any():
        raise ValueError("rotated farm has insufficient boundary clearance")
    return result


def profile_text(nz, height, speed, hub=70., roughness=.0002):
    dz = height / nz
    text = "z_m,u_m_s,v_m_s,w_upper_m_s,scalar,u_rms_m_s,v_rms_m_s,w_upper_rms_m_s,scalar_rms\n"
    for i in range(nz):
        z = (i + .5) * dz
        u = speed * math.log(z / roughness) / math.log(hub / roughness)
        # Small seeding perturbations, NOT prescribed measured turbulence.
        w_rms = .1 * math.sin(math.pi * (i + 1) / nz) if i + 1 < nz else 0.
        text += ",".join(map(str, (z, u, 0, 0, 0, .1, .1, w_rms, 0))) + "\n"
    return text


def write_once(path, text):
    """Reuse identical assets; refuse to silently replace a different case."""
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f"different configuration already exists: {path}; choose another --directory")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(text)


def generate(direction=270., wind_speed=None, directory=None, run_root=None, smoke=False, *, reference_run=None):
    if not math.isfinite(direction):
        raise ValueError("wind direction must be finite")
    direction %= 360.
    directory = Path(directory or CASE_ROOT / "directions").resolve()
    run_root = Path(run_root or ROOT / "outputs/hornsrev1_directional").resolve()
    with (CASE_ROOT / "wind_rose.csv").open() as stream:
        rose = list(csv.DictReader(stream))
    sector = min(rose, key=lambda r: abs((float(r["wind_direction_deg"]) - direction + 180.) % 360. - 180.))
    target = float(sector["mean_wind_speed_m_s"]) if wind_speed is None else wind_speed
    if not math.isfinite(target) or target <= 0.:
        raise ValueError("wind speed must be finite and positive")
    reference_speed = float(next(r for r in rose if float(r["wind_direction_deg"]) == 270.)["mean_wind_speed_m_s"])
    suffix = "_smoke" if smoke else ""
    reference_dir = directory / ("reference" + suffix)
    reference_output = Path(reference_run).resolve() if reference_run is not None else run_root / ("reference" + suffix)
    label = "wd" + f"{direction:07.3f}".replace(".", "p") + suffix
    if wind_speed is not None:
        label += "_ws" + f"{target:g}".replace(".", "p")
    main_dir, main_output = directory / label, run_root / label

    ref = deepcopy(load_case(CASE_ROOT / "fv_000deg_precursor.toml").document)
    ref["case"].update(name="hornsrev1_shared_reference" + suffix,
        citation="Shared neutral reference: supplied 270-degree sector mean assumed at 70 m",
        initial_profile="initial_profile.csv")
    if smoke:
        ref["mesh"] = {"cells": [32, 24, 64], "lengths_m": [512., 384., 256.]}
    nz, height = ref["mesh"]["cells"][2], ref["mesh"]["lengths_m"][2]
    ustar = .4 * reference_speed / math.log(70. / .0002)
    ref["physics"]["flow"]["pressure_acceleration_m_s2"] = [ustar**2 / height, 0.]
    ref["diagnostics"].update(reference_velocity_m_s=reference_speed,
        reference_length_m=height, inversion_search_max_height_m=height,
        sample_every_steps=2 if smoke else 5)
    warm_steps, precursor_steps, main_steps = (4, 4, 4) if smoke else (6000, 14400, 14400)
    ref["time"].update(dt_seconds=.25 if smoke else 6., cfl=.9, steps=warm_steps, chunk_steps=2 if smoke else 120,
        checkpoint_every_steps=2 if smoke else 600, frame_count=2 if smoke else 100)
    if not smoke:
        ref["time"]["checkpoint_every_seconds"] = 3600.
    ref["output"]["directory"] = str(reference_output / "warmup")
    ref_workflow = {"schema_version": 1, "output": {"directory": str(reference_output)},
        "stages": {
            "warmup": {"case": "precursor.toml", "operation": "periodic"},
            "precursor": {"case": "precursor.toml", "operation": "record-inflow", "fixed_dt": True,
                "inputs": {"checkpoint": "@warmup/checkpoint"},
                "options": {"record_plane": 0},
                "overrides": {"time": {"steps": precursor_steps, "dt_seconds": .25,
                    "checkpoint_every_steps": 2 if smoke else 14400},
                    "diagnostics": {"sample_every_steps": 2 if smoke else 120}}}}}
    main = deepcopy(load_case(CASE_ROOT / "fv_hornsrev1_80_uniform10_open_gmg_300s_muscl.toml").document)
    main["physics"].pop("inflow")
    main["case"].update(name="hornsrev1_" + label,
        citation=f"Meteorological wind FROM {direction:g} deg; shared amplitude-scaled precursor",
        initial_profile=str(reference_dir / "initial_profile.csv"))
    main["mesh"] = deepcopy(ref["mesh"])
    main["diagnostics"] = deepcopy(ref["diagnostics"])
    main["diagnostics"]["reference_velocity_m_s"] = target
    main["time"] = {**ref["time"], "steps": main_steps, "dt_seconds": .25,
        "checkpoint_every_steps": 2 if smoke else 14400}
    main["time"].pop("cfl", None)
    main["diagnostics"]["sample_every_steps"] = 2 if smoke else 120
    layout = main["physics"]["wind_farm"]["layout"]
    if smoke:
        layout = [dict(id="T01", x_m=256., y_m=192., hub_height_m=70., initial_rpm=0.)]
        # Smoke is an adapter/control test, not a geometrically scaled farm.
        main["physics"]["turbine"].update(x_m=256., y_m=192.)
    else:
        layout = rotate_layout(layout, direction, main["mesh"]["lengths_m"])
    main["physics"]["wind_farm"]["layout"] = layout
    inputs = {"checkpoint": str(reference_output / "warmup/checkpoint.npz"),
              "inflow": str(reference_output / "precursor/inflow")}
    options = {"lateral_boundary": "outflow", "target_hub_wind_speed_m_s": target}
    main["initial_conditions"] = {"operation": "open-inflow", "artifacts": inputs, "stage_options": options}
    main["output"] = {"directory": str(main_output / "main")}
    workflow = {"schema_version": 1, "output": {"directory": str(main_output)}, "stages": {
        "main": {"case": "main.toml", "operation": "open-inflow", "inputs": inputs, "options": options}}}
    metadata = {"wind_direction_from_deg": direction, "selected_wind_rose_sector_deg": float(sector["wind_direction_deg"]),
        "target_mean_hub_speed_m_s": target, "reference_nominal_hub_speed_m_s": reference_speed,
        "hub_height_m": 70., "roughness_length_m": .0002,
        "reference_friction_velocity_m_s": ustar,
        "scaling": "all velocity components; actual recorded mean hub speed; unchanged timestamps",
        "axes": "x downstream; y cross-stream; z up; geographic East/North layout rotated about its centre",
        "frequency_sum_percent_as_supplied": sum(float(r["frequency_percent"]) for r in rose),
        "reference_workflow": str(reference_dir / "workflow.toml"),
        "main_workflow": str(main_dir / "workflow.toml"), "smoke": smoke}
    write_once(reference_dir / "initial_profile.csv", profile_text(nz, height, reference_speed))
    write_once(reference_dir / "precursor.toml", dumps(ref))
    write_once(reference_dir / "workflow.toml", dumps(ref_workflow))
    write_once(main_dir / "main.toml", dumps(main))
    write_once(main_dir / "workflow.toml", dumps(workflow))
    write_once(main_dir / "direction.json", json.dumps(metadata, indent=2) + "\n")
    # Target log-law mean for documentation; main actually starts from scaled
    # developed reference fields, not from this idealized profile.
    write_once(main_dir / "target_log_profile.csv", profile_text(nz, height, target))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wind-direction", type=float, default=270., help="meteorological FROM degrees; default 270")
    parser.add_argument("--wind-speed", type=float, help="override mean wind at 70 m; default nearest wind-rose sector mean")
    parser.add_argument("--directory", type=Path, help="generated case root (also contains shared reference)")
    parser.add_argument("--run-root", type=Path, help="shared reference and directional result root")
    parser.add_argument("--smoke", action="store_true", help="4 steps/stage, one turbine, small grid; no production run")
    args = parser.parse_args()
    result = generate(args.wind_direction, args.wind_speed, args.directory, args.run_root, args.smoke)
    print(json.dumps(result, indent=2))
    print("Run reference ONCE: python -m jaxwind workflow", result["reference_workflow"])
    print("Then run direction: python -m jaxwind workflow", result["main_workflow"])


if __name__ == "__main__":
    main()

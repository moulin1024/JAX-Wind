#!/usr/bin/env python3
import argparse
import math
import re
from pathlib import Path
from string import Template


def parse_config(path):
    config = {}
    for raw_line in Path(path).read_text().splitlines():
        line = raw_line.split("#", 1)[0]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        config[key.strip()] = value.strip()
    return config


def as_int(config, key):
    config[key] = int(config[key])


def as_float(config, key):
    config[key] = float(config[key])


def process_config(config):
    for key in ("sim_flag", "resub_flag", "double_flag", "dom_flag"):
        as_int(config, key)

    for key in ("nx", "ny", "nz"):
        as_int(config, key)

    if config["dom_flag"] == 0:
        as_float(config, "z_i")
        as_float(config, "l_z")
        as_int(config, "l_r")
    elif config["dom_flag"] == 1:
        as_float(config, "lx")
        as_float(config, "ly")
        as_float(config, "lz")
        config["z_i"] = config["lx"] / (2.0 * math.pi)
        config["l_z"] = config["lz"]
        config["l_r"] = int(config["lx"] / config["ly"])
    else:
        raise ValueError(f"Unsupported dom_flag: {config['dom_flag']}")

    config["lx"] = config["z_i"] * 2.0 * math.pi
    config["ly"] = config["z_i"] * 2.0 * math.pi / config["l_r"]
    config["lz"] = config["l_z"]
    config["dx"] = config["lx"] / config["nx"]
    config["dy"] = config["ly"] / config["ny"]
    config["dz"] = config["lz"] / (config["nz"] - 1)

    as_int(config, "time_flag")
    as_int(config, "nsteps")
    if config["time_flag"] == 0:
        as_float(config, "dt")
    elif config["time_flag"] == 1:
        as_float(config, "dtr")
        config["dt"] = config["dtr"] / config["z_i"]
    else:
        raise ValueError(f"Unsupported time_flag: {config['time_flag']}")
    config["dtr"] = config["dt"] * config["z_i"]
    config["t_tot"] = config["nsteps"] * config["dtr"]

    for key in ("zo", "u_fric", "bl_height", "fgr", "tfr"):
        as_float(config, key)
    for key in ("model", "cs_count", "turb_flag", "turb_nb", "turb_count"):
        as_int(config, key)

    for key in ("turb_r", "tow_r", "tow_c", "nac_r", "nac_c"):
        as_float(config, key)

    for key in ("inflow_istart", "inflow_iend", "inflow_count"):
        as_int(config, key)
    config["inflow_nx"] = config["inflow_iend"] - config["inflow_istart"] + 1

    for key in ("log_flag", "c_count", "p_count", "ta_flag", "ta_mask"):
        as_int(config, key)
    if config["ta_mask"] == 1:
        for key in ("ta_istart", "ta_iend", "ta_jstart", "ta_jend", "ta_kend", "ta_tstart"):
            as_int(config, key)
    else:
        config["ta_istart"] = 1
        config["ta_iend"] = config["nx"]
        config["ta_jstart"] = 1
        config["ta_jend"] = config["ny"]
        config["ta_kend"] = config["nz"]
        as_int(config, "ta_tstart")
    config["ta_nx"] = config["ta_iend"] - config["ta_istart"] + 1
    config["ta_ny"] = config["ta_jend"] - config["ta_jstart"] + 1
    config["ta_ns"] = int((config["nsteps"] - config["ta_tstart"] + 1) / config["p_count"])

    for key in ("ts_flag", "ts_mask"):
        as_int(config, key)
    if config["ts_mask"] == 1:
        for key in ("ts_istart", "ts_iend", "ts_jstart", "ts_jend", "ts_kend", "ts_tstart"):
            as_int(config, key)
    else:
        config["ts_istart"] = 1
        config["ts_iend"] = config["nx"]
        config["ts_jstart"] = 1
        config["ts_jend"] = config["ny"]
        config["ts_kend"] = config["nz"]
        as_int(config, "ts_tstart")
    config["ts_nx"] = config["ts_iend"] - config["ts_istart"] + 1
    config["ts_ny"] = config["ts_jend"] - config["ts_jstart"] + 1
    config["ts_ns"] = int((config["nsteps"] - config["ts_tstart"] + 1) / (config["c_count"] * 10))

    as_int(config, "job_np")
    config["nzb"] = config["nz"] // config["job_np"]
    config["nz2"] = config["nzb"] + 2

    return {key: str(value) for key, value in config.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    values = process_config(parse_config(args.config))
    rendered = Template(Path(args.template).read_text()).safe_substitute(values)
    active_source = "\n".join(line.split("!", 1)[0] for line in rendered.splitlines())
    unresolved = sorted(set(re.findall(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?", active_source)))
    if unresolved:
        names = ", ".join(unresolved)
        raise SystemExit(f"Unresolved dimen.cuf template variable(s): {names}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)


if __name__ == "__main__":
    main()

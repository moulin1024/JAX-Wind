#!/usr/bin/env python
import os
import shutil
import subprocess
import sys
import csv

import numpy as np

from fctlib import get_case_path, get_config
from app_pre import compute_vel, compute_zo


def _ensure_relative_symlink(path, target):
    if os.path.lexists(path):
        return
    os.symlink(target, path)


def _validate_finite(name, values):
    if np.all(np.isfinite(values)):
        return
    print('ERROR: generated {} contains NaN or Inf values.'.format(name))
    sys.exit(1)


def _print_range(name, values):
    print('{}: min={:.6g}, max={:.6g}'.format(name, float(np.min(values)), float(np.max(values))))


def _validate_turbines(runtime_input, config):
    if int(config['turb_flag']) <= 0:
        return

    path = os.path.join(runtime_input, 'turb_loc.dat')
    if not os.path.isfile(path):
        print('ERROR: turb_flag > 0 but input/turb_loc.dat is missing.')
        sys.exit(1)

    with open(path, newline='') as handle:
        rows = list(csv.reader(handle))

    data_rows = [row for row in rows[1:] if row]
    if len(data_rows) < int(config['turb_nb']):
        print('ERROR: turb_nb = {}, but input/turb_loc.dat has only {} turbine row(s).'.format(
            config['turb_nb'], len(data_rows)))
        sys.exit(1)

    margin_x = 16 * config['dx']
    margin_y = 32 * config['dy']
    for index, row in enumerate(data_rows[:int(config['turb_nb'])], start=1):
        if len(row) < 5:
            print('ERROR: input/turb_loc.dat row {} must contain x,y,z,yaw,tilt.'.format(index))
            sys.exit(1)

        try:
            x, y, z = (float(row[0]), float(row[1]), float(row[2]))
        except ValueError:
            print('ERROR: input/turb_loc.dat row {} contains a non-numeric coordinate.'.format(index))
            sys.exit(1)

        if not np.all(np.isfinite([x, y, z])):
            print('ERROR: input/turb_loc.dat row {} contains NaN or Inf.'.format(index))
            sys.exit(1)
        if x < margin_x or x > config['lx'] - margin_x:
            print('ERROR: turbine {} x={} is too close to/outside the x boundary.'.format(index, x))
            print('Required for local run: {:.6g} <= x <= {:.6g}'.format(margin_x, config['lx'] - margin_x))
            sys.exit(1)
        if y < margin_y or y > config['ly'] - margin_y:
            print('ERROR: turbine {} y={} is too close to/outside the y boundary.'.format(index, y))
            print('Required for local run: {:.6g} <= y <= {:.6g}'.format(margin_y, config['ly'] - margin_y))
            sys.exit(1)
        if z <= float(config['turb_r']) or z >= config['lz'] - float(config['turb_r']):
            print('ERROR: turbine {} z={} leaves the rotor too close to/outside the z boundary.'.format(index, z))
            print('Required for local run: {:.6g} < z < {:.6g}'.format(float(config['turb_r']), config['lz'] - float(config['turb_r'])))
            sys.exit(1)


def _prepare_input(case_path, runtime_path, config):
    source_input = os.path.join(case_path, 'input')
    runtime_input = os.path.join(runtime_path, 'input')

    if os.path.islink(runtime_input):
        os.unlink(runtime_input)
    os.makedirs(runtime_input, exist_ok=True)

    for name in os.listdir(source_input):
        source = os.path.join(source_input, name)
        destination = os.path.join(runtime_input, name)
        if os.path.isfile(source):
            shutil.copy2(source, destination)

    if config['sim_flag'] == 0 and config['resub_flag'] == 0:
        u_init, v_init, w_init = compute_vel(config)
        _validate_finite('u.bin', u_init)
        _validate_finite('v.bin', v_init)
        _validate_finite('w.bin', w_init)
        _print_range('u.bin', u_init)
        _print_range('v.bin', v_init)
        _print_range('w.bin', w_init)
        u_init.tofile(os.path.join(runtime_input, 'u.bin'))
        v_init.tofile(os.path.join(runtime_input, 'v.bin'))
        w_init.tofile(os.path.join(runtime_input, 'w.bin'))
    elif not all(os.path.isfile(os.path.join(runtime_input, name)) for name in ('u.bin', 'v.bin', 'w.bin')):
        print('ERROR: missing input/u.bin, input/v.bin, or input/w.bin for this run mode.')
        print('Run `wl pre {}` or provide restart/precursor input files.'.format(os.path.basename(case_path)))
        sys.exit(1)

    zo = compute_zo(config)
    _validate_finite('zo.bin', zo)
    _print_range('zo.bin', zo)
    zo.tofile(os.path.join(runtime_input, 'zo.bin'))
    _validate_turbines(runtime_input, config)


def _find_binary(case_path):
    candidates = (
        os.path.join(case_path, 'build', 'src', 'wireles_src'),
        os.path.join(case_path, 'build', 'bin', 'wireles_src'),
        os.path.join(case_path, 'build', 'wireles_src'),
        os.path.join(case_path, 'src', 'wireles_src'),
    )
    for candidate in candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return os.path.abspath(candidate)

    print('ERROR: wireles_src binary not found. Run `wl make {}` first.'.format(os.path.basename(case_path)))
    print('Checked:')
    for candidate in candidates:
        print('  ' + candidate)
    sys.exit(1)


def run(PATH, case_name):
    case_path = get_case_path(PATH, case_name)
    config = get_config(case_path)
    nprocs = int(config['job_np'])
    binary = _find_binary(case_path)

    if nprocs != 1:
        print('ERROR: local `wl run` is single-GPU only and requires job_np = 1.')
        print('Current job_np = {}'.format(nprocs))
        print('Edit job/{}/input/config, set job_np = 1, then rebuild with `wl make {}`.'.format(case_name, case_name))
        sys.exit(1)

    runtime_path = os.path.join(case_path, 'src')
    output_path = os.path.join(case_path, 'output')
    init_path = os.path.join(case_path, 'init_data')

    os.makedirs(runtime_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(init_path, exist_ok=True)

    _prepare_input(case_path, runtime_path, config)
    _ensure_relative_symlink(os.path.join(runtime_path, 'output'), '../output')

    command = ['mpiexec', '-n', str(nprocs), binary]
    print('Run command:')
    print('  ' + ' '.join(command))
    print('Working directory:')
    print('  ' + runtime_path)

    try:
        subprocess.run(command, cwd=runtime_path, check=True)
    except FileNotFoundError as exc:
        missing = exc.filename
        print('ERROR: command not found: ' + str(missing))
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)

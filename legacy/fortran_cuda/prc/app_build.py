#!/usr/bin/env python
import os
import subprocess
import sys

from fctlib import get_case_path, get_config


def _bool_option(value):
    return 'ON' if value else 'OFF'


def _run(command):
    print('  ' + ' '.join(command))
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        print('ERROR: command not found: ' + str(exc.filename))
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)


def build(PATH, case_name):
    case_path = get_case_path(PATH, case_name)
    config = get_config(case_path)

    source_dir = os.path.abspath(os.getcwd())
    build_dir = os.path.join(case_path, 'build')
    config_file = os.path.abspath(os.path.join(case_path, 'input', 'config'))

    compiler = os.environ.get('WIRELES_FC', os.environ.get('FC', 'nvfortran'))
    gpu_arch = os.environ.get('WIRELES_GPU_ARCH', os.environ.get('GPU_ARCH', 'sm_80'))
    use_nvtx = os.environ.get('WIRELES_USE_NVTX', 'ON')
    build_jobs = os.environ.get('WIRELES_BUILD_JOBS', '4')
    precision_double = _bool_option(int(config['double_flag']) != 0)

    os.makedirs(build_dir, exist_ok=True)

    configure_command = [
        'cmake',
        '-S', source_dir,
        '-B', build_dir,
        '-DCMAKE_Fortran_COMPILER={}'.format(compiler),
        '-DGPU_ARCH={}'.format(gpu_arch),
        '-DUSE_NVTX={}'.format(use_nvtx),
        '-DPRECISION_DOUBLE={}'.format(precision_double),
        '-DWIRELES_CONFIG_FILE={}'.format(config_file),
    ]

    build_command = [
        'cmake',
        '--build', build_dir,
        '-j', str(build_jobs),
    ]

    print('CMake configure:')
    _run(configure_command)
    print('CMake build:')
    _run(build_command)

    candidates = (
        os.path.join(build_dir, 'src', 'wireles_src'),
        os.path.join(build_dir, 'bin', 'wireles_src'),
        os.path.join(build_dir, 'wireles_src'),
    )
    for candidate in candidates:
        if os.path.isfile(candidate):
            print('Build complete. Binary at: ' + candidate)
            return

    print('Build complete, but wireles_src was not found in expected locations.')


def make(PATH, case_name):
    build(PATH, case_name)

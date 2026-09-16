"""Checks that the validation report cannot hide local or temporal profile errors."""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import run_backflow_validation as runner
from plot_dtu10mw_precursor import cell_average_log
from jaxwind.config.document import load_case
from jaxwind.config.toml import dumps


@pytest.fixture
def profile_archive(tmp_path):
    args = runner.parser().parse_args(['--output', str(tmp_path), '--smoke'])
    settings = runner.prepare(args)
    settings['smoke'] = False
    case = load_case(tmp_path/'cases/precursor.toml').document
    case['mesh'] = {'cells': [8, 4, 128], 'lengths_m': [512., 128., 1024.]}
    case['physics']['flow']['pressure_acceleration_m_s2'] = [.4**2/1024., 0.]
    (tmp_path/'cases/precursor.toml').write_text(dumps(case))
    flow = case['physics']['flow']
    zf = np.linspace(0., 1024., 129)
    exact = .4/flow['von_karman']*cell_average_log(zf, flow['roughness_length_m'])
    fields = np.broadcast_to(exact[None,None,:,None], (6,3,128,8)).copy()
    directory = tmp_path/'loglaw'
    directory.mkdir()
    runner.write_json(directory/'summary.json', {'status': 'complete'})

    def save(values=fields, times=np.arange(1., 7.)):
        np.savez(directory/'samples.npz', time_seconds=times, mean_u_y=values,
                 wall_stress=np.full((6,3), .4**2), maximum_step_cfl=np.full((6,3), .1),
                 maximum_divergence=np.zeros((6,3)), z_faces=zf, x_faces=np.linspace(0.,512.,9))
    save()
    return tmp_path, settings, fields, save


def test_exact_log_profile_passes(profile_archive):
    output, settings, _, _ = profile_archive
    report = runner.analyze(output, settings)
    assert report['status'] == 'PASS'
    assert report['averaging']['samples'] == 4
    assert report['metrics']['corrected']['log_law_rmse_m_s'] < 1e-12
    assert (output/'analysis/log_law_comparison.pdf').is_file()


def test_opposite_downstream_errors_cannot_cancel(profile_archive):
    output, settings, fields, save = profile_archive
    fields[:,2,:,:2] += .5
    fields[:,2,:,2:4] -= .5
    save()
    report = runner.analyze(output, settings)
    assert report['metrics']['corrected']['log_law_rmse_m_s'] < 1e-12
    assert report['status'] == 'FAIL'
    assert not report['checks']['profile_change_vs_central']


def test_temporal_drift_cannot_cancel(profile_archive):
    output, settings, fields, save = profile_archive
    fields[2:4,2] -= .3
    fields[4:6,2] += .3
    save()
    report = runner.analyze(output, settings)
    assert report['metrics']['corrected']['log_law_rmse_m_s'] < 1e-12
    assert report['status'] == 'FAIL'
    assert not report['checks']['split_window_stationarity']


def test_smoke_is_never_physical_validation(profile_archive):
    output, settings, _, _ = profile_archive
    settings['smoke'] = True
    report = runner.analyze(output, settings)
    assert report['status'] == 'SMOKE_ONLY'
    assert report['log_law_retained_within_declared_tolerances'] is None


def test_missing_time_sample_is_rejected(profile_archive):
    output, settings, fields, save = profile_archive
    save(fields[:-1], np.arange(1., 6.))
    with pytest.raises(AssertionError):
        runner.analyze(output, settings)


def test_timing_must_align_with_samples():
    with pytest.raises(ValueError, match='integer multiple'):
        runner.integral_steps(10., 3., 'sample interval')


def test_changed_prepared_input_is_rejected(tmp_path):
    original = tmp_path/'input.txt'
    original.write_text('original')
    runner.write_json(tmp_path/'source_manifest.json', {str(original): runner.digest(original)})
    runner.verify_sources(tmp_path)
    original.write_text('changed')
    with pytest.raises(RuntimeError, match='changed'):
        runner.verify_sources(tmp_path)


@pytest.mark.parametrize('dry_run', [False, True])
def test_submission_preserves_arguments_and_dry_run(tmp_path, monkeypatch, dry_run):
    import json
    import os
    import subprocess

    executable = tmp_path/'sbatch'
    captured = tmp_path/'submitted.json'
    executable.write_text(f'#!{sys.executable}\nimport json, os, sys\nfrom pathlib import Path\nPath(os.environ["CAPTURE_SUBMISSION"]).write_text(json.dumps(sys.argv[1:]))\nprint("12345")\n')
    executable.chmod(0o755)
    monkeypatch.setenv('PATH', str(tmp_path)+os.pathsep+os.environ['PATH'])
    monkeypatch.setenv('CAPTURE_SUBMISSION', str(captured))
    monkeypatch.setenv('JAXWIND_PYTHON', sys.executable)
    monkeypatch.setenv('DEPENDENCY', 'afterok:123')
    monkeypatch.setenv('ACCOUNT', '')
    output = tmp_path/'output with spaces and $literal'
    argv = ['bash', str(ROOT/'tools/submit_backflow_validation.sh'), '--smoke', '--output', str(output)]
    if dry_run:
        argv.append('--dry-run')
    subprocess.run(argv, check=True, capture_output=True, text=True)
    assert (output/'status.json').is_file()
    if dry_run:
        assert not captured.exists()
    else:
        arguments = json.loads(captured.read_text())
        assert arguments[-3:] == ['--smoke', '--output', str(output)]
        assert '--dependency=afterok:123' in arguments
        assert not any(value.startswith('--account') for value in arguments)


def test_near_wall_damage_is_not_hidden_by_log_band(profile_archive):
    output, settings, fields, save = profile_archive
    fields[:,2,:2] += .4
    save()
    report = runner.analyze(output, settings)
    assert report['metrics']['corrected']['log_law_rmse_m_s'] < 1e-12
    assert report['checks']['profile_change_vs_central']
    assert not report['checks']['near_wall_profile_change_vs_central']
    assert report['status'] == 'FAIL'


def test_wall_stress_change_has_its_own_gate(profile_archive):
    output, settings, _, _ = profile_archive
    path = output/'loglaw/samples.npz'
    with np.load(path) as archive:
        data = dict(archive)
    data['wall_stress'][:,2] *= .8**2
    np.savez(path, **data)
    report = runner.analyze(output, settings)
    assert report['checks']['profile_change_vs_central']
    assert not report['checks']['wall_friction_velocity_change_vs_central']
    assert report['status'] == 'FAIL'

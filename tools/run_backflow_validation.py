"""Submit helper: small wake regression and a paired turbulent open-domain log-law check.

The log-law experiment advances one periodic precursor and two turbine-free
open domains synchronously. Both open domains receive the exact same precursor
plane at every step. Numerical advancement uses the production solver builders.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = ("precursor", "central", "corrected")
DEFAULT_CHECKPOINT = ROOT / "outputs/dtu10mw_central_cfl09_20260916/run/precursor/checkpoint.npz"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT,
                   help='Developed periodic DTU checkpoint; its mesh must match --reference-case.')
    p.add_argument('--reference-case', type=Path, default=ROOT/'cases/DTU10MWPrecursor/fv_central_backflow.toml')
    p.add_argument('--spinup-seconds', type=float, default=1800.)
    p.add_argument('--average-seconds', type=float, default=3600.)
    p.add_argument('--dt', type=float, default=.5, help='Common fixed timestep for all log-law branches.')
    p.add_argument('--sample-seconds', type=float, default=10.)
    p.add_argument('--checkpoint-seconds', type=float, default=600.)
    p.add_argument('--cfl-limit', type=float, default=.9, help='Abort if any actual step exceeds this limit.')
    p.add_argument('--max-profile-change', type=float, default=.2, help='Screening tolerance [m/s], corrected minus central, in each streamwise quarter.')
    p.add_argument('--max-wall-ustar-relative-change', type=float, default=.05, help='Allowed fractional wall-friction-velocity change from central.')
    p.add_argument('--max-log-rmse', type=float, default=1., help='Absolute 20–100 m log-law RMSE tolerance [m/s] in each streamwise quarter.')
    p.add_argument('--max-log-rmse-increase', type=float, default=.1, help='Allowed increase in log-law RMSE over central [m/s].')
    p.add_argument('--max-half-window-change', type=float, default=.2, help='Split-window profile drift tolerance [m/s].')
    p.add_argument('--corrected-scheme', choices=('central', 'central-open-upwind', 'central-open-filter'), default='central-open-upwind')
    p.add_argument('--wake-normal-width', type=float, default=0., help='Corrected AD-BEM minimum streamwise Gaussian width [m]; zero retains legacy widths.')
    p.add_argument('--wake-stabilization', type=float, default=0., help='Rotor-local momentum damping coefficient, between 0 and 1/16.')
    p.add_argument('--sponge-start-fraction', type=float, default=None, help='Enable corrected-branch outlet sponge from this fraction of Lx.')
    p.add_argument('--sponge-timescale', type=float, default=5., help='Outlet sponge relaxation time at high-x [s].')
    p.add_argument('--upstream-sponge-end', type=float, default=None, help='End fraction of an eighth-order upstream mode-absorbing buffer.')
    p.add_argument('--upstream-sponge-timescale', type=float, default=1., help='Peak grid-mode damping time [s].')
    p.add_argument('--skip-wake', action='store_true', help='Run only the turbine-free log-law comparison.')
    p.add_argument('--smoke', action='store_true', help='Tiny synthetic 6 s plumbing test; skips wake and cannot establish log-law retention.')
    p.add_argument('--configure-only', action='store_true', help='Validate inputs and write cases; no device initialization or simulation.')
    p.add_argument('--stage', choices=('all', 'loglaw', 'analyze'), default='all', help=argparse.SUPPRESS)
    return p


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def integral_steps(seconds, dt, name):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f'{name} must be finite and positive')
    steps = round(seconds/dt)
    if steps < 1 or not math.isclose(steps*dt, seconds, abs_tol=1e-9):
        raise ValueError(f'{name} must be an integer multiple of dt')
    return steps


def configuration(args):
    from jaxwind.config.document import load_case
    if not math.isfinite(args.dt) or args.dt <= 0:
        raise ValueError('dt must be finite and positive')
    for key in ('cfl_limit', 'max_wall_ustar_relative_change', 'max_profile_change', 'max_log_rmse', 'max_log_rmse_increase', 'max_half_window_change'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if not math.isfinite(args.wake_stabilization) or not 0. <= args.wake_stabilization <= 1./16.:
        raise ValueError('wake_stabilization must be between 0 and 1/16')
    if not math.isfinite(args.wake_normal_width) or args.wake_normal_width < 0.:
        raise ValueError('wake_normal_width must be finite and nonnegative')
    if args.sponge_start_fraction is not None and (not math.isfinite(args.sponge_start_fraction) or not 0. < args.sponge_start_fraction < 1.):
        raise ValueError('sponge_start_fraction must be between 0 and 1')
    if not math.isfinite(args.sponge_timescale) or args.sponge_timescale <= 0.:
        raise ValueError('sponge_timescale must be finite and positive')
    if args.upstream_sponge_end is not None and (not math.isfinite(args.upstream_sponge_end) or not 0. < args.upstream_sponge_end < 1.):
        raise ValueError('upstream_sponge_end must be between 0 and 1')
    if not math.isfinite(args.upstream_sponge_timescale) or args.upstream_sponge_timescale <= 0.:
        raise ValueError('upstream_sponge_timescale must be finite and positive')
    output = args.output.resolve()
    settings = {k: v for k, v in vars(args).items() if k not in ('configure_only', 'stage')}
    settings.update(output=str(output), checkpoint=str(args.checkpoint.resolve()), reference_case=str(args.reference_case.resolve()))
    if args.smoke:
        settings.update(spinup_seconds=2., average_seconds=4., sample_seconds=1., checkpoint_seconds=2., skip_wake=True, dt=.5)
    dt = settings['dt']
    for name in ('spinup_seconds', 'average_seconds', 'sample_seconds', 'checkpoint_seconds'):
        integral_steps(settings[name], dt, name)
    for name in ('spinup_seconds', 'average_seconds', 'checkpoint_seconds'):
        integral_steps(settings[name], settings['sample_seconds'], name+' / sample interval')
    if settings['average_seconds']/settings['sample_seconds'] < 4:
        raise ValueError('at least four averaged profile samples are required')
    doc = deepcopy(load_case(args.reference_case).document)
    for key in ('workflow', 'initial_conditions'):
        doc.pop(key, None)
    for key in ('turbine', 'wind_farm', 'inflow', 'cooling', 'moisture', 'water_spray'):
        doc['physics'].pop(key, None)
    flow = doc['physics']['flow']
    if flow['pressure_acceleration_m_s2'][0] <= 0 or np.any(np.asarray(flow['coriolis_s']) != 0) or doc['physics']['scalar']['buoyancy_acceleration_per_unit'] != 0:
        raise ValueError('this log-law check requires neutral pressure-driven flow without Coriolis')
    if args.smoke:
        doc['mesh'] = {'cells': [8, 4, 16], 'lengths_m': [512., 128., 128.]}
        flow['pressure_acceleration_m_s2'] = [.4**2/128., 0.]
        doc['case']['initial_profile'] = str(output/'cases/smoke_profile.csv')
        doc['case'].pop('profile_resampling', None)
    elif not args.checkpoint.is_file():
        raise FileNotFoundError(f'{args.checkpoint}: supply --checkpoint with a developed matching periodic checkpoint')
    else:
        from jaxwind.io.checkpoint import checkpoint_metadata
        header = checkpoint_metadata(args.checkpoint)
        if header['formulation'] != 'boussinesq':
            raise ValueError('checkpoint must be Boussinesq')
        for axis, count, length in zip('xyz', doc['mesh']['cells'], doc['mesh']['lengths_m']):
            expected = np.linspace(0., length, count+1)
            if not np.array_equal(expected, np.asarray(header['mesh'][axis+'_faces'])):
                raise ValueError('checkpoint mesh must exactly match the uniform reference case')
        if header['state'].get('record') != 'AtmosphericSolution':
            raise ValueError('checkpoint must be a turbine-free periodic atmospheric state')
        with np.load(args.checkpoint, allow_pickle=False) as saved:
            settings['checkpoint_time_seconds'] = float(saved['state/time'])
    if np.prod(doc['mesh']['cells']) > 1024*1024:
        raise ValueError('use a small reference grid (at most 1M cells) for this debug validation')
    steps = round((settings['spinup_seconds']+settings['average_seconds'])/dt)
    sample_steps = round(settings['sample_seconds']/dt)
    doc['time'] = {'dt_seconds': dt, 'steps': steps, 'chunk_steps': sample_steps, 'checkpoint_every_steps': steps+1}
    doc['numerics'].update(time_integration='rk3', pressure_backend='fft', wall_gradient_correction=True,
                            momentum_advection_scheme='central', outlet_backflow='none', spectrum_diagnostic='none')
    doc['diagnostics'].update(sample_start_step=round(settings['spinup_seconds']/dt), sample_every_steps=sample_steps,
                             reference_length_m=doc['mesh']['lengths_m'][2], inversion_search_max_height_m=doc['mesh']['lengths_m'][2])
    doc['case']['name'] = 'backflow_loglaw_periodic_reference'
    doc['output']['directory'] = str(output/'loglaw/precursor')
    return settings, doc


def verify_sources(output):
    manifest = json.loads((output/'source_manifest.json').read_text())
    changed = [path for path, expected in manifest.items() if not Path(path).is_file() or digest(path) != expected]
    if changed:
        raise RuntimeError('Prepared inputs/source changed; use a new output directory: '+', '.join(changed))


def prepare(args):
    from jaxwind.config.document import load_case
    from jaxwind.config.toml import dumps
    settings, doc = configuration(args)
    output = Path(settings['output'])
    if (output/'settings.json').exists():
        if json.loads((output/'settings.json').read_text()) != settings:
            raise ValueError('output already has different settings; choose a fresh --output')
        verify_sources(output)
        return settings
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'output is not empty: {output}')
    (output/'cases').mkdir(parents=True)
    if args.smoke:
        from plot_dtu10mw_precursor import cell_average_log
        faces = np.linspace(0., 128., 17)
        u = .4/doc['physics']['flow']['von_karman'] * cell_average_log(faces, doc['physics']['flow']['roughness_length_m'])
        data = np.column_stack((.5*(faces[1:]+faces[:-1]), u, np.zeros((16,3)), np.full((16,3), .01), np.zeros(16)))
        np.savetxt(output/'cases/smoke_profile.csv', data, delimiter=',', comments='', header='z_m,u_m_s,v_m_s,w_upper_m_s,scalar,u_rms_m_s,v_rms_m_s,w_upper_rms_m_s,scalar_rms')
    for label in LABELS:
        variant = deepcopy(doc)
        if label != 'precursor':
            variant['numerics']['pressure_backend'] = 'gmg'
            variant['numerics']['outlet_backflow'] = 'energy' if label == 'corrected' else 'none'
            variant['numerics']['momentum_advection_scheme'] = settings['corrected_scheme'] if label == 'corrected' else 'central'
        if label == 'corrected' and settings['sponge_start_fraction'] is not None:
            variant['numerics']['outlet_sponge_start_fraction'] = settings['sponge_start_fraction']
            variant['numerics']['outlet_sponge_timescale_seconds'] = settings['sponge_timescale']
        if label == 'corrected' and settings['upstream_sponge_end'] is not None:
            variant['numerics']['upstream_mode_sponge_end_fraction'] = settings['upstream_sponge_end']
            variant['numerics']['upstream_mode_sponge_timescale_seconds'] = settings['upstream_sponge_timescale']
        variant['case']['name'] = 'backflow_loglaw_'+label
        variant['output']['directory'] = str(output/'loglaw'/label)
        path = output/'cases'/f'{label}.toml'
        path.write_text(dumps(variant))
        load_case(path)
    for label, filename in (('wake_central', 'fv_v80_small_central_outlet_debug.toml'), ('wake_corrected', 'fv_v80_small_central_open_upwind.toml')):
        wake = deepcopy(load_case(ROOT/'cases/HornsRev1'/filename).document)
        if label == 'wake_corrected':
            wake['numerics']['momentum_advection_scheme'] = settings['corrected_scheme']
            wake['physics']['turbine']['minimum_normal_smoothing_width_m'] = settings['wake_normal_width']
            wake['physics']['turbine']['momentum_stabilization_coefficient'] = settings['wake_stabilization']
        if label == 'wake_corrected' and settings['sponge_start_fraction'] is not None:
            wake['numerics']['outlet_sponge_start_fraction'] = settings['sponge_start_fraction']
            wake['numerics']['outlet_sponge_timescale_seconds'] = settings['sponge_timescale']
        if label == 'wake_corrected' and settings['upstream_sponge_end'] is not None:
            wake['numerics']['upstream_mode_sponge_end_fraction'] = settings['upstream_sponge_end']
            wake['numerics']['upstream_mode_sponge_timescale_seconds'] = settings['upstream_sponge_timescale']
        wake['output']['directory'] = str(output/label)
        path = output/'cases'/f'{label}.toml'
        path.write_text(dumps(wake))
        load_case(path)
    files = list((ROOT/'src/jaxwind').rglob('*.py')) + list((output/'cases').iterdir())
    files += [Path(doc['case']['initial_profile']), Path(doc['case']['reference_results'])]
    files += [ROOT/'tools'/name for name in ('run_backflow_validation.py', 'submit_backflow_validation.sh', 'backflow_validation.sbatch', 'analyze_hornsrev_backflow.py', 'analyze_open_v80_debug.py', 'plot_dtu10mw_precursor.py', 'render_fv_wake_gif.py')]
    files += [p for p in (ROOT/'cases/HornsRev1/turbines/V80').rglob('*') if p.is_file()]
    if not args.smoke:
        files.append(args.checkpoint.resolve())
    manifest = {str(p.resolve()): digest(p) for p in files}
    write_json(output/'settings.json', settings)
    write_json(output/'source_manifest.json', manifest)
    # Large input checkpoint stays in place, bound by its content hash.
    with tarfile.open(output/'source_snapshot.tar.gz', 'w:gz') as archive:
        for p in files:
            if p.resolve() != args.checkpoint.resolve():
                archive.add(p, arcname=str(p.resolve()).lstrip('/'))
    write_json(output/'status.json', {'status': 'prepared', 'smoke': args.smoke})
    print(f'Prepared {output}; log-law grid {doc["mesh"]["cells"]}; spinup {settings["spinup_seconds"]:g}s + averaging {settings["average_seconds"]:g}s', flush=True)
    return settings


def run_loglaw(output, settings):
    import jax
    import jax.numpy as jnp
    from jaxwind import (build_open_atmospheric_step, build_pressure_poisson,
                         initial_atmospheric_solution, periodic_to_open_velocity,
                         enforce_open_velocity, extract_inflow_plane, courant_number, divergence)
    from jaxwind.config.abl import load_fv_abl
    from jaxwind.config.document import load_case
    from jaxwind.simulation.abl import build_models, build_periodic_advance, initialize_periodic
    from jaxwind.io.state_fields import atmospheric_state
    from jaxwind.io.checkpoint import save_checkpoint
    from jaxwind.numerics.discretization import cell_velocity
    from jaxwind.wall import surface_stress
    configs = [load_fv_abl(output/'cases'/f'{label}.toml') for label in LABELS]
    configured = configs[0]
    jax.config.update('jax_enable_x64', configured.physical.dtype == 'float64')
    grid = configured.physical.physical_grid
    warm = initialize_periodic(configured, jax, jnp) if settings['smoke'] else atmospheric_state(settings['checkpoint'], grid)
    # Reset clocks for a new synchronous RK3 experiment; RK3 ignores prior-step tendencies.
    warm = warm._replace(time=jnp.asarray(0., warm.time.dtype), step=jnp.asarray(0, warm.step.dtype))
    periodic_step, _ = build_periodic_advance(configured)
    steps = [periodic_step]
    states = [warm]
    walls = [build_models(configured, periodic_x=True)[1].surface]
    first = extract_inflow_plane(warm, grid, 0)
    for config in configs[1:]:
        boundaries, momentum, scalar, buoyancy, surface = build_models(config, periodic_x=False, pressure_force_enabled=True, evolve_scalar=False)
        solver = build_pressure_poisson(grid, backend='gmg', periodic_x=False, dtype=config.physical.dtype,
                                       config={'tolerance': config.options.gmg_tolerance or 1e-5,
                                               'presweeps': config.options.gmg_presweeps, 'postsweeps': config.options.gmg_postsweeps,
                                               'anisotropy_aware': config.options.gmg_anisotropy_aware})
        steps.append(build_open_atmospheric_step(grid, boundaries, solver, momentum, scalar, buoyancy, surface, scheme='rk3'))
        velocity = enforce_open_velocity(periodic_to_open_velocity(warm.velocity, grid), first, grid)
        states.append(initial_atmospheric_solution(grid, velocity, warm.scalar, dtype=config.physical.dtype))
        walls.append(momentum.surface)
    dt = settings['dt']
    count = round(settings['sample_seconds']/dt)
    blocks = round((settings['spinup_seconds']+settings['average_seconds'])/settings['sample_seconds'])

    @jax.jit
    def advance(states):
        def one(current, _):
            plane = extract_inflow_plane(current[0], grid, 0)
            new = (steps[0](current[0], dt), steps[1](current[1], dt, plane), steps[2](current[2], dt, plane))
            return new, jnp.stack([courant_number(s.velocity, grid, dt) for s in new])
        final, cfl = jax.lax.scan(one, states, None, length=count)
        return final, jnp.max(cfl, axis=0)

    @jax.jit
    def diagnostics(states):
        result = []
        for state, wall in zip(states, walls):
            u, _, w = cell_velocity(state.velocity)
            tau, _ = surface_stress(state.velocity, grid, wall)
            if tau.shape[-1] == grid.nx+1:
                tau = .5*(tau[:,:-1]+tau[:,1:])
            fluctuations = u-jnp.mean(u, axis=(1,2), keepdims=True)
            result.append((jnp.mean(u, axis=1), jnp.mean(fluctuations*w, axis=(1,2)),
                           jnp.mean(tau), jnp.max(jnp.abs(divergence(state.velocity, grid))),
                           jnp.min(state.velocity.x[..., -1])))
        return tuple(result)

    directory = output/'loglaw'
    directory.mkdir(exist_ok=True)
    rows, times, cfls = [], [], []
    states = tuple(states)
    def save_samples():
        # time, branch, z, x; means are over y at each equal-volume x-z cell.
        np.savez_compressed(directory/'samples.npz', time_seconds=np.asarray(times),
                            mean_u_y=np.asarray([[branch[0] for branch in row] for row in rows]),
                            resolved_uw=np.asarray([[branch[1] for branch in row] for row in rows]),
                            wall_stress=np.asarray([[branch[2] for branch in row] for row in rows]),
                            maximum_divergence=np.asarray([[branch[3] for branch in row] for row in rows]),
                            outlet_minimum_u=np.asarray([[branch[4] for branch in row] for row in rows]),
                            maximum_step_cfl=np.asarray(cfls), z_faces=np.asarray(grid.z_faces), x_faces=np.asarray(grid.x_faces))
    started = time.monotonic()
    for block in range(1, blocks+1):
        states, cfl = advance(states)
        cfl = np.asarray(cfl)
        if not np.isfinite(cfl).all() or cfl.max() > settings['cfl_limit']:
            raise RuntimeError(f'Actual step CFL {cfl.tolist()} exceeds limit {settings["cfl_limit"]}; use a smaller dt in a fresh run')
        row = jax.device_get(diagnostics(states))
        if not all(np.isfinite(value).all() for branch in row for value in branch):
            raise RuntimeError('non-finite profile diagnostics')
        now = block*settings['sample_seconds']
        rows.append(row); times.append(now); cfls.append(cfl)
        print(f'loglaw t={now:g}s CFL={cfl.max():.4g} elapsed={time.monotonic()-started:.1f}s', flush=True)
        checkpoint_due = math.isclose(now % settings['checkpoint_seconds'], 0., abs_tol=1e-7) or block == blocks
        if checkpoint_due:
            save_samples()
            for label, state, config in zip(LABELS, states, configs):
                save_checkpoint(directory/f'{label}_checkpoint.npz', state, metadata={
                    'formulation': 'boussinesq', 'fingerprint': 'paired-validation-'+label,
                    'mesh': {axis+'_faces': np.asarray(getattr(grid, axis+'_faces')).tolist() for axis in 'xyz'},
                    'initial_time': 0., 'target_time': blocks*settings['sample_seconds'],
                    'resolved_case': load_case(output/'cases'/f'{label}.toml').document,
                    'note': 'Diagnostic restart fields; rerun this validation in a fresh output directory.'})
    write_json(directory/'summary.json', {'status': 'complete', 'time_seconds': times[-1], 'samples': len(times),
                                          'elapsed_seconds': time.monotonic()-started,
                                          'devices': [str(x) for x in jax.devices()], 'same_inflow_at_every_step': True})


def analyze(output, settings):
    from plot_dtu10mw_precursor import cell_average_log
    from jaxwind.config.document import load_case
    with np.load(output/'loglaw/samples.npz') as data:
        time_s = data['time_seconds']; fields = data['mean_u_y']; tau = data['wall_stress']
        cfl = data['maximum_step_cfl']; div = data['maximum_divergence']
        zf, xf = data['z_faces'], data['x_faces']
    summary = json.loads((output/'loglaw/summary.json').read_text())
    if summary['status'] != 'complete':
        raise ValueError('log-law experiment is incomplete')
    duration = settings['spinup_seconds']+settings['average_seconds']
    expected_times = np.arange(1, round(duration/settings['sample_seconds'])+1)*settings['sample_seconds']
    np.testing.assert_allclose(time_s, expected_times, rtol=0., atol=1e-6)
    if not np.isfinite(fields).all():
        raise ValueError('non-finite profile archive')
    keep = time_s > settings['spinup_seconds']+1e-7
    expected = round(settings['average_seconds']/settings['sample_seconds'])
    if keep.sum() != expected:
        raise ValueError('averaging window has the wrong number of samples')
    z = .5*(zf[1:]+zf[:-1]); band = (z >= 20.) & (z <= min(100., .1*zf[-1]))
    # Smoke grids deliberately have a smaller H; use enough levels to test plotting.
    if settings['smoke']:
        band = (z >= 20.) & (z <= 100.)
    if band.sum() < 3:
        raise ValueError('at least three levels are required in the log-law band')
    flow = load_case(output/'cases/precursor.toml').document['physics']['flow']
    ustar = math.sqrt(flow['pressure_acceleration_m_s2'][0]*zf[-1])
    log_u = ustar/flow['von_karman']*cell_average_log(zf, flow['roughness_length_m'])
    sample = fields[keep]  # regular time, branch, z, x; already volume-averaged in y
    profiles = sample.mean(axis=(0,3), dtype=np.float64)
    quarters = np.array_split(np.arange(fields.shape[-1]), 4)
    quarter_profiles = np.stack([sample[:,:,:,indices].mean(axis=(0,3), dtype=np.float64) for indices in quarters], axis=1)
    rmse = lambda value: float(np.sqrt(np.mean(np.asarray(value)**2)))
    middle = len(sample)//2
    half_change = [rmse(sample[middle:,i].mean(axis=(0,2), dtype=np.float64)[band]-sample[:middle,i].mean(axis=(0,2), dtype=np.float64)[band]) for i in range(3)]
    metrics = {}
    for i, label in enumerate(LABELS):
        metrics[label] = {
            'log_law_rmse_m_s': rmse(profiles[i,band]-log_u[band]),
            'log_law_bias_m_s': float(np.mean(profiles[i,band]-log_u[band])),
            'quarter_log_law_rmse_m_s': [rmse(p[band]-log_u[band]) for p in quarter_profiles[i]],
            'half_window_profile_change_m_s': half_change[i],
            'wall_stress_ustar_m_s': float(np.sqrt(np.mean(tau[keep,i]))),
            'maximum_step_cfl': float(cfl[:,i].max()),
            'maximum_sampled_divergence_s': float(div[:,i].max()),
        }
    quarter_delta = [rmse((quarter_profiles[2,k]-quarter_profiles[1,k])[band]) for k in range(4)]
    quarter_increase = np.asarray(metrics['corrected']['quarter_log_law_rmse_m_s'])-np.asarray(metrics['central']['quarter_log_law_rmse_m_s'])
    near_wall = z < 20.
    near_wall_delta = float(np.max(np.abs(quarter_profiles[2,:,near_wall]-quarter_profiles[1,:,near_wall]))) if near_wall.any() else 0.
    wall_change = abs(metrics['corrected']['wall_stress_ustar_m_s']/metrics['central']['wall_stress_ustar_m_s']-1.)
    checks = {
        'near_wall_profile_change_vs_central': near_wall_delta <= settings['max_profile_change'],
        'wall_friction_velocity_change_vs_central': wall_change <= settings.get('max_wall_ustar_relative_change', .05),
        'absolute_log_law_error': max(metrics['corrected']['quarter_log_law_rmse_m_s']) <= settings['max_log_rmse'],
        'profile_change_vs_central': max(quarter_delta) <= settings['max_profile_change'],
        'log_law_error_increase_vs_central': float(quarter_increase.max()) <= settings['max_log_rmse_increase'],
        'split_window_stationarity': max(half_change) <= settings['max_half_window_change'],
        'cfl': float(cfl.max()) <= settings['cfl_limit'],
        'divergence': float(div.max()) <= 1e-5,
    }
    report = {
        'status': 'SMOKE_ONLY' if settings['smoke'] else ('PASS' if all(checks.values()) else 'FAIL'),
        'log_law_retained_within_declared_tolerances': None if settings['smoke'] else all(checks.values()),
        'settings': settings, 'metrics': metrics, 'checks': checks,
        'near_wall_maximum_profile_change_m_s': near_wall_delta,
        'wall_ustar_relative_change': wall_change,
        'quarter_corrected_minus_central_rmse_m_s': quarter_delta,
        'quarter_log_rmse_increase_m_s': quarter_increase.tolist(),
        'outlet_sponge': {'start_fraction': settings.get('sponge_start_fraction'), 'timescale_seconds': settings.get('sponge_timescale'), 'target': 'Instantaneous inlet spanwise-mean u(z), v(z); w=0', 'quarter_x_ranges_m': [[float(xf[q[0]]), float(xf[q[-1]+1])] for q in quarters]},
        'upstream_mode_sponge': {'end_fraction': settings.get('upstream_sponge_end'), 'timescale_seconds': settings.get('upstream_sponge_timescale'), 'operator': '-D4.T W D4 / (256 tau), open x, compact buffer, unchanged AD-BEM'},
        'averaging': {'samples': int(keep.sum()), 'first_time_s': float(time_s[keep][0]), 'last_time_s': float(time_s[keep][-1]),
                      'definition': 'Cell-centered u, uniform cell-volume mean over x,y at each z, then equal-time mean of regularly spaced samples after spin-up; also four streamwise subvolumes.'},
        'reference': {'ustar_m_s': ustar, 'roughness_m': flow['roughness_length_m'], 'von_karman': flow['von_karman'],
                      'band_m': [float(z[band][0]), float(z[band][-1])], 'vertical_cell_integral': True},
        'limitations': 'Declared tolerances are engineering regression screens, not a proof of statistical convergence or wake recovery. Log-law runs have no turbines and periodic lateral boundaries. Both open branches receive identical turbulent inflow at every step. A short smoke test cannot establish profile preservation.',
    }
    wake_path = output/'wake_corrected/backflow_analysis/metrics.json'
    if wake_path.is_file():
        old = json.loads((output/'wake_central/backflow_analysis/metrics.json').read_text())
        new = json.loads(wake_path.read_text())
        ratio = new['final_upstream_second_difference_rms_m_s']/max(old['final_upstream_second_difference_rms_m_s'], 1e-15)
        report['wake'] = {'upstream_oscillation_ratio': ratio, 'reduction_percent': 100*(1-ratio),
                          'central_face_jump_m_s': old['final_raw_upstream_maximum_adjacent_jump_m_s'],
                          'corrected_face_jump_m_s': new['final_raw_upstream_maximum_adjacent_jump_m_s'],
                          'screen_pass': ratio <= .1 and new['maximum_divergence_s'] <= 1e-5 and new['maximum_inlet_error_m_s'] <= 1e-5}
        if not report['wake']['screen_pass'] and not settings['smoke']:
            report['status'] = 'FAIL'
    target = output/'analysis'; target.mkdir(exist_ok=True)
    write_json(target/'report.json', report)
    columns = [z, log_u, *profiles]
    np.savetxt(target/'profiles.csv', np.column_stack(columns), delimiter=',', comments='',
               header='z_m,cell_average_log_u_m_s,precursor_u_m_s,central_open_u_m_s,corrected_open_u_m_s')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(15,5),layout='constrained')
    colors = ('#444444','#e69f00','#0072b2')
    for i, label in enumerate(LABELS):
        axes[0].plot(profiles[i],z,label=label,color=colors[i])
        axes[1].plot(profiles[i,band]-log_u[band],z[band],label=label,color=colors[i])
    axes[0].plot(log_u,z,'k--',label='cell-integrated log law')
    axes[0].set(xlabel='Mean u [m/s]',ylabel='z [m]',ylim=(0,min(200,zf[-1])),title='Time / volume mean')
    axes[0].legend(fontsize=9)
    axes[1].axvline(0,color='k',lw=.6); axes[1].set(xlabel='u − log law [m/s]',ylabel='z [m]',title='Surface-layer error')
    for k in range(4):
        axes[2].plot((quarter_profiles[2,k]-quarter_profiles[1,k])[band],z[band],label=f'x={xf[quarters[k][0]]:g}–{xf[quarters[k][-1]+1]:g} m')
    axes[2].axvline(0,color='k',lw=.6);axes[2].set(xlabel='Corrected − central [m/s]',ylabel='z [m]',title='Downstream profile change');axes[2].legend(fontsize=8)
    for ax in axes: ax.grid(alpha=.2)
    profile_verdict = "PASS" if all(checks.values()) else "FAIL"
    fig.suptitle('SMOKE TEST — no physical validation' if settings['smoke'] else f'Log-law screening: {profile_verdict} | Combined screening: {report["status"]}')
    fig.savefig(target/'log_law_comparison.png',dpi=170);fig.savefig(target/'log_law_comparison.pdf');plt.close(fig)
    lines = [f'# Backflow correction validation: {report["status"]}', '',
             ("SMOKE_ONLY: no physical validation." if settings["smoke"] else f"Log-law checks: **{profile_verdict}**. The combined verdict also includes the wake check when available."), "",
             f'Spin-up: {settings["spinup_seconds"]:g} s; averaging: {settings["average_seconds"]:g} s; {int(keep.sum())} samples.',
             '', 'One periodic precursor supplies the identical turbulent inlet to turbine-free central and corrected open domains. Porté-Agel is enabled in all branches. Both open domains retain the pressure forcing.', '',
             '| Branch | Log-law RMSE [m/s] | Half-window change [m/s] | Wall u* [m/s] |', '|---|---:|---:|---:|']
    for label in LABELS:
        m=metrics[label];lines.append(f'| {label} | {m["log_law_rmse_m_s"]:.5f} | {m["half_window_profile_change_m_s"]:.5f} | {m["wall_stress_ustar_m_s"]:.5f} |')
    lines += ['', 'The checks include all four streamwise quarters, not only the whole-domain mean.',
              f'Declared limits: absolute quarter log-law RMSE ≤ {settings["max_log_rmse"]:g} m/s; corrected-vs-central profile RMSE ≤ {settings["max_profile_change"]:g} m/s; increase in log-law RMSE ≤ {settings["max_log_rmse_increase"]:g} m/s; split-window change ≤ {settings["max_half_window_change"]:g} m/s.', '',
              '[Profile plot](log_law_comparison.png) · [Detailed metrics and checks](report.json) · [Profile data](profiles.csv)', '',report['limitations']]
    lines += ['', f'Near-wall (<20 m) maximum quarter-profile change: {near_wall_delta:.5f} m/s (limit {settings["max_profile_change"]:g}); relative wall-u* change: {wall_change:.3%} (limit {settings.get("max_wall_ustar_relative_change", .05):.0%}).']
    if settings['smoke']:
        lines += ['', '**SMOKE_ONLY: successful execution is not evidence of log-law retention.**']
    if 'wake' in report:
        lines += ['', f'Small wake oscillation reduction: {report["wake"]["reduction_percent"]:.2f}%.',
                  '[Wake comparison](../wake_corrected/backflow_analysis/upstream_comparison.png)']
    (target/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status': report['status'], 'report': str(target/'README.md'), 'checks': checks},indent=2), flush=True)
    return report


def command(argv, log):
    print('Running: '+' '.join(map(str, argv)), flush=True)
    with log.open('w') as stream:
        subprocess.run(list(map(str, argv)), stdout=stream, stderr=subprocess.STDOUT, check=True, cwd=ROOT)


def main():
    args = parser().parse_args()
    output = args.output.resolve()
    if args.stage == 'analyze':
        report = analyze(output, json.loads((output/'settings.json').read_text()))
        return 2 if report['status'] == 'FAIL' else 0
    settings = prepare(args)
    if args.configure_only:
        return 0
    if args.stage == 'loglaw':
        run_loglaw(output, settings)
        return 0
    previous = json.loads((output/'status.json').read_text())
    if previous['status'] != 'prepared':
        raise ValueError('this output has already run; use a new output directory (or --stage analyze for saved results)')
    write_json(output/'status.json', {'status': 'running', 'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'smoke': args.smoke})
    try:
        command([sys.executable, Path(__file__).resolve(), *sys.argv[1:], '--stage', 'loglaw'], output/'loglaw.log')
        if not settings['skip_wake']:
            for label in ('wake_central', 'wake_corrected'):
                verify_sources(output)
                command([sys.executable, '-m', 'jaxwind', 'run', output/'cases'/f'{label}.toml'], output/f'{label}.log')
                command([sys.executable, ROOT/'tools/analyze_open_v80_debug.py', output/label], output/f'{label}_render.log')
                argv = [sys.executable, ROOT/'tools/analyze_hornsrev_backflow.py', output/label]
                if label == 'wake_corrected':
                    argv += ['--baseline', output/'wake_central']
                command(argv, output/f'{label}_analysis.log')
        verify_sources(output)
        report = analyze(output, settings)
        write_json(output/'status.json', {'status': 'complete', 'screening': report['status'], 'slurm_job_id': os.environ.get('SLURM_JOB_ID')})
        return 2 if report['status'] == 'FAIL' else 0
    except BaseException as error:
        write_json(output/'status.json', {'status': 'failed', 'error': str(error), 'slurm_job_id': os.environ.get('SLURM_JOB_ID')})
        raise


if __name__ == '__main__':
    raise SystemExit(main())

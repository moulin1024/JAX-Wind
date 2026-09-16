#!/usr/bin/env bash
# CPU post-processing only. Requires numpy, matplotlib, and imageio-ffmpeg
# (or system ffmpeg), plus Python 3.11+ or tomli. No GIF is generated.
# Usage: bash tools/postprocess_hitsz_ln2.sh [SETUP_DIRECTORY]
set -euo pipefail
(( $# <= 1 )) || { echo "Expected at most one setup-directory argument." >&2; exit 2; }
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export JAX_PLATFORMS=cpu MPLBACKEND=Agg
"${PYTHON:-python3}" - "${1:-${REPO_ROOT}/outputs/hitsz_ln2_512_setup}" <<'PY'
import json
from pathlib import Path
import sys
from types import SimpleNamespace
try:
    import tomllib
except ImportError:
    import tomli as tomllib
import numpy as np
import matplotlib.pyplot as plt
from tools.render_fv_wake_gif import _render_four_panel, _ffmpeg_executable
setup = Path(sys.argv[1]).resolve()
base = tomllib.loads((setup / 'case.toml').read_text())
graph = tomllib.loads((setup / 'workflow.toml').read_text())
root = Path(graph['output']['directory'])
manifest = json.loads((root / 'workflow.json').read_text())
for stage in ('warmup', 'precursor', 'control', 'main'):
    if manifest['stages'].get(stage, {}).get('status') != 'complete':
        raise SystemExit(f'{stage} is incomplete; resume the simulation before analysis.')
_ffmpeg_executable()  # Fail before analysis if no MP4 encoder is installed.
turbine = base['physics']['turbine']
hub, xt = turbine['hub_height_m'], turbine['x_m']
D = base['diagnostics']['reference_length_m']
frames = {}
for name in ('control', 'main'):
    with np.load(root / name / 'flow_frames.npz', allow_pickle=False) as archive:
        frames[name] = {k: archive[k] for k in archive.files}
    a = frames[name]
    for key in ('u_hub_yx', 'u_center_zx', 'scalar_hub_yx', 'scalar_center_zx'):
        if a[key].shape[0] != 100 or not np.isfinite(a[key]).all():
            raise ValueError(f'{name}: expected 100 finite frames for {key}')
a, b = frames['control'], frames['main']
for key in ('x_m', 'y_m', 'z_m', 'time_seconds'):
    if not np.allclose(a[key], b[key], rtol=0, atol=1e-5):
        raise ValueError(f'Control and LN2 {key} do not match')
times = b['time_seconds']
if np.any(np.diff(times) <= 0) or not np.allclose(times, np.arange(1, 101)*1.8, atol=1e-4, rtol=0):
    raise ValueError('Expected 100 sample times from 1.8 through 180 s')
# Stream precursor chunks; use one common full-record hub-height reference.
inflow = root / 'precursor/inflow'
meta = json.loads((inflow / 'metadata.json').read_text())
total, count = None, 0
for chunk in meta['chunks']:
    with np.load(inflow / chunk['file'], allow_pickle=False) as archive:
        u = archive['x_velocity']
        if not np.isfinite(u).all():
            raise ValueError('Nonfinite precursor velocity')
        summed = u.sum(axis=0, dtype=np.float64)
        total = summed if total is None else total + summed
        count += len(u)
if count != meta['samples'] or count == 0:
    raise ValueError('Incomplete precursor recording')
mean = total / count
z, y = a['z_m'], a['y_m']
if mean.shape != (len(z), len(y)):
    raise ValueError(f'Unexpected inlet plane shape: {mean.shape}')
uref = float(np.interp(hub, z, [np.interp(turbine['y_m'], y, row) for row in mean]))
if not np.isfinite(uref) or uref <= 0:
    raise ValueError('Invalid reference velocity')
hi = np.searchsorted(z, hub)
if not 0 < hi < len(z):
    raise ValueError('Hub outside cell-center interpolation range')
w = (hub-z[hi-1])/(z[hi]-z[hi-1])
selected = (times > 90.0001) & (times <= 180.0001)
if selected.sum() != 50:
    raise ValueError('Expected 50 matched frames in the last 90 seconds')
def profile(f):
    u = f['u_center_zx'][selected].mean(axis=0, dtype=np.float64)
    return (1-w)*u[hi-1] + w*u[hi]
uc, uj = profile(a), profile(b)
dc, dj = 1-uc/uref, 1-uj/uref
xd = (a['x_m']-xt)/D
out = root / 'postprocessing'
out.mkdir(exist_ok=True)
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 7), constrained_layout=True)
axes[0].plot(xd, dc, label='No LN2')
axes[0].plot(xd, dj, label='LN2')
axes[0].legend()
axes[0].set(ylabel='Centerline deficit: 1 − mean(u)/Uref', title=f'HITSZ: 50 matched frames, 91.8–180 s; Uref={uref:.4f} m/s')
axes[1].plot(xd, 100*(dj-dc))
axes[1].axhline(0, color='grey', lw=0.8)
axes[1].set(xlabel='(x − turbine x) / D', ylabel='LN2 − no LN2 [percentage points]', xlim=(0, float(xd.max())))
for ax in axes:
    ax.grid(alpha=0.25)
for ext in ('png', 'pdf'):
    fig.savefig(out / f'centerline_deficit.{ext}', dpi=180)
plt.close(fig)
np.savetxt(out / 'centerline_deficit.csv', np.column_stack((a['x_m'], xd, uc, uj, dc, dj, 100*(dj-dc))), delimiter=',', comments='', header='x_m,x_over_D,control_u_m_s,ln2_u_m_s,control_deficit,ln2_deficit,change_percentage_points')
far = (xd >= 4) & (xd <= 12)
summary = dict(reference_velocity_m_s=uref, reference='full precursor, interpolated at hub y/z', frames_averaged=50, averaging_window_s=[float(times[selected][0]),float(times[selected][-1])], mean_change_percentage_points_4_12D=float(100*(dj-dc)[far].mean()), interpretation='Fixed hub centerline, no spatial smoothing; not a whole-wake momentum or turbine-power measure.')
(out / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
nx, ny, nz = base['mesh']['cells']
lx, ly, lz = base['mesh']['lengths_m']
grid = SimpleNamespace(x_faces=np.linspace(0,lx,nx+1), y_faces=np.linspace(0,ly,ny+1), z_faces=np.linspace(0,lz,nz+1))
disk = SimpleNamespace(x=xt, y=turbine['y_m'], z=hub, tip_radius=D/2)
for name, f in frames.items():
    _render_four_panel(f['u_hub_yx'], f['u_center_zx'], f['scalar_hub_yx'], f['scalar_center_zx'], times, grid, disk, out / f'{name}_wake.mp4', 10, f'HITSZ {name}')
print(f'Wrote deficit PNG/PDF/CSV, summary JSON, and two 100-frame MP4s to {out}')
PY

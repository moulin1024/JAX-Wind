#!/usr/bin/env python3
"""Plot the instantaneous axial jet from a standalone solver snapshot."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('snapshot', type=Path)
args = parser.parse_args()
with np.load(args.snapshot) as data:
    u = .5*(data['u'][1:]+data['u'][:-1])
    length, h, t = data['length'], data['h'], float(data['time'])
    parcels = data['parcels']
nx, ny, nz = u.shape
x, y, z = [(np.arange(n)+.5)*d for n, d in zip(u.shape, h)]
yc, zc = length[1]/2, length[2]/2
# Average the two planes bracketing the geometric centre on an even grid.
def mid(a, axis):
    n = a.shape[axis]
    return np.take(a, [n//2-1, n//2] if n % 2 == 0 else [n//2], axis=axis).mean(axis=axis)
uxy, uxz = mid(u, 2), mid(u, 1)
centre = mid(uxy, 1)
fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
vmin, vmax = min(0., float(u.min())), float(u.max())
for ax, field, coord, extent, label in (
        (axes[0, 0], uxy, 1, [0, length[0], -yc, yc], 'y'),
        (axes[1, 0], uxz, 2, [0, length[0], -zc, zc], 'z')):
    im = ax.imshow(field.T, origin='lower', extent=extent, aspect='auto',
                   vmin=vmin, vmax=vmax, cmap='turbo', interpolation='nearest')
    ax.set(xlabel='x [m]', ylabel=f'{label} - centre [m]',
           title=f'Axial gas velocity, central x–{label} plane')
    other = 2 if coord == 1 else 1
    selected = abs(parcels[:, other]-length[other]/2) < h[other]
    ax.scatter(parcels[selected, 0], parcels[selected, coord]-length[coord]/2,
               s=2, c='k', alpha=.3, linewidths=0)
fig.colorbar(im, ax=axes[:, 0], label='u [m/s]', shrink=.85)
axes[0, 1].plot(x, centre)
axes[0, 1].axhline(3, color='grey', ls='--', lw=1, label='Inlet: 3 m/s')
axes[0, 1].set(xlabel='x [m]', ylabel='u [m/s]', title='Instantaneous centreline')
axes[0, 1].legend()
for station in (.1, .4, .95, 1.85):
    profile = np.array([np.interp(station, x, uxy[:, j]) for j in range(ny)])
    axes[1, 1].plot(y-yc, profile, label=f'x={station:g} m')
axes[1, 1].set(xlabel='y - centre [m]', ylabel='u [m/s]', title='Central-plane transverse profiles')
axes[1, 1].legend()
for ax in axes[:, 1]:
    ax.grid(alpha=.2)
fig.suptitle(f'Standalone classical-Smagorinsky LES — {nx}×{ny}×{nz}, t={t:.4f} s\n'
             'Developing jet; instantaneous fields, uniform inlet')
output = args.snapshot.with_name(args.snapshot.stem+'_jet.png')
fig.savefig(output, dpi=170)
print(output)
print(f'Peak axial gas velocity: {u.max():.6f} m/s; centreline peak: {centre.max():.6f} m/s')

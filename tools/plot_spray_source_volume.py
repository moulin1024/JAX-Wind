"""Plot archived source-volume diagnostics; no simulation."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(__doc__);p.add_argument('directory',type=Path);a=p.parse_args()
f=np.load(a.directory/'fields.npz');d=json.loads((a.directory/'summary.json').read_text())
x=f['x'];z=f['z'];ny=len(f['y'])
fig,ax=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
for pane,key,title in zip(ax.flat[:3],['heat_s-1','vapor_s-1','total_s-1'],['Cooling contribution','Evaporation contribution','Net volume source']):
 field=f[key][:,ny//2-1:ny//2+1,:].mean(1)
 bound=max(abs(field.min()),abs(field.max()))
 im=pane.pcolormesh(x,z,field,cmap='RdBu_r',vmin=-bound,vmax=bound,shading='nearest')
 fig.colorbar(im,ax=pane,label='Divergence source [1/s]')
 pane.set(xlabel='x [m]',ylabel='z − centre [m]',title=title)
for name,label in [('mixed_log','Pressure outlet'),('uniform_outlet_log','Uniform outlet response'),('mixed_eos_rate','EOS density control')]:
 if name not in d['controls']:continue
 rows=d['controls'][name]['stations']
 ax[1,1].plot([r['x_requested_m'] for r in rows],[r['centre_du_m_s'] for r in rows],marker='o',label=label)
ax[1,1].axhline(0,color='k',lw=.7);ax[1,1].legend(fontsize=8)
ax[1,1].set(xlabel='x [m]',ylabel='Centre axial velocity correction [m/s]',title='Frozen potential response (not coupled low-Mach)')
fig.suptitle('64 × 32 × 32, saved 4 s state; native parcel thermal source')
fig.savefig(a.directory/'source_volume.png',dpi=170)

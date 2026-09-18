"""Compare existing production archive with Figure 9; no simulation advancement."""
import sys,json,csv,argparse,tomllib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from plot_spray_radial_momentum import digitize
parser=argparse.ArgumentParser(__doc__)
parser.add_argument('run',type=Path)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
BASE=ROOT/'outputs/inertial_centre_goal/momentum_budget'
OUT=args.output
RUN=args.run
doc=tomllib.loads((RUN/'resolved_case.toml').read_text())
if doc['physics']['flow']['streamwise_velocity_m_s'] != 3.0:
 raise ValueError('sim.pdf Figure 9 describes case 3 (3 m/s inlet); use compare_spray_saved_velocity.py for other cases')
OUT.mkdir(parents=True,exist_ok=True)
mesh_label=' × '.join(str(n) for n in doc['mesh']['cells'])
a=np.load(RUN/'flow_frames.npz');t=a['time_seconds'];sel=(t>=3)&(t<=4)
x=a['x_m'];z=a['z_m']-.2925;stations=np.array([.1,.4,.7,1.,1.3,1.6,1.9])

def at_x(v):
 return np.stack([np.interp(stations,x,row) for row in v])

mean=[];std=[];instant=[]
for name in ['u_center_zx','scalar_center_zx','vapor_center_zx']:
 v=a[name][sel]
 if name.startswith('scalar'):v=v+doc['physics']['moisture']['temperature_offset_k']-273.15
 if name.startswith('vapor'):v=1000*v/(1+v)
 avg=np.trapezoid(v,t[sel],axis=0)/(t[sel][-1]-t[sel][0])
 variance=np.trapezoid((v-avg)**2,t[sel],axis=0)/(t[sel][-1]-t[sel][0])
 mean.append(at_x(avg));std.append(at_x(np.sqrt(variance)));instant.append(at_x(v[-1]))
mean=np.array(mean);std=np.array(std);instant=np.array(instant)
# Interpolate staggered vector components onto the vertical centre plane.
j=len(a['y_m'])//2
c=np.load(RUN/'checkpoint.npz');uf=c['state/velocity/x'];up=.25*(uf[:,j-1,:-1]+uf[:,j,:-1]+uf[:,j-1,1:]+uf[:,j,1:]);del uf
vf=c['state/velocity/y'];vp=.25*(vf[:,j-1,:]+2*vf[:,j,:]+vf[:,j+1,:]);del vf
wf=c['state/velocity/z'];wp=.25*(wf[:-1,j-1,:]+wf[:-1,j,:]+wf[1:,j-1,:]+wf[1:,j,:]);del wf
speed=at_x(np.sqrt(up**2+vp**2+wp**2));axial=at_x(up)
np.testing.assert_allclose(axial,instant[0],atol=1e-12)
zp,paper=digitize(BASE/'figure9_original.jpg')
fig,axes=plt.subplots(1,7,figsize=(18,5),sharey=True,layout='constrained')
for i,(ax,s) in enumerate(zip(axes,stations)):
 ax.plot(paper[0,i],zp,'k-',lw=1.5,label='Paper: steady speed')
 ax.plot(mean[0,:,i],z,color='#d94b28',lw=1.8,label='Ours: mean axial u, 3–4 s')
 ax.fill_betweenx(z,mean[0,:,i]-std[0,:,i],mean[0,:,i]+std[0,:,i],color='#d94b28',alpha=.15,label='Temporal ±1 std (not CI)')
 ax.plot(speed[:,i],z,'--',color='#2765ac',lw=1,label='Ours: speed at 4 s')
 ax.set(title=f'x = {s:.1f} m',xlabel='Velocity (m/s)',xlim=(0,17),ylim=(-.3,.3));ax.grid(alpha=.25)
axes[0].set_ylabel('Height relative to nozzle (m)');axes[0].legend(loc='lower left',fontsize=7)
fig.suptitle(f'{mesh_label} · {doc["case"].get("carrier_turbulence_model","LES")} versus sim.pdf Figure 9a')
fig.savefig(OUT/'velocity_profiles.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(3,7,figsize=(18,9),sharey=True,layout='constrained')
labels=['Axial u / paper speed (m/s)','DBT (°C)','Vapour mass fraction ×1000']
for k in range(3):
 for i,ax in enumerate(axes[k]):
  ax.plot(paper[k,i],zp,'k-',lw=1.3,label='Paper, Fig. 9')
  ax.plot(mean[k,:,i],z,color='#d94b28',label='Saved 3–4 s mean')
  ax.fill_betweenx(z,mean[k,:,i]-std[k,:,i],mean[k,:,i]+std[k,:,i],color='#d94b28',alpha=.15)
  ax.set(xlabel=labels[k],ylim=(-.3,.3));ax.grid(alpha=.2)
  if k==0:ax.set_title(f'x = {stations[i]:.1f} m')
  if i==0:ax.set_ylabel('z − 0.2925 (m)')
axes[0,0].legend(fontsize=7)
fig.suptitle(f'{mesh_label} · {doc["case"].get("carrier_turbulence_model","LES")} · saved profiles; band = temporal variability')
fig.savefig(OUT/'figure9_profiles.png',dpi=150);plt.close(fig)

def halfwidth(z,v):
 peak=int(np.nanargmax(v));level=3+.5*(v[peak]-3)
 left=np.flatnonzero(v[:peak]<=level);right=np.flatnonzero(v[peak+1:]<=level)
 if not len(left) or not len(right):return None
 l=left[-1];r=peak+1+right[0]
 zl=z[l]+(level-v[l])/(v[l+1]-v[l])*(z[l+1]-z[l]);zr=z[r-1]+(level-v[r-1])/(v[r]-v[r-1])*(z[r]-z[r-1])
 return float((zr-zl)/2)
rows=[]
for i,s in enumerate(stations):
 rows.append({'x_m':float(s),'paper_centre_speed_m_s':float(np.interp(0,zp,paper[0,i])),
  'our_mean_centre_axial_m_s':float(np.interp(0,z,mean[0,:,i])),
  'our_final_centre_speed_m_s':float(np.interp(0,z,speed[:,i])),
  'our_final_centre_axial_m_s':float(np.interp(0,z,axial[:,i])),
  'paper_half_excess_width_m':halfwidth(zp,paper[0,i]),'our_half_excess_width_m':halfwidth(z,mean[0,:,i]),
  'paper_centre_T_C':float(np.interp(0,zp,paper[1,i])),'our_mean_centre_T_C':float(np.interp(0,z,mean[1,:,i])),
  'paper_centre_vapor_mass_fraction':float(np.interp(0,zp,paper[2,i])/1000),
  'our_mean_centre_vapor_mass_fraction':float(np.interp(0,z,mean[2,:,i])/1000)})
summary={'averaging_window_s':[3,4],'saved_frames_in_window':int(sel.sum()),'stations':rows,
 'checkpoint_vs_frame_axial_max_error':float(np.max(abs(axial-instant[0]))),
 'digitization':'Raster Fig. 9 curve extraction; approximate. Three near-wall vapor samples unresolved and left NaN.',
 'limitations':['Saved time series contains axial u, not all velocity components; paper reports speed magnitude.',
 'Blue dashed curve uses full-vector instantaneous speed at t=4 s, not a temporal mean.',
 f'Outlet uses last cell centre x={x[-1]:.9f} m; other stations linearly interpolated.',
 'Half-width uses half peak excess over inlet 3 m/s; paper curves have digitization uncertainty.',
 'Paper is steady RANS CFD, not measured velocity profiles.']}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (OUT/'centreline_comparison.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
np.savez_compressed(OUT/'profiles.npz',z_relative_m=z,stations_m=stations,means=mean,temporal_std=std,final_speed=speed,paper_z_relative_m=zp,paper_profiles=paper)
print(json.dumps(summary,indent=2))

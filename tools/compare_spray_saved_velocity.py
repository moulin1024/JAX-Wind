"""Compare saved axial-velocity profiles across matched cases; no paper overlay.

Vertical centre-plane time means, not full-section momentum budgets. Widths use
half the peak velocity excess above the prescribed inlet speed. The last station
uses the last saved cell centre when the requested outlet is beyond that centre.
"""
import argparse
import json
import tomllib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def width(z,u,background):
    peak=int(np.argmax(u)); level=(u[peak]+background)/2
    left=np.flatnonzero(u[:peak]<=level);right=np.flatnonzero(u[peak+1:]<=level)
    if len(left)==0 or len(right)==0: return None
    i=left[-1];j=peak+1+right[0]
    lo=z[i]+(level-u[i])*(z[i+1]-z[i])/(u[i+1]-u[i])
    hi=z[j-1]+(level-u[j-1])*(z[j]-z[j-1])/(u[j]-u[j-1])
    return float((hi-lo)/2)


def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('runs',nargs='+',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    stations=np.array([.1,.4,.7,1.,1.3,1.6,1.9])
    fig,axes=plt.subplots(1,7,figsize=(17,4.5),sharey=True,layout='constrained')
    results=[];reference=None
    for run in args.runs:
        doc=tomllib.loads((run/'resolved_case.toml').read_text())
        background=doc['physics']['flow']['streamwise_velocity_m_s']
        if reference is None: reference=background
        assert background==reference,'Do not conflate different inlet cases'
        with np.load(run/'flow_frames.npz') as a:
            time=a['time_seconds'];mask=(time>=3-1e-10)&(time<=4+1e-10)
            time=time[mask];np.testing.assert_allclose(time[[0,-1]],[3,4],atol=1e-9)
            frames=a['u_center_zx'][mask];x=a['x_m'];z=a['z_m']-doc['mesh']['lengths_m'][2]/2
        mean=np.trapezoid(frames,time,axis=0)/(time[-1]-time[0])
        std=np.sqrt(np.trapezoid((frames-mean)**2,time,axis=0)/(time[-1]-time[0]))
        profiles=np.stack([np.interp(stations,x,row) for row in mean])
        deviations=np.stack([np.interp(stations,x,row) for row in std])
        model=doc['case'].get('carrier_turbulence_model','les')
        parcel_rate=doc['case'].get('parcels_per_step',0)/doc['time']['dt_seconds']
        mesh='×'.join(map(str,doc['mesh']['cells']))
        inlet='face inlet' if doc['case'].get('physical_transverse_inlet',False) else 'cell inlet'
        label=f"{model}, {inlet}, {mesh}, dt={doc['time']['dt_seconds']:g}, {parcel_rate/1000:g}k parcels/s"
        rows=[]
        for j,(ax,station) in enumerate(zip(axes,stations)):
            u=profiles[:,j]
            line=ax.plot(u,z,label=label)[0]
            ax.fill_betweenx(z,u-deviations[:,j],u+deviations[:,j],alpha=.12,color=line.get_color())
            ax.set(title=f'x={station:g} m',xlabel='Mean axial u (m/s)',ylim=(-.3,.3))
            ax.grid(alpha=.2)
            rows.append({'x_m':float(station),'centre_u_m_s':float(np.interp(0,z,u)), 'peak_u_m_s':float(u.max()),'peak_z_relative_m':float(z[np.argmax(u)]),'half_excess_width_m':width(z,u,background),'centre_temporal_std_m_s':float(np.interp(0,z,deviations[:,j]))})
        results.append({'run':str(run),'model':model,'dt':doc['time']['dt_seconds'],'mesh':doc['mesh']['cells'],'parcel_rate_per_s':parcel_rate,'physical_transverse_inlet':doc['case'].get('physical_transverse_inlet',False),'stations':rows})
    axes[0].set_ylabel('Height relative to nozzle (m)');axes[0].legend(fontsize=7)
    fig.suptitle(f'Inlet {reference:g} m/s; archived 3–4 s means, bands = temporal std (not uncertainty)')
    args.output.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output/'velocity_profiles.png',dpi=180)
    (args.output/'summary.json').write_text(json.dumps({'scope':__doc__,'runs':results},indent=2)+'\n')
    print(json.dumps([{'run':r['run'],'outlet':r['stations'][-1]} for r in results],indent=2))


if __name__=='__main__':
    main()

"""Inspect actual outlet reversal separately from upstream velocity oscillations."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import tomllib
import numpy as np


def analyze(run, baseline=None):
    doc=tomllib.loads((run/'resolved_case.toml').read_text())
    summary=json.loads((run/'summary.json').read_text())
    if summary['status'] != 'complete': raise ValueError('run must be complete')
    history=np.atleast_1d(np.genfromtxt(run/'history.csv',delimiter=',',names=True))
    with np.load(run/'flow_frames.npz') as data:
        fields=data['u_hub_yx'].copy(); times=data['time_seconds'].copy()
        xf=data['x_faces_m'].copy();yf=data['y_faces_m'].copy();zf=data['z_faces_m'].copy()
    if not np.isfinite(fields).all(): raise ValueError('nonfinite velocity frames')
    expected=doc['time']['dt_seconds']*doc['time']['steps']
    np.testing.assert_allclose(times[-1],expected,atol=1e-3)
    assert len(times)==doc['time']['frame_count']
    turbines=doc['physics']['wind_farm']['layout']
    case_label='Single V80' if len(turbines)==1 else 'Horns Rev'
    transport_label=doc['numerics']['momentum_advection_scheme']
    treatment_label='Open horizontal upwind' if transport_label=='central-open-upwind' else 'Backflow pressure'
    if doc['numerics'].get('outlet_sponge_start_fraction') is not None:
        treatment_label='Central + outlet sponge'
    if doc['numerics'].get('upstream_mode_sponge_end_fraction') is not None:
        treatment_label='Central + mode buffer + outlet sponge'
    movie='v80_single_hub_u.mp4' if len(turbines)==1 else 'hornsrev1_hub_u.mp4'
    speed=doc['physics']['inflow']['speed_m_s']
    hub=doc['physics']['turbine']['hub_height_m']
    xc,yc,zc=(.5*(f[1:]+f[:-1]) for f in (xf,yf,zf))
    zidx=int(np.argmin(abs(zc-hub)))
    upstream_stop=min(len(xc),max(3,int((min(t['x_m'] for t in turbines)-160.)/(xf[1]-xf[0]))))
    row=int(np.argmin(abs(yc-turbines[0]['y_m'])))
    with np.load(run/'checkpoint.npz') as data:
        u=data['state/velocity/x']
        raw_hub=u[zidx].copy()
        final_x_out=u[...,-1].copy()
        index=np.unravel_index(np.argmin(u),u.shape)
        min_location=[float(xf[index[2]]),float(yc[index[1]]),float(zc[index[0]])]
        negative_count=int(np.count_nonzero(u<0))
        nearest=min(turbines,key=lambda q:(q['x_m']-min_location[0])**2+(q['y_m']-min_location[1])**2)
        final_negative={'minimum_u_m_s':float(u[index]),'minimum_location_xyz_m':min_location,
            'negative_face_count':negative_count,'total_u_face_count':int(u.size),
            'nearest_turbine_id':nearest['id'],
            'minimum_horizontal_distance_to_turbine_m':float(np.hypot(nearest['x_m']-min_location[0],nearest['y_m']-min_location[1]))}
        del u
        v=data['state/velocity/y']
        final_y_low=-v[:,0].copy();final_y_high=v[:,-1].copy()
        del v
    metrics={
        'run':str(run.resolve()),'duration_seconds':float(times[-1]),'turbine_count':len(turbines),
        'grid_cells_xyz':doc['mesh']['cells'],'uniform_inlet_m_s':speed,'raw_face_height_m':float(zc[zidx]),
        'final_internal_reversal':final_negative,
        'numerics':doc['numerics'],'frame_count':len(times),'history_samples':len(history),
        'sample_interval_seconds':float(np.median(np.diff(history['time_seconds']))) if 'time_seconds' in history.dtype.names else 3.,
        'maximum_cfl':float(np.max(history['maximum_cfl'])),
        'maximum_divergence_s':float(np.max(history['maximum_divergence_s'])),
        'maximum_inlet_error_m_s':float(np.max(history['inlet_maximum_error_m_s'])),
        'minimum_domain_u_m_s':float(np.min(history['domain_minimum_u_m_s'])),
        'minimum_downstream_outlet_u_m_s':float(np.min(history['outlet_x_minimum_u_m_s'])),
        'final_hub_downstream_outlet_minimum_u_m_s':float(final_x_out[zidx].min()),
        'final_hub_downstream_outlet_maximum_u_m_s':float(final_x_out[zidx].max()),
        'maximum_downstream_backflow_area_fraction':float(np.max(history['outlet_x_backflow_area_fraction'])),
        'maximum_downstream_reverse_volume_flux_m3_s':float(np.max(history['outlet_x_reverse_volume_flux_m3_s'])),
        'maximum_lateral_reverse_volume_flux_m3_s':float(np.max(history['outlet_y_reverse_volume_flux_m3_s'])),
        'minimum_low_y_outward_velocity_m_s':float(np.min(history['outlet_y_low_minimum_normal_m_s'])),
        'minimum_high_y_outward_velocity_m_s':float(np.min(history['outlet_y_high_minimum_normal_m_s'])),
        'maximum_low_y_backflow_area_fraction_below_001':float(np.max(history['outlet_y_low_backflow_area_fraction_below_001'])),
        'maximum_high_y_backflow_area_fraction_below_001':float(np.max(history['outlet_y_high_backflow_area_fraction_below_001'])),
        'maximum_net_boundary_flux_m3_s':float(np.max(abs(history['boundary_net_volume_flux_m3_s']))),
        'final_upstream_second_difference_rms_m_s':float(history['upstream_hub_second_difference_rms_m_s'][-1]),
        'maximum_upstream_second_difference_rms_m_s':float(np.max(history['upstream_hub_second_difference_rms_m_s'])),
        'final_upstream_minimum_u_m_s':float(history['upstream_hub_minimum_u_m_s'][-1]),
        'final_upstream_maximum_u_m_s':float(history['upstream_hub_maximum_u_m_s'][-1]),
        'final_raw_upstream_maximum_adjacent_jump_m_s':float(np.max(abs(np.diff(raw_hub[:,:upstream_stop],axis=1)))),
        'final_farm_lookup_power_mw':float(history['farm_lookup_power_w'][-1]/1e6),
        'upstream_region_x_max_m':float(xf[upstream_stop-1]),
        'last_row_to_outlet_advection_time_at_inlet_speed_s':float((xf[-1]-max(t['x_m'] for t in turbines))/speed),
        'note':'Boundary reversal is measured from outward-normal face velocities. Negative lateral normal velocity is entrainment, not proof of streamwise wake wraparound. Samples do not rule out shorter events between outputs. Upstream second differences include physical gradients as well as numerical oscillations.'}
    inlet_flux=speed*(yf[-1]-yf[0])*(zf[-1]-zf[0])
    metrics['maximum_net_boundary_flux_relative_to_inlet']=metrics['maximum_net_boundary_flux_m3_s']/inlet_flux
    previous=None
    if baseline is not None:
        previous=json.loads((baseline/'backflow_analysis/metrics.json').read_text())
        metrics['baseline']=str(baseline)
        metrics['baseline_metrics']={k:previous[k] for k in (
            'minimum_downstream_outlet_u_m_s','maximum_lateral_reverse_volume_flux_m3_s',
            'minimum_low_y_outward_velocity_m_s','minimum_high_y_outward_velocity_m_s',
            'final_upstream_second_difference_rms_m_s','final_raw_upstream_maximum_adjacent_jump_m_s')}
    output=run/'backflow_analysis';output.mkdir(exist_ok=True)
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(output/'final_outlet_faces.npz',x_high_u=final_x_out,y_low_outward_v=final_y_low,y_high_outward_v=final_y_high)
    np.savez_compressed(output/'upstream_raw_faces.npz',u=raw_hub[:,:upstream_stop],x=xf[:upstream_stop],y=yc,row_index=row)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    fig,axes=plt.subplots(2,3,figsize=(16,8.6),layout='constrained')
    image=axes[0,0].pcolormesh(xf/1000,yf/1000,fields[-1],vmin=0,vmax=12,cmap='viridis')
    segments=[[(t['x_m']/1000,(t['y_m']-40)/1000),(t['x_m']/1000,(t['y_m']+40)/1000)] for t in turbines]
    axes[0,0].add_collection(LineCollection(segments,colors='k',linewidths=.7))
    axes[0,0].set(title=f'{len(turbines)}-turbine wake at {times[-1]:g} s',xlabel='x [km]',ylabel='y [km]',aspect='equal')
    fig.colorbar(image,ax=axes[0,0],label='Hub-height u [m/s]')
    anomaly=fields[-1,:,:upstream_stop]-speed
    limit=max(.05,float(np.max(abs(anomaly))))
    image=axes[0,1].pcolormesh(xf[:upstream_stop+1],yf/1000,anomaly,cmap='RdBu_r',vmin=-limit,vmax=limit)
    axes[0,1].set(title='Upstream velocity departures',xlabel='x [m]',ylabel='y [km]')
    fig.colorbar(image,ax=axes[0,1],label='u − 10 [m/s]')
    axes[0,2].plot(xf[:upstream_stop],raw_hub[row,:upstream_stop]-speed,'.-',ms=3)
    axes[0,2].set(title=f'Raw upstream faces at y={yc[row]:g} m',xlabel='x [m]',ylabel='u − 10 [m/s]')
    t=history['time_seconds'] if 'time_seconds' in history.dtype.names else history['time_hours']*3600
    axes[1,0].plot(t,history['outlet_x_minimum_u_m_s'],label='Downstream outlet')
    axes[1,0].plot(t,history['domain_minimum_u_m_s'],label='Entire domain')
    axes[1,0].axhline(0,color='k',ls=':');axes[1,0].legend(fontsize=9)
    axes[1,0].set(title='Actual streamwise reversal',xlabel='Time [s]',ylabel='Minimum u [m/s]')
    for key,label in [('outlet_x_backflow_area_fraction_below_001','Downstream'),('outlet_y_low_backflow_area_fraction_below_001','Low-y side'),('outlet_y_high_backflow_area_fraction_below_001','High-y side')]:
        axes[1,1].plot(t,100*history[key],label=label)
    axes[1,1].legend(fontsize=9)
    axes[1,1].set(title='Boundary area with inward speed >0.01 m/s',xlabel='Time [s]',ylabel='Area [%]')
    axes[1,2].plot(t,history['upstream_hub_second_difference_rms_m_s'])
    axes[1,2].set(title='Upstream grid-scale variation',xlabel='Time [s]',ylabel='Second-difference RMS [m/s]')
    for ax in axes.flat:ax.grid(alpha=.15)
    fig.suptitle(f"{case_label} · uniform 10 m/s · {transport_label}/RK3 · outlet treatment: {doc['numerics'].get('outlet_backflow','none')}")
    fig.savefig(output/'backflow_summary.png',dpi=170);fig.savefig(output/'backflow_summary.pdf');plt.close(fig)
    text=f'''# {case_label} five-minute backflow check

Completed {times[-1]:g} s with {len(turbines)} turbines, {transport_label}/RK3, and `{doc['numerics'].get('outlet_backflow','none')}` outlet backflow treatment. Uniform inlet: {speed:g} m/s. Grid: {doc['mesh']['cells']}. Fixed dt: {doc['time']['dt_seconds']:g} s.

- Minimum sampled downstream-outlet u: {metrics['minimum_downstream_outlet_u_m_s']:.6g} m/s.
- Maximum downstream reverse-flow area: {100*metrics['maximum_downstream_backflow_area_fraction']:.6g}%.
- Minimum sampled low/high-y outward velocity: {metrics['minimum_low_y_outward_velocity_m_s']:.6g}, {metrics['minimum_high_y_outward_velocity_m_s']:.6g} m/s.
- Maximum lateral inward volume flux: {metrics['maximum_lateral_reverse_volume_flux_m3_s']:.6g} m³/s.
- Final upstream raw-face adjacent jump: {metrics['final_raw_upstream_maximum_adjacent_jump_m_s']:.6g} m/s.
- Final upstream second-difference RMS: {metrics['final_upstream_second_difference_rms_m_s']:.6g} m/s.
- Maximum divergence: {metrics['maximum_divergence_s']:.6g} /s; maximum CFL: {metrics['maximum_cfl']:.6g}.
- Maximum inlet error: {metrics['maximum_inlet_error_m_s']:.6g} m/s.

Pressure-based stabilization permits physical entrainment. It is not a no-reverse-flow constraint. Three-second boundary sampling cannot exclude shorter events; the 100 saved frames show the evolution. Upstream departures include pressure induction and numerical dispersion. A single treated run cannot establish how much the treatment changed them.

[Summary figure](backflow_summary.png) · [Metrics](metrics.json) · [Wake animation](../{movie})
'''
    if baseline is not None:
        prior=np.atleast_1d(np.genfromtxt(baseline/'history.csv',delimiter=',',names=True))
        pt=prior['time_hours']*3600.
        fig,axes=plt.subplots(2,3,figsize=(16,8),layout='constrained')
        specifications=(('outlet_x_minimum_u_m_s','Downstream outlet minimum u [m/s]'),
                        ('outlet_y_reverse_volume_flux_m3_s','Lateral inward volume flux [m³/s]'),
                        ('upstream_hub_second_difference_rms_m_s','Upstream second-difference RMS [m/s]'),
                        ('domain_minimum_u_m_s','Domain minimum u [m/s]'),
                        ('outlet_y_low_minimum_normal_m_s','Low-y minimum outward velocity [m/s]'),
                        ('outlet_y_high_minimum_normal_m_s','High-y minimum outward velocity [m/s]'))
        for ax,(key,label) in zip(axes.flat,specifications):
            ax.plot(pt,prior[key],label='Ordinary pressure outlets',color='#888888')
            ax.plot(t,history[key],label=treatment_label,color='#126782')
            ax.set(xlabel='Time [s]',ylabel=label);ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=9)
        fig.suptitle(f'Matched {case_label} comparison · {len(turbines)} turbines · {transport_label}/RK3 · uniform {speed:g} m/s')
        fig.savefig(output/'comparison_with_baseline.png',dpi=170)
        fig.savefig(output/'comparison_with_baseline.pdf');plt.close(fig)
        text=text.replace('A single treated run cannot establish how much the treatment changed them.',
            'A matched ordinary-outlet run uses the same initial flow, central/RK3, grid, turbines, timestep, and output schedule.')
        text+='\n## Matched outlet comparison\n\n| Metric | Ordinary outlets | Backflow treatment |\n|---|---:|---:|\n'
        for label,key in [('Minimum downstream u [m/s]','minimum_downstream_outlet_u_m_s'),
                          ('Maximum lateral inward flux [m³/s]','maximum_lateral_reverse_volume_flux_m3_s'),
                          ('Minimum low-y outward velocity [m/s]','minimum_low_y_outward_velocity_m_s'),
                          ('Minimum high-y outward velocity [m/s]','minimum_high_y_outward_velocity_m_s'),
                          ('Final upstream second-difference RMS [m/s]','final_upstream_second_difference_rms_m_s'),
                          ('Final raw upstream adjacent jump [m/s]','final_raw_upstream_maximum_adjacent_jump_m_s')]:
            text+=f"| {label} | {previous[key]:.6g} | {metrics[key]:.6g} |\n"
        with np.load(baseline/'backflow_analysis/upstream_raw_faces.npz') as data:
            prior_raw=data['u'].copy()
            np.testing.assert_array_equal(data['x'],xf[:upstream_stop])
            np.testing.assert_array_equal(data['y'],yc)
        current_raw=raw_hub[:,:upstream_stop]
        lim=max(.05,float(np.max(abs(prior_raw-speed))),float(np.max(abs(current_raw-speed))))
        fig,axes=plt.subplots(1,3,figsize=(14,5),layout='constrained')
        for ax,field,label in zip(axes[:2],(prior_raw,current_raw),('Ordinary pressure outlets',treatment_label)):
            mesh=ax.pcolormesh(xf[:upstream_stop],yc/1000,field-speed,shading='nearest',cmap='RdBu_r',vmin=-lim,vmax=lim)
            ax.set(title=label,xlabel='Upstream x [m]',ylabel='y [km]')
        fig.colorbar(mesh,ax=list(axes[:2]),label='Raw face u − 10 [m/s]')
        axes[2].plot(xf[:upstream_stop],prior_raw[row]-speed,label='Ordinary outlets',color='#888888')
        axes[2].plot(xf[:upstream_stop],current_raw[row]-speed,label=treatment_label,color='#126782')
        axes[2].set(title=f'Row at y={yc[row]:g} m',xlabel='Upstream x [m]',ylabel='Raw face u − 10 [m/s]')
        axes[2].legend(fontsize=9);axes[2].grid(alpha=.2)
        fig.suptitle(f'Upstream flow at t={times[-1]:g} s · raw u faces at z={zc[zidx]:g} m')
        fig.savefig(output/'upstream_comparison.png',dpi=170)
        fig.savefig(output/'upstream_comparison.pdf');plt.close(fig)
        with np.load(baseline/'backflow_analysis/final_outlet_faces.npz') as data:
            prior_sides=(data['y_low_outward_v'].copy(),data['y_high_outward_v'].copy())
        fig,axes=plt.subplots(2,1,figsize=(12,6),layout='constrained')
        for ax,old,new,label in zip(axes,prior_sides,(final_y_low,final_y_high),('Low-y side','High-y side')):
            ax.plot(xc,old[0],label='Ordinary pressure outlet',color='#888888',lw=1)
            ax.plot(xc,new[0],label=treatment_label,color='#126782',lw=.8)
            ax.axhline(0,color='k',ls=':',lw=.7)
            ax.set(title=label,xlabel='x [m]',ylabel='Outward normal velocity [m/s]');ax.grid(alpha=.2)
        axes[0].legend(fontsize=9)
        fig.suptitle(f'Lateral boundary velocities at z={zc[0]:g} m, t={times[-1]:g} s')
        fig.savefig(output/'lateral_near_wall_comparison.png',dpi=170)
        fig.savefig(output/'lateral_near_wall_comparison.pdf');plt.close(fig)
        text+='\n[Matched comparison figure](comparison_with_baseline.png) · [Raw upstream comparison](upstream_comparison.png) · [Near-wall lateral boundaries](lateral_near_wall_comparison.png)\n'
    (output/'README.md').write_text(text)
    print(json.dumps(metrics,indent=2))
    return metrics

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--baseline',type=Path)
    args=p.parse_args();analyze(args.run,args.baseline)

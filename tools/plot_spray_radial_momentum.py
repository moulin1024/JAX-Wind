"""Postprocess finite-window radial transport and digitized paper profiles."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def digitize(path):
    a=np.asarray(Image.open(path).convert('L'));xframes=np.array([92,379,667,954,1242,1529,1817,2103])
    bounds=[(81.5,469),(633.5,1022.5),(1182.5,1571.5)]
    limits=[(0,12),(28,40),(4,12)]
    z=np.linspace(-.285,.285,115);result=np.zeros((3,7,len(z)))
    for k,((top,bottom),(vmin,vmax)) in enumerate(zip(bounds,limits)):
        for s in range(7):
            lo=xframes[s]+6;hi=xframes[s+1]-6
            for i,zz in enumerate(z):
                row=round(top+(.3-zz)/.6*(bottom-top))
                ys,xs=np.where(a[row-1:row+2,lo:hi]<120)
                if len(xs)==0:
                    result[k,s,i]=np.nan
                    continue
                result[k,s,i]=vmin+(np.median(xs)+lo-xframes[s])/(xframes[s+1]-xframes[s])*(vmax-vmin)
    return z,result


def radial_terms(fields,names,y,z,rho):
    f=dict(zip(names,fields));yy=y[None,:,None]-.2925;zz=z[:,None,None]-.2925
    r=np.hypot(yy,zz);ny=yy/r;nz=zz/r
    U,V,W=f['u'],f['v'],f['w'];vr=V*ny+W*nz
    return {'mean_advection':rho*U*vr,
        'resolved_turbulence':rho*((f['uv']-U*V)*ny+(f['uw']-U*W)*nz),
        'SGS':-rho*(f['sgs_xy']*ny+f['sgs_xz']*nz),
        'molecular':-rho*(f['molecular_xy']*ny+f['molecular_xz']*nz)},r[:,:,0]


def annular_average(a,r,edges):
    return np.array([np.mean(a[(r>=lo)&(r<hi)],axis=0) for lo,hi in zip(edges[:-1],edges[1:])])


def main():
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('directory',type=Path);ap.add_argument('--run-directory',type=Path);args=ap.parse_args();base=args.directory
    run=args.run_directory or base/'continuation'
    data=np.load(run/'statistics.npz');blocks=data['block_fields'];fields=blocks.mean(axis=0)
    names=list(data['names']);f=dict(zip(names,fields));y,z=data['y_m'],data['z_m'];stations=data['stations_m'];rho=1.125
    times=data['time'];window=[float(times[0]),float(times[-1])];out=run/'analysis';out.mkdir(exist_ok=True)
    zp,paper=digitize(base/'figure9_original.jpg');np.savez(out/'digitized_figure9.npz',z_relative_m=zp,profiles=paper,stations_m=stations)
    # Pointwise means first. Middle y-plane lies halfway between two cells.
    centre=lambda a:(a[:,len(y)//2-1,:]+a[:,len(y)//2,:])*.5
    profiles=np.stack([centre(f['speed']),centre(f['T_C']),centre(f['Y_vapor'])*1000])
    fig,axes=plt.subplots(3,7,figsize=(18,8),sharey=True,layout='constrained')
    labels=['Speed (m/s)','DBT (°C)','Vapour mass fraction ×1000']
    for k in range(3):
        for j,x in enumerate(stations):
            ax=axes[k,j];ax.plot(paper[k,j],zp,'k--',lw=1.3,label='Paper Fig. 9');ax.plot(profiles[k,:,j],z-.2925,color='#d94b28',label=f'Ours {window[0]:g}–{window[1]:g} s')
            for b in blocks:
                bf=dict(zip(names,b));p=centre(bf[['speed','T_C','Y_vapor'][k]])[:,j]*(1000 if k==2 else 1)
                ax.plot(p,z-.2925,color='#d94b28',alpha=.25,lw=.7)
            ax.set(ylim=(-.3,.3),xlabel=labels[k]);ax.grid(alpha=.2)
            if k==0:ax.set_title(f'x = {x:.1f} m')
            if j==0:ax.set_ylabel('z − 0.2925 (m)')
    axes[0,0].legend(fontsize=7);fig.suptitle('Uniform 256 × 128 × 128 versus published steady RANS; thin orange = temporal half-windows')
    fig.savefig(out/'figure9_comparison.png',dpi=150);plt.close(fig)
    terms,r=radial_terms(fields,names,y,z,rho);edges=np.linspace(0,.28,max(2,int(.28/max(float(y[1]-y[0]),float(z[1]-z[0]),.01)))+1);rc=.5*(edges[:-1]+edges[1:]);radial={k:annular_average(v,r,edges) for k,v in terms.items()}
    fig,axes=plt.subplots(2,4,figsize=(16,8),layout='constrained')
    colors={'mean_advection':'#3366aa','resolved_turbulence':'#dd7722','SGS':'#339955','molecular':'#777777'}
    for j,ax in enumerate(axes.flat):
        if j==7:ax.axis('off');continue
        for k,v in radial.items():ax.plot(rc,v[:,j],label=k.replace('_',' '),color=colors[k])
        ax.axhline(0,color='k',lw=.5);ax.set(title=f'x = {stations[j]:.1f} m',xlabel='Radius from axis (m)',ylabel='Outward axial momentum flux (Pa)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('Radial transport decomposition; positive = outward; temporal covariance before annular averaging')
    fig.savefig(out/'radial_momentum_flux.png',dpi=150);plt.close(fig)
    rings=data['block_rings'].mean(axis=0);redges=data['radius_edges_m'];rx=.5*(redges[:-1]+redges[1:]);x=data['x_m'];dx=x[1]-x[0]
    totals=rings.sum(axis=(1,2))*dx
    fig,axes=plt.subplots(1,3,figsize=(16,4.7),layout='constrained')
    # Ring-integrated force / dx / dr: N/m², not volumetric force density.
    im=axes[0].pcolormesh(x,rx,rings[0]/np.diff(redges)[:,None],shading='nearest',cmap='coolwarm');fig.colorbar(im,ax=axes[0],label='Drag d²F/(dx dr) (N/m²)')
    axes[0].set(xlabel='x (m)',ylabel='Radius (m)',title='Where droplet drag deposits axial momentum')
    for j,lab in enumerate(['Drag','Evaporated-mass momentum']):axes[1].plot(x,rings[j].sum(axis=0),label=lab)
    axes[1].set(xlabel='x (m)',ylabel='dF/dx (N/m)',title='All radii, including wall-contact parcels');axes[1].legend()
    for cutoff in [.05,.1,.2,.42]:axes[2].plot(x,np.cumsum(rings[0,rx<cutoff].sum(axis=0))*dx,label=f'r < {cutoff:g} m')
    axes[2].set(xlabel='x (m)',ylabel='Cumulative drag force (N)',title='Integrated source');axes[2].legend()
    fig.savefig(out/'parcel_momentum_deposition.png',dpi=150);plt.close(fig)
    block_rad=[]
    for b in blocks:
        bt,_=radial_terms(b,names,y,z,rho);block_rad.append({k:annular_average(v,r,edges) for k,v in bt.items()})
    summaries=[]
    for j,s in enumerate(stations):
        U=annular_average(f['u'],r,edges)[:,j];core=U[0];target=3+.5*(core-3);eligible=np.flatnonzero(U<=target);half=float(rc[eligible[0]]) if len(eligible) else None
        pick=int(np.argmin(abs(rc-(half or .1))))
        summaries.append({'x_m':float(s),'centre_speed_m_s':float(np.interp(0,z-.2925,profiles[0,:,j])),
            'paper_centre_speed_m_s':float(np.interp(0,zp,paper[0,j])),
            'radial_half_excess_speed_radius_m':half,'flux_radius_m':float(rc[pick]),
            'radial_flux_Pa_at_radius':{k:float(v[pick,j]) for k,v in radial.items()},
            'half_window_fluxes_Pa_at_radius':[{k:float(v[pick,j]) for k,v in br.items()} for br in block_rad],
            'mean_SGS_viscosity_at_radius_m2_s':float(annular_average(f['nu_sgs'],r,edges)[pick,j])})
    summary={'window_seconds':window,'completed_blocks':len(blocks),'stations':summaries,'integrated_drag_force_N':float(totals[0]),'integrated_evaporation_momentum_force_N':float(totals[1]),
        'drag_fraction_x_lt_0p1':float(rings[0,:,x<.1].sum()*dx/totals[0]),'drag_fraction_x_lt_0p4':float(rings[0,:,x<.4].sum()*dx/totals[0]),
        'drag_fraction_r_lt_0p05':float(rings[0,rx<.05].sum()*dx/totals[0]),
        'maximum_deposition_impulse_residual':float(np.abs(data['source_residual_kg_m_s']).max()),
        'limitations':['Paper curves digitized from raster, approximate; raw Figure 9 retained.','Radial profiles average complete annuli only, radius <0.28 m.',
        'Not a full discrete budget closure; numerical advection, axial transport, pressure, wall fluxes and storage not balanced.',
        'Finite-window LES comparison to steady RANS; half-window changes are not confidence intervals.']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (out/'radial_profiles.csv').open('w') as stream:
        writer=csv.writer(stream);writer.writerow(['x_m','radius_m',*radial.keys()])
        for j,s in enumerate(stations):
            for i,rr in enumerate(rc):writer.writerow([s,rr,*[v[i,j] for v in radial.values()]])
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()

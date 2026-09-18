"""Approximate paired diameter/temperature digitization from local Fig. 11 raster.

Coordinates apply only to the stored page rendered at 1.5 pixels/PDF point.
Uses corresponding spatial pixels in the top/bottom panels and their colorbars.
Color antialiasing, scatter overlap and panel registration limit accuracy; these
are raster color estimates, not original droplet records or flux-weighted means.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def decode(colors, palette, values):
    delta=np.sum((colors[:,None,:].astype(float)-palette[None,:,:])**2,axis=2)
    index=delta.argmin(axis=1)
    return values[index],np.sqrt(delta[np.arange(len(colors)),index])


def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('image',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--registration-shift',type=int,default=0,choices=[-2,0,2])
    args=ap.parse_args()
    a=np.asarray(Image.open(args.image).convert('RGB'))
    assert a.shape[:2]==(1191,893),a.shape
    tp=a[120:299,176:180].mean(axis=1)
    dp=a[328:507,176:180].mean(axis=1)
    tv=np.linspace(35,20,len(tp)); dv=np.linspace(518,78,len(dp))
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True,layout='constrained')
    reports=[]
    for ax,(lo,hi),fraction in zip(axes,[(194,373),(380,559),(567,746)],[.25,.5,.75]):
        shift=args.registration_shift
        tcolors=a[120+shift:299+shift,lo+shift:hi+shift].reshape(-1,3)
        dcolors=a[328:507,lo:hi].reshape(-1,3)
        t,terr=decode(tcolors,tp,tv); d,derr=decode(dcolors,dp,dv)
        valid=(terr<25)&(derr<25)&(np.ptp(tcolors.astype(float),axis=1)>80)&(np.ptp(dcolors.astype(float),axis=1)>80)
        ax.scatter(d[valid],t[valid],s=1,alpha=.15)
        ax.set(title=f'x/L = {fraction}',xlabel='Raster diameter (µm)',xlim=(70,520),ylim=(19,36))
        ax.grid(alpha=.2)
        bins=[]
        for lower,upper in [(70,150),(150,250),(250,350),(350,450),(450,520)]:
            m=valid&(d>=lower)&(d<upper)
            if m.any(): bins.append({'diameter_um':[lower,upper],'pixels':int(m.sum()),'T_percentiles_5_50_95_C':np.percentile(t[m],[5,50,95]).tolist()})
        reports.append({'x_over_L':fraction,'bins':bins})
    axes[0].set_ylabel('Raster temperature (°C)')
    args.output.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output/'figure11_color_pairs.png',dpi=160)
    (args.output/'summary.json').write_text(json.dumps({'scope':__doc__,'registration_shift_pixels':args.registration_shift,'sections':reports},indent=2)+'\n')
    print(json.dumps(reports,indent=2))


if __name__=='__main__':
    main()

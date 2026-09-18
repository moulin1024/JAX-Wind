"""Compare paper Fig10 diameter moments with the passive 20-size cohort.

Paper raster readings are approximate. Native moments use conserved physical
number weights at first crossings; no occupancy or image-pixel mass weighting.
"""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(__doc__);p.add_argument('job',type=Path);p.add_argument('--figure',type=Path,required=True);a=p.parse_args()
im=np.asarray(Image.open(a.figure)).astype(float);xx=np.array([955+64.4*k for k in range(11)]);yguess=np.array([118,146,185,228,273,321,369,420,469,516,559]);D10=[];D32=[]
for k,x in enumerate(xx):
 ix=int(round(x));region=im[max(0,yguess[k]-14):yguess[k]+15,ix-2:ix+3];dark=np.max(region,axis=2)<65;ys=np.where(dark)[0]+yguess[k]-14;assert len(ys)>0;D10.append(199-(np.median(ys)-57.5)*5/(631-57.5))
 region=im[70:615,ix-2:ix+3];orange=(region[...,0]>120)&(region[...,0]-region[...,1]>45)&(region[...,1]-region[...,2]>25);ys=np.where(orange)[0]+70;assert len(ys)>0;D32.append(293-(np.median(ys)-57.5)*2/(631-57.5))
f=np.load(a.job/'result/q20_dt3.125e-05.npz');d0=f['initial_diameter_m'];weights=f['initial_mass_weights'];N=weights/d0**3;fields=list(f['fields']);loss=np.vstack([np.zeros(len(d0)),f['records'][:,fields.index('mass_loss_fraction')]]);diam=d0[None,:]*(1-loss)**(1/3);m10=np.sum(N*diam,axis=1)/sum(N)*1e6;m32=np.sum(N*diam**3,axis=1)/np.sum(N*diam**2,axis=1)*1e6
x=np.r_[0,f['stations_m']];px=np.arange(11)*.19;report={'scope':__doc__,'graph_reading_allowance_um':{'D10':.04,'D32':.02},'paper':{'x_m':px.tolist(),'D10_um':D10,'D32_um':D32},'cohort':{'x_m':x.tolist(),'D10_um':m10.tolist(),'D32_um':m32.tolist()}}
(a.job/'diameter_moments.json').write_text(json.dumps(report,indent=2)+'\n')
fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for col,(label,ref,got) in enumerate([('D10',np.array(D10),m10),('D32',np.array(D32),m32)]):
 for row in range(2):
  ax=axes[row,col];offset_ref=ref[0] if row else 0;offset_got=got[0] if row else 0
  ax.plot(px,ref-offset_ref,'s-',label='Paper CFD (raster)');ax.plot(x,got-offset_got,'o-',label='Native 20-size passive cohort');ax.set_xlim(0,.95);ax.set_xlabel('Distance from nozzle (m)');ax.set_ylabel(('Change from own inlet ' if row else '')+label+' (µm)');ax.grid(alpha=.25);ax.legend(fontsize=8)
fig.suptitle('Diameter moments: initial distribution versus downstream depletion')
fig.savefig(a.job/'diameter_moments.png',dpi=180);plt.close(fig)
print('paper inlet/midpoint:',D10[0],D10[5],D32[0],D32[5]);print('native inlet/midpoint:',m10[0],m10[-1],m32[0],m32[-1])

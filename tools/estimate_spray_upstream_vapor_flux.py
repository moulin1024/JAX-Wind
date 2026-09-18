"""Axisymmetric upstream excess-vapor flux estimates from saved Fig9 profiles.

Only x=.1 and .4m before wall interaction. Paper speed is used as axial speed;
vertical upper/lower half profiles are separate symmetry sensitivity controls.
Not a closed experimental or square-duct mass budget. Local far-field vapor
baseline is subtracted to avoid integrating raster baseline offsets over area.
"""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(__doc__);p.add_argument('profiles',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
f=np.load(a.profiles);results=[]
for label,z,values in [('paper',f['paper_z_relative_m'],f['paper_profiles']),('solver',f['z_relative_m'],f['means'].transpose(0,2,1))]:
 order=np.argsort(z);z=z[order];values=values[...,order]
 for j,station in enumerate(f['stations_m'][:2]):
  U=values[0,j];Y=values[2,j]/1000
  finite=np.isfinite(U)&np.isfinite(Y)
  zz=z[finite];uu=U[finite];yy=Y[finite]
  outer=(abs(zz)>.20)&(abs(zz)<.26)
  background=float(np.median(yy[outer]))
  rows=[]
  for radius in [.15,.20,.25]:
   r=np.linspace(0,radius,2001)
   for side in [-1,1]:
    u=np.interp(side*r,zz,uu);y=np.interp(side*r,zz,yy)
    excess_q=y/(1-y)-background/(1-background)
    flux=1.125*2*np.pi*np.trapezoid(u*excess_q*r,r)
    rows.append({'radius_m':radius,'side':side,'excess_vapor_kg_s':float(flux)})
  results.append({'source':label,'x_m':float(station),'far_field_Y':background,'centre_Y':float(np.interp(0,zz,yy)),'estimates':rows})
report={'scope':__doc__,'assumed_dry_density_kg_m3':1.125,'profiles':results}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

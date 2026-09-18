"""Generate conditional axial-speed surrogate fields and run passive cohorts.

Paper speed is treated as axial speed, while native transverse velocity and
thermal fields are retained. These hybrids are not solenoidal carrier solutions
or reconstructions of the full paper velocity vector. Native-profile controls
use speed too, exposing component-choice and mapping effects.
"""
import argparse,json,subprocess,sys,tomllib
from pathlib import Path
import numpy as np


def main():
 p=argparse.ArgumentParser(__doc__);p.add_argument('run',type=Path);p.add_argument('--profiles',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 doc=tomllib.loads((a.run/'resolved_case.toml').read_text());nx,ny,nz=doc['mesh']['cells'];lx,ly,lz=doc['mesh']['lengths_m'];xc=(np.arange(nx)+.5)*lx/nx;yc=(np.arange(ny)+.5)*ly/ny;zc=(np.arange(nz)+.5)*lz/nz
 with np.load(a.run/'checkpoint.npz') as f:
  uv=[]
  for c,axis in zip('xyz',[2,1,0]):
   v=f['state/velocity/'+c];uv.append((np.take(v,range(v.shape[axis]-1),axis)+np.take(v,range(1,v.shape[axis]),axis))/2)
  original=np.stack([*uv,f['state/scalar']+doc['physics']['moisture']['temperature_offset_k'],f['state/moisture/vapor']])
 with np.load(a.profiles) as f:xp=np.r_[0,f['stations_m'][:4]];pz=f['paper_z_relative_m'];paper=f['paper_profiles'][0,:4]
 assert np.isfinite(paper).all();native_speed=np.sqrt(np.sum(original[:3]**2,axis=0));center=native_speed[:,ny//2-1:ny//2+1,:].mean(1);native=np.array([np.interp(xp[1:],xc,row) for row in center]).T
 radius=np.hypot(zc[:,None]-lz/2,yc[None,:]-ly/2);variants={'native_3d':original.copy()};metadata={}
 for name,profiles,z in [('native_speed',native,zc-lz/2),('paper_speed',paper,pz)]:
  order=np.argsort(z);z=z[order];profiles=profiles[:,order]
  radial=np.array([(.5*(np.interp(radius.ravel(),z,row)+np.interp(-radius.ravel(),z,row))).reshape(nz,ny) for row in profiles])
  for bridge in ['linear','flat']:
   inlet=np.full((1,nz,ny),doc['physics']['flow']['streamwise_velocity_m_s']) if bridge=='linear' else radial[:1]
   table=np.concatenate([inlet,radial]);reconstructed=np.array([np.interp(xc,xp,row) for row in table.reshape(5,-1).T]).reshape(nz,ny,nx)
   data=original.copy();data[0]=reconstructed;key=name+'_'+bridge;variants[key]=data
   metadata[key]={'max_abs_axial_change_m_s':float(np.max(abs(data[0]-original[0]))),'scope':'Only axial gas velocity changed; upper/lower radial profiles averaged, endpoint clamping explicit. Paper speed as axial is not the measured vector.'}
 reports=[]
 for name,field in variants.items():
  archive=a.output/(name+'_fields.npz');np.savez_compressed(archive,fields=field)
  result=a.output/name
  subprocess.run([sys.executable,str(Path(__file__).with_name('audit_spray_upstream_cohort.py')),str(a.run),'--output',str(result),'--diameter-rule','linear-point-density','--carrier-fields',str(archive)],check=True)
  reports.append({'name':name,'result':str(result/'summary.json')});print('finished field',name,flush=True)
 (a.output/'summary.json').write_text(json.dumps({'scope':__doc__,'metadata':metadata,'variants':reports},indent=2)+'\n')
if __name__=='__main__':main()

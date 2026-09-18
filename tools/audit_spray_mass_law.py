"""Compare instantaneous Stefan and Fluent-12 concentration evaporation rates.

Frozen saved gas and parcel states; identical transport properties and heat law.
This isolates the driving-force formula, not a coupled alternate-model result.
Reference: https://www.afs.enea.it/project/neptunius/docs/fluent/html/th/node253.htm
"""
import argparse
import json
import tomllib
from pathlib import Path
import numpy as np
from scipy.ndimage import map_coordinates


def saturation_pressure(t):
    logt = np.log(t)
    return np.exp(54.842763 - 6763.22/t - 4.210*logt + .000367*t
                  + np.tanh(.0415*(t-218.8))*(53.878-1331.22/t-9.44523*logt+.014025*t))


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('run', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    doc = tomllib.loads((args.run/'resolved_case.toml').read_text())
    with np.load(args.run/'checkpoint.npz') as a:
        active = a['state/parcels/active']
        p = {k: a['state/parcels/'+k][..., active] for k in ['position','velocity','mass','temperature','multiplicity']}
        lengths = np.array(doc['mesh']['lengths_m'])
        cells = np.array(doc['mesh']['cells'])
        coordinates = (p['position']/(lengths/cells)[:,None]-.5)[::-1]
        def sample(field):
            return map_coordinates(field, coordinates, order=1, mode='nearest', prefilter=False)
        tg = sample(a['state/scalar']) + doc['physics']['moisture']['temperature_offset_k']
        q = sample(a['state/moisture/vapor'])
        u=[]
        for component, axis in zip('xyz', [2,1,0]):
            v=a['state/velocity/'+component]
            u.append(sample((np.take(v,range(v.shape[axis]-1),axis)+np.take(v,range(1,v.shape[axis]),axis))/2))
    td=p['temperature']; mass=p['mass']; number=p['multiplicity']
    rho=doc['physics']['moisture']['dry_air_density_kg_m3']
    pressure=doc['physics']['moisture']['pressure_pa']
    d=np.cbrt(6*mass/(np.pi*997))
    slip=np.linalg.norm(p['velocity']-np.asarray(u),axis=0)
    re=rho*slip*d/1.9e-5
    sh=2+.6*np.sqrt(re)*(1.9e-5/(rho*2.5e-5))**(1/3)
    ps=saturation_pressure(td); eps=287.05/461.5
    qs=eps*ps/(pressure-ps)
    stefan=np.pi*d*2.5e-5*sh*rho*np.maximum(np.log1p(qs)-np.log1p(q),0)
    concentration=np.pi*d*2.5e-5*sh*np.maximum(ps/(461.5*td)-pressure*q/(q+eps)/(461.5*tg),0)
    groups={}
    for name,mask in [('all',np.ones(len(mass),bool)),('upstream_half',p['position'][0]<lengths[0]/2),('large_upstream', (d>450e-6)&(p['position'][0]<lengths[0]/2))]:
        s=float(np.sum(stefan[mask]*number[mask])); c=float(np.sum(concentration[mask]*number[mask]))
        groups[name]={'stefan_evaporation_kg_s':s,'concentration_evaporation_kg_s':c,'relative_change':c/s-1,'parcels':int(mask.sum())}
    result={'run':str(args.run),'scope':__doc__,'groups':groups}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()

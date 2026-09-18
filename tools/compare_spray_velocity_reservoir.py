"""Verify and summarize passive native/paper velocity-surrogate cohorts."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(__doc__);p.add_argument('job',type=Path);p.add_argument('--baseline',type=Path,required=True);a=p.parse_args();index=json.loads((a.job/'result/summary.json').read_text())
h=json.loads((a.job/'hashes.json').read_text());assert all(hashlib.sha256((a.job/k).read_bytes()).hexdigest()==v for k,v in h.items())
base=np.load(a.job/'result/native_3d_fields.npz')['fields'];entries={};verification={}
for variant in index['variants']:
 name=variant['name'];folder=a.job/'result'/name;r=json.loads((folder/'summary.json').read_text());raw=np.load(folder/'q20_dt3.125e-05.npz');field=np.load(a.job/'result'/f'{name}_fields.npz')['fields'];assert np.array_equal(field[1:],base[1:]),name
 old=np.load(a.baseline/'result/q20_dt3.125e-05.npz')
 if name=='native_3d':
  assert np.array_equal(raw['records'],old['records']);assert np.array_equal(field,base)
 maxdt=0
 for lo,hi in zip(r['results'][0]['stations'],r['results'][-1]['stations']):maxdt=max(maxdt,abs(lo['evaporation_kg_s']/hi['evaporation_kg_s']-1)*100)
 rows=r['results'][-1]['stations'];entries[name]=rows
 fields=list(raw['fields']);k=int(np.flatnonzero(np.isclose(raw['stations_m'],.4))[0]);weights=raw['initial_mass_weights'];radius=raw['records'][k,fields.index('radius_m')];support=.285 if name.startswith('paper') else .283359375
 verification[name]={'max_timestep_evaporation_change_percent':maxdt,'max_crossing_radius_at_04_m':float(radius.max()),'birth_mass_outside_profile_support_at_04':float(weights[radius>support].sum()),'minimum_crossing_axial_velocity_m_s':float(raw['records'][:,fields.index('axial_velocity_m_s')].min()),'max_crossing_energy_identity_J_kg':r['results'][-1]['max_crossing_energy_identity_J_per_kg_birth']}
results=[]
for bridge in ['linear','flat']:
 for i,x in enumerate([row['x_m'] for row in entries['native_3d']]):
  n=entries['native_speed_'+bridge][i];q=entries['paper_speed_'+bridge][i]
  results.append({'bridge':bridge,'x_m':x,'native_evaporation_g_s':1000*n['evaporation_kg_s'],'paper_evaporation_g_s':1000*q['evaporation_kg_s'],'evaporation_change_percent':100*(q['evaporation_kg_s']/n['evaporation_kg_s']-1),'native_mean_age_ms':1000*n['mean_age_s'],'paper_mean_age_ms':1000*q['mean_age_s'],'age_change_percent':100*(q['mean_age_s']/n['mean_age_s']-1)})
report={'scope':index['scope'],'native_3d_replay':'Raw half-dt crossing records identical to archived cohort; all field components except axial velocity are bitwise unchanged in every surrogate.','verification':verification,'matched_pairs':results};(a.job/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for name,rows in entries.items():
 x=[r['x_m'] for r in rows];axes[0].plot(x,[1000*r['evaporation_kg_s'] for r in rows],label=name.replace('_',' '));axes[1].plot(x,[1000*r['mean_age_s'] for r in rows],label=name.replace('_',' '))
axes[0].errorbar([.1,.4],[.224,1.047],yerr=.042,fmt='ks',label='Paper Fig.10 (raster)')
for ax in axes:ax.set_xlabel('Distance from nozzle (m)');ax.grid(alpha=.25)
axes[0].set_ylabel('Cumulative evaporation (g/s)');axes[1].set_ylabel('Mean droplet crossing age (ms)');axes[0].legend(fontsize=7)
fig.suptitle('Passive kinematic-field sensitivity: native thermal and transverse fields retained\nPaper speed treated as axial; these surrogate fields are not coupled CFD solutions',fontsize=10);fig.savefig(a.job/'velocity_reservoir_comparison.png',dpi=180);plt.close(fig)
print(json.dumps([r for r in results if r['x_m']==.4],indent=2));print(json.dumps(verification,indent=2))

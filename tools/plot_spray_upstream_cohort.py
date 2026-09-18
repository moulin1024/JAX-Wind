"""Plot the frozen cohort against the paper's liquid mass-loss curve."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(__doc__);p.add_argument('job',type=Path);p.add_argument('--paper',type=Path,required=True);a=p.parse_args()
f=json.loads((a.job/'result/summary.json').read_text());paper=json.loads(a.paper.read_text());runs=f['results'];rows=runs[-1]['stations']
x=np.r_[0,[r['x_m'] for r in rows]];E=np.r_[0,[1000*r['evaporation_kg_s'] for r in rows]];Q=np.r_[0,[r['gas_sensible_W'] for r in rows]]
px=np.array([r['x_over_L']*1.9 for r in paper['markers']]);py=np.array([(1-r['normalized_flow'])*f['injected_water_kg_s']*1000 for r in paper['markers']]);py[0]=0
fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
ax=axes[0,0];ax.plot(x,E,'o-',label='Native passive cohort');sel=px<=.951;ax.errorbar(px[sel],py[sel],yerr=.042,fmt='s-',label='Paper CFD Fig. 10 (digitized)');ax.set_ylabel('Cumulative evaporation (g/s)');ax.legend(fontsize=8)
ax=axes[0,1];ax.plot((x[1:]+x[:-1])/2,np.diff(E)/np.diff(x),'o-',label='Native cohort');ax.plot((px[1:]+px[:-1])[sel[1:]]/2,(np.diff(py)/np.diff(px))[sel[1:]],'s-',label='Paper CFD');ax.set_ylabel('Evaporation per length (g/s/m)');ax.legend(fontsize=8)
ax=axes[1,0]
for j,b in enumerate(rows[-1]['size_bins']):ax.plot(x,np.r_[0,[r['size_bins'][j]['evaporation_kg_s']*1000 for r in rows]],label=f"{b['birth_diameter_um'][0]}–{b['birth_diameter_um'][1]} µm")
ax.set_ylabel('Size-bin contribution (g/s)');ax.legend(fontsize=8)
ax=axes[1,1];ax.plot(x,E*2500,label='Latent energy of evaporated water');ax.plot(x,Q,label='Sensible heat from air');ax.plot(x,E*2500-Q,label='Net loss of liquid sensible enthalpy');ax.set_ylabel('Energy rate in native convention (W)');ax.legend(fontsize=8)
for ax in axes.flat:ax.set_xlabel('Distance from nozzle (m)');ax.grid(alpha=.25);ax.set_xlim(0,.95)
fig.suptitle('Upstream evaporation: frozen RANS field (64 × 32 × 32)\nPassive native droplets; no coupled feedback or fitted inputs',fontsize=12)
fig.savefig(a.job/'upstream_evaporation.png',dpi=180);plt.close(fig)
controls=[]
for r in runs[:-1]:
 values=np.array([v['evaporation_kg_s'] for v in r['stations']]);ref=np.array([v['evaporation_kg_s'] for v in rows]);controls.append({'quantiles':r['mass_quantiles'],'dt_s':r['dt_s'],'max_relative_evaporation_difference_percent':float(np.max(abs(values/ref-1))*100),'difference_at_04_percent':float((values[4]/ref[4]-1)*100)})
r=rows[4];report={'controls_against_finest':controls,'at_04':{'evaporation_g_s':r['evaporation_kg_s']*1000,'gas_heat_W':r['gas_sensible_W'],'latent_W':r['evaporation_kg_s']*2.5e6,'liquid_sensible_enthalpy_fraction':1-r['gas_sensible_W']/(r['evaporation_kg_s']*2.5e6),'fraction_from_150_350um':sum(b['evaporation_kg_s'] for b in r['size_bins'][1:3])/r['evaporation_kg_s']}}
(a.job/'comparison.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

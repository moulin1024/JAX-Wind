"""Summarize fixed-path thermal reservoir controls at the fully covered station."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(__doc__);p.add_argument('job',type=Path);a=p.parse_args();r=json.loads((a.job/'result/summary.json').read_text());fine=r['results'][-1];coarse=r['results'][0]
rows=fine['stations'][1]['variants'];assert fine['stations'][1]['x_m']==.4
lookup={row['name']:row for row in rows};pairs=[]
for bridge in ['', '_flat_inlet']:
 for branch in ['upper','lower']:
  p='paper'+bridge+'_'+branch;n='native_snapshot'+bridge+'_'+branch;pv=lookup[p]['evaporation_g_s'];nv=lookup[n]['evaporation_g_s'];pairs.append({'paper':p,'native':n,'increase_g_s':pv-nv,'increase_percent':100*(pv/nv-1)})
errors=[]
for low,high in zip(coarse['stations'],fine['stations']):
 for l,h in zip(low['variants'],high['variants']):errors.append(abs(l['evaporation_g_s']/h['evaporation_g_s']-1)*100)
report={'matched_pairs_at_04':pairs,'max_timestep_evaporation_difference_percent':max(errors),'native_inlet_q':r['native_inlet_q'],'paper_inferred_inlet_Y':r['paper_inferred_inlet_Y']};(a.job/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
keep=['native_3d','native_snapshot_upper','native_snapshot_lower','native_snapshot_flat_inlet_upper','native_snapshot_flat_inlet_lower','paper_upper','paper_lower','paper_flat_inlet_upper','paper_flat_inlet_lower','paper_T_only','paper_q_only']
labels=['Native 3D','Native profiles: upper','Native profiles: lower','Native profiles: flat inlet, upper','Native profiles: flat inlet, lower','Paper profiles: upper','Paper profiles: lower','Paper profiles: flat inlet, upper','Paper profiles: flat inlet, lower','Paper T only','Paper humidity only']
values=[lookup[k]['evaporation_g_s'] for k in keep];colors=['#555555']+['#377eb8']*4+['#e7852c']*4+['#8c66a8']*2
fig,ax=plt.subplots(figsize=(10,6),layout='constrained');yy=np.arange(len(keep));ax.barh(yy,values,color=colors);ax.set_yticks(yy,labels);ax.invert_yaxis();ax.axvspan(1.047-.042,1.047+.042,color='black',alpha=.13,label='Paper Fig. 10: 1.047 ± 0.042 g/s (graph reading)');ax.axvline(1.047,color='black',ls='--');ax.set_xlim(0,2.6)
for y,v in zip(yy,values):ax.text(v+.02,y,f'{v:.3f}',va='center',fontsize=9)
ax.set_xlabel('Cumulative evaporation at x = 0.4 m (g/s)');ax.set_title('Same native droplet paths and transfer law; alternative gas T and humidity\nProfile reconstruction is a sensitivity test, not a coupled CFD result');ax.legend(loc='lower left',fontsize=8);fig.savefig(a.job/'reservoir_comparison.png',dpi=180);plt.close(fig);print(json.dumps(report,indent=2))

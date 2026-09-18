"""Exact interval decomposition of passive cohort loss into age-rate and residence.

This is algebra on crossing records, not a new simulation. Relative to the
first interval, <r*s>-<r0*s0> = <(r-r0)*s0> + <r0*(s-s0)>
+ <(r-r0)*(s-s0)>, with r = loss per age and s = age per length.
Equal injected-mass weights are used without replacing mean products by
products of means. The per-age term includes all thermal/slip/gas changes.
"""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(__doc__);p.add_argument('job',type=Path);a=p.parse_args()
r=json.loads((a.job/'result/summary.json').read_text());flow=r['injected_water_kg_s']
f=np.load(a.job/'result/q64_dt3.125e-05.npz');names=list(f['fields']);raw=f['records'];ds=f['initial_diameter_m']*1e6
x=np.r_[0,f['stations_m']];age=np.vstack([np.zeros(len(ds)),raw[:,names.index('age_s')]]);loss=np.vstack([np.zeros(len(ds)),raw[:,names.index('mass_loss_fraction')]])
da=np.diff(age,axis=0);dx=np.diff(x);dl=np.diff(loss,axis=0);assert (da>0).all();rate=dl/da;residence=da/dx[:,None]
rows=[]
for lo,hi in [(74,519),(74,150),(150,250),(250,350),(350,450),(450,519)]:
 mask=(ds>=lo)&(ds<hi);w=flow*1000*mask/len(ds);r0=rate[0];s0=residence[0];base=float(np.sum(w*r0*s0))
 intervals=[]
 for j in range(len(dx)):
  thermal=float(np.sum(w*(rate[j]-r0)*s0));kinematic=float(np.sum(w*r0*(residence[j]-s0)));cross=float(np.sum(w*(rate[j]-r0)*(residence[j]-s0)));native=float(np.sum(w*rate[j]*residence[j]));error=native-base-thermal-kinematic-cross;assert abs(error)<1e-10
  intervals.append({'x_interval_m':[float(x[j]),float(x[j+1])],'evaporation_per_length_g_s_m':native,'first_interval_baseline_g_s_m':base,'per_age_rate_change_g_s_m':thermal,'residence_change_g_s_m':kinematic,'interaction_g_s_m':cross,'identity_error_g_s_m':error,'mean_loss_per_age_s_1':float(np.mean(rate[j,mask])),'mean_residence_s_m':float(np.mean(residence[j,mask]))})
 rows.append({'initial_diameter_um':[lo,hi],'intervals':intervals})
report={'scope':__doc__,'results':rows};(a.job/'rate_residence_decomposition.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(rows[0],indent=2))

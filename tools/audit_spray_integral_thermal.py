"""Compare archived source/boundary accounting with sparse sensor results.

Outlet fluxes use saved cell values and instantaneous face velocities; their
history quadrature is not an exact RK-stage discrete conservation budget.
No simulation is advanced. Enthalpy labeled solver uses its constant-latent-
heat convention; sensor enthalpy includes the diagnostic vapour sensible term.
"""
import argparse,csv,json,sys,tomllib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from assess_inertial_centre import assess
from overlay_sim_fig7 import average

parser=argparse.ArgumentParser(__doc__)
parser.add_argument('runs', nargs='+', type=Path)
parser.add_argument('--output', required=True, type=Path)
args=parser.parse_args()
results=[]
for directory in args.runs:
 name=directory.name
 p=Path(directory)
 doc=tomllib.loads((p/'resolved_case.toml').read_text())
 with (p/'history.csv').open() as f: rows=list(csv.DictReader(f))
 t=np.array([float(r['time_hours'])*3600 for r in rows])
 def values(key):return np.array([float(r[key]) for r in rows])
 def mean(key):return float(average(t,values(key)[:,None],3.,4.)[0])
 def rate(key):return float(np.interp(4.,t,values(key))-np.interp(3.,t,values(key)))
 cp,lv,cpl=1005.,2.5e6,4182.
 rho=doc['physics']['moisture']['dry_air_density_kg_m3']
 lengths=doc['mesh']['lengths_m']; mdry=rho*mean('inlet_bulk_u_m_s')*lengths[1]*lengths[2]
 t_in=doc['case']['reference']['dry_bulb_c']
 qin=mean('inlet_vapor_mixing_ratio')
 cooling=mean('cooling_power_w')
 vapor=mean('vapor_outflow_kg_s')
 cooling_source=rate('parcel_gas_sensible_energy_loss_j')
 evap=rate('parcel_evaporated_mass_kg')
 drain_mass=rate('parcel_escaped_mass_kg')
 drain_h=rate('parcel_escaped_enthalpy_j')
 a=assess(p)
 with np.load(p/'checkpoint.npz') as f:
  td=f['state/scalar'][:,:,-1]+t_in
  q=f['state/moisture/vapor'][:,:,-1]
  ux=f['state/velocity/x'][:,:,-1]
  h=1.005*td+q*(2500+1.859*td)
  wf=ux/ux.mean()
 result={'name':name,'dry_air_mass_flow_kg_s':mdry,'evaporation_source_kg_s':evap,'gas_cooling_source_W':cooling_source,'gas_outlet_cooling_power_W':cooling,'dryair_flux_weighted_outlet_T_C':t_in-cooling/(cp*mdry),'dryair_flux_weighted_outlet_q_g_kg':1000*vapor/mdry,'solver_flux_weighted_outlet_h_kJ_kg':(cp*t_in-cooling/mdry+lv*vapor/mdry)/1000,'gas_sensible_balance_residual_W':rate('gas_sensible_anomaly_j')+cooling_source-cooling,'gas_water_balance_residual_kg_s':rate('water_inventory_kg')-evap-mdry*qin+vapor,'escaped_liquid_mass_rate_kg_s':drain_mass,'escaped_liquid_T_C':drain_h/(cpl*drain_mass),'parcel_inventory_change_kg_s':rate('parcel_inventory_kg'),'predicted_9sensor_mean_DBT_WBT_h':np.asarray(a['predicted']).mean(1).tolist(),'experimental_9sensor_mean_DBT_WBT_h':np.asarray(a['experimental']).mean(1).tolist(),'snapshot_4s_area_mean_T_C':float(td.mean()),'snapshot_4s_area_mean_h_kJ_kg':float(h.mean()),'snapshot_4s_flux_mean_T_C':float((td*wf).mean()),'snapshot_4s_flux_mean_h_kJ_kg':float((h*wf).mean())}
 wall_mass=rate('parcel_escaped_wall_mass_kg')
 wall_h=rate('parcel_escaped_wall_enthalpy_j')
 result.update({'wall_touched_drain_kg_s':wall_mass,'wall_touched_drain_T_C':wall_h/(cpl*wall_mass),'never_wall_drain_kg_s':drain_mass-wall_mass,'never_wall_drain_T_C':(drain_h-wall_h)/(cpl*(drain_mass-wall_mass))})
 results.append(result)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))

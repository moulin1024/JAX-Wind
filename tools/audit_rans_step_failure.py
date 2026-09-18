"""Replay a short RANS startup and record the first rejected single step."""
import argparse
import json
from pathlib import Path
import jax
import jax.numpy as jnp
from jaxwind import FREE_SLIP, OPEN, Boundaries, Wall
from jaxwind.config.document import load_case
from jaxwind.rans_kepsilon import constrain_wall_epsilon
from jaxwind.rans_realizable import turbulent_viscosity
from jaxwind.simulation.api import RunControls
from jaxwind.simulation.water_spray_benchmark import build_simulation


def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('case',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    assert jax.default_backend()=='gpu'
    jax.config.update('jax_enable_x64',True)
    case=load_case(args.case); sim=build_simulation(case)
    g=sim.grid; dt=case.document['time']['dt_seconds']
    nu=case.document['physics']['flow']['kinematic_viscosity_m2_s']
    metric=sum(float(min(getattr(g,a+'_widths')))**-2 for a in 'xyz')
    bc=Boundaries(Wall(FREE_SLIP),Wall(FREE_SLIP),streamwise=OPEN,spanwise=FREE_SLIP)
    @jax.jit
    def diagnose(s):
        turbulence=constrain_wall_epsilon(s.turbulence,s.velocity,g,nu)
        nut=turbulent_viscosity(turbulence,s.velocity,g,bc)
        return jnp.array([s.time,jnp.max(nut),2*dt*jnp.max(nut+nu)*metric,
                          jnp.min(turbulence.kinetic_energy),jnp.min(turbulence.dissipation)])
    state=sim.initial_state; records=[]; failure=None
    for i in range(int(.3/dt)):
        values=list(map(float,diagnose(state)))
        record=dict(zip(['time_s','maximum_nut','diffusion_number','minimum_k','minimum_epsilon'],values))
        if i%40==0 or values[2]>.48:
            records.append(record); print(record,flush=True)
        try:
            state=sim.advance(state,RunControls(1,(i+1)*dt))
        except RuntimeError as exc:
            failure={'before_rejected_step':record,'exception':str(exc)}
            print('Failure:',failure,flush=True)
            break
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'case':str(args.case),'failure':failure,'samples':records},indent=2)+'\n')


if __name__=='__main__':
    main()

"""Exercise central/RK3 with a vortex crossing a high-x pressure outlet.

Run on a compute node. This is a boundary verification case, not DTU wake
validation. Both variants use central transport, no SGS, periodic y, slip z.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from jaxwind import (Boundaries, FlowModel, InflowPlane, OPEN, StaggeredVelocity,
                     build_open_atmospheric_step, build_pressure_poisson,
                     divergence, initial_atmospheric_solution)
from jaxwind.domain import UniformGrid
from jaxwind.numerics.discretization import stable_timestep, courant_number


def vortex(grid):
    x, y = jnp.asarray(grid.x_faces), jnp.asarray(grid.y_faces[:-1])
    psi = 1.2 * jnp.exp(-((x[None, :] - .88*grid.lx)/(.2*grid.lx))**2) * jnp.sin(2*jnp.pi*y[:, None]/grid.ly)
    u = 1. + (jnp.roll(psi, -1, axis=0) - psi) / grid.dy
    v = -(psi[:, 1:] - psi[:, :-1]) / grid.dx
    return StaggeredVelocity(jnp.broadcast_to(u, (grid.nz, *u.shape)),
                            jnp.broadcast_to(v, (grid.nz, *v.shape)),
                            jnp.zeros((grid.nz+1, grid.ny, grid.nx)))


def run(output, cfl):
    output.mkdir(parents=True, exist_ok=False)
    grid = UniformGrid(24, 16, 4, 6., 4., 1.)
    velocity = vortex(grid)
    plane = InflowPlane(*(f[..., 0] for f in velocity), jnp.zeros_like(velocity.x[..., 0]))
    initial = initial_atmospheric_solution(grid, velocity, dtype="float64")
    solver = build_pressure_poisson(grid, backend="gmg", periodic_x=False, dtype="float64")
    report = {"grid_cells_xyz": [grid.nx, grid.ny, grid.nz], "cfl_limit": cfl,
              "duration_seconds": 12., "initial_min_outlet_u_m_s": float(jnp.min(velocity.x[..., -1])),
              "note": "Synthetic reversing-vortex verification; no turbine, wall model, or SGS. Not a proof of nonlinear or unconditional energy stability.", "runs": {}}
    histories = {}
    for treatment in ("none", "energy"):
        advance = build_open_atmospheric_step(grid, Boundaries(streamwise=OPEN), solver,
            FlowModel(momentum_advection_scheme="central", outlet_backflow=treatment), None, scheme="rk3")
        @jax.jit
        def block(state, target):
            def body(carry):
                state, max_cfl = carry
                dt = jnp.minimum(stable_timestep(state.velocity, grid, 0., courant=cfl), target-state.time)
                actual_cfl = courant_number(state.velocity, grid, dt)
                state = advance(state, dt, plane)
                return state, jnp.maximum(max_cfl, actual_cfl)
            return jax.lax.while_loop(lambda carry: carry[0].time < target, body, (state, jnp.asarray(0.)))
        state = initial
        history, frames = [], []
        for target in np.linspace(.1, 12., 120):
            state, measured_cfl = block(state, jnp.asarray(target))
            u, v, w = (np.asarray(f) for f in state.velocity)
            if not all(np.isfinite(f).all() for f in (u,v,w)):
                raise RuntimeError(f"{treatment}: non-finite state at {target}")
            uc = .5*(u[..., 1:]+u[..., :-1])
            vc = .5*(v+np.roll(v,-1,axis=1))
            wc = .5*(w[1:]+w[:-1])
            history.append((float(state.time), float(measured_cfl), float(jnp.max(jnp.abs(divergence(state.velocity,grid)))),
                            float(np.max(np.abs(u))), float(np.min(u[..., -1])), float(np.mean(u[..., -1] < 0)),
                            float(np.mean(.5*(uc**2+vc**2+wc**2))), float(np.sum(u[..., -1]-u[..., 0])*grid.dy*grid.dz)))
            frames.append(uc[grid.nz//2])
        history = np.asarray(history)
        histories[treatment] = history
        names = 'time_seconds,maximum_step_cfl,maximum_divergence_s,maximum_abs_u_m_s,minimum_outlet_u_m_s,backflow_area_fraction,mean_kinetic_energy_m2_s2,net_flux_m3_s'
        np.savetxt(output/f'{treatment}_history.csv',history,delimiter=',',header=names,comments='')
        np.savez_compressed(output/f'{treatment}_fields.npz',initial_u=.5*(np.asarray(velocity.x)[0,:,1:]+np.asarray(velocity.x)[0,:,:-1]),frames=np.asarray(frames),x_faces=grid.x_faces,y_faces=grid.y_faces,final_u=u,final_v=v,final_w=w)
        report['runs'][treatment] = {'steps': int(state.step), 'maximum_step_cfl': float(history[:,1].max()),
            'maximum_divergence_s':float(history[:,2].max()), 'maximum_abs_u_m_s':float(history[:,3].max()),
            'minimum_outlet_u_m_s':float(history[:,4].min()),'maximum_backflow_area_fraction':float(history[:,5].max()),
            'final_mean_kinetic_energy_m2_s2':float(history[-1,6]), 'maximum_net_flux_m3_s':float(np.abs(history[:,7]).max())}
        print(treatment, json.dumps(report['runs'][treatment]), flush=True)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    for treatment,h in histories.items():
        for ax,column in zip(axes,(4,6,2)):
            ax.plot(h[:,0],h[:,column],label=treatment)
    for ax,title,label in zip(axes,('Outlet reversal','Domain kinetic energy','Mass conservation'),('Minimum outlet u [m/s]','Mean kinetic energy [m²/s²]','Maximum divergence [1/s]')):
        ax.set(xlabel='Time [s]',ylabel=label,title=title);ax.grid(alpha=.2)
    axes[0].axhline(0,color='k',ls=':',lw=1); axes[0].legend(title='Backflow treatment')
    axes[2].set_yscale('log')
    fig.suptitle(f'Central / RK3 · reversing outlet vortex · CFL limit {cfl:g}')
    fig.savefig(output/'comparison.png',dpi=170)
    plt.close(fig)
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--cfl',type=float,default=.9)
    args=parser.parse_args()
    if not 0 < args.cfl <= .9: parser.error('CFL must be in (0, 0.9]')
    run(args.output,args.cfl)

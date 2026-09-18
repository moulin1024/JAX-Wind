"""Coupled physical-inlet RANS advances conserved inertial parcels."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import jax.numpy as jnp
import pytest
from jaxwind.config.document import load_case
from jaxwind.numerics.discretization import divergence
from jaxwind.simulation.api import RunControls
from jaxwind.simulation.water_spray_benchmark import build_simulation

@pytest.mark.parametrize('model',['standard-k-epsilon','realizable-k-epsilon'])
def test_rans_physical_inlet_coupling(model):
    root=Path(__file__).resolve().parents[2]
    case=load_case(root/'cases/WaterSprayMontazeri2015/centre_investigation/core_64x32x32_realizable_physical_inlet.toml')
    doc=deepcopy(case.document)
    doc['case']['carrier_turbulence_model']=model
    doc['mesh']['cells']=[8,8,8]
    doc['case']['parcel_capacity']=32
    doc['case']['parcels_per_step']=4
    doc['case']['parcel_substeps']=1
    sim=build_simulation(replace(case,document=doc))
    result=sim.advance(sim.initial_state,RunControls(count=2,target_time=.0005))
    assert int(result.step)==2
    assert float(result.parcels.injected_mass)>0
    assert bool(jnp.all(jnp.isfinite(jnp.stack(result.turbulence))))
    assert bool(jnp.all(jnp.stack(result.turbulence)>0))
    assert float(jnp.max(jnp.abs(divergence(result.velocity,sim.grid))))<1e-7
    diag=sim.state_diagnostics(result)
    assert abs(float(diag['parcel_mass_balance_error_kg']))<1e-14
    assert abs(float(diag['parcel_enthalpy_balance_error_j']))<1e-9


def test_realizable_diagnostic_uses_physical_inlet_strain():
    from jaxwind.rans_kepsilon import KEpsilonState
    import numpy as np
    root=Path(__file__).resolve().parents[2]
    case=load_case(root/'cases/WaterSprayMontazeri2015/centre_investigation/core_64x32x32_realizable_physical_inlet.toml')
    doc=deepcopy(case.document);doc['mesh']['cells']=[8,8,8]
    doc['case']['parcel_capacity']=32
    sim=build_simulation(replace(case,document=doc));state=sim.initial_state
    # Constant transverse velocity in the first two cells: only the prescribed
    # zero inlet introduces bulk first-cell shear. Large k isolates its max nu.
    amplitude=.1
    v=state.velocity.y.at[:,1:-1,:2].set(amplitude)
    k=jnp.full_like(state.turbulence.kinetic_energy,1e-4).at[...,0].set(1.)
    eps=jnp.full_like(k,.1)
    state=state._replace(velocity=state.velocity._replace(y=v),turbulence=KEpsilonState(k,eps))
    measured=float(sim.state_diagnostics(state)['rans_nut_max'])
    shear=amplitude/sim.grid.dx
    expected=10./(4.04+3./np.sqrt(2.)*shear/.1)
    np.testing.assert_allclose(measured,expected,rtol=2e-13)

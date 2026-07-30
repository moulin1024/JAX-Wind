from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .config import Params
from .grid import center_z, upper_face_to_center
from .scalar import virtual_potential_temperature
from .state import Diagnostics
from .state import FlowState


def save_npz(path: str | Path, state: FlowState) -> None:
    arrays = {name: np.asarray(value) for name, value in state._asdict().items()}
    np.savez(path, **arrays)


def save_velocity_h5(path: str | Path, state: FlowState, params: Params, diag: Diagnostics) -> None:
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = jax.block_until_ready(state)
    with h5py.File(path, "w") as handle:
        fields = handle.create_group("fields")
        coords = handle.create_group("coords")
        coords.create_dataset(
            "x",
            data=np.asarray(
                jnp.arange(params.nx, dtype=params.dtype)
                * params.dx
                * params.z_i
            ),
        )
        coords.create_dataset(
            "y",
            data=np.asarray(
                jnp.arange(params.ny, dtype=params.dtype)
                * params.dy
                * params.z_i
            ),
        )
        coords.create_dataset("z_center", data=np.asarray(center_z(params) * params.z_i))
        coords.create_dataset(
            "z_face",
            data=np.asarray(
                jnp.arange(params.nz + 1, dtype=params.dtype)
                * params.dz
                * params.z_i
            ),
        )
        fields.create_dataset("u", data=np.asarray(state.u))
        fields.create_dataset("v", data=np.asarray(state.v))
        bottom_w = jnp.zeros_like(state.w[:, :, :1])
        fields.create_dataset(
            "w_face", data=np.asarray(jnp.concatenate((bottom_w, state.w), axis=2))
        )
        fields.create_dataset("w", data=np.asarray(upper_face_to_center(state.w)))
        fields.create_dataset("p", data=np.asarray(state.p))
        fields.create_dataset("theta", data=np.asarray(state.theta))
        fields.create_dataset("qv", data=np.asarray(state.qv))
        theta_v = virtual_potential_temperature(state.theta, state.qv, params)
        fields.create_dataset("theta_v", data=np.asarray(theta_v))
        fields.create_dataset("scalar_c_theta", data=np.asarray(state.scalar_c[..., 0]))
        fields.create_dataset("scalar_c_qv", data=np.asarray(state.scalar_c[..., 1]))

        handle.attrs["step"] = int(diag.step)
        handle.attrs["time"] = float(diag.step) * float(params.dt_physical)
        handle.attrs["dt"] = float(params.dt_physical)
        handle.attrs["time_scaled"] = float(diag.step) * float(params.dt)
        handle.attrs["dt_scaled"] = float(params.dt)
        handle.attrs["nx"] = int(params.nx)
        handle.attrs["ny"] = int(params.ny)
        handle.attrs["nz"] = int(params.nz)
        handle.attrs["lx"] = float(params.lx * params.z_i)
        handle.attrs["ly"] = float(params.ly * params.z_i)
        handle.attrs["lz"] = float(params.lz * params.z_i)
        handle.attrs["z_i"] = float(params.z_i)
        handle.attrs["u_fric"] = float(params.u_fric)
        handle.attrs["zo"] = float(params.zo)
        handle.attrs["vonk"] = float(params.vonk)
        handle.attrs["bl_height"] = float(params.bl_height)
        handle.attrs["pressure_force"] = np.nan if params.pressure_force is None else float(params.pressure_force)
        handle.attrs["pressure_force_height"] = (
            np.nan
            if params.pressure_force_height is None
            else float(params.pressure_force_height)
        )
        handle.attrs["coriolis_f"] = float(params.coriolis_f)
        handle.attrs["geostrophic_u"] = float(params.geostrophic_u)
        handle.attrs["geostrophic_v"] = float(params.geostrophic_v)
        handle.attrs["sponge_enabled"] = bool(params.sponge_enabled)
        handle.attrs["sponge_start_height"] = float(params.sponge_start_height)
        handle.attrs["sponge_timescale"] = float(params.sponge_timescale)
        handle.attrs["sponge_power"] = float(params.sponge_power)
        handle.attrs["sponge_target"] = str(params.sponge_target)
        handle.attrs["initial_condition"] = str(params.initial_condition)
        handle.attrs["momentum_wall_model"] = str(params.momentum_wall_model)
        handle.attrs["wall_stress_model"] = str(params.wall_stress_model)
        handle.attrs["molecular_viscosity"] = float(params.rayleigh_molecular_viscosity)
        handle.attrs["molecular_diffusivity"] = float(params.rayleigh_molecular_diffusivity)
        handle.attrs["rayleigh_number"] = -1.0 if params.rayleigh_number is None else float(params.rayleigh_number)
        handle.attrs["rayleigh_prandtl"] = float(params.rayleigh_prandtl)
        handle.attrs["sgs_model"] = str(params.sgs_model)
        handle.attrs["thermo_enabled"] = bool(params.thermo_enabled)
        handle.attrs["moisture_enabled"] = bool(params.moisture_enabled)
        handle.attrs["theta0"] = float(params.theta0)
        handle.attrs["theta_bc"] = str(params.theta_bc)
        handle.attrs["theta_profile"] = str(params.theta_profile)
        handle.attrs["theta_top_gradient"] = np.nan if params.theta_top_gradient is None else float(params.theta_top_gradient)
        handle.attrs["theta_bottom"] = np.nan if params.theta_bottom is None else float(params.theta_bottom)
        handle.attrs["theta_top"] = np.nan if params.theta_top is None else float(params.theta_top)
        handle.attrs["theta_initial_gradient"] = float(params.theta_initial_gradient)
        handle.attrs["theta_perturbation_amplitude"] = float(params.theta_perturbation_amplitude)
        handle.attrs["theta_perturbation_height"] = (
            np.nan if params.theta_perturbation_height is None else float(params.theta_perturbation_height)
        )
        handle.attrs["cbl_mixed_layer_height"] = (
            np.nan if params.cbl_mixed_layer_height is None else float(params.cbl_mixed_layer_height)
        )
        handle.attrs["cbl_inversion_strength"] = float(params.cbl_inversion_strength)
        handle.attrs["cbl_inversion_thickness"] = float(params.cbl_inversion_thickness)
        handle.attrs["cbl_free_atmosphere_gradient"] = float(params.cbl_free_atmosphere_gradient)
        handle.attrs["surface_theta_flux"] = float(params.surface_theta_flux)
        handle.attrs["qv0"] = float(params.qv0)
        handle.attrs["qv_initial_gradient"] = float(params.qv_initial_gradient)
        handle.attrs["surface_qv_flux"] = float(params.surface_qv_flux)
        handle.attrs["qv_floor"] = float(params.qv_floor)
        handle.attrs["scalar_sgs_model"] = str(params.scalar_sgs_model)
        handle.attrs["prandtl_t"] = float(params.prandtl_t)
        handle.attrs["schmidt_t"] = float(params.schmidt_t)
        handle.attrs["layout"] = (
            "center fields use (nx,ny,nz); fields/w is cell-centred and "
            "fields/w_face uses (nx,ny,nz+1); no persistent ghost cells"
        )


def load_npz(path: str | Path) -> FlowState:
    data = np.load(path)
    u = jnp.asarray(data["u"])
    zeros = jnp.zeros_like(u)
    cs2_default = jnp.full_like(zeros, 0.16 * 0.16)
    scalar_zeros = jnp.zeros(u.shape + (2,), dtype=u.dtype)
    scalar_c_default = jnp.full_like(scalar_zeros, 0.16 * 0.16)

    def field(name: str, default: jax.Array) -> jax.Array:
        if name in data:
            return jnp.asarray(data[name])
        return default

    return FlowState(
        u=u,
        v=jnp.asarray(data["v"]),
        w=jnp.asarray(data["w"]),
        p=jnp.asarray(data["p"]),
        theta=field("theta", zeros),
        qv=field("qv", zeros),
        rhs_u_prev=jnp.asarray(data["rhs_u_prev"]),
        rhs_v_prev=jnp.asarray(data["rhs_v_prev"]),
        rhs_w_prev=jnp.asarray(data["rhs_w_prev"]),
        rhs_theta_prev=field("rhs_theta_prev", zeros),
        rhs_qv_prev=field("rhs_qv_prev", zeros),
        lm_old=field("lm_old", zeros),
        mm_old=field("mm_old", zeros),
        qn_old=field("qn_old", zeros),
        nn_old=field("nn_old", zeros),
        cs2=field("cs2", cs2_default),
        scalar_c=field("scalar_c", scalar_c_default),
        scalar_lm_old=field("scalar_lm_old", scalar_zeros),
        scalar_mm_old=field("scalar_mm_old", scalar_zeros),
        scalar_qn_old=field("scalar_qn_old", scalar_zeros),
        scalar_nn_old=field("scalar_nn_old", scalar_zeros),
        u_lag=field("u_lag", zeros),
        v_lag=field("v_lag", zeros),
        w_lag=field("w_lag", zeros),
        step=jnp.asarray(data["step"]),
    )

"""Audit completed inertial-spray checkpoints for loading and modeled mixing."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from jaxwind import FREE_SLIP, OPEN, Boundaries, StaggeredVelocity, Wall
from jaxwind.config.moisture import load_moisture
from jaxwind.cryogenic import _cic_coordinates, _cic_deposit_many, _cic_sample_many
from jaxwind.domain import UniformGrid
from jaxwind.io.checkpoint import checkpoint_metadata
from jaxwind.numerics.discretization import cell_velocity
from jaxwind.physics.moisture import advance_water_droplet
from jaxwind.sgs import AnisotropicMinimumDissipation, eddy_viscosity
from jaxwind.simulation.water_spray_benchmark import inlet_mixing_ratio


def audit(directory):
    path = directory / "checkpoint.npz"
    header = checkpoint_metadata(path)
    doc = header["resolved_case"]
    grid = UniformGrid(*doc["mesh"]["cells"], *doc["mesh"]["lengths_m"])
    moist, _ = load_moisture(doc["physics"])
    config = moist.thermodynamics
    jax.config.update("jax_enable_x64", True)
    with np.load(path, allow_pickle=False) as data:
        velocity = StaggeredVelocity(
            *(jnp.asarray(data[f"state/velocity/{a}"]) for a in "xyz")
        )
        position = jnp.asarray(data["state/parcels/position"])
        droplet_mass = np.asarray(data["state/parcels/mass"])
        multiplicity = np.asarray(
            data["state/parcels/multiplicity"] * data["state/parcels/active"]
        )
        parcel_temperature = np.asarray(data["state/parcels/temperature"])
        parcel_velocity = np.asarray(data["state/parcels/velocity"])
        gas_temperature = np.asarray(data["state/scalar"]) + moist.temperature_offset_k
        gas_vapor = np.asarray(data["state/moisture/vapor"])
        physical_mass = jnp.asarray(
            data["state/parcels/mass"]
            * data["state/parcels/multiplicity"]
            * data["state/parcels/active"]
        )
    # A slab flux estimator, not a cumulative outlet-event ledger. Its bias
    # and sampling error need assessment with time averages and mesh refinement.
    outlet_mask = np.asarray(position[0]) >= grid.lx - grid.x_widths[-1]
    outlet_weights = (
        np.asarray(physical_mass) * np.maximum(parcel_velocity[0], 0) * outlet_mask
    )
    outlet_temperature = float(
        np.sum(outlet_weights * (parcel_temperature - 273.15)) / np.sum(outlet_weights)
    )
    outlet_mass_flow = float(outlet_weights.sum() / grid.x_widths[-1])
    areas = np.asarray(grid.z_widths)[:, None] * np.asarray(grid.y_widths)[None, :]
    ambient_vapor = inlet_mixing_ratio(doc["case"]["reference"], config)
    net_vapor_rate = config.dry_air_density * float(
        np.sum(np.asarray(velocity.x)[..., -1] * gas_vapor[..., -1] * areas)
        - np.sum(np.asarray(velocity.x)[..., 0] * ambient_vapor * areas)
    )
    coordinates = _cic_coordinates(*position, grid)
    gas_fields = jnp.stack(
        (*cell_velocity(velocity), jnp.asarray(gas_temperature), jnp.asarray(gas_vapor))
    )
    sampled = _cic_sample_many(gas_fields, coordinates)
    h = doc["time"]["dt_seconds"] / doc["case"].get("parcel_substeps", 4)
    update = jax.jit(
        lambda: advance_water_droplet(
            jnp.asarray(droplet_mass),
            jnp.asarray(parcel_temperature),
            sampled[3],
            sampled[4],
            jnp.linalg.norm(jnp.asarray(parcel_velocity) - sampled[:3], axis=0),
            h,
            config,
        )
    )()
    evaporation = np.asarray(update.evaporated_mass) * multiplicity / h
    sensible_loss = np.asarray(update.gas_sensible_energy_loss) * multiplicity / h
    wall_contact = (
        (np.asarray(position[1]) < 1e-10)
        | (np.asarray(position[1]) > grid.ly - 1e-10)
        | (np.asarray(position[2]) < 1e-10)
        | (np.asarray(position[2]) > grid.lz - 1e-10)
    )
    wall_metrics = {
        "instantaneous_spray_evaporation_kg_s": float(evaporation.sum()),
        "instantaneous_wall_contact_evaporation_kg_s": float(
            evaporation[wall_contact].sum()
        ),
        "instantaneous_spray_sensible_loss_w": float(sensible_loss.sum()),
        "instantaneous_wall_contact_sensible_loss_w": float(
            sensible_loss[wall_contact].sum()
        ),
        "instantaneous_wall_contact_liquid_mass_fraction": float(
            np.asarray(physical_mass)[wall_contact].sum()
            / np.asarray(physical_mass).sum()
        ),
        "outlet_slab_wall_contact_liquid_flux_fraction": float(
            outlet_weights[wall_contact].sum() / outlet_weights.sum()
        ),
    }
    boundaries = Boundaries(
        Wall(FREE_SLIP), Wall(FREE_SLIP), streamwise=OPEN, spanwise=FREE_SLIP
    )
    nu = np.asarray(
        jax.jit(
            lambda v: eddy_viscosity(
                v, grid, boundaries, AnisotropicMinimumDissipation()
            )
        )(velocity)
    )
    loading = np.asarray(
        jax.jit(
            lambda p, m: _cic_deposit_many(
                m[None], _cic_coordinates(*p, grid), (grid.nz, grid.ny, grid.nx)
            )[0]
        )(position, physical_mass)
    )
    volume_fraction = loading / (config.water_density * np.asarray(grid.cell_volumes))
    iz, iy, ix = np.unravel_index(np.argmax(volume_fraction), volume_fraction.shape)
    # The paper's Eq. (1) states k=(U I)^2 (not 3/2 times that value).
    k = (doc["physics"]["flow"]["streamwise_velocity_m_s"] * 0.1) ** 2
    length_scale = 0.07 * grid.ly
    epsilon = 0.09**0.75 * k**1.5 / length_scale
    inlet_rans_viscosity = 0.09 * k * k / epsilon
    profiles = []
    diameter_um = np.cbrt(6 * droplet_mass / (np.pi * config.water_density)) * 1e6
    radial = np.hypot(
        np.asarray(position[1]) - grid.ly / 2, np.asarray(position[2]) - grid.lz / 2
    )
    for station in (0.25, 0.5, 0.75):
        for low, high in ((74, 150), (250, 350), (400, 520), (500, 520)):
            mask = (
                (multiplicity > 0)
                & (abs(np.asarray(position[0]) - station * grid.lx) < 0.025)
                & (diameter_um >= low)
                & (diameter_um < high)
            )
            weights = np.asarray(physical_mass) * mask
            mass = weights.sum()
            profiles.append(
                {
                    "x_over_length": station,
                    "axial_half_width_m": 0.025,
                    "current_diameter_range_um": [low, high],
                    "parcel_count": int(mask.sum()),
                    "mass_weighted_temperature_c": float(
                        np.sum(weights * (parcel_temperature - 273.15)) / mass
                    )
                    if mass > 0
                    else None,
                    "mass_weighted_radial_distance_m": float(
                        np.sum(weights * radial) / mass
                    )
                    if mass > 0
                    else None,
                }
            )
    return {
        "instantaneous_size_resolved_droplet_profiles": profiles,
        "directory": str(directory),
        **wall_metrics,
        "outlet_liquid_temperature_c_slab_estimate": outlet_temperature,
        "outlet_liquid_mass_flow_kg_s_slab_estimate": outlet_mass_flow,
        "net_outlet_vapor_mass_flow_kg_s": net_vapor_rate,
        "minimum_gas_temperature_k": float(gas_temperature.min()),
        "final_maximum_liquid_volume_fraction": float(volume_fraction.max()),
        "first_0p1_m_maximum_liquid_volume_fraction": float(
            volume_fraction[..., np.asarray(grid.x_centers) < 0.1].max()
        ),
        "maximum_location_xyz_m": [
            float(grid.x_centers[ix]),
            float(grid.y_centers[iy]),
            float(grid.z_centers[iz]),
        ],
        "domain_volume_fraction_above_liquid_fraction_0p001": float(
            np.mean(volume_fraction > 0.001)
        ),
        "liquid_mass_fraction_in_cells_above_volume_fraction_0p001": float(
            loading[volume_fraction > 0.001].sum() / loading.sum()
        ),
        "amd_viscosity_percentiles_0_50_95_100_m2_s": np.percentile(
            nu, [0, 50, 95, 100]
        ).tolist(),
        "amd_volume_mean_viscosity_m2_s": float(np.mean(nu)),
        "reference_inlet_k_m2_s2": k,
        "reference_inlet_length_scale_m": length_scale,
        "reference_inlet_epsilon_m2_s3": epsilon,
        "reference_inlet_k_epsilon_viscosity_estimate_m2_s": inlet_rans_viscosity,
        "interpretation": "LES subgrid viscosity and RANS total turbulent viscosity are not equivalent. This comparison exposes a modeling difference; it is not a viscosity calibration prescription. Consult the resolved case for imposed inlet turbulence.",
    }


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cases/WaterSprayMontazeri2015/investigation/state_audit.json"),
    )
    args = parser.parse_args()
    result = [audit(d) for d in args.runs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

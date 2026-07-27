#pragma once

namespace wireles {

struct ThermodynamicConstants {
    double reference_pressure = 100000.0;
    double dry_air_gas_constant = 287.04;
    double water_vapor_gas_constant = 461.5;
    double dry_air_heat_capacity = 1004.0;
    double latent_heat_vaporization = 2.5e6;
    double triple_point_temperature = 273.15;
    double triple_point_vapor_pressure = 611.2;
};

struct MoistThermodynamicState {
    double temperature = 0.0;
    double potential_temperature = 0.0;
    double liquid_water_potential_temperature = 0.0;
    double total_water_mixing_ratio = 0.0;
    double water_vapor_mixing_ratio = 0.0;
    double liquid_water_mixing_ratio = 0.0;
    double saturation_mixing_ratio = 0.0;
    double temperature_residual = 0.0;
    int iterations = 0;
    bool saturated = false;
};

struct MoistConservedJacobians {
    double dliquid_water_dtheta_l = 0.0;
    double dliquid_water_dtotal_water = 0.0;
    double dvirtual_theta_dtheta_l = 0.0;
    double dvirtual_theta_dtotal_water = 0.0;
};

double exner_function(
    double pressure,
    const ThermodynamicConstants& constants = {});
double hydrostatic_base_pressure(
    double height,
    double surface_pressure,
    double reference_potential_temperature,
    double gravity,
    const ThermodynamicConstants& constants = {});
double saturation_vapor_pressure(
    double temperature,
    const ThermodynamicConstants& constants = {});
double saturation_vapor_pressure_derivative(
    double temperature,
    const ThermodynamicConstants& constants = {});
double saturation_water_vapor_mixing_ratio(
    double temperature,
    double pressure,
    const ThermodynamicConstants& constants = {});
double liquid_water_potential_temperature(
    double potential_temperature,
    double liquid_water_mixing_ratio,
    double pressure,
    const ThermodynamicConstants& constants = {});

// Saturation adjustment for the conserved warm-cloud variables theta_l and q_t.
// The closure uses theta_l = theta - Lv*q_l/(cp*Pi) and q_t = q_v + q_l.
MoistThermodynamicState saturation_adjustment(
    double liquid_water_potential_temperature,
    double total_water_mixing_ratio,
    double pressure,
    const ThermodynamicConstants& constants = {},
    double temperature_tolerance = 1.0e-10,
    int max_iterations = 100);

// Local Jacobian of diagnostic q_l and theta_v with respect to the conserved
// warm-cloud variables theta_l and q_t. The input state must already be in
// saturation-adjustment equilibrium, as it is in the solver state.
MoistConservedJacobians moist_conserved_jacobians(
    double potential_temperature,
    double water_vapor_mixing_ratio,
    double liquid_water_mixing_ratio,
    double pressure,
    const ThermodynamicConstants& constants = {});

// Equilibrate an arbitrary warm-cloud state while conserving q_t and
// cp*T - Lv*q_l, allowing both condensation and evaporation tests.
MoistThermodynamicState equilibrate_moisture(
    double temperature,
    double water_vapor_mixing_ratio,
    double liquid_water_mixing_ratio,
    double pressure,
    const ThermodynamicConstants& constants = {},
    double temperature_tolerance = 1.0e-10,
    int max_iterations = 100);

}  // namespace wireles

#include "wireles/thermodynamics.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace wireles {
namespace {

void require_finite_positive(double value, const char* name) {
    if (!std::isfinite(value) || value <= 0.0) {
        throw std::invalid_argument(std::string(name) + " must be finite and positive");
    }
}

void validate_constants(const ThermodynamicConstants& constants) {
    require_finite_positive(constants.reference_pressure, "reference pressure");
    require_finite_positive(constants.dry_air_gas_constant, "dry-air gas constant");
    require_finite_positive(constants.water_vapor_gas_constant, "water-vapor gas constant");
    require_finite_positive(constants.dry_air_heat_capacity, "dry-air heat capacity");
    require_finite_positive(constants.latent_heat_vaporization, "latent heat");
    require_finite_positive(constants.triple_point_temperature, "triple-point temperature");
    require_finite_positive(constants.triple_point_vapor_pressure, "triple-point vapor pressure");
}

}  // namespace

double exner_function(double pressure, const ThermodynamicConstants& constants) {
    validate_constants(constants);
    require_finite_positive(pressure, "pressure");
    const double kappa = constants.dry_air_gas_constant / constants.dry_air_heat_capacity;
    return std::pow(pressure / constants.reference_pressure, kappa);
}

double hydrostatic_base_pressure(
    double height,
    double surface_pressure,
    double reference_potential_temperature,
    double gravity,
    const ThermodynamicConstants& constants) {
    validate_constants(constants);
    if (!std::isfinite(height) || height < 0.0) {
        throw std::invalid_argument("height must be finite and non-negative");
    }
    require_finite_positive(surface_pressure, "surface pressure");
    require_finite_positive(reference_potential_temperature, "reference potential temperature");
    require_finite_positive(gravity, "gravity");
    const double surface_exner = exner_function(surface_pressure, constants);
    const double exner = surface_exner
        - gravity * height / (constants.dry_air_heat_capacity * reference_potential_temperature);
    if (exner <= 0.0) {
        throw std::invalid_argument("hydrostatic base-state Exner function is non-positive");
    }
    return constants.reference_pressure
        * std::pow(exner, constants.dry_air_heat_capacity / constants.dry_air_gas_constant);
}

double saturation_vapor_pressure(double temperature, const ThermodynamicConstants& constants) {
    validate_constants(constants);
    require_finite_positive(temperature, "temperature");
    // Bolton's warm-water fit, with pressure returned in Pa.
    const double temperature_celsius = temperature - constants.triple_point_temperature;
    return constants.triple_point_vapor_pressure
        * std::exp(17.67 * temperature_celsius / (temperature_celsius + 243.5));
}

double saturation_vapor_pressure_derivative(
    double temperature,
    const ThermodynamicConstants& constants) {
    validate_constants(constants);
    require_finite_positive(temperature, "temperature");
    const double denominator = temperature - constants.triple_point_temperature + 243.5;
    const double logarithmic_derivative = 17.67 * 243.5 / (denominator * denominator);
    return saturation_vapor_pressure(temperature, constants) * logarithmic_derivative;
}

double saturation_water_vapor_mixing_ratio(
    double temperature,
    double pressure,
    const ThermodynamicConstants& constants) {
    validate_constants(constants);
    require_finite_positive(pressure, "pressure");
    const double vapor_pressure = saturation_vapor_pressure(temperature, constants);
    if (vapor_pressure >= pressure) {
        throw std::invalid_argument("saturation vapor pressure must be below ambient pressure");
    }
    const double epsilon = constants.dry_air_gas_constant / constants.water_vapor_gas_constant;
    return epsilon * vapor_pressure / (pressure - vapor_pressure);
}

double liquid_water_potential_temperature(
    double potential_temperature,
    double liquid_water_mixing_ratio,
    double pressure,
    const ThermodynamicConstants& constants) {
    validate_constants(constants);
    require_finite_positive(potential_temperature, "potential temperature");
    if (!std::isfinite(liquid_water_mixing_ratio) || liquid_water_mixing_ratio < 0.0) {
        throw std::invalid_argument("liquid-water mixing ratio must be finite and non-negative");
    }
    const double exner = exner_function(pressure, constants);
    return potential_temperature
        - constants.latent_heat_vaporization * liquid_water_mixing_ratio
            / (constants.dry_air_heat_capacity * exner);
}

MoistThermodynamicState saturation_adjustment(
    double theta_l,
    double total_water,
    double pressure,
    const ThermodynamicConstants& constants,
    double temperature_tolerance,
    int max_iterations) {
    validate_constants(constants);
    require_finite_positive(theta_l, "liquid-water potential temperature");
    require_finite_positive(pressure, "pressure");
    require_finite_positive(temperature_tolerance, "temperature tolerance");
    if (!std::isfinite(total_water) || total_water < 0.0) {
        throw std::invalid_argument("total-water mixing ratio must be finite and non-negative");
    }
    if (max_iterations <= 0) {
        throw std::invalid_argument("maximum iteration count must be positive");
    }

    const double exner = exner_function(pressure, constants);
    const double dry_temperature = exner * theta_l;
    const double dry_saturation = saturation_water_vapor_mixing_ratio(dry_temperature, pressure, constants);

    MoistThermodynamicState state;
    state.liquid_water_potential_temperature = theta_l;
    state.total_water_mixing_ratio = total_water;
    if (total_water <= dry_saturation) {
        state.temperature = dry_temperature;
        state.potential_temperature = theta_l;
        state.water_vapor_mixing_ratio = total_water;
        state.liquid_water_mixing_ratio = 0.0;
        state.saturation_mixing_ratio = dry_saturation;
        state.saturated = std::abs(total_water - dry_saturation) <= 1.0e-14;
        return state;
    }

    const double latent_over_cp = constants.latent_heat_vaporization / constants.dry_air_heat_capacity;
    auto residual = [&](double temperature) {
        const double qsat = saturation_water_vapor_mixing_ratio(temperature, pressure, constants);
        return temperature - dry_temperature - latent_over_cp * (total_water - qsat);
    };

    double lower = dry_temperature;
    double upper = dry_temperature + latent_over_cp * total_water;
    if (!(residual(lower) <= 0.0 && residual(upper) >= 0.0)) {
        throw std::runtime_error("failed to bracket saturation-adjustment temperature");
    }

    double temperature = 0.5 * (lower + upper);
    double f_temperature = residual(temperature);
    int iterations = 0;
    for (; iterations < max_iterations; ++iterations) {
        temperature = 0.5 * (lower + upper);
        f_temperature = residual(temperature);
        if (std::abs(f_temperature) <= temperature_tolerance
            || 0.5 * (upper - lower) <= temperature_tolerance) {
            ++iterations;
            break;
        }
        if (f_temperature > 0.0) {
            upper = temperature;
        } else {
            lower = temperature;
        }
    }
    if (iterations >= max_iterations && std::abs(f_temperature) > temperature_tolerance) {
        throw std::runtime_error("saturation adjustment did not converge");
    }

    state.temperature = temperature;
    state.potential_temperature = temperature / exner;
    state.saturation_mixing_ratio = saturation_water_vapor_mixing_ratio(temperature, pressure, constants);
    state.water_vapor_mixing_ratio = std::min(total_water, state.saturation_mixing_ratio);
    state.liquid_water_mixing_ratio = std::max(total_water - state.water_vapor_mixing_ratio, 0.0);
    state.temperature_residual = f_temperature;
    state.iterations = iterations;
    state.saturated = true;
    return state;
}

MoistConservedJacobians moist_conserved_jacobians(
    double theta,
    double water_vapor,
    double liquid_water,
    double pressure,
    const ThermodynamicConstants& constants) {
    validate_constants(constants);
    require_finite_positive(theta, "potential temperature");
    require_finite_positive(pressure, "pressure");
    if (!std::isfinite(water_vapor) || water_vapor < 0.0
        || !std::isfinite(liquid_water) || liquid_water < 0.0) {
        throw std::invalid_argument("water mixing ratios must be finite and non-negative");
    }

    MoistConservedJacobians jacobian;
    if (liquid_water <= 0.0) {
        jacobian.dvirtual_theta_dtheta_l = 1.0 + 0.61 * water_vapor;
        jacobian.dvirtual_theta_dtotal_water = 0.61 * theta;
        return jacobian;
    }

    const double exner = exner_function(pressure, constants);
    const double temperature = exner * theta;
    const double vapor_pressure = saturation_vapor_pressure(temperature, constants);
    const double vapor_pressure_derivative =
        saturation_vapor_pressure_derivative(temperature, constants);
    const double epsilon = constants.dry_air_gas_constant / constants.water_vapor_gas_constant;
    const double pressure_minus_vapor = pressure - vapor_pressure;
    const double dqsat_dtemperature = epsilon * pressure * vapor_pressure_derivative
        / (pressure_minus_vapor * pressure_minus_vapor);
    const double latent_over_cp =
        constants.latent_heat_vaporization / constants.dry_air_heat_capacity;
    const double denominator = 1.0 + latent_over_cp * dqsat_dtemperature;

    const double dtemperature_dtheta_l = exner / denominator;
    const double dtemperature_dtotal_water = latent_over_cp / denominator;
    const double dtheta_dtheta_l = 1.0 / denominator;
    const double dtheta_dtotal_water = latent_over_cp / (exner * denominator);
    const double dqv_dtheta_l = dqsat_dtemperature * dtemperature_dtheta_l;
    const double dqv_dtotal_water = dqsat_dtemperature * dtemperature_dtotal_water;
    jacobian.dliquid_water_dtheta_l = -dqv_dtheta_l;
    jacobian.dliquid_water_dtotal_water = 1.0 - dqv_dtotal_water;

    const double virtual_factor = 1.0 + 0.61 * water_vapor - liquid_water;
    jacobian.dvirtual_theta_dtheta_l = dtheta_dtheta_l * virtual_factor
        + theta * (0.61 * dqv_dtheta_l - jacobian.dliquid_water_dtheta_l);
    jacobian.dvirtual_theta_dtotal_water = dtheta_dtotal_water * virtual_factor
        + theta * (0.61 * dqv_dtotal_water - jacobian.dliquid_water_dtotal_water);
    return jacobian;
}

MoistThermodynamicState equilibrate_moisture(
    double temperature,
    double water_vapor,
    double liquid_water,
    double pressure,
    const ThermodynamicConstants& constants,
    double temperature_tolerance,
    int max_iterations) {
    validate_constants(constants);
    require_finite_positive(temperature, "temperature");
    if (!std::isfinite(water_vapor) || water_vapor < 0.0
        || !std::isfinite(liquid_water) || liquid_water < 0.0) {
        throw std::invalid_argument("water mixing ratios must be finite and non-negative");
    }
    const double exner = exner_function(pressure, constants);
    const double theta = temperature / exner;
    const double theta_l = liquid_water_potential_temperature(theta, liquid_water, pressure, constants);
    return saturation_adjustment(
        theta_l,
        water_vapor + liquid_water,
        pressure,
        constants,
        temperature_tolerance,
        max_iterations);
}

}  // namespace wireles

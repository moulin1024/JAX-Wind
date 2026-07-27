#include "wireles/field.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

#include "wireles/bomex.hpp"
#include "wireles/thermodynamics.hpp"

namespace wireles {

FlowState::FlowState(const Params& params)
    : u(params.real_size(), 0.0),
      v(params.real_size(), 0.0),
      w(params.z_face_size(), 0.0),
      p(params.real_size(), 0.0),
      theta(params.real_size(), 0.0),
      qv(params.real_size(), 0.0),
      theta_l(params.real_size(), 0.0),
      qt(params.real_size(), 0.0),
      ql(params.real_size(), 0.0),
      sgs_tke(params.real_size(), params.tke_floor),
      base_pressure(static_cast<std::size_t>(params.nz), 0.0),
      rhs_u_prev(params.real_size(), 0.0),
      rhs_v_prev(params.real_size(), 0.0),
      rhs_w_prev(params.z_face_size(), 0.0),
      rhs_theta_prev(params.real_size(), 0.0),
      rhs_qv_prev(params.real_size(), 0.0),
      rhs_sgs_tke_prev(params.real_size(), 0.0),
      cs2(params.real_size(), params.smagorinsky_cs * params.smagorinsky_cs),
      lm_old(params.real_size(), 0.0),
      mm_old(params.real_size(), 0.0),
      qn_old(params.real_size(), 0.0),
      nn_old(params.real_size(), 0.0),
      scalar_c(params.real_size(), (params.smagorinsky_cs * params.smagorinsky_cs) / params.prandtl_t),
      scalar_lm_old(params.real_size(), 0.0),
      scalar_mm_old(params.real_size(), 0.0),
      scalar_qn_old(params.real_size(), 0.0),
      scalar_nn_old(params.real_size(), 0.0),
      qt_scalar_c(params.real_size(), (params.smagorinsky_cs * params.smagorinsky_cs) / params.schmidt_t),
      qt_scalar_lm_old(params.real_size(), 0.0),
      qt_scalar_mm_old(params.real_size(), 0.0),
      qt_scalar_qn_old(params.real_size(), 0.0),
      qt_scalar_nn_old(params.real_size(), 0.0),
      u_lag(params.real_size(), 0.0),
      v_lag(params.real_size(), 0.0),
      w_lag(params.real_size(), 0.0) {}

void enforce_walls(Field& w, const Params& params) {
    for (int j = 0; j < params.ny; ++j) {
        for (int i = 0; i < params.nx; ++i) {
            w[z_face_idx(params, i, j, 0)] = 0.0;
            w[z_face_idx(params, i, j, params.nz)] = 0.0;
        }
    }
}

void update_moist_thermodynamics(FlowState& state, const Params& params) {
    if (!params.moisture_enabled) {
        return;
    }
    for (int k = 0; k < params.nz; ++k) {
        const double pressure = state.base_pressure[static_cast<std::size_t>(k)];
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                const MoistThermodynamicState moist = saturation_adjustment(
                    state.theta_l[n], state.qt[n], pressure);
                state.theta[n] = moist.potential_temperature;
                state.qv[n] = moist.water_vapor_mixing_ratio;
                state.ql[n] = moist.liquid_water_mixing_ratio;
            }
        }
    }
}

void initialize_moist_thermodynamics(FlowState& state, const Params& params) {
    if (!params.moisture_enabled) {
        return;
    }
    for (int k = 0; k < params.nz; ++k) {
        const double z = (static_cast<double>(k) + 0.5) * params.dz();
        state.base_pressure[static_cast<std::size_t>(k)] = hydrostatic_base_pressure(
            z, params.surface_pressure, params.theta0, params.g);
    }
    state.theta_l = state.theta;
    state.qt = state.qv;
    update_moist_thermodynamics(state, params);
}

void initialize(FlowState& state, const Params& params) {
    if (params.initial_condition == "taylor_green") {
        for (int k = 0; k < params.nz; ++k) {
            const double z = (static_cast<double>(k) + 0.5) * params.dz();
            for (int j = 0; j < params.ny; ++j) {
                const double y = static_cast<double>(j) * params.dy();
                for (int i = 0; i < params.nx; ++i) {
                    const double x = static_cast<double>(i) * params.dx();
                    state.u[idx(params, i, j, k)] = std::sin(x) * std::cos(y);
                    state.v[idx(params, i, j, k)] = -std::cos(x) * std::sin(y);
                    state.theta[idx(params, i, j, k)] = params.theta0 + params.theta_initial_gradient * z;
                    state.qv[idx(params, i, j, k)] = params.qv0 + params.qv_initial_gradient * z;
                }
            }
        }
    } else if (params.initial_condition == "largeeddy1993") {
        if (params.surface_theta_flux <= 0.0) {
            throw std::runtime_error("largeeddy1993 initial condition requires positive surface_theta_flux");
        }
        const double zi1 = params.largeeddy_initial_zi1_fraction * params.z_i;
        const double wstar = std::cbrt((params.g / params.theta0) * params.surface_theta_flux * params.z_i);
        const double theta_star = params.surface_theta_flux / wstar;
        std::mt19937 rng(static_cast<std::mt19937::result_type>(params.random_seed));
        std::uniform_real_distribution<double> uniform(-0.5, 0.5);

        for (int k = 0; k < params.nz; ++k) {
            const double z = (static_cast<double>(k) + 0.5) * params.dz();
            const double lower_weight = std::max(1.0 - z / zi1, 0.0);
            const bool in_mixed_layer = z < zi1;
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const double perturb = 0.1 * uniform(rng) * lower_weight;
                    state.theta[idx(params, i, j, k)] = in_mixed_layer
                        ? params.theta0 + perturb * theta_star
                        : params.theta0 + (z - zi1) * params.theta_initial_gradient;
                    state.qv[idx(params, i, j, k)] = params.qv0 + params.qv_initial_gradient * z;
                }
            }
        }
        for (int k = 1; k < params.nz; ++k) {
            const double z = static_cast<double>(k) * params.dz();
            const double lower_weight = std::max(1.0 - z / zi1, 0.0);
            const bool in_mixed_layer = z < zi1;
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const double perturb = 0.1 * uniform(rng) * lower_weight;
                    state.w[z_face_idx(params, i, j, k)] = in_mixed_layer ? perturb * wstar : 0.0;
                }
            }
        }
    } else if (params.initial_condition == "neutral_ekman") {
        const double perturbation_height = params.initial_perturbation_height > 0.0
            ? params.initial_perturbation_height
            : 0.25 * params.lz;
        std::mt19937 rng(static_cast<std::mt19937::result_type>(params.random_seed));
        std::uniform_real_distribution<double> uniform(-0.5, 0.5);

        for (int k = 0; k < params.nz; ++k) {
            const double z = (static_cast<double>(k) + 0.5) * params.dz();
            const double lower_weight = perturbation_height > 0.0
                ? std::max(1.0 - z / perturbation_height, 0.0)
                : 0.0;
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const std::size_t n = idx(params, i, j, k);
                    state.u[n] = params.geostrophic_u
                        + params.initial_velocity_perturbation * lower_weight * uniform(rng);
                    state.v[n] = params.geostrophic_v
                        + params.initial_velocity_perturbation * lower_weight * uniform(rng);
                    state.theta[n] = params.theta0 + params.theta_initial_gradient * z;
                    state.qv[n] = params.qv0 + params.qv_initial_gradient * z;
                }
            }
        }
        for (int k = 1; k < params.nz; ++k) {
            const double z = static_cast<double>(k) * params.dz();
            const double lower_weight = perturbation_height > 0.0
                ? std::max(1.0 - z / perturbation_height, 0.0)
                : 0.0;
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    state.w[z_face_idx(params, i, j, k)] =
                        params.initial_velocity_perturbation * lower_weight * uniform(rng);
                }
            }
        }
    } else if (params.initial_condition == "bomex") {
        std::mt19937 rng(static_cast<std::mt19937::result_type>(params.random_seed));
        std::uniform_real_distribution<double> uniform(-1.0, 1.0);
        for (int k = 0; k < params.nz; ++k) {
            const double z = (static_cast<double>(k) + 0.5) * params.dz();
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const std::size_t n = idx(params, i, j, k);
                    state.u[n] = bomex_initial_u(z);
                    state.v[n] = 0.0;
                    state.theta[n] = bomex_initial_theta_l(z);
                    state.qv[n] = bomex_initial_qt(z);
                    const bool perturb = params.initial_perturbation_height > 0.0
                        ? z < params.initial_perturbation_height
                        : k < 4;
                    if (perturb) {
                        state.theta[n] += params.bomex_theta_perturbation * uniform(rng);
                        state.qv[n] = std::max(0.0, state.qv[n] + params.bomex_qt_perturbation * uniform(rng));
                    }
                }
            }
        }
    } else {
        throw std::runtime_error("unsupported initial_condition: " + params.initial_condition);
    }
    initialize_moist_thermodynamics(state, params);
    enforce_walls(state.w, params);
}

double max_abs(const Field& q) {
    double value = 0.0;
    for (const double item : q) {
        if (!std::isfinite(item)) {
            return std::numeric_limits<double>::infinity();
        }
        value = std::max(value, std::abs(item));
    }
    return value;
}

}  // namespace wireles

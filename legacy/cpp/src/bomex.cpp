#include "wireles/bomex.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "wireles/thermodynamics.hpp"
#include "wireles/wall.hpp"

namespace wireles {
namespace {

constexpr double seconds_per_day = 86400.0;

double linear_profile(double z, double z0, double value0, double z1, double value1) {
    const double weight = (z - z0) / (z1 - z0);
    return value0 + weight * (value1 - value0);
}

std::vector<double> plane_mean(const Field& field, const Params& params) {
    std::vector<double> mean(static_cast<std::size_t>(params.nz), 0.0);
    const double inverse_plane = 1.0 / static_cast<double>(params.nx * params.ny);
    for (int k = 0; k < params.nz; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                mean[static_cast<std::size_t>(k)] += field[idx(params, i, j, k)] * inverse_plane;
            }
        }
    }
    return mean;
}

Field centered_vertical_velocity(const Field& w, const Params& params) {
    Field centered(params.real_size(), 0.0);
    for (int k = 0; k < params.nz; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                centered[idx(params, i, j, k)] = 0.5 * (
                    w[z_face_idx(params, i, j, k)]
                    + w[z_face_idx(params, i, j, k + 1)]);
            }
        }
    }
    return centered;
}

double vertical_derivative(const std::vector<double>& profile, int k, const Params& params) {
    if (params.nz == 1) {
        return 0.0;
    }
    if (k == 0) {
        return (profile[1] - profile[0]) / params.dz();
    }
    if (k == params.nz - 1) {
        return (profile[static_cast<std::size_t>(k)] - profile[static_cast<std::size_t>(k - 1)]) / params.dz();
    }
    return (profile[static_cast<std::size_t>(k + 1)] - profile[static_cast<std::size_t>(k - 1)])
        / (2.0 * params.dz());
}

void ensure_profile_storage(BomexAccumulator& accumulator, const Params& params) {
    const std::size_t nz = static_cast<std::size_t>(params.nz);
    const std::size_t threshold_count = bomex_cloud_thresholds().size();
    if (accumulator.theta_l.empty()) {
        accumulator.theta_l.assign(nz, 0.0);
        accumulator.qt.assign(nz, 0.0);
        accumulator.qv.assign(nz, 0.0);
        accumulator.ql.assign(nz, 0.0);
        accumulator.u.assign(nz, 0.0);
        accumulator.v.assign(nz, 0.0);
        accumulator.tke.assign(nz, 0.0);
        accumulator.cloud_fraction.assign(nz, 0.0);
        accumulator.cloud_fraction_by_threshold.assign(threshold_count * nz, 0.0);
        accumulator.core_fraction.assign(nz, 0.0);
        accumulator.w_variance.assign(nz, 0.0);
        accumulator.u_variance.assign(nz, 0.0);
        accumulator.v_variance.assign(nz, 0.0);
        accumulator.mean_eddy_viscosity.assign(nz, 0.0);
        accumulator.mean_sgs_tke.assign(nz, 0.0);
        accumulator.mean_strain_squared.assign(nz, 0.0);
        accumulator.mean_sgs_dissipation.assign(nz, 0.0);
        accumulator.zero_eddy_viscosity_fraction.assign(nz, 0.0);
        accumulator.mean_theta_l_scalar_c.assign(nz, 0.0);
        accumulator.mean_qt_scalar_c.assign(nz, 0.0);
        accumulator.mean_theta_l_scalar_diffusivity.assign(nz, 0.0);
        accumulator.mean_qt_scalar_diffusivity.assign(nz, 0.0);
        accumulator.zero_theta_l_scalar_diffusivity_fraction.assign(nz, 0.0);
        accumulator.zero_qt_scalar_diffusivity_fraction.assign(nz, 0.0);
        accumulator.mean_theta_v.assign(nz, 0.0);
        accumulator.resolved_vw_flux.assign(nz, 0.0);
        accumulator.sgs_vw_flux.assign(nz, 0.0);
        accumulator.resolved_theta_l_flux.assign(nz, 0.0);
        accumulator.resolved_qt_flux.assign(nz, 0.0);
        accumulator.resolved_ql_flux.assign(nz, 0.0);
        accumulator.resolved_theta_v_flux.assign(nz, 0.0);
        accumulator.resolved_uw_flux.assign(nz, 0.0);
        accumulator.sgs_theta_l_flux.assign(nz, 0.0);
        accumulator.sgs_qt_flux.assign(nz, 0.0);
        accumulator.sgs_ql_flux.assign(nz, 0.0);
        accumulator.sgs_theta_v_flux.assign(nz, 0.0);
        accumulator.sgs_uw_flux.assign(nz, 0.0);
        accumulator.cloud_theta_l.assign(nz, 0.0);
        accumulator.core_theta_l.assign(nz, 0.0);
        accumulator.cloud_qt.assign(nz, 0.0);
        accumulator.core_qt.assign(nz, 0.0);
        accumulator.cloud_theta_v.assign(nz, 0.0);
        accumulator.core_theta_v.assign(nz, 0.0);
        accumulator.cloud_ql.assign(nz, 0.0);
        accumulator.core_ql.assign(nz, 0.0);
        accumulator.cloud_w.assign(nz, 0.0);
        accumulator.core_w.assign(nz, 0.0);
        accumulator.resolved_w_tke_flux.assign(nz, 0.0);
        accumulator.resolved_w_pressure_flux.assign(nz, 0.0);
        accumulator.wall_fluctuation_tke_work.assign(nz, 0.0);
        accumulator.cloud_conditional_samples.assign(nz, 0);
        accumulator.core_conditional_samples.assign(nz, 0);
        accumulator.total_cloud_cover_by_threshold.assign(threshold_count, 0.0);
    }
}

}  // namespace

double bomex_initial_theta_l(double z) {
    if (z <= 520.0) {
        return 298.7;
    }
    if (z <= 1480.0) {
        return linear_profile(z, 520.0, 298.7, 1480.0, 302.4);
    }
    if (z <= 2000.0) {
        return linear_profile(z, 1480.0, 302.4, 2000.0, 308.2);
    }
    return 308.2 + 3.65e-3 * (z - 2000.0);
}

double bomex_specific_to_mixing_ratio(double q) {
    if (q < 0.0 || q >= 1.0) {
        throw std::invalid_argument("specific humidity must lie in [0, 1)");
    }
    return q / (1.0 - q);
}

double bomex_mixing_to_specific_humidity(double r) {
    if (r < 0.0) {
        throw std::invalid_argument("mixing ratio must be non-negative");
    }
    return r / (1.0 + r);
}

double bomex_surface_qt_mixing_ratio_flux(double specified_flux) {
    // Linearize r=q/(1-q) about the specified BOMEX surface q_t=0.017.
    constexpr double surface_specific_humidity = 0.017;
    const double jacobian = 1.0 / std::pow(1.0 - surface_specific_humidity, 2.0);
    return specified_flux * jacobian;
}

double bomex_initial_qt(double z) {
    double qt_g_per_kg = 0.0;
    if (z <= 520.0) {
        qt_g_per_kg = linear_profile(z, 0.0, 17.0, 520.0, 16.3);
    } else if (z <= 1480.0) {
        qt_g_per_kg = linear_profile(z, 520.0, 16.3, 1480.0, 10.7);
    } else if (z <= 2000.0) {
        qt_g_per_kg = linear_profile(z, 1480.0, 10.7, 2000.0, 4.2);
    } else {
        qt_g_per_kg = 4.2 - 1.2e-3 * (z - 2000.0);
    }
    // Table B1 specifies moist-air specific humidity, while the solver
    // transports dry-air mixing ratio.
    return bomex_specific_to_mixing_ratio(std::max(0.0, 1.0e-3 * qt_g_per_kg));
}

double bomex_geostrophic_u(double z) {
    return -10.0 + 1.8e-3 * z;
}

double bomex_initial_u(double z) {
    return z <= 700.0 ? -8.75 : bomex_geostrophic_u(z);
}

double bomex_subsidence(double z) {
    if (z <= 1500.0) {
        return -0.0065 * z / 1500.0;
    }
    if (z <= 2100.0) {
        return -0.0065 * (2100.0 - z) / 600.0;
    }
    return 0.0;
}

double bomex_radiative_tendency(double z) {
    if (z <= 1500.0) {
        return -2.0 / seconds_per_day;
    }
    if (z <= 3000.0) {
        return (-2.0 / seconds_per_day) * (3000.0 - z) / 1500.0;
    }
    return 0.0;
}

double bomex_moisture_advection_tendency(double z) {
    if (z <= 300.0) {
        return -1.2e-8;
    }
    if (z <= 500.0) {
        return -1.2e-8 * (500.0 - z) / 200.0;
    }
    return 0.0;
}

const std::vector<double>& bomex_cloud_thresholds() {
    static const std::vector<double> thresholds{
        1.0e-8,
        1.0e-7,
        1.0e-6,
        1.0e-5,
    };
    return thresholds;
}

double bomex_column_water_large_scale_tendency(const FlowState& state, const Params& params) {
    if (params.initial_condition != "bomex") {
        return 0.0;
    }
    const std::vector<double> qt_mean = plane_mean(state.qt, params);
    double tendency = 0.0;
    for (int k = 0; k < params.nz; ++k) {
        const double z = (static_cast<double>(k) + 0.5) * params.dz();
        const double qt_specific = bomex_mixing_to_specific_humidity(qt_mean[static_cast<std::size_t>(k)]);
        const double jacobian = 1.0 / std::pow(1.0 - qt_specific, 2.0);
        tendency += (
            -bomex_subsidence(z) * vertical_derivative(qt_mean, k, params)
            + bomex_moisture_advection_tendency(z) * jacobian) * params.dz();
    }
    return tendency;
}

void add_bomex_large_scale_forcing(
    Field& rhs_u,
    Field& rhs_v,
    Field& rhs_theta_l,
    Field& rhs_qt,
    const FlowState& state,
    const Params& params) {
    if (params.initial_condition != "bomex") {
        return;
    }
    const std::vector<double> u_mean = plane_mean(state.u, params);
    const std::vector<double> v_mean = plane_mean(state.v, params);
    const std::vector<double> theta_l_mean = plane_mean(state.theta_l, params);
    const std::vector<double> qt_mean = plane_mean(state.qt, params);

    for (int k = 0; k < params.nz; ++k) {
        const double z = (static_cast<double>(k) + 0.5) * params.dz();
        const double subsidence = bomex_subsidence(z);
        const double u_tendency = -subsidence * vertical_derivative(u_mean, k, params);
        const double v_tendency = -subsidence * vertical_derivative(v_mean, k, params);
        const double theta_tendency = -subsidence * vertical_derivative(theta_l_mean, k, params)
            + bomex_radiative_tendency(z);
        const double qt_specific = bomex_mixing_to_specific_humidity(qt_mean[static_cast<std::size_t>(k)]);
        const double specific_to_mixing_jacobian = 1.0 / std::pow(1.0 - qt_specific, 2.0);
        const double qt_tendency = -subsidence * vertical_derivative(qt_mean, k, params)
            + bomex_moisture_advection_tendency(z) * specific_to_mixing_jacobian;
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                rhs_u[n] += u_tendency;
                rhs_v[n] += v_tendency;
                rhs_theta_l[n] += theta_tendency;
                rhs_qt[n] += qt_tendency;
            }
        }
    }
}

void add_bomex_sample(BomexAccumulator& accumulator, const FlowState& state, const Params& params) {
    if (!params.bomex_diagnostics_enabled || params.initial_condition != "bomex") {
        return;
    }
    ensure_profile_storage(accumulator, params);
    const Field w_center = centered_vertical_velocity(state.w, params);
    const std::vector<double> theta_l_mean = plane_mean(state.theta_l, params);
    const std::vector<double> qt_mean = plane_mean(state.qt, params);
    const std::vector<double> qv_mean = plane_mean(state.qv, params);
    const std::vector<double> ql_mean = plane_mean(state.ql, params);
    const std::vector<double> w_mean = plane_mean(w_center, params);
    const std::vector<double> u_mean = plane_mean(state.u, params);
    const std::vector<double> v_mean = plane_mean(state.v, params);
    const std::vector<double> p_mean = plane_mean(state.p, params);
    Field theta_v(state.theta.size(), 0.0);
    for (std::size_t n = 0; n < theta_v.size(); ++n) {
        theta_v[n] = state.theta[n] * (1.0 + 0.61 * state.qv[n] - state.ql[n]);
    }
    const std::vector<double> theta_v_mean = plane_mean(theta_v, params);

    const double inverse_plane = 1.0 / static_cast<double>(params.nx * params.ny);
    const auto& thresholds = bomex_cloud_thresholds();
    const std::size_t threshold_count = thresholds.size();
    std::vector<bool> cloudy_column(static_cast<std::size_t>(params.nx * params.ny), false);
    std::vector<bool> cloudy_column_by_threshold(
        threshold_count * static_cast<std::size_t>(params.nx * params.ny),
        false);
    double liquid_water_path = 0.0;
    double integrated_tke = 0.0;
    double sample_max_cloud_fraction = 0.0;
    for (int k = 0; k < params.nz; ++k) {
        double cloud_fraction = 0.0;
        std::vector<double> cloud_fraction_by_threshold(threshold_count, 0.0);
        double core_fraction = 0.0;
        double w_variance = 0.0;
        double u_variance = 0.0;
        double v_variance = 0.0;
        double tke = 0.0;
        double theta_flux = 0.0;
        double qt_flux = 0.0;
        double ql_flux = 0.0;
        double theta_v_flux = 0.0;
        double uw_flux = 0.0;
        double vw_flux = 0.0;
        double cloud_theta_l = 0.0;
        double core_theta_l = 0.0;
        double cloud_qt = 0.0;
        double core_qt = 0.0;
        double cloud_theta_v = 0.0;
        double core_theta_v = 0.0;
        double cloud_ql = 0.0;
        double core_ql = 0.0;
        double cloud_w = 0.0;
        double core_w = 0.0;
        std::size_t cloud_count = 0;
        std::size_t core_count = 0;
        double theta_l_scalar_c = 0.0;
        double qt_scalar_c = 0.0;
        double w_tke_flux = 0.0;
        double w_pressure_flux = 0.0;
        double wall_tke_work = 0.0;
        const bool wall_stress_is_local = params.wall_stress_model == "prescribed_ustar_local";
        const bool sample_wall_work = k == 0
            && params.momentum_wall_model == "abl"
            && (params.wall_stress_model == "prescribed_ustar" || wall_stress_is_local);
        double wall_local_drag = 0.0;
        if (sample_wall_work && wall_stress_is_local) {
            double mean_speed_u = 0.0;
            double mean_speed_v = 0.0;
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const std::size_t n = idx(params, i, j, 0);
                    const double speed = std::hypot(state.u[n], state.v[n]);
                    mean_speed_u += speed * state.u[n] * inverse_plane;
                    mean_speed_v += speed * state.v[n] * inverse_plane;
                }
            }
            wall_local_drag = prescribed_ustar_local_drag_coefficient(
                mean_speed_u, mean_speed_v, params.u_fric);
        }
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                const std::size_t column = static_cast<std::size_t>(j * params.nx + i);
                for (std::size_t threshold_index = 0; threshold_index < threshold_count; ++threshold_index) {
                    const bool cloudy_at_threshold = state.ql[n] > thresholds[threshold_index];
                    cloud_fraction_by_threshold[threshold_index] += cloudy_at_threshold ? inverse_plane : 0.0;
                    cloudy_column_by_threshold[threshold_index * static_cast<std::size_t>(params.nx * params.ny) + column] =
                        cloudy_column_by_threshold[
                            threshold_index * static_cast<std::size_t>(params.nx * params.ny) + column]
                        || cloudy_at_threshold;
                }
                const bool cloudy = state.ql[n] > thresholds.front();
                const bool core = cloudy && theta_v[n] > theta_v_mean[static_cast<std::size_t>(k)];
                cloud_fraction += cloudy ? inverse_plane : 0.0;
                core_fraction += core ? inverse_plane : 0.0;
                cloudy_column[column] = cloudy_column[column] || cloudy;
                const double w_prime = w_center[n] - w_mean[static_cast<std::size_t>(k)];
                const double u_prime = state.u[n] - u_mean[static_cast<std::size_t>(k)];
                const double v_prime = state.v[n] - v_mean[static_cast<std::size_t>(k)];
                w_variance += w_prime * w_prime * inverse_plane;
                u_variance += u_prime * u_prime * inverse_plane;
                v_variance += v_prime * v_prime * inverse_plane;
                tke += 0.5 * (u_prime*u_prime + v_prime*v_prime + w_prime*w_prime) * inverse_plane;
                theta_flux += w_prime * (state.theta_l[n] - theta_l_mean[static_cast<std::size_t>(k)]) * inverse_plane;
                qt_flux += w_prime * (state.qt[n] - qt_mean[static_cast<std::size_t>(k)]) * inverse_plane;
                ql_flux += w_prime * (state.ql[n] - ql_mean[static_cast<std::size_t>(k)]) * inverse_plane;
                theta_v_flux += w_prime * (theta_v[n] - theta_v_mean[static_cast<std::size_t>(k)]) * inverse_plane;
                uw_flux += u_prime * w_prime * inverse_plane;
                vw_flux += v_prime * w_prime * inverse_plane;
                const double resolved_tke_point =
                    0.5 * (u_prime * u_prime + v_prime * v_prime + w_prime * w_prime);
                w_tke_flux += w_prime * resolved_tke_point * inverse_plane;
                w_pressure_flux += w_prime
                    * (state.p[n] - p_mean[static_cast<std::size_t>(k)]) * inverse_plane;
                if (sample_wall_work) {
                    const double speed = std::hypot(state.u[n], state.v[n]);
                    if (speed > 1.0e-12) {
                        double tau_x;
                        double tau_y;
                        if (wall_stress_is_local) {
                            tau_x = -wall_local_drag * speed * state.u[n];
                            tau_y = -wall_local_drag * speed * state.v[n];
                        } else {
                            const double tau = -params.u_fric * params.u_fric;
                            tau_x = tau * state.u[n] / speed;
                            tau_y = tau * state.v[n] / speed;
                        }
                        wall_tke_work += (tau_x * u_prime + tau_y * v_prime)
                            / params.dz() * inverse_plane;
                    }
                }
                if (cloudy) {
                    ++cloud_count;
                    cloud_theta_l += state.theta_l[n];
                    cloud_qt += state.qt[n];
                    cloud_theta_v += theta_v[n];
                    cloud_ql += state.ql[n];
                    cloud_w += w_center[n];
                }
                if (core) {
                    ++core_count;
                    core_theta_l += state.theta_l[n];
                    core_qt += state.qt[n];
                    core_theta_v += theta_v[n];
                    core_ql += state.ql[n];
                    core_w += w_center[n];
                }
                theta_l_scalar_c += state.scalar_c[n] * inverse_plane;
                qt_scalar_c += state.qt_scalar_c[n] * inverse_plane;
            }
        }
        sample_max_cloud_fraction = std::max(sample_max_cloud_fraction, cloud_fraction);
        const double pressure = state.base_pressure[static_cast<std::size_t>(k)];
        const double temperature = theta_l_mean[static_cast<std::size_t>(k)] * exner_function(pressure);
        const ThermodynamicConstants constants;
        const double density = pressure / (constants.dry_air_gas_constant * temperature);
        liquid_water_path += density * ql_mean[static_cast<std::size_t>(k)] * params.dz();
        integrated_tke += density * tke * params.dz();

        const std::size_t kk = static_cast<std::size_t>(k);
        accumulator.theta_l[kk] += theta_l_mean[kk];
        accumulator.qt[kk] += qt_mean[kk];
        accumulator.qv[kk] += qv_mean[kk];
        accumulator.ql[kk] += ql_mean[kk];
        accumulator.u[kk] += u_mean[kk];
        accumulator.v[kk] += v_mean[kk];
        accumulator.mean_theta_v[kk] += theta_v_mean[kk];
        accumulator.cloud_fraction[kk] += cloud_fraction;
        for (std::size_t threshold_index = 0; threshold_index < threshold_count; ++threshold_index) {
            accumulator.cloud_fraction_by_threshold[threshold_index * static_cast<std::size_t>(params.nz) + kk] +=
                cloud_fraction_by_threshold[threshold_index];
        }
        accumulator.core_fraction[kk] += core_fraction;
        accumulator.w_variance[kk] += w_variance;
        accumulator.u_variance[kk] += u_variance;
        accumulator.v_variance[kk] += v_variance;
        accumulator.tke[kk] += tke;
        accumulator.resolved_theta_l_flux[kk] += theta_flux;
        accumulator.resolved_qt_flux[kk] += qt_flux;
        accumulator.resolved_ql_flux[kk] += ql_flux;
        accumulator.resolved_theta_v_flux[kk] += theta_v_flux;
        accumulator.resolved_uw_flux[kk] += uw_flux;
        accumulator.resolved_vw_flux[kk] += vw_flux;
        accumulator.cloud_theta_l[kk] += cloud_theta_l;
        accumulator.core_theta_l[kk] += core_theta_l;
        accumulator.cloud_qt[kk] += cloud_qt;
        accumulator.core_qt[kk] += core_qt;
        accumulator.cloud_theta_v[kk] += cloud_theta_v;
        accumulator.core_theta_v[kk] += core_theta_v;
        accumulator.cloud_ql[kk] += cloud_ql;
        accumulator.core_ql[kk] += core_ql;
        accumulator.cloud_w[kk] += cloud_w;
        accumulator.core_w[kk] += core_w;
        accumulator.cloud_conditional_samples[kk] += cloud_count;
        accumulator.core_conditional_samples[kk] += core_count;
        accumulator.mean_theta_l_scalar_c[kk] += theta_l_scalar_c;
        accumulator.mean_qt_scalar_c[kk] += qt_scalar_c;
        accumulator.resolved_w_tke_flux[kk] += w_tke_flux;
        accumulator.resolved_w_pressure_flux[kk] += w_pressure_flux;
        accumulator.wall_fluctuation_tke_work[kk] += wall_tke_work;
    }
    const double cloudy_columns = static_cast<double>(std::count(cloudy_column.begin(), cloudy_column.end(), true));
    const double sample_total_cloud_cover = cloudy_columns * inverse_plane;
    accumulator.total_cloud_cover += sample_total_cloud_cover;
    std::vector<double> sample_total_cloud_cover_by_threshold(threshold_count, 0.0);
    for (std::size_t threshold_index = 0; threshold_index < threshold_count; ++threshold_index) {
        const auto begin = cloudy_column_by_threshold.begin()
            + static_cast<std::ptrdiff_t>(threshold_index * static_cast<std::size_t>(params.nx * params.ny));
        const auto end = begin + static_cast<std::ptrdiff_t>(params.nx * params.ny);
        sample_total_cloud_cover_by_threshold[threshold_index] =
            static_cast<double>(std::count(begin, end, true)) * inverse_plane;
        accumulator.total_cloud_cover_by_threshold[threshold_index] += sample_total_cloud_cover_by_threshold[threshold_index];
    }
    accumulator.liquid_water_path += liquid_water_path;
    accumulator.sample_step.push_back(state.step_count);
    accumulator.sample_time_s.push_back(static_cast<double>(state.step_count) * params.dt);
    accumulator.sample_total_cloud_cover.push_back(sample_total_cloud_cover);
    accumulator.sample_total_cloud_cover_by_threshold.insert(
        accumulator.sample_total_cloud_cover_by_threshold.end(),
        sample_total_cloud_cover_by_threshold.begin(),
        sample_total_cloud_cover_by_threshold.end());
    accumulator.sample_max_cloud_fraction.push_back(sample_max_cloud_fraction);
    accumulator.sample_liquid_water_path.push_back(liquid_water_path);
    accumulator.sample_integrated_tke.push_back(integrated_tke);
    double column_qt = 0.0;
    for (double value : qt_mean) column_qt += value * params.dz();
    accumulator.sample_column_qt_m.push_back(column_qt);
    accumulator.sample_qt_large_scale_tendency_m_s.push_back(
        bomex_column_water_large_scale_tendency(state, params));
    ++accumulator.samples;
}

void print_bomex_summary(const BomexAccumulator& accumulator, const Params& params) {
    if (!params.bomex_diagnostics_enabled || params.initial_condition != "bomex") {
        return;
    }
    if (accumulator.samples == 0) {
        std::cout << "[bomex] no samples in configured averaging window\n";
        return;
    }
    const double inverse_samples = 1.0 / static_cast<double>(accumulator.samples);
    const double max_cloud_fraction = *std::max_element(
        accumulator.cloud_fraction.begin(), accumulator.cloud_fraction.end()) * inverse_samples;
    std::cout << "[bomex] samples=" << accumulator.samples
              << ", cloud_cover=" << accumulator.total_cloud_cover * inverse_samples
              << ", max_cloud_fraction=" << max_cloud_fraction
              << ", LWP=" << accumulator.liquid_water_path * inverse_samples << " kg/m2\n";
}

void write_bomex_outputs(const BomexAccumulator& accumulator, const FlowState& state, const Params& params) {
    if (!params.bomex_diagnostics_enabled || params.bomex_output_dir.empty()
        || params.initial_condition != "bomex" || accumulator.samples == 0) {
        return;
    }
    std::filesystem::create_directories(params.bomex_output_dir);
    const double inverse_samples = 1.0 / static_cast<double>(accumulator.samples);
    const auto averaged = [inverse_samples](double value) { return value * inverse_samples; };
    const double max_cloud_fraction = averaged(*std::max_element(
        accumulator.cloud_fraction.begin(), accumulator.cloud_fraction.end()));
    double cloud_base = -1.0;
    double cloud_top = -1.0;
    for (int k = 0; k < params.nz; ++k) {
        if (averaged(accumulator.cloud_fraction[static_cast<std::size_t>(k)]) > 0.0) {
            const double z = (static_cast<double>(k) + 0.5) * params.dz();
            if (cloud_base < 0.0) {
                cloud_base = z;
            }
            cloud_top = z;
        }
    }

    {
        std::ofstream out(std::filesystem::path(params.bomex_output_dir) / "bomex_summary.csv");
        if (!out) {
            throw std::runtime_error("failed to open BOMEX summary output");
        }
        out << "samples,total_cloud_cover,max_cloud_fraction,cloud_base_m,cloud_top_m,liquid_water_path_kg_m2,reference_cloud_cover,reference_max_cloud_fraction\n";
        out << accumulator.samples << ',' << std::setprecision(12)
            << averaged(accumulator.total_cloud_cover) << ',' << max_cloud_fraction << ','
            << cloud_base << ',' << cloud_top << ',' << averaged(accumulator.liquid_water_path)
            << ",0.13,0.06\n";
    }
    {
        std::ofstream out(std::filesystem::path(params.bomex_output_dir) / "bomex_timeseries.csv");
        if (!out) {
            throw std::runtime_error("failed to open BOMEX time series output");
        }
        const auto& thresholds = bomex_cloud_thresholds();
        out << "sample,step,time_s,time_h,total_cloud_cover";
        for (double threshold : thresholds) {
            out << ",total_cloud_cover_ql_gt_" << std::scientific << std::setprecision(0) << threshold;
        }
        out << std::defaultfloat
            << ",max_cloud_fraction,liquid_water_path_kg_m2,integrated_tke_kg_m-1_s-2,column_qt_m,"
               "qt_large_scale_tendency_m_s,surface_qt_flux_m_s,reference_cloud_cover\n";
        out << std::setprecision(12);
        for (std::size_t sample = 0; sample < accumulator.sample_total_cloud_cover.size(); ++sample) {
            const double time_s = accumulator.sample_time_s[sample];
            out << sample << ','
                << accumulator.sample_step[sample] << ','
                << time_s << ','
                << time_s / 3600.0 << ','
                << accumulator.sample_total_cloud_cover[sample];
            for (std::size_t threshold_index = 0; threshold_index < thresholds.size(); ++threshold_index) {
                const std::size_t offset = sample * thresholds.size() + threshold_index;
                out << ',' << accumulator.sample_total_cloud_cover_by_threshold[offset];
            }
            out << ','
                << accumulator.sample_max_cloud_fraction[sample] << ','
                << accumulator.sample_liquid_water_path[sample] << ','
                << accumulator.sample_integrated_tke[sample] << ','
                << accumulator.sample_column_qt_m[sample] << ','
                << accumulator.sample_qt_large_scale_tendency_m_s[sample] << ','
                << bomex_surface_qt_mixing_ratio_flux(params.surface_qv_flux) << ','
                << "0.13\n";
        }
    }
    {
        std::ofstream out(std::filesystem::path(params.bomex_output_dir) / "bomex_cloud_thresholds.csv");
        if (!out) {
            throw std::runtime_error("failed to open BOMEX threshold summary output");
        }
        out << "ql_threshold_kg_kg,total_cloud_cover,max_cloud_fraction,cloud_base_m,cloud_top_m\n";
        const auto& thresholds = bomex_cloud_thresholds();
        for (std::size_t threshold_index = 0; threshold_index < thresholds.size(); ++threshold_index) {
            double threshold_max_cloud_fraction = 0.0;
            double threshold_cloud_base = -1.0;
            double threshold_cloud_top = -1.0;
            for (int k = 0; k < params.nz; ++k) {
                const std::size_t offset = threshold_index * static_cast<std::size_t>(params.nz)
                    + static_cast<std::size_t>(k);
                const double fraction = averaged(accumulator.cloud_fraction_by_threshold[offset]);
                threshold_max_cloud_fraction = std::max(threshold_max_cloud_fraction, fraction);
                if (fraction > 0.0) {
                    const double z = (static_cast<double>(k) + 0.5) * params.dz();
                    if (threshold_cloud_base < 0.0) {
                        threshold_cloud_base = z;
                    }
                    threshold_cloud_top = z;
                }
            }
            out << std::setprecision(12)
                << thresholds[threshold_index] << ','
                << averaged(accumulator.total_cloud_cover_by_threshold[threshold_index]) << ','
                << threshold_max_cloud_fraction << ','
                << threshold_cloud_base << ','
                << threshold_cloud_top << '\n';
        }
    }
    {
        std::ofstream out(std::filesystem::path(params.bomex_output_dir) / "bomex_profiles.csv");
        if (!out) {
            throw std::runtime_error("failed to open BOMEX profile output");
        }
        out << "z_m,p0_pa,theta_l_k,qt_kg_kg,qv_kg_kg,ql_kg_kg,u_m_s,v_m_s,tke_m2_s2,"
               "cloud_fraction,core_fraction,w_variance_m2_s2,u_variance_m2_s2,v_variance_m2_s2,"
               "mean_eddy_viscosity_m2_s,mean_sgs_tke_m2_s2,mean_strain_squared_s-2,mean_sgs_dissipation_m2_s3,"
               "zero_eddy_viscosity_fraction,mean_theta_l_scalar_c,mean_qt_scalar_c,"
               "mean_theta_l_scalar_diffusivity_m2_s,mean_qt_scalar_diffusivity_m2_s,"
               "zero_theta_l_scalar_diffusivity_fraction,zero_qt_scalar_diffusivity_fraction,"
               "mean_theta_v_k,resolved_vw_flux_m2_s2,sgs_vw_flux_m2_s2,"
               "resolved_w_theta_l_flux_k_m_s,"
               "resolved_w_qt_flux_kg_kg_m_s,resolved_w_ql_flux_kg_kg_m_s,"
               "resolved_w_theta_v_flux_k_m_s,resolved_uw_flux_m2_s2,"
               "sgs_w_theta_l_flux_k_m_s,total_w_theta_l_flux_k_m_s,"
               "sgs_w_qt_flux_kg_kg_m_s,total_w_qt_flux_kg_kg_m_s,"
               "sgs_w_ql_flux_kg_kg_m_s,total_w_ql_flux_kg_kg_m_s,"
               "sgs_w_theta_v_flux_k_m_s,total_w_theta_v_flux_k_m_s,"
               "sgs_uw_flux_m2_s2,total_uw_flux_m2_s2,"
               "cloud_theta_l_k,core_theta_l_k,cloud_qt_kg_kg,core_qt_kg_kg,"
               "cloud_theta_v_k,core_theta_v_k,cloud_ql_kg_kg,core_ql_kg_kg,"
               "cloud_w_m_s,core_w_m_s,"
               "resolved_w_tke_flux_m3_s3,resolved_w_pressure_flux_m3_s3,"
               "wall_fluctuation_tke_work_m2_s3\n";
        out << std::setprecision(12);
        for (int k = 0; k < params.nz; ++k) {
            const std::size_t kk = static_cast<std::size_t>(k);
            const auto conditional = [](double sum, std::size_t count) {
                return count > 0 ? sum / static_cast<double>(count) : 0.0;
            };
            out << (static_cast<double>(k) + 0.5) * params.dz() << ','
                << state.base_pressure[kk] << ','
                << averaged(accumulator.theta_l[kk]) << ','
                << averaged(accumulator.qt[kk]) << ','
                << averaged(accumulator.qv[kk]) << ','
                << averaged(accumulator.ql[kk]) << ','
                << averaged(accumulator.u[kk]) << ','
                << averaged(accumulator.v[kk]) << ','
                << averaged(accumulator.tke[kk]) << ','
                << averaged(accumulator.cloud_fraction[kk]) << ','
                << averaged(accumulator.core_fraction[kk]) << ','
                << averaged(accumulator.w_variance[kk]) << ','
                << averaged(accumulator.u_variance[kk]) << ','
                << averaged(accumulator.v_variance[kk]) << ','
                << averaged(accumulator.mean_eddy_viscosity[kk]) << ','
                << averaged(accumulator.mean_sgs_tke[kk]) << ','
                << averaged(accumulator.mean_strain_squared[kk]) << ','
                << averaged(accumulator.mean_sgs_dissipation[kk]) << ','
                << averaged(accumulator.zero_eddy_viscosity_fraction[kk]) << ','
                << averaged(accumulator.mean_theta_l_scalar_c[kk]) << ','
                << averaged(accumulator.mean_qt_scalar_c[kk]) << ','
                << averaged(accumulator.mean_theta_l_scalar_diffusivity[kk]) << ','
                << averaged(accumulator.mean_qt_scalar_diffusivity[kk]) << ','
                << averaged(accumulator.zero_theta_l_scalar_diffusivity_fraction[kk]) << ','
                << averaged(accumulator.zero_qt_scalar_diffusivity_fraction[kk]) << ','
                << averaged(accumulator.mean_theta_v[kk]) << ','
                << averaged(accumulator.resolved_vw_flux[kk]) << ','
                << averaged(accumulator.sgs_vw_flux[kk]) << ','
                << averaged(accumulator.resolved_theta_l_flux[kk]) << ','
                << averaged(accumulator.resolved_qt_flux[kk]) << ','
                << averaged(accumulator.resolved_ql_flux[kk]) << ','
                << averaged(accumulator.resolved_theta_v_flux[kk]) << ','
                << averaged(accumulator.resolved_uw_flux[kk]) << ','
                << averaged(accumulator.sgs_theta_l_flux[kk]) << ','
                << averaged(accumulator.resolved_theta_l_flux[kk] + accumulator.sgs_theta_l_flux[kk]) << ','
                << averaged(accumulator.sgs_qt_flux[kk]) << ','
                << averaged(accumulator.resolved_qt_flux[kk] + accumulator.sgs_qt_flux[kk]) << ','
                << averaged(accumulator.sgs_ql_flux[kk]) << ','
                << averaged(accumulator.resolved_ql_flux[kk] + accumulator.sgs_ql_flux[kk]) << ','
                << averaged(accumulator.sgs_theta_v_flux[kk]) << ','
                << averaged(accumulator.resolved_theta_v_flux[kk] + accumulator.sgs_theta_v_flux[kk]) << ','
                << averaged(accumulator.sgs_uw_flux[kk]) << ','
                << averaged(accumulator.resolved_uw_flux[kk] + accumulator.sgs_uw_flux[kk]) << ','
                << conditional(accumulator.cloud_theta_l[kk], accumulator.cloud_conditional_samples[kk]) << ','
                << conditional(accumulator.core_theta_l[kk], accumulator.core_conditional_samples[kk]) << ','
                << conditional(accumulator.cloud_qt[kk], accumulator.cloud_conditional_samples[kk]) << ','
                << conditional(accumulator.core_qt[kk], accumulator.core_conditional_samples[kk]) << ','
                << conditional(accumulator.cloud_theta_v[kk], accumulator.cloud_conditional_samples[kk]) << ','
                << conditional(accumulator.core_theta_v[kk], accumulator.core_conditional_samples[kk]) << ','
                << conditional(accumulator.cloud_ql[kk], accumulator.cloud_conditional_samples[kk]) << ','
                << conditional(accumulator.core_ql[kk], accumulator.core_conditional_samples[kk]) << ','
                << conditional(accumulator.cloud_w[kk], accumulator.cloud_conditional_samples[kk]) << ','
                << conditional(accumulator.core_w[kk], accumulator.core_conditional_samples[kk]) << ','
                << averaged(accumulator.resolved_w_tke_flux[kk]) << ','
                << averaged(accumulator.resolved_w_pressure_flux[kk]) << ','
                << averaged(accumulator.wall_fluctuation_tke_work[kk]) << '\n';
        }
    }
    {
        std::ofstream out(std::filesystem::path(params.bomex_output_dir) / "bomex_surface_fluxes.csv");
        if (!out) {
            throw std::runtime_error("failed to open BOMEX surface-flux output");
        }
        const double inv_samples = 1.0 / static_cast<double>(accumulator.samples);
        const double theta_l_surface = accumulator.theta_l.front() * inv_samples;
        const double qv_surface = accumulator.qv.front() * inv_samples;
        const double u_surface = accumulator.u.front() * inv_samples;
        const double v_surface = accumulator.v.front() * inv_samples;
        const double speed = std::hypot(u_surface, v_surface);
        const double qt_flux = bomex_surface_qt_mixing_ratio_flux(params.surface_qv_flux);
        const double theta_v_flux =
            (1.0 + 0.61 * qv_surface) * params.surface_theta_flux
            + 0.61 * theta_l_surface * qt_flux;
        const double uw_flux = speed > 0.0 ? -params.u_fric * params.u_fric * u_surface / speed : 0.0;
        out << "z_m,total_w_theta_l_flux_k_m_s,total_w_qt_flux_kg_kg_m_s,"
               "total_w_ql_flux_kg_kg_m_s,total_w_theta_v_flux_k_m_s,total_uw_flux_m2_s2\n";
        out << std::setprecision(12) << 0.0 << ',' << params.surface_theta_flux << ',' << qt_flux << ','
            << 0.0 << ',' << theta_v_flux << ',' << uw_flux << '\n';
    }
    {
        std::ofstream out(std::filesystem::path(params.bomex_output_dir) / "bomex_cloud_fraction_threshold_profiles.csv");
        if (!out) {
            throw std::runtime_error("failed to open BOMEX threshold profile output");
        }
        const auto& thresholds = bomex_cloud_thresholds();
        out << "z_m";
        for (double threshold : thresholds) {
            out << ",cloud_fraction_ql_gt_" << std::scientific << std::setprecision(0) << threshold;
        }
        out << std::defaultfloat << '\n';
        out << std::setprecision(12);
        for (int k = 0; k < params.nz; ++k) {
            const std::size_t kk = static_cast<std::size_t>(k);
            out << (static_cast<double>(k) + 0.5) * params.dz();
            for (std::size_t threshold_index = 0; threshold_index < thresholds.size(); ++threshold_index) {
                const std::size_t offset = threshold_index * static_cast<std::size_t>(params.nz) + kk;
                out << ',' << averaged(accumulator.cloud_fraction_by_threshold[offset]);
            }
            out << '\n';
        }
    }
    std::cout << "[bomex] wrote comparison diagnostics to " << params.bomex_output_dir << '\n';
}

}  // namespace wireles

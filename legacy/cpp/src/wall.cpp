#include "wireles/wall.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace wireles {

WallStress dynamic_neutral_wall_stress(const FlowState& state, const Params& params, FftwXY& fft) {
    const std::size_t plane_size = static_cast<std::size_t>(params.nx) * static_cast<std::size_t>(params.ny);
    WallStress stress{
        Field(plane_size, 0.0),
        Field(plane_size, 0.0),
        Field(plane_size, 0.0),
    };

    if (params.momentum_wall_model != "abl") {
        return stress;
    }
    if (params.nz < 3) {
        throw std::runtime_error("ABL wall stress requires nz >= 3");
    }
    if (params.wall_ref_height() <= params.zo) {
        throw std::runtime_error("ABL wall stress requires wall_ref_height > zo");
    }

    Field u0(plane_size, 0.0);
    Field v0(plane_size, 0.0);
    if (params.wall_stress_model == "dynamic_neutral") {
        const double filter_width = params.fgr * params.tfr;
        fft.filter_plane(state.u, 0, u0, params, filter_width);
        fft.filter_plane(state.v, 0, v0, params, filter_width);
    } else {
        // The BOMEX prescribed-u* boundary condition defines the stress
        // direction from the velocity at the lowest grid level itself.  A
        // test-filtered direction belongs to the dynamic wall model and would
        // leave the near-wall high-wavenumber velocity fluctuations unstressed.
        std::copy_n(state.u.begin(), plane_size, u0.begin());
        std::copy_n(state.v.begin(), plane_size, v0.begin());
    }

    const double denom = std::log(params.wall_ref_height() / params.zo);
    constexpr double eps = 1.0e-12;
    double local_drag = 0.0;
    if (params.wall_stress_model == "prescribed_ustar_local") {
        const double inverse_plane = 1.0 / static_cast<double>(plane_size);
        double mean_speed_u = 0.0;
        double mean_speed_v = 0.0;
        for (std::size_t n = 0; n < plane_size; ++n) {
            const double speed = std::sqrt(u0[n] * u0[n] + v0[n] * v0[n]);
            mean_speed_u += speed * u0[n] * inverse_plane;
            mean_speed_v += speed * v0[n] * inverse_plane;
        }
        local_drag = prescribed_ustar_local_drag_coefficient(
            mean_speed_u, mean_speed_v, params.u_fric);
    }
    for (int j = 0; j < params.ny; ++j) {
        for (int i = 0; i < params.nx; ++i) {
            const std::size_t n = static_cast<std::size_t>(j) * static_cast<std::size_t>(params.nx) + static_cast<std::size_t>(i);
            const double speed = std::sqrt(u0[n] * u0[n] + v0[n] * v0[n]);
            if (speed <= eps || std::abs(denom) <= eps) {
                continue;
            }
            if (params.wall_stress_model == "prescribed_ustar_local") {
                // Log-law-consistent quadratic local drag, rescaled every
                // step so the plane-mean stress magnitude stays at the
                // prescribed u*^2.  Unlike the uniform-magnitude form this
                // preferentially retards fast streaks and therefore does
                // negative work on the first-level velocity fluctuations.
                stress.tau_xz[n] = -local_drag * speed * u0[n];
                stress.tau_yz[n] = -local_drag * speed * v0[n];
                stress.ustar[n] = std::sqrt(local_drag) * speed;
                continue;
            }
            double ustar = params.u_fric;
            if (params.wall_stress_model == "dynamic_neutral") {
                ustar = speed * params.vonk / denom;
            }
            const double tau = -(ustar * ustar);
            stress.tau_xz[n] = tau * u0[n] / speed;
            stress.tau_yz[n] = tau * v0[n] / speed;
            stress.ustar[n] = ustar;
        }
    }
    return stress;
}

double prescribed_ustar_local_drag_coefficient(
    double plane_mean_speed_times_u,
    double plane_mean_speed_times_v,
    double u_fric) {
    const double magnitude = std::hypot(plane_mean_speed_times_u, plane_mean_speed_times_v);
    return magnitude > 1.0e-12 ? u_fric * u_fric / magnitude : 0.0;
}

std::array<double, 2> wall_model_mean_velocity_gradient(
    double mean_u,
    double mean_v,
    const Params& params) {
    const double speed = std::hypot(mean_u, mean_v);
    if (speed <= 1.0e-12 || params.vonk <= 0.0 || params.wall_ref_height() <= 0.0) {
        return {0.0, 0.0};
    }
    double ustar = params.u_fric;
    if (params.wall_stress_model == "dynamic_neutral") {
        const double log_argument = params.wall_ref_height() / params.zo;
        if (log_argument <= 1.0) {
            return {0.0, 0.0};
        }
        ustar = speed * params.vonk / std::log(log_argument);
    }
    const double magnitude = ustar / (params.vonk * params.wall_ref_height());
    return {magnitude * mean_u / speed, magnitude * mean_v / speed};
}

void apply_wall_stress(Field& rhs_u, Field& rhs_v, const FlowState& state, const Params& params, FftwXY& fft) {
    if (params.momentum_wall_model != "abl") {
        return;
    }
    const WallStress stress = dynamic_neutral_wall_stress(state, params, fft);
    const double inv_dz = 1.0 / params.dz();
    for (int j = 0; j < params.ny; ++j) {
        for (int i = 0; i < params.nx; ++i) {
            const std::size_t plane = static_cast<std::size_t>(j) * static_cast<std::size_t>(params.nx) + static_cast<std::size_t>(i);
            const std::size_t cell = idx(params, i, j, 0);
            rhs_u[cell] += stress.tau_xz[plane] * inv_dz;
            rhs_v[cell] += stress.tau_yz[plane] * inv_dz;
        }
    }
}

}  // namespace wireles

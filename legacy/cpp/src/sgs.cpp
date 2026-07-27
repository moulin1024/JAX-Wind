#include "wireles/sgs.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "wireles/operators.hpp"
#include "wireles/scalar.hpp"
#include "wireles/wall.hpp"

namespace wireles {
namespace {

using SymFields = std::array<Field, 6>;
using VecFields = std::array<Field, 3>;

double safe_divide(double num, double den) {
    return std::abs(den) > 1.0e-30 ? num / den : 0.0;
}

double sym_dot_at(const std::array<double, 6>& a, const std::array<double, 6>& b) {
    return a[0] * b[0] + 2.0 * a[1] * b[1] + 2.0 * a[2] * b[2] + a[3] * b[3] + 2.0 * a[4] * b[4] + a[5] * b[5];
}

double sym_strain_magnitude_at(const std::array<double, 6>& sij) {
    return std::sqrt(std::max(2.0 * sym_dot_at(sij, sij), 0.0));
}

SymFields strain_components(const VelocityGradients& grad, const Params& params) {
    SymFields sij;
    for (Field& q : sij) {
        q.assign(params.real_size(), 0.0);
    }
    for (std::size_t n = 0; n < params.real_size(); ++n) {
        sij[0][n] = grad.dudx[n];
        sij[1][n] = 0.5 * (grad.dudy[n] + grad.dvdx[n]);
        sij[2][n] = 0.5 * (grad.dudz[n] + grad.dwdx[n]);
        sij[3][n] = grad.dvdy[n];
        sij[4][n] = 0.5 * (grad.dvdz[n] + grad.dwdy[n]);
        sij[5][n] = grad.dwdz[n];
    }
    return sij;
}

VecFields centered_velocity(const FlowState& state, const Params& params) {
    return VecFields{state.u, state.v, w_to_center(state.w, params)};
}

struct LmMm {
    Field lm;
    Field mm;
    Field strain_hat;
};

LmMm momentum_lm_mm(
    const FlowState& state,
    const VecFields& vel,
    const SymFields& sij,
    const Field& strain,
    const Params& params,
    FftwXY& fft,
    double test_ratio) {
    VecFields vel_hat;
    for (int c = 0; c < 3; ++c) {
        vel_hat[c] = horizontal_spectral_filter(vel[c], params, fft, params.fgr * test_ratio);
    }

    SymFields uu;
    uu[0].assign(params.real_size(), 0.0);
    uu[1].assign(params.real_size(), 0.0);
    uu[2].assign(params.real_size(), 0.0);
    uu[3].assign(params.real_size(), 0.0);
    uu[4].assign(params.real_size(), 0.0);
    uu[5].assign(params.real_size(), 0.0);
    SymFields ssij;
    for (Field& q : ssij) {
        q.assign(params.real_size(), 0.0);
    }
    for (std::size_t n = 0; n < params.real_size(); ++n) {
        uu[0][n] = vel[0][n] * vel[0][n];
        uu[1][n] = vel[0][n] * vel[1][n];
        uu[3][n] = vel[1][n] * vel[1][n];
        uu[5][n] = vel[2][n] * vel[2][n];
        for (int c = 0; c < 6; ++c) {
            ssij[c][n] = strain[n] * sij[c][n];
        }
    }

    SymFields uu_hat;
    SymFields sij_hat;
    SymFields ssij_hat;
    for (int c = 0; c < 6; ++c) {
        uu_hat[c] = horizontal_spectral_filter(uu[c], params, fft, params.fgr * test_ratio);
        sij_hat[c] = horizontal_spectral_filter(sij[c], params, fft, params.fgr * test_ratio);
        ssij_hat[c] = horizontal_spectral_filter(ssij[c], params, fft, params.fgr * test_ratio);
    }

    LmMm out;
    out.lm.assign(params.real_size(), 0.0);
    out.mm.assign(params.real_size(), 0.0);
    out.strain_hat.assign(params.real_size(), 0.0);
    const double delta2 = params.sgs_delta() * params.sgs_delta();
    const double ratio2 = test_ratio * test_ratio;
    for (std::size_t n = 0; n < params.real_size(); ++n) {
        const std::array<double, 6> l{
            uu_hat[0][n] - vel_hat[0][n] * vel_hat[0][n],
            uu_hat[1][n] - vel_hat[0][n] * vel_hat[1][n],
            uu_hat[2][n] - vel_hat[0][n] * vel_hat[2][n],
            uu_hat[3][n] - vel_hat[1][n] * vel_hat[1][n],
            uu_hat[4][n] - vel_hat[1][n] * vel_hat[2][n],
            uu_hat[5][n] - vel_hat[2][n] * vel_hat[2][n],
        };
        const std::array<double, 6> sh{
            sij_hat[0][n], sij_hat[1][n], sij_hat[2][n], sij_hat[3][n], sij_hat[4][n], sij_hat[5][n],
        };
        out.strain_hat[n] = sym_strain_magnitude_at(sh);
        std::array<double, 6> m{};
        for (int c = 0; c < 6; ++c) {
            m[c] = 2.0 * delta2 * (ssij_hat[c][n] - ratio2 * out.strain_hat[n] * sij_hat[c][n]);
        }
        out.lm[n] = sym_dot_at(l, m);
        out.mm[n] = sym_dot_at(m, m);
    }
    return out;
}

void update_lagrangian_velocity(FlowState& state, const Params& params) {
    const Field w_center = w_to_center(state.w, params);
    const double inv_count = 1.0 / static_cast<double>(params.cs_count);
    for (std::size_t n = 0; n < params.real_size(); ++n) {
        state.u_lag[n] += state.u[n] * inv_count;
        state.v_lag[n] += state.v[n] * inv_count;
        state.w_lag[n] += w_center[n] * inv_count;
    }
}

Field eddy_viscosity_from_cs2(const Field& cs2, const Field& strain, const Params& params) {
    Field nu_t(params.real_size(), 0.0);
    const double delta2 = params.sgs_delta() * params.sgs_delta();
    for (std::size_t n = 0; n < nu_t.size(); ++n) {
        nu_t[n] = std::max(cs2[n], 0.0) * delta2 * strain[n];
    }
    return nu_t;
}

}  // namespace

Field horizontal_spectral_filter(const Field& in, const Params& params, FftwXY& fft, double filter_width) {
    Field out;
    fft.filter_planes_fortran_sharp(in, params.nz, out, params, filter_width);
    return out;
}

void apply_center_history_bc(Field& q, const Params& params) {
    if (params.nz < 2) {
        return;
    }
    for (int j = 0; j < params.ny; ++j) {
        for (int i = 0; i < params.nx; ++i) {
            q[idx(params, i, j, 0)] = q[idx(params, i, j, 1)];
            q[idx(params, i, j, params.nz - 1)] = q[idx(params, i, j, params.nz - 2)];
        }
    }
}

Field lagrangian_interp_center(
    const Field& q,
    const Field& u_lag,
    const Field& v_lag,
    const Field& w_lag,
    const Params& params) {
    Field out(params.real_size(), 0.0);
    const double dt_lag = params.dt * static_cast<double>(params.cs_count);
    auto periodic_coordinate = [](double coordinate, int extent) {
        coordinate = std::fmod(coordinate, static_cast<double>(extent));
        return coordinate < 0.0 ? coordinate + static_cast<double>(extent) : coordinate;
    };
    for (int k = 0; k < params.nz; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                const double xi = periodic_coordinate(
                    static_cast<double>(i) - u_lag[n] * dt_lag / params.dx(), params.nx);
                const double eta = periodic_coordinate(
                    static_cast<double>(j) - v_lag[n] * dt_lag / params.dy(), params.ny);
                const double zeta = std::clamp(
                    static_cast<double>(k) - w_lag[n] * dt_lag / params.dz(),
                    0.0,
                    static_cast<double>(params.nz - 1));
                const int i0 = static_cast<int>(std::floor(xi));
                const int j0 = static_cast<int>(std::floor(eta));
                const int k0 = static_cast<int>(std::floor(zeta));
                const int i1 = (i0 + 1) % params.nx;
                const int j1 = (j0 + 1) % params.ny;
                const int k1 = std::min(k0 + 1, params.nz - 1);
                const double fx = xi - static_cast<double>(i0);
                const double fy = eta - static_cast<double>(j0);
                const double fz = zeta - static_cast<double>(k0);
                const double q00 = (1.0 - fx) * q[idx(params, i0, j0, k0)]
                    + fx * q[idx(params, i1, j0, k0)];
                const double q10 = (1.0 - fx) * q[idx(params, i0, j1, k0)]
                    + fx * q[idx(params, i1, j1, k0)];
                const double q01 = (1.0 - fx) * q[idx(params, i0, j0, k1)]
                    + fx * q[idx(params, i1, j0, k1)];
                const double q11 = (1.0 - fx) * q[idx(params, i0, j1, k1)]
                    + fx * q[idx(params, i1, j1, k1)];
                const double q0 = (1.0 - fy) * q00 + fy * q10;
                const double q1 = (1.0 - fy) * q01 + fy * q11;
                out[n] = (1.0 - fz) * q0 + fz * q1;
            }
        }
    }
    return out;
}

std::pair<Field, Field> lagrangian_average_fields(
    const Field& current_a,
    const Field& current_b,
    const Field& old_a,
    const Field& old_b,
    const Field& u_lag,
    const Field& v_lag,
    const Field& w_lag,
    const Params& params,
    const Field* timescale_a,
    const Field* timescale_b) {
    const Field a_interp = lagrangian_interp_center(old_a, u_lag, v_lag, w_lag, params);
    const Field b_interp = lagrangian_interp_center(old_b, u_lag, v_lag, w_lag, params);
    Field avg_a(params.real_size(), 0.0);
    Field avg_b(params.real_size(), 0.0);
    const double dt_lag = params.dt * static_cast<double>(params.cs_count);
    for (std::size_t n = 0; n < params.real_size(); ++n) {
        const double time_a = timescale_a == nullptr ? old_a[n] : (*timescale_a)[n];
        const double time_b = timescale_b == nullptr ? old_b[n] : (*timescale_b)[n];
        const double product = time_a * time_b;
        const bool valid = time_a > 0.0 && time_b >= 0.0 && product > 0.0;
        double eps = 0.0;
        if (valid) {
            const double tn = 1.5 * params.sgs_delta() * std::pow(product, -0.125);
            eps = (dt_lag / tn) / (1.0 + dt_lag / tn);
        }
        const double raw_a = eps * current_a[n] + (1.0 - eps) * a_interp[n];
        const double raw_b = eps * current_b[n] + (1.0 - eps) * b_interp[n];
        // The legacy NCAR momentum LASD transports the signed numerator.
        // Calaf et al. (2011), Eq. (A10), applies H{.} to scalar histories;
        // scalar callers are identified by their explicit common-timescale
        // fields below.
        const bool ramp_scalar_numerator = timescale_a != nullptr && timescale_b != nullptr;
        avg_a[n] = ramp_scalar_numerator ? (raw_a > 0.0 ? raw_a : 1.0e-32) : raw_a;
        avg_b[n] = std::max(raw_b, 0.0);
    }
    return {avg_a, avg_b};
}

void reset_lasd_velocity_accumulators(FlowState& state) {
    std::fill(state.u_lag.begin(), state.u_lag.end(), 0.0);
    std::fill(state.v_lag.begin(), state.v_lag.end(), 0.0);
    std::fill(state.w_lag.begin(), state.w_lag.end(), 0.0);
}

VelocityGradients velocity_gradients(const FlowState& state, const Params& params, FftwXY& fft) {
    VelocityGradients grad;
    fft.derivative_x(state.u, grad.dudx, params);
    fft.derivative_y(state.u, grad.dudy, params);
    fft.derivative_x(state.v, grad.dvdx, params);
    fft.derivative_y(state.v, grad.dvdy, params);
    const Field w_center = w_to_center(state.w, params);
    fft.derivative_x(w_center, grad.dwdx, params);
    fft.derivative_y(w_center, grad.dwdy, params);
    grad.dudz = ddz_center(state.u, params);
    grad.dvdz = ddz_center(state.v, params);
    grad.dwdz = ddz_w_to_center(state.w, params);
    if (params.amd_wall_model_gradients && params.nz >= 2) {
        const double inverse_plane = 1.0 / static_cast<double>(params.nx * params.ny);
        double mean_u0 = 0.0;
        double mean_v0 = 0.0;
        double mean_u1 = 0.0;
        double mean_v1 = 0.0;
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                mean_u0 += state.u[idx(params, i, j, 0)] * inverse_plane;
                mean_v0 += state.v[idx(params, i, j, 0)] * inverse_plane;
                mean_u1 += state.u[idx(params, i, j, 1)] * inverse_plane;
                mean_v1 += state.v[idx(params, i, j, 1)] * inverse_plane;
            }
        }
        const auto mean_gradient = wall_model_mean_velocity_gradient(mean_u0, mean_v0, params);
        const double inverse_dz = 1.0 / params.dz();
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t lower = idx(params, i, j, 0);
                const std::size_t upper = idx(params, i, j, 1);
                grad.dudz[lower] = mean_gradient[0]
                    + ((state.u[upper] - mean_u1) - (state.u[lower] - mean_u0)) * inverse_dz;
                grad.dvdz[lower] = mean_gradient[1]
                    + ((state.v[upper] - mean_v1) - (state.v[lower] - mean_v0)) * inverse_dz;
            }
        }
    }
    return grad;
}

std::array<double, 3> amd_scaled_cell_width(const Params& params) {
    // Abkar et al. (2016): 1/sqrt(12) for the pseudo-spectral horizontal
    // derivatives and 1/sqrt(3) for the second-order vertical derivative.
    // With sharp 2/3 dealiasing every state field is truncated at the
    // wavenumber pi/(1.5 dx), so the horizontal length entering the
    // minimum-dissipation bound is the effective filter width 1.5 dx (the
    // same two_thirds_filter_width the truncation itself uses), not the raw
    // grid spacing.  There is no vertical truncation, so dz is unchanged.
    const bool sharp_dealias_active = params.amd_dealiased_cell_width
        && params.horizontal_dealias && params.dealiasing == "sharp";
    const double horizontal_width_scale = sharp_dealias_active ? 1.5 : 1.0;
    return {
        horizontal_width_scale * params.dx() / std::sqrt(12.0),
        horizontal_width_scale * params.dy() / std::sqrt(12.0),
        params.dz() / std::sqrt(3.0),
    };
}

double amd_eddy_viscosity_at(
    const std::array<double, 9>& velocity_gradient,
    const std::array<double, 3>& buoyancy_gradient,
    const std::array<double, 3>& scaled_cell_width) {
    auto gradient = [&](int component, int direction) {
        return velocity_gradient[static_cast<std::size_t>(3 * component + direction)];
    };
    double numerator = 0.0;
    double denominator = 0.0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const double sij = 0.5 * (gradient(i, j) + gradient(j, i));
            double scaled_gradient_product = 0.0;
            for (int k = 0; k < 3; ++k) {
                const double gi = scaled_cell_width[static_cast<std::size_t>(k)] * gradient(i, k);
                const double gj = scaled_cell_width[static_cast<std::size_t>(k)] * gradient(j, k);
                scaled_gradient_product += gi * gj;
            }
            numerator -= scaled_gradient_product * sij;
        }
        for (int k = 0; k < 3; ++k) {
            const double value = gradient(i, k);
            denominator += value * value;
        }
    }
    for (int k = 0; k < 3; ++k) {
        const double length = scaled_cell_width[static_cast<std::size_t>(k)];
        numerator += (length * gradient(2, k))
            * (length * buoyancy_gradient[static_cast<std::size_t>(k)]);
    }
    if (!(denominator > 0.0) || !std::isfinite(numerator) || !std::isfinite(denominator)) {
        return 0.0;
    }
    return std::max(numerator, 0.0) / denominator;
}

double amd_invariant_ratio(const AmdInvariant& invariant) {
    if (!(invariant.denominator > 0.0)
        || !std::isfinite(invariant.numerator)
        || !std::isfinite(invariant.denominator)) {
        return 0.0;
    }
    return std::max(invariant.numerator, 0.0) / invariant.denominator;
}

void smooth_amd_invariant_field(Field& q, const Params& params) {
    if (q.size() != params.real_size()) {
        throw std::runtime_error("AMD invariant smoothing field-size mismatch");
    }
    const Field src = q;
    constexpr std::array<double, 3> stencil{0.25, 0.5, 0.25};
    for (int k = 0; k < params.nz; ++k) {
        double vertical_weight = 0.0;
        for (int dk = -1; dk <= 1; ++dk) {
            const int kk = k + dk;
            if (kk >= 0 && kk < params.nz) {
                vertical_weight += stencil[static_cast<std::size_t>(dk + 1)];
            }
        }
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                double accumulated = 0.0;
                for (int dk = -1; dk <= 1; ++dk) {
                    const int kk = k + dk;
                    if (kk < 0 || kk >= params.nz) {
                        continue;
                    }
                    const double wz = stencil[static_cast<std::size_t>(dk + 1)];
                    for (int dj = -1; dj <= 1; ++dj) {
                        const int jj = (j + dj + params.ny) % params.ny;
                        const double wyz = wz * stencil[static_cast<std::size_t>(dj + 1)];
                        for (int di = -1; di <= 1; ++di) {
                            const int ii = (i + di + params.nx) % params.nx;
                            accumulated += wyz * stencil[static_cast<std::size_t>(di + 1)]
                                * src[idx(params, ii, jj, kk)];
                        }
                    }
                }
                q[idx(params, i, j, k)] = accumulated / vertical_weight;
            }
        }
    }
}

double amd_eddy_viscosity_staggered_at(
    const std::array<double, 9>& center_velocity_gradient,
    const std::array<double, 4>& lower_face_gradient,
    const std::array<double, 4>& upper_face_gradient,
    const std::array<double, 3>& buoyancy_gradient,
    const std::array<double, 3>& scaled_cell_width) {
    return amd_invariant_ratio(amd_eddy_viscosity_staggered_invariant_at(
        center_velocity_gradient,
        lower_face_gradient,
        upper_face_gradient,
        buoyancy_gradient,
        scaled_cell_width));
}

AmdInvariant amd_eddy_viscosity_staggered_invariant_at(
    const std::array<double, 9>& center_velocity_gradient,
    const std::array<double, 4>& lower_face_gradient,
    const std::array<double, 4>& upper_face_gradient,
    const std::array<double, 3>& buoyancy_gradient,
    const std::array<double, 3>& scaled_cell_width) {
    auto center = [&](int component, int direction) {
        return center_velocity_gradient[static_cast<std::size_t>(3 * component + direction)];
    };
    auto face_product_mean = [&](int a, int b) {
        return 0.5 * (lower_face_gradient[static_cast<std::size_t>(a)]
                * lower_face_gradient[static_cast<std::size_t>(b)]
            + upper_face_gradient[static_cast<std::size_t>(a)]
                * upper_face_gradient[static_cast<std::size_t>(b)]);
    };

    const double s11 = center(0, 0);
    const double s22 = center(1, 1);
    const double s33 = center(2, 2);
    const double s12 = 0.5 * (center(0, 1) + center(1, 0));
    const double lower_s13 = 0.5 * (lower_face_gradient[0] + lower_face_gradient[2]);
    const double upper_s13 = 0.5 * (upper_face_gradient[0] + upper_face_gradient[2]);
    const double lower_s23 = 0.5 * (lower_face_gradient[1] + lower_face_gradient[3]);
    const double upper_s23 = 0.5 * (upper_face_gradient[1] + upper_face_gradient[3]);
    const double face_wx_s13 = 0.5 * (
        lower_face_gradient[2] * lower_s13 + upper_face_gradient[2] * upper_s13);
    const double face_wx_s23 = 0.5 * (
        lower_face_gradient[2] * lower_s23 + upper_face_gradient[2] * upper_s23);
    const double face_wy_s13 = 0.5 * (
        lower_face_gradient[3] * lower_s13 + upper_face_gradient[3] * upper_s13);
    const double face_wy_s23 = 0.5 * (
        lower_face_gradient[3] * lower_s23 + upper_face_gradient[3] * upper_s23);
    const double face_uz_s13 = 0.5 * (
        lower_face_gradient[0] * lower_s13 + upper_face_gradient[0] * upper_s13);
    const double face_vz_s23 = 0.5 * (
        lower_face_gradient[1] * lower_s23 + upper_face_gradient[1] * upper_s23);

    auto horizontal_contraction = [&](int direction, int w_face_component) {
        const double u_direction = center(0, direction);
        const double v_direction = center(1, direction);
        const double w2 = face_product_mean(w_face_component, w_face_component);
        const double w_s13 = direction == 0 ? face_wx_s13 : face_wy_s13;
        const double w_s23 = direction == 0 ? face_wx_s23 : face_wy_s23;
        return s11 * u_direction * u_direction
            + s22 * v_direction * v_direction
            + s33 * w2
            + 2.0 * s12 * u_direction * v_direction
            + 2.0 * u_direction * w_s13
            + 2.0 * v_direction * w_s23;
    };

    const double vertical_contraction =
        s11 * face_product_mean(0, 0)
        + s22 * face_product_mean(1, 1)
        + s33 * center(2, 2) * center(2, 2)
        + 2.0 * s12 * face_product_mean(0, 1)
        + 2.0 * center(2, 2) * face_uz_s13
        + 2.0 * center(2, 2) * face_vz_s23;

    double numerator =
        -scaled_cell_width[0] * scaled_cell_width[0] * horizontal_contraction(0, 2)
        -scaled_cell_width[1] * scaled_cell_width[1] * horizontal_contraction(1, 3)
        -scaled_cell_width[2] * scaled_cell_width[2] * vertical_contraction;
    for (int direction = 0; direction < 3; ++direction) {
        const double length = scaled_cell_width[static_cast<std::size_t>(direction)];
        numerator += length * length * center(2, direction)
            * buoyancy_gradient[static_cast<std::size_t>(direction)];
    }

    const double denominator =
        center(0, 0) * center(0, 0) + center(1, 0) * center(1, 0)
            + face_product_mean(2, 2)
        + center(0, 1) * center(0, 1) + center(1, 1) * center(1, 1)
            + face_product_mean(3, 3)
        + face_product_mean(0, 0) + face_product_mean(1, 1)
            + center(2, 2) * center(2, 2);
    return AmdInvariant{numerator, denominator};
}

double amd_scalar_diffusivity_at(
    const std::array<double, 9>& velocity_gradient,
    const std::array<double, 3>& scalar_gradient,
    const std::array<double, 3>& scaled_cell_width) {
    double numerator = 0.0;
    double denominator = 0.0;
    for (int i = 0; i < 3; ++i) {
        const double scalar_gradient_i = scalar_gradient[static_cast<std::size_t>(i)];
        denominator += scalar_gradient_i * scalar_gradient_i;
        for (int k = 0; k < 3; ++k) {
            const double length = scaled_cell_width[static_cast<std::size_t>(k)];
            const double velocity_derivative = velocity_gradient[static_cast<std::size_t>(3 * i + k)];
            numerator -= (length * velocity_derivative)
                * (length * scalar_gradient[static_cast<std::size_t>(k)])
                * scalar_gradient_i;
        }
    }
    if (!(denominator > 0.0) || !std::isfinite(numerator) || !std::isfinite(denominator)) {
        return 0.0;
    }
    return std::max(numerator, 0.0) / denominator;
}

double amd_scalar_diffusivity_staggered_at(
    const std::array<double, 9>& center_velocity_gradient,
    const std::array<double, 3>& center_scalar_gradient,
    double lower_scalar_dz,
    double upper_scalar_dz,
    const std::array<double, 3>& scaled_cell_width) {
    return amd_invariant_ratio(amd_scalar_diffusivity_staggered_invariant_at(
        center_velocity_gradient,
        center_scalar_gradient,
        lower_scalar_dz,
        upper_scalar_dz,
        scaled_cell_width));
}

AmdInvariant amd_scalar_diffusivity_staggered_invariant_at(
    const std::array<double, 9>& center_velocity_gradient,
    const std::array<double, 3>& center_scalar_gradient,
    double lower_scalar_dz,
    double upper_scalar_dz,
    const std::array<double, 3>& scaled_cell_width) {
    double numerator = 0.0;
    for (int i = 0; i < 3; ++i) {
        const double scalar_gradient_i = center_scalar_gradient[static_cast<std::size_t>(i)];
        for (int k = 0; k < 3; ++k) {
            const double length = scaled_cell_width[static_cast<std::size_t>(k)];
            const double velocity_derivative =
                center_velocity_gradient[static_cast<std::size_t>(3 * i + k)];
            numerator -= (length * velocity_derivative)
                * (length * center_scalar_gradient[static_cast<std::size_t>(k)])
                * scalar_gradient_i;
        }
    }
    const double denominator =
        center_scalar_gradient[0] * center_scalar_gradient[0]
        + center_scalar_gradient[1] * center_scalar_gradient[1]
        + 0.5 * (lower_scalar_dz * lower_scalar_dz + upper_scalar_dz * upper_scalar_dz);
    return AmdInvariant{numerator, denominator};
}

AmdInvariant amd_scalar_diffusivity_face_product_invariant_at(
    const std::array<double, 9>& center_velocity_gradient,
    const std::array<double, 4>& lower_face_gradient,
    const std::array<double, 4>& upper_face_gradient,
    double center_scalar_dx,
    double center_scalar_dy,
    double lower_scalar_dz,
    double upper_scalar_dz,
    const std::array<double, 3>& scaled_cell_width) {
    auto center = [&](int component, int direction) {
        return center_velocity_gradient[static_cast<std::size_t>(3 * component + direction)];
    };
    // Face layout: {du/dz, dv/dz, dw/dx, dw/dy}; the scalar vertical gradient
    // shares the same faces, so all vertical/face quadratic products are
    // formed in place and averaged to the center afterwards.
    auto face_mean_with_scalar_dz = [&](int component) {
        return 0.5 * (lower_face_gradient[static_cast<std::size_t>(component)] * lower_scalar_dz
            + upper_face_gradient[static_cast<std::size_t>(component)] * upper_scalar_dz);
    };
    const double face_scalar_dz_squared =
        0.5 * (lower_scalar_dz * lower_scalar_dz + upper_scalar_dz * upper_scalar_dz);

    const double dx2 = scaled_cell_width[0] * scaled_cell_width[0];
    const double dy2 = scaled_cell_width[1] * scaled_cell_width[1];
    const double dz2 = scaled_cell_width[2] * scaled_cell_width[2];
    double numerator = 0.0;
    numerator -= dx2 * (center(0, 0) * center_scalar_dx * center_scalar_dx
        + center(1, 0) * center_scalar_dx * center_scalar_dy
        + face_mean_with_scalar_dz(2) * center_scalar_dx);
    numerator -= dy2 * (center(0, 1) * center_scalar_dy * center_scalar_dx
        + center(1, 1) * center_scalar_dy * center_scalar_dy
        + face_mean_with_scalar_dz(3) * center_scalar_dy);
    numerator -= dz2 * (face_mean_with_scalar_dz(0) * center_scalar_dx
        + face_mean_with_scalar_dz(1) * center_scalar_dy
        + center(2, 2) * face_scalar_dz_squared);
    const double denominator = center_scalar_dx * center_scalar_dx
        + center_scalar_dy * center_scalar_dy
        + face_scalar_dz_squared;
    return AmdInvariant{numerator, denominator};
}

double amd_shared_scalar_diffusivity_at(
    const std::array<double, 9>& velocity_gradient,
    const std::array<double, 3>& theta_l_gradient,
    const std::array<double, 3>& qt_gradient,
    const std::array<double, 3>& scaled_cell_width) {
    return std::max(
        amd_scalar_diffusivity_at(
            velocity_gradient, theta_l_gradient, scaled_cell_width),
        amd_scalar_diffusivity_at(
            velocity_gradient, qt_gradient, scaled_cell_width));
}

namespace {

Field amd_buoyancy_eddy_viscosity(
    const FlowState& state,
    const VelocityGradients& grad,
    const Params& params,
    FftwXY& fft) {
    const bool buoyancy_active = params.thermo_enabled && params.amd_buoyancy_correction;
    Field buoyancy_prime(params.real_size(), 0.0);
    if (buoyancy_active) {
        buoyancy_prime = virtual_potential_temperature(state, params);
        const double inverse_plane = 1.0 / static_cast<double>(params.nx * params.ny);
        const double coefficient = params.g
            / (params.moisture_enabled
                    ? params.theta0 * (1.0 + 0.61 * params.qv0)
                    : params.theta0);
        for (int k = 0; k < params.nz; ++k) {
            double mean = 0.0;
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    mean += buoyancy_prime[idx(params, i, j, k)];
                }
            }
            mean *= inverse_plane;
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const std::size_t n = idx(params, i, j, k);
                    buoyancy_prime[n] = coefficient * (buoyancy_prime[n] - mean);
                }
            }
        }
    }
    Field dbdx(params.real_size(), 0.0);
    Field dbdy(params.real_size(), 0.0);
    Field dbdz(params.real_size(), 0.0);
    if (buoyancy_active) {
        fft.derivative_x(buoyancy_prime, dbdx, params);
        fft.derivative_y(buoyancy_prime, dbdy, params);
        dbdz = ddz_center(buoyancy_prime, params);
    }
    Field dwdx_face;
    Field dwdy_face;
    fft.derivative_x_planes(state.w, params.nz + 1, dwdx_face, params);
    fft.derivative_y_planes(state.w, params.nz + 1, dwdy_face, params);
    Field dudz_face = ddz_center_to_w(state.u, params);
    Field dvdz_face = ddz_center_to_w(state.v, params);
    if (params.amd_wall_model_gradients) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t center = idx(params, i, j, 0);
                const std::size_t face = z_face_idx(params, i, j, 0);
                dudz_face[face] = grad.dudz[center];
                dvdz_face[face] = grad.dvdz[center];
            }
        }
    }
    const std::array<double, 3> length = amd_scaled_cell_width(params);
    Field numerator(params.real_size(), 0.0);
    Field denominator(params.real_size(), 0.0);
    for (int k = 0; k < params.nz; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                const std::size_t lower = z_face_idx(params, i, j, k);
                const std::size_t upper = z_face_idx(params, i, j, k + 1);
                const std::array<double, 9> velocity_gradient{
                    grad.dudx[n], grad.dudy[n], grad.dudz[n],
                    grad.dvdx[n], grad.dvdy[n], grad.dvdz[n],
                    grad.dwdx[n], grad.dwdy[n], grad.dwdz[n],
                };
                const AmdInvariant invariant = amd_eddy_viscosity_staggered_invariant_at(
                    velocity_gradient,
                    {dudz_face[lower], dvdz_face[lower], dwdx_face[lower], dwdy_face[lower]},
                    {dudz_face[upper], dvdz_face[upper], dwdx_face[upper], dwdy_face[upper]},
                    {dbdx[n], dbdy[n], dbdz[n]},
                    length);
                numerator[n] = invariant.numerator;
                denominator[n] = invariant.denominator;
            }
        }
    }
    if (params.amd_invariant_averaging) {
        smooth_amd_invariant_field(numerator, params);
        smooth_amd_invariant_field(denominator, params);
    }
    Field nu_t(params.real_size(), 0.0);
    for (std::size_t n = 0; n < nu_t.size(); ++n) {
        nu_t[n] = amd_invariant_ratio(AmdInvariant{numerator[n], denominator[n]});
    }
    if (params.amd_multiscale_averaging) {
        enforce_multiscale_amd_invariant_bound(
            nu_t, std::move(numerator), std::move(denominator), params);
    }
    if (params.amd_dissipation_averaging) {
        average_amd_eddy_viscosity_by_local_dissipation(
            nu_t, strain_magnitude(grad, params), params);
    }
    return nu_t;
}

}  // namespace

Field strain_magnitude(const VelocityGradients& grad, const Params& params) {
    Field mag(params.real_size(), 0.0);
    for (std::size_t n = 0; n < mag.size(); ++n) {
        const double s11 = grad.dudx[n];
        const double s22 = grad.dvdy[n];
        const double s33 = grad.dwdz[n];
        const double s12 = 0.5 * (grad.dudy[n] + grad.dvdx[n]);
        const double s13 = 0.5 * (grad.dudz[n] + grad.dwdx[n]);
        const double s23 = 0.5 * (grad.dvdz[n] + grad.dwdy[n]);
        const double sij_sij = s11 * s11 + s22 * s22 + s33 * s33 + 2.0 * (s12 * s12 + s13 * s13 + s23 * s23);
        mag[n] = std::sqrt(std::max(2.0 * sij_sij, 0.0));
    }
    return mag;
}

void redistribute_amd_eddy_viscosity_by_plane_dissipation(
    Field& eddy_viscosity,
    const Field& strain,
    const Params& params) {
    if (eddy_viscosity.size() != params.real_size() || strain.size() != params.real_size()) {
        throw std::runtime_error("plane AMD redistribution field-size mismatch");
    }
    for (int k = 0; k < params.nz; ++k) {
        double plane_dissipation = 0.0;
        double plane_strain_squared = 0.0;
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                const double s2 = strain[n] * strain[n];
                plane_dissipation += std::max(eddy_viscosity[n], 0.0) * s2;
                plane_strain_squared += s2;
            }
        }
        const double plane_viscosity = plane_strain_squared > 0.0
            ? plane_dissipation / plane_strain_squared
            : 0.0;
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                eddy_viscosity[idx(params, i, j, k)] = plane_viscosity;
            }
        }
    }
}

void average_amd_eddy_viscosity_by_local_dissipation(
    Field& eddy_viscosity,
    const Field& strain,
    const Params& params) {
    if (eddy_viscosity.size() != params.real_size() || strain.size() != params.real_size()) {
        throw std::runtime_error("local AMD dissipation-averaging field-size mismatch");
    }
    Field dissipation(params.real_size(), 0.0);
    Field strain_squared(params.real_size(), 0.0);
    for (std::size_t n = 0; n < eddy_viscosity.size(); ++n) {
        strain_squared[n] = strain[n] * strain[n];
        dissipation[n] = std::max(eddy_viscosity[n], 0.0) * strain_squared[n];
    }
    smooth_amd_invariant_field(dissipation, params);
    smooth_amd_invariant_field(strain_squared, params);
    for (std::size_t n = 0; n < eddy_viscosity.size(); ++n) {
        eddy_viscosity[n] = strain_squared[n] > 0.0
            ? dissipation[n] / strain_squared[n]
            : 0.0;
    }
}

void enforce_multiscale_amd_invariant_bound(
    Field& eddy_viscosity,
    Field positive_numerator,
    Field denominator,
    const Params& params) {
    if (eddy_viscosity.size() != params.real_size()
        || positive_numerator.size() != params.real_size()
        || denominator.size() != params.real_size()) {
        throw std::runtime_error("multiscale AMD invariant field-size mismatch");
    }
    for (double& value : positive_numerator) value = std::max(value, 0.0);
    smooth_amd_invariant_field(positive_numerator, params);
    smooth_amd_invariant_field(denominator, params);
    for (std::size_t n = 0; n < eddy_viscosity.size(); ++n) {
        const double filter_box_viscosity = amd_invariant_ratio(
            AmdInvariant{positive_numerator[n], denominator[n]});
        eddy_viscosity[n] = std::max(eddy_viscosity[n], filter_box_viscosity);
    }
}

Field test_filtered_strain_magnitude(
    const VelocityGradients& grad,
    const Params& params,
    FftwXY& fft,
    double filter_width) {
    const SymFields sij = strain_components(grad, params);
    SymFields sij_hat;
    for (int c = 0; c < 6; ++c) {
        sij_hat[c] = horizontal_spectral_filter(sij[c], params, fft, filter_width);
    }
    Field magnitude(params.real_size(), 0.0);
    for (std::size_t n = 0; n < magnitude.size(); ++n) {
        const std::array<double, 6> filtered{
            sij_hat[0][n], sij_hat[1][n], sij_hat[2][n],
            sij_hat[3][n], sij_hat[4][n], sij_hat[5][n],
        };
        magnitude[n] = sym_strain_magnitude_at(filtered);
    }
    return magnitude;
}

Field smagorinsky_eddy_viscosity(const VelocityGradients& grad, const Params& params) {
    if (params.sgs_model == "none") {
        return Field(params.real_size(), 0.0);
    }

    const Field strain = strain_magnitude(grad, params);
    Field nu_t(params.real_size(), 0.0);
    const double length = params.smagorinsky_cs * params.sgs_delta();
    const double coeff = length * length;
    for (std::size_t n = 0; n < nu_t.size(); ++n) {
        nu_t[n] = coeff * strain[n];
    }
    return nu_t;
}

Field smagorinsky_eddy_viscosity(
    const FlowState& state,
    const VelocityGradients& grad,
    const Params& params) {
    if (!params.smagorinsky_buoyancy_correction || !params.thermo_enabled) {
        return smagorinsky_eddy_viscosity(grad, params);
    }
    const Field strain = strain_magnitude(grad, params);
    Field buoyancy_scalar(params.real_size(), 0.0);
    for (std::size_t n = 0; n < buoyancy_scalar.size(); ++n) {
        buoyancy_scalar[n] = params.moisture_enabled
            ? state.theta[n] * (1.0 + 0.61 * state.qv[n] - state.ql[n])
            : state.theta[n];
    }
    const Field dbuoyancy_dz = ddz_center(buoyancy_scalar, params);
    Field nu_t(params.real_size(), 0.0);
    const double length = params.smagorinsky_cs * params.sgs_delta();
    const double coeff = length * length;
    for (std::size_t n = 0; n < nu_t.size(); ++n) {
        const double n2 = params.g * dbuoyancy_dz[n] / std::max(buoyancy_scalar[n], 1.0);
        const double effective_strain_squared = strain[n] * strain[n] - n2 / params.prandtl_t;
        const double corrected = coeff * std::sqrt(std::max(effective_strain_squared, 0.0));
        const double shear_floor = params.smagorinsky_min_shear_fraction * coeff * strain[n];
        nu_t[n] = std::max(corrected, shear_floor);
    }
    return nu_t;
}

Field current_sgs_eddy_viscosity(
    const FlowState& state,
    const VelocityGradients& grad,
    const Params& params,
    FftwXY& fft) {
    const Field strain = strain_magnitude(grad, params);
    if (params.sgs_model == "none") {
        return Field(params.real_size(), 0.0);
    }
    if (params.sgs_model == "lasd") {
        return eddy_viscosity_from_cs2(state.cs2, strain, params);
    }
    if (params.sgs_model == "amd" || params.sgs_model == "amd_plane_dissipation") {
        Field eddy_viscosity = amd_buoyancy_eddy_viscosity(state, grad, params, fft);
        if (params.sgs_model == "amd_plane_dissipation") {
            redistribute_amd_eddy_viscosity_by_plane_dissipation(
                eddy_viscosity, strain, params);
        }
        return eddy_viscosity;
    }
    return smagorinsky_eddy_viscosity(state, grad, params);
}

SgsViscosity update_sgs_eddy_viscosity(FlowState& state, const VelocityGradients& grad, const Params& params, FftwXY& fft) {
    SgsViscosity result;
    result.strain = strain_magnitude(grad, params);
    if (params.sgs_model == "none") {
        result.eddy_viscosity.assign(params.real_size(), 0.0);
        return result;
    }
    if (params.sgs_model == "smagorinsky") {
        result.eddy_viscosity = smagorinsky_eddy_viscosity(state, grad, params);
        return result;
    }
    if (params.sgs_model == "amd" || params.sgs_model == "amd_plane_dissipation") {
        result.eddy_viscosity = amd_buoyancy_eddy_viscosity(state, grad, params, fft);
        if (params.sgs_model == "amd_plane_dissipation") {
            redistribute_amd_eddy_viscosity_by_plane_dissipation(
                result.eddy_viscosity, result.strain, params);
        }
        return result;
    }
    if (params.sgs_model != "lasd") {
        throw std::runtime_error("unsupported SGS model: " + params.sgs_model);
    }

    update_lagrangian_velocity(state, params);
    const bool should_update = state.step_count > 0 && (state.step_count % params.cs_count) == 0;
    if (!should_update) {
        result.eddy_viscosity = eddy_viscosity_from_cs2(state.cs2, result.strain, params);
        return result;
    }

    const SymFields sij = strain_components(grad, params);
    const VecFields vel = centered_velocity(state, params);
    const LmMm two_delta = momentum_lm_mm(state, vel, sij, result.strain, params, fft, params.tfr);
    const LmMm four_delta = momentum_lm_mm(state, vel, sij, result.strain, params, fft, params.tfr * params.tfr);
    result.test_strain_2d = two_delta.strain_hat;
    result.test_strain_4d = four_delta.strain_hat;

    Field lm_old = state.lm_old;
    Field mm_old = state.mm_old;
    Field qn_old = state.qn_old;
    Field nn_old = state.nn_old;
    if (state.step_count == params.cs_count) {
        for (std::size_t n = 0; n < params.real_size(); ++n) {
            lm_old[n] = 0.03 * two_delta.mm[n];
            mm_old[n] = two_delta.mm[n];
            qn_old[n] = 0.03 * four_delta.mm[n];
            nn_old[n] = four_delta.mm[n];
        }
    }
    apply_center_history_bc(lm_old, params);
    apply_center_history_bc(mm_old, params);
    apply_center_history_bc(qn_old, params);
    apply_center_history_bc(nn_old, params);

    auto [lm_avg, mm_avg] = lagrangian_average_fields(
        two_delta.lm, two_delta.mm, lm_old, mm_old, state.u_lag, state.v_lag, state.w_lag, params);
    auto [qn_avg, nn_avg] = lagrangian_average_fields(
        four_delta.lm, four_delta.mm, qn_old, nn_old, state.u_lag, state.v_lag, state.w_lag, params);

    const double exponent = std::log(params.tfr) / (std::log(params.tfr * params.tfr) - std::log(params.tfr));
    const double beta_min = 1.0 / (params.tfr * params.tfr * params.tfr);
    for (std::size_t n = 0; n < params.real_size(); ++n) {
        const double cs2_2d = std::max(safe_divide(lm_avg[n], mm_avg[n]), 0.0);
        const double cs2_4d = std::max(safe_divide(qn_avg[n], nn_avg[n]), 0.0);
        double beta = std::pow(std::max(safe_divide(cs2_4d, cs2_2d), 0.0), exponent);
        beta = std::max(beta, beta_min);
        state.cs2[n] = std::clamp(safe_divide(cs2_2d, beta), 1.0e-6, 0.81);
    }
    state.lm_old.swap(lm_avg);
    state.mm_old.swap(mm_avg);
    state.qn_old.swap(qn_avg);
    state.nn_old.swap(nn_avg);
    result.lasd_updated = true;
    result.eddy_viscosity = eddy_viscosity_from_cs2(state.cs2, result.strain, params);
    return result;
}

void add_sgs_momentum_forcing(
    Field& rhs_u,
    Field& rhs_v,
    Field& rhs_w,
    const FlowState& state,
    const VelocityGradients& grad,
    const Field& nu_t,
    const Params& params,
    FftwXY& fft) {
    if (params.sgs_model == "none") {
        return;
    }

    Field txx(params.real_size(), 0.0);
    Field txy(params.real_size(), 0.0);
    Field tyy(params.real_size(), 0.0);
    Field tzz(params.real_size(), 0.0);
    for (std::size_t n = 0; n < txx.size(); ++n) {
        txx[n] = 2.0 * nu_t[n] * grad.dudx[n];
        txy[n] = nu_t[n] * (grad.dudy[n] + grad.dvdx[n]);
        tyy[n] = 2.0 * nu_t[n] * grad.dvdy[n];
        tzz[n] = 2.0 * nu_t[n] * grad.dwdz[n];
    }

    Field dwdx_face;
    Field dwdy_face;
    fft.derivative_x_planes(state.w, params.nz + 1, dwdx_face, params);
    fft.derivative_y_planes(state.w, params.nz + 1, dwdy_face, params);
    const Field dudz_face = ddz_center_to_w(state.u, params);
    const Field dvdz_face = ddz_center_to_w(state.v, params);
    const Field nu_t_face = center_to_w(nu_t, params);

    Field txz(params.z_face_size(), 0.0);
    Field tyz(params.z_face_size(), 0.0);
    for (int k = 1; k < params.nz; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t face = z_face_idx(params, i, j, k);
                txz[face] = nu_t_face[face] * (dudz_face[face] + dwdx_face[face]);
                tyz[face] = nu_t_face[face] * (dvdz_face[face] + dwdy_face[face]);
            }
        }
    }

    Field dtxx_dx;
    Field dtxy_dy;
    Field dtxy_dx;
    Field dtyy_dy;
    fft.derivative_x(txx, dtxx_dx, params);
    fft.derivative_y(txy, dtxy_dy, params);
    fft.derivative_x(txy, dtxy_dx, params);
    fft.derivative_y(tyy, dtyy_dy, params);
    const Field dtxz_dz = ddz_w_to_center(txz, params);
    const Field dtyz_dz = ddz_w_to_center(tyz, params);

    for (std::size_t n = 0; n < rhs_u.size(); ++n) {
        rhs_u[n] += dtxx_dx[n] + dtxy_dy[n] + dtxz_dz[n];
        rhs_v[n] += dtxy_dx[n] + dtyy_dy[n] + dtyz_dz[n];
    }

    Field dtxz_dx;
    Field dtyz_dy;
    fft.derivative_x_planes(txz, params.nz + 1, dtxz_dx, params);
    fft.derivative_y_planes(tyz, params.nz + 1, dtyz_dy, params);
    const Field dtzz_dz = ddz_center_to_w(tzz, params);
    for (int k = 1; k < params.nz; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t face = z_face_idx(params, i, j, k);
                rhs_w[face] += dtxz_dx[face] + dtyz_dy[face] + dtzz_dz[face];
            }
        }
    }
}

}  // namespace wireles

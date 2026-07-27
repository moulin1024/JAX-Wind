#include "wireles/scalar.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "wireles/bomex.hpp"
#include "wireles/operators.hpp"
#include "wireles/sgs.hpp"

namespace wireles {
namespace {

constexpr double virtual_temperature_coefficient = 0.61;

double safe_divide(double num, double den) {
    return std::abs(den) > 1.0e-30 ? num / den : 0.0;
}

double reference_virtual_potential_temperature(const Params& params) {
    if (!params.moisture_enabled) {
        return params.theta0;
    }
    return params.theta0 * (1.0 + virtual_temperature_coefficient * params.qv0);
}

struct ScalarLmMm {
    Field lm;
    Field mm;
};

ScalarLmMm scalar_lm_mm(
    const FlowState& state,
    const Field& transported_scalar,
    const Field& dtheta_dx,
    const Field& dtheta_dy,
    const Field& strain,
    const Field& test_strain,
    const Params& params,
    FftwXY& fft,
    double test_ratio) {
    const double width = params.fgr * test_ratio;
    const Field u_hat = horizontal_spectral_filter(state.u, params, fft, width);
    const Field v_hat = horizontal_spectral_filter(state.v, params, fft, width);
    const Field w_center = w_to_center(state.w, params);
    const Field w_hat = horizontal_spectral_filter(w_center, params, fft, width);
    const Field theta_hat = horizontal_spectral_filter(transported_scalar, params, fft, width);
    const Field dtheta_dz = ddz_center(transported_scalar, params);

    Field utheta(params.real_size(), 0.0);
    Field vtheta(params.real_size(), 0.0);
    Field wtheta(params.real_size(), 0.0);
    Field sx(params.real_size(), 0.0);
    Field sy(params.real_size(), 0.0);
    Field sz(params.real_size(), 0.0);
    for (std::size_t n = 0; n < params.real_size(); ++n) {
        utheta[n] = state.u[n] * transported_scalar[n];
        vtheta[n] = state.v[n] * transported_scalar[n];
        wtheta[n] = w_center[n] * transported_scalar[n];
        sx[n] = strain[n] * dtheta_dx[n];
        sy[n] = strain[n] * dtheta_dy[n];
        sz[n] = strain[n] * dtheta_dz[n];
    }

    const Field utheta_hat = horizontal_spectral_filter(utheta, params, fft, width);
    const Field vtheta_hat = horizontal_spectral_filter(vtheta, params, fft, width);
    const Field wtheta_hat = horizontal_spectral_filter(wtheta, params, fft, width);
    const Field sx_hat = horizontal_spectral_filter(sx, params, fft, width);
    const Field sy_hat = horizontal_spectral_filter(sy, params, fft, width);
    const Field sz_hat = horizontal_spectral_filter(sz, params, fft, width);
    const Field dx_hat = horizontal_spectral_filter(dtheta_dx, params, fft, width);
    const Field dy_hat = horizontal_spectral_filter(dtheta_dy, params, fft, width);
    const Field dz_hat = horizontal_spectral_filter(dtheta_dz, params, fft, width);

    ScalarLmMm out;
    out.lm.assign(params.real_size(), 0.0);
    out.mm.assign(params.real_size(), 0.0);
    const double delta2 = params.sgs_delta() * params.sgs_delta();
    const double ratio2 = test_ratio * test_ratio;
    for (std::size_t n = 0; n < params.real_size(); ++n) {
        const double l0 = utheta_hat[n] - u_hat[n] * theta_hat[n];
        const double l1 = vtheta_hat[n] - v_hat[n] * theta_hat[n];
        const double l2 = wtheta_hat[n] - w_hat[n] * theta_hat[n];
        const double sh = std::max(test_strain[n], 0.0);
        const double m0 = delta2 * (sx_hat[n] - ratio2 * sh * dx_hat[n]);
        const double m1 = delta2 * (sy_hat[n] - ratio2 * sh * dy_hat[n]);
        const double m2 = delta2 * (sz_hat[n] - ratio2 * sh * dz_hat[n]);
        out.lm[n] = l0 * m0 + l1 * m1 + l2 * m2;
        out.mm[n] = m0 * m0 + m1 * m1 + m2 * m2;
    }
    return out;
}

void update_scalar_lasd_coefficients(
    FlowState& state,
    const Field& transported_scalar,
    const Field& dtheta_dx,
    const Field& dtheta_dy,
    const Field& strain,
    const Field* test_strain_2d,
    const Field* test_strain_4d,
    Field& scalar_c,
    Field& scalar_lm_old,
    Field& scalar_mm_old,
    Field& scalar_qn_old,
    Field& scalar_nn_old,
    const Params& params,
    FftwXY& fft) {
    if (params.scalar_sgs_model != "lasd") {
        return;
    }
    const bool should_update = state.step_count > 0 && (state.step_count % params.cs_count) == 0;
    if (!should_update) {
        return;
    }

    Field local_test_strain_2d;
    Field local_test_strain_4d;
    if (test_strain_2d == nullptr || test_strain_4d == nullptr) {
        const VelocityGradients velocity_grad = velocity_gradients(state, params, fft);
        local_test_strain_2d = test_filtered_strain_magnitude(
            velocity_grad, params, fft, params.fgr * params.tfr);
        local_test_strain_4d = test_filtered_strain_magnitude(
            velocity_grad, params, fft, params.fgr * params.tfr * params.tfr);
        test_strain_2d = &local_test_strain_2d;
        test_strain_4d = &local_test_strain_4d;
    }
    const ScalarLmMm two_delta = scalar_lm_mm(
        state, transported_scalar, dtheta_dx, dtheta_dy, strain, *test_strain_2d, params, fft, params.tfr);
    const ScalarLmMm four_delta = scalar_lm_mm(
        state, transported_scalar, dtheta_dx, dtheta_dy, strain, *test_strain_4d, params, fft, params.tfr * params.tfr);

    Field lm_old = scalar_lm_old;
    Field mm_old = scalar_mm_old;
    Field qn_old = scalar_qn_old;
    Field nn_old = scalar_nn_old;
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

    // Use the SGS turnover time supplied by the momentum histories.  Building
    // T from scalar contractions makes T change when the same scalar is
    // expressed in K versus a rescaled unit, even though the dynamic
    // diffusivity coefficient itself is dimensionless.
    auto [lm_avg, mm_avg] = lagrangian_average_fields(
        two_delta.lm, two_delta.mm, lm_old, mm_old, state.u_lag, state.v_lag, state.w_lag, params,
        &state.lm_old, &state.mm_old);
    auto [qn_avg, nn_avg] = lagrangian_average_fields(
        four_delta.lm, four_delta.mm, qn_old, nn_old, state.u_lag, state.v_lag, state.w_lag, params,
        &state.qn_old, &state.nn_old);

    const double exponent = std::log(params.tfr) / (std::log(params.tfr * params.tfr) - std::log(params.tfr));
    const double beta_min = 1.0 / (params.tfr * params.tfr * params.tfr);
    for (std::size_t n = 0; n < params.real_size(); ++n) {
        const double c_2d = std::max(safe_divide(lm_avg[n], mm_avg[n]), 0.0);
        const double c_4d = std::max(safe_divide(qn_avg[n], nn_avg[n]), 0.0);
        double beta = std::pow(std::max(safe_divide(c_4d, c_2d), 0.0), exponent);
        beta = std::max(beta, beta_min);
        scalar_c[n] = std::clamp(safe_divide(c_2d, beta), params.scalar_lasd_min, params.scalar_lasd_max);
    }
    scalar_lm_old.swap(lm_avg);
    scalar_mm_old.swap(mm_avg);
    scalar_qn_old.swap(qn_avg);
    scalar_nn_old.swap(nn_avg);
}

Field scalar_eddy_diffusivity_with_coefficient(
    const FlowState& state,
    const Field& eddy_viscosity,
    const Field& strain,
    const Field& scalar_c,
    double molecular_diffusivity,
    double turbulent_ratio,
    const Params& params) {
    Field kappa(params.real_size(), 0.0);
    const Field buoyancy_scalar = params.scalar_stability_correction
        ? virtual_potential_temperature(state, params)
        : Field{};
    const Field dbuoyancy_scalar_dz = params.scalar_stability_correction
        ? ddz_center(buoyancy_scalar, params)
        : Field{};
    const double strain_coeff = std::pow(params.smagorinsky_cs * params.sgs_delta(), 2.0);

    for (std::size_t n = 0; n < kappa.size(); ++n) {
        double diffusivity = molecular_diffusivity;
        if (params.scalar_sgs_model == "lasd") {
            diffusivity += std::max(scalar_c[n], 0.0) * params.sgs_delta() * params.sgs_delta() * strain[n];
        } else if (params.scalar_sgs_model == "fixed_smagorinsky") {
            diffusivity += strain_coeff * strain[n] / turbulent_ratio;
        } else {
            diffusivity += eddy_viscosity[n] / turbulent_ratio;
        }
        if (params.scalar_stability_correction) {
            const double local_strain = params.scalar_sgs_model == "lasd"
                ? strain[n]
                : ((strain_coeff > 0.0) ? eddy_viscosity[n] / strain_coeff : 0.0);
            const double n2 = (params.g / reference_virtual_potential_temperature(params))
                * dbuoyancy_scalar_dz[n];
            const double ri = std::max(n2, 0.0) / std::max(local_strain * local_strain, 1.0e-24);
            diffusivity *= std::pow(1.0 + params.scalar_stability_beta * ri, -params.scalar_stability_power);
        }
        kappa[n] = diffusivity;
    }
    return kappa;
}

Field amd_scalar_eddy_diffusivity_field(
    const FlowState& state,
    const Field& transported_scalar,
    double molecular_diffusivity,
    const Params& params,
    FftwXY& fft) {
    const VelocityGradients velocity_gradient = velocity_gradients(state, params, fft);
    Field dscalar_dx;
    Field dscalar_dy;
    fft.derivative_x(transported_scalar, dscalar_dx, params);
    fft.derivative_y(transported_scalar, dscalar_dy, params);
    const Field dscalar_dz = ddz_center(transported_scalar, params);
    const Field dscalar_dz_w = ddz_center_to_w(transported_scalar, params);
    Field dudz_face;
    Field dvdz_face;
    Field dwdx_face;
    Field dwdy_face;
    if (params.scalar_amd_face_products) {
        dudz_face = ddz_center_to_w(state.u, params);
        dvdz_face = ddz_center_to_w(state.v, params);
        fft.derivative_x_planes(state.w, params.nz + 1, dwdx_face, params);
        fft.derivative_y_planes(state.w, params.nz + 1, dwdy_face, params);
        if (params.amd_wall_model_gradients) {
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const std::size_t center = idx(params, i, j, 0);
                    const std::size_t face = z_face_idx(params, i, j, 0);
                    dudz_face[face] = velocity_gradient.dudz[center];
                    dvdz_face[face] = velocity_gradient.dvdz[center];
                }
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
                const std::array<double, 9> gradient{
                    velocity_gradient.dudx[n], velocity_gradient.dudy[n], velocity_gradient.dudz[n],
                    velocity_gradient.dvdx[n], velocity_gradient.dvdy[n], velocity_gradient.dvdz[n],
                    velocity_gradient.dwdx[n], velocity_gradient.dwdy[n], velocity_gradient.dwdz[n],
                };
                const std::size_t lower = z_face_idx(params, i, j, k);
                const std::size_t upper = z_face_idx(params, i, j, k + 1);
                const AmdInvariant invariant = params.scalar_amd_face_products
                    ? amd_scalar_diffusivity_face_product_invariant_at(
                        gradient,
                        {dudz_face[lower], dvdz_face[lower], dwdx_face[lower], dwdy_face[lower]},
                        {dudz_face[upper], dvdz_face[upper], dwdx_face[upper], dwdy_face[upper]},
                        dscalar_dx[n], dscalar_dy[n],
                        dscalar_dz_w[lower], dscalar_dz_w[upper],
                        length)
                    : amd_scalar_diffusivity_staggered_invariant_at(
                        gradient,
                        {dscalar_dx[n], dscalar_dy[n], dscalar_dz[n]},
                        dscalar_dz_w[lower],
                        dscalar_dz_w[upper],
                        length);
                numerator[n] = invariant.numerator;
                denominator[n] = invariant.denominator;
            }
        }
    }
    if (params.scalar_amd_invariant_averaging) {
        smooth_amd_invariant_field(numerator, params);
        smooth_amd_invariant_field(denominator, params);
    }
    Field diffusivity(params.real_size(), molecular_diffusivity);
    for (std::size_t n = 0; n < diffusivity.size(); ++n) {
        diffusivity[n] += amd_invariant_ratio(AmdInvariant{numerator[n], denominator[n]});
    }
    if (params.scalar_sgs_model == "amd_plane_dissipation") {
        redistribute_scalar_diffusivity_by_plane_dissipation(
            diffusivity,
            dscalar_dx,
            dscalar_dy,
            dscalar_dz_w,
            molecular_diffusivity,
            params);
    }
    return diffusivity;
}

Field amd_shared_scalar_eddy_diffusivity_field(
    const FlowState& state,
    double molecular_diffusivity,
    const Params& params,
    FftwXY& fft) {
    const VelocityGradients velocity_gradient = velocity_gradients(state, params, fft);
    Field dtheta_l_dx;
    Field dtheta_l_dy;
    Field dqt_dx;
    Field dqt_dy;
    fft.derivative_x(state.theta_l, dtheta_l_dx, params);
    fft.derivative_y(state.theta_l, dtheta_l_dy, params);
    fft.derivative_x(state.qt, dqt_dx, params);
    fft.derivative_y(state.qt, dqt_dy, params);
    const Field dtheta_l_dz = ddz_center(state.theta_l, params);
    const Field dqt_dz = ddz_center(state.qt, params);
    const Field dtheta_l_dz_w = ddz_center_to_w(state.theta_l, params);
    const Field dqt_dz_w = ddz_center_to_w(state.qt, params);
    const std::array<double, 3> length = amd_scaled_cell_width(params);
    Field theta_l_numerator(params.real_size(), 0.0);
    Field theta_l_denominator(params.real_size(), 0.0);
    Field qt_numerator(params.real_size(), 0.0);
    Field qt_denominator(params.real_size(), 0.0);
    for (int k = 0; k < params.nz; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                const std::array<double, 9> gradient{
                    velocity_gradient.dudx[n], velocity_gradient.dudy[n], velocity_gradient.dudz[n],
                    velocity_gradient.dvdx[n], velocity_gradient.dvdy[n], velocity_gradient.dvdz[n],
                    velocity_gradient.dwdx[n], velocity_gradient.dwdy[n], velocity_gradient.dwdz[n],
                };
                const std::size_t lower = z_face_idx(params, i, j, k);
                const std::size_t upper = z_face_idx(params, i, j, k + 1);
                const AmdInvariant theta_l_invariant = amd_scalar_diffusivity_staggered_invariant_at(
                    gradient,
                    {dtheta_l_dx[n], dtheta_l_dy[n], dtheta_l_dz[n]},
                    dtheta_l_dz_w[lower], dtheta_l_dz_w[upper], length);
                const AmdInvariant qt_invariant = amd_scalar_diffusivity_staggered_invariant_at(
                    gradient,
                    {dqt_dx[n], dqt_dy[n], dqt_dz[n]},
                    dqt_dz_w[lower], dqt_dz_w[upper], length);
                theta_l_numerator[n] = theta_l_invariant.numerator;
                theta_l_denominator[n] = theta_l_invariant.denominator;
                qt_numerator[n] = qt_invariant.numerator;
                qt_denominator[n] = qt_invariant.denominator;
            }
        }
    }
    if (params.scalar_amd_invariant_averaging) {
        smooth_amd_invariant_field(theta_l_numerator, params);
        smooth_amd_invariant_field(theta_l_denominator, params);
        smooth_amd_invariant_field(qt_numerator, params);
        smooth_amd_invariant_field(qt_denominator, params);
    }
    Field diffusivity(params.real_size(), molecular_diffusivity);
    for (std::size_t n = 0; n < diffusivity.size(); ++n) {
        diffusivity[n] += std::max(
            amd_invariant_ratio(AmdInvariant{theta_l_numerator[n], theta_l_denominator[n]}),
            amd_invariant_ratio(AmdInvariant{qt_numerator[n], qt_denominator[n]}));
    }
    return diffusivity;
}

}  // namespace

void redistribute_scalar_diffusivity_by_plane_dissipation(
    Field& diffusivity,
    const Field& dscalar_dx,
    const Field& dscalar_dy,
    const Field& dscalar_dz_w,
    double molecular_diffusivity,
    const Params& params) {
    const std::size_t size = params.real_size();
    if (diffusivity.size() != size || dscalar_dx.size() != size
        || dscalar_dy.size() != size || dscalar_dz_w.size() != params.z_face_size()) {
        throw std::runtime_error("plane scalar-dissipation redistribution field-size mismatch");
    }

    for (int k = 0; k < params.nz; ++k) {
        double plane_sgs_dissipation = 0.0;
        double plane_gradient_energy = 0.0;
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                double gradient_energy = dscalar_dx[n] * dscalar_dx[n]
                    + dscalar_dy[n] * dscalar_dy[n];
                if (k > 0) {
                    const double lower = dscalar_dz_w[z_face_idx(params, i, j, k)];
                    gradient_energy += 0.5 * lower * lower;
                }
                if (k + 1 < params.nz) {
                    const double upper = dscalar_dz_w[z_face_idx(params, i, j, k + 1)];
                    gradient_energy += 0.5 * upper * upper;
                }
                plane_gradient_energy += gradient_energy;
                plane_sgs_dissipation +=
                    std::max(diffusivity[n] - molecular_diffusivity, 0.0) * gradient_energy;
            }
        }
        const double plane_sgs_diffusivity = plane_gradient_energy > 0.0
            ? plane_sgs_dissipation / plane_gradient_energy
            : 0.0;
        const double plane_diffusivity = molecular_diffusivity + plane_sgs_diffusivity;
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                diffusivity[idx(params, i, j, k)] = plane_diffusivity;
            }
        }
    }
}

Field virtual_potential_temperature(const FlowState& state, const Params& params) {
    if (!params.moisture_enabled) {
        return state.theta;
    }
    Field theta_v(params.real_size(), 0.0);
    for (std::size_t n = 0; n < theta_v.size(); ++n) {
        theta_v[n] = state.theta[n]
            * (1.0 + virtual_temperature_coefficient * state.qv[n] - state.ql[n]);
    }
    return theta_v;
}

Field scalar_eddy_diffusivity(
    const FlowState& state,
    const Field& eddy_viscosity,
    const Field& strain,
    const Params& params,
    FftwXY& fft) {
    if (params.scalar_sgs_model == "amd"
        || params.scalar_sgs_model == "amd_plane_dissipation") {
        const Field& transported_theta = params.moisture_enabled ? state.theta_l : state.theta;
        return amd_scalar_eddy_diffusivity_field(
            state, transported_theta, params.scalar_diffusivity, params, fft);
    }
    if (params.scalar_sgs_model == "amd_shared") {
        return amd_shared_scalar_eddy_diffusivity_field(
            state, params.scalar_diffusivity, params, fft);
    }
    return scalar_eddy_diffusivity_with_coefficient(
        state,
        eddy_viscosity,
        strain,
        state.scalar_c,
        params.scalar_diffusivity,
        params.prandtl_t,
        params);
}

Field scalar_rhs(
    FlowState& state,
    const Field& eddy_viscosity,
    const Field& strain,
    const Params& params,
    FftwXY& fft,
    const Field* test_strain_2d,
    const Field* test_strain_4d) {
    Field rhs(params.real_size(), 0.0);
    if (!params.thermo_enabled) {
        return rhs;
    }
    const Field& transported_theta = params.moisture_enabled ? state.theta_l : state.theta;

    Field theta_flux_x(params.real_size(), 0.0);
    Field theta_flux_y(params.real_size(), 0.0);
    Field theta_flux_z(params.z_face_size(), 0.0);
    const Field theta_on_w = center_to_w(transported_theta, params);
    for (std::size_t n = 0; n < rhs.size(); ++n) {
        theta_flux_x[n] = state.u[n] * transported_theta[n];
        theta_flux_y[n] = state.v[n] * transported_theta[n];
    }
    for (std::size_t n = 0; n < state.w.size(); ++n) {
        theta_flux_z[n] = state.w[n] * theta_on_w[n];
    }
    if (params.horizontal_dealias && params.dealiasing == "sharp") {
        Field filtered;
        constexpr double two_thirds_filter_width = 1.5;
        fft.filter_planes_fortran_sharp(theta_flux_x, params.nz, filtered, params, two_thirds_filter_width);
        theta_flux_x.swap(filtered);
        fft.filter_planes_fortran_sharp(theta_flux_y, params.nz, filtered, params, two_thirds_filter_width);
        theta_flux_y.swap(filtered);
    }

    Field div_adv_x;
    Field div_adv_y;
    Field div_adv_xy;
    if (params.horizontal_dealias && params.dealiasing == "padding_3_2") {
        fft.horizontal_flux_divergence_3_2(state.u, state.v, transported_theta, 0, params.nz, div_adv_xy, params);
    } else {
        fft.derivative_x(theta_flux_x, div_adv_x, params);
        fft.derivative_y(theta_flux_y, div_adv_y, params);
    }
    const Field div_adv_z = ddz_w_to_center(theta_flux_z, params);
    for (std::size_t n = 0; n < rhs.size(); ++n) {
        const double horizontal_adv = params.horizontal_dealias && params.dealiasing == "padding_3_2"
            ? div_adv_xy[n]
            : div_adv_x[n] + div_adv_y[n];
        rhs[n] = -(horizontal_adv + div_adv_z[n]);
    }

    Field dtheta_dx;
    Field dtheta_dy;
    fft.derivative_x(transported_theta, dtheta_dx, params);
    fft.derivative_y(transported_theta, dtheta_dy, params);
    const Field dtheta_dz_w = ddz_center_to_w(transported_theta, params);

    update_scalar_lasd_coefficients(
        state,
        transported_theta,
        dtheta_dx,
        dtheta_dy,
        strain,
        test_strain_2d,
        test_strain_4d,
        state.scalar_c,
        state.scalar_lm_old,
        state.scalar_mm_old,
        state.scalar_qn_old,
        state.scalar_nn_old,
        params,
        fft);

    Field qx(params.real_size(), 0.0);
    Field qy(params.real_size(), 0.0);
    Field kappa_center = scalar_eddy_diffusivity(state, eddy_viscosity, strain, params, fft);
    for (std::size_t n = 0; n < rhs.size(); ++n) {
        qx[n] = -kappa_center[n] * dtheta_dx[n];
        qy[n] = -kappa_center[n] * dtheta_dy[n];
    }
    Field qz(params.z_face_size(), 0.0);
    const Field kappa_w = center_to_w(kappa_center, params);
    for (int k = 1; k < params.nz; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t face = z_face_idx(params, i, j, k);
                qz[face] = -kappa_w[face] * dtheta_dz_w[face];
            }
        }
    }

    Field div_qx;
    Field div_qy;
    fft.derivative_x(qx, div_qx, params);
    fft.derivative_y(qy, div_qy, params);
    if (params.surface_theta_flux != 0.0) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                qz[z_face_idx(params, i, j, 0)] = params.surface_theta_flux;
            }
        }
    }
    const Field div_qz = ddz_w_to_center(qz, params);
    for (std::size_t n = 0; n < rhs.size(); ++n) {
        rhs[n] -= div_qx[n] + div_qy[n] + div_qz[n];
    }
    return rhs;
}

Field moisture_rhs(
    FlowState& state,
    const Field& eddy_viscosity,
    const Field& strain,
    const Params& params,
    FftwXY& fft,
    double* diffusion_number,
    const Field* test_strain_2d,
    const Field* test_strain_4d) {
    Field rhs(params.real_size(), 0.0);
    if (diffusion_number != nullptr) {
        *diffusion_number = 0.0;
    }
    if (!params.moisture_enabled) {
        return rhs;
    }

    Field flux_x(params.real_size(), 0.0);
    Field flux_y(params.real_size(), 0.0);
    Field flux_z(params.z_face_size(), 0.0);
    const Field qt_on_w = center_to_w(state.qt, params);
    for (std::size_t n = 0; n < rhs.size(); ++n) {
        flux_x[n] = state.u[n] * state.qt[n];
        flux_y[n] = state.v[n] * state.qt[n];
    }
    for (std::size_t n = 0; n < flux_z.size(); ++n) {
        flux_z[n] = state.w[n] * qt_on_w[n];
    }
    if (params.horizontal_dealias && params.dealiasing == "sharp") {
        Field filtered;
        constexpr double two_thirds_filter_width = 1.5;
        fft.filter_planes_fortran_sharp(flux_x, params.nz, filtered, params, two_thirds_filter_width);
        flux_x.swap(filtered);
        fft.filter_planes_fortran_sharp(flux_y, params.nz, filtered, params, two_thirds_filter_width);
        flux_y.swap(filtered);
    }

    Field div_adv_x;
    Field div_adv_y;
    Field div_adv_xy;
    if (params.horizontal_dealias && params.dealiasing == "padding_3_2") {
        fft.horizontal_flux_divergence_3_2(state.u, state.v, state.qt, 0, params.nz, div_adv_xy, params);
    } else {
        fft.derivative_x(flux_x, div_adv_x, params);
        fft.derivative_y(flux_y, div_adv_y, params);
    }
    const Field div_adv_z = ddz_w_to_center(flux_z, params);
    for (std::size_t n = 0; n < rhs.size(); ++n) {
        const double horizontal_adv = params.horizontal_dealias && params.dealiasing == "padding_3_2"
            ? div_adv_xy[n]
            : div_adv_x[n] + div_adv_y[n];
        rhs[n] = -(horizontal_adv + div_adv_z[n]);
    }

    Field dqt_dx;
    Field dqt_dy;
    fft.derivative_x(state.qt, dqt_dx, params);
    fft.derivative_y(state.qt, dqt_dy, params);
    const Field dqt_dz_w = ddz_center_to_w(state.qt, params);

    update_scalar_lasd_coefficients(
        state,
        state.qt,
        dqt_dx,
        dqt_dy,
        strain,
        test_strain_2d,
        test_strain_4d,
        state.qt_scalar_c,
        state.qt_scalar_lm_old,
        state.qt_scalar_mm_old,
        state.qt_scalar_qn_old,
        state.qt_scalar_nn_old,
        params,
        fft);

    const Field kappa = moisture_eddy_diffusivity(state, eddy_viscosity, strain, params, fft);
    if (diffusion_number != nullptr) {
        *diffusion_number = moisture_diffusion_number(kappa, params);
    }
    Field diffusive_flux_x(params.real_size(), 0.0);
    Field diffusive_flux_y(params.real_size(), 0.0);
    for (std::size_t n = 0; n < rhs.size(); ++n) {
        diffusive_flux_x[n] = -kappa[n] * dqt_dx[n];
        diffusive_flux_y[n] = -kappa[n] * dqt_dy[n];
    }
    Field diffusive_flux_z(params.z_face_size(), 0.0);
    const Field kappa_w = center_to_w(kappa, params);
    for (int k = 1; k < params.nz; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t face = z_face_idx(params, i, j, k);
                diffusive_flux_z[face] = -kappa_w[face] * dqt_dz_w[face];
            }
        }
    }
    if (params.surface_qv_flux != 0.0) {
        const double surface_flux = params.initial_condition == "bomex"
            ? bomex_surface_qt_mixing_ratio_flux(params.surface_qv_flux)
            : params.surface_qv_flux;
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                diffusive_flux_z[z_face_idx(params, i, j, 0)] = surface_flux;
            }
        }
    }

    Field div_diffusive_x;
    Field div_diffusive_y;
    fft.derivative_x(diffusive_flux_x, div_diffusive_x, params);
    fft.derivative_y(diffusive_flux_y, div_diffusive_y, params);
    const Field div_diffusive_z = ddz_w_to_center(diffusive_flux_z, params);
    for (std::size_t n = 0; n < rhs.size(); ++n) {
        rhs[n] -= div_diffusive_x[n] + div_diffusive_y[n] + div_diffusive_z[n];
    }
    return rhs;
}

Field moisture_eddy_diffusivity(
    const FlowState& state,
    const Field& eddy_viscosity,
    const Field& strain,
    const Params& params,
    FftwXY& fft) {
    if (params.scalar_sgs_model == "amd"
        || params.scalar_sgs_model == "amd_plane_dissipation") {
        return amd_scalar_eddy_diffusivity_field(
            state, state.qt, params.moisture_diffusivity, params, fft);
    }
    if (params.scalar_sgs_model == "amd_shared") {
        return amd_shared_scalar_eddy_diffusivity_field(
            state, params.moisture_diffusivity, params, fft);
    }
    return scalar_eddy_diffusivity_with_coefficient(
        state,
        eddy_viscosity,
        strain,
        state.qt_scalar_c,
        params.moisture_diffusivity,
        params.schmidt_t,
        params);
}

double moisture_diffusion_number(const Field& diffusivity, const Params& params) {
    if (diffusivity.empty()) {
        return 0.0;
    }
    const double max_diffusivity = *std::max_element(diffusivity.begin(), diffusivity.end());
    const double max_laplacian_eigenvalue = std::pow(pi / params.dx(), 2.0)
        + std::pow(pi / params.dy(), 2.0)
        + 4.0 / std::pow(params.dz(), 2.0);
    return params.dt * max_diffusivity * max_laplacian_eigenvalue;
}

double column_integrated_water(const Field& qv, const Params& params) {
    return std::accumulate(qv.begin(), qv.end(), 0.0) * params.dz()
        / static_cast<double>(params.nx * params.ny);
}

double enforce_nonnegative_conservative(Field& qv) {
    double negative_mass = 0.0;
    double positive_mass = 0.0;
    for (const double value : qv) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("non-finite water mixing ratio encountered");
        }
        if (value < 0.0) {
            negative_mass -= value;
        } else {
            positive_mass += value;
        }
    }
    if (negative_mass == 0.0) {
        return 0.0;
    }
    if (positive_mass <= negative_mass) {
        throw std::runtime_error("water positivity correction cannot preserve a non-positive total mass");
    }
    const double scale = (positive_mass - negative_mass) / positive_mass;
    for (double& value : qv) {
        value = value > 0.0 ? value * scale : 0.0;
    }
    return negative_mass;
}

void add_buoyancy(Field& rhs_w, const FlowState& state, const Params& params) {
    if (!params.thermo_enabled) {
        return;
    }
    const double theta_v0 = reference_virtual_potential_temperature(params);
    const double coeff = params.g / theta_v0;
    const double inv_plane = 1.0 / static_cast<double>(params.nx * params.ny);
    Field theta_v_prime = virtual_potential_temperature(state, params);
    for (int k = 0; k < params.nz; ++k) {
        double mean_theta_v = 0.0;
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                mean_theta_v += theta_v_prime[idx(params, i, j, k)];
            }
        }
        mean_theta_v *= inv_plane;
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                theta_v_prime[n] -= mean_theta_v;
            }
        }
    }
    for (int k = 1; k < params.nz; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                rhs_w[z_face_idx(params, i, j, k)] += 0.5 * coeff * (
                    theta_v_prime[idx(params, i, j, k - 1)] + theta_v_prime[idx(params, i, j, k)]);
            }
        }
    }
}

void step_scalar(FlowState& state, const Field& rhs_theta, const Params& params) {
    if (!params.thermo_enabled) {
        return;
    }
    Field& scalar = params.moisture_enabled ? state.theta_l : state.theta;
    for (std::size_t n = 0; n < scalar.size(); ++n) {
        scalar[n] += params.dt * rhs_theta[n];
    }
}

}  // namespace wireles

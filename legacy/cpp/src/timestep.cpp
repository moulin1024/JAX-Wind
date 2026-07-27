#include "wireles/timestep.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "wireles/bomex.hpp"
#include "wireles/cuda_solver.hpp"
#include "wireles/operators.hpp"
#include "wireles/pressure.hpp"
#include "wireles/scalar.hpp"
#include "wireles/sgs.hpp"
#include "wireles/wall.hpp"

namespace wireles {
namespace {

void advance_field(Field& q, const Field& rhs, Field& rhs_prev, bool use_ab2, double dt) {
    for (std::size_t n = 0; n < q.size(); ++n) {
        const double tendency = use_ab2 ? (1.5 * rhs[n] - 0.5 * rhs_prev[n]) : rhs[n];
        q[n] += dt * tendency;
        rhs_prev[n] = rhs[n];
    }
}

void horizontal_dealias_field(Field& q, int planes, const Params& params, FftwXY& fft) {
    Field filtered;
    constexpr double two_thirds_filter_width = 1.5;
    fft.filter_planes(q, planes, filtered, params, two_thirds_filter_width);
    q.swap(filtered);
}

void horizontal_clear_nyquist_field(Field& q, int planes, const Params& params, FftwXY& fft) {
    Field filtered;
    fft.clear_nyquist_planes(q, planes, filtered, params);
    q.swap(filtered);
}

void horizontal_dealias_state(FlowState& state, const Params& params, FftwXY& fft) {
    if (!params.horizontal_dealias) {
        return;
    }
    auto apply_horizontal_dealias = [&](Field& q, int planes) {
        if (params.dealiasing == "padding_3_2") {
            horizontal_clear_nyquist_field(q, planes, params, fft);
        } else {
            horizontal_dealias_field(q, planes, params, fft);
        }
    };
    apply_horizontal_dealias(state.u, params.nz);
    apply_horizontal_dealias(state.v, params.nz);
    apply_horizontal_dealias(state.w, params.nz + 1);
    if (params.moisture_enabled) {
        apply_horizontal_dealias(state.theta_l, params.nz);
        apply_horizontal_dealias(state.qt, params.nz);
    } else if (params.thermo_enabled) {
        apply_horizontal_dealias(state.theta, params.nz);
    }
    enforce_walls(state.w, params);
}

double sponge_strength(double z, const Params& params) {
    if (!params.sponge_enabled || z <= params.sponge_start_height) {
        return 0.0;
    }
    const double depth = std::max(params.lz - params.sponge_start_height, params.dz());
    const double eta = std::clamp((z - params.sponge_start_height) / depth, 0.0, 1.0);
    return std::pow(eta, params.sponge_power) / params.sponge_timescale;
}

double center_sponge_target_u(double z, const Params& params) {
    return params.initial_condition == "bomex" ? bomex_geostrophic_u(z) : params.geostrophic_u;
}

double center_sponge_target_v(const Params& params) {
    return params.initial_condition == "bomex" ? 0.0 : params.geostrophic_v;
}

double center_sponge_target_theta(double z, const Params& params) {
    if (params.initial_condition == "bomex") {
        return bomex_initial_theta_l(z);
    }
    return params.theta0 + params.theta_initial_gradient * z;
}

double center_sponge_target_qt(double z, const Params& params) {
    if (params.initial_condition == "bomex") {
        return bomex_initial_qt(z);
    }
    return std::max(0.0, params.qv0 + params.qv_initial_gradient * z);
}

void relax_to_target(double& value, double target, double strength, double dt) {
    if (strength <= 0.0) {
        return;
    }
    const double factor = std::exp(-strength * dt);
    value = target + (value - target) * factor;
}

void apply_rayleigh_sponge(FlowState& state, const Params& params) {
    if (!params.sponge_enabled) {
        return;
    }
    for (int k = 0; k < params.nz; ++k) {
        const double z = (static_cast<double>(k) + 0.5) * params.dz();
        const double strength = sponge_strength(z, params);
        if (strength == 0.0) {
            continue;
        }
        const double target_u = center_sponge_target_u(z, params);
        const double target_v = center_sponge_target_v(params);
        const double target_theta = center_sponge_target_theta(z, params);
        const double target_qt = center_sponge_target_qt(z, params);
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                relax_to_target(state.u[n], target_u, strength, params.dt);
                relax_to_target(state.v[n], target_v, strength, params.dt);
                // The BOMEX specification has no scalar restoring above the
                // inversion.  Damping theta_l or q_t here creates artificial
                // column sources, so its sponge acts on velocity only.
                if (params.moisture_enabled && params.initial_condition != "bomex") {
                    relax_to_target(state.theta_l[n], target_theta, strength, params.dt);
                    relax_to_target(state.qt[n], target_qt, strength, params.dt);
                } else if (params.thermo_enabled && params.initial_condition != "bomex") {
                    relax_to_target(state.theta[n], target_theta, strength, params.dt);
                }
            }
        }
    }
    for (int k = 1; k < params.nz; ++k) {
        const double z = static_cast<double>(k) * params.dz();
        const double strength = sponge_strength(z, params);
        if (strength == 0.0) {
            continue;
        }
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                relax_to_target(state.w[z_face_idx(params, i, j, k)], 0.0, strength, params.dt);
            }
        }
    }
    enforce_walls(state.w, params);
}

void add_coriolis_geostrophic_forcing(
    Field& rhs_u,
    Field& rhs_v,
    const FlowState& state,
    const Params& params) {
    if (params.coriolis_f == 0.0) {
        return;
    }
    for (int k = 0; k < params.nz; ++k) {
        const double z = (static_cast<double>(k) + 0.5) * params.dz();
        const double geostrophic_u = params.initial_condition == "bomex"
            ? bomex_geostrophic_u(z)
            : params.geostrophic_u;
        const double geostrophic_v = params.initial_condition == "bomex" ? 0.0 : params.geostrophic_v;
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                rhs_u[n] += params.coriolis_f * (state.v[n] - geostrophic_v);
                rhs_v[n] += -params.coriolis_f * (state.u[n] - geostrophic_u);
            }
        }
    }
}

void resize_center(Field& q, const Params& params) {
    q.resize(params.real_size());
}

void resize_face(Field& q, const Params& params) {
    q.resize(params.z_face_size());
}

}  // namespace

TimestepWorkspace::TimestepWorkspace(const Params& params) {
    ensure(params);
}

TimestepWorkspace::~TimestepWorkspace() = default;

void TimestepWorkspace::ensure(const Params& params) {
    resize_center(rhs_u, params);
    resize_center(rhs_v, params);
    resize_face(rhs_w, params);
    resize_center(rhs_theta, params);
    resize_center(rhs_qv, params);
    resize_face(dwdx_face, params);
    resize_face(dwdy_face, params);
    resize_face(dwdz_face, params);
    resize_center(w_center, params);
    resize_face(u_on_w, params);
    resize_face(v_on_w, params);
    resize_center(lap_u, params);
    resize_center(lap_v, params);
    resize_face(lap_w, params);
    resize_center(grad.dudx, params);
    resize_center(grad.dudy, params);
    resize_center(grad.dudz, params);
    resize_center(grad.dvdx, params);
    resize_center(grad.dvdy, params);
    resize_center(grad.dvdz, params);
    resize_center(grad.dwdx, params);
    resize_center(grad.dwdy, params);
    resize_center(grad.dwdz, params);
    if (params.cuda_enabled && !cuda) {
        cuda = std::make_unique<CudaFlowState>(params);
    }
}

Diagnostics diagnostics(const FlowState& state, const Params& params, FftwXY& fft) {
    Diagnostics diag;
    const Field w_center = w_to_center(state.w, params);
    for (std::size_t n = 0; n < state.u.size(); ++n) {
        if (!std::isfinite(state.u[n]) || !std::isfinite(state.v[n]) || !std::isfinite(w_center[n])) {
            diag.ke_max = std::numeric_limits<double>::infinity();
            diag.cfl = std::numeric_limits<double>::infinity();
            continue;
        }
        const double ke = 0.5 * (state.u[n] * state.u[n] + state.v[n] * state.v[n] + w_center[n] * w_center[n]);
        diag.ke_max = std::max(diag.ke_max, ke);
    }
    const Field div = divergence(state.u, state.v, state.w, params, fft);
    diag.div_max = max_abs(div);
    const double inv_dx = 1.0 / params.dx();
    const double inv_dy = 1.0 / params.dy();
    const double inv_dz = 1.0 / params.dz();
    for (std::size_t n = 0; n < state.u.size(); ++n) {
        if (!std::isfinite(state.u[n]) || !std::isfinite(state.v[n]) || !std::isfinite(w_center[n])) {
            diag.cfl = std::numeric_limits<double>::infinity();
            continue;
        }
        diag.cfl = std::max(
            diag.cfl,
            params.dt * (std::abs(state.u[n]) * inv_dx + std::abs(state.v[n]) * inv_dy + std::abs(w_center[n]) * inv_dz));
    }
    if (params.moisture_enabled) {
        const auto [qv_min, qv_max] = std::minmax_element(state.qv.begin(), state.qv.end());
        diag.qv_min = *qv_min;
        diag.qv_max = *qv_max;
        const auto [qt_min, qt_max] = std::minmax_element(state.qt.begin(), state.qt.end());
        diag.qt_min = *qt_min;
        diag.qt_max = *qt_max;
        diag.ql_max = *std::max_element(state.ql.begin(), state.ql.end());
        diag.column_water = column_integrated_water(state.qt, params);
        const VelocityGradients grad = velocity_gradients(state, params, fft);
        const Field strain = strain_magnitude(grad, params);
        const Field nu_t = current_sgs_eddy_viscosity(state, grad, params, fft);
        const Field kappa_qv = moisture_eddy_diffusivity(state, nu_t, strain, params, fft);
        diag.moisture_diffusion_number = moisture_diffusion_number(kappa_qv, params);
        if (!std::isfinite(diag.qv_min) || !std::isfinite(diag.qv_max)) {
            diag.ke_max = std::numeric_limits<double>::infinity();
            diag.cfl = std::numeric_limits<double>::infinity();
        }
    }
    return diag;
}

void compute_rhs(
    FlowState& state,
    Field& rhs_u,
    Field& rhs_v,
    Field& rhs_w,
    Field& rhs_theta,
    const Params& params,
    FftwXY& fft) {
    TimestepWorkspace workspace(params);
    compute_rhs(state, params, fft, workspace);
    rhs_u = workspace.rhs_u;
    rhs_v = workspace.rhs_v;
    rhs_w = workspace.rhs_w;
    rhs_theta = workspace.rhs_theta;
}

void compute_rhs(
    FlowState& state,
    const Params& params,
    FftwXY& fft,
    TimestepWorkspace& workspace) {
    if (params.cuda_enabled) {
        throw std::runtime_error("compute_rhs is a host-side API; CUDA mode uses step() with the device-resident timestep path");
    }
    workspace.ensure(params);
    VelocityGradients& grad = workspace.grad;
    Field& rhs_u = workspace.rhs_u;
    Field& rhs_v = workspace.rhs_v;
    Field& rhs_w = workspace.rhs_w;

    w_to_center(state.w, workspace.w_center, params);
    workspace.moisture_advective_cfl = 0.0;
    if (params.moisture_enabled) {
        for (std::size_t n = 0; n < state.u.size(); ++n) {
            workspace.moisture_advective_cfl = std::max(
                workspace.moisture_advective_cfl,
                params.dt * (
                    std::abs(state.u[n]) / params.dx()
                    + std::abs(state.v[n]) / params.dy()
                    + std::abs(workspace.w_center[n]) / params.dz()));
        }
    }
    fft.derivative_x(state.u, grad.dudx, params);
    fft.derivative_y(state.u, grad.dudy, params);
    fft.derivative_x(state.v, grad.dvdx, params);
    fft.derivative_y(state.v, grad.dvdy, params);
    fft.derivative_x(workspace.w_center, grad.dwdx, params);
    fft.derivative_y(workspace.w_center, grad.dwdy, params);
    ddz_center(state.u, grad.dudz, params);
    ddz_center(state.v, grad.dvdz, params);
    ddz_w_to_center(state.w, grad.dwdz, params);
    laplacian_center(state.u, workspace.lap_u, params, fft);
    laplacian_center(state.v, workspace.lap_v, params, fft);

    fft.derivative_x_planes(state.w, params.nz + 1, workspace.dwdx_face, params);
    fft.derivative_y_planes(state.w, params.nz + 1, workspace.dwdy_face, params);
    ddz_w(state.w, workspace.dwdz_face, params);
    center_to_w(state.u, workspace.u_on_w, params);
    center_to_w(state.v, workspace.v_on_w, params);
    laplacian_w(state.w, workspace.lap_w, params, fft);

    std::fill(rhs_u.begin(), rhs_u.end(), 0.0);
    std::fill(rhs_v.begin(), rhs_v.end(), 0.0);
    std::fill(rhs_w.begin(), rhs_w.end(), 0.0);
    Field horizontal_adv_u;
    Field horizontal_adv_v;
    Field horizontal_adv_w;
    if (params.horizontal_dealias && params.dealiasing == "padding_3_2") {
        fft.horizontal_advective_derivative_3_2(state.u, state.v, state.u, 0, params.nz, horizontal_adv_u, params);
        fft.horizontal_advective_derivative_3_2(state.u, state.v, state.v, 0, params.nz, horizontal_adv_v, params);
        fft.horizontal_advective_derivative_3_2(
            workspace.u_on_w,
            workspace.v_on_w,
            state.w,
            0,
            params.nz + 1,
            horizontal_adv_w,
            params);
    }
    for (std::size_t n = 0; n < state.u.size(); ++n) {
        const double horizontal_u = params.horizontal_dealias && params.dealiasing == "padding_3_2"
            ? horizontal_adv_u[n]
            : state.u[n] * grad.dudx[n] + state.v[n] * grad.dudy[n];
        const double horizontal_v = params.horizontal_dealias && params.dealiasing == "padding_3_2"
            ? horizontal_adv_v[n]
            : state.u[n] * grad.dvdx[n] + state.v[n] * grad.dvdy[n];
        rhs_u[n] = -(horizontal_u + workspace.w_center[n] * grad.dudz[n])
            + params.nu * workspace.lap_u[n];
        rhs_v[n] = -(horizontal_v + workspace.w_center[n] * grad.dvdz[n])
            + params.nu * workspace.lap_v[n];
    }
    for (int k = 1; k < params.nz; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t face = z_face_idx(params, i, j, k);
                const double horizontal_w = params.horizontal_dealias && params.dealiasing == "padding_3_2"
                    ? horizontal_adv_w[face]
                    : workspace.u_on_w[face] * workspace.dwdx_face[face]
                        + workspace.v_on_w[face] * workspace.dwdy_face[face];
                rhs_w[face] =
                    -(horizontal_w + state.w[face] * workspace.dwdz_face[face])
                    + params.nu * workspace.lap_w[face];
            }
        }
    }
    add_coriolis_geostrophic_forcing(rhs_u, rhs_v, state, params);
    const SgsViscosity sgs = update_sgs_eddy_viscosity(state, grad, params, fft);
    add_sgs_momentum_forcing(rhs_u, rhs_v, rhs_w, state, grad, sgs.eddy_viscosity, params, fft);
    apply_wall_stress(rhs_u, rhs_v, state, params, fft);
    const Field* test_strain_2d = sgs.test_strain_2d.empty() ? nullptr : &sgs.test_strain_2d;
    const Field* test_strain_4d = sgs.test_strain_4d.empty() ? nullptr : &sgs.test_strain_4d;
    workspace.rhs_theta = scalar_rhs(
        state, sgs.eddy_viscosity, sgs.strain, params, fft, test_strain_2d, test_strain_4d);
    double theta_diffusion_number = 0.0;
    if (params.moisture_enabled) {
        const Field theta_kappa = scalar_eddy_diffusivity(state, sgs.eddy_viscosity, sgs.strain, params, fft);
        theta_diffusion_number = moisture_diffusion_number(theta_kappa, params);
    }
    workspace.rhs_qv = moisture_rhs(
        state,
        sgs.eddy_viscosity,
        sgs.strain,
        params,
        fft,
        &workspace.moisture_diffusion_number,
        test_strain_2d,
        test_strain_4d);
    workspace.moisture_diffusion_number = std::max(
        workspace.moisture_diffusion_number, theta_diffusion_number);
    add_bomex_large_scale_forcing(
        rhs_u,
        rhs_v,
        workspace.rhs_theta,
        workspace.rhs_qv,
        state,
        params);
    if (sgs.lasd_updated) {
        reset_lasd_velocity_accumulators(state);
    }
    add_buoyancy(rhs_w, state, params);
}

void step(FlowState& state, const Params& params, FftwXY& fft) {
    TimestepWorkspace workspace(params);
    step(state, params, fft, workspace);
}

void step(FlowState& state, const Params& params, FftwXY& fft, TimestepWorkspace& workspace) {
    if (params.cuda_enabled) {
        workspace.ensure(params);
        const bool use_ab2 = params.time_scheme == "ab2" && state.has_rhs_prev;
        cuda_step_device_resident(state, workspace, use_ab2, params);
        state.has_rhs_prev = true;
        ++state.step_count;
        return;
    }

    compute_rhs(state, params, fft, workspace);
    if (params.moisture_enabled && workspace.moisture_advective_cfl > 1.0) {
        throw std::runtime_error(
            "moisture advection CFL exceeds 1: "
            + std::to_string(workspace.moisture_advective_cfl));
    }
    if (params.moisture_enabled && params.sgs_model == "lasd"
        && static_cast<double>(params.cs_count) * workspace.moisture_advective_cfl > 1.0) {
        throw std::runtime_error(
            "LASD update-interval CFL exceeds 1: "
            + std::to_string(static_cast<double>(params.cs_count) * workspace.moisture_advective_cfl));
    }
    if (params.moisture_enabled && workspace.moisture_diffusion_number > 1.0) {
        throw std::runtime_error(
            "moisture explicit-diffusion stability number exceeds 1: "
            + std::to_string(workspace.moisture_diffusion_number));
    }
    const bool use_ab2 = params.time_scheme == "ab2" && state.has_rhs_prev;
    advance_field(state.u, workspace.rhs_u, state.rhs_u_prev, use_ab2, params.dt);
    advance_field(state.v, workspace.rhs_v, state.rhs_v_prev, use_ab2, params.dt);
    advance_field(state.w, workspace.rhs_w, state.rhs_w_prev, use_ab2, params.dt);
    if (params.moisture_enabled) {
        advance_field(state.theta_l, workspace.rhs_theta, state.rhs_theta_prev, use_ab2, params.dt);
    } else if (params.thermo_enabled) {
        advance_field(state.theta, workspace.rhs_theta, state.rhs_theta_prev, use_ab2, params.dt);
    } else {
        state.rhs_theta_prev = workspace.rhs_theta;
    }
    if (params.moisture_enabled) {
        advance_field(state.qt, workspace.rhs_qv, state.rhs_qv_prev, use_ab2, params.dt);
    } else {
        state.rhs_qv_prev = workspace.rhs_qv;
    }
    state.has_rhs_prev = true;
    apply_rayleigh_sponge(state, params);
    enforce_walls(state.w, params);
    horizontal_dealias_state(state, params, fft);
    if (params.moisture_enabled) {
        const double correction = enforce_nonnegative_conservative(state.qt);
        if (correction > 0.0) {
            ++state.moisture_limiter_activations;
            state.moisture_limiter_column_correction += correction * params.dz()
                / static_cast<double>(params.nx * params.ny);
        }
        update_moist_thermodynamics(state, params);
    }
    project(state, params, fft);
    ++state.step_count;
}

}  // namespace wireles

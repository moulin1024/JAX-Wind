#pragma once

#include <memory>

#include "wireles/fft.hpp"
#include "wireles/field.hpp"
#include "wireles/sgs.hpp"

namespace wireles {

class CudaFlowState;

struct Diagnostics {
    double ke_max = 0.0;
    double div_max = 0.0;
    double cfl = 0.0;
    double qv_min = 0.0;
    double qv_max = 0.0;
    double qt_min = 0.0;
    double qt_max = 0.0;
    double ql_max = 0.0;
    double column_water = 0.0;
    double moisture_diffusion_number = 0.0;
    double lasd_cs2_mean = 0.0;
    double lasd_cs2_max = 0.0;
    double lasd_beta_mean = 0.0;
    double lasd_beta_floor_fraction = 0.0;
    double lasd_theta_c_mean = 0.0;
    double lasd_theta_c_max = 0.0;
    double lasd_theta_beta_floor_fraction = 0.0;
    double lasd_qt_c_mean = 0.0;
    double lasd_qt_c_max = 0.0;
    double lasd_qt_beta_floor_fraction = 0.0;
};

struct TimestepWorkspace {
    Field rhs_u;
    Field rhs_v;
    Field rhs_w;
    Field rhs_theta;
    Field rhs_qv;
    Field dwdx_face;
    Field dwdy_face;
    Field dwdz_face;
    Field w_center;
    Field u_on_w;
    Field v_on_w;
    Field lap_u;
    Field lap_v;
    Field lap_w;
    VelocityGradients grad;
    double moisture_advective_cfl = 0.0;
    double moisture_diffusion_number = 0.0;
    std::unique_ptr<CudaFlowState> cuda;

    TimestepWorkspace() = default;
    explicit TimestepWorkspace(const Params& params);
    ~TimestepWorkspace();
    void ensure(const Params& params);
};

Diagnostics diagnostics(const FlowState& state, const Params& params, FftwXY& fft);
void compute_rhs(
    FlowState& state,
    Field& rhs_u,
    Field& rhs_v,
    Field& rhs_w,
    Field& rhs_theta,
    const Params& params,
    FftwXY& fft);
void compute_rhs(
    FlowState& state,
    const Params& params,
    FftwXY& fft,
    TimestepWorkspace& workspace);
void step(FlowState& state, const Params& params, FftwXY& fft);
void step(FlowState& state, const Params& params, FftwXY& fft, TimestepWorkspace& workspace);

}  // namespace wireles

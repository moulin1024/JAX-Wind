#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "wireles/field.hpp"
#include "wireles/params.hpp"

namespace wireles {

double bomex_initial_theta_l(double z);
double bomex_initial_qt(double z);
double bomex_specific_to_mixing_ratio(double specific_humidity);
double bomex_mixing_to_specific_humidity(double mixing_ratio);
double bomex_surface_qt_mixing_ratio_flux(double specified_specific_humidity_flux);
double bomex_initial_u(double z);
double bomex_geostrophic_u(double z);
double bomex_subsidence(double z);
double bomex_radiative_tendency(double z);
double bomex_moisture_advection_tendency(double z);
double bomex_column_water_large_scale_tendency(const FlowState& state, const Params& params);
const std::vector<double>& bomex_cloud_thresholds();

void add_bomex_large_scale_forcing(
    Field& rhs_u,
    Field& rhs_v,
    Field& rhs_theta_l,
    Field& rhs_qt,
    const FlowState& state,
    const Params& params);

struct BomexAccumulator {
    std::size_t samples = 0;
    std::vector<double> theta_l;
    std::vector<double> qt;
    std::vector<double> qv;
    std::vector<double> ql;
    std::vector<double> u;
    std::vector<double> v;
    std::vector<double> tke;
    std::vector<double> cloud_fraction;
    std::vector<double> cloud_fraction_by_threshold;
    std::vector<double> core_fraction;
    std::vector<double> w_variance;
    std::vector<double> u_variance;
    std::vector<double> v_variance;
    std::vector<double> mean_eddy_viscosity;
    std::vector<double> mean_sgs_tke;
    std::vector<double> mean_strain_squared;
    std::vector<double> mean_sgs_dissipation;
    std::vector<double> zero_eddy_viscosity_fraction;
    std::vector<double> mean_theta_l_scalar_c;
    std::vector<double> mean_qt_scalar_c;
    std::vector<double> mean_theta_l_scalar_diffusivity;
    std::vector<double> mean_qt_scalar_diffusivity;
    std::vector<double> zero_theta_l_scalar_diffusivity_fraction;
    std::vector<double> zero_qt_scalar_diffusivity_fraction;
    std::vector<double> mean_theta_v;
    std::vector<double> resolved_vw_flux;
    std::vector<double> sgs_vw_flux;
    std::vector<double> resolved_theta_l_flux;
    std::vector<double> resolved_qt_flux;
    std::vector<double> resolved_ql_flux;
    std::vector<double> resolved_theta_v_flux;
    std::vector<double> resolved_uw_flux;
    std::vector<double> sgs_theta_l_flux;
    std::vector<double> sgs_qt_flux;
    std::vector<double> sgs_ql_flux;
    std::vector<double> sgs_theta_v_flux;
    std::vector<double> sgs_uw_flux;
    std::vector<double> cloud_theta_l;
    std::vector<double> core_theta_l;
    std::vector<double> cloud_qt;
    std::vector<double> core_qt;
    std::vector<double> cloud_theta_v;
    std::vector<double> core_theta_v;
    std::vector<double> cloud_ql;
    std::vector<double> core_ql;
    std::vector<double> cloud_w;
    std::vector<double> core_w;
    // Resolved-TKE budget ingredients (handoff section 13.4): vertical fluxes
    // of resolved TKE and pressure, plus the fluctuation work done by the
    // prescribed surface stress at the first model level (zero elsewhere).
    std::vector<double> resolved_w_tke_flux;
    std::vector<double> resolved_w_pressure_flux;
    std::vector<double> wall_fluctuation_tke_work;
    std::vector<std::size_t> cloud_conditional_samples;
    std::vector<std::size_t> core_conditional_samples;
    double total_cloud_cover = 0.0;
    std::vector<double> total_cloud_cover_by_threshold;
    double liquid_water_path = 0.0;
    std::vector<int> sample_step;
    std::vector<double> sample_time_s;
    std::vector<double> sample_total_cloud_cover;
    std::vector<double> sample_total_cloud_cover_by_threshold;
    std::vector<double> sample_max_cloud_fraction;
    std::vector<double> sample_liquid_water_path;
    // Density-weighted vertical integral of resolved TKE, matching the units
    // of BOMEX Figure 2c (kg m^-1 s^-2).
    std::vector<double> sample_integrated_tke;
    std::vector<double> sample_column_qt_m;
    std::vector<double> sample_qt_large_scale_tendency_m_s;
};

void add_bomex_sample(BomexAccumulator& accumulator, const FlowState& state, const Params& params);
void print_bomex_summary(const BomexAccumulator& accumulator, const Params& params);
void write_bomex_outputs(const BomexAccumulator& accumulator, const FlowState& state, const Params& params);

}  // namespace wireles

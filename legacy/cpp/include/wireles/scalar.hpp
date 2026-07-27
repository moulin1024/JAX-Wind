#pragma once

#include "wireles/fft.hpp"
#include "wireles/field.hpp"

namespace wireles {

Field virtual_potential_temperature(const FlowState& state, const Params& params);
Field scalar_eddy_diffusivity(
    const FlowState& state,
    const Field& eddy_viscosity,
    const Field& strain,
    const Params& params,
    FftwXY& fft);
Field scalar_rhs(
    FlowState& state,
    const Field& eddy_viscosity,
    const Field& strain,
    const Params& params,
    FftwXY& fft,
    const Field* test_strain_2d = nullptr,
    const Field* test_strain_4d = nullptr);
Field moisture_rhs(
    FlowState& state,
    const Field& eddy_viscosity,
    const Field& strain,
    const Params& params,
    FftwXY& fft,
    double* diffusion_number = nullptr,
    const Field* test_strain_2d = nullptr,
    const Field* test_strain_4d = nullptr);
Field moisture_eddy_diffusivity(
    const FlowState& state,
    const Field& eddy_viscosity,
    const Field& strain,
    const Params& params,
    FftwXY& fft);
// Diagnostic redistribution for horizontally homogeneous cases.  Replaces
// the local SGS diffusivity in each x-y plane with the gradient-energy-
// weighted plane value, preserving sum(K_sgs * |grad(phi)|^2) at every z.
void redistribute_scalar_diffusivity_by_plane_dissipation(
    Field& diffusivity,
    const Field& dscalar_dx,
    const Field& dscalar_dy,
    const Field& dscalar_dz_w,
    double molecular_diffusivity,
    const Params& params);
double moisture_diffusion_number(const Field& diffusivity, const Params& params);
double column_integrated_water(const Field& qv, const Params& params);
double enforce_nonnegative_conservative(Field& qv);
void add_buoyancy(Field& rhs_w, const FlowState& state, const Params& params);
void step_scalar(FlowState& state, const Field& rhs_theta, const Params& params);

}  // namespace wireles

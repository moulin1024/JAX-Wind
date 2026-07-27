#pragma once

#include <array>
#include <utility>

#include "wireles/fft.hpp"
#include "wireles/field.hpp"

namespace wireles {

struct VelocityGradients {
    Field dudx;
    Field dudy;
    Field dudz;
    Field dvdx;
    Field dvdy;
    Field dvdz;
    Field dwdx;
    Field dwdy;
    Field dwdz;
};

struct SgsViscosity {
    Field eddy_viscosity;
    Field strain;
    Field test_strain_2d;
    Field test_strain_4d;
    bool lasd_updated = false;
};

// Signed numerator and non-negative denominator of a minimum-dissipation
// coefficient before the clipped ratio is formed.  Keeping them separate
// allows the invariants to be volume-averaged (the Poincare bound underlying
// AMD holds over a filter volume, not pointwise) before clipping.
struct AmdInvariant {
    double numerator = 0.0;
    double denominator = 0.0;
};

// max(numerator, 0) / denominator with the same finiteness/positivity guards
// used by the pointwise AMD evaluations.
double amd_invariant_ratio(const AmdInvariant& invariant);

// In-place local (1-2-1)^3 average of an AMD invariant field on cell centers:
// periodic horizontally, clamped and renormalized at the vertical boundaries.
void smooth_amd_invariant_field(Field& q, const Params& params);

// Row-major velocity-gradient layout: {du/dx, du/dy, du/dz,
//                                      dv/dx, dv/dy, dv/dz,
//                                      dw/dx, dw/dy, dw/dz}.
// scaled_cell_width contains sqrt(C_i) * Delta_i, where C_i is the
// direction-dependent modified Poincare constant.
double amd_eddy_viscosity_at(
    const std::array<double, 9>& velocity_gradient,
    const std::array<double, 3>& buoyancy_gradient,
    const std::array<double, 3>& scaled_cell_width);
// Discrete AMD evaluation for the vertical stagger used by WIRELES.  The
// face arrays contain {du/dz, dv/dz, dw/dx, dw/dy} on the lower and upper
// w faces of a scalar cell.  Quadratic products are formed at their natural
// locations before interpolation to the cell center.
double amd_eddy_viscosity_staggered_at(
    const std::array<double, 9>& center_velocity_gradient,
    const std::array<double, 4>& lower_face_gradient,
    const std::array<double, 4>& upper_face_gradient,
    const std::array<double, 3>& buoyancy_gradient,
    const std::array<double, 3>& scaled_cell_width);
AmdInvariant amd_eddy_viscosity_staggered_invariant_at(
    const std::array<double, 9>& center_velocity_gradient,
    const std::array<double, 4>& lower_face_gradient,
    const std::array<double, 4>& upper_face_gradient,
    const std::array<double, 3>& buoyancy_gradient,
    const std::array<double, 3>& scaled_cell_width);
double amd_scalar_diffusivity_at(
    const std::array<double, 9>& velocity_gradient,
    const std::array<double, 3>& scalar_gradient,
    const std::array<double, 3>& scaled_cell_width);
double amd_scalar_diffusivity_staggered_at(
    const std::array<double, 9>& center_velocity_gradient,
    const std::array<double, 3>& center_scalar_gradient,
    double lower_scalar_dz,
    double upper_scalar_dz,
    const std::array<double, 3>& scaled_cell_width);
AmdInvariant amd_scalar_diffusivity_staggered_invariant_at(
    const std::array<double, 9>& center_velocity_gradient,
    const std::array<double, 3>& center_scalar_gradient,
    double lower_scalar_dz,
    double upper_scalar_dz,
    const std::array<double, 3>& scaled_cell_width);
// Fully staggered scalar AMD invariant: every quadratic product involving a
// vertical derivative or a w derivative is formed on its natural w-face
// location before interpolation to the cell center, mirroring the accepted
// staggered momentum AMD.  The face arrays contain {du/dz, dv/dz, dw/dx,
// dw/dy} on the lower and upper faces; the partially staggered variant above
// instead contracts center-interpolated gradients in the numerator.
AmdInvariant amd_scalar_diffusivity_face_product_invariant_at(
    const std::array<double, 9>& center_velocity_gradient,
    const std::array<double, 4>& lower_face_gradient,
    const std::array<double, 4>& upper_face_gradient,
    double center_scalar_dx,
    double center_scalar_dy,
    double lower_scalar_dz,
    double upper_scalar_dz,
    const std::array<double, 3>& scaled_cell_width);
// Smallest common diffusivity satisfying the AMD dissipation requirement for
// both conserved moist scalars, without introducing a fitted blend.
double amd_shared_scalar_diffusivity_at(
    const std::array<double, 9>& velocity_gradient,
    const std::array<double, 3>& theta_l_gradient,
    const std::array<double, 3>& qt_gradient,
    const std::array<double, 3>& scaled_cell_width);
std::array<double, 3> amd_scaled_cell_width(const Params& params);
// Diagnostic homogeneous-plane redistribution of local AMD viscosity.  It
// preserves sum(nu_t |S|^2) independently on every horizontal plane.
void redistribute_amd_eddy_viscosity_by_plane_dissipation(
    Field& eddy_viscosity,
    const Field& strain,
    const Params& params);
// Compact-filter analogue of the plane redistribution: replace nu_t by the
// ratio of locally averaged AMD dissipation and locally averaged |S|^2.
void average_amd_eddy_viscosity_by_local_dissipation(
    Field& eddy_viscosity,
    const Field& strain,
    const Params& params);
void enforce_multiscale_amd_invariant_bound(
    Field& eddy_viscosity,
    Field positive_numerator,
    Field denominator,
    const Params& params);

VelocityGradients velocity_gradients(const FlowState& state, const Params& params, FftwXY& fft);
Field strain_magnitude(const VelocityGradients& grad, const Params& params);
Field test_filtered_strain_magnitude(
    const VelocityGradients& grad,
    const Params& params,
    FftwXY& fft,
    double filter_width);
Field smagorinsky_eddy_viscosity(const VelocityGradients& grad, const Params& params);
Field smagorinsky_eddy_viscosity(
    const FlowState& state,
    const VelocityGradients& grad,
    const Params& params);
SgsViscosity update_sgs_eddy_viscosity(FlowState& state, const VelocityGradients& grad, const Params& params, FftwXY& fft);
Field current_sgs_eddy_viscosity(
    const FlowState& state,
    const VelocityGradients& grad,
    const Params& params,
    FftwXY& fft);
void reset_lasd_velocity_accumulators(FlowState& state);
Field horizontal_spectral_filter(const Field& in, const Params& params, FftwXY& fft, double filter_width);
void apply_center_history_bc(Field& q, const Params& params);
Field lagrangian_interp_center(
    const Field& q,
    const Field& u_lag,
    const Field& v_lag,
    const Field& w_lag,
    const Params& params);
std::pair<Field, Field> lagrangian_average_fields(
    const Field& current_a,
    const Field& current_b,
    const Field& old_a,
    const Field& old_b,
    const Field& u_lag,
    const Field& v_lag,
    const Field& w_lag,
    const Params& params,
    const Field* timescale_a = nullptr,
    const Field* timescale_b = nullptr);
void add_sgs_momentum_forcing(
    Field& rhs_u,
    Field& rhs_v,
    Field& rhs_w,
    const FlowState& state,
    const VelocityGradients& grad,
    const Field& nu_t,
    const Params& params,
    FftwXY& fft);

}  // namespace wireles

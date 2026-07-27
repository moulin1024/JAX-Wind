#pragma once

#include <array>

#include "wireles/fft.hpp"
#include "wireles/field.hpp"

namespace wireles {

struct WallStress {
    Field tau_xz;
    Field tau_yz;
    Field ustar;
};

WallStress dynamic_neutral_wall_stress(const FlowState& state, const Params& params, FftwXY& fft);
void apply_wall_stress(Field& rhs_u, Field& rhs_v, const FlowState& state, const Params& params, FftwXY& fft);

// Drag coefficient of the local prescribed-u* surface stress
// tau_i = -C |U| U_i, chosen so that the plane-mean stress vector has
// magnitude exactly u_fric^2 for the instantaneous first-level velocity.
// Inputs are the plane means of |U| u and |U| v.
double prescribed_ustar_local_drag_coefficient(
    double plane_mean_speed_times_u,
    double plane_mean_speed_times_v,
    double u_fric);

// Plane-mean first-level velocity gradient from the neutral wall-similarity
// relation, aligned with the plane-mean horizontal velocity.
std::array<double, 2> wall_model_mean_velocity_gradient(
    double mean_u,
    double mean_v,
    const Params& params);

}  // namespace wireles

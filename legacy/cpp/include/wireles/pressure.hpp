#pragma once

#include "wireles/fft.hpp"
#include "wireles/field.hpp"

namespace wireles {

SpectralField solve_pressure_hat(const SpectralField& div_hat, const Params& params);
void project(FlowState& state, const Params& params, FftwXY& fft);

}  // namespace wireles

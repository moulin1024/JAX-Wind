#pragma once

#include <complex>
#include <cstddef>
#include <vector>

#include "wireles/params.hpp"

namespace wireles {

using Complex = std::complex<double>;
using Field = std::vector<double>;
using SpectralField = std::vector<Complex>;

inline std::size_t z_face_idx(const Params& p, int i, int j, int k) {
    return (static_cast<std::size_t>(k) * static_cast<std::size_t>(p.ny) + static_cast<std::size_t>(j))
        * static_cast<std::size_t>(p.nx)
        + static_cast<std::size_t>(i);
}

struct FlowState {
    Field u;
    Field v;
    Field w;
    Field p;
    Field theta;
    Field qv;
    Field theta_l;
    Field qt;
    Field ql;
    Field sgs_tke;
    Field base_pressure;
    Field rhs_u_prev;
    Field rhs_v_prev;
    Field rhs_w_prev;
    Field rhs_theta_prev;
    Field rhs_qv_prev;
    Field rhs_sgs_tke_prev;
    Field cs2;
    Field lm_old;
    Field mm_old;
    Field qn_old;
    Field nn_old;
    Field scalar_c;
    Field scalar_lm_old;
    Field scalar_mm_old;
    Field scalar_qn_old;
    Field scalar_nn_old;
    Field qt_scalar_c;
    Field qt_scalar_lm_old;
    Field qt_scalar_mm_old;
    Field qt_scalar_qn_old;
    Field qt_scalar_nn_old;
    Field u_lag;
    Field v_lag;
    Field w_lag;
    int step_count = 0;
    bool has_rhs_prev = false;
    std::size_t moisture_limiter_activations = 0;
    double moisture_limiter_column_correction = 0.0;

    explicit FlowState(const Params& params);
};

void initialize(FlowState& state, const Params& params);
void initialize_moist_thermodynamics(FlowState& state, const Params& params);
void update_moist_thermodynamics(FlowState& state, const Params& params);
void enforce_walls(Field& w, const Params& params);
double max_abs(const Field& q);

}  // namespace wireles

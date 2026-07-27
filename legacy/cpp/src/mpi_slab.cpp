#include "wireles/mpi_slab.hpp"

#include <mpi.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <utility>
#include <vector>

#include "wireles/bomex.hpp"
#include "wireles/fft.hpp"
#include "wireles/field.hpp"
#include "wireles/operators.hpp"
#include "wireles/pressure.hpp"
#include "wireles/sgs.hpp"
#include "wireles/timestep.hpp"
#include "wireles/thermodynamics.hpp"
#include "wireles/wall.hpp"

namespace wireles {
namespace {

struct MpiRuntime {
    MpiRuntime(int argc, char** argv) {
        MPI_Init(&argc, &argv);
    }
    MpiRuntime(const MpiRuntime&) = delete;
    MpiRuntime& operator=(const MpiRuntime&) = delete;
    ~MpiRuntime() {
        MPI_Finalize();
    }
};

struct Slab {
    int rank = 0;
    int size = 1;
    int k_begin = 0;
    int k_count = 0;
    int face_begin = 0;
    int face_count = 0;
    std::vector<int> center_counts;
    std::vector<int> center_displs;
    std::vector<int> face_counts;
    std::vector<int> face_displs;
};

std::size_t plane_size(const Params& params) {
    return static_cast<std::size_t>(params.nx) * static_cast<std::size_t>(params.ny);
}

struct LocalField {
    Field values;
    int plane_begin = 0;
    int plane_count = 0;
    int owned_begin = 0;
    int owned_count = 0;
    int total_planes = 0;
    std::size_t plane_stride = 0;

    void resize(int begin, int count, int total, const Params& params, bool with_halo = true) {
        owned_begin = begin;
        owned_count = count;
        total_planes = total;
        plane_stride = plane_size(params);
        const int lower_halo = with_halo && begin > 0 ? 1 : 0;
        const int upper_halo = with_halo && begin + count < total ? 1 : 0;
        plane_begin = begin - lower_halo;
        plane_count = count + lower_halo + upper_halo;
        values.resize(static_cast<std::size_t>(plane_count) * plane_stride);
    }

    void clear_owned() {
        for (int k = owned_begin; k < owned_begin + owned_count; ++k) {
            auto begin = values.begin() + static_cast<std::ptrdiff_t>((k - plane_begin) * plane_stride);
            std::fill(begin, begin + static_cast<std::ptrdiff_t>(plane_stride), 0.0);
        }
    }

    std::size_t local_offset_from_global_flat(std::size_t global_flat) const {
        const int k_global = static_cast<int>(global_flat / plane_stride);
        const std::size_t in_plane = global_flat % plane_stride;
        return static_cast<std::size_t>(k_global - plane_begin) * plane_stride + in_plane;
    }

    double& operator[](std::size_t global_flat) {
        return values[local_offset_from_global_flat(global_flat)];
    }

    const double& operator[](std::size_t global_flat) const {
        return values[local_offset_from_global_flat(global_flat)];
    }
};

struct LocalVelocityGradients {
    LocalField dudx;
    LocalField dudy;
    LocalField dudz;
    LocalField dvdx;
    LocalField dvdy;
    LocalField dvdz;
    LocalField dwdx;
    LocalField dwdy;
    LocalField dwdz;

    void resize(const Params& params, const Slab& slab) {
        dudx.resize(slab.k_begin, slab.k_count, params.nz, params);
        dudy.resize(slab.k_begin, slab.k_count, params.nz, params);
        dudz.resize(slab.k_begin, slab.k_count, params.nz, params);
        dvdx.resize(slab.k_begin, slab.k_count, params.nz, params);
        dvdy.resize(slab.k_begin, slab.k_count, params.nz, params);
        dvdz.resize(slab.k_begin, slab.k_count, params.nz, params);
        dwdx.resize(slab.k_begin, slab.k_count, params.nz, params);
        dwdy.resize(slab.k_begin, slab.k_count, params.nz, params);
        dwdz.resize(slab.k_begin, slab.k_count, params.nz, params);
    }
};

struct MpiLocalFlowState {
    LocalField u;
    LocalField v;
    LocalField w;
    LocalField p;
    LocalField theta;
    LocalField qv;
    LocalField theta_l;
    LocalField qt;
    LocalField ql;
    LocalField sgs_tke;
    std::vector<double> base_pressure;
    LocalField rhs_u_prev;
    LocalField rhs_v_prev;
    LocalField rhs_w_prev;
    LocalField rhs_theta_prev;
    LocalField rhs_qt_prev;
    LocalField rhs_sgs_tke_prev;
    LocalField rhs_u_prev2;
    LocalField rhs_v_prev2;
    LocalField rhs_w_prev2;
    LocalField rhs_theta_prev2;
    LocalField rhs_qt_prev2;
    LocalField rhs_sgs_tke_prev2;
    LocalField cs2;
    LocalField lm_old;
    LocalField mm_old;
    LocalField qn_old;
    LocalField nn_old;
    LocalField scalar_c;
    LocalField scalar_lm_old;
    LocalField scalar_mm_old;
    LocalField scalar_qn_old;
    LocalField scalar_nn_old;
    LocalField qt_scalar_c;
    LocalField qt_scalar_lm_old;
    LocalField qt_scalar_mm_old;
    LocalField qt_scalar_qn_old;
    LocalField qt_scalar_nn_old;
    LocalField u_lag;
    LocalField v_lag;
    LocalField w_lag;
    int step_count = 0;
    bool has_rhs_prev = false;

    explicit MpiLocalFlowState(const Params& params, const Slab& slab) {
        resize(params, slab);
    }

    void resize(const Params& params, const Slab& slab) {
        u.resize(slab.k_begin, slab.k_count, params.nz, params);
        v.resize(slab.k_begin, slab.k_count, params.nz, params);
        w.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        p.resize(slab.k_begin, slab.k_count, params.nz, params);
        theta.resize(slab.k_begin, slab.k_count, params.nz, params);
        qv.resize(slab.k_begin, slab.k_count, params.nz, params);
        theta_l.resize(slab.k_begin, slab.k_count, params.nz, params);
        qt.resize(slab.k_begin, slab.k_count, params.nz, params);
        ql.resize(slab.k_begin, slab.k_count, params.nz, params);
        sgs_tke.resize(slab.k_begin, slab.k_count, params.nz, params);
        base_pressure.assign(static_cast<std::size_t>(params.nz), 0.0);
        rhs_u_prev.resize(slab.k_begin, slab.k_count, params.nz, params);
        rhs_v_prev.resize(slab.k_begin, slab.k_count, params.nz, params);
        rhs_w_prev.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        rhs_theta_prev.resize(slab.k_begin, slab.k_count, params.nz, params);
        rhs_qt_prev.resize(slab.k_begin, slab.k_count, params.nz, params);
        rhs_sgs_tke_prev.resize(slab.k_begin, slab.k_count, params.nz, params);
        rhs_u_prev2.resize(slab.k_begin, slab.k_count, params.nz, params);
        rhs_v_prev2.resize(slab.k_begin, slab.k_count, params.nz, params);
        rhs_w_prev2.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        rhs_theta_prev2.resize(slab.k_begin, slab.k_count, params.nz, params);
        rhs_qt_prev2.resize(slab.k_begin, slab.k_count, params.nz, params);
        rhs_sgs_tke_prev2.resize(slab.k_begin, slab.k_count, params.nz, params);
        cs2.resize(slab.k_begin, slab.k_count, params.nz, params);
        lm_old.resize(slab.k_begin, slab.k_count, params.nz, params);
        mm_old.resize(slab.k_begin, slab.k_count, params.nz, params);
        qn_old.resize(slab.k_begin, slab.k_count, params.nz, params);
        nn_old.resize(slab.k_begin, slab.k_count, params.nz, params);
        scalar_c.resize(slab.k_begin, slab.k_count, params.nz, params);
        scalar_lm_old.resize(slab.k_begin, slab.k_count, params.nz, params);
        scalar_mm_old.resize(slab.k_begin, slab.k_count, params.nz, params);
        scalar_qn_old.resize(slab.k_begin, slab.k_count, params.nz, params);
        scalar_nn_old.resize(slab.k_begin, slab.k_count, params.nz, params);
        qt_scalar_c.resize(slab.k_begin, slab.k_count, params.nz, params);
        qt_scalar_lm_old.resize(slab.k_begin, slab.k_count, params.nz, params);
        qt_scalar_mm_old.resize(slab.k_begin, slab.k_count, params.nz, params);
        qt_scalar_qn_old.resize(slab.k_begin, slab.k_count, params.nz, params);
        qt_scalar_nn_old.resize(slab.k_begin, slab.k_count, params.nz, params);
        u_lag.resize(slab.k_begin, slab.k_count, params.nz, params);
        v_lag.resize(slab.k_begin, slab.k_count, params.nz, params);
        w_lag.resize(slab.k_begin, slab.k_count, params.nz, params);

        const double cs2_initial = params.smagorinsky_cs * params.smagorinsky_cs;
        const double scalar_c_initial = cs2_initial / params.prandtl_t;
        const double qt_scalar_c_initial = cs2_initial / params.schmidt_t;
        for (LocalField* field : {
                 &u,
                 &v,
                 &w,
                 &p,
                 &theta,
                 &qv,
                 &theta_l,
                 &qt,
                 &ql,
                 &sgs_tke,
                 &rhs_u_prev,
                 &rhs_v_prev,
                 &rhs_w_prev,
                 &rhs_theta_prev,
                 &rhs_qt_prev,
                 &rhs_sgs_tke_prev,
                 &rhs_u_prev2,
                 &rhs_v_prev2,
                 &rhs_w_prev2,
                 &rhs_theta_prev2,
                 &rhs_qt_prev2,
                 &rhs_sgs_tke_prev2,
                 &lm_old,
                 &mm_old,
                 &qn_old,
                 &nn_old,
                 &scalar_lm_old,
                 &scalar_mm_old,
                 &scalar_qn_old,
                 &scalar_nn_old,
                 &qt_scalar_lm_old,
                 &qt_scalar_mm_old,
                 &qt_scalar_qn_old,
                 &qt_scalar_nn_old,
                 &u_lag,
                 &v_lag,
                 &w_lag,
             }) {
            std::fill(field->values.begin(), field->values.end(), 0.0);
        }
        std::fill(cs2.values.begin(), cs2.values.end(), cs2_initial);
        std::fill(scalar_c.values.begin(), scalar_c.values.end(), scalar_c_initial);
        std::fill(qt_scalar_c.values.begin(), qt_scalar_c.values.end(), qt_scalar_c_initial);
        std::fill(sgs_tke.values.begin(), sgs_tke.values.end(), params.tke_floor);
        step_count = 0;
        has_rhs_prev = false;
    }
};

using LocalSymFields = std::array<LocalField, 6>;
using LocalVecFields = std::array<LocalField, 3>;
using ConstLocalVecFields = std::array<const LocalField*, 3>;

void release_local_field_storage(LocalField& field) {
    Field{}.swap(field.values);
    field.plane_begin = 0;
    field.plane_count = 0;
    field.owned_begin = 0;
    field.owned_count = 0;
    field.total_planes = 0;
    field.plane_stride = 0;
}

void release_local_vec_field_storage(LocalVecFields& fields) {
    for (LocalField& field : fields) {
        release_local_field_storage(field);
    }
}

void match_local_layout(LocalField& output, const LocalField& input) {
    output.plane_begin = input.plane_begin;
    output.plane_count = input.plane_count;
    output.owned_begin = input.owned_begin;
    output.owned_count = input.owned_count;
    output.total_planes = input.total_planes;
    output.plane_stride = input.plane_stride;
    output.values.resize(input.values.size());
}

void copy_center_owned(const LocalField& source, LocalField& dest, const Params& params, const Slab& slab) {
    for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                dest[n] = source[n];
            }
        }
    }
}

struct LmMm {
    LocalField lm;
    LocalField mm;
};

struct ScalarLmMm {
    LocalField lm;
    LocalField mm;
};

struct LocalHaloField {
    LocalField* data = nullptr;
    int owned_begin = 0;
    int owned_count = 0;
    int total_planes = 0;
};

struct HaloExchangeScratch {
    std::vector<double> send_lower;
    std::vector<double> send_upper;
    std::vector<double> recv_lower;
    std::vector<double> recv_upper;
    std::vector<MPI_Request> requests;
};

struct LasdFilterCacheSlot {
    double width = -1.0;
    int face_begin = -1;
    int face_count = -1;
    bool u_hat_valid = false;
    bool v_hat_valid = false;
    bool w_center_hat_valid = false;
    bool w_face_hat_valid = false;
    bool strain_hat_valid = false;
    bool strain_face_hat_valid = false;
    LocalField u_hat;
    LocalField v_hat;
    LocalField w_center_hat;
    LocalField w_face_hat;
    LocalField strain_hat;
    LocalField strain_face_hat;

    void reset_metadata() {
        width = -1.0;
        face_begin = -1;
        face_count = -1;
        u_hat_valid = false;
        v_hat_valid = false;
        w_center_hat_valid = false;
        w_face_hat_valid = false;
        strain_hat_valid = false;
        strain_face_hat_valid = false;
    }

    void release_storage() {
        reset_metadata();
        release_local_field_storage(u_hat);
        release_local_field_storage(v_hat);
        release_local_field_storage(w_center_hat);
        release_local_field_storage(w_face_hat);
        release_local_field_storage(strain_hat);
        release_local_field_storage(strain_face_hat);
    }
};

struct LasdFilterCache {
    int step = -1;
    std::array<LasdFilterCacheSlot, 2> slots;

    void reset_for_step(int new_step) {
        if (step == new_step) {
            return;
        }
        step = new_step;
        for (LasdFilterCacheSlot& slot : slots) {
            slot.reset_metadata();
        }
    }

    LasdFilterCacheSlot& slot_for(double width) {
        constexpr double tolerance = 1.0e-12;
        for (LasdFilterCacheSlot& slot : slots) {
            if (slot.width > 0.0 && std::abs(slot.width - width) <= tolerance * std::max(1.0, std::abs(width))) {
                return slot;
            }
        }
        for (LasdFilterCacheSlot& slot : slots) {
            if (slot.width <= 0.0) {
                slot.width = width;
                return slot;
            }
        }
        slots[0].reset_metadata();
        slots[0].width = width;
        return slots[0];
    }

    void release_storage() {
        step = -1;
        for (LasdFilterCacheSlot& slot : slots) {
            slot.release_storage();
        }
    }
};

struct LasdGermanoScratch {
    LocalVecFields momentum_vel_hat_storage;
    LocalField momentum_center_work0;
    LocalField momentum_center_work1;
    LocalField momentum_center_hat0;
    LocalField momentum_center_hat1;
    LocalField momentum_center_hat2;
    LocalField momentum_strain_hat;
    LocalField momentum_dwdx_face;
    LocalField momentum_dwdy_face;
    LocalField momentum_dudz_face;
    LocalField momentum_dvdz_face;
    LocalField momentum_u_face;
    LocalField momentum_v_face;
    LocalField momentum_strain_face;
    LocalField momentum_u_face_hat;
    LocalField momentum_w_face_hat_storage;
    LocalField momentum_strain_face_hat_storage;
    LocalField momentum_face_work0;
    LocalField momentum_face_work1;
    LocalField momentum_face_work2;
    LocalField momentum_face_hat0;
    LocalField momentum_face_hat1;
    LocalField momentum_face_hat2;

    LocalField scalar_u_hat_storage;
    LocalField scalar_v_hat_storage;
    LocalField scalar_theta_hat;
    LocalField scalar_dx_hat;
    LocalField scalar_strain_hat;
    LocalField scalar_center_work0;
    LocalField scalar_center_work1;
    LocalField scalar_center_hat0;
    LocalField scalar_center_hat1;
    LocalField scalar_theta_w;
    LocalField scalar_strain_w;
    LocalField scalar_dtheta_dz_face;
    LocalField scalar_w_face_hat_storage;
    LocalField scalar_theta_w_hat;
    LocalField scalar_dz_hat;
    LocalField scalar_strain_w_hat_storage;
    LocalField scalar_face_work0;
    LocalField scalar_face_work1;
    LocalField scalar_face_hat0;
    LocalField scalar_face_hat1;

    LocalField lag_avg0;
    LocalField lag_avg1;
    LocalField lag_avg2;
    LocalField lag_avg3;
    LocalField lag_interp0;
    LocalField lag_interp1;
    LocalField lag_interp_scratch;

    std::vector<const LocalField*> filter_inputs;
    std::vector<LocalField*> filter_outputs;

    void reset_filter_batch() {
        filter_inputs.clear();
        filter_outputs.clear();
    }

    void add_filter(const LocalField& input, LocalField& output) {
        filter_inputs.push_back(&input);
        filter_outputs.push_back(&output);
    }

    void release_storage() {
        release_local_vec_field_storage(momentum_vel_hat_storage);
        release_local_field_storage(momentum_center_work0);
        release_local_field_storage(momentum_center_work1);
        release_local_field_storage(momentum_center_hat0);
        release_local_field_storage(momentum_center_hat1);
        release_local_field_storage(momentum_center_hat2);
        release_local_field_storage(momentum_strain_hat);
        release_local_field_storage(momentum_dwdx_face);
        release_local_field_storage(momentum_dwdy_face);
        release_local_field_storage(momentum_dudz_face);
        release_local_field_storage(momentum_dvdz_face);
        release_local_field_storage(momentum_u_face);
        release_local_field_storage(momentum_v_face);
        release_local_field_storage(momentum_strain_face);
        release_local_field_storage(momentum_u_face_hat);
        release_local_field_storage(momentum_w_face_hat_storage);
        release_local_field_storage(momentum_strain_face_hat_storage);
        release_local_field_storage(momentum_face_work0);
        release_local_field_storage(momentum_face_work1);
        release_local_field_storage(momentum_face_work2);
        release_local_field_storage(momentum_face_hat0);
        release_local_field_storage(momentum_face_hat1);
        release_local_field_storage(momentum_face_hat2);

        release_local_field_storage(scalar_u_hat_storage);
        release_local_field_storage(scalar_v_hat_storage);
        release_local_field_storage(scalar_theta_hat);
        release_local_field_storage(scalar_dx_hat);
        release_local_field_storage(scalar_strain_hat);
        release_local_field_storage(scalar_center_work0);
        release_local_field_storage(scalar_center_work1);
        release_local_field_storage(scalar_center_hat0);
        release_local_field_storage(scalar_center_hat1);
        release_local_field_storage(scalar_theta_w);
        release_local_field_storage(scalar_strain_w);
        release_local_field_storage(scalar_dtheta_dz_face);
        release_local_field_storage(scalar_w_face_hat_storage);
        release_local_field_storage(scalar_theta_w_hat);
        release_local_field_storage(scalar_dz_hat);
        release_local_field_storage(scalar_strain_w_hat_storage);
        release_local_field_storage(scalar_face_work0);
        release_local_field_storage(scalar_face_work1);
        release_local_field_storage(scalar_face_hat0);
        release_local_field_storage(scalar_face_hat1);

        release_local_field_storage(lag_avg0);
        release_local_field_storage(lag_avg1);
        release_local_field_storage(lag_avg2);
        release_local_field_storage(lag_avg3);
        release_local_field_storage(lag_interp0);
        release_local_field_storage(lag_interp1);
        release_local_field_storage(lag_interp_scratch);
        reset_filter_batch();
    }
};

struct PressureWorkspace {
    Field u_local;
    Field v_local;
    Field dwdz_local;
    Field p_local;
    Field dpdx_local;
    Field dpdy_local;

    SpectralField u_hat_local;
    SpectralField v_hat_local;
    SpectralField dwdz_hat_local;
    SpectralField div_hat_local;
    SpectralField p_hat_local;
    SpectralField dpdx_hat_local;
    SpectralField dpdy_hat_local;

    std::vector<Complex> transpose_send;
    std::vector<Complex> transpose_recv;
    std::vector<Complex> y_pencil;

    std::vector<Complex> thomas_lower;
    std::vector<Complex> thomas_cp;
    std::vector<Complex> thomas_inv_denom;
    std::vector<Complex> thomas_dp;
    int thomas_nx = -1;
    int thomas_ny = -1;
    int thomas_nz = -1;
    int thomas_rank = -1;
    int thomas_size = -1;
};

void clear_center_slab_field(LocalField& field, const Params& params, const Slab& slab) {
    if (field.owned_begin != slab.k_begin || field.owned_count != slab.k_count || field.total_planes != params.nz) {
        field.resize(slab.k_begin, slab.k_count, params.nz, params);
    }
    field.clear_owned();
}

void clear_face_slab_field(LocalField& field, const Params& params, const Slab& slab) {
    if (field.owned_begin != slab.face_begin || field.owned_count != slab.face_count || field.total_planes != params.nz + 1) {
        field.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
    }
    field.clear_owned();
}

struct MpiSlabWorkspace {
    LocalField rhs_u;
    LocalField rhs_v;
    LocalField rhs_w;
    LocalField rhs_theta;
    LocalField rhs_qt;
    LocalField rhs_sgs_tke;
    LocalField w_center;
    LocalField lap_u;
    LocalField lap_v;
    LocalField lap_w;
    LocalField dwdx_face;
    LocalField dwdy_face;
    LocalField dwdz_face;
    LocalField u_on_w;
    LocalField v_on_w;
    LocalField momentum_dudz_face;
    LocalField momentum_dvdz_face;
    LocalField momentum_flux_u_z;
    LocalField momentum_flux_v_z;
    LocalField momentum_div_u_z;
    LocalField momentum_div_v_z;
    LocalField momentum_flux_w_z;
    LocalField momentum_adv_w_z;
    LocalField momentum_div_w_z;
    LocalField strain;
    LocalField nu_t;
    LocalField amd_buoyancy_prime;
    LocalField amd_db_dx;
    LocalField amd_db_dy;
    LocalField amd_db_dz;
    LocalField amd_invariant_num;
    LocalField amd_invariant_den;
    LocalField tke_length;
    LocalField tke_kh;
    LocalField tke_diffusivity;
    LocalField tke_dtheta_v_dz;
    LocalField sgs_txx;
    LocalField sgs_txy;
    LocalField sgs_tyy;
    LocalField sgs_tzz;
    LocalField sgs_dwdx_face;
    LocalField sgs_dwdy_face;
    LocalField sgs_dudz_face;
    LocalField sgs_dvdz_face;
    LocalField sgs_nu_t_face;
    LocalField sgs_txz;
    LocalField sgs_tyz;
    LocalField sgs_div_u_xy;
    LocalField sgs_div_v_xy;
    LocalField sgs_dtxz_dz;
    LocalField sgs_dtyz_dz;
    LocalField sgs_div_w_xy;
    LocalField sgs_dtzz_dz;
    LocalField scalar_theta_flux_x;
    LocalField scalar_theta_flux_y;
    LocalField scalar_theta_flux_z;
    LocalField scalar_theta_on_w;
    LocalField scalar_div_adv_xy;
    LocalField scalar_div_adv_z;
    LocalField scalar_dtheta_dx;
    LocalField scalar_dtheta_dy;
    LocalField scalar_dtheta_dz_center;
    LocalField scalar_dtheta_dz_w;
    LocalField scalar_kappa_center;
    LocalField moisture_kappa_center;
    LocalField tke_scalar_kappa_center;
    LocalField scalar_kappa_w;
    LocalField scalar_qx;
    LocalField scalar_qy;
    LocalField scalar_qz;
    LocalField scalar_div_qxy;
    LocalField scalar_div_qz;
    LocalVelocityGradients grad;
    LasdFilterCache lasd_filter_cache;
    LasdGermanoScratch lasd_germano_scratch;
    PressureWorkspace pressure;
    HaloExchangeScratch halo_exchange_scratch;
    double projection_energy_change_sum = 0.0;
    double projection_max_energy_increase = 0.0;
    std::size_t projection_energy_samples = 0;
    double momentum_advection_power_sum = 0.0;
    double momentum_advection_u_sum = 0.0;
    double momentum_advection_v_sum = 0.0;
    std::size_t momentum_advection_power_samples = 0;

    explicit MpiSlabWorkspace(const Params& params, const Slab& slab) {
        resize(params, slab);
    }

    void resize(const Params& params, const Slab& slab) {
        rhs_u.resize(slab.k_begin, slab.k_count, params.nz, params);
        rhs_v.resize(slab.k_begin, slab.k_count, params.nz, params);
        rhs_w.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        rhs_theta.resize(slab.k_begin, slab.k_count, params.nz, params);
        rhs_qt.resize(slab.k_begin, slab.k_count, params.nz, params);
        rhs_sgs_tke.resize(slab.k_begin, slab.k_count, params.nz, params);
        w_center.resize(slab.k_begin, slab.k_count, params.nz, params);
        lap_u.resize(slab.k_begin, slab.k_count, params.nz, params);
        lap_v.resize(slab.k_begin, slab.k_count, params.nz, params);
        lap_w.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        dwdx_face.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        dwdy_face.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        dwdz_face.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        u_on_w.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        v_on_w.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        momentum_dudz_face.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        momentum_dvdz_face.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        momentum_flux_u_z.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        momentum_flux_v_z.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        momentum_div_u_z.resize(slab.k_begin, slab.k_count, params.nz, params);
        momentum_div_v_z.resize(slab.k_begin, slab.k_count, params.nz, params);
        momentum_flux_w_z.resize(slab.k_begin, slab.k_count, params.nz, params);
        momentum_adv_w_z.resize(slab.k_begin, slab.k_count, params.nz, params);
        momentum_div_w_z.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        strain.resize(slab.k_begin, slab.k_count, params.nz, params);
        nu_t.resize(slab.k_begin, slab.k_count, params.nz, params);
        amd_buoyancy_prime.resize(slab.k_begin, slab.k_count, params.nz, params);
        amd_db_dx.resize(slab.k_begin, slab.k_count, params.nz, params);
        amd_db_dy.resize(slab.k_begin, slab.k_count, params.nz, params);
        amd_db_dz.resize(slab.k_begin, slab.k_count, params.nz, params);
        amd_invariant_num.resize(slab.k_begin, slab.k_count, params.nz, params);
        amd_invariant_den.resize(slab.k_begin, slab.k_count, params.nz, params);
        tke_length.resize(slab.k_begin, slab.k_count, params.nz, params);
        tke_kh.resize(slab.k_begin, slab.k_count, params.nz, params);
        tke_diffusivity.resize(slab.k_begin, slab.k_count, params.nz, params);
        tke_dtheta_v_dz.resize(slab.k_begin, slab.k_count, params.nz, params);
        sgs_txx.resize(slab.k_begin, slab.k_count, params.nz, params);
        sgs_txy.resize(slab.k_begin, slab.k_count, params.nz, params);
        sgs_tyy.resize(slab.k_begin, slab.k_count, params.nz, params);
        sgs_tzz.resize(slab.k_begin, slab.k_count, params.nz, params);
        sgs_dwdx_face.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        sgs_dwdy_face.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        sgs_dudz_face.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        sgs_dvdz_face.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        sgs_nu_t_face.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        sgs_txz.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        sgs_tyz.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        sgs_div_u_xy.resize(slab.k_begin, slab.k_count, params.nz, params);
        sgs_div_v_xy.resize(slab.k_begin, slab.k_count, params.nz, params);
        sgs_dtxz_dz.resize(slab.k_begin, slab.k_count, params.nz, params);
        sgs_dtyz_dz.resize(slab.k_begin, slab.k_count, params.nz, params);
        sgs_div_w_xy.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        sgs_dtzz_dz.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        scalar_theta_flux_x.resize(slab.k_begin, slab.k_count, params.nz, params);
        scalar_theta_flux_y.resize(slab.k_begin, slab.k_count, params.nz, params);
        scalar_theta_flux_z.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        scalar_theta_on_w.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        scalar_div_adv_xy.resize(slab.k_begin, slab.k_count, params.nz, params);
        scalar_div_adv_z.resize(slab.k_begin, slab.k_count, params.nz, params);
        scalar_dtheta_dx.resize(slab.k_begin, slab.k_count, params.nz, params);
        scalar_dtheta_dy.resize(slab.k_begin, slab.k_count, params.nz, params);
        scalar_dtheta_dz_center.resize(slab.k_begin, slab.k_count, params.nz, params);
        scalar_dtheta_dz_w.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        scalar_kappa_center.resize(slab.k_begin, slab.k_count, params.nz, params);
        moisture_kappa_center.resize(slab.k_begin, slab.k_count, params.nz, params);
        tke_scalar_kappa_center.resize(slab.k_begin, slab.k_count, params.nz, params);
        scalar_kappa_w.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        scalar_qx.resize(slab.k_begin, slab.k_count, params.nz, params);
        scalar_qy.resize(slab.k_begin, slab.k_count, params.nz, params);
        scalar_qz.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        scalar_div_qxy.resize(slab.k_begin, slab.k_count, params.nz, params);
        scalar_div_qz.resize(slab.k_begin, slab.k_count, params.nz, params);
        grad.resize(params, slab);

        const std::size_t local_real =
            static_cast<std::size_t>(params.nx) * static_cast<std::size_t>(params.ny) * static_cast<std::size_t>(slab.k_count);
        const std::size_t local_spectral =
            static_cast<std::size_t>(params.nkx()) * static_cast<std::size_t>(params.ny) * static_cast<std::size_t>(slab.k_count);
        pressure.u_local.reserve(local_real);
        pressure.v_local.reserve(local_real);
        pressure.dwdz_local.reserve(local_real);
        pressure.p_local.reserve(local_real);
        pressure.dpdx_local.reserve(local_real);
        pressure.dpdy_local.reserve(local_real);
        pressure.u_hat_local.reserve(local_spectral);
        pressure.v_hat_local.reserve(local_spectral);
        pressure.dwdz_hat_local.reserve(local_spectral);
        pressure.div_hat_local.reserve(local_spectral);
        pressure.p_hat_local.reserve(local_spectral);
        pressure.dpdx_hat_local.reserve(local_spectral);
        pressure.dpdy_hat_local.reserve(local_spectral);
    }
};

double local_kinetic_energy_sum(const MpiLocalFlowState& state, const Params& params, const Slab& slab) {
    double energy = 0.0;
    for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                energy += 0.5 * (state.u[n] * state.u[n] + state.v[n] * state.v[n]);
            }
        }
    }
    for (int k = slab.face_begin; k < slab.face_begin + slab.face_count; ++k) {
        if (k <= 0 || k >= params.nz) continue;
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const double w = state.w[z_face_idx(params, i, j, k)];
                energy += 0.5 * w * w;
            }
        }
    }
    return energy;
}

double safe_divide(double num, double den) {
    return std::abs(den) > 1.0e-30 ? num / den : 0.0;
}

bool uses_moeng_tke(const Params& params) {
    return params.sgs_model == "tke" || params.sgs_model == "moeng_tke";
}

enum class MpiTimerId : std::size_t {
    rhs_derivatives = 0,
    rhs_momentum,
    rhs_sgs_update,
    rhs_sgs_forcing,
    sgs_center_stress_build,
    sgs_face_derivatives,
    sgs_face_stress_build,
    sgs_stress_halo,
    sgs_center_divergence,
    sgs_face_divergence,
    rhs_wall_stress,
    rhs_scalar,
    scalar_advective_flux,
    scalar_advective_divergence,
    scalar_gradients,
    scalar_lasd_update,
    scalar_diffusivity_halo,
    scalar_diffusive_flux,
    scalar_diffusive_divergence,
    rhs_buoyancy,
    rhs_total,
    advance,
    dealias,
    state_halo_after,
    projection,
    step_total,
    diagnostics,
    count
};

constexpr std::size_t mpi_timer_count = static_cast<std::size_t>(MpiTimerId::count);

struct MpiTimingStats {
    std::array<double, mpi_timer_count> seconds{};
    int measured_steps = 0;
    int diagnostic_calls = 0;
};

class MpiTimerScope {
public:
    MpiTimerScope(MpiTimingStats* stats, MpiTimerId id)
        : stats_(stats), id_(id) {
        if (stats_ != nullptr) {
            start_ = std::chrono::steady_clock::now();
        }
    }

    MpiTimerScope(const MpiTimerScope&) = delete;
    MpiTimerScope& operator=(const MpiTimerScope&) = delete;

    ~MpiTimerScope() {
        if (stats_ == nullptr) {
            return;
        }
        const auto stop = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed = stop - start_;
        stats_->seconds[static_cast<std::size_t>(id_)] += elapsed.count();
    }

private:
    MpiTimingStats* stats_ = nullptr;
    MpiTimerId id_;
    std::chrono::steady_clock::time_point start_{};
};

double sym_component_weight(int component) {
    return (component == 1 || component == 2 || component == 4) ? 2.0 : 1.0;
}

Slab make_slab(const Params& params, MPI_Comm comm) {
    Slab slab;
    MPI_Comm_rank(comm, &slab.rank);
    MPI_Comm_size(comm, &slab.size);
    if (slab.size > params.nz) {
        throw std::runtime_error("MPI z-slab path requires number of ranks <= nz");
    }
    if (params.ny % slab.size != 0) {
        throw std::runtime_error("Fortran-style MPI pressure path requires ny divisible by number of ranks");
    }
    const int base = params.nz / slab.size;
    const int remainder = params.nz % slab.size;
    const int plane_size = params.nx * params.ny;
    slab.center_counts.assign(static_cast<std::size_t>(slab.size), 0);
    slab.center_displs.assign(static_cast<std::size_t>(slab.size), 0);
    slab.face_counts.assign(static_cast<std::size_t>(slab.size), 0);
    slab.face_displs.assign(static_cast<std::size_t>(slab.size), 0);

    int k_offset = 0;
    for (int r = 0; r < slab.size; ++r) {
        const int k_count = base + (r < remainder ? 1 : 0);
        const int face_count = k_count + (r == slab.size - 1 ? 1 : 0);
        slab.center_counts[static_cast<std::size_t>(r)] = k_count * plane_size;
        slab.center_displs[static_cast<std::size_t>(r)] = k_offset * plane_size;
        slab.face_counts[static_cast<std::size_t>(r)] = face_count * plane_size;
        slab.face_displs[static_cast<std::size_t>(r)] = k_offset * plane_size;
        if (r == slab.rank) {
            slab.k_begin = k_offset;
            slab.k_count = k_count;
            slab.face_begin = k_offset;
            slab.face_count = face_count;
        }
        k_offset += k_count;
    }
    return slab;
}

bool owns_center_plane(const Slab& slab, int k) {
    return k >= slab.k_begin && k < slab.k_begin + slab.k_count;
}

bool owns_face_plane(const Slab& slab, int k) {
    return k >= slab.face_begin && k < slab.face_begin + slab.face_count;
}

void enforce_walls_slab(LocalField& w, const Params& params, const Slab& slab) {
    if (owns_face_plane(slab, 0)) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                w[z_face_idx(params, i, j, 0)] = 0.0;
            }
        }
    }
    if (owns_face_plane(slab, params.nz)) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                w[z_face_idx(params, i, j, params.nz)] = 0.0;
            }
        }
    }
}

void update_moist_thermodynamics_slab(
    MpiLocalFlowState& state,
    const Params& params,
    const Slab& slab) {
    if (!params.moisture_enabled) {
        return;
    }
    for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
        const double pressure = state.base_pressure[static_cast<std::size_t>(k)];
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                const MoistThermodynamicState moist = saturation_adjustment(
                    state.theta_l[n], state.qt[n], pressure);
                state.theta[n] = moist.potential_temperature;
                state.qv[n] = moist.water_vapor_mixing_ratio;
                state.ql[n] = moist.liquid_water_mixing_ratio;
            }
        }
    }
}

void initialize_moist_thermodynamics_slab(
    MpiLocalFlowState& state,
    const Params& params,
    const Slab& slab) {
    if (!params.moisture_enabled) {
        return;
    }
    for (int k = 0; k < params.nz; ++k) {
        const double z = (static_cast<double>(k) + 0.5) * params.dz();
        state.base_pressure[static_cast<std::size_t>(k)] = hydrostatic_base_pressure(
            z, params.surface_pressure, params.theta0, params.g);
    }
    for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                state.theta_l[n] = state.theta[n];
                state.qt[n] = state.qv[n];
            }
        }
    }
    update_moist_thermodynamics_slab(state, params, slab);
}

void initialize_local(MpiLocalFlowState& state, const Params& params, const Slab& slab) {
    if (params.initial_condition == "taylor_green") {
        for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
            const double z = (static_cast<double>(k) + 0.5) * params.dz();
            for (int j = 0; j < params.ny; ++j) {
                const double y = static_cast<double>(j) * params.dy();
                for (int i = 0; i < params.nx; ++i) {
                    const double x = static_cast<double>(i) * params.dx();
                    state.u[idx(params, i, j, k)] = std::sin(x) * std::cos(y);
                    state.v[idx(params, i, j, k)] = -std::cos(x) * std::sin(y);
                    state.theta[idx(params, i, j, k)] = params.theta0 + params.theta_initial_gradient * z;
                    state.qv[idx(params, i, j, k)] = params.qv0 + params.qv_initial_gradient * z;
                }
            }
        }
    } else if (params.initial_condition == "largeeddy1993") {
        if (params.surface_theta_flux <= 0.0) {
            throw std::runtime_error("largeeddy1993 initial condition requires positive surface_theta_flux");
        }
        const double zi1 = params.largeeddy_initial_zi1_fraction * params.z_i;
        const double wstar = std::cbrt((params.g / params.theta0) * params.surface_theta_flux * params.z_i);
        const double theta_star = params.surface_theta_flux / wstar;
        std::mt19937 rng(static_cast<std::mt19937::result_type>(params.random_seed));
        std::uniform_real_distribution<double> uniform(-0.5, 0.5);

        for (int k = 0; k < params.nz; ++k) {
            const double z = (static_cast<double>(k) + 0.5) * params.dz();
            const double lower_weight = std::max(1.0 - z / zi1, 0.0);
            const bool in_mixed_layer = z < zi1;
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const double perturb = 0.1 * uniform(rng) * lower_weight;
                    if (owns_center_plane(slab, k)) {
                        state.theta[idx(params, i, j, k)] = in_mixed_layer
                            ? params.theta0 + perturb * theta_star
                            : params.theta0 + (z - zi1) * params.theta_initial_gradient;
                        state.qv[idx(params, i, j, k)] = params.qv0 + params.qv_initial_gradient * z;
                    }
                }
            }
        }
        for (int k = 1; k < params.nz; ++k) {
            const double z = static_cast<double>(k) * params.dz();
            const double lower_weight = std::max(1.0 - z / zi1, 0.0);
            const bool in_mixed_layer = z < zi1;
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const double perturb = 0.1 * uniform(rng) * lower_weight;
                    if (owns_face_plane(slab, k)) {
                        state.w[z_face_idx(params, i, j, k)] = in_mixed_layer ? perturb * wstar : 0.0;
                    }
                }
            }
        }
    } else if (params.initial_condition == "neutral_ekman") {
        const double perturbation_height = params.initial_perturbation_height > 0.0
            ? params.initial_perturbation_height
            : 0.25 * params.lz;
        std::mt19937 rng(static_cast<std::mt19937::result_type>(params.random_seed));
        std::uniform_real_distribution<double> uniform(-0.5, 0.5);

        for (int k = 0; k < params.nz; ++k) {
            const double z = (static_cast<double>(k) + 0.5) * params.dz();
            const double lower_weight = perturbation_height > 0.0
                ? std::max(1.0 - z / perturbation_height, 0.0)
                : 0.0;
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const double u_perturb = params.initial_velocity_perturbation * lower_weight * uniform(rng);
                    const double v_perturb = params.initial_velocity_perturbation * lower_weight * uniform(rng);
                    if (owns_center_plane(slab, k)) {
                        const std::size_t n = idx(params, i, j, k);
                        state.u[n] = params.geostrophic_u + u_perturb;
                        state.v[n] = params.geostrophic_v + v_perturb;
                        state.theta[n] = params.theta0 + params.theta_initial_gradient * z;
                        state.qv[n] = params.qv0 + params.qv_initial_gradient * z;
                    }
                }
            }
        }
        for (int k = 1; k < params.nz; ++k) {
            const double z = static_cast<double>(k) * params.dz();
            const double lower_weight = perturbation_height > 0.0
                ? std::max(1.0 - z / perturbation_height, 0.0)
                : 0.0;
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const double perturb = params.initial_velocity_perturbation * lower_weight * uniform(rng);
                    if (owns_face_plane(slab, k)) {
                        state.w[z_face_idx(params, i, j, k)] = perturb;
                    }
                }
            }
        }
    } else if (params.initial_condition == "bomex") {
        std::mt19937 rng(static_cast<std::mt19937::result_type>(params.random_seed));
        std::uniform_real_distribution<double> uniform(-1.0, 1.0);
        for (int k = 0; k < params.nz; ++k) {
            const double z = (static_cast<double>(k) + 0.5) * params.dz();
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const bool perturb = params.initial_perturbation_height > 0.0
                        ? z < params.initial_perturbation_height
                        : k < 4;
                    const double theta_perturbation = perturb
                        ? params.bomex_theta_perturbation * uniform(rng)
                        : 0.0;
                    const double qt_perturbation = perturb
                        ? params.bomex_qt_perturbation * uniform(rng)
                        : 0.0;
                    if (owns_center_plane(slab, k)) {
                        const std::size_t n = idx(params, i, j, k);
                        state.u[n] = bomex_initial_u(z);
                        state.v[n] = 0.0;
                        state.theta[n] = bomex_initial_theta_l(z) + theta_perturbation;
                        state.qv[n] = std::max(0.0, bomex_initial_qt(z) + qt_perturbation);
                        if (uses_moeng_tke(params)) {
                            // BOMEX intercomparison prescription for models with
                            // prognostic subgrid TKE (Siebesma et al. 2003, App. B).
                            state.sgs_tke[n] = std::max(1.0 - z / 3000.0, params.tke_floor);
                        }
                    }
                }
            }
        }
    } else {
        throw std::runtime_error("unsupported initial_condition: " + params.initial_condition);
    }
    initialize_moist_thermodynamics_slab(state, params, slab);
    enforce_walls_slab(state.w, params, slab);
}

void advance_center_slab(
    LocalField& q,
    const LocalField& rhs,
    LocalField& rhs_prev,
    LocalField& rhs_prev2,
    int step_count,
    double dt,
    const Params& params,
    const Slab& slab) {
    for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                double tendency = rhs[n];
                if (params.time_scheme == "ab3" && step_count >= 2) {
                    tendency = (23.0 * rhs[n] - 16.0 * rhs_prev[n] + 5.0 * rhs_prev2[n]) / 12.0;
                } else if ((params.time_scheme == "ab2" && step_count >= 1)
                    || (params.time_scheme == "ab3" && step_count == 1)) {
                    tendency = 1.5 * rhs[n] - 0.5 * rhs_prev[n];
                }
                q[n] += dt * tendency;
                rhs_prev2[n] = rhs_prev[n];
                rhs_prev[n] = rhs[n];
            }
        }
    }
}

void advance_w_slab(
    LocalField& w,
    const LocalField& rhs,
    LocalField& rhs_prev,
    LocalField& rhs_prev2,
    int step_count,
    double dt,
    const Params& params,
    const Slab& slab) {
    for (int k = slab.face_begin; k < slab.face_begin + slab.face_count; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = z_face_idx(params, i, j, k);
                double tendency = rhs[n];
                if (params.time_scheme == "ab3" && step_count >= 2) {
                    tendency = (23.0 * rhs[n] - 16.0 * rhs_prev[n] + 5.0 * rhs_prev2[n]) / 12.0;
                } else if ((params.time_scheme == "ab2" && step_count >= 1)
                    || (params.time_scheme == "ab3" && step_count == 1)) {
                    tendency = 1.5 * rhs[n] - 0.5 * rhs_prev[n];
                }
                w[n] += dt * tendency;
                rhs_prev2[n] = rhs_prev[n];
                rhs_prev[n] = rhs[n];
            }
        }
    }
}

std::size_t local_sidx(const Params& params, int ih, int j, int k_local) {
    return (static_cast<std::size_t>(k_local) * static_cast<std::size_t>(params.ny) + static_cast<std::size_t>(j))
        * static_cast<std::size_t>(params.nkx())
        + static_cast<std::size_t>(ih);
}

std::size_t pencil_sidx(const Params& params, int ih, int j_local, int k_global, int nj) {
    return (static_cast<std::size_t>(k_global) * static_cast<std::size_t>(nj) + static_cast<std::size_t>(j_local))
        * static_cast<std::size_t>(params.nkx())
        + static_cast<std::size_t>(ih);
}

Field planes_to_local(const LocalField& q, const Params& params, int plane_begin, int plane_count) {
    const int plane_size = params.nx * params.ny;
    Field local(static_cast<std::size_t>(plane_count * plane_size), 0.0);
    for (int k_local = 0; k_local < plane_count; ++k_local) {
        const int k_global = plane_begin + k_local;
        const auto source = q.values.begin() + static_cast<std::ptrdiff_t>((k_global - q.plane_begin) * plane_size);
        std::copy(source, source + static_cast<std::ptrdiff_t>(plane_size),
            local.begin() + static_cast<std::ptrdiff_t>(k_local * plane_size));
    }
    return local;
}

void scatter_local_planes(
    const Field& local,
    LocalField& global,
    const Params& params,
    int plane_begin,
    int plane_count) {
    const int plane_size = params.nx * params.ny;
    for (int k_local = 0; k_local < plane_count; ++k_local) {
        const int k_global = plane_begin + k_local;
        const auto source = local.begin() + static_cast<std::ptrdiff_t>(k_local * plane_size);
        std::copy(source, source + static_cast<std::ptrdiff_t>(plane_size),
            global.values.begin() + static_cast<std::ptrdiff_t>((k_global - global.plane_begin) * plane_size));
    }
}

void derivative_x_center_slab(const LocalField& q, LocalField& out, const Params& params, const Slab& slab, FftwXY& fft) {
    const Field local = planes_to_local(q, params, slab.k_begin, slab.k_count);
    Field local_out;
    fft.derivative_x_planes(local, slab.k_count, local_out, params);
    scatter_local_planes(local_out, out, params, slab.k_begin, slab.k_count);
}

void derivative_y_center_slab(const LocalField& q, LocalField& out, const Params& params, const Slab& slab, FftwXY& fft) {
    const Field local = planes_to_local(q, params, slab.k_begin, slab.k_count);
    Field local_out;
    fft.derivative_y_planes(local, slab.k_count, local_out, params);
    scatter_local_planes(local_out, out, params, slab.k_begin, slab.k_count);
}

void derivative_x_face_slab(const LocalField& q, LocalField& out, const Params& params, const Slab& slab, FftwXY& fft) {
    const Field local = planes_to_local(q, params, slab.face_begin, slab.face_count);
    Field local_out;
    fft.derivative_x_planes(local, slab.face_count, local_out, params);
    scatter_local_planes(local_out, out, params, slab.face_begin, slab.face_count);
}

void derivative_y_face_slab(const LocalField& q, LocalField& out, const Params& params, const Slab& slab, FftwXY& fft) {
    const Field local = planes_to_local(q, params, slab.face_begin, slab.face_count);
    Field local_out;
    fft.derivative_y_planes(local, slab.face_count, local_out, params);
    scatter_local_planes(local_out, out, params, slab.face_begin, slab.face_count);
}

void add_d2dz2_center_slab(const LocalField& q, LocalField& out, const Params& params, const Slab& slab);
void add_d2dz2_w_face_slab(const LocalField& w, LocalField& out, const Params& params, const Slab& slab);

void horizontal_divergence_center_slab(
    const LocalField& flux_x,
    const LocalField& flux_y,
    LocalField& out,
    const Params& params,
    const Slab& slab,
    FftwXY& fft) {
    const int local_begin = slab.k_begin - flux_x.plane_begin;
    fft.horizontal_divergence_plane_range(flux_x.values, flux_y.values, local_begin, slab.k_count, out.values, params);
}

void horizontal_divergence_face_slab(
    const LocalField& flux_x,
    const LocalField& flux_y,
    LocalField& out,
    const Params& params,
    const Slab& slab,
    FftwXY& fft) {
    const int local_begin = slab.face_begin - flux_x.plane_begin;
    fft.horizontal_divergence_plane_range(flux_x.values, flux_y.values, local_begin, slab.face_count, out.values, params);
}

void horizontal_derivatives_center_slab(
    const LocalField& q,
    LocalField& dx,
    LocalField& dy,
    const Params& params,
    const Slab& slab,
    FftwXY& fft) {
    const int local_begin = slab.k_begin - q.plane_begin;
    fft.horizontal_derivatives_plane_range(q.values, local_begin, slab.k_count, dx.values, dy.values, params);
}

void horizontal_derivatives_laplacian_center_slab(
    const LocalField& q,
    LocalField& dx,
    LocalField& dy,
    LocalField& lap,
    const Params& params,
    const Slab& slab,
    FftwXY& fft) {
    const Field local = planes_to_local(q, params, slab.k_begin, slab.k_count);
    Field local_dx;
    Field local_dy;
    Field local_lap;
    fft.horizontal_derivatives_laplacian_plane_range(local, 0, slab.k_count, local_dx, local_dy, local_lap, params);
    scatter_local_planes(local_dx, dx, params, slab.k_begin, slab.k_count);
    scatter_local_planes(local_dy, dy, params, slab.k_begin, slab.k_count);
    scatter_local_planes(local_lap, lap, params, slab.k_begin, slab.k_count);
    add_d2dz2_center_slab(q, lap, params, slab);
}

void horizontal_derivatives_laplacian_face_slab(
    const LocalField& q,
    LocalField& dx,
    LocalField& dy,
    LocalField& lap,
    const Params& params,
    const Slab& slab,
    FftwXY& fft) {
    const Field local = planes_to_local(q, params, slab.face_begin, slab.face_count);
    Field local_dx;
    Field local_dy;
    Field local_lap;
    fft.horizontal_derivatives_laplacian_plane_range(local, 0, slab.face_count, local_dx, local_dy, local_lap, params);
    scatter_local_planes(local_dx, dx, params, slab.face_begin, slab.face_count);
    scatter_local_planes(local_dy, dy, params, slab.face_begin, slab.face_count);
    scatter_local_planes(local_lap, lap, params, slab.face_begin, slab.face_count);
    add_d2dz2_w_face_slab(q, lap, params, slab);
}

void ddz_center_slab(const LocalField& q, LocalField& out, const Params& params, const Slab& slab) {
    clear_center_slab_field(out, params, slab);
    const double dz = params.dz();
    for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                double value = 0.0;
                if (k == 0) {
                    value = (q[idx(params, i, j, 1)] - q[idx(params, i, j, 0)]) / dz;
                } else if (k == params.nz - 1) {
                    value = (q[idx(params, i, j, params.nz - 1)] - q[idx(params, i, j, params.nz - 2)]) / dz;
                } else {
                    value = (q[idx(params, i, j, k + 1)] - q[idx(params, i, j, k - 1)]) / (2.0 * dz);
                }
                out[idx(params, i, j, k)] = value;
            }
        }
    }
}

void apply_amd_wall_model_gradients_slab(
    const MpiLocalFlowState& state,
    LocalVelocityGradients& grad,
    LocalField* dudz_face,
    LocalField* dvdz_face,
    const Params& params,
    const Slab& slab) {
    if (!params.amd_wall_model_gradients || slab.k_begin != 0 || params.nz < 2) {
        return;
    }
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
            if (dudz_face != nullptr && dvdz_face != nullptr) {
                const std::size_t face = z_face_idx(params, i, j, 0);
                (*dudz_face)[face] = grad.dudz[lower];
                (*dvdz_face)[face] = grad.dvdz[lower];
            }
        }
    }
}

void add_d2dz2_center_slab(const LocalField& q, LocalField& out, const Params& params, const Slab& slab) {
    const double inv_dz2 = 1.0 / (params.dz() * params.dz());
    for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                double value = 0.0;
                if (k == 0) {
                    value = (q[idx(params, i, j, 1)] - q[idx(params, i, j, 0)]) * inv_dz2;
                } else if (k == params.nz - 1) {
                    value = (q[idx(params, i, j, params.nz - 2)] - q[idx(params, i, j, params.nz - 1)]) * inv_dz2;
                } else {
                    value = (q[idx(params, i, j, k - 1)] - 2.0 * q[idx(params, i, j, k)] + q[idx(params, i, j, k + 1)])
                        * inv_dz2;
                }
                out[idx(params, i, j, k)] += value;
            }
        }
    }
}

void w_to_center_slab(const LocalField& w, LocalField& out, const Params& params, const Slab& slab) {
    clear_center_slab_field(out, params, slab);
    for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                out[idx(params, i, j, k)] = 0.5 * (
                    w[z_face_idx(params, i, j, k)] + w[z_face_idx(params, i, j, k + 1)]);
            }
        }
    }
}

void center_to_w_face_slab(const LocalField& q, LocalField& out, const Params& params, const Slab& slab) {
    clear_face_slab_field(out, params, slab);
    for (int k = slab.face_begin; k < slab.face_begin + slab.face_count; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                double value = 0.0;
                if (k == 0) {
                    value = q[idx(params, i, j, 0)];
                } else if (k == params.nz) {
                    value = q[idx(params, i, j, params.nz - 1)];
                } else {
                    value = 0.5 * (q[idx(params, i, j, k - 1)] + q[idx(params, i, j, k)]);
                }
                out[z_face_idx(params, i, j, k)] = value;
            }
        }
    }
}

void ddz_w_to_center_slab(const LocalField& w, LocalField& out, const Params& params, const Slab& slab) {
    clear_center_slab_field(out, params, slab);
    const double inv_dz = 1.0 / params.dz();
    for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                out[idx(params, i, j, k)] =
                    (w[z_face_idx(params, i, j, k + 1)] - w[z_face_idx(params, i, j, k)]) * inv_dz;
            }
        }
    }
}

void ddz_center_to_w_face_slab(const LocalField& q, LocalField& out, const Params& params, const Slab& slab) {
    clear_face_slab_field(out, params, slab);
    const double inv_dz = 1.0 / params.dz();
    for (int k = slab.face_begin; k < slab.face_begin + slab.face_count; ++k) {
        if (k <= 0 || k >= params.nz) {
            continue;
        }
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                out[z_face_idx(params, i, j, k)] =
                    (q[idx(params, i, j, k)] - q[idx(params, i, j, k - 1)]) * inv_dz;
            }
        }
    }
}

void ddz_w_face_slab(const LocalField& w, LocalField& out, const Params& params, const Slab& slab) {
    clear_face_slab_field(out, params, slab);
    const double inv_dz = 1.0 / params.dz();
    for (int k = slab.face_begin; k < slab.face_begin + slab.face_count; ++k) {
        if (k <= 0 || k >= params.nz) {
            continue;
        }
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                out[z_face_idx(params, i, j, k)] =
                    (w[z_face_idx(params, i, j, k + 1)] - w[z_face_idx(params, i, j, k - 1)]) * (0.5 * inv_dz);
            }
        }
    }
}

void add_d2dz2_w_face_slab(const LocalField& w, LocalField& out, const Params& params, const Slab& slab) {
    const double inv_dz2 = 1.0 / (params.dz() * params.dz());
    for (int k = slab.face_begin; k < slab.face_begin + slab.face_count; ++k) {
        if (k <= 0 || k >= params.nz) {
            continue;
        }
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                out[z_face_idx(params, i, j, k)] =
                    out[z_face_idx(params, i, j, k)]
                    + (w[z_face_idx(params, i, j, k - 1)] - 2.0 * w[z_face_idx(params, i, j, k)]
                     + w[z_face_idx(params, i, j, k + 1)])
                    * inv_dz2;
            }
        }
    }
}

void horizontal_spectral_filter_center_slab_many(
    const std::vector<const LocalField*>& inputs,
    const std::vector<LocalField*>& outputs,
    const Params& params,
    const Slab& slab,
    FftwXY& fft,
    double filter_width) {
    if (inputs.size() != outputs.size()) {
        throw std::runtime_error("center filter batch requires matching input/output counts");
    }
    for (std::size_t n = 0; n < inputs.size(); ++n) {
        if (inputs[n] == nullptr || outputs[n] == nullptr) {
            throw std::runtime_error("center filter batch requires non-null fields");
        }
        match_local_layout(*outputs[n], *inputs[n]);
        fft.filter_plane_range_fortran_sharp(
            inputs[n]->values,
            slab.k_begin - inputs[n]->plane_begin,
            slab.k_count,
            outputs[n]->values,
            params,
            filter_width);
    }
}

void horizontal_spectral_filter_face_range_many(
    const std::vector<const LocalField*>& inputs,
    int face_begin,
    int face_count,
    const std::vector<LocalField*>& outputs,
    const Params& params,
    FftwXY& fft,
    double filter_width) {
    if (inputs.size() != outputs.size()) {
        throw std::runtime_error("face filter batch requires matching input/output counts");
    }
    for (std::size_t n = 0; n < inputs.size(); ++n) {
        if (inputs[n] == nullptr || outputs[n] == nullptr) {
            throw std::runtime_error("face filter batch requires non-null fields");
        }
        match_local_layout(*outputs[n], *inputs[n]);
        fft.filter_plane_range_fortran_sharp(
            inputs[n]->values,
            face_begin - inputs[n]->plane_begin,
            face_count,
            outputs[n]->values,
            params,
            filter_width);
    }
}

void derivative_x_face_range(
    const LocalField& q,
    int face_begin,
    int face_count,
    LocalField& out,
    const Params& params,
    FftwXY& fft) {
    Field local = planes_to_local(q, params, face_begin, face_count);
    Field local_out;
    fft.derivative_x_planes(local, face_count, local_out, params);
    out.resize(face_begin, face_count, params.nz + 1, params, false);
    scatter_local_planes(local_out, out, params, face_begin, face_count);
}

void derivative_y_face_range(
    const LocalField& q,
    int face_begin,
    int face_count,
    LocalField& out,
    const Params& params,
    FftwXY& fft) {
    Field local = planes_to_local(q, params, face_begin, face_count);
    Field local_out;
    fft.derivative_y_planes(local, face_count, local_out, params);
    out.resize(face_begin, face_count, params.nz + 1, params, false);
    scatter_local_planes(local_out, out, params, face_begin, face_count);
}

int needed_face_begin(const Slab& slab) {
    return slab.k_begin;
}

int needed_face_count(const Params& params, const Slab& slab) {
    const int begin = needed_face_begin(slab);
    const int end = std::min(params.nz, slab.k_begin + slab.k_count);
    return end - begin + 1;
}

void center_to_w_face_range(
    const LocalField& q,
    int face_begin,
    int face_count,
    LocalField& out,
    const Params& params) {
    out.resize(face_begin, face_count, params.nz + 1, params, false);
    for (int k = face_begin; k < face_begin + face_count; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                double value = 0.0;
                if (k == 0) {
                    value = q[idx(params, i, j, 0)];
                } else if (k == params.nz) {
                    value = q[idx(params, i, j, params.nz - 1)];
                } else {
                    value = 0.5 * (q[idx(params, i, j, k - 1)] + q[idx(params, i, j, k)]);
                }
                out[z_face_idx(params, i, j, k)] = value;
            }
        }
    }
}

void ddz_center_to_w_face_range(
    const LocalField& q,
    int face_begin,
    int face_count,
    LocalField& out,
    const Params& params) {
    out.resize(face_begin, face_count, params.nz + 1, params, false);
    const double inv_dz = 1.0 / params.dz();
    for (int k = face_begin; k < face_begin + face_count; ++k) {
        if (k <= 0 || k >= params.nz) {
            continue;
        }
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                out[z_face_idx(params, i, j, k)] =
                    (q[idx(params, i, j, k)] - q[idx(params, i, j, k - 1)]) * inv_dz;
            }
        }
    }
}

Field local_divergence_slab(const MpiLocalFlowState& state, const Params& params, const Slab& slab, FftwXY& fft) {
    const Field u_local = planes_to_local(state.u, params, slab.k_begin, slab.k_count);
    const Field v_local = planes_to_local(state.v, params, slab.k_begin, slab.k_count);
    Field dudx;
    Field dvdy;
    fft.derivative_x_planes(u_local, slab.k_count, dudx, params);
    fft.derivative_y_planes(v_local, slab.k_count, dvdy, params);

    Field div(static_cast<std::size_t>(slab.k_count) * static_cast<std::size_t>(params.nx) * static_cast<std::size_t>(params.ny), 0.0);
    const double inv_dz = 1.0 / params.dz();
    for (int k_local = 0; k_local < slab.k_count; ++k_local) {
        const int k_global = slab.k_begin + k_local;
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n_local = idx(params, i, j, k_local);
                div[n_local] = dudx[n_local] + dvdy[n_local]
                    + (state.w[z_face_idx(params, i, j, k_global + 1)] - state.w[z_face_idx(params, i, j, k_global)]) * inv_dz;
            }
        }
    }
    return div;
}

void build_pressure_divergence_hat_slab(
    const MpiLocalFlowState& state,
    const Params& params,
    const Slab& slab,
    FftwXY& fft,
    PressureWorkspace& workspace) {
    const std::size_t local_real_size =
        static_cast<std::size_t>(slab.k_count) * static_cast<std::size_t>(params.nx) * static_cast<std::size_t>(params.ny);
    workspace.u_local.resize(local_real_size);
    workspace.v_local.resize(local_real_size);
    workspace.dwdz_local.resize(local_real_size);

    const double inv_dz = 1.0 / params.dz();
    for (int k_local = 0; k_local < slab.k_count; ++k_local) {
        const int k_global = slab.k_begin + k_local;
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t local_n = idx(params, i, j, k_local);
                const std::size_t global_n = idx(params, i, j, k_global);
                workspace.u_local[local_n] = state.u[global_n];
                workspace.v_local[local_n] = state.v[global_n];
                workspace.dwdz_local[local_n] =
                    (state.w[z_face_idx(params, i, j, k_global + 1)] - state.w[z_face_idx(params, i, j, k_global)]) * inv_dz;
            }
        }
    }

    fft.forward_planes(workspace.u_local, slab.k_count, workspace.u_hat_local, params);
    fft.forward_planes(workspace.v_local, slab.k_count, workspace.v_hat_local, params);
    fft.forward_planes(workspace.dwdz_local, slab.k_count, workspace.dwdz_hat_local, params);

    workspace.div_hat_local.resize(workspace.u_hat_local.size());
    for (int k_local = 0; k_local < slab.k_count; ++k_local) {
        for (int j = 0; j < params.ny; ++j) {
            const Complex i_ky{0.0, ky_derivative_value(params, j)};
            for (int ih = 0; ih < params.nkx(); ++ih) {
                const std::size_t n = local_sidx(params, ih, j, k_local);
                const Complex i_kx{0.0, kx_derivative_value(params, ih)};
                workspace.div_hat_local[n] =
                    i_kx * workspace.u_hat_local[n] + i_ky * workspace.v_hat_local[n] + workspace.dwdz_hat_local[n];
            }
        }
    }
}

void mpi_ialltoallv_complex(
    const std::vector<Complex>& send,
    const std::vector<int>& send_counts,
    const std::vector<int>& send_displs,
    std::vector<Complex>& recv,
    const std::vector<int>& recv_counts,
    const std::vector<int>& recv_displs,
    MPI_Comm comm) {
    std::vector<int> send_counts_double(send_counts.size());
    std::vector<int> send_displs_double(send_displs.size());
    std::vector<int> recv_counts_double(recv_counts.size());
    std::vector<int> recv_displs_double(recv_displs.size());
    for (std::size_t r = 0; r < send_counts.size(); ++r) {
        send_counts_double[r] = 2 * send_counts[r];
        send_displs_double[r] = 2 * send_displs[r];
        recv_counts_double[r] = 2 * recv_counts[r];
        recv_displs_double[r] = 2 * recv_displs[r];
    }
    const int recv_size = recv_counts.empty()
        ? 0
        : recv_displs.back() + recv_counts.back();
    recv.resize(static_cast<std::size_t>(recv_size));
    MPI_Request request = MPI_REQUEST_NULL;
    MPI_Ialltoallv(
        reinterpret_cast<const double*>(send.data()),
        send_counts_double.data(),
        send_displs_double.data(),
        MPI_DOUBLE,
        reinterpret_cast<double*>(recv.data()),
        recv_counts_double.data(),
        recv_displs_double.data(),
        MPI_DOUBLE,
        comm,
        &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
}

std::size_t pressure_coeff_idx(const Params& params, int ih, int j_local, int k) {
    return (static_cast<std::size_t>(j_local) * static_cast<std::size_t>(params.nkx()) + static_cast<std::size_t>(ih))
        * static_cast<std::size_t>(params.nz)
        + static_cast<std::size_t>(k);
}

void ensure_pressure_thomas_coefficients(PressureWorkspace& workspace, const Params& params, const Slab& slab) {
    if (workspace.thomas_nx == params.nx
        && workspace.thomas_ny == params.ny
        && workspace.thomas_nz == params.nz
        && workspace.thomas_rank == slab.rank
        && workspace.thomas_size == slab.size) {
        return;
    }
    const int nj = params.ny / slab.size;
    const std::size_t coeff_count =
        static_cast<std::size_t>(params.nkx()) * static_cast<std::size_t>(nj) * static_cast<std::size_t>(params.nz);
    workspace.thomas_lower.resize(coeff_count);
    workspace.thomas_cp.resize(coeff_count);
    workspace.thomas_inv_denom.resize(coeff_count);
    workspace.thomas_dp.resize(static_cast<std::size_t>(params.nz));

    const double inv_dz2 = 1.0 / (params.dz() * params.dz());
    for (int j_local = 0; j_local < nj; ++j_local) {
        const int j_global = slab.rank * nj + j_local;
        const double ky = ky_derivative_value(params, j_global);
        for (int ih = 0; ih < params.nkx(); ++ih) {
            const double kx = kx_derivative_value(params, ih);
            const double kh2 = kx * kx + ky * ky;
            for (int k = 0; k < params.nz; ++k) {
                Complex lower{0.0, 0.0};
                Complex diag{0.0, 0.0};
                Complex upper{0.0, 0.0};
                if (kh2 == 0.0) {
                    if (k == 0) {
                        diag = Complex{1.0, 0.0};
                    } else if (k == params.nz - 1) {
                        lower = Complex{inv_dz2, 0.0};
                        diag = Complex{-inv_dz2, 0.0};
                    } else {
                        lower = Complex{inv_dz2, 0.0};
                        diag = Complex{-2.0 * inv_dz2, 0.0};
                        upper = Complex{inv_dz2, 0.0};
                    }
                } else {
                    if (k == 0) {
                        diag = Complex{-inv_dz2 - kh2, 0.0};
                        upper = Complex{inv_dz2, 0.0};
                    } else if (k == params.nz - 1) {
                        lower = Complex{inv_dz2, 0.0};
                        diag = Complex{-inv_dz2 - kh2, 0.0};
                    } else {
                        lower = Complex{inv_dz2, 0.0};
                        diag = Complex{-2.0 * inv_dz2 - kh2, 0.0};
                        upper = Complex{inv_dz2, 0.0};
                    }
                }
                const std::size_t n = pressure_coeff_idx(params, ih, j_local, k);
                const Complex denom = k == 0 ? diag : diag - lower * workspace.thomas_cp[pressure_coeff_idx(params, ih, j_local, k - 1)];
                if (std::abs(denom) == 0.0) {
                    throw std::runtime_error("singular distributed pressure tridiagonal pivot");
                }
                workspace.thomas_lower[n] = lower;
                workspace.thomas_inv_denom[n] = Complex{1.0, 0.0} / denom;
                workspace.thomas_cp[n] = (k == params.nz - 1) ? Complex{0.0, 0.0} : upper / denom;
            }
        }
    }
    workspace.thomas_nx = params.nx;
    workspace.thomas_ny = params.ny;
    workspace.thomas_nz = params.nz;
    workspace.thomas_rank = slab.rank;
    workspace.thomas_size = slab.size;
}

void solve_pressure_y_pencils(std::vector<Complex>& y_pencil, const Params& params, const Slab& slab, PressureWorkspace& workspace) {
    ensure_pressure_thomas_coefficients(workspace, params, slab);
    const int nj = params.ny / slab.size;
    for (int j_local = 0; j_local < nj; ++j_local) {
        const int j_global = slab.rank * nj + j_local;
        const bool zero_mode = ky_derivative_value(params, j_global) == 0.0;
        for (int ih = 0; ih < params.nkx(); ++ih) {
            const bool zero_pressure_mode = zero_mode && kx_derivative_value(params, ih) == 0.0;
            for (int k = 0; k < params.nz; ++k) {
                const std::size_t coeff = pressure_coeff_idx(params, ih, j_local, k);
                Complex rhs = y_pencil[pencil_sidx(params, ih, j_local, k, nj)] / params.dt;
                if (zero_pressure_mode && k == 0) {
                    rhs = Complex{0.0, 0.0};
                }
                workspace.thomas_dp[static_cast<std::size_t>(k)] = k == 0
                    ? rhs * workspace.thomas_inv_denom[coeff]
                    : (rhs - workspace.thomas_lower[coeff] * workspace.thomas_dp[static_cast<std::size_t>(k - 1)])
                        * workspace.thomas_inv_denom[coeff];
            }
            Complex solution = workspace.thomas_dp[static_cast<std::size_t>(params.nz - 1)];
            y_pencil[pencil_sidx(params, ih, j_local, params.nz - 1, nj)] = solution;
            for (int k = params.nz - 2; k >= 0; --k) {
                const std::size_t coeff = pressure_coeff_idx(params, ih, j_local, k);
                solution = workspace.thomas_dp[static_cast<std::size_t>(k)] - workspace.thomas_cp[coeff] * solution;
                y_pencil[pencil_sidx(params, ih, j_local, k, nj)] = solution;
            }
        }
    }
}

void transpose_z_slab_to_y_pencil(
    const SpectralField& local_hat,
    std::vector<Complex>& y_pencil,
    const Params& params,
    const Slab& slab,
    MPI_Comm comm,
    PressureWorkspace& workspace) {
    const int nj = params.ny / slab.size;
    const int local_chunk = params.nkx() * nj * slab.k_count;
    std::vector<Complex>& send = workspace.transpose_send;
    std::vector<Complex>& recv = workspace.transpose_recv;
    send.resize(static_cast<std::size_t>(local_chunk * slab.size));

    for (int dest = 0; dest < slab.size; ++dest) {
        for (int k_local = 0; k_local < slab.k_count; ++k_local) {
            for (int j_local = 0; j_local < nj; ++j_local) {
                const int j_global = dest * nj + j_local;
                for (int ih = 0; ih < params.nkx(); ++ih) {
                    const std::size_t packed =
                        static_cast<std::size_t>(dest * local_chunk + (k_local * nj + j_local) * params.nkx() + ih);
                    send[packed] = local_hat[local_sidx(params, ih, j_global, k_local)];
                }
            }
        }
    }

    std::vector<int> send_counts(static_cast<std::size_t>(slab.size), local_chunk);
    std::vector<int> send_displs(static_cast<std::size_t>(slab.size), 0);
    std::vector<int> recv_counts(static_cast<std::size_t>(slab.size), 0);
    std::vector<int> recv_displs(static_cast<std::size_t>(slab.size), 0);
    for (int r = 0; r < slab.size; ++r) {
        send_displs[static_cast<std::size_t>(r)] = r * local_chunk;
        const int source_k_count = slab.center_counts[static_cast<std::size_t>(r)] / (params.nx * params.ny);
        recv_counts[static_cast<std::size_t>(r)] = params.nkx() * nj * source_k_count;
        if (r > 0) {
            recv_displs[static_cast<std::size_t>(r)] = recv_displs[static_cast<std::size_t>(r - 1)]
                + recv_counts[static_cast<std::size_t>(r - 1)];
        }
    }
    mpi_ialltoallv_complex(
        send, send_counts, send_displs, recv, recv_counts, recv_displs, comm);

    y_pencil.resize(static_cast<std::size_t>(params.nkx()) * static_cast<std::size_t>(nj) * static_cast<std::size_t>(params.nz));
    for (int source = 0; source < slab.size; ++source) {
        const int source_k_begin = slab.center_displs[static_cast<std::size_t>(source)] / (params.nx * params.ny);
        const int source_k_count = slab.center_counts[static_cast<std::size_t>(source)] / (params.nx * params.ny);
        for (int k_local = 0; k_local < source_k_count; ++k_local) {
            const int k_global = source_k_begin + k_local;
            for (int j_local = 0; j_local < nj; ++j_local) {
                for (int ih = 0; ih < params.nkx(); ++ih) {
                    const std::size_t packed =
                        static_cast<std::size_t>(recv_displs[static_cast<std::size_t>(source)]
                            + (k_local * nj + j_local) * params.nkx() + ih);
                    y_pencil[pencil_sidx(params, ih, j_local, k_global, nj)] = recv[packed];
                }
            }
        }
    }
}

void transpose_y_pencil_to_z_slab(
    const SpectralField& y_pencil,
    SpectralField& local_hat,
    const Params& params,
    const Slab& slab,
    MPI_Comm comm,
    PressureWorkspace& workspace) {
    const int nj = params.ny / slab.size;
    std::vector<Complex>& send = workspace.transpose_send;
    std::vector<Complex>& recv = workspace.transpose_recv;
    const int pencil_size = params.nkx() * nj * params.nz;
    send.resize(static_cast<std::size_t>(pencil_size));

    std::vector<int> send_counts(static_cast<std::size_t>(slab.size), 0);
    std::vector<int> send_displs(static_cast<std::size_t>(slab.size), 0);
    std::vector<int> recv_counts(static_cast<std::size_t>(slab.size), params.nkx() * nj * slab.k_count);
    std::vector<int> recv_displs(static_cast<std::size_t>(slab.size), 0);
    for (int r = 0; r < slab.size; ++r) {
        const int dest_k_count = slab.center_counts[static_cast<std::size_t>(r)] / (params.nx * params.ny);
        send_counts[static_cast<std::size_t>(r)] = params.nkx() * nj * dest_k_count;
        if (r > 0) {
            send_displs[static_cast<std::size_t>(r)] = send_displs[static_cast<std::size_t>(r - 1)]
                + send_counts[static_cast<std::size_t>(r - 1)];
            recv_displs[static_cast<std::size_t>(r)] = recv_displs[static_cast<std::size_t>(r - 1)]
                + recv_counts[static_cast<std::size_t>(r - 1)];
        }
    }

    for (int dest = 0; dest < slab.size; ++dest) {
        const int dest_k_begin = slab.center_displs[static_cast<std::size_t>(dest)] / (params.nx * params.ny);
        const int dest_k_count = slab.center_counts[static_cast<std::size_t>(dest)] / (params.nx * params.ny);
        for (int k_local = 0; k_local < dest_k_count; ++k_local) {
            const int k_global = dest_k_begin + k_local;
            for (int j_local = 0; j_local < nj; ++j_local) {
                for (int ih = 0; ih < params.nkx(); ++ih) {
                    const std::size_t packed =
                        static_cast<std::size_t>(send_displs[static_cast<std::size_t>(dest)]
                            + (k_local * nj + j_local) * params.nkx() + ih);
                    send[packed] = y_pencil[pencil_sidx(params, ih, j_local, k_global, nj)];
                }
            }
        }
    }

    mpi_ialltoallv_complex(
        send, send_counts, send_displs, recv, recv_counts, recv_displs, comm);

    local_hat.resize(static_cast<std::size_t>(params.nkx()) * static_cast<std::size_t>(params.ny) * static_cast<std::size_t>(slab.k_count));
    for (int source = 0; source < slab.size; ++source) {
        for (int k_local = 0; k_local < slab.k_count; ++k_local) {
            for (int j_local = 0; j_local < nj; ++j_local) {
                const int j_global = source * nj + j_local;
                for (int ih = 0; ih < params.nkx(); ++ih) {
                    const std::size_t packed =
                        static_cast<std::size_t>(recv_displs[static_cast<std::size_t>(source)]
                            + (k_local * nj + j_local) * params.nkx() + ih);
                    local_hat[local_sidx(params, ih, j_global, k_local)] = recv[packed];
                }
            }
        }
    }
}

void exchange_neighbor_planes(
    LocalField& q,
    int owned_begin,
    int owned_count,
    int total_planes,
    int tag_base,
    const Params& params,
    const Slab& slab,
    MPI_Comm comm,
    HaloExchangeScratch* scratch = nullptr);

void exchange_neighbor_field_pack(
    const std::vector<LocalHaloField>& fields,
    int tag_base,
    const Params& params,
    const Slab& slab,
    MPI_Comm comm,
    HaloExchangeScratch* scratch = nullptr);

void exchange_state_halos(
    MpiLocalFlowState& state,
    const Params& params,
    const Slab& slab,
    MPI_Comm comm,
    HaloExchangeScratch* scratch = nullptr);

void project_mpi_y_pencil(
    MpiLocalFlowState& state,
    const Params& params,
    const Slab& slab,
    FftwXY& fft,
    MPI_Comm comm,
    PressureWorkspace& workspace,
    HaloExchangeScratch* halo_scratch = nullptr) {
    build_pressure_divergence_hat_slab(state, params, slab, fft, workspace);

    transpose_z_slab_to_y_pencil(workspace.div_hat_local, workspace.y_pencil, params, slab, comm, workspace);
    solve_pressure_y_pencils(workspace.y_pencil, params, slab, workspace);
    transpose_y_pencil_to_z_slab(workspace.y_pencil, workspace.p_hat_local, params, slab, comm, workspace);

    workspace.dpdx_hat_local.resize(workspace.p_hat_local.size());
    workspace.dpdy_hat_local.resize(workspace.p_hat_local.size());
    for (int k_local = 0; k_local < slab.k_count; ++k_local) {
        for (int j = 0; j < params.ny; ++j) {
            const double ky = ky_derivative_value(params, j);
            for (int ih = 0; ih < params.nkx(); ++ih) {
                const std::size_t n = local_sidx(params, ih, j, k_local);
                workspace.dpdx_hat_local[n] = Complex{0.0, kx_derivative_value(params, ih)} * workspace.p_hat_local[n];
                workspace.dpdy_hat_local[n] = Complex{0.0, ky} * workspace.p_hat_local[n];
            }
        }
    }

    fft.inverse_planes(workspace.p_hat_local, slab.k_count, workspace.p_local, params);
    fft.inverse_planes(workspace.dpdx_hat_local, slab.k_count, workspace.dpdx_local, params);
    fft.inverse_planes(workspace.dpdy_hat_local, slab.k_count, workspace.dpdy_local, params);

    for (int k_local = 0; k_local < slab.k_count; ++k_local) {
        const int k_global = slab.k_begin + k_local;
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t local_n = idx(params, i, j, k_local);
                const std::size_t global_n = idx(params, i, j, k_global);
                state.p[global_n] = workspace.p_local[local_n];
                state.u[global_n] -= params.dt * workspace.dpdx_local[local_n];
                state.v[global_n] -= params.dt * workspace.dpdy_local[local_n];
            }
        }
    }

    exchange_neighbor_planes(state.p, slab.k_begin, slab.k_count, params.nz, 600, params, slab, comm, halo_scratch);
    const double inv_dz = 1.0 / params.dz();
    for (int k = slab.face_begin; k < slab.face_begin + slab.face_count; ++k) {
        if (k <= 0 || k >= params.nz) {
            continue;
        }
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                state.w[z_face_idx(params, i, j, k)] -=
                    params.dt * (state.p[idx(params, i, j, k)] - state.p[idx(params, i, j, k - 1)]) * inv_dz;
            }
        }
    }
    enforce_walls_slab(state.w, params, slab);
    exchange_state_halos(state, params, slab, comm, halo_scratch);
    enforce_walls_slab(state.w, params, slab);
}

void exchange_neighbor_field_pack(
    const std::vector<LocalHaloField>& fields,
    int tag_base,
    const Params& params,
    const Slab& slab,
    MPI_Comm comm,
    HaloExchangeScratch* scratch) {
    const int plane_size = params.nx * params.ny;
    const int lower_rank = slab.rank > 0 ? slab.rank - 1 : MPI_PROC_NULL;
    const int upper_rank = slab.rank + 1 < slab.size ? slab.rank + 1 : MPI_PROC_NULL;
    auto has_lower_recv = [&](const LocalHaloField& field) {
        return field.data != nullptr && lower_rank != MPI_PROC_NULL && field.owned_begin > 0;
    };
    auto has_upper_recv = [&](const LocalHaloField& field) {
        return field.data != nullptr
            && upper_rank != MPI_PROC_NULL
            && field.owned_begin + field.owned_count < field.total_planes;
    };
    auto has_lower_send = [&](const LocalHaloField& field) {
        return field.data != nullptr && lower_rank != MPI_PROC_NULL && field.owned_count > 0;
    };
    auto has_upper_send = [&](const LocalHaloField& field) {
        return field.data != nullptr && upper_rank != MPI_PROC_NULL && field.owned_count > 0;
    };

    std::size_t lower_recv_count = 0;
    std::size_t upper_recv_count = 0;
    std::size_t lower_send_count = 0;
    std::size_t upper_send_count = 0;
    for (const LocalHaloField& field : fields) {
        lower_recv_count += has_lower_recv(field) ? static_cast<std::size_t>(plane_size) : 0;
        upper_recv_count += has_upper_recv(field) ? static_cast<std::size_t>(plane_size) : 0;
        lower_send_count += has_lower_send(field) ? static_cast<std::size_t>(plane_size) : 0;
        upper_send_count += has_upper_send(field) ? static_cast<std::size_t>(plane_size) : 0;
    }

    thread_local HaloExchangeScratch fallback_scratch;
    HaloExchangeScratch& work = scratch == nullptr ? fallback_scratch : *scratch;
    work.send_lower.clear();
    work.send_upper.clear();
    work.recv_lower.resize(lower_recv_count);
    work.recv_upper.resize(upper_recv_count);
    work.requests.clear();
    work.requests.reserve(4);

    if (!work.recv_lower.empty()) {
        MPI_Request request = MPI_REQUEST_NULL;
        MPI_Irecv(work.recv_lower.data(), static_cast<int>(work.recv_lower.size()), MPI_DOUBLE, lower_rank, tag_base + 1, comm, &request);
        work.requests.push_back(request);
    }
    if (!work.recv_upper.empty()) {
        MPI_Request request = MPI_REQUEST_NULL;
        MPI_Irecv(work.recv_upper.data(), static_cast<int>(work.recv_upper.size()), MPI_DOUBLE, upper_rank, tag_base + 0, comm, &request);
        work.requests.push_back(request);
    }
    if (lower_send_count > 0) {
        work.send_lower.resize(lower_send_count);
        std::size_t offset = 0;
        for (const LocalHaloField& field : fields) {
            if (!has_lower_send(field)) {
                continue;
            }
            const int send_plane = field.owned_begin;
            const auto begin = field.data->values.begin()
                + static_cast<std::ptrdiff_t>((send_plane - field.data->plane_begin) * plane_size);
            std::copy_n(begin, plane_size, work.send_lower.begin() + static_cast<std::ptrdiff_t>(offset));
            offset += static_cast<std::size_t>(plane_size);
        }
        MPI_Request request = MPI_REQUEST_NULL;
        MPI_Isend(work.send_lower.data(), static_cast<int>(work.send_lower.size()), MPI_DOUBLE, lower_rank, tag_base + 0, comm, &request);
        work.requests.push_back(request);
    }
    if (upper_send_count > 0) {
        work.send_upper.resize(upper_send_count);
        std::size_t offset = 0;
        for (const LocalHaloField& field : fields) {
            if (!has_upper_send(field)) {
                continue;
            }
            const int send_plane = field.owned_begin + field.owned_count - 1;
            const auto begin = field.data->values.begin()
                + static_cast<std::ptrdiff_t>((send_plane - field.data->plane_begin) * plane_size);
            std::copy_n(begin, plane_size, work.send_upper.begin() + static_cast<std::ptrdiff_t>(offset));
            offset += static_cast<std::size_t>(plane_size);
        }
        MPI_Request request = MPI_REQUEST_NULL;
        MPI_Isend(work.send_upper.data(), static_cast<int>(work.send_upper.size()), MPI_DOUBLE, upper_rank, tag_base + 1, comm, &request);
        work.requests.push_back(request);
    }
    if (!work.requests.empty()) {
        MPI_Waitall(static_cast<int>(work.requests.size()), work.requests.data(), MPI_STATUSES_IGNORE);
    }
    std::size_t lower_offset = 0;
    for (const LocalHaloField& field : fields) {
        if (!has_lower_recv(field)) {
            continue;
        }
        const int target_plane = field.owned_begin - 1;
        std::copy_n(
            work.recv_lower.begin() + static_cast<std::ptrdiff_t>(lower_offset),
            plane_size,
            field.data->values.begin() + static_cast<std::ptrdiff_t>((target_plane - field.data->plane_begin) * plane_size));
        lower_offset += static_cast<std::size_t>(plane_size);
    }
    std::size_t upper_offset = 0;
    for (const LocalHaloField& field : fields) {
        if (!has_upper_recv(field)) {
            continue;
        }
        const int target_plane = field.owned_begin + field.owned_count;
        std::copy_n(
            work.recv_upper.begin() + static_cast<std::ptrdiff_t>(upper_offset),
            plane_size,
            field.data->values.begin() + static_cast<std::ptrdiff_t>((target_plane - field.data->plane_begin) * plane_size));
        upper_offset += static_cast<std::size_t>(plane_size);
    }
}

void exchange_neighbor_planes(
    LocalField& q,
    int owned_begin,
    int owned_count,
    int total_planes,
    int tag_base,
    const Params& params,
    const Slab& slab,
    MPI_Comm comm,
    HaloExchangeScratch* scratch) {
    exchange_neighbor_field_pack({LocalHaloField{&q, owned_begin, owned_count, total_planes}}, tag_base, params, slab, comm, scratch);
}

void exchange_state_halos(
    MpiLocalFlowState& state,
    const Params& params,
    const Slab& slab,
    MPI_Comm comm,
    HaloExchangeScratch* scratch) {
    std::vector<LocalHaloField> fields{
        LocalHaloField{&state.u, slab.k_begin, slab.k_count, params.nz},
        LocalHaloField{&state.v, slab.k_begin, slab.k_count, params.nz},
        LocalHaloField{&state.w, slab.face_begin, slab.face_count, params.nz + 1},
    };
    if (params.thermo_enabled) {
        fields.push_back(LocalHaloField{&state.theta, slab.k_begin, slab.k_count, params.nz});
    }
    if (params.moisture_enabled) {
        fields.push_back(LocalHaloField{&state.qv, slab.k_begin, slab.k_count, params.nz});
        fields.push_back(LocalHaloField{&state.theta_l, slab.k_begin, slab.k_count, params.nz});
        fields.push_back(LocalHaloField{&state.qt, slab.k_begin, slab.k_count, params.nz});
        fields.push_back(LocalHaloField{&state.ql, slab.k_begin, slab.k_count, params.nz});
    }
    if (uses_moeng_tke(params)) {
        fields.push_back(LocalHaloField{&state.sgs_tke, slab.k_begin, slab.k_count, params.nz});
    }
    exchange_neighbor_field_pack(fields, 100, params, slab, comm, scratch);
}

void require_distributed_rhs_supported(const Params& params) {
    if (params.scalar_sgs_model == "lasd" && params.sgs_model != "lasd") {
        throw std::runtime_error("scalar LASD requires momentum LASD in MPI slab mode");
    }
    if ((params.scalar_sgs_model == "amd"
            || params.scalar_sgs_model == "amd_shared"
            || params.scalar_sgs_model == "amd_plane_dissipation")
        && params.sgs_model != "amd" && params.sgs_model != "amd_plane_dissipation") {
        throw std::runtime_error("scalar AMD requires momentum AMD in MPI slab mode");
    }
}

void horizontal_dealias_field_slab(
    LocalField& q,
    int plane_begin,
    int plane_count,
    const Params& params,
    FftwXY& fft) {
    constexpr double two_thirds_filter_width = 1.5;
    const int local_begin = plane_begin - q.plane_begin;
    Field filtered;
    fft.filter_plane_range(q.values, local_begin, plane_count, filtered, params, two_thirds_filter_width);
    const std::size_t plane_size = static_cast<std::size_t>(params.nx) * static_cast<std::size_t>(params.ny);
    for (int k = plane_begin; k < plane_begin + plane_count; ++k) {
        const std::size_t local_offset = static_cast<std::size_t>(k - q.plane_begin) * plane_size;
        std::copy_n(filtered.begin() + static_cast<std::ptrdiff_t>(local_offset),
            plane_size,
            q.values.begin() + static_cast<std::ptrdiff_t>(local_offset));
    }
}

void horizontal_clear_nyquist_field_slab(
    LocalField& q,
    int plane_begin,
    int plane_count,
    const Params& params,
    FftwXY& fft) {
    const int local_begin = plane_begin - q.plane_begin;
    Field filtered;
    fft.clear_nyquist_plane_range(q.values, local_begin, plane_count, filtered, params);
    const std::size_t plane_size = static_cast<std::size_t>(params.nx) * static_cast<std::size_t>(params.ny);
    for (int k = plane_begin; k < plane_begin + plane_count; ++k) {
        const std::size_t local_offset = static_cast<std::size_t>(k - q.plane_begin) * plane_size;
        std::copy_n(filtered.begin() + static_cast<std::ptrdiff_t>(local_offset),
            plane_size,
            q.values.begin() + static_cast<std::ptrdiff_t>(local_offset));
    }
}

void horizontal_dealias_state_slab(MpiLocalFlowState& state, const Params& params, const Slab& slab, FftwXY& fft) {
    if (!params.horizontal_dealias) {
        return;
    }
    auto apply_horizontal_dealias = [&](LocalField& q, int plane_begin, int plane_count) {
        if (params.dealiasing == "padding_3_2") {
            horizontal_clear_nyquist_field_slab(q, plane_begin, plane_count, params, fft);
        } else {
            horizontal_dealias_field_slab(q, plane_begin, plane_count, params, fft);
        }
    };
    apply_horizontal_dealias(state.u, slab.k_begin, slab.k_count);
    apply_horizontal_dealias(state.v, slab.k_begin, slab.k_count);
    apply_horizontal_dealias(state.w, slab.face_begin, slab.face_count);
    if (params.moisture_enabled) {
        apply_horizontal_dealias(state.theta_l, slab.k_begin, slab.k_count);
        apply_horizontal_dealias(state.qt, slab.k_begin, slab.k_count);
    } else if (params.thermo_enabled) {
        apply_horizontal_dealias(state.theta, slab.k_begin, slab.k_count);
    }
    if (uses_moeng_tke(params)) {
        apply_horizontal_dealias(state.sgs_tke, slab.k_begin, slab.k_count);
    }
    enforce_walls_slab(state.w, params, slab);
}

void add_coriolis_geostrophic_forcing_slab(
    LocalField& rhs_u,
    LocalField& rhs_v,
    const MpiLocalFlowState& state,
    const Params& params,
    const Slab& slab) {
    if (params.coriolis_f == 0.0) {
        return;
    }
    for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
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

void strain_magnitude_slab(const LocalVelocityGradients& grad, LocalField& mag, const Params& params, const Slab& slab) {
    clear_center_slab_field(mag, params, slab);
    for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                const double s11 = grad.dudx[n];
                const double s22 = grad.dvdy[n];
                const double s33 = grad.dwdz[n];
                const double s12 = 0.5 * (grad.dudy[n] + grad.dvdx[n]);
                const double s13 = 0.5 * (grad.dudz[n] + grad.dwdx[n]);
                const double s23 = 0.5 * (grad.dvdz[n] + grad.dwdy[n]);
                const double sij_sij = s11 * s11 + s22 * s22 + s33 * s33
                    + 2.0 * (s12 * s12 + s13 * s13 + s23 * s23);
                mag[n] = std::sqrt(std::max(2.0 * sij_sij, 0.0));
            }
        }
    }
}

void update_moeng_tke_coefficients_slab(
    const MpiLocalFlowState& state,
    LocalField& eddy_viscosity,
    LocalField& mixing_length,
    LocalField& scalar_diffusivity,
    LocalField& tke_diffusivity,
    LocalField& dtheta_v_dz,
    const Params& params,
    const Slab& slab) {
    LocalField theta_v;
    theta_v.resize(slab.k_begin, slab.k_count, params.nz, params);
    clear_center_slab_field(dtheta_v_dz, params, slab);
    for (int k = theta_v.plane_begin; k < theta_v.plane_begin + theta_v.plane_count; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                theta_v[n] = params.moisture_enabled
                    ? state.theta[n] * (1.0 + 0.61 * state.qv[n] - state.ql[n])
                    : state.theta[n];
            }
        }
    }
    ddz_center_slab(theta_v, dtheta_v_dz, params, slab);

    clear_center_slab_field(eddy_viscosity, params, slab);
    clear_center_slab_field(mixing_length, params, slab);
    clear_center_slab_field(scalar_diffusivity, params, slab);
    clear_center_slab_field(tke_diffusivity, params, slab);
    const double delta = params.sgs_delta();
    for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
        const double z = (static_cast<double>(k) + 0.5) * params.dz();
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                const double e = std::max(state.sgs_tke[n], params.tke_floor);
                const double n2 = (params.g / params.theta0) * dtheta_v_dz[n];
                // Deardorff/Moeng baseline wall limit used by the BOMEX NCAR
                // configuration.  The more aggressive kappa*z cap was tested
                // separately and produced excessive resolved near-wall TKE.
                double length = std::min(delta, 1.8 * z);
                if (n2 > 0.0) {
                    length = std::min(length, params.tke_length_coefficient * std::sqrt(e / n2));
                }
                length = std::max(length, 1.0e-12 * delta);
                const double km = params.tke_ck * length * std::sqrt(e);
                mixing_length[n] = length;
                eddy_viscosity[n] = km;
                scalar_diffusivity[n] = (1.0 + 2.0 * length / delta) * km;
                tke_diffusivity[n] = 2.0 * km;
            }
        }
    }
}

LocalSymFields strain_components_slab(const LocalVelocityGradients& grad, const Params& params, const Slab& slab) {
    LocalSymFields sij;
    for (LocalField& q : sij) {
        q.resize(slab.k_begin, slab.k_count, params.nz, params);
        q.clear_owned();
    }
    for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                sij[0][n] = grad.dudx[n];
                sij[1][n] = 0.5 * (grad.dudy[n] + grad.dvdx[n]);
                sij[2][n] = 0.5 * (grad.dudz[n] + grad.dwdx[n]);
                sij[3][n] = grad.dvdy[n];
                sij[4][n] = 0.5 * (grad.dvdz[n] + grad.dwdy[n]);
                sij[5][n] = grad.dwdz[n];
            }
        }
    }
    return sij;
}

ConstLocalVecFields centered_velocity_slab(const LocalField& u, const LocalField& v, const LocalField& w_center) {
    return ConstLocalVecFields{&u, &v, &w_center};
}

void apply_center_history_bc_slab(LocalField& q, const Params& params, const Slab& slab) {
    if (params.nz < 2) {
        return;
    }
    if (slab.rank == 0) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                q[idx(params, i, j, 0)] = q[idx(params, i, j, 1)];
            }
        }
    }
    if (slab.rank == slab.size - 1) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                q[idx(params, i, j, params.nz - 1)] = q[idx(params, i, j, params.nz - 2)];
            }
        }
    }
}

void update_lagrangian_velocity_slab(
    MpiLocalFlowState& state,
    const LocalField& w_center,
    const Params& params,
    const Slab& slab,
    MPI_Comm comm) {
    const double inv_count = 1.0 / static_cast<double>(params.cs_count);
    for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                state.u_lag[n] += state.u[n] * inv_count;
                state.v_lag[n] += state.v[n] * inv_count;
                state.w_lag[n] += w_center[n] * inv_count;
            }
        }
    }
    exchange_neighbor_field_pack(
        {
            LocalHaloField{&state.u_lag, slab.k_begin, slab.k_count, params.nz},
            LocalHaloField{&state.v_lag, slab.k_begin, slab.k_count, params.nz},
            LocalHaloField{&state.w_lag, slab.k_begin, slab.k_count, params.nz},
        },
        700,
        params,
        slab,
        comm);
}

void reset_lasd_velocity_accumulators_slab(MpiLocalFlowState& state, const Params& params, const Slab& slab) {
    for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                state.u_lag[n] = 0.0;
                state.v_lag[n] = 0.0;
                state.w_lag[n] = 0.0;
            }
        }
    }
}

void eddy_viscosity_from_cs2_slab(const LocalField& cs2, const LocalField& strain, LocalField& nu_t, const Params& params, const Slab& slab) {
    clear_center_slab_field(nu_t, params, slab);
    const double delta2 = params.sgs_delta() * params.sgs_delta();
    for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                nu_t[n] = std::max(cs2[n], 0.0) * delta2 * strain[n];
            }
        }
    }
}

void smagorinsky_eddy_viscosity_slab(
    const MpiLocalFlowState& state,
    const LocalField& strain,
    LocalField& nu_t,
    const Params& params,
    const Slab& slab) {
    clear_center_slab_field(nu_t, params, slab);
    const double coeff = std::pow(params.smagorinsky_cs * params.sgs_delta(), 2.0);
    if (!params.smagorinsky_buoyancy_correction || !params.thermo_enabled) {
        for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const std::size_t n = idx(params, i, j, k);
                    nu_t[n] = coeff * strain[n];
                }
            }
        }
        return;
    }
    LocalField buoyancy_scalar;
    LocalField dbuoyancy_dz;
    buoyancy_scalar.resize(slab.k_begin, slab.k_count, params.nz, params);
    dbuoyancy_dz.resize(slab.k_begin, slab.k_count, params.nz, params);
    for (int k = buoyancy_scalar.plane_begin; k < buoyancy_scalar.plane_begin + buoyancy_scalar.plane_count; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                buoyancy_scalar[n] = params.moisture_enabled
                    ? state.theta[n] * (1.0 + 0.61 * state.qv[n] - state.ql[n])
                    : state.theta[n];
            }
        }
    }
    ddz_center_slab(buoyancy_scalar, dbuoyancy_dz, params, slab);
    for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                const double n2 = params.g * dbuoyancy_dz[n] / std::max(buoyancy_scalar[n], 1.0);
                const double effective_strain_squared = strain[n] * strain[n] - n2 / params.prandtl_t;
                const double corrected = coeff * std::sqrt(std::max(effective_strain_squared, 0.0));
                const double shear_floor = params.smagorinsky_min_shear_fraction * coeff * strain[n];
                nu_t[n] = std::max(corrected, shear_floor);
            }
        }
    }
}

void lagrangian_interp_center_slab_into(
    const LocalField& q,
    const LocalField& u_lag,
    const LocalField& v_lag,
    const LocalField& w_lag,
    const Params& params,
    const Slab& slab,
    LocalField& out,
    LocalField& /*scratch*/) {
    const double dt_lag = params.dt * static_cast<double>(params.cs_count);
    clear_center_slab_field(out, params, slab);
    auto periodic_coordinate = [](double coordinate, int extent) {
        coordinate = std::fmod(coordinate, static_cast<double>(extent));
        return coordinate < 0.0 ? coordinate + static_cast<double>(extent) : coordinate;
    };
    for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                // The update guard enforces cs_count*CFL <= 1, so the
                // departure point requires at most the one exchanged z halo.
                const double xi = periodic_coordinate(
                    static_cast<double>(i)
                        + std::clamp(-u_lag[n] * dt_lag / params.dx(), -1.0, 1.0),
                    params.nx);
                const double eta = periodic_coordinate(
                    static_cast<double>(j)
                        + std::clamp(-v_lag[n] * dt_lag / params.dy(), -1.0, 1.0),
                    params.ny);
                const double zeta = std::clamp(
                    static_cast<double>(k)
                        + std::clamp(-w_lag[n] * dt_lag / params.dz(), -1.0, 1.0),
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
                out[n] = (1.0 - fz) * ((1.0 - fy) * q00 + fy * q10)
                    + fz * ((1.0 - fy) * q01 + fy * q11);
            }
        }
    }
}

void lagrangian_average_fields_slab_into(
    const LocalField& current_a,
    const LocalField& current_b,
    const LocalField& old_a,
    const LocalField& old_b,
    const LocalField& u_lag,
    const LocalField& v_lag,
    const LocalField& w_lag,
    const Params& params,
    const Slab& slab,
    LocalField& avg_a,
    LocalField& avg_b,
    LocalField& a_interp,
    LocalField& b_interp,
    LocalField& interp_scratch,
    const LocalField* timescale_a = nullptr,
    const LocalField* timescale_b = nullptr) {
    lagrangian_interp_center_slab_into(old_a, u_lag, v_lag, w_lag, params, slab, a_interp, interp_scratch);
    lagrangian_interp_center_slab_into(old_b, u_lag, v_lag, w_lag, params, slab, b_interp, interp_scratch);
    clear_center_slab_field(avg_a, params, slab);
    clear_center_slab_field(avg_b, params, slab);
    const double dt_lag = params.dt * static_cast<double>(params.cs_count);
    for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
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
                const bool ramp_scalar_numerator = timescale_a != nullptr && timescale_b != nullptr;
                avg_a[n] = ramp_scalar_numerator ? (raw_a > 0.0 ? raw_a : 1.0e-32) : raw_a;
                avg_b[n] = std::max(raw_b, 0.0);
            }
        }
    }
}

LmMm momentum_lm_mm_slab(
    const LocalField& u,
    const LocalField& v,
    const LocalField& w,
    const ConstLocalVecFields& vel,
    const LocalSymFields& sij,
    const LocalField& strain,
    const Params& params,
    const Slab& slab,
    FftwXY& fft,
    double test_ratio,
    LasdGermanoScratch& scratch,
    LasdFilterCache* filter_cache,
    MPI_Comm comm) {
    const double width = params.fgr * test_ratio;
    LasdFilterCacheSlot* cache_slot = filter_cache == nullptr ? nullptr : &filter_cache->slot_for(width);
    LocalVecFields& vel_hat_storage = scratch.momentum_vel_hat_storage;
    ConstLocalVecFields vel_hat{};
    scratch.reset_filter_batch();
    if (cache_slot != nullptr && cache_slot->u_hat_valid) {
        vel_hat[0] = &cache_slot->u_hat;
    } else if (cache_slot != nullptr) {
        vel_hat[0] = &cache_slot->u_hat;
        scratch.add_filter(*vel[0], cache_slot->u_hat);
    } else {
        vel_hat[0] = &vel_hat_storage[0];
        scratch.add_filter(*vel[0], vel_hat_storage[0]);
    }
    if (cache_slot != nullptr && cache_slot->v_hat_valid) {
        vel_hat[1] = &cache_slot->v_hat;
    } else if (cache_slot != nullptr) {
        vel_hat[1] = &cache_slot->v_hat;
        scratch.add_filter(*vel[1], cache_slot->v_hat);
    } else {
        vel_hat[1] = &vel_hat_storage[1];
        scratch.add_filter(*vel[1], vel_hat_storage[1]);
    }
    if (cache_slot != nullptr && cache_slot->w_center_hat_valid) {
        vel_hat[2] = &cache_slot->w_center_hat;
    } else if (cache_slot != nullptr) {
        vel_hat[2] = &cache_slot->w_center_hat;
        scratch.add_filter(*vel[2], cache_slot->w_center_hat);
    } else {
        vel_hat[2] = &vel_hat_storage[2];
        scratch.add_filter(*vel[2], vel_hat_storage[2]);
    }
    horizontal_spectral_filter_center_slab_many(
        scratch.filter_inputs, scratch.filter_outputs, params, slab, fft, width);
    if (cache_slot != nullptr) {
        cache_slot->u_hat_valid = true;
        cache_slot->v_hat_valid = true;
        cache_slot->w_center_hat_valid = true;
    }

    LmMm out;
    clear_center_slab_field(out.lm, params, slab);
    clear_center_slab_field(out.mm, params, slab);
    const double delta2 = params.sgs_delta() * params.sgs_delta();
    const double ratio2 = test_ratio * test_ratio;
    LocalField& strain_hat = cache_slot == nullptr
        ? scratch.momentum_strain_hat
        : cache_slot->strain_hat;
    LocalField& center_hat0 = scratch.momentum_center_hat0;
    clear_center_slab_field(strain_hat, params, slab);
    for (int c = 0; c < 6; ++c) {
        scratch.reset_filter_batch();
        scratch.add_filter(sij[c], center_hat0);
        horizontal_spectral_filter_center_slab_many(
            scratch.filter_inputs, scratch.filter_outputs, params, slab, fft, width);
        const double weight = sym_component_weight(c);
        for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const std::size_t n = idx(params, i, j, k);
                    strain_hat[n] += weight * center_hat0[n] * center_hat0[n];
                }
            }
        }
    }
    for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                strain_hat[n] = std::sqrt(std::max(2.0 * strain_hat[n], 0.0));
            }
        }
    }
    exchange_neighbor_planes(strain_hat, slab.k_begin, slab.k_count, params.nz, 731, params, slab, comm);
    if (cache_slot != nullptr) {
        cache_slot->strain_hat_valid = true;
    }

    LocalField& center_work0 = scratch.momentum_center_work0;
    LocalField& center_work1 = scratch.momentum_center_work1;
    LocalField& center_hat1 = scratch.momentum_center_hat1;
    LocalField& center_hat2 = scratch.momentum_center_hat2;
    for (int c = 0; c < 6; ++c) {
        clear_center_slab_field(center_work0, params, slab);
        clear_center_slab_field(center_work1, params, slab);
        for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const std::size_t n = idx(params, i, j, k);
                    if (c == 0) {
                        center_work0[n] = (*vel[0])[n] * (*vel[0])[n];
                    } else if (c == 1) {
                        center_work0[n] = (*vel[0])[n] * (*vel[1])[n];
                    } else if (c == 2) {
                        center_work0[n] = (*vel[0])[n] * (*vel[2])[n];
                    } else if (c == 3) {
                        center_work0[n] = (*vel[1])[n] * (*vel[1])[n];
                    } else if (c == 4) {
                        center_work0[n] = (*vel[1])[n] * (*vel[2])[n];
                    } else {
                        center_work0[n] = (*vel[2])[n] * (*vel[2])[n];
                    }
                    center_work1[n] = strain[n] * sij[c][n];
                }
            }
        }
        scratch.reset_filter_batch();
        scratch.add_filter(center_work0, center_hat0);
        scratch.add_filter(center_work1, center_hat1);
        scratch.add_filter(sij[c], center_hat2);
        horizontal_spectral_filter_center_slab_many(
            scratch.filter_inputs, scratch.filter_outputs, params, slab, fft, width);

        const double weight = sym_component_weight(c);
        for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const std::size_t n = idx(params, i, j, k);
                    double velocity_product = (*vel_hat[2])[n] * (*vel_hat[2])[n];
                    if (c == 0) {
                        velocity_product = (*vel_hat[0])[n] * (*vel_hat[0])[n];
                    } else if (c == 1) {
                        velocity_product = (*vel_hat[0])[n] * (*vel_hat[1])[n];
                    } else if (c == 2) {
                        velocity_product = (*vel_hat[0])[n] * (*vel_hat[2])[n];
                    } else if (c == 3) {
                        velocity_product = (*vel_hat[1])[n] * (*vel_hat[1])[n];
                    } else if (c == 4) {
                        velocity_product = (*vel_hat[1])[n] * (*vel_hat[2])[n];
                    }
                    const double l = center_hat0[n] - velocity_product;
                    const double m = 2.0 * delta2 * (center_hat1[n] - ratio2 * strain_hat[n] * center_hat2[n]);
                    out.lm[n] += weight * l * m;
                    out.mm[n] += weight * m * m;
                }
            }
        }
    }

    return out;

}

// Slab-parallel version of smooth_amd_invariant_field: a local (1-2-1)^3
// average of an AMD numerator or denominator on cell centers, periodic
// horizontally and clamped/renormalized at the physical vertical boundaries.
// Owned planes must be filled before the call; halo planes are refreshed here.
void smooth_amd_invariant_slab(
    LocalField& q,
    int halo_tag,
    const Params& params,
    const Slab& slab,
    MPI_Comm comm) {
    exchange_neighbor_planes(q, slab.k_begin, slab.k_count, params.nz, halo_tag, params, slab, comm);
    const LocalField src = q;
    constexpr std::array<double, 3> stencil{0.25, 0.5, 0.25};
    for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
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

bool update_sgs_eddy_viscosity_slab(
    MpiLocalFlowState& state,
    const LocalVelocityGradients& grad,
    const LocalField& dudz_face,
    const LocalField& dvdz_face,
    const LocalField& dwdx_face,
    const LocalField& dwdy_face,
    LocalField& strain,
    const LocalField& w_center,
    LocalField& nu_t,
    LocalField& buoyancy_prime,
    LocalField& db_dx,
    LocalField& db_dy,
    LocalField& db_dz,
    LocalField& invariant_num,
    LocalField& invariant_den,
    const Params& params,
    const Slab& slab,
    FftwXY& fft,
    MPI_Comm comm,
    LasdGermanoScratch& germano_scratch,
    LasdFilterCache* filter_cache) {
    clear_center_slab_field(nu_t, params, slab);
    if (params.sgs_model == "none") {
        return false;
    }
    if (params.sgs_model == "smagorinsky") {
        smagorinsky_eddy_viscosity_slab(state, strain, nu_t, params, slab);
        return false;
    }
    if (params.sgs_model == "amd" || params.sgs_model == "amd_plane_dissipation") {
        clear_center_slab_field(buoyancy_prime, params, slab);
        clear_center_slab_field(db_dx, params, slab);
        clear_center_slab_field(db_dy, params, slab);
        clear_center_slab_field(db_dz, params, slab);
        if (params.thermo_enabled && params.amd_buoyancy_correction) {
            const double coefficient = params.g
                / (params.moisture_enabled
                        ? params.theta0 * (1.0 + 0.61 * params.qv0)
                        : params.theta0);
            for (int k = buoyancy_prime.plane_begin;
                 k < buoyancy_prime.plane_begin + buoyancy_prime.plane_count;
                 ++k) {
                double mean_theta_v = 0.0;
                for (int j = 0; j < params.ny; ++j) {
                    for (int i = 0; i < params.nx; ++i) {
                        const std::size_t n = idx(params, i, j, k);
                        const double theta_v = params.moisture_enabled
                            ? state.theta[n] * (1.0 + 0.61 * state.qv[n] - state.ql[n])
                            : state.theta[n];
                        buoyancy_prime[n] = theta_v;
                        mean_theta_v += theta_v;
                    }
                }
                mean_theta_v /= static_cast<double>(params.nx * params.ny);
                for (int j = 0; j < params.ny; ++j) {
                    for (int i = 0; i < params.nx; ++i) {
                        const std::size_t n = idx(params, i, j, k);
                        buoyancy_prime[n] = coefficient * (buoyancy_prime[n] - mean_theta_v);
                    }
                }
            }
            horizontal_derivatives_center_slab(
                buoyancy_prime, db_dx, db_dy, params, slab, fft);
            ddz_center_slab(buoyancy_prime, db_dz, params, slab);
        }
        const std::array<double, 3> length = amd_scaled_cell_width(params);
        invariant_num.clear_owned();
        invariant_den.clear_owned();
        for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const std::size_t n = idx(params, i, j, k);
                    const std::array<double, 9> velocity_gradient{
                        grad.dudx[n], grad.dudy[n], grad.dudz[n],
                        grad.dvdx[n], grad.dvdy[n], grad.dvdz[n],
                        grad.dwdx[n], grad.dwdy[n], grad.dwdz[n],
                    };
                    const std::size_t lower = z_face_idx(params, i, j, k);
                    const std::size_t upper = z_face_idx(params, i, j, k + 1);
                    const AmdInvariant invariant = amd_eddy_viscosity_staggered_invariant_at(
                        velocity_gradient,
                        {dudz_face[lower], dvdz_face[lower], dwdx_face[lower], dwdy_face[lower]},
                        {dudz_face[upper], dvdz_face[upper], dwdx_face[upper], dwdy_face[upper]},
                        {db_dx[n], db_dy[n], db_dz[n]},
                        length);
                    invariant_num[n] = invariant.numerator;
                    invariant_den[n] = invariant.denominator;
                }
            }
        }
        if (params.amd_invariant_averaging) {
            smooth_amd_invariant_slab(invariant_num, 930, params, slab, comm);
            smooth_amd_invariant_slab(invariant_den, 931, params, slab, comm);
        }
        for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const std::size_t n = idx(params, i, j, k);
                    nu_t[n] = amd_invariant_ratio(
                        AmdInvariant{invariant_num[n], invariant_den[n]});
                }
            }
        }
        if (params.amd_multiscale_averaging) {
            for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
                for (int j = 0; j < params.ny; ++j) {
                    for (int i = 0; i < params.nx; ++i) {
                        const std::size_t n = idx(params, i, j, k);
                        invariant_num[n] = std::max(invariant_num[n], 0.0);
                    }
                }
            }
            smooth_amd_invariant_slab(invariant_num, 934, params, slab, comm);
            smooth_amd_invariant_slab(invariant_den, 935, params, slab, comm);
            for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
                for (int j = 0; j < params.ny; ++j) {
                    for (int i = 0; i < params.nx; ++i) {
                        const std::size_t n = idx(params, i, j, k);
                        nu_t[n] = std::max(
                            nu_t[n],
                            amd_invariant_ratio(
                                AmdInvariant{invariant_num[n], invariant_den[n]}));
                    }
                }
            }
        }
        if (params.amd_dissipation_averaging) {
            for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
                for (int j = 0; j < params.ny; ++j) {
                    for (int i = 0; i < params.nx; ++i) {
                        const std::size_t n = idx(params, i, j, k);
                        const double strain_squared = strain[n] * strain[n];
                        invariant_num[n] = nu_t[n] * strain_squared;
                        invariant_den[n] = strain_squared;
                    }
                }
            }
            smooth_amd_invariant_slab(invariant_num, 932, params, slab, comm);
            smooth_amd_invariant_slab(invariant_den, 933, params, slab, comm);
            for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
                for (int j = 0; j < params.ny; ++j) {
                    for (int i = 0; i < params.nx; ++i) {
                        const std::size_t n = idx(params, i, j, k);
                        nu_t[n] = invariant_den[n] > 0.0
                            ? invariant_num[n] / invariant_den[n]
                            : 0.0;
                    }
                }
            }
        }
        if (params.sgs_model == "amd_plane_dissipation") {
            for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
                double plane_dissipation = 0.0;
                double plane_strain_squared = 0.0;
                for (int j = 0; j < params.ny; ++j) {
                    for (int i = 0; i < params.nx; ++i) {
                        const std::size_t n = idx(params, i, j, k);
                        const double s2 = strain[n] * strain[n];
                        plane_dissipation += nu_t[n] * s2;
                        plane_strain_squared += s2;
                    }
                }
                const double plane_viscosity = plane_strain_squared > 0.0
                    ? plane_dissipation / plane_strain_squared
                    : 0.0;
                for (int j = 0; j < params.ny; ++j) {
                    for (int i = 0; i < params.nx; ++i) {
                        nu_t[idx(params, i, j, k)] = plane_viscosity;
                    }
                }
            }
        }
        return false;
    }
    if (params.sgs_model != "lasd") {
        throw std::runtime_error("unsupported SGS model: " + params.sgs_model);
    }

    update_lagrangian_velocity_slab(state, w_center, params, slab, comm);
    const bool should_update = state.step_count > 0 && (state.step_count % params.cs_count) == 0;
    if (!should_update) {
        eddy_viscosity_from_cs2_slab(state.cs2, strain, nu_t, params, slab);
        return false;
    }

    exchange_neighbor_planes(strain, slab.k_begin, slab.k_count, params.nz, 730, params, slab, comm);
    const LocalSymFields sij = strain_components_slab(grad, params, slab);
    const ConstLocalVecFields vel = centered_velocity_slab(state.u, state.v, w_center);
    const LmMm two_delta =
        momentum_lm_mm_slab(
            state.u, state.v, state.w, vel, sij, strain, params, slab, fft,
            params.tfr, germano_scratch, filter_cache, comm);
    const LmMm four_delta = momentum_lm_mm_slab(
        state.u, state.v, state.w, vel, sij, strain, params, slab, fft,
        params.tfr * params.tfr, germano_scratch, filter_cache, comm);

    if (state.step_count == params.cs_count) {
        for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const std::size_t n = idx(params, i, j, k);
                    state.lm_old[n] = 0.03 * two_delta.mm[n];
                    state.mm_old[n] = two_delta.mm[n];
                    state.qn_old[n] = 0.03 * four_delta.mm[n];
                    state.nn_old[n] = four_delta.mm[n];
                }
            }
        }
    }
    apply_center_history_bc_slab(state.lm_old, params, slab);
    apply_center_history_bc_slab(state.mm_old, params, slab);
    apply_center_history_bc_slab(state.qn_old, params, slab);
    apply_center_history_bc_slab(state.nn_old, params, slab);
    exchange_neighbor_field_pack(
        {
            LocalHaloField{&state.lm_old, slab.k_begin, slab.k_count, params.nz},
            LocalHaloField{&state.mm_old, slab.k_begin, slab.k_count, params.nz},
            LocalHaloField{&state.qn_old, slab.k_begin, slab.k_count, params.nz},
            LocalHaloField{&state.nn_old, slab.k_begin, slab.k_count, params.nz},
        },
        740,
        params,
        slab,
        comm);

    lagrangian_average_fields_slab_into(
        two_delta.lm,
        two_delta.mm,
        state.lm_old,
        state.mm_old,
        state.u_lag,
        state.v_lag,
        state.w_lag,
        params,
        slab,
        germano_scratch.lag_avg0,
        germano_scratch.lag_avg1,
        germano_scratch.lag_interp0,
        germano_scratch.lag_interp1,
        germano_scratch.lag_interp_scratch);
    lagrangian_average_fields_slab_into(
        four_delta.lm,
        four_delta.mm,
        state.qn_old,
        state.nn_old,
        state.u_lag,
        state.v_lag,
        state.w_lag,
        params,
        slab,
        germano_scratch.lag_avg2,
        germano_scratch.lag_avg3,
        germano_scratch.lag_interp0,
        germano_scratch.lag_interp1,
        germano_scratch.lag_interp_scratch);

    const double exponent = std::log(params.tfr) / (std::log(params.tfr * params.tfr) - std::log(params.tfr));
    const double beta_min = 1.0 / (params.tfr * params.tfr * params.tfr);
    for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                const double cs2_2d = std::max(safe_divide(germano_scratch.lag_avg0[n], germano_scratch.lag_avg1[n]), 0.0);
                const double cs2_4d = std::max(safe_divide(germano_scratch.lag_avg2[n], germano_scratch.lag_avg3[n]), 0.0);
                double beta = std::pow(std::max(safe_divide(cs2_4d, cs2_2d), 0.0), exponent);
                beta = std::max(beta, beta_min);
                state.cs2[n] = std::clamp(safe_divide(cs2_2d, beta), 1.0e-6, 0.81);
            }
        }
    }
    copy_center_owned(germano_scratch.lag_avg0, state.lm_old, params, slab);
    copy_center_owned(germano_scratch.lag_avg1, state.mm_old, params, slab);
    copy_center_owned(germano_scratch.lag_avg2, state.qn_old, params, slab);
    copy_center_owned(germano_scratch.lag_avg3, state.nn_old, params, slab);
    eddy_viscosity_from_cs2_slab(state.cs2, strain, nu_t, params, slab);
    return true;
}

void add_sgs_momentum_forcing_slab(
    LocalField& rhs_u,
    LocalField& rhs_v,
    LocalField& rhs_w,
    const MpiLocalFlowState& state,
    const LocalVelocityGradients& grad,
    const LocalField& nu_t,
    const Params& params,
    const Slab& slab,
    FftwXY& fft,
    MPI_Comm comm,
    MpiSlabWorkspace& workspace,
    MpiTimingStats* timing) {
    if (params.sgs_model == "none") {
        return;
    }

    LocalField& txx = workspace.sgs_txx;
    LocalField& txy = workspace.sgs_txy;
    LocalField& tyy = workspace.sgs_tyy;
    LocalField& tzz = workspace.sgs_tzz;
    {
        MpiTimerScope scope(timing, MpiTimerId::sgs_center_stress_build);
        clear_center_slab_field(txx, params, slab);
        clear_center_slab_field(txy, params, slab);
        clear_center_slab_field(tyy, params, slab);
        clear_center_slab_field(tzz, params, slab);
        for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const std::size_t n = idx(params, i, j, k);
                    txx[n] = 2.0 * nu_t[n] * grad.dudx[n];
                    txy[n] = nu_t[n] * (grad.dudy[n] + grad.dvdx[n]);
                    tyy[n] = 2.0 * nu_t[n] * grad.dvdy[n];
                    tzz[n] = 2.0 * nu_t[n] * grad.dwdz[n];
                }
            }
        }
    }

    {
        MpiTimerScope scope(timing, MpiTimerId::sgs_stress_halo);
        exchange_neighbor_planes(tzz, slab.k_begin, slab.k_count, params.nz, 510, params, slab, comm);
    }

    LocalField& dwdx_face = workspace.sgs_dwdx_face;
    LocalField& dwdy_face = workspace.sgs_dwdy_face;
    LocalField& dudz_face = workspace.sgs_dudz_face;
    LocalField& dvdz_face = workspace.sgs_dvdz_face;
    LocalField& nu_t_face = workspace.sgs_nu_t_face;
    {
        MpiTimerScope scope(timing, MpiTimerId::sgs_face_derivatives);
        derivative_x_face_slab(state.w, dwdx_face, params, slab, fft);
        derivative_y_face_slab(state.w, dwdy_face, params, slab, fft);
        ddz_center_to_w_face_slab(state.u, dudz_face, params, slab);
        ddz_center_to_w_face_slab(state.v, dvdz_face, params, slab);
        center_to_w_face_slab(nu_t, nu_t_face, params, slab);
    }

    LocalField& txz = workspace.sgs_txz;
    LocalField& tyz = workspace.sgs_tyz;
    {
        MpiTimerScope scope(timing, MpiTimerId::sgs_face_stress_build);
        clear_face_slab_field(txz, params, slab);
        clear_face_slab_field(tyz, params, slab);
        for (int k = slab.face_begin; k < slab.face_begin + slab.face_count; ++k) {
            if (k <= 0 || k >= params.nz) {
                continue;
            }
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const std::size_t face = z_face_idx(params, i, j, k);
                    txz[face] = nu_t_face[face] * (dudz_face[face] + dwdx_face[face]);
                    tyz[face] = nu_t_face[face] * (dvdz_face[face] + dwdy_face[face]);
                }
            }
        }
    }

    {
        MpiTimerScope scope(timing, MpiTimerId::sgs_stress_halo);
        exchange_neighbor_field_pack(
            {
                LocalHaloField{&txz, slab.face_begin, slab.face_count, params.nz + 1},
                LocalHaloField{&tyz, slab.face_begin, slab.face_count, params.nz + 1},
            },
            520,
            params,
            slab,
            comm);
    }

    {
        MpiTimerScope scope(timing, MpiTimerId::sgs_center_divergence);
        LocalField& div_u_xy = workspace.sgs_div_u_xy;
        LocalField& div_v_xy = workspace.sgs_div_v_xy;
        LocalField& dtxz_dz = workspace.sgs_dtxz_dz;
        LocalField& dtyz_dz = workspace.sgs_dtyz_dz;
        horizontal_divergence_center_slab(txx, txy, div_u_xy, params, slab, fft);
        horizontal_divergence_center_slab(txy, tyy, div_v_xy, params, slab, fft);
        ddz_w_to_center_slab(txz, dtxz_dz, params, slab);
        ddz_w_to_center_slab(tyz, dtyz_dz, params, slab);

        for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const std::size_t n = idx(params, i, j, k);
                    rhs_u[n] += div_u_xy[n] + dtxz_dz[n];
                    rhs_v[n] += div_v_xy[n] + dtyz_dz[n];
                }
            }
        }
    }

    {
        MpiTimerScope scope(timing, MpiTimerId::sgs_face_divergence);
        LocalField& div_w_xy = workspace.sgs_div_w_xy;
        LocalField& dtzz_dz = workspace.sgs_dtzz_dz;
        horizontal_divergence_face_slab(txz, tyz, div_w_xy, params, slab, fft);
        ddz_center_to_w_face_slab(tzz, dtzz_dz, params, slab);
        for (int k = slab.face_begin; k < slab.face_begin + slab.face_count; ++k) {
            if (k <= 0 || k >= params.nz) {
                continue;
            }
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const std::size_t face = z_face_idx(params, i, j, k);
                    rhs_w[face] += div_w_xy[face] + dtzz_dz[face];
                }
            }
        }
    }
}

void apply_wall_stress_slab(
    LocalField& rhs_u,
    LocalField& rhs_v,
    const MpiLocalFlowState& state,
    const Params& params,
    const Slab& slab,
    FftwXY& fft) {
    if (params.momentum_wall_model != "abl" || slab.rank != 0) {
        return;
    }
    if (params.wall_ref_height() <= params.zo) {
        throw std::runtime_error("ABL wall stress requires wall_ref_height > zo");
    }
    const std::size_t plane_size = static_cast<std::size_t>(params.nx)
        * static_cast<std::size_t>(params.ny);
    Field u0(plane_size, 0.0);
    Field v0(plane_size, 0.0);
    if (params.wall_stress_model == "dynamic_neutral") {
        const double filter_width = params.fgr * params.tfr;
        fft.filter_plane(state.u.values, 0 - state.u.plane_begin, u0, params, filter_width);
        fft.filter_plane(state.v.values, 0 - state.v.plane_begin, v0, params, filter_width);
    } else {
        const std::size_t offset = static_cast<std::size_t>(-state.u.plane_begin) * plane_size;
        std::copy_n(state.u.values.begin() + static_cast<std::ptrdiff_t>(offset), plane_size, u0.begin());
        std::copy_n(state.v.values.begin() + static_cast<std::ptrdiff_t>(offset), plane_size, v0.begin());
    }
    const double denom = std::log(params.wall_ref_height() / params.zo);
    const double inv_dz = 1.0 / params.dz();
    constexpr double eps = 1.0e-12;
    double local_drag = 0.0;
    if (params.wall_stress_model == "prescribed_ustar_local") {
        const double inverse_plane = 1.0 / static_cast<double>(plane_size);
        double mean_speed_u = 0.0;
        double mean_speed_v = 0.0;
        for (std::size_t n = 0; n < plane_size; ++n) {
            const double speed = std::sqrt(u0[n] * u0[n] + v0[n] * v0[n]);
            mean_speed_u += speed * u0[n] * inverse_plane;
            mean_speed_v += speed * v0[n] * inverse_plane;
        }
        local_drag = prescribed_ustar_local_drag_coefficient(
            mean_speed_u, mean_speed_v, params.u_fric);
    }
    for (int j = 0; j < params.ny; ++j) {
        for (int i = 0; i < params.nx; ++i) {
            const std::size_t plane = static_cast<std::size_t>(j) * static_cast<std::size_t>(params.nx)
                + static_cast<std::size_t>(i);
            const double speed = std::sqrt(u0[plane] * u0[plane] + v0[plane] * v0[plane]);
            if (speed <= eps || std::abs(denom) <= eps) {
                continue;
            }
            const std::size_t cell = idx(params, i, j, 0);
            if (params.wall_stress_model == "prescribed_ustar_local") {
                rhs_u[cell] += -local_drag * speed * u0[plane] * inv_dz;
                rhs_v[cell] += -local_drag * speed * v0[plane] * inv_dz;
                continue;
            }
            double ustar = params.u_fric;
            if (params.wall_stress_model == "dynamic_neutral") {
                ustar = speed * params.vonk / denom;
            }
            const double tau = -(ustar * ustar);
            rhs_u[cell] += tau * u0[plane] / speed * inv_dz;
            rhs_v[cell] += tau * v0[plane] / speed * inv_dz;
        }
    }
}

double plane_mean(const LocalField& q, const Params& params, int k) {
    double mean = 0.0;
    for (int j = 0; j < params.ny; ++j) {
        for (int i = 0; i < params.nx; ++i) {
            mean += q[idx(params, i, j, k)];
        }
    }
    return mean / static_cast<double>(params.nx * params.ny);
}

double plane_mean_vertical_derivative(const LocalField& q, const Params& params, int k) {
    if (params.nz <= 1) {
        return 0.0;
    }
    if (k == 0) {
        return (plane_mean(q, params, 1) - plane_mean(q, params, 0)) / params.dz();
    }
    if (k == params.nz - 1) {
        return (plane_mean(q, params, k) - plane_mean(q, params, k - 1)) / params.dz();
    }
    return (plane_mean(q, params, k + 1) - plane_mean(q, params, k - 1)) / (2.0 * params.dz());
}

void add_bomex_large_scale_forcing_slab(
    LocalField& rhs_u,
    LocalField& rhs_v,
    LocalField& rhs_theta_l,
    LocalField& rhs_qt,
    const MpiLocalFlowState& state,
    const Params& params,
    const Slab& slab) {
    if (params.initial_condition != "bomex") {
        return;
    }
    for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
        const double z = (static_cast<double>(k) + 0.5) * params.dz();
        const double subsidence = bomex_subsidence(z);
        const double u_tendency = -subsidence * plane_mean_vertical_derivative(state.u, params, k);
        const double v_tendency = -subsidence * plane_mean_vertical_derivative(state.v, params, k);
        const double theta_tendency = -subsidence * plane_mean_vertical_derivative(state.theta_l, params, k)
            + bomex_radiative_tendency(z);
        const double qt_mean = plane_mean(state.qt, params, k);
        const double qt_specific = bomex_mixing_to_specific_humidity(qt_mean);
        const double specific_to_mixing_jacobian = 1.0 / std::pow(1.0 - qt_specific, 2.0);
        const double qt_tendency = -subsidence * plane_mean_vertical_derivative(state.qt, params, k)
            + bomex_moisture_advection_tendency(z) * specific_to_mixing_jacobian;
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                rhs_u[n] += u_tendency;
                rhs_v[n] += v_tendency;
                rhs_theta_l[n] += theta_tendency;
                rhs_qt[n] += qt_tendency;
            }
        }
    }
}

void add_buoyancy_slab(LocalField& rhs_w, const MpiLocalFlowState& state, const Params& params, const Slab& slab) {
    if (!params.thermo_enabled) {
        return;
    }
    const double coeff = params.g / params.theta0;
    for (int k = slab.face_begin; k < slab.face_begin + slab.face_count; ++k) {
        if (k <= 0 || k >= params.nz) {
            continue;
        }
        double mean_lower = 0.0;
        double mean_upper = 0.0;
        if (params.moisture_enabled) {
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const std::size_t lower = idx(params, i, j, k - 1);
                    const std::size_t upper = idx(params, i, j, k);
                    mean_lower += state.theta[lower] * (1.0 + 0.61 * state.qv[lower] - state.ql[lower]);
                    mean_upper += state.theta[upper] * (1.0 + 0.61 * state.qv[upper] - state.ql[upper]);
                }
            }
            const double inv_plane = 1.0 / static_cast<double>(params.nx * params.ny);
            mean_lower *= inv_plane;
            mean_upper *= inv_plane;
        } else {
            mean_lower = plane_mean(state.theta, params, k - 1);
            mean_upper = plane_mean(state.theta, params, k);
        }
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t lower = idx(params, i, j, k - 1);
                const std::size_t upper = idx(params, i, j, k);
                const double theta_lower = params.moisture_enabled
                    ? state.theta[lower] * (1.0 + 0.61 * state.qv[lower] - state.ql[lower]) - mean_lower
                    : state.theta[lower] - mean_lower;
                const double theta_upper = params.moisture_enabled
                    ? state.theta[upper] * (1.0 + 0.61 * state.qv[upper] - state.ql[upper]) - mean_upper
                    : state.theta[upper] - mean_upper;
                rhs_w[z_face_idx(params, i, j, k)] += 0.5 * coeff * (theta_lower + theta_upper);
            }
        }
    }
}

ScalarLmMm scalar_lm_mm_slab(
    const LocalField& u,
    const LocalField& v,
    const LocalField& w,
    const LocalField& theta,
    const LocalField& dtheta_dx,
    const LocalField& dtheta_dy,
    const LocalField& strain,
    const Params& params,
    const Slab& slab,
    FftwXY& fft,
    double test_ratio,
    LasdGermanoScratch& scratch,
    LasdFilterCache* filter_cache) {
    const double width = params.fgr * test_ratio;
    LasdFilterCacheSlot* cache_slot = filter_cache == nullptr ? nullptr : &filter_cache->slot_for(width);
    LocalField& u_hat_storage = scratch.scalar_u_hat_storage;
    LocalField& v_hat_storage = scratch.scalar_v_hat_storage;
    const LocalField* u_hat = nullptr;
    const LocalField* v_hat = nullptr;
    LocalField& w_center = scratch.scalar_theta_w;
    LocalField& dtheta_dz = scratch.scalar_dtheta_dz_face;
    w_to_center_slab(w, w_center, params, slab);
    ddz_center_slab(theta, dtheta_dz, params, slab);
    const LocalField* w_hat = nullptr;
    scratch.reset_filter_batch();
    if (cache_slot != nullptr && cache_slot->u_hat_valid) {
        u_hat = &cache_slot->u_hat;
    } else {
        u_hat = &u_hat_storage;
        scratch.add_filter(u, u_hat_storage);
    }
    if (cache_slot != nullptr && cache_slot->v_hat_valid) {
        v_hat = &cache_slot->v_hat;
    } else {
        v_hat = &v_hat_storage;
        scratch.add_filter(v, v_hat_storage);
    }
    if (cache_slot != nullptr && cache_slot->w_center_hat_valid) {
        w_hat = &cache_slot->w_center_hat;
    } else {
        LocalField& w_hat_storage = scratch.scalar_w_face_hat_storage;
        w_hat = &w_hat_storage;
        scratch.add_filter(w_center, w_hat_storage);
    }
    LocalField& theta_hat = scratch.scalar_theta_hat;
    scratch.add_filter(theta, theta_hat);
    horizontal_spectral_filter_center_slab_many(
        scratch.filter_inputs, scratch.filter_outputs, params, slab, fft, width);

    if (cache_slot == nullptr || !cache_slot->strain_hat_valid) {
        throw std::runtime_error(
            "scalar LASD requires the momentum LASD test-scale strain cache");
    }
    const LocalField& strain_hat = cache_slot->strain_hat;

    ScalarLmMm out;
    clear_center_slab_field(out.lm, params, slab);
    clear_center_slab_field(out.mm, params, slab);
    const double delta2 = params.sgs_delta() * params.sgs_delta();
    const double ratio2 = test_ratio * test_ratio;
    LocalField& center_work0 = scratch.scalar_center_work0;
    LocalField& center_work1 = scratch.scalar_center_work1;
    LocalField& center_hat0 = scratch.scalar_center_hat0;
    LocalField& center_hat1 = scratch.scalar_center_hat1;
    LocalField& gradient_hat = scratch.scalar_dx_hat;
    for (int component = 0; component < 3; ++component) {
        clear_center_slab_field(center_work0, params, slab);
        clear_center_slab_field(center_work1, params, slab);
        const LocalField& velocity = component == 0 ? u : (component == 1 ? v : w_center);
        const LocalField& gradient = component == 0 ? dtheta_dx : (component == 1 ? dtheta_dy : dtheta_dz);
        const LocalField* velocity_hat = component == 0 ? u_hat : (component == 1 ? v_hat : w_hat);
        for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const std::size_t n = idx(params, i, j, k);
                    center_work0[n] = velocity[n] * theta[n];
                    center_work1[n] = strain[n] * gradient[n];
                }
            }
        }
        scratch.reset_filter_batch();
        scratch.add_filter(center_work0, center_hat0);
        scratch.add_filter(center_work1, center_hat1);
        scratch.add_filter(gradient, gradient_hat);
        horizontal_spectral_filter_center_slab_many(
            scratch.filter_inputs, scratch.filter_outputs, params, slab, fft, width);
        for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const std::size_t n = idx(params, i, j, k);
                    const double l = center_hat0[n] - (*velocity_hat)[n] * theta_hat[n];
                    const double m = delta2 * (center_hat1[n] - ratio2 * std::max(strain_hat[n], 0.0) * gradient_hat[n]);
                    out.lm[n] += l * m;
                    out.mm[n] += m * m;
                }
            }
        }
    }

    return out;

}

void update_scalar_lasd_coefficients_slab(
    MpiLocalFlowState& state,
    const LocalField& transported_scalar,
    const LocalField& dtheta_dx,
    const LocalField& dtheta_dy,
    const LocalField& strain,
    LocalField& scalar_c,
    LocalField& scalar_lm_old,
    LocalField& scalar_mm_old,
    LocalField& scalar_qn_old,
    LocalField& scalar_nn_old,
    int halo_tag_base,
    const Params& params,
    const Slab& slab,
    FftwXY& fft,
    MPI_Comm comm,
    LasdGermanoScratch& germano_scratch,
    LasdFilterCache* filter_cache) {
    if (params.scalar_sgs_model != "lasd") {
        return;
    }
    const bool should_update = state.step_count > 0 && (state.step_count % params.cs_count) == 0;
    if (!should_update) {
        return;
    }

    const ScalarLmMm two_delta = scalar_lm_mm_slab(
        state.u,
        state.v,
        state.w,
        transported_scalar,
        dtheta_dx,
        dtheta_dy,
        strain,
        params,
        slab,
        fft,
        params.tfr,
        germano_scratch,
        filter_cache);
    const ScalarLmMm four_delta = scalar_lm_mm_slab(
        state.u,
        state.v,
        state.w,
        transported_scalar,
        dtheta_dx,
        dtheta_dy,
        strain,
        params,
        slab,
        fft,
        params.tfr * params.tfr,
        germano_scratch,
        filter_cache);

    if (state.step_count == params.cs_count) {
        for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const std::size_t n = idx(params, i, j, k);
                    scalar_lm_old[n] = 0.03 * two_delta.mm[n];
                    scalar_mm_old[n] = two_delta.mm[n];
                    scalar_qn_old[n] = 0.03 * four_delta.mm[n];
                    scalar_nn_old[n] = four_delta.mm[n];
                }
            }
        }
    }
    apply_center_history_bc_slab(scalar_lm_old, params, slab);
    apply_center_history_bc_slab(scalar_mm_old, params, slab);
    apply_center_history_bc_slab(scalar_qn_old, params, slab);
    apply_center_history_bc_slab(scalar_nn_old, params, slab);
    exchange_neighbor_field_pack(
        {
            LocalHaloField{&scalar_lm_old, slab.k_begin, slab.k_count, params.nz},
            LocalHaloField{&scalar_mm_old, slab.k_begin, slab.k_count, params.nz},
            LocalHaloField{&scalar_qn_old, slab.k_begin, slab.k_count, params.nz},
            LocalHaloField{&scalar_nn_old, slab.k_begin, slab.k_count, params.nz},
        },
        halo_tag_base,
        params,
        slab,
        comm);

    // Scalar-history magnitudes carry scalar units.  The momentum histories
    // provide the common SGS turnover time and keep the scalar closure
    // invariant under a change of scalar units.
    lagrangian_average_fields_slab_into(
        two_delta.lm,
        two_delta.mm,
        scalar_lm_old,
        scalar_mm_old,
        state.u_lag,
        state.v_lag,
        state.w_lag,
        params,
        slab,
        germano_scratch.lag_avg0,
        germano_scratch.lag_avg1,
        germano_scratch.lag_interp0,
        germano_scratch.lag_interp1,
        germano_scratch.lag_interp_scratch,
        &state.lm_old,
        &state.mm_old);
    lagrangian_average_fields_slab_into(
        four_delta.lm,
        four_delta.mm,
        scalar_qn_old,
        scalar_nn_old,
        state.u_lag,
        state.v_lag,
        state.w_lag,
        params,
        slab,
        germano_scratch.lag_avg2,
        germano_scratch.lag_avg3,
        germano_scratch.lag_interp0,
        germano_scratch.lag_interp1,
        germano_scratch.lag_interp_scratch,
        &state.qn_old,
        &state.nn_old);

    const double exponent = std::log(params.tfr) / (std::log(params.tfr * params.tfr) - std::log(params.tfr));
    const double beta_min = 1.0 / (params.tfr * params.tfr * params.tfr);
    for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                const double c_2d = std::max(safe_divide(germano_scratch.lag_avg0[n], germano_scratch.lag_avg1[n]), 0.0);
                const double c_4d = std::max(safe_divide(germano_scratch.lag_avg2[n], germano_scratch.lag_avg3[n]), 0.0);
                double beta = std::pow(std::max(safe_divide(c_4d, c_2d), 0.0), exponent);
                beta = std::max(beta, beta_min);
                scalar_c[n] = std::clamp(safe_divide(c_2d, beta), params.scalar_lasd_min, params.scalar_lasd_max);
            }
        }
    }
    copy_center_owned(germano_scratch.lag_avg0, scalar_lm_old, params, slab);
    copy_center_owned(germano_scratch.lag_avg1, scalar_mm_old, params, slab);
    copy_center_owned(germano_scratch.lag_avg2, scalar_qn_old, params, slab);
    copy_center_owned(germano_scratch.lag_avg3, scalar_nn_old, params, slab);
}

void scalar_eddy_diffusivity_slab(
    const MpiLocalFlowState& state,
    const LocalVelocityGradients& velocity_gradient,
    const LocalField& dscalar_dx,
    const LocalField& dscalar_dy,
    const LocalField& dscalar_dz,
    const LocalField& dscalar_dz_w,
    const LocalField* paired_dscalar_dx,
    const LocalField* paired_dscalar_dy,
    const LocalField* paired_dscalar_dz,
    const LocalField* paired_dscalar_dz_w,
    const LocalField* dudz_face,
    const LocalField* dvdz_face,
    const LocalField* dwdx_face,
    const LocalField* dwdy_face,
    const LocalField& eddy_viscosity,
    const LocalField& strain,
    const LocalField& scalar_c,
    double molecular_diffusivity,
    double turbulent_ratio,
    const Params& params,
    const Slab& slab,
    MPI_Comm comm,
    int amd_halo_tag_base,
    LocalField& kappa) {
    clear_center_slab_field(kappa, params, slab);
    LocalField dtheta_dz;
    if (params.scalar_stability_correction) {
        dtheta_dz.resize(slab.k_begin, slab.k_count, params.nz, params);
        if (params.moisture_enabled) {
            LocalField theta_v;
            theta_v.resize(slab.k_begin, slab.k_count, params.nz, params);
            for (int k = theta_v.plane_begin; k < theta_v.plane_begin + theta_v.plane_count; ++k) {
                for (int j = 0; j < params.ny; ++j) {
                    for (int i = 0; i < params.nx; ++i) {
                        const std::size_t n = idx(params, i, j, k);
                        theta_v[n] = state.theta[n] * (1.0 + 0.61 * state.qv[n] - state.ql[n]);
                    }
                }
            }
            ddz_center_slab(theta_v, dtheta_dz, params, slab);
        } else {
            ddz_center_slab(state.theta, dtheta_dz, params, slab);
        }
    }
    const double strain_coeff = std::pow(params.smagorinsky_cs * params.sgs_delta(), 2.0);
    const std::array<double, 3> amd_length = amd_scaled_cell_width(params);
    const bool amd_scalar_model = params.scalar_sgs_model == "amd"
        || params.scalar_sgs_model == "amd_shared"
        || params.scalar_sgs_model == "amd_plane_dissipation";
    LocalField amd_num;
    LocalField amd_den;
    LocalField paired_amd_num;
    LocalField paired_amd_den;
    if (amd_scalar_model) {
        amd_num.resize(slab.k_begin, slab.k_count, params.nz, params);
        amd_den.resize(slab.k_begin, slab.k_count, params.nz, params);
        if (params.scalar_sgs_model == "amd_shared") {
            if (paired_dscalar_dx == nullptr || paired_dscalar_dy == nullptr
                || paired_dscalar_dz == nullptr || paired_dscalar_dz_w == nullptr) {
                throw std::runtime_error("shared scalar AMD requires paired scalar gradients");
            }
            paired_amd_num.resize(slab.k_begin, slab.k_count, params.nz, params);
            paired_amd_den.resize(slab.k_begin, slab.k_count, params.nz, params);
        }
        if (params.scalar_amd_face_products
            && (dudz_face == nullptr || dvdz_face == nullptr
                || dwdx_face == nullptr || dwdy_face == nullptr)) {
            throw std::runtime_error("face-product scalar AMD requires face velocity gradients");
        }
        for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
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
                            {(*dudz_face)[lower], (*dvdz_face)[lower],
                             (*dwdx_face)[lower], (*dwdy_face)[lower]},
                            {(*dudz_face)[upper], (*dvdz_face)[upper],
                             (*dwdx_face)[upper], (*dwdy_face)[upper]},
                            dscalar_dx[n], dscalar_dy[n],
                            dscalar_dz_w[lower], dscalar_dz_w[upper],
                            amd_length)
                        : amd_scalar_diffusivity_staggered_invariant_at(
                            gradient,
                            {dscalar_dx[n], dscalar_dy[n], dscalar_dz[n]},
                            dscalar_dz_w[lower], dscalar_dz_w[upper], amd_length);
                    amd_num[n] = invariant.numerator;
                    amd_den[n] = invariant.denominator;
                    if (params.scalar_sgs_model == "amd_shared") {
                        const AmdInvariant paired_invariant =
                            amd_scalar_diffusivity_staggered_invariant_at(
                                gradient,
                                {(*paired_dscalar_dx)[n], (*paired_dscalar_dy)[n], (*paired_dscalar_dz)[n]},
                                (*paired_dscalar_dz_w)[lower], (*paired_dscalar_dz_w)[upper], amd_length);
                        paired_amd_num[n] = paired_invariant.numerator;
                        paired_amd_den[n] = paired_invariant.denominator;
                    }
                }
            }
        }
        if (params.scalar_amd_invariant_averaging) {
            smooth_amd_invariant_slab(amd_num, amd_halo_tag_base, params, slab, comm);
            smooth_amd_invariant_slab(amd_den, amd_halo_tag_base + 1, params, slab, comm);
            if (params.scalar_sgs_model == "amd_shared") {
                smooth_amd_invariant_slab(paired_amd_num, amd_halo_tag_base + 2, params, slab, comm);
                smooth_amd_invariant_slab(paired_amd_den, amd_halo_tag_base + 3, params, slab, comm);
            }
        }
    }
    for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                double diffusivity = molecular_diffusivity;
                if (amd_scalar_model) {
                    if (params.scalar_sgs_model == "amd_shared") {
                        diffusivity += std::max(
                            amd_invariant_ratio(AmdInvariant{amd_num[n], amd_den[n]}),
                            amd_invariant_ratio(
                                AmdInvariant{paired_amd_num[n], paired_amd_den[n]}));
                    } else {
                        diffusivity += amd_invariant_ratio(AmdInvariant{amd_num[n], amd_den[n]});
                    }
                } else if (params.scalar_sgs_model == "lasd") {
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
                    const double n2 = (params.g / params.theta0) * dtheta_dz[n];
                    const double ri = std::max(n2, 0.0) / std::max(local_strain * local_strain, 1.0e-24);
                    diffusivity *= std::pow(1.0 + params.scalar_stability_beta * ri, -params.scalar_stability_power);
                }
                kappa[n] = diffusivity;
            }
        }
    }
    if (params.scalar_sgs_model == "amd_plane_dissipation") {
        for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
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
                        std::max(kappa[n] - molecular_diffusivity, 0.0) * gradient_energy;
                }
            }
            const double plane_sgs_diffusivity = plane_gradient_energy > 0.0
                ? plane_sgs_dissipation / plane_gradient_energy
                : 0.0;
            const double plane_diffusivity = molecular_diffusivity + plane_sgs_diffusivity;
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    kappa[idx(params, i, j, k)] = plane_diffusivity;
                }
            }
        }
    }
}

void scalar_rhs_slab(
    MpiLocalFlowState& state,
    const LocalField& transported_scalar,
    const LocalField* paired_scalar,
    const LocalVelocityGradients& velocity_gradient,
    const LocalField& eddy_viscosity,
    const LocalField& strain,
    double molecular_diffusivity,
    double turbulent_ratio,
    double surface_flux,
    LocalField& scalar_c,
    LocalField& scalar_lm_old,
    LocalField& scalar_mm_old,
    LocalField& scalar_qn_old,
    LocalField& scalar_nn_old,
    int lasd_halo_tag_base,
    LocalField& kappa_center,
    const Params& params,
    const Slab& slab,
    FftwXY& fft,
    MPI_Comm comm,
    LocalField& rhs,
    MpiSlabWorkspace& workspace,
    MpiTimingStats* timing,
    LasdFilterCache* filter_cache) {
    clear_center_slab_field(rhs, params, slab);
    if (!params.thermo_enabled) {
        return;
    }

    LocalField& theta_flux_x = workspace.scalar_theta_flux_x;
    LocalField& theta_flux_y = workspace.scalar_theta_flux_y;
    LocalField& theta_flux_z = workspace.scalar_theta_flux_z;
    LocalField& theta_on_w = workspace.scalar_theta_on_w;
    {
        MpiTimerScope scope(timing, MpiTimerId::scalar_advective_flux);
        clear_center_slab_field(theta_flux_x, params, slab);
        clear_center_slab_field(theta_flux_y, params, slab);
        clear_face_slab_field(theta_flux_z, params, slab);
        center_to_w_face_slab(transported_scalar, theta_on_w, params, slab);
        for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const std::size_t n = idx(params, i, j, k);
                    theta_flux_x[n] = state.u[n] * transported_scalar[n];
                    theta_flux_y[n] = state.v[n] * transported_scalar[n];
                }
            }
        }
        for (int k = slab.face_begin; k < slab.face_begin + slab.face_count; ++k) {
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const std::size_t face = z_face_idx(params, i, j, k);
                    theta_flux_z[face] = state.w[face] * theta_on_w[face];
                }
            }
        }
        if (params.horizontal_dealias && params.dealiasing == "sharp") {
            constexpr double two_thirds_filter_width = 1.5;
            const int local_begin = slab.k_begin - theta_flux_x.plane_begin;
            Field filtered_x;
            Field filtered_y;
            fft.filter_plane_range_fortran_sharp(
                theta_flux_x.values,
                local_begin,
                slab.k_count,
                filtered_x,
                params,
                two_thirds_filter_width);
            fft.filter_plane_range_fortran_sharp(
                theta_flux_y.values,
                local_begin,
                slab.k_count,
                filtered_y,
                params,
                two_thirds_filter_width);
            const std::size_t plane_size = static_cast<std::size_t>(params.nx * params.ny);
            for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
                const std::size_t offset = static_cast<std::size_t>(k - theta_flux_x.plane_begin) * plane_size;
                std::copy_n(
                    filtered_x.begin() + static_cast<std::ptrdiff_t>(offset),
                    plane_size,
                    theta_flux_x.values.begin() + static_cast<std::ptrdiff_t>(offset));
                std::copy_n(
                    filtered_y.begin() + static_cast<std::ptrdiff_t>(offset),
                    plane_size,
                    theta_flux_y.values.begin() + static_cast<std::ptrdiff_t>(offset));
            }
        }
        exchange_neighbor_planes(theta_flux_z, slab.face_begin, slab.face_count, params.nz + 1, 550, params, slab, comm);
    }

    {
        MpiTimerScope scope(timing, MpiTimerId::scalar_advective_divergence);
        LocalField& div_adv_xy = workspace.scalar_div_adv_xy;
        LocalField& div_adv_z = workspace.scalar_div_adv_z;
        if (params.horizontal_dealias && params.dealiasing == "padding_3_2") {
            const int local_begin = slab.k_begin - transported_scalar.plane_begin;
            fft.horizontal_flux_divergence_3_2(
                state.u.values,
                state.v.values,
                transported_scalar.values,
                local_begin,
                slab.k_count,
                div_adv_xy.values,
                params);
        } else {
            horizontal_divergence_center_slab(theta_flux_x, theta_flux_y, div_adv_xy, params, slab, fft);
        }
        ddz_w_to_center_slab(theta_flux_z, div_adv_z, params, slab);
        for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const std::size_t n = idx(params, i, j, k);
                    rhs[n] = -(div_adv_xy[n] + div_adv_z[n]);
                }
            }
        }
    }

    LocalField& dtheta_dx = workspace.scalar_dtheta_dx;
    LocalField& dtheta_dy = workspace.scalar_dtheta_dy;
    LocalField& dtheta_dz_center = workspace.scalar_dtheta_dz_center;
    LocalField& dtheta_dz_w = workspace.scalar_dtheta_dz_w;
    LocalField paired_dtheta_dx;
    LocalField paired_dtheta_dy;
    LocalField paired_dtheta_dz_center;
    LocalField paired_dtheta_dz_w;
    LocalField face_dudz;
    LocalField face_dvdz;
    LocalField face_dwdx;
    LocalField face_dwdy;
    const bool build_face_products = params.scalar_amd_face_products
        && (params.scalar_sgs_model == "amd"
            || params.scalar_sgs_model == "amd_shared"
            || params.scalar_sgs_model == "amd_plane_dissipation");
    if (build_face_products) {
        // Recompute every center-bounding face gradient with the same
        // face-range operators on every rank.  Reusing the momentum-side
        // workspace faces would mix values produced by the plane-batched and
        // range-batched FFT paths, which agree only to round-off and would
        // break the byte-identical n1/n4 guarantee.
        const int face_begin = needed_face_begin(slab);
        const int face_count = needed_face_count(params, slab);
        face_dudz.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        face_dvdz.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        face_dwdx.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        face_dwdy.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        derivative_x_face_range(state.w, face_begin, face_count, face_dwdx, params, fft);
        derivative_y_face_range(state.w, face_begin, face_count, face_dwdy, params, fft);
        ddz_center_to_w_face_range(state.u, face_begin, face_count, face_dudz, params);
        ddz_center_to_w_face_range(state.v, face_begin, face_count, face_dvdz, params);
        if (params.amd_wall_model_gradients && slab.k_begin == 0) {
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const std::size_t center = idx(params, i, j, 0);
                    const std::size_t face = z_face_idx(params, i, j, 0);
                    face_dudz[face] = velocity_gradient.dudz[center];
                    face_dvdz[face] = velocity_gradient.dvdz[center];
                }
            }
        }
    }
    {
        MpiTimerScope scope(timing, MpiTimerId::scalar_gradients);
        derivative_x_center_slab(transported_scalar, dtheta_dx, params, slab, fft);
        derivative_y_center_slab(transported_scalar, dtheta_dy, params, slab, fft);
        ddz_center_slab(transported_scalar, dtheta_dz_center, params, slab);
        if (params.scalar_sgs_model == "amd"
            || params.scalar_sgs_model == "amd_shared"
            || params.scalar_sgs_model == "amd_plane_dissipation") {
            ddz_center_to_w_face_range(
                transported_scalar,
                needed_face_begin(slab),
                needed_face_count(params, slab),
                dtheta_dz_w,
                params);
        } else {
            ddz_center_to_w_face_slab(transported_scalar, dtheta_dz_w, params, slab);
        }
        if (params.scalar_sgs_model == "amd_shared") {
            if (paired_scalar == nullptr) {
                throw std::runtime_error("shared scalar AMD requires a paired conserved scalar");
            }
            paired_dtheta_dx.resize(slab.k_begin, slab.k_count, params.nz, params);
            paired_dtheta_dy.resize(slab.k_begin, slab.k_count, params.nz, params);
            paired_dtheta_dz_center.resize(slab.k_begin, slab.k_count, params.nz, params);
            derivative_x_center_slab(*paired_scalar, paired_dtheta_dx, params, slab, fft);
            derivative_y_center_slab(*paired_scalar, paired_dtheta_dy, params, slab, fft);
            ddz_center_slab(*paired_scalar, paired_dtheta_dz_center, params, slab);
            ddz_center_to_w_face_range(
                *paired_scalar,
                needed_face_begin(slab),
                needed_face_count(params, slab),
                paired_dtheta_dz_w,
                params);
        }
    }

    {
        MpiTimerScope scope(timing, MpiTimerId::scalar_lasd_update);
        update_scalar_lasd_coefficients_slab(
            state,
            transported_scalar,
            dtheta_dx,
            dtheta_dy,
            strain,
            scalar_c,
            scalar_lm_old,
            scalar_mm_old,
            scalar_qn_old,
            scalar_nn_old,
            lasd_halo_tag_base,
            params,
            slab,
            fft,
            comm,
            workspace.lasd_germano_scratch,
            filter_cache);
    }

    LocalField& kappa_w = workspace.scalar_kappa_w;
    {
        MpiTimerScope scope(timing, MpiTimerId::scalar_diffusivity_halo);
        scalar_eddy_diffusivity_slab(
            state,
            velocity_gradient,
            dtheta_dx,
            dtheta_dy,
            dtheta_dz_center,
            dtheta_dz_w,
            params.scalar_sgs_model == "amd_shared" ? &paired_dtheta_dx : nullptr,
            params.scalar_sgs_model == "amd_shared" ? &paired_dtheta_dy : nullptr,
            params.scalar_sgs_model == "amd_shared" ? &paired_dtheta_dz_center : nullptr,
            params.scalar_sgs_model == "amd_shared" ? &paired_dtheta_dz_w : nullptr,
            build_face_products ? &face_dudz : nullptr,
            build_face_products ? &face_dvdz : nullptr,
            build_face_products ? &face_dwdx : nullptr,
            build_face_products ? &face_dwdy : nullptr,
            eddy_viscosity,
            strain,
            scalar_c,
            molecular_diffusivity,
            turbulent_ratio,
            params,
            slab,
            comm,
            lasd_halo_tag_base + 7,
            kappa_center);
        exchange_neighbor_planes(kappa_center, slab.k_begin, slab.k_count, params.nz, 560, params, slab, comm);
        center_to_w_face_slab(kappa_center, kappa_w, params, slab);
    }

    LocalField& qx = workspace.scalar_qx;
    LocalField& qy = workspace.scalar_qy;
    LocalField& qz = workspace.scalar_qz;
    {
        MpiTimerScope scope(timing, MpiTimerId::scalar_diffusive_flux);
        clear_center_slab_field(qx, params, slab);
        clear_center_slab_field(qy, params, slab);
        clear_face_slab_field(qz, params, slab);
        for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const std::size_t n = idx(params, i, j, k);
                    qx[n] = -kappa_center[n] * dtheta_dx[n];
                    qy[n] = -kappa_center[n] * dtheta_dy[n];
                }
            }
        }

        for (int k = slab.face_begin; k < slab.face_begin + slab.face_count; ++k) {
            if (k <= 0 || k >= params.nz) {
                continue;
            }
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const std::size_t face = z_face_idx(params, i, j, k);
                    qz[face] = -kappa_w[face] * dtheta_dz_w[face];
                }
            }
        }
        if (surface_flux != 0.0 && slab.rank == 0) {
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    qz[z_face_idx(params, i, j, 0)] = surface_flux;
                }
            }
        }
        exchange_neighbor_planes(qz, slab.face_begin, slab.face_count, params.nz + 1, 570, params, slab, comm);
    }

    {
        MpiTimerScope scope(timing, MpiTimerId::scalar_diffusive_divergence);
        LocalField& div_qxy = workspace.scalar_div_qxy;
        LocalField& div_qz = workspace.scalar_div_qz;
        horizontal_divergence_center_slab(qx, qy, div_qxy, params, slab, fft);
        ddz_w_to_center_slab(qz, div_qz, params, slab);
        for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const std::size_t n = idx(params, i, j, k);
                    rhs[n] -= div_qxy[n] + div_qz[n];
                }
            }
        }
    }
}

void compute_rhs_slab(
    MpiLocalFlowState& state,
    const Params& params,
    const Slab& slab,
    FftwXY& fft,
    MPI_Comm comm,
    MpiSlabWorkspace& workspace,
    MpiTimingStats* timing) {
    MpiTimerScope rhs_scope(timing, MpiTimerId::rhs_total);
    require_distributed_rhs_supported(params);

    workspace.resize(params, slab);
    workspace.lasd_filter_cache.reset_for_step(state.step_count);
    LocalField& rhs_u = workspace.rhs_u;
    LocalField& rhs_v = workspace.rhs_v;
    LocalField& rhs_w = workspace.rhs_w;
    LocalVelocityGradients& grad = workspace.grad;

    {
        MpiTimerScope scope(timing, MpiTimerId::rhs_derivatives);
        horizontal_derivatives_laplacian_center_slab(
            state.u, grad.dudx, grad.dudy, workspace.lap_u, params, slab, fft);
        horizontal_derivatives_laplacian_center_slab(
            state.v, grad.dvdx, grad.dvdy, workspace.lap_v, params, slab, fft);
        ddz_center_slab(state.u, grad.dudz, params, slab);
        ddz_center_slab(state.v, grad.dvdz, params, slab);
        w_to_center_slab(state.w, workspace.w_center, params, slab);
        horizontal_derivatives_center_slab(workspace.w_center, grad.dwdx, grad.dwdy, params, slab, fft);
        ddz_w_to_center_slab(state.w, grad.dwdz, params, slab);

        horizontal_derivatives_laplacian_face_slab(
            state.w, workspace.dwdx_face, workspace.dwdy_face, workspace.lap_w, params, slab, fft);
        ddz_w_face_slab(state.w, workspace.dwdz_face, params, slab);
        center_to_w_face_slab(state.u, workspace.u_on_w, params, slab);
        center_to_w_face_slab(state.v, workspace.v_on_w, params, slab);
        if (params.sgs_model == "amd" || params.sgs_model == "amd_plane_dissipation") {
            // AMD quadratic products must be formed on the natural w-face
            // locations before being interpolated to viscosity centers.  A
            // rank's last center needs the first face owned by the next rank,
            // so evaluate the complete center-bounding range from state halos.
            const int face_begin = needed_face_begin(slab);
            const int face_count = needed_face_count(params, slab);
            derivative_x_face_range(
                state.w, face_begin, face_count, workspace.sgs_dwdx_face, params, fft);
            derivative_y_face_range(
                state.w, face_begin, face_count, workspace.sgs_dwdy_face, params, fft);
            ddz_center_to_w_face_range(
                state.u, face_begin, face_count, workspace.sgs_dudz_face, params);
            ddz_center_to_w_face_range(
                state.v, face_begin, face_count, workspace.sgs_dvdz_face, params);
            apply_amd_wall_model_gradients_slab(
                state,
                grad,
                &workspace.sgs_dudz_face,
                &workspace.sgs_dvdz_face,
                params,
                slab);
        }
        if (params.momentum_advection_form == "skew_symmetric"
            || params.momentum_advection_form == "rotational") {
            ddz_center_to_w_face_slab(state.u, workspace.momentum_dudz_face, params, slab);
            ddz_center_to_w_face_slab(state.v, workspace.momentum_dvdz_face, params, slab);
        }
        if (params.momentum_advection_form == "rotational") {
            // NCAR vector-invariant staggering: omega_x and omega_y live on
            // w faces, while omega_z lives at u/v centers.  Store the paired
            // w*omega_y and w*omega_x products in the existing face scratch.
            clear_face_slab_field(workspace.momentum_flux_u_z, params, slab);
            clear_face_slab_field(workspace.momentum_flux_v_z, params, slab);
            for (int k = slab.face_begin; k < slab.face_begin + slab.face_count; ++k) {
                if (k <= 0 || k >= params.nz) continue;
                for (int j = 0; j < params.ny; ++j) {
                    for (int i = 0; i < params.nx; ++i) {
                        const std::size_t face = z_face_idx(params, i, j, k);
                        const double omega_x = workspace.dwdy_face[face] - workspace.momentum_dvdz_face[face];
                        const double omega_y = workspace.momentum_dudz_face[face] - workspace.dwdx_face[face];
                        workspace.momentum_flux_u_z[face] = state.w[face] * omega_y;
                        workspace.momentum_flux_v_z[face] = state.w[face] * omega_x;
                    }
                }
            }
            exchange_neighbor_field_pack(
                {
                    LocalHaloField{&workspace.momentum_flux_u_z, slab.face_begin, slab.face_count, params.nz + 1},
                    LocalHaloField{&workspace.momentum_flux_v_z, slab.face_begin, slab.face_count, params.nz + 1},
                },
                493,
                params,
                slab,
                comm);
            w_to_center_slab(workspace.momentum_flux_u_z, workspace.momentum_div_u_z, params, slab);
            w_to_center_slab(workspace.momentum_flux_v_z, workspace.momentum_div_v_z, params, slab);
        }
        if (params.momentum_advection_form == "skew_symmetric") {
            clear_face_slab_field(workspace.momentum_flux_u_z, params, slab);
            clear_face_slab_field(workspace.momentum_flux_v_z, params, slab);
            for (int k = slab.face_begin; k < slab.face_begin + slab.face_count; ++k) {
                for (int j = 0; j < params.ny; ++j) {
                    for (int i = 0; i < params.nx; ++i) {
                        const std::size_t face = z_face_idx(params, i, j, k);
                        workspace.momentum_flux_u_z[face] = state.w[face] * workspace.u_on_w[face];
                        workspace.momentum_flux_v_z[face] = state.w[face] * workspace.v_on_w[face];
                    }
                }
            }
            exchange_neighbor_field_pack(
                {
                    LocalHaloField{&workspace.momentum_dudz_face, slab.face_begin, slab.face_count, params.nz + 1},
                    LocalHaloField{&workspace.momentum_dvdz_face, slab.face_begin, slab.face_count, params.nz + 1},
                    LocalHaloField{&workspace.momentum_flux_u_z, slab.face_begin, slab.face_count, params.nz + 1},
                    LocalHaloField{&workspace.momentum_flux_v_z, slab.face_begin, slab.face_count, params.nz + 1},
                },
                495,
                params,
                slab,
                comm);
            ddz_w_to_center_slab(workspace.momentum_flux_u_z, workspace.momentum_div_u_z, params, slab);
            ddz_w_to_center_slab(workspace.momentum_flux_v_z, workspace.momentum_div_v_z, params, slab);
            clear_center_slab_field(workspace.momentum_flux_w_z, params, slab);
            clear_center_slab_field(workspace.momentum_adv_w_z, params, slab);
            for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
                for (int j = 0; j < params.ny; ++j) {
                    for (int i = 0; i < params.nx; ++i) {
                        const std::size_t n = idx(params, i, j, k);
                        workspace.momentum_flux_w_z[n] = workspace.w_center[n] * workspace.w_center[n];
                        workspace.momentum_adv_w_z[n] = workspace.w_center[n] * grad.dwdz[n];
                    }
                }
            }
            exchange_neighbor_field_pack(
                {
                    LocalHaloField{&workspace.momentum_flux_w_z, slab.k_begin, slab.k_count, params.nz},
                    LocalHaloField{&workspace.momentum_adv_w_z, slab.k_begin, slab.k_count, params.nz},
                },
                496,
                params,
                slab,
                comm);
            ddz_center_to_w_face_slab(workspace.momentum_flux_w_z, workspace.momentum_div_w_z, params, slab);
        }
    }

    {
        MpiTimerScope scope(timing, MpiTimerId::rhs_momentum);
        clear_center_slab_field(rhs_u, params, slab);
        clear_center_slab_field(rhs_v, params, slab);
        clear_face_slab_field(rhs_w, params, slab);
        Field horizontal_adv_u;
        Field horizontal_adv_v;
        Field horizontal_adv_w;
        Field horizontal_conservative_u;
        Field horizontal_conservative_v;
        Field horizontal_conservative_w;
        const int center_begin = slab.k_begin - state.u.plane_begin;
        const int face_begin = slab.face_begin - state.w.plane_begin;
        if (params.horizontal_dealias && params.dealiasing == "padding_3_2") {
            fft.horizontal_advective_derivative_3_2(
                state.u.values,
                state.v.values,
                state.u.values,
                center_begin,
                slab.k_count,
                horizontal_adv_u,
                params);
            fft.horizontal_advective_derivative_3_2(
                state.u.values,
                state.v.values,
                state.v.values,
                center_begin,
                slab.k_count,
                horizontal_adv_v,
                params);
            fft.horizontal_advective_derivative_3_2(
                workspace.u_on_w.values,
                workspace.v_on_w.values,
                state.w.values,
                face_begin,
                slab.face_count,
                horizontal_adv_w,
                params);
            if (params.momentum_advection_form == "skew_symmetric") {
                fft.horizontal_flux_divergence_3_2(
                    state.u.values,
                    state.v.values,
                    state.u.values,
                    center_begin,
                    slab.k_count,
                    horizontal_conservative_u,
                    params);
                fft.horizontal_flux_divergence_3_2(
                    state.u.values,
                    state.v.values,
                    state.v.values,
                    center_begin,
                    slab.k_count,
                    horizontal_conservative_v,
                    params);
                fft.horizontal_flux_divergence_3_2(
                    workspace.u_on_w.values,
                    workspace.v_on_w.values,
                    state.w.values,
                    face_begin,
                    slab.face_count,
                    horizontal_conservative_w,
                    params);
            }
        } else if (params.momentum_advection_form == "skew_symmetric") {
            Field flux_x(state.u.values.size(), 0.0);
            Field flux_y(state.u.values.size(), 0.0);
            for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
                for (int j = 0; j < params.ny; ++j) {
                    for (int i = 0; i < params.nx; ++i) {
                        const std::size_t n = idx(params, i, j, k);
                        const std::size_t local = state.u.local_offset_from_global_flat(n);
                        flux_x[local] = state.u[n] * state.u[n];
                        flux_y[local] = state.v[n] * state.u[n];
                    }
                }
            }
            fft.horizontal_divergence_plane_range(
                flux_x, flux_y, center_begin, slab.k_count, horizontal_conservative_u, params);
            std::fill(flux_x.begin(), flux_x.end(), 0.0);
            std::fill(flux_y.begin(), flux_y.end(), 0.0);
            for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
                for (int j = 0; j < params.ny; ++j) {
                    for (int i = 0; i < params.nx; ++i) {
                        const std::size_t n = idx(params, i, j, k);
                        const std::size_t local = state.v.local_offset_from_global_flat(n);
                        flux_x[local] = state.u[n] * state.v[n];
                        flux_y[local] = state.v[n] * state.v[n];
                    }
                }
            }
            fft.horizontal_divergence_plane_range(
                flux_x, flux_y, center_begin, slab.k_count, horizontal_conservative_v, params);

            Field face_flux_x(state.w.values.size(), 0.0);
            Field face_flux_y(state.w.values.size(), 0.0);
            for (int k = slab.face_begin; k < slab.face_begin + slab.face_count; ++k) {
                for (int j = 0; j < params.ny; ++j) {
                    for (int i = 0; i < params.nx; ++i) {
                        const std::size_t face = z_face_idx(params, i, j, k);
                        const std::size_t local = state.w.local_offset_from_global_flat(face);
                        face_flux_x[local] = workspace.u_on_w[face] * state.w[face];
                        face_flux_y[local] = workspace.v_on_w[face] * state.w[face];
                    }
                }
            }
            fft.horizontal_divergence_plane_range(
                face_flux_x, face_flux_y, face_begin, slab.face_count, horizontal_conservative_w, params);
        }
        for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const std::size_t n = idx(params, i, j, k);
                    const std::size_t local = state.u.local_offset_from_global_flat(n);
                    if (params.momentum_advection_form == "rotational") {
                        const double omega_z = grad.dvdx[n] - grad.dudy[n];
                        rhs_u[n] = state.v[n] * omega_z - workspace.momentum_div_u_z[n]
                            + params.nu * workspace.lap_u[n];
                        rhs_v[n] = workspace.momentum_div_v_z[n] - state.u[n] * omega_z
                            + params.nu * workspace.lap_v[n];
                        continue;
                    }
                    const double horizontal_u = params.horizontal_dealias && params.dealiasing == "padding_3_2"
                        ? horizontal_adv_u[local]
                        : state.u[n] * grad.dudx[n] + state.v[n] * grad.dudy[n];
                    const double horizontal_v = params.horizontal_dealias && params.dealiasing == "padding_3_2"
                        ? horizontal_adv_v[local]
                        : state.u[n] * grad.dvdx[n] + state.v[n] * grad.dvdy[n];
                    double vertical_u = workspace.w_center[n] * grad.dudz[n];
                    double vertical_v = workspace.w_center[n] * grad.dvdz[n];
                    double horizontal_u_used = horizontal_u;
                    double horizontal_v_used = horizontal_v;
                    if (params.momentum_advection_form == "skew_symmetric") {
                        horizontal_u_used = 0.5 * (horizontal_u + horizontal_conservative_u[local]);
                        horizontal_v_used = 0.5 * (horizontal_v + horizontal_conservative_v[local]);
                        const std::size_t lower_face = z_face_idx(params, i, j, k);
                        const std::size_t upper_face = z_face_idx(params, i, j, k + 1);
                        const double vertical_advective_u = 0.5 * (
                            state.w[lower_face] * workspace.momentum_dudz_face[lower_face]
                            + state.w[upper_face] * workspace.momentum_dudz_face[upper_face]);
                        const double vertical_advective_v = 0.5 * (
                            state.w[lower_face] * workspace.momentum_dvdz_face[lower_face]
                            + state.w[upper_face] * workspace.momentum_dvdz_face[upper_face]);
                        vertical_u = 0.5 * (vertical_advective_u + workspace.momentum_div_u_z[n]);
                        vertical_v = 0.5 * (vertical_advective_v + workspace.momentum_div_v_z[n]);
                    }
                    rhs_u[n] = -(horizontal_u_used + vertical_u)
                        + params.nu * workspace.lap_u[n];
                    rhs_v[n] = -(horizontal_v_used + vertical_v)
                        + params.nu * workspace.lap_v[n];
                }
            }
        }
        for (int k = slab.face_begin; k < slab.face_begin + slab.face_count; ++k) {
            if (k <= 0 || k >= params.nz) {
                continue;
            }
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const std::size_t face = z_face_idx(params, i, j, k);
                    const std::size_t local = state.w.local_offset_from_global_flat(face);
                    if (params.momentum_advection_form == "rotational") {
                        const double omega_x = workspace.dwdy_face[face] - workspace.momentum_dvdz_face[face];
                        const double omega_y = workspace.momentum_dudz_face[face] - workspace.dwdx_face[face];
                        rhs_w[face] = workspace.u_on_w[face] * omega_y
                            - workspace.v_on_w[face] * omega_x
                            + params.nu * workspace.lap_w[face];
                        continue;
                    }
                    const double horizontal_w = params.horizontal_dealias && params.dealiasing == "padding_3_2"
                        ? horizontal_adv_w[local]
                        : workspace.u_on_w[face] * workspace.dwdx_face[face]
                            + workspace.v_on_w[face] * workspace.dwdy_face[face];
                    const double horizontal_w_used = params.momentum_advection_form == "skew_symmetric"
                        ? 0.5 * (horizontal_w + horizontal_conservative_w[local])
                        : horizontal_w;
                    double vertical_w_used = state.w[face] * workspace.dwdz_face[face];
                    if (params.momentum_advection_form == "skew_symmetric") {
                        const std::size_t lower = idx(params, i, j, k - 1);
                        const std::size_t upper = idx(params, i, j, k);
                        const double vertical_advective_w = 0.5 * (
                            workspace.momentum_adv_w_z[lower] + workspace.momentum_adv_w_z[upper]);
                        vertical_w_used = 0.5 * (vertical_advective_w + workspace.momentum_div_w_z[face]);
                    }
                    rhs_w[face] =
                        -(horizontal_w_used + vertical_w_used)
                        + params.nu * workspace.lap_w[face];
                }
            }
        }
        double local_advection_power = 0.0;
        double local_advection_u = 0.0;
        double local_advection_v = 0.0;
        for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const std::size_t n = idx(params, i, j, k);
                    const double advection_u = rhs_u[n] - params.nu * workspace.lap_u[n];
                    const double advection_v = rhs_v[n] - params.nu * workspace.lap_v[n];
                    local_advection_power += state.u[n] * advection_u + state.v[n] * advection_v;
                    local_advection_u += advection_u;
                    local_advection_v += advection_v;
                }
            }
        }
        for (int k = slab.face_begin; k < slab.face_begin + slab.face_count; ++k) {
            if (k <= 0 || k >= params.nz) continue;
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const std::size_t face = z_face_idx(params, i, j, k);
                    local_advection_power += state.w[face] * (rhs_w[face] - params.nu * workspace.lap_w[face]);
                }
            }
        }
        workspace.momentum_advection_power_sum += local_advection_power;
        workspace.momentum_advection_u_sum += local_advection_u;
        workspace.momentum_advection_v_sum += local_advection_v;
        ++workspace.momentum_advection_power_samples;
        add_coriolis_geostrophic_forcing_slab(rhs_u, rhs_v, state, params, slab);
    }

    bool lasd_updated = false;
    {
        MpiTimerScope scope(timing, MpiTimerId::rhs_sgs_update);
        strain_magnitude_slab(grad, workspace.strain, params, slab);
        if (uses_moeng_tke(params)) {
            update_moeng_tke_coefficients_slab(
                state,
                workspace.nu_t,
                workspace.tke_length,
                workspace.tke_kh,
                workspace.tke_diffusivity,
                workspace.tke_dtheta_v_dz,
                params,
                slab);
        } else {
            lasd_updated = update_sgs_eddy_viscosity_slab(
                state,
                grad,
                workspace.sgs_dudz_face,
                workspace.sgs_dvdz_face,
                workspace.sgs_dwdx_face,
                workspace.sgs_dwdy_face,
                workspace.strain,
                workspace.w_center,
                workspace.nu_t,
                workspace.amd_buoyancy_prime,
                workspace.amd_db_dx,
                workspace.amd_db_dy,
                workspace.amd_db_dz,
                workspace.amd_invariant_num,
                workspace.amd_invariant_den,
                params,
                slab,
                fft,
                comm,
                workspace.lasd_germano_scratch,
                &workspace.lasd_filter_cache);
        }
        if (params.sgs_model != "none") {
            exchange_neighbor_planes(workspace.nu_t, slab.k_begin, slab.k_count, params.nz, 500, params, slab, comm);
        }
        if (uses_moeng_tke(params)) {
            exchange_neighbor_planes(workspace.tke_kh, slab.k_begin, slab.k_count, params.nz, 501, params, slab, comm);
        }
    }

    {
        MpiTimerScope scope(timing, MpiTimerId::rhs_sgs_forcing);
        add_sgs_momentum_forcing_slab(
            rhs_u, rhs_v, rhs_w, state, grad, workspace.nu_t, params, slab, fft, comm, workspace, timing);
    }
    {
        MpiTimerScope scope(timing, MpiTimerId::rhs_wall_stress);
        apply_wall_stress_slab(rhs_u, rhs_v, state, params, slab, fft);
    }
    {
        MpiTimerScope scope(timing, MpiTimerId::rhs_scalar);
        scalar_rhs_slab(
            state,
            params.moisture_enabled ? state.theta_l : state.theta,
            params.scalar_sgs_model == "amd_shared" ? &state.qt : nullptr,
            grad,
            uses_moeng_tke(params) ? workspace.tke_kh : workspace.nu_t,
            workspace.strain,
            params.scalar_diffusivity,
            uses_moeng_tke(params) ? 1.0 : params.prandtl_t,
            params.surface_theta_flux,
            state.scalar_c,
            state.scalar_lm_old,
            state.scalar_mm_old,
            state.scalar_qn_old,
            state.scalar_nn_old,
            790,
            workspace.scalar_kappa_center,
            params,
            slab,
            fft,
            comm,
            workspace.rhs_theta,
            workspace,
            timing,
            &workspace.lasd_filter_cache);
        if (params.moisture_enabled) {
            scalar_rhs_slab(
                state,
                state.qt,
                params.scalar_sgs_model == "amd_shared" ? &state.theta_l : nullptr,
                grad,
                uses_moeng_tke(params) ? workspace.tke_kh : workspace.nu_t,
                workspace.strain,
                params.moisture_diffusivity,
                uses_moeng_tke(params) ? 1.0 : params.schmidt_t,
                params.initial_condition == "bomex"
                    ? bomex_surface_qt_mixing_ratio_flux(params.surface_qv_flux)
                    : params.surface_qv_flux,
                state.qt_scalar_c,
                state.qt_scalar_lm_old,
                state.qt_scalar_mm_old,
                state.qt_scalar_qn_old,
                state.qt_scalar_nn_old,
                800,
                workspace.moisture_kappa_center,
                params,
                slab,
                fft,
                comm,
                workspace.rhs_qt,
                workspace,
                timing,
                &workspace.lasd_filter_cache);
            add_bomex_large_scale_forcing_slab(
                rhs_u,
                rhs_v,
                workspace.rhs_theta,
                workspace.rhs_qt,
                state,
                params,
                slab);
        }
        if (uses_moeng_tke(params)) {
            scalar_rhs_slab(
                state,
                state.sgs_tke,
                nullptr,
                grad,
                workspace.tke_diffusivity,
                workspace.strain,
                0.0,
                1.0,
                0.0,
                state.scalar_c,
                state.scalar_lm_old,
                state.scalar_mm_old,
                state.scalar_qn_old,
                state.scalar_nn_old,
                790,
                workspace.tke_scalar_kappa_center,
                params,
                slab,
                fft,
                comm,
                workspace.rhs_sgs_tke,
                workspace,
                timing,
                nullptr);

            const double delta = params.sgs_delta();
            const double surface_qt_flux = params.initial_condition == "bomex"
                ? bomex_surface_qt_mixing_ratio_flux(params.surface_qv_flux)
                : params.surface_qv_flux;
            auto theta_v_at = [&](std::size_t n) {
                return params.moisture_enabled
                    ? state.theta[n] * (1.0 + 0.61 * state.qv[n] - state.ql[n])
                    : state.theta[n];
            };
            auto theta_v_sgs_flux = [&](int face_k, int i, int j) {
                if (face_k <= 0) {
                    if (!params.moisture_enabled) {
                        return params.surface_theta_flux;
                    }
                    const std::size_t n = idx(params, i, j, 0);
                    return (1.0 + 0.61 * state.qv[n]) * params.surface_theta_flux
                        + 0.61 * state.theta_l[n] * surface_qt_flux;
                }
                if (face_k >= params.nz) {
                    return 0.0;
                }
                const std::size_t lower = idx(params, i, j, face_k - 1);
                const std::size_t upper = idx(params, i, j, face_k);
                const double kh_face = 0.5 * (workspace.tke_kh[lower] + workspace.tke_kh[upper]);
                return -kh_face * (theta_v_at(upper) - theta_v_at(lower)) / params.dz();
            };
            for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
                for (int j = 0; j < params.ny; ++j) {
                    for (int i = 0; i < params.nx; ++i) {
                        const std::size_t n = idx(params, i, j, k);
                        const double e = std::max(state.sgs_tke[n], params.tke_floor);
                        const double length = workspace.tke_length[n];
                        const double c_e = params.tke_dissipation_base
                            + params.tke_dissipation_slope * length / delta;
                        // P_s = 2 K_m S_ij S_ij = K_m |S|^2.  The moist
                        // buoyancy term uses the same K_h as theta_l and q_t.
                        const double shear_production = workspace.nu_t[n]
                            * workspace.strain[n] * workspace.strain[n];
                        const double buoyancy_production = (params.g / params.theta0) * 0.5
                            * (theta_v_sgs_flux(k, i, j) + theta_v_sgs_flux(k + 1, i, j));
                        const double dissipation = c_e * e * std::sqrt(e) / length;
                        workspace.rhs_sgs_tke[n] += shear_production + buoyancy_production - dissipation;
                    }
                }
            }
        }
    }
    if (lasd_updated) {
        reset_lasd_velocity_accumulators_slab(state, params, slab);
    }
    {
        MpiTimerScope scope(timing, MpiTimerId::rhs_buoyancy);
        add_buoyancy_slab(rhs_w, state, params, slab);
    }
    workspace.lasd_filter_cache.release_storage();
    workspace.lasd_germano_scratch.release_storage();
}

Diagnostics diagnostics_mpi_slab(const MpiLocalFlowState& state, const Params& params, const Slab& slab, FftwXY& fft, MPI_Comm comm) {
    struct LasdStats {
        double coefficient_sum = 0.0;
        double coefficient_max = 0.0;
        double beta_sum = 0.0;
        double active_count = 0.0;
        double beta_floor_count = 0.0;
    };
    LasdStats local_momentum_lasd;
    LasdStats local_theta_l_lasd;
    LasdStats local_qt_lasd;
    const double beta_min = 1.0 / (params.tfr * params.tfr * params.tfr);
    auto accumulate_lasd = [&](LasdStats& stats, double coefficient,
                               double numerator_2d, double denominator_2d,
                               double numerator_4d, double denominator_4d) {
        stats.coefficient_sum += coefficient;
        stats.coefficient_max = std::max(stats.coefficient_max, coefficient);
        if (denominator_2d <= 0.0 || denominator_4d <= 0.0) return;
        const double c2 = std::max(numerator_2d / denominator_2d, 0.0);
        const double c4 = std::max(numerator_4d / denominator_4d, 0.0);
        if (c2 <= 1.0e-30) return;
        const double raw_beta = std::max(c4 / c2, 0.0);
        stats.beta_sum += std::max(raw_beta, beta_min);
        stats.active_count += 1.0;
        stats.beta_floor_count += raw_beta <= beta_min ? 1.0 : 0.0;
    };
    double local_ke = 0.0;
    double local_cfl = 0.0;
    double local_qv_min = std::numeric_limits<double>::infinity();
    double local_qv_max = -std::numeric_limits<double>::infinity();
    double local_qt_min = std::numeric_limits<double>::infinity();
    double local_qt_max = -std::numeric_limits<double>::infinity();
    double local_ql_max = 0.0;
    double local_column_water_sum = 0.0;
    const double inv_dx = 1.0 / params.dx();
    const double inv_dy = 1.0 / params.dy();
    const double inv_dz = 1.0 / params.dz();
    for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                const double wc = 0.5 * (
                    state.w[z_face_idx(params, i, j, k)] + state.w[z_face_idx(params, i, j, k + 1)]);
                if (!std::isfinite(state.u[n]) || !std::isfinite(state.v[n]) || !std::isfinite(wc)) {
                    local_ke = std::numeric_limits<double>::infinity();
                    local_cfl = std::numeric_limits<double>::infinity();
                    continue;
                }
                local_ke = std::max(local_ke, 0.5 * (state.u[n] * state.u[n] + state.v[n] * state.v[n] + wc * wc));
                local_cfl = std::max(
                    local_cfl,
                    params.dt * (std::abs(state.u[n]) * inv_dx + std::abs(state.v[n]) * inv_dy + std::abs(wc) * inv_dz));
                if (params.sgs_model == "lasd") {
                    accumulate_lasd(local_momentum_lasd, state.cs2[n],
                        state.lm_old[n], state.mm_old[n], state.qn_old[n], state.nn_old[n]);
                    if (params.scalar_sgs_model == "lasd") {
                        accumulate_lasd(local_theta_l_lasd, state.scalar_c[n],
                            state.scalar_lm_old[n], state.scalar_mm_old[n],
                            state.scalar_qn_old[n], state.scalar_nn_old[n]);
                        accumulate_lasd(local_qt_lasd, state.qt_scalar_c[n],
                            state.qt_scalar_lm_old[n], state.qt_scalar_mm_old[n],
                            state.qt_scalar_qn_old[n], state.qt_scalar_nn_old[n]);
                    }
                }
                if (params.moisture_enabled) {
                    local_qv_min = std::min(local_qv_min, state.qv[n]);
                    local_qv_max = std::max(local_qv_max, state.qv[n]);
                    local_qt_min = std::min(local_qt_min, state.qt[n]);
                    local_qt_max = std::max(local_qt_max, state.qt[n]);
                    local_ql_max = std::max(local_ql_max, state.ql[n]);
                    local_column_water_sum += state.qt[n];
                }
            }
        }
    }
    const Field div_local = local_divergence_slab(state, params, slab, fft);
    double local_div = max_abs(div_local);

    Diagnostics diag;
    MPI_Allreduce(&local_ke, &diag.ke_max, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&local_cfl, &diag.cfl, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&local_div, &diag.div_max, 1, MPI_DOUBLE, MPI_MAX, comm);
    if (params.sgs_model == "lasd") {
        auto reduce_lasd = [&](const LasdStats& local, double& coefficient_mean,
                               double& coefficient_max, double& beta_mean,
                               double& beta_floor_fraction) {
            LasdStats global;
            MPI_Allreduce(&local.coefficient_sum, &global.coefficient_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
            MPI_Allreduce(&local.coefficient_max, &global.coefficient_max, 1, MPI_DOUBLE, MPI_MAX, comm);
            MPI_Allreduce(&local.beta_sum, &global.beta_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
            MPI_Allreduce(&local.active_count, &global.active_count, 1, MPI_DOUBLE, MPI_SUM, comm);
            MPI_Allreduce(&local.beta_floor_count, &global.beta_floor_count, 1, MPI_DOUBLE, MPI_SUM, comm);
            const double cell_count = static_cast<double>(params.nx * params.ny * params.nz);
            coefficient_mean = global.coefficient_sum / cell_count;
            coefficient_max = global.coefficient_max;
            beta_mean = global.active_count > 0.0 ? global.beta_sum / global.active_count : 0.0;
            beta_floor_fraction = global.active_count > 0.0
                ? global.beta_floor_count / global.active_count : 0.0;
        };
        reduce_lasd(local_momentum_lasd, diag.lasd_cs2_mean, diag.lasd_cs2_max,
            diag.lasd_beta_mean, diag.lasd_beta_floor_fraction);
        double unused_beta_mean = 0.0;
        reduce_lasd(local_theta_l_lasd, diag.lasd_theta_c_mean, diag.lasd_theta_c_max,
            unused_beta_mean, diag.lasd_theta_beta_floor_fraction);
        reduce_lasd(local_qt_lasd, diag.lasd_qt_c_mean, diag.lasd_qt_c_max,
            unused_beta_mean, diag.lasd_qt_beta_floor_fraction);
    }
    if (params.moisture_enabled) {
        double global_column_water_sum = 0.0;
        MPI_Allreduce(&local_qv_min, &diag.qv_min, 1, MPI_DOUBLE, MPI_MIN, comm);
        MPI_Allreduce(&local_qv_max, &diag.qv_max, 1, MPI_DOUBLE, MPI_MAX, comm);
        MPI_Allreduce(&local_qt_min, &diag.qt_min, 1, MPI_DOUBLE, MPI_MIN, comm);
        MPI_Allreduce(&local_qt_max, &diag.qt_max, 1, MPI_DOUBLE, MPI_MAX, comm);
        MPI_Allreduce(&local_ql_max, &diag.ql_max, 1, MPI_DOUBLE, MPI_MAX, comm);
        MPI_Allreduce(&local_column_water_sum, &global_column_water_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
        diag.column_water = global_column_water_sum * params.dz()
            / static_cast<double>(params.nx * params.ny);
    }
    return diag;
}

void add_mpi_bomex_sample(
    BomexAccumulator& accumulator,
    const MpiLocalFlowState& state,
    const Params& params,
    const Slab& slab,
    MPI_Comm comm,
    const MpiSlabWorkspace& workspace) {
    constexpr int metric_count = 51;
    const std::size_t nz = static_cast<std::size_t>(params.nz);
    const auto& thresholds = bomex_cloud_thresholds();
    const std::size_t threshold_count = thresholds.size();
    const std::size_t column_count = static_cast<std::size_t>(params.nx * params.ny);
    std::vector<double> local(static_cast<std::size_t>(metric_count) * nz, 0.0);
    std::vector<double> local_cloud_fraction_by_threshold(threshold_count * nz, 0.0);
    std::vector<int> local_cloudy_columns(column_count, 0);
    std::vector<int> local_cloudy_columns_by_threshold(threshold_count * column_count, 0);
    double local_lwp = 0.0;
    double local_integrated_tke = 0.0;
    double local_max_cloud_fraction = 0.0;
    const double inverse_plane = 1.0 / static_cast<double>(params.nx * params.ny);

    for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
        const std::size_t kk = static_cast<std::size_t>(k);
        double theta_l_mean = 0.0;
        double qt_mean = 0.0;
        double qv_mean = 0.0;
        double ql_mean = 0.0;
        double w_mean = 0.0;
        double theta_v_mean = 0.0;
        double u_mean = 0.0;
        double v_mean = 0.0;
        double p_mean = 0.0;
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                const double wc = 0.5 * (
                    state.w[z_face_idx(params, i, j, k)]
                    + state.w[z_face_idx(params, i, j, k + 1)]);
                theta_l_mean += state.theta_l[n] * inverse_plane;
                qt_mean += state.qt[n] * inverse_plane;
                qv_mean += state.qv[n] * inverse_plane;
                ql_mean += state.ql[n] * inverse_plane;
                w_mean += wc * inverse_plane;
                theta_v_mean += state.theta[n] * (1.0 + 0.61 * state.qv[n] - state.ql[n]) * inverse_plane;
                u_mean += state.u[n] * inverse_plane;
                v_mean += state.v[n] * inverse_plane;
                p_mean += state.p[n] * inverse_plane;
            }
        }
        double cloud_fraction = 0.0;
        std::vector<double> cloud_fraction_by_threshold(threshold_count, 0.0);
        double core_fraction = 0.0;
        double w_variance = 0.0;
        double u_variance = 0.0;
        double v_variance = 0.0;
        double theta_l_flux = 0.0;
        double qt_flux = 0.0;
        double ql_flux = 0.0;
        double theta_v_flux = 0.0;
        double uw_flux = 0.0;
        double tke = 0.0;
        double cloud_theta_l = 0.0, core_theta_l = 0.0;
        double cloud_qt = 0.0, core_qt = 0.0;
        double cloud_theta_v = 0.0, core_theta_v = 0.0;
        double cloud_ql = 0.0, core_ql = 0.0;
        double cloud_w = 0.0, core_w = 0.0;
        int cloud_count = 0, core_count = 0;
        double sgs_theta_l_flux = 0.0, sgs_qt_flux = 0.0, sgs_ql_flux = 0.0;
        double sgs_theta_v_flux = 0.0, sgs_uw_flux = 0.0;
        double mean_nu_t = 0.0, mean_sgs_tke = 0.0, mean_strain_squared = 0.0, mean_sgs_dissipation = 0.0;
        double mean_theta_l_scalar_c = 0.0, mean_qt_scalar_c = 0.0;
        double mean_theta_l_scalar_diffusivity = 0.0, mean_qt_scalar_diffusivity = 0.0;
        double zero_theta_l_scalar_diffusivity_fraction = 0.0;
        double zero_qt_scalar_diffusivity_fraction = 0.0;
        double zero_nu_t_fraction = 0.0, vw_flux = 0.0, sgs_vw_flux = 0.0;
        double w_tke_flux = 0.0, w_pressure_flux = 0.0, wall_tke_work = 0.0;
        const bool wall_stress_is_local = params.wall_stress_model == "prescribed_ustar_local";
        const bool sample_wall_work = k == 0
            && params.momentum_wall_model == "abl"
            && (params.wall_stress_model == "prescribed_ustar" || wall_stress_is_local);
        double wall_local_drag = 0.0;
        if (sample_wall_work && wall_stress_is_local) {
            double mean_speed_u = 0.0;
            double mean_speed_v = 0.0;
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const std::size_t n = idx(params, i, j, 0);
                    const double speed = std::hypot(state.u[n], state.v[n]);
                    mean_speed_u += speed * state.u[n] * inverse_plane;
                    mean_speed_v += speed * state.v[n] * inverse_plane;
                }
            }
            wall_local_drag = prescribed_ustar_local_drag_coefficient(
                mean_speed_u, mean_speed_v, params.u_fric);
        }
        const double surface_qt_flux = bomex_surface_qt_mixing_ratio_flux(params.surface_qv_flux);
        const double speed = std::hypot(u_mean, v_mean);
        const double surface_uw_flux = speed > 0.0 ? -params.u_fric * params.u_fric * u_mean / speed : 0.0;
        const double surface_theta_v_flux =
            (1.0 + 0.61 * qv_mean) * params.surface_theta_flux
            + 0.61 * theta_l_mean * surface_qt_flux;
        auto scalar_face_flux = [&](
            const LocalField& field,
            const LocalField& diffusivity,
            int face_k,
            int i,
            int j,
            double surface) {
            if (face_k <= 0) return surface;
            if (face_k >= params.nz) return 0.0;
            const std::size_t lower = idx(params, i, j, face_k - 1);
            const std::size_t upper = idx(params, i, j, face_k);
            const double diffusivity_face = 0.5 * (diffusivity[lower] + diffusivity[upper]);
            return -diffusivity_face * (field[upper] - field[lower]) / params.dz();
        };
        auto diagnosed_moist_face_flux = [&](int face_k, int i, int j) {
            if (face_k <= 0) return std::pair<double, double>{0.0, surface_theta_v_flux};
            if (face_k >= params.nz) return std::pair<double, double>{0.0, 0.0};
            const std::size_t lower = idx(params, i, j, face_k - 1);
            const std::size_t upper = idx(params, i, j, face_k);
            const double theta_l_flux_face = scalar_face_flux(
                state.theta_l, workspace.scalar_kappa_center,
                face_k, i, j, params.surface_theta_flux);
            const double qt_flux_face = scalar_face_flux(
                state.qt, workspace.moisture_kappa_center,
                face_k, i, j, surface_qt_flux);
            const MoistConservedJacobians lower_jacobian = moist_conserved_jacobians(
                state.theta[lower], state.qv[lower], state.ql[lower],
                state.base_pressure[static_cast<std::size_t>(face_k - 1)]);
            const MoistConservedJacobians upper_jacobian = moist_conserved_jacobians(
                state.theta[upper], state.qv[upper], state.ql[upper],
                state.base_pressure[static_cast<std::size_t>(face_k)]);
            auto average = [](double lower_value, double upper_value) {
                return 0.5 * (lower_value + upper_value);
            };
            const double ql_flux_face =
                average(lower_jacobian.dliquid_water_dtheta_l,
                    upper_jacobian.dliquid_water_dtheta_l) * theta_l_flux_face
                + average(lower_jacobian.dliquid_water_dtotal_water,
                    upper_jacobian.dliquid_water_dtotal_water) * qt_flux_face;
            const double theta_v_flux_face =
                average(lower_jacobian.dvirtual_theta_dtheta_l,
                    upper_jacobian.dvirtual_theta_dtheta_l) * theta_l_flux_face
                + average(lower_jacobian.dvirtual_theta_dtotal_water,
                    upper_jacobian.dvirtual_theta_dtotal_water) * qt_flux_face;
            return std::pair<double, double>{ql_flux_face, theta_v_flux_face};
        };
        auto uw_face_flux = [&](int face_k, int i, int j) {
            if (face_k <= 0) return surface_uw_flux;
            if (face_k >= params.nz) return 0.0;
            return -workspace.sgs_txz[z_face_idx(params, i, j, face_k)];
        };
        auto vw_face_flux = [&](int face_k, int i, int j) {
            if (face_k <= 0 || face_k >= params.nz) return 0.0;
            return -workspace.sgs_tyz[z_face_idx(params, i, j, face_k)];
        };
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                const double wc = 0.5 * (
                    state.w[z_face_idx(params, i, j, k)]
                    + state.w[z_face_idx(params, i, j, k + 1)]);
                const double theta_v = state.theta[n] * (1.0 + 0.61 * state.qv[n] - state.ql[n]);
                const std::size_t column = static_cast<std::size_t>(j * params.nx + i);
                for (std::size_t threshold_index = 0; threshold_index < threshold_count; ++threshold_index) {
                    const bool cloudy_at_threshold = state.ql[n] > thresholds[threshold_index];
                    cloud_fraction_by_threshold[threshold_index] += cloudy_at_threshold ? inverse_plane : 0.0;
                    if (cloudy_at_threshold) {
                        local_cloudy_columns_by_threshold[threshold_index * column_count + column] = 1;
                    }
                }
                const bool cloudy = state.ql[n] > thresholds.front();
                const bool core = cloudy && theta_v > theta_v_mean;
                cloud_fraction += cloudy ? inverse_plane : 0.0;
                core_fraction += core ? inverse_plane : 0.0;
                if (cloudy) {
                    local_cloudy_columns[column] = 1;
                }
                const double w_prime = wc - w_mean;
                const double u_prime = state.u[n] - u_mean;
                const double v_prime = state.v[n] - v_mean;
                w_variance += w_prime * w_prime * inverse_plane;
                u_variance += u_prime * u_prime * inverse_plane;
                v_variance += v_prime * v_prime * inverse_plane;
                tke += 0.5 * (u_prime * u_prime + v_prime * v_prime + w_prime * w_prime) * inverse_plane;
                theta_l_flux += w_prime * (state.theta_l[n] - theta_l_mean) * inverse_plane;
                qt_flux += w_prime * (state.qt[n] - qt_mean) * inverse_plane;
                ql_flux += w_prime * (state.ql[n] - ql_mean) * inverse_plane;
                theta_v_flux += w_prime * (theta_v - theta_v_mean) * inverse_plane;
                uw_flux += u_prime * w_prime * inverse_plane;
                vw_flux += v_prime * w_prime * inverse_plane;
                const double resolved_tke_point =
                    0.5 * (u_prime * u_prime + v_prime * v_prime + w_prime * w_prime);
                w_tke_flux += w_prime * resolved_tke_point * inverse_plane;
                w_pressure_flux += w_prime * (state.p[n] - p_mean) * inverse_plane;
                if (sample_wall_work) {
                    const double speed = std::hypot(state.u[n], state.v[n]);
                    if (speed > 1.0e-12) {
                        double tau_x;
                        double tau_y;
                        if (wall_stress_is_local) {
                            tau_x = -wall_local_drag * speed * state.u[n];
                            tau_y = -wall_local_drag * speed * state.v[n];
                        } else {
                            const double tau = -params.u_fric * params.u_fric;
                            tau_x = tau * state.u[n] / speed;
                            tau_y = tau * state.v[n] / speed;
                        }
                        wall_tke_work += (tau_x * u_prime + tau_y * v_prime)
                            / params.dz() * inverse_plane;
                    }
                }
                mean_nu_t += workspace.nu_t[n] * inverse_plane;
                mean_theta_l_scalar_c += state.scalar_c[n] * inverse_plane;
                mean_qt_scalar_c += state.qt_scalar_c[n] * inverse_plane;
                mean_theta_l_scalar_diffusivity += workspace.scalar_kappa_center[n] * inverse_plane;
                mean_qt_scalar_diffusivity += workspace.moisture_kappa_center[n] * inverse_plane;
                zero_theta_l_scalar_diffusivity_fraction +=
                    workspace.scalar_kappa_center[n] <= 1.0e-14 ? inverse_plane : 0.0;
                zero_qt_scalar_diffusivity_fraction +=
                    workspace.moisture_kappa_center[n] <= 1.0e-14 ? inverse_plane : 0.0;
                mean_sgs_tke += state.sgs_tke[n] * inverse_plane;
                const double strain_squared = workspace.strain[n] * workspace.strain[n];
                mean_strain_squared += strain_squared * inverse_plane;
                if (uses_moeng_tke(params)) {
                    const double e = std::max(state.sgs_tke[n], params.tke_floor);
                    const double length = workspace.tke_length[n];
                    const double c_e = params.tke_dissipation_base
                        + params.tke_dissipation_slope * length / params.sgs_delta();
                    mean_sgs_dissipation += c_e * e * std::sqrt(e) / length * inverse_plane;
                } else {
                    mean_sgs_dissipation += workspace.nu_t[n] * strain_squared * inverse_plane;
                }
                zero_nu_t_fraction += workspace.nu_t[n] <= 1.0e-14 ? inverse_plane : 0.0;
                sgs_theta_l_flux += 0.5 * (
                    scalar_face_flux(state.theta_l, workspace.scalar_kappa_center, k, i, j, params.surface_theta_flux)
                    + scalar_face_flux(state.theta_l, workspace.scalar_kappa_center, k + 1, i, j, params.surface_theta_flux)) * inverse_plane;
                sgs_qt_flux += 0.5 * (
                    scalar_face_flux(state.qt, workspace.moisture_kappa_center, k, i, j, surface_qt_flux)
                    + scalar_face_flux(state.qt, workspace.moisture_kappa_center, k + 1, i, j, surface_qt_flux)) * inverse_plane;
                const auto lower_diagnostic_flux = diagnosed_moist_face_flux(k, i, j);
                const auto upper_diagnostic_flux = diagnosed_moist_face_flux(k + 1, i, j);
                sgs_ql_flux += 0.5 * (
                    lower_diagnostic_flux.first + upper_diagnostic_flux.first) * inverse_plane;
                sgs_theta_v_flux += 0.5 * (
                    lower_diagnostic_flux.second + upper_diagnostic_flux.second) * inverse_plane;
                sgs_uw_flux += 0.5 * (
                    uw_face_flux(k, i, j) + uw_face_flux(k + 1, i, j)) * inverse_plane;
                sgs_vw_flux += 0.5 * (
                    vw_face_flux(k, i, j) + vw_face_flux(k + 1, i, j)) * inverse_plane;
                if (cloudy) {
                    ++cloud_count;
                    cloud_theta_l += state.theta_l[n]; cloud_qt += state.qt[n];
                    cloud_theta_v += theta_v; cloud_ql += state.ql[n]; cloud_w += wc;
                }
                if (core) {
                    ++core_count;
                    core_theta_l += state.theta_l[n]; core_qt += state.qt[n];
                    core_theta_v += theta_v; core_ql += state.ql[n]; core_w += wc;
                }
            }
        }
        local_max_cloud_fraction = std::max(local_max_cloud_fraction, cloud_fraction);
        local[0 * nz + kk] = theta_l_mean;
        local[1 * nz + kk] = qt_mean;
        local[2 * nz + kk] = qv_mean;
        local[3 * nz + kk] = ql_mean;
        local[4 * nz + kk] = cloud_fraction;
        local[5 * nz + kk] = core_fraction;
        local[6 * nz + kk] = w_variance;
        local[7 * nz + kk] = theta_l_flux;
        local[8 * nz + kk] = qt_flux;
        local[9 * nz + kk] = u_mean;
        local[10 * nz + kk] = v_mean;
        local[11 * nz + kk] = tke;
        local[12 * nz + kk] = ql_flux;
        local[13 * nz + kk] = theta_v_flux;
        local[14 * nz + kk] = uw_flux;
        if (cloud_count > 0) {
            local[15 * nz + kk] = cloud_theta_l; local[17 * nz + kk] = cloud_qt;
            local[19 * nz + kk] = cloud_theta_v; local[21 * nz + kk] = cloud_ql;
            local[23 * nz + kk] = cloud_w;
        }
        if (core_count > 0) {
            local[16 * nz + kk] = core_theta_l; local[18 * nz + kk] = core_qt;
            local[20 * nz + kk] = core_theta_v; local[22 * nz + kk] = core_ql;
            local[24 * nz + kk] = core_w;
        }
        local[25 * nz + kk] = static_cast<double>(cloud_count);
        local[26 * nz + kk] = static_cast<double>(core_count);
        local[27 * nz + kk] = sgs_theta_l_flux;
        local[28 * nz + kk] = sgs_qt_flux;
        local[29 * nz + kk] = sgs_ql_flux;
        local[30 * nz + kk] = sgs_theta_v_flux;
        local[31 * nz + kk] = sgs_uw_flux;
        local[32 * nz + kk] = u_variance;
        local[33 * nz + kk] = v_variance;
        local[34 * nz + kk] = mean_nu_t;
        local[35 * nz + kk] = mean_strain_squared;
        local[36 * nz + kk] = mean_sgs_dissipation;
        local[37 * nz + kk] = zero_nu_t_fraction;
        local[38 * nz + kk] = theta_v_mean;
        local[39 * nz + kk] = vw_flux;
        local[40 * nz + kk] = sgs_vw_flux;
        local[41 * nz + kk] = mean_sgs_tke;
        local[42 * nz + kk] = mean_theta_l_scalar_c;
        local[43 * nz + kk] = mean_qt_scalar_c;
        local[44 * nz + kk] = mean_theta_l_scalar_diffusivity;
        local[45 * nz + kk] = mean_qt_scalar_diffusivity;
        local[46 * nz + kk] = zero_theta_l_scalar_diffusivity_fraction;
        local[47 * nz + kk] = zero_qt_scalar_diffusivity_fraction;
        local[48 * nz + kk] = w_tke_flux;
        local[49 * nz + kk] = w_pressure_flux;
        local[50 * nz + kk] = wall_tke_work;
        for (std::size_t threshold_index = 0; threshold_index < threshold_count; ++threshold_index) {
            local_cloud_fraction_by_threshold[threshold_index * nz + kk] =
                cloud_fraction_by_threshold[threshold_index];
        }
        const double pressure = state.base_pressure[kk];
        const double temperature = theta_l_mean * exner_function(pressure);
        const ThermodynamicConstants constants;
        const double density = pressure / (constants.dry_air_gas_constant * temperature);
        local_lwp += density * ql_mean * params.dz();
        local_integrated_tke += density * tke * params.dz();
    }

    std::vector<double> global(local.size(), 0.0);
    std::vector<double> global_cloud_fraction_by_threshold(local_cloud_fraction_by_threshold.size(), 0.0);
    std::vector<int> global_cloudy_columns(local_cloudy_columns.size(), 0);
    std::vector<int> global_cloudy_columns_by_threshold(local_cloudy_columns_by_threshold.size(), 0);
    double global_lwp = 0.0;
    double global_integrated_tke = 0.0;
    double global_max_cloud_fraction = 0.0;
    MPI_Reduce(local.data(), global.data(), static_cast<int>(local.size()), MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(
        local_cloud_fraction_by_threshold.data(),
        global_cloud_fraction_by_threshold.data(),
        static_cast<int>(local_cloud_fraction_by_threshold.size()),
        MPI_DOUBLE,
        MPI_SUM,
        0,
        comm);
    MPI_Reduce(
        local_cloudy_columns.data(),
        global_cloudy_columns.data(),
        static_cast<int>(local_cloudy_columns.size()),
        MPI_INT,
        MPI_MAX,
        0,
        comm);
    MPI_Reduce(
        local_cloudy_columns_by_threshold.data(),
        global_cloudy_columns_by_threshold.data(),
        static_cast<int>(local_cloudy_columns_by_threshold.size()),
        MPI_INT,
        MPI_MAX,
        0,
        comm);
    MPI_Reduce(&local_lwp, &global_lwp, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(
        &local_integrated_tke,
        &global_integrated_tke,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        0,
        comm);
    MPI_Reduce(&local_max_cloud_fraction, &global_max_cloud_fraction, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (slab.rank != 0) {
        return;
    }
    if (accumulator.theta_l.empty()) {
        accumulator.theta_l.assign(nz, 0.0);
        accumulator.qt.assign(nz, 0.0);
        accumulator.qv.assign(nz, 0.0);
        accumulator.ql.assign(nz, 0.0);
        accumulator.u.assign(nz, 0.0);
        accumulator.v.assign(nz, 0.0);
        accumulator.tke.assign(nz, 0.0);
        accumulator.cloud_fraction.assign(nz, 0.0);
        accumulator.cloud_fraction_by_threshold.assign(threshold_count * nz, 0.0);
        accumulator.core_fraction.assign(nz, 0.0);
        accumulator.w_variance.assign(nz, 0.0);
        accumulator.u_variance.assign(nz, 0.0);
        accumulator.v_variance.assign(nz, 0.0);
        accumulator.mean_eddy_viscosity.assign(nz, 0.0);
        accumulator.mean_sgs_tke.assign(nz, 0.0);
        accumulator.mean_strain_squared.assign(nz, 0.0);
        accumulator.mean_sgs_dissipation.assign(nz, 0.0);
        accumulator.zero_eddy_viscosity_fraction.assign(nz, 0.0);
        accumulator.mean_theta_l_scalar_c.assign(nz, 0.0);
        accumulator.mean_qt_scalar_c.assign(nz, 0.0);
        accumulator.mean_theta_l_scalar_diffusivity.assign(nz, 0.0);
        accumulator.mean_qt_scalar_diffusivity.assign(nz, 0.0);
        accumulator.zero_theta_l_scalar_diffusivity_fraction.assign(nz, 0.0);
        accumulator.zero_qt_scalar_diffusivity_fraction.assign(nz, 0.0);
        accumulator.mean_theta_v.assign(nz, 0.0);
        accumulator.resolved_vw_flux.assign(nz, 0.0);
        accumulator.sgs_vw_flux.assign(nz, 0.0);
        accumulator.resolved_theta_l_flux.assign(nz, 0.0);
        accumulator.resolved_qt_flux.assign(nz, 0.0);
        accumulator.resolved_ql_flux.assign(nz, 0.0);
        accumulator.resolved_theta_v_flux.assign(nz, 0.0);
        accumulator.resolved_uw_flux.assign(nz, 0.0);
        accumulator.sgs_theta_l_flux.assign(nz, 0.0);
        accumulator.sgs_qt_flux.assign(nz, 0.0);
        accumulator.sgs_ql_flux.assign(nz, 0.0);
        accumulator.sgs_theta_v_flux.assign(nz, 0.0);
        accumulator.sgs_uw_flux.assign(nz, 0.0);
        accumulator.cloud_theta_l.assign(nz, 0.0);
        accumulator.core_theta_l.assign(nz, 0.0);
        accumulator.cloud_qt.assign(nz, 0.0);
        accumulator.core_qt.assign(nz, 0.0);
        accumulator.cloud_theta_v.assign(nz, 0.0);
        accumulator.core_theta_v.assign(nz, 0.0);
        accumulator.cloud_ql.assign(nz, 0.0);
        accumulator.core_ql.assign(nz, 0.0);
        accumulator.cloud_w.assign(nz, 0.0);
        accumulator.core_w.assign(nz, 0.0);
        accumulator.resolved_w_tke_flux.assign(nz, 0.0);
        accumulator.resolved_w_pressure_flux.assign(nz, 0.0);
        accumulator.wall_fluctuation_tke_work.assign(nz, 0.0);
        accumulator.cloud_conditional_samples.assign(nz, 0);
        accumulator.core_conditional_samples.assign(nz, 0);
        accumulator.total_cloud_cover_by_threshold.assign(threshold_count, 0.0);
    }
    for (std::size_t k = 0; k < nz; ++k) {
        accumulator.theta_l[k] += global[0 * nz + k];
        accumulator.qt[k] += global[1 * nz + k];
        accumulator.qv[k] += global[2 * nz + k];
        accumulator.ql[k] += global[3 * nz + k];
        accumulator.cloud_fraction[k] += global[4 * nz + k];
        for (std::size_t threshold_index = 0; threshold_index < threshold_count; ++threshold_index) {
            accumulator.cloud_fraction_by_threshold[threshold_index * nz + k] +=
                global_cloud_fraction_by_threshold[threshold_index * nz + k];
        }
        accumulator.core_fraction[k] += global[5 * nz + k];
        accumulator.w_variance[k] += global[6 * nz + k];
        accumulator.resolved_theta_l_flux[k] += global[7 * nz + k];
        accumulator.resolved_qt_flux[k] += global[8 * nz + k];
        accumulator.u[k] += global[9 * nz + k];
        accumulator.v[k] += global[10 * nz + k];
        accumulator.tke[k] += global[11 * nz + k];
        accumulator.resolved_ql_flux[k] += global[12 * nz + k];
        accumulator.resolved_theta_v_flux[k] += global[13 * nz + k];
        accumulator.resolved_uw_flux[k] += global[14 * nz + k];
        accumulator.cloud_theta_l[k] += global[15 * nz + k];
        accumulator.core_theta_l[k] += global[16 * nz + k];
        accumulator.cloud_qt[k] += global[17 * nz + k];
        accumulator.core_qt[k] += global[18 * nz + k];
        accumulator.cloud_theta_v[k] += global[19 * nz + k];
        accumulator.core_theta_v[k] += global[20 * nz + k];
        accumulator.cloud_ql[k] += global[21 * nz + k];
        accumulator.core_ql[k] += global[22 * nz + k];
        accumulator.cloud_w[k] += global[23 * nz + k];
        accumulator.core_w[k] += global[24 * nz + k];
        accumulator.cloud_conditional_samples[k] += static_cast<std::size_t>(std::llround(global[25 * nz + k]));
        accumulator.core_conditional_samples[k] += static_cast<std::size_t>(std::llround(global[26 * nz + k]));
        accumulator.sgs_theta_l_flux[k] += global[27 * nz + k];
        accumulator.sgs_qt_flux[k] += global[28 * nz + k];
        accumulator.sgs_ql_flux[k] += global[29 * nz + k];
        accumulator.sgs_theta_v_flux[k] += global[30 * nz + k];
        accumulator.sgs_uw_flux[k] += global[31 * nz + k];
        accumulator.u_variance[k] += global[32 * nz + k];
        accumulator.v_variance[k] += global[33 * nz + k];
        accumulator.mean_eddy_viscosity[k] += global[34 * nz + k];
        accumulator.mean_strain_squared[k] += global[35 * nz + k];
        accumulator.mean_sgs_dissipation[k] += global[36 * nz + k];
        accumulator.zero_eddy_viscosity_fraction[k] += global[37 * nz + k];
        accumulator.mean_theta_v[k] += global[38 * nz + k];
        accumulator.resolved_vw_flux[k] += global[39 * nz + k];
        accumulator.sgs_vw_flux[k] += global[40 * nz + k];
        accumulator.mean_sgs_tke[k] += global[41 * nz + k];
        accumulator.mean_theta_l_scalar_c[k] += global[42 * nz + k];
        accumulator.mean_qt_scalar_c[k] += global[43 * nz + k];
        accumulator.mean_theta_l_scalar_diffusivity[k] += global[44 * nz + k];
        accumulator.mean_qt_scalar_diffusivity[k] += global[45 * nz + k];
        accumulator.zero_theta_l_scalar_diffusivity_fraction[k] += global[46 * nz + k];
        accumulator.zero_qt_scalar_diffusivity_fraction[k] += global[47 * nz + k];
        accumulator.resolved_w_tke_flux[k] += global[48 * nz + k];
        accumulator.resolved_w_pressure_flux[k] += global[49 * nz + k];
        accumulator.wall_fluctuation_tke_work[k] += global[50 * nz + k];
    }
    const double sample_total_cloud_cover = static_cast<double>(std::count(
        global_cloudy_columns.begin(), global_cloudy_columns.end(), 1)) * inverse_plane;
    accumulator.total_cloud_cover += sample_total_cloud_cover;
    std::vector<double> sample_total_cloud_cover_by_threshold(threshold_count, 0.0);
    for (std::size_t threshold_index = 0; threshold_index < threshold_count; ++threshold_index) {
        const auto begin = global_cloudy_columns_by_threshold.begin()
            + static_cast<std::ptrdiff_t>(threshold_index * column_count);
        const auto end = begin + static_cast<std::ptrdiff_t>(column_count);
        sample_total_cloud_cover_by_threshold[threshold_index] =
            static_cast<double>(std::count(begin, end, 1)) * inverse_plane;
        accumulator.total_cloud_cover_by_threshold[threshold_index] += sample_total_cloud_cover_by_threshold[threshold_index];
    }
    accumulator.liquid_water_path += global_lwp;
    accumulator.sample_step.push_back(state.step_count);
    accumulator.sample_time_s.push_back(static_cast<double>(state.step_count) * params.dt);
    accumulator.sample_total_cloud_cover.push_back(sample_total_cloud_cover);
    accumulator.sample_total_cloud_cover_by_threshold.insert(
        accumulator.sample_total_cloud_cover_by_threshold.end(),
        sample_total_cloud_cover_by_threshold.begin(),
        sample_total_cloud_cover_by_threshold.end());
    accumulator.sample_max_cloud_fraction.push_back(global_max_cloud_fraction);
    accumulator.sample_liquid_water_path.push_back(global_lwp);
    accumulator.sample_integrated_tke.push_back(global_integrated_tke);
    std::vector<double> qt_mean_profile(nz, 0.0);
    double column_qt = 0.0;
    for (std::size_t k = 0; k < nz; ++k) {
        qt_mean_profile[k] = global[1 * nz + k];
        column_qt += qt_mean_profile[k] * params.dz();
    }
    double qt_large_scale_tendency = 0.0;
    auto profile_derivative = [&](int k) {
        if (params.nz <= 1) return 0.0;
        if (k == 0) return (qt_mean_profile[1] - qt_mean_profile[0]) / params.dz();
        if (k == params.nz - 1) {
            return (qt_mean_profile[static_cast<std::size_t>(k)]
                - qt_mean_profile[static_cast<std::size_t>(k - 1)]) / params.dz();
        }
        return (qt_mean_profile[static_cast<std::size_t>(k + 1)]
            - qt_mean_profile[static_cast<std::size_t>(k - 1)]) / (2.0 * params.dz());
    };
    for (int k = 0; k < params.nz; ++k) {
        const double z = (static_cast<double>(k) + 0.5) * params.dz();
        const double qt_specific = bomex_mixing_to_specific_humidity(qt_mean_profile[static_cast<std::size_t>(k)]);
        const double jacobian = 1.0 / std::pow(1.0 - qt_specific, 2.0);
        qt_large_scale_tendency += (
            -bomex_subsidence(z) * profile_derivative(k)
            + bomex_moisture_advection_tendency(z) * jacobian) * params.dz();
    }
    accumulator.sample_column_qt_m.push_back(column_qt);
    accumulator.sample_qt_large_scale_tendency_m_s.push_back(qt_large_scale_tendency);
    ++accumulator.samples;
}

struct MpiBenchmarkSnapshot {
    std::vector<double> heat_flux_total;
    std::vector<double> heat_flux_resolved;
    std::vector<double> heat_flux_sgs;
    std::vector<double> heat_flux_face_total;
    std::vector<double> heat_flux_face_resolved;
    std::vector<double> heat_flux_face_sgs;
    std::vector<double> u_mean;
    std::vector<double> v_mean;
    std::vector<double> w_mean;
    std::vector<double> p_mean;
    std::vector<double> theta_mean;
    std::vector<double> u_var;
    std::vector<double> v_var;
    std::vector<double> w_var;
    std::vector<double> theta_var;
    std::vector<double> p_var;
    std::vector<double> w3;
    std::vector<double> w_transport;
    std::vector<double> p_transport;
    std::vector<double> alpha_u;
    std::vector<double> w_u;
    std::vector<double> theta_u_excess;
    std::vector<double> epsilon;
    std::vector<double> cs2_mean;
    std::vector<double> scalar_c_mean;
    std::vector<double> kappa_mean;
    double zi = 0.0;
    double wstar = 0.0;
};

struct MpiBenchmarkAccumulator {
    int sample_count = 0;
    double zi_sum = 0.0;
    double wstar_sum = 0.0;
    std::vector<double> heat_flux_sum;
    std::vector<double> heat_flux_resolved_sum;
    std::vector<double> heat_flux_sgs_sum;
    std::vector<double> heat_flux_face_sum;
    std::vector<double> heat_flux_face_resolved_sum;
    std::vector<double> heat_flux_face_sgs_sum;
    std::vector<double> u_mean_sum;
    std::vector<double> v_mean_sum;
    std::vector<double> w_mean_sum;
    std::vector<double> p_mean_sum;
    std::vector<double> theta_mean_sum;
    std::vector<double> u_var_sum;
    std::vector<double> v_var_sum;
    std::vector<double> w_var_sum;
    std::vector<double> theta_var_sum;
    std::vector<double> p_var_sum;
    std::vector<double> w3_sum;
    std::vector<double> w_transport_sum;
    std::vector<double> p_transport_sum;
    std::vector<double> alpha_u_sum;
    std::vector<double> w_u_sum;
    std::vector<double> theta_u_excess_sum;
    std::vector<double> epsilon_sum;
    std::vector<double> cs2_mean_sum;
    std::vector<double> scalar_c_mean_sum;
    std::vector<double> kappa_mean_sum;
};

double convective_wstar_slab(const Params& params, double zi) {
    return std::cbrt((params.g / params.theta0) * params.surface_theta_flux * zi);
}

void init_mpi_benchmark_snapshot(MpiBenchmarkSnapshot& snapshot, const Params& params) {
    const std::size_t centers = static_cast<std::size_t>(params.nz);
    const std::size_t faces = static_cast<std::size_t>(params.nz + 1);
    auto init_center = [centers](std::vector<double>& values) {
        values.assign(centers, 0.0);
    };
    auto init_face = [faces](std::vector<double>& values) {
        values.assign(faces, 0.0);
    };
    init_center(snapshot.heat_flux_total);
    init_center(snapshot.heat_flux_resolved);
    init_center(snapshot.heat_flux_sgs);
    init_face(snapshot.heat_flux_face_total);
    init_face(snapshot.heat_flux_face_resolved);
    init_face(snapshot.heat_flux_face_sgs);
    init_center(snapshot.u_mean);
    init_center(snapshot.v_mean);
    init_center(snapshot.w_mean);
    init_center(snapshot.p_mean);
    init_center(snapshot.theta_mean);
    init_center(snapshot.u_var);
    init_center(snapshot.v_var);
    init_center(snapshot.w_var);
    init_center(snapshot.theta_var);
    init_center(snapshot.p_var);
    init_center(snapshot.w3);
    init_center(snapshot.w_transport);
    init_center(snapshot.p_transport);
    init_center(snapshot.alpha_u);
    init_center(snapshot.w_u);
    init_center(snapshot.theta_u_excess);
    init_center(snapshot.epsilon);
    init_center(snapshot.cs2_mean);
    init_center(snapshot.scalar_c_mean);
    init_center(snapshot.kappa_mean);
    snapshot.zi = 0.0;
    snapshot.wstar = 0.0;
}

void reduce_mpi_benchmark_vector(
    const std::vector<double>& local,
    std::vector<double>& global,
    int root,
    MPI_Comm comm) {
    if (global.size() != local.size()) {
        global.assign(local.size(), 0.0);
    }
    MPI_Reduce(local.data(), global.data(), static_cast<int>(local.size()), MPI_DOUBLE, MPI_SUM, root, comm);
}

void finalize_mpi_benchmark_root(MpiBenchmarkSnapshot& snapshot, const Params& params) {
    double min_heat_flux = std::numeric_limits<double>::infinity();
    int min_k = 0;
    for (int k = 0; k < params.nz; ++k) {
        const double value = snapshot.heat_flux_total[static_cast<std::size_t>(k)];
        if (value < min_heat_flux) {
            min_heat_flux = value;
            min_k = k;
        }
    }
    snapshot.zi = (static_cast<double>(min_k) + 0.5) * params.dz();
    snapshot.wstar = convective_wstar_slab(params, snapshot.zi);
}

void collect_mpi_benchmark_sample(
    MpiLocalFlowState& state,
    const Params& params,
    const Slab& slab,
    FftwXY& fft,
    MPI_Comm comm,
    MpiSlabWorkspace& workspace,
    MpiBenchmarkSnapshot& global_snapshot) {
    MpiBenchmarkSnapshot local;
    init_mpi_benchmark_snapshot(local, params);
    init_mpi_benchmark_snapshot(global_snapshot, params);

    const double inv_plane = 1.0 / static_cast<double>(params.nx * params.ny);
    w_to_center_slab(state.w, workspace.w_center, params, slab);

    for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                const std::size_t kk = static_cast<std::size_t>(k);
                local.u_mean[kk] += state.u[n];
                local.v_mean[kk] += state.v[n];
                local.w_mean[kk] += workspace.w_center[n];
                local.p_mean[kk] += state.p[n];
                local.theta_mean[kk] += state.theta[n];
                local.cs2_mean[kk] += state.cs2[n];
                local.scalar_c_mean[kk] += state.scalar_c[n];
            }
        }
        const std::size_t kk = static_cast<std::size_t>(k);
        local.u_mean[kk] *= inv_plane;
        local.v_mean[kk] *= inv_plane;
        local.w_mean[kk] *= inv_plane;
        local.p_mean[kk] *= inv_plane;
        local.theta_mean[kk] *= inv_plane;
        local.cs2_mean[kk] *= inv_plane;
        local.scalar_c_mean[kk] *= inv_plane;
    }

    for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
        int updraft_count = 0;
        const std::size_t kk = static_cast<std::size_t>(k);
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                const double up = state.u[n] - local.u_mean[kk];
                const double vp = state.v[n] - local.v_mean[kk];
                const double wp = workspace.w_center[n] - local.w_mean[kk];
                const double pp = state.p[n] - local.p_mean[kk];
                const double thetap = state.theta[n] - local.theta_mean[kk];
                const double energy = 0.5 * (up * up + vp * vp + wp * wp);
                local.heat_flux_resolved[kk] += wp * thetap;
                local.u_var[kk] += up * up;
                local.v_var[kk] += vp * vp;
                local.w_var[kk] += wp * wp;
                local.theta_var[kk] += thetap * thetap;
                local.p_var[kk] += pp * pp;
                local.w3[kk] += wp * wp * wp;
                local.w_transport[kk] += wp * energy;
                local.p_transport[kk] += pp * wp;
                if (wp > 0.0) {
                    ++updraft_count;
                    local.w_u[kk] += workspace.w_center[n];
                    local.theta_u_excess[kk] += thetap;
                }
            }
        }
        local.heat_flux_resolved[kk] *= inv_plane;
        local.u_var[kk] *= inv_plane;
        local.v_var[kk] *= inv_plane;
        local.w_var[kk] *= inv_plane;
        local.theta_var[kk] *= inv_plane;
        local.p_var[kk] *= inv_plane;
        local.w3[kk] *= inv_plane;
        local.w_transport[kk] *= inv_plane;
        local.p_transport[kk] *= inv_plane;
        local.alpha_u[kk] = static_cast<double>(updraft_count) * inv_plane;
        if (updraft_count > 0) {
            local.w_u[kk] /= static_cast<double>(updraft_count);
            local.theta_u_excess[kk] /= static_cast<double>(updraft_count);
        }
    }

    horizontal_derivatives_center_slab(state.u, workspace.grad.dudx, workspace.grad.dudy, params, slab, fft);
    horizontal_derivatives_center_slab(state.v, workspace.grad.dvdx, workspace.grad.dvdy, params, slab, fft);
    ddz_center_slab(state.u, workspace.grad.dudz, params, slab);
    ddz_center_slab(state.v, workspace.grad.dvdz, params, slab);
    horizontal_derivatives_center_slab(workspace.w_center, workspace.grad.dwdx, workspace.grad.dwdy, params, slab, fft);
    ddz_w_to_center_slab(state.w, workspace.grad.dwdz, params, slab);
    strain_magnitude_slab(workspace.grad, workspace.strain, params, slab);

    clear_center_slab_field(workspace.nu_t, params, slab);
    if (params.sgs_model == "smagorinsky") {
        smagorinsky_eddy_viscosity_slab(state, workspace.strain, workspace.nu_t, params, slab);
    } else if (params.sgs_model == "lasd") {
        eddy_viscosity_from_cs2_slab(state.cs2, workspace.strain, workspace.nu_t, params, slab);
    } else if (params.sgs_model == "amd" || params.sgs_model == "amd_plane_dissipation") {
        const int face_begin = needed_face_begin(slab);
        const int face_count = needed_face_count(params, slab);
        derivative_x_face_range(
            state.w, face_begin, face_count, workspace.sgs_dwdx_face, params, fft);
        derivative_y_face_range(
            state.w, face_begin, face_count, workspace.sgs_dwdy_face, params, fft);
        ddz_center_to_w_face_range(
            state.u, face_begin, face_count, workspace.sgs_dudz_face, params);
        ddz_center_to_w_face_range(
            state.v, face_begin, face_count, workspace.sgs_dvdz_face, params);
        (void)update_sgs_eddy_viscosity_slab(
            state,
            workspace.grad,
            workspace.sgs_dudz_face,
            workspace.sgs_dvdz_face,
            workspace.sgs_dwdx_face,
            workspace.sgs_dwdy_face,
            workspace.strain,
            workspace.w_center,
            workspace.nu_t,
            workspace.amd_buoyancy_prime,
            workspace.amd_db_dx,
            workspace.amd_db_dy,
            workspace.amd_db_dz,
            workspace.amd_invariant_num,
            workspace.amd_invariant_den,
            params,
            slab,
            fft,
            comm,
            workspace.lasd_germano_scratch,
            &workspace.lasd_filter_cache);
    }

    horizontal_derivatives_center_slab(
        state.theta, workspace.scalar_dtheta_dx, workspace.scalar_dtheta_dy, params, slab, fft);
    ddz_center_slab(state.theta, workspace.scalar_dtheta_dz_center, params, slab);
    if (params.scalar_sgs_model == "amd"
        || params.scalar_sgs_model == "amd_shared"
        || params.scalar_sgs_model == "amd_plane_dissipation") {
        ddz_center_to_w_face_range(
            state.theta,
            needed_face_begin(slab),
            needed_face_count(params, slab),
            workspace.scalar_dtheta_dz_w,
            params);
    } else {
        ddz_center_to_w_face_slab(state.theta, workspace.scalar_dtheta_dz_w, params, slab);
    }
    scalar_eddy_diffusivity_slab(
        state,
        workspace.grad,
        workspace.scalar_dtheta_dx,
        workspace.scalar_dtheta_dy,
        workspace.scalar_dtheta_dz_center,
        workspace.scalar_dtheta_dz_w,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        &workspace.sgs_dudz_face,
        &workspace.sgs_dvdz_face,
        &workspace.sgs_dwdx_face,
        &workspace.sgs_dwdy_face,
        workspace.nu_t,
        workspace.strain,
        state.scalar_c,
        params.scalar_diffusivity,
        params.prandtl_t,
        params,
        slab,
        comm,
        940,
        workspace.scalar_kappa_center);
    exchange_neighbor_planes(workspace.scalar_kappa_center, slab.k_begin, slab.k_count, params.nz, 910, params, slab, comm);
    center_to_w_face_slab(workspace.scalar_kappa_center, workspace.scalar_kappa_w, params, slab);
    clear_face_slab_field(workspace.scalar_qz, params, slab);
    for (int k = slab.face_begin; k < slab.face_begin + slab.face_count; ++k) {
        if (k <= 0 || k >= params.nz) {
            continue;
        }
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t face = z_face_idx(params, i, j, k);
                workspace.scalar_qz[face] = -workspace.scalar_kappa_w[face] * workspace.scalar_dtheta_dz_w[face];
            }
        }
    }
    if (slab.rank == 0) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                workspace.scalar_qz[z_face_idx(params, i, j, 0)] = params.surface_theta_flux;
            }
        }
    }
    if (slab.rank == slab.size - 1) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                workspace.scalar_qz[z_face_idx(params, i, j, params.nz)] = 0.0;
            }
        }
    }
    exchange_neighbor_planes(workspace.scalar_qz, slab.face_begin, slab.face_count, params.nz + 1, 920, params, slab, comm);

    center_to_w_face_slab(state.theta, workspace.scalar_theta_on_w, params, slab);
    std::vector<double> w_face_mean(static_cast<std::size_t>(params.nz + 1), 0.0);
    std::vector<double> theta_face_mean(static_cast<std::size_t>(params.nz + 1), 0.0);
    for (int k = slab.face_begin; k < slab.face_begin + slab.face_count; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t face = z_face_idx(params, i, j, k);
                w_face_mean[static_cast<std::size_t>(k)] += state.w[face];
                theta_face_mean[static_cast<std::size_t>(k)] += workspace.scalar_theta_on_w[face];
            }
        }
        w_face_mean[static_cast<std::size_t>(k)] *= inv_plane;
        theta_face_mean[static_cast<std::size_t>(k)] *= inv_plane;
    }
    for (int k = slab.face_begin; k < slab.face_begin + slab.face_count; ++k) {
        double resolved = 0.0;
        double sgs = 0.0;
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t face = z_face_idx(params, i, j, k);
                resolved += (state.w[face] - w_face_mean[static_cast<std::size_t>(k)])
                    * (workspace.scalar_theta_on_w[face] - theta_face_mean[static_cast<std::size_t>(k)]);
                sgs += workspace.scalar_qz[face];
            }
        }
        const std::size_t kk = static_cast<std::size_t>(k);
        local.heat_flux_face_resolved[kk] = resolved * inv_plane;
        local.heat_flux_face_sgs[kk] = sgs * inv_plane;
        local.heat_flux_face_total[kk] = local.heat_flux_face_resolved[kk] + local.heat_flux_face_sgs[kk];
    }

    for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
        double qz_center_mean = 0.0;
        double kappa_mean = 0.0;
        double epsilon_mean = 0.0;
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                qz_center_mean += 0.5 * (
                    workspace.scalar_qz[z_face_idx(params, i, j, k)]
                    + workspace.scalar_qz[z_face_idx(params, i, j, k + 1)]);
                kappa_mean += workspace.scalar_kappa_center[n];
                epsilon_mean += workspace.nu_t[n] * workspace.strain[n] * workspace.strain[n];
            }
        }
        const std::size_t kk = static_cast<std::size_t>(k);
        local.heat_flux_sgs[kk] = qz_center_mean * inv_plane;
        local.kappa_mean[kk] = kappa_mean * inv_plane;
        local.epsilon[kk] = epsilon_mean * inv_plane;
        local.heat_flux_total[kk] = local.heat_flux_resolved[kk] + local.heat_flux_sgs[kk];
    }

    constexpr int root = 0;
    reduce_mpi_benchmark_vector(local.heat_flux_total, global_snapshot.heat_flux_total, root, comm);
    reduce_mpi_benchmark_vector(local.heat_flux_resolved, global_snapshot.heat_flux_resolved, root, comm);
    reduce_mpi_benchmark_vector(local.heat_flux_sgs, global_snapshot.heat_flux_sgs, root, comm);
    reduce_mpi_benchmark_vector(local.heat_flux_face_total, global_snapshot.heat_flux_face_total, root, comm);
    reduce_mpi_benchmark_vector(local.heat_flux_face_resolved, global_snapshot.heat_flux_face_resolved, root, comm);
    reduce_mpi_benchmark_vector(local.heat_flux_face_sgs, global_snapshot.heat_flux_face_sgs, root, comm);
    reduce_mpi_benchmark_vector(local.u_mean, global_snapshot.u_mean, root, comm);
    reduce_mpi_benchmark_vector(local.v_mean, global_snapshot.v_mean, root, comm);
    reduce_mpi_benchmark_vector(local.w_mean, global_snapshot.w_mean, root, comm);
    reduce_mpi_benchmark_vector(local.p_mean, global_snapshot.p_mean, root, comm);
    reduce_mpi_benchmark_vector(local.theta_mean, global_snapshot.theta_mean, root, comm);
    reduce_mpi_benchmark_vector(local.u_var, global_snapshot.u_var, root, comm);
    reduce_mpi_benchmark_vector(local.v_var, global_snapshot.v_var, root, comm);
    reduce_mpi_benchmark_vector(local.w_var, global_snapshot.w_var, root, comm);
    reduce_mpi_benchmark_vector(local.theta_var, global_snapshot.theta_var, root, comm);
    reduce_mpi_benchmark_vector(local.p_var, global_snapshot.p_var, root, comm);
    reduce_mpi_benchmark_vector(local.w3, global_snapshot.w3, root, comm);
    reduce_mpi_benchmark_vector(local.w_transport, global_snapshot.w_transport, root, comm);
    reduce_mpi_benchmark_vector(local.p_transport, global_snapshot.p_transport, root, comm);
    reduce_mpi_benchmark_vector(local.alpha_u, global_snapshot.alpha_u, root, comm);
    reduce_mpi_benchmark_vector(local.w_u, global_snapshot.w_u, root, comm);
    reduce_mpi_benchmark_vector(local.theta_u_excess, global_snapshot.theta_u_excess, root, comm);
    reduce_mpi_benchmark_vector(local.epsilon, global_snapshot.epsilon, root, comm);
    reduce_mpi_benchmark_vector(local.cs2_mean, global_snapshot.cs2_mean, root, comm);
    reduce_mpi_benchmark_vector(local.scalar_c_mean, global_snapshot.scalar_c_mean, root, comm);
    reduce_mpi_benchmark_vector(local.kappa_mean, global_snapshot.kappa_mean, root, comm);

    if (slab.rank == root) {
        finalize_mpi_benchmark_root(global_snapshot, params);
    }
}

void add_mpi_benchmark_sample(MpiBenchmarkAccumulator& accumulator, const MpiBenchmarkSnapshot& snapshot) {
    if (accumulator.heat_flux_sum.empty()) {
        auto init = [](std::vector<double>& target, const std::vector<double>& source) {
            target.assign(source.size(), 0.0);
        };
        init(accumulator.heat_flux_sum, snapshot.heat_flux_total);
        init(accumulator.heat_flux_resolved_sum, snapshot.heat_flux_resolved);
        init(accumulator.heat_flux_sgs_sum, snapshot.heat_flux_sgs);
        init(accumulator.heat_flux_face_sum, snapshot.heat_flux_face_total);
        init(accumulator.heat_flux_face_resolved_sum, snapshot.heat_flux_face_resolved);
        init(accumulator.heat_flux_face_sgs_sum, snapshot.heat_flux_face_sgs);
        init(accumulator.u_mean_sum, snapshot.u_mean);
        init(accumulator.v_mean_sum, snapshot.v_mean);
        init(accumulator.w_mean_sum, snapshot.w_mean);
        init(accumulator.p_mean_sum, snapshot.p_mean);
        init(accumulator.theta_mean_sum, snapshot.theta_mean);
        init(accumulator.u_var_sum, snapshot.u_var);
        init(accumulator.v_var_sum, snapshot.v_var);
        init(accumulator.w_var_sum, snapshot.w_var);
        init(accumulator.theta_var_sum, snapshot.theta_var);
        init(accumulator.p_var_sum, snapshot.p_var);
        init(accumulator.w3_sum, snapshot.w3);
        init(accumulator.w_transport_sum, snapshot.w_transport);
        init(accumulator.p_transport_sum, snapshot.p_transport);
        init(accumulator.alpha_u_sum, snapshot.alpha_u);
        init(accumulator.w_u_sum, snapshot.w_u);
        init(accumulator.theta_u_excess_sum, snapshot.theta_u_excess);
        init(accumulator.epsilon_sum, snapshot.epsilon);
        init(accumulator.cs2_mean_sum, snapshot.cs2_mean);
        init(accumulator.scalar_c_mean_sum, snapshot.scalar_c_mean);
        init(accumulator.kappa_mean_sum, snapshot.kappa_mean);
    }
    ++accumulator.sample_count;
    accumulator.zi_sum += snapshot.zi;
    accumulator.wstar_sum += snapshot.wstar;
    auto add = [](std::vector<double>& target, const std::vector<double>& source) {
        for (std::size_t k = 0; k < source.size(); ++k) {
            target[k] += source[k];
        }
    };
    add(accumulator.heat_flux_sum, snapshot.heat_flux_total);
    add(accumulator.heat_flux_resolved_sum, snapshot.heat_flux_resolved);
    add(accumulator.heat_flux_sgs_sum, snapshot.heat_flux_sgs);
    add(accumulator.heat_flux_face_sum, snapshot.heat_flux_face_total);
    add(accumulator.heat_flux_face_resolved_sum, snapshot.heat_flux_face_resolved);
    add(accumulator.heat_flux_face_sgs_sum, snapshot.heat_flux_face_sgs);
    add(accumulator.u_mean_sum, snapshot.u_mean);
    add(accumulator.v_mean_sum, snapshot.v_mean);
    add(accumulator.w_mean_sum, snapshot.w_mean);
    add(accumulator.p_mean_sum, snapshot.p_mean);
    add(accumulator.theta_mean_sum, snapshot.theta_mean);
    add(accumulator.u_var_sum, snapshot.u_var);
    add(accumulator.v_var_sum, snapshot.v_var);
    add(accumulator.w_var_sum, snapshot.w_var);
    add(accumulator.theta_var_sum, snapshot.theta_var);
    add(accumulator.p_var_sum, snapshot.p_var);
    add(accumulator.w3_sum, snapshot.w3);
    add(accumulator.w_transport_sum, snapshot.w_transport);
    add(accumulator.p_transport_sum, snapshot.p_transport);
    add(accumulator.alpha_u_sum, snapshot.alpha_u);
    add(accumulator.w_u_sum, snapshot.w_u);
    add(accumulator.theta_u_excess_sum, snapshot.theta_u_excess);
    add(accumulator.epsilon_sum, snapshot.epsilon);
    add(accumulator.cs2_mean_sum, snapshot.cs2_mean);
    add(accumulator.scalar_c_mean_sum, snapshot.scalar_c_mean);
    add(accumulator.kappa_mean_sum, snapshot.kappa_mean);
}

void ensure_directory_slab(const std::string& path) {
    if (path.empty()) {
        return;
    }
    std::string current;
    std::size_t start = 0;
    if (!path.empty() && path[0] == '/') {
        current = "/";
        start = 1;
    }
    while (start <= path.size()) {
        const std::size_t slash = path.find('/', start);
        const std::string part = path.substr(start, slash == std::string::npos ? std::string::npos : slash - start);
        if (!part.empty()) {
            if (!current.empty() && current.back() != '/') {
                current += '/';
            }
            current += part;
            if (::mkdir(current.c_str(), 0775) != 0 && errno != EEXIST) {
                throw std::runtime_error("failed to create directory: " + current);
            }
        }
        if (slash == std::string::npos) {
            break;
        }
        start = slash + 1;
    }
}

std::string join_path_slab(const std::string& directory, const std::string& name) {
    if (directory.empty()) {
        return name;
    }
    return directory.back() == '/' ? directory + name : directory + "/" + name;
}

bool should_write_frame_slab(const Params& params, int step_number) {
    if (!params.frame_dump_enabled) {
        return false;
    }
    if (step_number < params.frame_dump_start_step) {
        return false;
    }
    if (params.frame_dump_end_step >= 0 && step_number > params.frame_dump_end_step) {
        return false;
    }
    return step_number % params.frame_dump_every == 0 || step_number == params.steps;
}

std::string frame_slice_filename_slab(int step_number, int y_index) {
    std::ostringstream name;
    name << "xz_slice_step" << std::setw(6) << std::setfill('0') << step_number
         << "_y" << y_index << ".csv";
    return name.str();
}

int frame_xy_k_index_slab(const Params& params) {
    if (params.frame_dump_z_height < 0.0) {
        return -1;
    }
    const double raw = params.frame_dump_z_height / params.dz() - 0.5;
    return std::clamp(static_cast<int>(std::floor(raw + 1.0e-12)), 0, params.nz - 1);
}

std::string frame_xy_filename_slab(int step_number, int k_index, const Params& params) {
    const double z = (static_cast<double>(k_index) + 0.5) * params.dz();
    std::ostringstream name;
    name << "xy_slice_step" << std::setw(6) << std::setfill('0') << step_number
         << "_z" << std::setw(5) << std::setfill('0') << static_cast<int>(std::lround(z)) << "m.csv";
    return name.str();
}

void write_mpi_xz_slice_csv(
    const MpiLocalFlowState& state,
    const Params& params,
    const Slab& slab,
    MPI_Comm comm,
    int step_number) {
    const int columns = 12;
    const int y_index = params.frame_dump_y_index < 0 ? params.ny / 2 : params.frame_dump_y_index;
    std::vector<double> local;
    local.reserve(static_cast<std::size_t>(slab.k_count * params.nx * columns));
    for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
        const double z = (static_cast<double>(k) + 0.5) * params.dz();
        for (int i = 0; i < params.nx; ++i) {
            const std::size_t n = idx(params, i, y_index, k);
            const double x = (static_cast<double>(i) + 0.5) * params.dx();
            const double wc = 0.5 * (
                state.w[z_face_idx(params, i, y_index, k)]
                + state.w[z_face_idx(params, i, y_index, k + 1)]);
            const double theta_l = params.moisture_enabled ? state.theta_l[n] : state.theta[n];
            const double qt = params.moisture_enabled ? state.qt[n] : 0.0;
            const double qv = params.moisture_enabled ? state.qv[n] : 0.0;
            const double ql = params.moisture_enabled ? state.ql[n] : 0.0;
            const double theta_v = state.theta[n] * (1.0 + 0.61 * qv - ql);
            local.push_back(static_cast<double>(i));
            local.push_back(static_cast<double>(k));
            local.push_back(x);
            local.push_back(z);
            local.push_back(state.u[n]);
            local.push_back(state.v[n]);
            local.push_back(wc);
            local.push_back(theta_l);
            local.push_back(qt);
            local.push_back(qv);
            local.push_back(ql);
            local.push_back(theta_v);
        }
    }

    const int local_count = static_cast<int>(local.size());
    std::vector<int> counts;
    if (slab.rank == 0) {
        counts.resize(static_cast<std::size_t>(slab.size));
    }
    MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm);

    std::vector<int> displacements;
    std::vector<double> global;
    if (slab.rank == 0) {
        displacements.resize(static_cast<std::size_t>(slab.size), 0);
        int total = 0;
        for (int rank = 0; rank < slab.size; ++rank) {
            displacements[static_cast<std::size_t>(rank)] = total;
            total += counts[static_cast<std::size_t>(rank)];
        }
        global.resize(static_cast<std::size_t>(total));
    }

    MPI_Gatherv(
        local.data(),
        local_count,
        MPI_DOUBLE,
        global.data(),
        counts.data(),
        displacements.data(),
        MPI_DOUBLE,
        0,
        comm);

    if (slab.rank != 0) {
        return;
    }
    ensure_directory_slab(params.frame_dump_output_dir);
    const std::string path = join_path_slab(
        params.frame_dump_output_dir,
        frame_slice_filename_slab(step_number, y_index));
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open MPI slab frame slice output: " + path);
    }
    out << "step,time_s,y_index,i,k,x_m,z_m,u,v,w,theta_l,qt,qv,ql,theta_v\n";
    out << std::setprecision(12);
    for (std::size_t offset = 0; offset < global.size(); offset += columns) {
        out << step_number << ','
            << static_cast<double>(step_number) * params.dt << ','
            << y_index;
        for (int c = 0; c < columns; ++c) {
            out << ',' << global[offset + static_cast<std::size_t>(c)];
        }
        out << '\n';
    }
}

void write_mpi_xy_slice_csv(
    const MpiLocalFlowState& state,
    const Params& params,
    const Slab& slab,
    MPI_Comm comm,
    int step_number) {
    const int k_index = frame_xy_k_index_slab(params);
    if (k_index < 0) {
        return;
    }
    const int columns = 14;
    std::vector<double> local;
    if (owns_center_plane(slab, k_index)) {
        local.reserve(static_cast<std::size_t>(params.nx * params.ny * columns));
        const double z = (static_cast<double>(k_index) + 0.5) * params.dz();
        for (int j = 0; j < params.ny; ++j) {
            const double y = (static_cast<double>(j) + 0.5) * params.dy();
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k_index);
                const double x = (static_cast<double>(i) + 0.5) * params.dx();
                const double wc = 0.5 * (
                    state.w[z_face_idx(params, i, j, k_index)]
                    + state.w[z_face_idx(params, i, j, k_index + 1)]);
                const double theta_l = params.moisture_enabled ? state.theta_l[n] : state.theta[n];
                const double qt = params.moisture_enabled ? state.qt[n] : 0.0;
                const double qv = params.moisture_enabled ? state.qv[n] : 0.0;
                const double ql = params.moisture_enabled ? state.ql[n] : 0.0;
                const double theta_v = state.theta[n] * (1.0 + 0.61 * qv - ql);
                local.push_back(static_cast<double>(i));
                local.push_back(static_cast<double>(j));
                local.push_back(static_cast<double>(k_index));
                local.push_back(x);
                local.push_back(y);
                local.push_back(z);
                local.push_back(state.u[n]);
                local.push_back(state.v[n]);
                local.push_back(wc);
                local.push_back(theta_l);
                local.push_back(qt);
                local.push_back(qv);
                local.push_back(ql);
                local.push_back(theta_v);
            }
        }
    }

    const int local_count = static_cast<int>(local.size());
    std::vector<int> counts;
    if (slab.rank == 0) {
        counts.resize(static_cast<std::size_t>(slab.size));
    }
    MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm);

    std::vector<int> displacements;
    std::vector<double> global;
    if (slab.rank == 0) {
        displacements.resize(static_cast<std::size_t>(slab.size), 0);
        int total = 0;
        for (int rank = 0; rank < slab.size; ++rank) {
            displacements[static_cast<std::size_t>(rank)] = total;
            total += counts[static_cast<std::size_t>(rank)];
        }
        global.resize(static_cast<std::size_t>(total));
    }

    MPI_Gatherv(
        local.data(),
        local_count,
        MPI_DOUBLE,
        global.data(),
        counts.data(),
        displacements.data(),
        MPI_DOUBLE,
        0,
        comm);

    if (slab.rank != 0) {
        return;
    }
    ensure_directory_slab(params.frame_dump_output_dir);
    const std::string path = join_path_slab(
        params.frame_dump_output_dir,
        frame_xy_filename_slab(step_number, k_index, params));
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open MPI slab x-y frame output: " + path);
    }
    out << "step,time_s,i,j,k_index,x_m,y_m,z_m,u,v,w,theta_l,qt,qv,ql,theta_v\n";
    out << std::setprecision(12);
    for (std::size_t offset = 0; offset < global.size(); offset += columns) {
        out << step_number << ','
            << static_cast<double>(step_number) * params.dt;
        for (int c = 0; c < columns; ++c) {
            out << ',' << global[offset + static_cast<std::size_t>(c)];
        }
        out << '\n';
    }
}

int nearest_center_k_index(double z, const Params& params) {
    const double raw = z / params.dz() - 0.5;
    return std::clamp(static_cast<int>(std::lround(raw)), 0, params.nz - 1);
}

void write_mpi_bomex_final_masks(
    const MpiLocalFlowState& state,
    const Params& params,
    const Slab& slab,
    MPI_Comm comm) {
    if (!params.bomex_diagnostics_enabled || params.bomex_output_dir.empty()
        || params.initial_condition != "bomex" || !params.moisture_enabled) {
        return;
    }

    const auto& thresholds = bomex_cloud_thresholds();
    const std::size_t threshold_count = thresholds.size();
    const std::size_t column_count = static_cast<std::size_t>(params.nx * params.ny);
    std::vector<int> local_column_masks(threshold_count * column_count, 0);
    for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                const std::size_t column = static_cast<std::size_t>(j * params.nx + i);
                for (std::size_t threshold_index = 0; threshold_index < threshold_count; ++threshold_index) {
                    if (state.ql[n] > thresholds[threshold_index]) {
                        local_column_masks[threshold_index * column_count + column] = 1;
                    }
                }
            }
        }
    }

    std::vector<int> global_column_masks(local_column_masks.size(), 0);
    MPI_Reduce(
        local_column_masks.data(),
        global_column_masks.data(),
        static_cast<int>(local_column_masks.size()),
        MPI_INT,
        MPI_MAX,
        0,
        comm);

    const std::vector<double> requested_heights{540.0, 620.0, 780.0, 980.0};
    std::vector<int> requested_k;
    requested_k.reserve(requested_heights.size());
    for (double height : requested_heights) {
        requested_k.push_back(nearest_center_k_index(height, params));
    }

    std::vector<std::vector<double>> local_ql_planes;
    local_ql_planes.reserve(requested_k.size());
    for (int k_index : requested_k) {
        std::vector<double> local_plane(column_count, 0.0);
        if (owns_center_plane(slab, k_index)) {
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    local_plane[static_cast<std::size_t>(j * params.nx + i)] =
                        state.ql[idx(params, i, j, k_index)];
                }
            }
        }
        local_ql_planes.push_back(std::move(local_plane));
    }

    std::vector<std::vector<double>> global_ql_planes;
    global_ql_planes.reserve(local_ql_planes.size());
    for (const auto& local_plane : local_ql_planes) {
        std::vector<double> global_plane(column_count, 0.0);
        MPI_Reduce(
            local_plane.data(),
            global_plane.data(),
            static_cast<int>(column_count),
            MPI_DOUBLE,
            MPI_SUM,
            0,
            comm);
        global_ql_planes.push_back(std::move(global_plane));
    }

    if (slab.rank != 0) {
        return;
    }

    ensure_directory_slab(params.bomex_output_dir);
    {
        const std::string path = join_path_slab(params.bomex_output_dir, "bomex_final_cloud_column_masks.csv");
        std::ofstream out(path);
        if (!out) {
            throw std::runtime_error("failed to open BOMEX cloud column mask output: " + path);
        }
        out << "i,j,x_m,y_m";
        for (double threshold : thresholds) {
            out << ",column_cloud_ql_gt_" << std::scientific << std::setprecision(0) << threshold;
        }
        out << std::defaultfloat << '\n';
        out << std::setprecision(12);
        for (int j = 0; j < params.ny; ++j) {
            const double y = (static_cast<double>(j) + 0.5) * params.dy();
            for (int i = 0; i < params.nx; ++i) {
                const double x = (static_cast<double>(i) + 0.5) * params.dx();
                const std::size_t column = static_cast<std::size_t>(j * params.nx + i);
                out << i << ',' << j << ',' << x << ',' << y;
                for (std::size_t threshold_index = 0; threshold_index < threshold_count; ++threshold_index) {
                    out << ',' << global_column_masks[threshold_index * column_count + column];
                }
                out << '\n';
            }
        }
    }
    {
        const std::string path = join_path_slab(params.bomex_output_dir, "bomex_final_column_cover_by_threshold.csv");
        std::ofstream out(path);
        if (!out) {
            throw std::runtime_error("failed to open BOMEX final column cover output: " + path);
        }
        out << "ql_threshold_kg_kg,total_cloud_cover\n";
        out << std::setprecision(12);
        for (std::size_t threshold_index = 0; threshold_index < threshold_count; ++threshold_index) {
            const auto begin = global_column_masks.begin()
                + static_cast<std::ptrdiff_t>(threshold_index * column_count);
            const auto end = begin + static_cast<std::ptrdiff_t>(column_count);
            const double cover = static_cast<double>(std::count(begin, end, 1))
                / static_cast<double>(params.nx * params.ny);
            out << thresholds[threshold_index] << ',' << cover << '\n';
        }
    }
    for (std::size_t plane_index = 0; plane_index < requested_k.size(); ++plane_index) {
        const int k_index = requested_k[plane_index];
        const double z = (static_cast<double>(k_index) + 0.5) * params.dz();
        std::ostringstream name;
        name << "bomex_final_ql_mask_z"
             << std::setw(5) << std::setfill('0') << static_cast<int>(std::lround(z)) << "m.csv";
        const std::string path = join_path_slab(params.bomex_output_dir, name.str());
        std::ofstream out(path);
        if (!out) {
            throw std::runtime_error("failed to open BOMEX ql mask output: " + path);
        }
        out << "i,j,k_index,x_m,y_m,z_m,ql_kg_kg";
        for (double threshold : thresholds) {
            out << ",ql_gt_" << std::scientific << std::setprecision(0) << threshold;
        }
        out << std::defaultfloat << '\n';
        out << std::setprecision(12);
        const auto& ql_plane = global_ql_planes[plane_index];
        for (int j = 0; j < params.ny; ++j) {
            const double y = (static_cast<double>(j) + 0.5) * params.dy();
            for (int i = 0; i < params.nx; ++i) {
                const double x = (static_cast<double>(i) + 0.5) * params.dx();
                const std::size_t column = static_cast<std::size_t>(j * params.nx + i);
                const double ql = ql_plane[column];
                out << i << ',' << j << ',' << k_index << ','
                    << x << ',' << y << ',' << z << ',' << ql;
                for (double threshold : thresholds) {
                    out << ',' << (ql > threshold ? 1 : 0);
                }
                out << '\n';
            }
        }
    }
}

double averaged_at_slab(const std::vector<double>& values, int k, double inv_count) {
    return values[static_cast<std::size_t>(k)] * inv_count;
}

double gradient_at_slab(const std::vector<double>& values, int k, const Params& params, double inv_count) {
    if (params.nz <= 1) {
        return 0.0;
    }
    if (k == 0) {
        return (averaged_at_slab(values, 1, inv_count) - averaged_at_slab(values, 0, inv_count)) / params.dz();
    }
    if (k == params.nz - 1) {
        return (averaged_at_slab(values, params.nz - 1, inv_count) - averaged_at_slab(values, params.nz - 2, inv_count)) / params.dz();
    }
    return (averaged_at_slab(values, k + 1, inv_count) - averaged_at_slab(values, k - 1, inv_count)) / (2.0 * params.dz());
}

void print_mpi_benchmark_summary(const MpiBenchmarkAccumulator& accumulator, const Params& params) {
    if (!params.benchmark_enabled) {
        return;
    }
    if (accumulator.sample_count == 0) {
        std::cout << "[benchmark] no samples in configured averaging window\n";
        return;
    }

    const double inv_count = 1.0 / static_cast<double>(accumulator.sample_count);
    const double wstar0 = convective_wstar_slab(params, params.z_i);
    double min_heat_flux = std::numeric_limits<double>::infinity();
    int min_heat_flux_k = 0;
    for (int k = 0; k < params.nz; ++k) {
        const double value = accumulator.heat_flux_sum[static_cast<std::size_t>(k)] * inv_count;
        if (value < min_heat_flux) {
            min_heat_flux = value;
            min_heat_flux_k = k;
        }
    }
    const double zi_mean = (static_cast<double>(min_heat_flux_k) + 0.5) * params.dz();
    const double wstar_mean = convective_wstar_slab(params, zi_mean);
    std::cout << "[benchmark] Moeng Table 3 comparison\n";
    std::cout << "  sample_count: " << accumulator.sample_count << '\n';
    std::cout << "  zi_over_zi0: " << std::setprecision(6) << zi_mean / params.z_i << '\n';
    std::cout << "  wstar_over_wstar0: " << std::setprecision(6) << wstar_mean / wstar0 << '\n';
    std::cout << "  entrainment_ratio: " << std::setprecision(6) << -min_heat_flux / params.surface_theta_flux << '\n';
}

void write_mpi_benchmark_outputs(const MpiBenchmarkAccumulator& accumulator, const Params& params) {
    if (!params.benchmark_enabled || params.benchmark_output_dir.empty()) {
        return;
    }
    if (accumulator.sample_count == 0 || accumulator.heat_flux_sum.empty()) {
        std::cout << "[benchmark] no profile CSV written because no samples were collected\n";
        return;
    }

    ensure_directory_slab(params.benchmark_output_dir);
    const double inv_count = 1.0 / static_cast<double>(accumulator.sample_count);
    const double wstar0 = convective_wstar_slab(params, params.z_i);
    double min_heat_flux = std::numeric_limits<double>::infinity();
    int min_heat_flux_k = 0;
    for (int k = 0; k < params.nz; ++k) {
        const double value = averaged_at_slab(accumulator.heat_flux_sum, k, inv_count);
        if (value < min_heat_flux) {
            min_heat_flux = value;
            min_heat_flux_k = k;
        }
    }

    const double zi_mean = (static_cast<double>(min_heat_flux_k) + 0.5) * params.dz();
    const double wstar = convective_wstar_slab(params, zi_mean);
    const double theta_star = params.surface_theta_flux / wstar;
    const double transport_norm = wstar * wstar * wstar / zi_mean;
    const double instantaneous_zi_mean = accumulator.zi_sum * inv_count;
    const double instantaneous_wstar_mean = accumulator.wstar_sum * inv_count;

    double theta_mixed_sum = 0.0;
    int theta_mixed_count = 0;
    for (int k = 0; k < params.nz; ++k) {
        const double z = (static_cast<double>(k) + 0.5) * params.dz();
        if (z < 0.8 * zi_mean) {
            theta_mixed_sum += averaged_at_slab(accumulator.theta_mean_sum, k, inv_count);
            ++theta_mixed_count;
        }
    }
    const double theta_mixed_mean = theta_mixed_count > 0 ? theta_mixed_sum / static_cast<double>(theta_mixed_count) : 0.0;

    {
        std::ofstream out(join_path_slab(params.benchmark_output_dir, "summary.csv"));
        if (!out) {
            throw std::runtime_error("failed to open benchmark summary output");
        }
        out << std::setprecision(17);
        out << "quantity,value\n";
        out << "sample_count," << accumulator.sample_count << "\n";
        out << "zi_mean," << zi_mean << "\n";
        out << "zi_over_zi0," << zi_mean / params.z_i << "\n";
        out << "instantaneous_zi_mean," << instantaneous_zi_mean << "\n";
        out << "instantaneous_zi_mean_over_zi0," << instantaneous_zi_mean / params.z_i << "\n";
        out << "wstar_mean," << wstar << "\n";
        out << "wstar_over_wstar0," << wstar / wstar0 << "\n";
        out << "instantaneous_wstar_mean," << instantaneous_wstar_mean << "\n";
        out << "instantaneous_wstar_mean_over_wstar0," << instantaneous_wstar_mean / wstar0 << "\n";
        out << "theta_star_mean," << theta_star << "\n";
        out << "theta_mixed_layer_mean," << theta_mixed_mean << "\n";
        out << "surface_theta_flux," << params.surface_theta_flux << "\n";
        out << "entrainment_ratio," << -min_heat_flux / params.surface_theta_flux << "\n";
        out << "entrainment_layer_z," << (static_cast<double>(min_heat_flux_k) + 0.5) * params.dz() << "\n";
        out << "entrainment_layer_z_over_zi0," << ((static_cast<double>(min_heat_flux_k) + 0.5) * params.dz()) / params.z_i << "\n";
        out << "z_i0," << params.z_i << "\n";
        out << "wstar0," << wstar0 << "\n";
        out << "theta0," << params.theta0 << "\n";
        out << "g," << params.g << "\n";
    }

    {
        std::ofstream out(join_path_slab(params.benchmark_output_dir, "profiles.csv"));
        if (!out) {
            throw std::runtime_error("failed to open benchmark profile output");
        }
        out << std::setprecision(17);
        out << "z,z_over_zi,z_over_zi0,u_mean,v_mean,w_mean,p_mean,theta_mean,"
            << "heat_flux_over_qs,heat_flux_resolved_over_qs,heat_flux_sgs_over_qs,heat_flux_total_over_qs,"
            << "epsilon_zi_over_wstar3,u_var_over_wstar_sq,v_var_over_wstar_sq,horizontal_var_over_wstar_sq,"
            << "w_var_over_wstar_sq,theta_var_over_thetastar_sq,p_var_over_wstar4,w3_over_wstar3,skewness,"
            << "alpha_u,w_u_over_wstar,theta_u_excess_over_thetastar,cs2_mean,scalar_c_mean,kappa_mean,"
            << "buoyancy_production,d_w_transport,d_p_transport\n";
        for (int k = 0; k < params.nz; ++k) {
            const double z = (static_cast<double>(k) + 0.5) * params.dz();
            const double heat_flux = averaged_at_slab(accumulator.heat_flux_sum, k, inv_count);
            const double heat_flux_resolved = averaged_at_slab(accumulator.heat_flux_resolved_sum, k, inv_count);
            const double heat_flux_sgs = averaged_at_slab(accumulator.heat_flux_sgs_sum, k, inv_count);
            const double u_var = averaged_at_slab(accumulator.u_var_sum, k, inv_count);
            const double v_var = averaged_at_slab(accumulator.v_var_sum, k, inv_count);
            const double w_var = averaged_at_slab(accumulator.w_var_sum, k, inv_count);
            const double theta_var = averaged_at_slab(accumulator.theta_var_sum, k, inv_count);
            const double p_var = averaged_at_slab(accumulator.p_var_sum, k, inv_count);
            const double w3 = averaged_at_slab(accumulator.w3_sum, k, inv_count);
            const double skewness = w_var > 0.0 ? w3 / std::pow(w_var, 1.5) : 0.0;
            const double buoyancy = (params.g / params.theta0) * heat_flux / transport_norm;
            const double d_w_transport = gradient_at_slab(accumulator.w_transport_sum, k, params, inv_count) / transport_norm;
            const double d_p_transport = gradient_at_slab(accumulator.p_transport_sum, k, params, inv_count) / transport_norm;

            out << z << ',' << z / zi_mean << ',' << z / params.z_i << ','
                << averaged_at_slab(accumulator.u_mean_sum, k, inv_count) << ','
                << averaged_at_slab(accumulator.v_mean_sum, k, inv_count) << ','
                << averaged_at_slab(accumulator.w_mean_sum, k, inv_count) << ','
                << averaged_at_slab(accumulator.p_mean_sum, k, inv_count) << ','
                << averaged_at_slab(accumulator.theta_mean_sum, k, inv_count) << ','
                << heat_flux / params.surface_theta_flux << ','
                << heat_flux_resolved / params.surface_theta_flux << ','
                << heat_flux_sgs / params.surface_theta_flux << ','
                << heat_flux / params.surface_theta_flux << ','
                << averaged_at_slab(accumulator.epsilon_sum, k, inv_count) * zi_mean / (wstar * wstar * wstar) << ','
                << u_var / (wstar * wstar) << ','
                << v_var / (wstar * wstar) << ','
                << 0.5 * (u_var + v_var) / (wstar * wstar) << ','
                << w_var / (wstar * wstar) << ','
                << theta_var / (theta_star * theta_star) << ','
                << p_var / std::pow(wstar, 4.0) << ','
                << w3 / (wstar * wstar * wstar) << ','
                << skewness << ','
                << averaged_at_slab(accumulator.alpha_u_sum, k, inv_count) << ','
                << averaged_at_slab(accumulator.w_u_sum, k, inv_count) / wstar << ','
                << averaged_at_slab(accumulator.theta_u_excess_sum, k, inv_count) / theta_star << ','
                << averaged_at_slab(accumulator.cs2_mean_sum, k, inv_count) << ','
                << averaged_at_slab(accumulator.scalar_c_mean_sum, k, inv_count) << ','
                << averaged_at_slab(accumulator.kappa_mean_sum, k, inv_count) << ','
                << buoyancy << ',' << d_w_transport << ',' << d_p_transport << '\n';
        }
    }

    {
        std::ofstream out(join_path_slab(params.benchmark_output_dir, "heat_flux_faces.csv"));
        if (!out) {
            throw std::runtime_error("failed to open benchmark face heat-flux output");
        }
        out << std::setprecision(17);
        out << "z,z_over_zi,z_over_zi0,heat_flux_over_qs,heat_flux_resolved_over_qs,heat_flux_sgs_over_qs\n";
        for (int k = 0; k <= params.nz; ++k) {
            const double z = static_cast<double>(k) * params.dz();
            const double heat_flux = averaged_at_slab(accumulator.heat_flux_face_sum, k, inv_count);
            const double resolved = averaged_at_slab(accumulator.heat_flux_face_resolved_sum, k, inv_count);
            const double sgs = averaged_at_slab(accumulator.heat_flux_face_sgs_sum, k, inv_count);
            out << z << ',' << z / zi_mean << ',' << z / params.z_i << ','
                << heat_flux / params.surface_theta_flux << ','
                << resolved / params.surface_theta_flux << ','
                << sgs / params.surface_theta_flux << '\n';
        }
    }

    std::cout << "[benchmark] wrote distributed profile diagnostics to " << params.benchmark_output_dir << '\n';
}

double enforce_nonnegative_conservative_slab(
    LocalField& qt,
    const Params& params,
    const Slab& slab,
    MPI_Comm comm) {
    double local_negative = 0.0;
    double local_positive = 0.0;
    for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const double value = qt[idx(params, i, j, k)];
                if (!std::isfinite(value)) {
                    local_negative = std::numeric_limits<double>::infinity();
                } else if (value < 0.0) {
                    local_negative -= value;
                } else {
                    local_positive += value;
                }
            }
        }
    }
    double global_negative = 0.0;
    double global_positive = 0.0;
    MPI_Allreduce(&local_negative, &global_negative, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&local_positive, &global_positive, 1, MPI_DOUBLE, MPI_SUM, comm);
    if (!std::isfinite(global_negative) || !std::isfinite(global_positive)) {
        throw std::runtime_error("non-finite total-water mixing ratio encountered in MPI slab mode");
    }
    if (global_negative == 0.0) {
        return 0.0;
    }
    if (global_positive < global_negative) {
        throw std::runtime_error("total-water positivity correction exceeds available water in MPI slab mode");
    }
    const double scale = (global_positive - global_negative) / global_positive;
    for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                double& value = qt[idx(params, i, j, k)];
                value = value > 0.0 ? value * scale : 0.0;
            }
        }
    }
    return global_negative;
}

void check_moisture_stability_slab(
    const MpiLocalFlowState& state,
    const MpiSlabWorkspace& workspace,
    const Params& params,
    const Slab& slab,
    MPI_Comm comm) {
    if (!params.moisture_enabled) {
        return;
    }
    double local_advective_cfl = 0.0;
    double local_max_diffusivity = 0.0;
    int local_max_i = 0;
    int local_max_j = 0;
    int local_max_k = slab.k_begin;
    for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                const double wc = 0.5 * (
                    state.w[z_face_idx(params, i, j, k)]
                    + state.w[z_face_idx(params, i, j, k + 1)]);
                local_advective_cfl = std::max(
                    local_advective_cfl,
                    params.dt * (std::abs(state.u[n]) / params.dx()
                        + std::abs(state.v[n]) / params.dy()
                        + std::abs(wc) / params.dz()));
                const double point_max_diffusivity = std::max(
                    workspace.scalar_kappa_center[n], workspace.moisture_kappa_center[n]);
                if (point_max_diffusivity > local_max_diffusivity) {
                    local_max_diffusivity = point_max_diffusivity;
                    local_max_i = i;
                    local_max_j = j;
                    local_max_k = k;
                }
            }
        }
    }
    double advective_cfl = 0.0;
    double max_diffusivity = 0.0;
    MPI_Allreduce(&local_advective_cfl, &advective_cfl, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&local_max_diffusivity, &max_diffusivity, 1, MPI_DOUBLE, MPI_MAX, comm);
    const double max_laplacian_eigenvalue = std::pow(pi / params.dx(), 2.0)
        + std::pow(pi / params.dy(), 2.0)
        + 4.0 / std::pow(params.dz(), 2.0);
    const double diffusion_number = params.dt * max_diffusivity * max_laplacian_eigenvalue;
    if (advective_cfl > 1.0) {
        throw std::runtime_error("MPI moisture advection CFL exceeds 1: " + std::to_string(advective_cfl));
    }
    const double lasd_update_interval_cfl = static_cast<double>(params.cs_count) * advective_cfl;
    if (params.sgs_model == "lasd" && lasd_update_interval_cfl > 1.0) {
        throw std::runtime_error(
            "MPI LASD update-interval CFL exceeds 1: " + std::to_string(lasd_update_interval_cfl));
    }
    if (diffusion_number > 1.0) {
        struct MaxLoc {
            double value;
            int rank;
        } local_maxloc{local_max_diffusivity, slab.rank}, global_maxloc{};
        MPI_Allreduce(&local_maxloc, &global_maxloc, 1, MPI_DOUBLE_INT, MPI_MAXLOC, comm);
        std::array<double, 26> max_details{};
        if (slab.rank == global_maxloc.rank) {
            const std::size_t n = idx(params, local_max_i, local_max_j, local_max_k);
            const std::size_t lower = z_face_idx(params, local_max_i, local_max_j, local_max_k);
            const std::size_t upper = z_face_idx(params, local_max_i, local_max_j, local_max_k + 1);
            const std::array<double, 9> velocity_gradient{
                workspace.grad.dudx[n], workspace.grad.dudy[n], workspace.grad.dudz[n],
                workspace.grad.dvdx[n], workspace.grad.dvdy[n], workspace.grad.dvdz[n],
                workspace.grad.dwdx[n], workspace.grad.dwdy[n], workspace.grad.dwdz[n],
            };
            const std::array<double, 3> scalar_gradient{
                workspace.scalar_dtheta_dx[n],
                workspace.scalar_dtheta_dy[n],
                workspace.scalar_dtheta_dz_center[n],
            };
            max_details = {
                static_cast<double>(local_max_i),
                static_cast<double>(local_max_j),
                static_cast<double>(local_max_k),
                workspace.strain[n],
                state.scalar_c[n],
                state.qt_scalar_c[n],
                workspace.scalar_kappa_center[n] >= workspace.moisture_kappa_center[n] ? 0.0 : 1.0,
                workspace.grad.dudx[n],
                workspace.grad.dudy[n],
                workspace.grad.dudz[n],
                workspace.grad.dvdx[n],
                workspace.grad.dvdy[n],
                workspace.grad.dvdz[n],
                workspace.grad.dwdx[n],
                workspace.grad.dwdy[n],
                workspace.grad.dwdz[n],
                workspace.scalar_dtheta_dx[n],
                workspace.scalar_dtheta_dy[n],
                workspace.scalar_dtheta_dz_center[n],
                workspace.scalar_dtheta_dz_w[lower],
                workspace.scalar_dtheta_dz_w[upper],
                workspace.nu_t[n],
                workspace.scalar_kappa_center[n],
                workspace.moisture_kappa_center[n],
                amd_scalar_diffusivity_at(
                    velocity_gradient, scalar_gradient, amd_scaled_cell_width(params)),
                amd_scalar_diffusivity_staggered_at(
                    velocity_gradient,
                    scalar_gradient,
                    workspace.scalar_dtheta_dz_w[lower],
                    workspace.scalar_dtheta_dz_w[upper],
                    amd_scaled_cell_width(params)),
            };
        }
        MPI_Bcast(max_details.data(), static_cast<int>(max_details.size()), MPI_DOUBLE, global_maxloc.rank, comm);
        throw std::runtime_error(
            "MPI moisture explicit-diffusion stability number exceeds 1: "
            + std::to_string(diffusion_number)
            + "; max_kappa=" + std::to_string(max_diffusivity)
            + " at (i,j,k)=(" + std::to_string(static_cast<int>(max_details[0]))
            + "," + std::to_string(static_cast<int>(max_details[1]))
            + "," + std::to_string(static_cast<int>(max_details[2])) + ")"
            + ", strain=" + std::to_string(max_details[3])
            + ", theta_l_c=" + std::to_string(max_details[4])
            + ", qt_c=" + std::to_string(max_details[5])
            + ", limiting_scalar=" + (max_details[6] == 0.0 ? "theta_l" : "qt")
            + "; grad_u=[" + std::to_string(max_details[7])
            + "," + std::to_string(max_details[8])
            + "," + std::to_string(max_details[9])
            + ";" + std::to_string(max_details[10])
            + "," + std::to_string(max_details[11])
            + "," + std::to_string(max_details[12])
            + ";" + std::to_string(max_details[13])
            + "," + std::to_string(max_details[14])
            + "," + std::to_string(max_details[15]) + "]"
            + "; grad_scalar_micro=[" + std::to_string(1.0e6 * max_details[16])
            + "," + std::to_string(1.0e6 * max_details[17])
            + "," + std::to_string(1.0e6 * max_details[18])
            + "; dz_faces=" + std::to_string(1.0e6 * max_details[19])
            + "," + std::to_string(1.0e6 * max_details[20]) + "]e-6"
            + "; nu_t=" + std::to_string(max_details[21])
            + ", kappa_theta_l=" + std::to_string(max_details[22])
            + ", kappa_qt=" + std::to_string(max_details[23])
            + ", recomputed_collocated=" + std::to_string(max_details[24])
            + ", recomputed_staggered=" + std::to_string(max_details[25]));
    }
}

double sponge_strength_slab(double z, const Params& params) {
    if (!params.sponge_enabled || z <= params.sponge_start_height) {
        return 0.0;
    }
    const double depth = std::max(params.lz - params.sponge_start_height, params.dz());
    const double eta = std::clamp((z - params.sponge_start_height) / depth, 0.0, 1.0);
    return std::pow(eta, params.sponge_power) / params.sponge_timescale;
}

double sponge_target_u_slab(double z, const Params& params) {
    return params.initial_condition == "bomex" ? bomex_geostrophic_u(z) : params.geostrophic_u;
}

double sponge_target_v_slab(const Params& params) {
    return params.initial_condition == "bomex" ? 0.0 : params.geostrophic_v;
}

double sponge_target_theta_slab(double z, const Params& params) {
    if (params.initial_condition == "bomex") {
        return bomex_initial_theta_l(z);
    }
    return params.theta0 + params.theta_initial_gradient * z;
}

double sponge_target_qt_slab(double z, const Params& params) {
    if (params.initial_condition == "bomex") {
        return bomex_initial_qt(z);
    }
    return std::max(0.0, params.qv0 + params.qv_initial_gradient * z);
}

void relax_to_target_slab(double& value, double target, double strength, double dt) {
    if (strength <= 0.0) {
        return;
    }
    const double factor = std::exp(-strength * dt);
    value = target + (value - target) * factor;
}

void apply_rayleigh_sponge_slab(MpiLocalFlowState& state, const Params& params, const Slab& slab) {
    if (!params.sponge_enabled) {
        return;
    }
    for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
        const double z = (static_cast<double>(k) + 0.5) * params.dz();
        const double strength = sponge_strength_slab(z, params);
        if (strength == 0.0) {
            continue;
        }
        const double target_u = sponge_target_u_slab(z, params);
        const double target_v = sponge_target_v_slab(params);
        const double target_theta = sponge_target_theta_slab(z, params);
        const double target_qt = sponge_target_qt_slab(z, params);
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                relax_to_target_slab(state.u[n], target_u, strength, params.dt);
                relax_to_target_slab(state.v[n], target_v, strength, params.dt);
                // BOMEX prescribes no theta_l/q_t restoring in the sponge.
                // Applying it would violate the conserved-scalar budgets.
                if (params.moisture_enabled && params.initial_condition != "bomex") {
                    relax_to_target_slab(state.theta_l[n], target_theta, strength, params.dt);
                    relax_to_target_slab(state.qt[n], target_qt, strength, params.dt);
                } else if (params.thermo_enabled && params.initial_condition != "bomex") {
                    relax_to_target_slab(state.theta[n], target_theta, strength, params.dt);
                }
            }
        }
    }
    for (int k = slab.face_begin; k < slab.face_begin + slab.face_count; ++k) {
        if (k <= 0 || k >= params.nz) {
            continue;
        }
        const double z = static_cast<double>(k) * params.dz();
        const double strength = sponge_strength_slab(z, params);
        if (strength == 0.0) {
            continue;
        }
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                relax_to_target_slab(state.w[z_face_idx(params, i, j, k)], 0.0, strength, params.dt);
            }
        }
    }
    enforce_walls_slab(state.w, params, slab);
}

void step_mpi_slab(
    MpiLocalFlowState& state,
    const Params& params,
    FftwXY& fft,
    const Slab& slab,
    MPI_Comm comm,
    MpiSlabWorkspace& workspace,
    MpiTimingStats& timing) {
    MpiTimingStats* active_timing =
        (params.mpi_profile_enabled && state.step_count >= params.mpi_profile_warmup_steps) ? &timing : nullptr;
    if (active_timing != nullptr) {
        ++active_timing->measured_steps;
    }
    MpiTimerScope step_scope(active_timing, MpiTimerId::step_total);

    compute_rhs_slab(state, params, slab, fft, comm, workspace, active_timing);
    check_moisture_stability_slab(state, workspace, params, slab, comm);

    {
        MpiTimerScope scope(active_timing, MpiTimerId::advance);
        advance_center_slab(state.u, workspace.rhs_u, state.rhs_u_prev, state.rhs_u_prev2, state.step_count, params.dt, params, slab);
        advance_center_slab(state.v, workspace.rhs_v, state.rhs_v_prev, state.rhs_v_prev2, state.step_count, params.dt, params, slab);
        advance_w_slab(state.w, workspace.rhs_w, state.rhs_w_prev, state.rhs_w_prev2, state.step_count, params.dt, params, slab);
        if (params.moisture_enabled) {
            advance_center_slab(state.theta_l, workspace.rhs_theta, state.rhs_theta_prev, state.rhs_theta_prev2, state.step_count, params.dt, params, slab);
            advance_center_slab(state.qt, workspace.rhs_qt, state.rhs_qt_prev, state.rhs_qt_prev2, state.step_count, params.dt, params, slab);
        } else if (params.thermo_enabled) {
            advance_center_slab(state.theta, workspace.rhs_theta, state.rhs_theta_prev, state.rhs_theta_prev2, state.step_count, params.dt, params, slab);
        } else {
            state.rhs_theta_prev = workspace.rhs_theta;
        }
        if (uses_moeng_tke(params)) {
            advance_center_slab(
                state.sgs_tke,
                workspace.rhs_sgs_tke,
                state.rhs_sgs_tke_prev,
                state.rhs_sgs_tke_prev2,
                state.step_count,
                params.dt,
                params,
                slab);
        }
        state.has_rhs_prev = true;
    }
    apply_rayleigh_sponge_slab(state, params, slab);

    {
        MpiTimerScope scope(active_timing, MpiTimerId::dealias);
        enforce_walls_slab(state.w, params, slab);
        horizontal_dealias_state_slab(state, params, slab, fft);
    }
    if (uses_moeng_tke(params)) {
        for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const std::size_t n = idx(params, i, j, k);
                    state.sgs_tke[n] = std::max(state.sgs_tke[n], params.tke_floor);
                }
            }
        }
    }
    if (params.moisture_enabled) {
        enforce_nonnegative_conservative_slab(state.qt, params, slab, comm);
        update_moist_thermodynamics_slab(state, params, slab);
    }
    {
        MpiTimerScope scope(active_timing, MpiTimerId::state_halo_after);
        exchange_state_halos(state, params, slab, comm, &workspace.halo_exchange_scratch);
    }
    const double kinetic_energy_before_projection = local_kinetic_energy_sum(state, params, slab);
    {
        MpiTimerScope scope(active_timing, MpiTimerId::projection);
        project_mpi_y_pencil(state, params, slab, fft, comm, workspace.pressure, &workspace.halo_exchange_scratch);
    }
    const double projection_energy_change = local_kinetic_energy_sum(state, params, slab)
        - kinetic_energy_before_projection;
    workspace.projection_energy_change_sum += projection_energy_change;
    workspace.projection_max_energy_increase = std::max(
        workspace.projection_max_energy_increase, projection_energy_change);
    ++workspace.projection_energy_samples;
    ++state.step_count;
}

void print_mpi_diagnostics(int step_number, const Diagnostics& diag, const Params& params) {
    std::cout << std::setw(6) << step_number
              << "  " << std::scientific << std::setprecision(6) << diag.ke_max
              << "  " << std::scientific << std::setprecision(6) << diag.div_max
              << "  " << std::fixed << std::setprecision(6) << diag.cfl;
    if (params.moisture_enabled) {
        std::cout << "  " << std::scientific << std::setprecision(6) << diag.qv_min
                  << "  " << diag.qv_max
                  << "  " << diag.ql_max
                  << "  " << diag.column_water;
    }
    std::cout << '\n';
    if (params.sgs_model == "lasd") {
        std::cout << "[lasd] cs2_mean=" << std::scientific << std::setprecision(4) << diag.lasd_cs2_mean
                  << " cs2_max=" << diag.lasd_cs2_max
                  << " beta_mean=" << diag.lasd_beta_mean
                  << " beta_floor=" << diag.lasd_beta_floor_fraction;
        if (params.scalar_sgs_model == "lasd") {
            std::cout << " theta_c_mean=" << diag.lasd_theta_c_mean
                      << " theta_c_max=" << diag.lasd_theta_c_max
                      << " theta_beta_floor=" << diag.lasd_theta_beta_floor_fraction
                      << " qt_c_mean=" << diag.lasd_qt_c_mean
                      << " qt_c_max=" << diag.lasd_qt_c_max
                      << " qt_beta_floor=" << diag.lasd_qt_beta_floor_fraction;
        }
        std::cout << '\n';
    }
}

void print_mpi_timing_report(const MpiTimingStats& timing, const Params& params, MPI_Comm comm) {
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    std::array<double, mpi_timer_count> sum_seconds{};
    std::array<double, mpi_timer_count> max_seconds{};
    MPI_Reduce(
        timing.seconds.data(),
        sum_seconds.data(),
        static_cast<int>(mpi_timer_count),
        MPI_DOUBLE,
        MPI_SUM,
        0,
        comm);
    MPI_Reduce(
        timing.seconds.data(),
        max_seconds.data(),
        static_cast<int>(mpi_timer_count),
        MPI_DOUBLE,
        MPI_MAX,
        0,
        comm);

    const int local_counts[2] = {timing.measured_steps, timing.diagnostic_calls};
    int max_counts[2] = {0, 0};
    MPI_Reduce(local_counts, max_counts, 2, MPI_INT, MPI_MAX, 0, comm);

    if (rank != 0) {
        return;
    }

    const int measured_steps = max_counts[0];
    const int diagnostic_calls = max_counts[1];
    std::cout << "[mpi-profile] measured_steps=" << measured_steps
              << " warmup_excluded=" << params.mpi_profile_warmup_steps
              << " diagnostic_calls=" << diagnostic_calls << '\n';
    if (measured_steps <= 0) {
        std::cout << "[mpi-profile] no timed steps; reduce profiling warmup or increase steps\n";
        return;
    }

    struct TimingRow {
        MpiTimerId id;
        const char* group;
        const char* module;
        bool diagnostic_average;
    };

    const std::array<TimingRow, 27> rows{{
        {MpiTimerId::rhs_derivatives, "rhs", "derivatives_and_laplacians", false},
        {MpiTimerId::rhs_momentum, "rhs", "base_momentum_rhs", false},
        {MpiTimerId::rhs_sgs_update, "rhs", "sgs_update_and_nut_halo", false},
        {MpiTimerId::rhs_sgs_forcing, "rhs", "sgs_momentum_forcing", false},
        {MpiTimerId::sgs_center_stress_build, "sgs", "center_stress_build", false},
        {MpiTimerId::sgs_face_derivatives, "sgs", "face_derivatives_and_nut", false},
        {MpiTimerId::sgs_face_stress_build, "sgs", "face_stress_build", false},
        {MpiTimerId::sgs_stress_halo, "sgs", "stress_halo_exchange", false},
        {MpiTimerId::sgs_center_divergence, "sgs", "center_stress_divergence", false},
        {MpiTimerId::sgs_face_divergence, "sgs", "face_stress_divergence", false},
        {MpiTimerId::rhs_wall_stress, "rhs", "wall_stress", false},
        {MpiTimerId::rhs_scalar, "rhs", "scalar_rhs", false},
        {MpiTimerId::scalar_advective_flux, "scalar", "advective_flux_build", false},
        {MpiTimerId::scalar_advective_divergence, "scalar", "advective_divergence", false},
        {MpiTimerId::scalar_gradients, "scalar", "gradients", false},
        {MpiTimerId::scalar_lasd_update, "scalar", "lasd_update", false},
        {MpiTimerId::scalar_diffusivity_halo, "scalar", "diffusivity_halo_and_interp", false},
        {MpiTimerId::scalar_diffusive_flux, "scalar", "diffusive_flux_build", false},
        {MpiTimerId::scalar_diffusive_divergence, "scalar", "diffusive_divergence", false},
        {MpiTimerId::rhs_buoyancy, "rhs", "buoyancy", false},
        {MpiTimerId::rhs_total, "rhs", "rhs_total", false},
        {MpiTimerId::advance, "time", "ab_update", false},
        {MpiTimerId::dealias, "filter", "dealias_and_walls", false},
        {MpiTimerId::state_halo_after, "comm", "state_halo_after", false},
        {MpiTimerId::projection, "projection", "pressure_projection", false},
        {MpiTimerId::step_total, "total", "step_total", false},
        {MpiTimerId::diagnostics, "diag", "diagnostics_per_call", true},
    }};

    const double step_max_ms =
        max_seconds[static_cast<std::size_t>(MpiTimerId::step_total)] / static_cast<double>(measured_steps) * 1000.0;
    std::cout << std::left << std::setw(12) << "group"
              << std::setw(30) << "module"
              << std::right << std::setw(16) << "rank_mean_ms"
              << std::setw(16) << "rank_max_ms"
              << std::setw(14) << "pct_step_max" << '\n';
    std::cout << std::fixed << std::setprecision(3);
    for (const TimingRow& row : rows) {
        const int denom = row.diagnostic_average ? diagnostic_calls : measured_steps;
        if (denom <= 0) {
            continue;
        }
        const std::size_t id = static_cast<std::size_t>(row.id);
        const double rank_mean_ms = sum_seconds[id] / (static_cast<double>(denom) * static_cast<double>(size)) * 1000.0;
        const double rank_max_ms = max_seconds[id] / static_cast<double>(denom) * 1000.0;
        std::cout << std::left << std::setw(12) << row.group
                  << std::setw(30) << row.module
                  << std::right << std::setw(16) << rank_mean_ms
                  << std::setw(16) << rank_max_ms;
        if (row.diagnostic_average || step_max_ms <= 0.0) {
            std::cout << std::setw(14) << "";
        } else {
            std::cout << std::setw(14) << (100.0 * rank_max_ms / step_max_ms);
        }
        std::cout << '\n';
    }
}

}  // namespace

int run_mpi_slab(const Params& params, int argc, char** argv) {
    MpiRuntime mpi(argc, argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    try {
        const Slab slab = make_slab(params, comm);
        require_distributed_rhs_supported(params);
        const int fft_max_planes = std::max({slab.k_count, slab.face_count, needed_face_count(params, slab)});
        FftwXY fft(params, fft_max_planes);
        MpiSlabWorkspace workspace(params, slab);
        MpiTimingStats timing;
        MpiBenchmarkAccumulator benchmark;
        BomexAccumulator bomex;
        BomexAccumulator bomex_last_hour;
        MpiLocalFlowState state(params, slab);
        initialize_local(state, params, slab);
        exchange_state_halos(state, params, slab, comm, &workspace.halo_exchange_scratch);
        project_mpi_y_pencil(state, params, slab, fft, comm, workspace.pressure, &workspace.halo_exchange_scratch);
        const Diagnostics initial_diag = diagnostics_mpi_slab(state, params, slab, fft, comm);
        int average_start_step = 0;
        int average_end_step = params.steps;
        if (params.benchmark_enabled) {
            const double wstar0 = convective_wstar_slab(params, params.z_i);
            const double tstar0 = params.z_i / wstar0;
            average_start_step =
                static_cast<int>(std::ceil(params.benchmark_average_start_tstar * tstar0 / params.dt));
            average_end_step = std::min(
                params.steps,
                static_cast<int>(std::floor(params.benchmark_average_end_tstar * tstar0 / params.dt)));
        }

        if (rank == 0) {
            std::cout << "# wireles non-blocking MPI z-slab solver\n";
            std::cout << "# ranks " << size << ", z-slab physical layout, y-pencil pressure all-to-all\n";
            std::cout << "# grid " << params.nx << "x" << params.ny << "x" << params.nz
                      << ", dt=" << params.dt << ", scheme=" << params.time_scheme
                      << ", nu=" << params.nu << '\n';
            std::cout << "# wall=" << params.momentum_wall_model
                      << ", sgs=" << params.sgs_model
                      << " (delta_scale=" << params.sgs_delta_scale << ")"
                      << ", thermo=" << (params.thermo_enabled ? "on" : "off")
                      << ", moisture=" << (params.moisture_enabled ? "on" : "off") << '\n';
            if (params.initial_condition == "bomex") {
                std::cout << "# BOMEX coriolis_f=" << params.coriolis_f
                          << ", perturbation_height="
                          << (params.initial_perturbation_height > 0.0
                                  ? params.initial_perturbation_height
                                  : 4.0 * params.dz())
                          << " m\n";
            }
            std::cout << "#  step        ke_max         div_max       cfl";
            if (params.moisture_enabled) {
                std::cout << "        qv_min         qv_max         ql_max    column_water";
            }
            std::cout << '\n';
            print_mpi_diagnostics(0, initial_diag, params);
        }
        if (should_write_frame_slab(params, 0)) {
            write_mpi_xz_slice_csv(state, params, slab, comm, 0);
            write_mpi_xy_slice_csv(state, params, slab, comm, 0);
        }

        for (int step_number = 1; step_number <= params.steps; ++step_number) {
            step_mpi_slab(state, params, fft, slab, comm, workspace, timing);
            if (should_write_frame_slab(params, step_number)) {
                write_mpi_xz_slice_csv(state, params, slab, comm, step_number);
                write_mpi_xy_slice_csv(state, params, slab, comm, step_number);
            }
            const bool needs_benchmark = params.benchmark_enabled && !params.moisture_enabled
                && (step_number % params.benchmark_sample_every == 0 || step_number == params.steps)
                && step_number >= average_start_step
                && step_number <= average_end_step;
            if (needs_benchmark) {
                MpiBenchmarkSnapshot snapshot;
                collect_mpi_benchmark_sample(state, params, slab, fft, comm, workspace, snapshot);
                if (rank == 0) {
                    add_mpi_benchmark_sample(benchmark, snapshot);
                }
            }
            const bool needs_bomex = params.bomex_diagnostics_enabled
                && params.initial_condition == "bomex"
                && (step_number % params.bomex_sample_every == 0 || step_number == params.steps)
                && static_cast<double>(step_number) * params.dt >= params.bomex_average_start_seconds;
            if (needs_bomex) {
                add_mpi_bomex_sample(bomex, state, params, slab, comm, workspace);
                const double time_s = static_cast<double>(step_number) * params.dt;
                const double last_hour_start = std::max(
                    params.bomex_average_start_seconds,
                    static_cast<double>(params.steps) * params.dt - 3600.0);
                if (time_s >= last_hour_start) {
                    add_mpi_bomex_sample(bomex_last_hour, state, params, slab, comm, workspace);
                }
            }
            if (step_number % params.log_every == 0 || step_number == params.steps) {
                MpiTimingStats* diagnostic_timing =
                    (params.mpi_profile_enabled && step_number > params.mpi_profile_warmup_steps) ? &timing : nullptr;
                if (diagnostic_timing != nullptr) {
                    ++diagnostic_timing->diagnostic_calls;
                }
                Diagnostics diag;
                {
                    MpiTimerScope scope(diagnostic_timing, MpiTimerId::diagnostics);
                    diag = diagnostics_mpi_slab(state, params, slab, fft, comm);
                }
                double global_projection_energy_change = 0.0;
                double global_momentum_advection_power = 0.0;
                double global_momentum_advection_u = 0.0;
                double global_momentum_advection_v = 0.0;
                MPI_Reduce(
                    &workspace.projection_energy_change_sum,
                    &global_projection_energy_change,
                    1,
                    MPI_DOUBLE,
                    MPI_SUM,
                    0,
                    comm);
                MPI_Reduce(
                    &workspace.momentum_advection_power_sum,
                    &global_momentum_advection_power,
                    1,
                    MPI_DOUBLE,
                    MPI_SUM,
                    0,
                    comm);
                MPI_Reduce(
                    &workspace.momentum_advection_u_sum,
                    &global_momentum_advection_u,
                    1,
                    MPI_DOUBLE,
                    MPI_SUM,
                    0,
                    comm);
                MPI_Reduce(
                    &workspace.momentum_advection_v_sum,
                    &global_momentum_advection_v,
                    1,
                    MPI_DOUBLE,
                    MPI_SUM,
                    0,
                    comm);
                if (rank == 0) {
                    print_mpi_diagnostics(step_number, diag, params);
                    const double normalization = static_cast<double>(params.nx * params.ny * params.nz)
                        * static_cast<double>(std::max<std::size_t>(workspace.projection_energy_samples, 1));
                    std::cout << "[projection-energy] mean_delta_ke_per_cell_step="
                              << std::scientific << std::setprecision(6)
                              << global_projection_energy_change / normalization << '\n';
                    const double advection_normalization = static_cast<double>(params.nx * params.ny * params.nz)
                        * static_cast<double>(std::max<std::size_t>(workspace.momentum_advection_power_samples, 1));
                    std::cout << "[advection-energy] mean_power_per_cell="
                              << global_momentum_advection_power / advection_normalization << '\n';
                    std::cout << "[advection-momentum] mean_du_dt_per_cell="
                              << global_momentum_advection_u / advection_normalization
                              << " mean_dv_dt_per_cell="
                              << global_momentum_advection_v / advection_normalization << '\n';
                    if (!std::isfinite(diag.ke_max) || !std::isfinite(diag.div_max) || !std::isfinite(diag.cfl)) {
                        throw std::runtime_error("non-finite diagnostics encountered; stopping run");
                    }
                }
                workspace.projection_energy_change_sum = 0.0;
                workspace.projection_max_energy_increase = 0.0;
                workspace.projection_energy_samples = 0;
                workspace.momentum_advection_power_sum = 0.0;
                workspace.momentum_advection_u_sum = 0.0;
                workspace.momentum_advection_v_sum = 0.0;
                workspace.momentum_advection_power_samples = 0;
            }
        }
        write_mpi_bomex_final_masks(state, params, slab, comm);
        if (rank == 0) {
            if (!params.moisture_enabled) {
                print_mpi_benchmark_summary(benchmark, params);
                write_mpi_benchmark_outputs(benchmark, params);
            }
            print_bomex_summary(bomex, params);
            FlowState output_state(params);
            output_state.base_pressure = state.base_pressure;
            write_bomex_outputs(bomex, output_state, params);
            if (bomex_last_hour.samples > 0) {
                Params last_hour_params = params;
                last_hour_params.bomex_output_dir =
                    (std::filesystem::path(params.bomex_output_dir) / "fig3_last_hour").string();
                write_bomex_outputs(bomex_last_hour, output_state, last_hour_params);
            }
        }
        if (params.mpi_profile_enabled) {
            print_mpi_timing_report(timing, params, comm);
        }
    } catch (const std::exception& exc) {
        std::cerr << "[rank " << rank << "] ERROR: " << exc.what() << '\n';
        MPI_Abort(comm, 1);
        return 1;
    }

    return 0;
}

}  // namespace wireles

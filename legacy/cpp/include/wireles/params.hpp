#pragma once

#include <cmath>
#include <cstddef>
#include <string>

namespace wireles {

constexpr double pi = 3.141592653589793238462643383279502884;

struct Params {
    int nx = 32;
    int ny = 32;
    int nz = 32;
    int steps = 20;
    int log_every = 5;
    double lx = 2.0 * pi;
    double ly = 2.0 * pi;
    double lz = 2.0 * pi;
    double z_i = 1.0;
    double dt = 1.0e-3;
    double nu = 1.0e-3;
    double scalar_diffusivity = 0.0;
    std::string time_scheme = "euler";
    std::string initial_condition = "taylor_green";
    std::string momentum_wall_model = "none";
    std::string wall_stress_model = "dynamic_neutral";
    std::string sgs_model = "none";
    double coriolis_f = 0.0;
    double geostrophic_u = 0.0;
    double geostrophic_v = 0.0;
    double u_fric = 0.4;
    double zo = 5.0e-3;
    double vonk = 0.4;
    double fgr = 1.5;
    double tfr = 2.0;
    int cs_count = 10;
    double smagorinsky_cs = 0.16;
    bool smagorinsky_buoyancy_correction = false;
    double smagorinsky_min_shear_fraction = 0.0;
    double sgs_delta_scale = 1.0;
    // Buoyancy-adjusted AMD momentum closure (Abkar & Moin).  Disabling it
    // recovers the plain Rozema/Verstappen minimum-dissipation numerator.
    bool amd_buoyancy_correction = true;
    // Evaluate the AMD numerator/denominator as local (1-2-1)^3 volume
    // averages before the clipped ratio.  The minimum-dissipation bound is
    // derived over a filter volume, so averaging the invariants is closer to
    // the derivation than the pointwise ratio and removes grid-scale sign
    // intermittency without destroying horizontal heterogeneity.
    bool amd_invariant_averaging = false;
    // Locally redistribute the already clipped AMD dissipation over a compact
    // filter box: nu = average(nu |S|^2) / average(|S|^2).  Unlike invariant
    // averaging this does not cancel positive dissipation demand against a
    // negative numerator, and unlike plane redistribution it remains local
    // and applicable to heterogeneous flows.
    bool amd_dissipation_averaging = false;
    // Enforce both the pointwise AMD coefficient and the coefficient obtained
    // from compactly averaged positive numerator/denominator invariants.  The
    // maximum satisfies the stricter of the cell and filter-box bounds
    // without a fitted blend coefficient.
    bool amd_multiscale_averaging = false;
    // At the first velocity level, replace the finite-difference mean vertical
    // velocity gradient by the wall-similarity value while retaining finite-
    // difference gradients of the resolved fluctuations (Gadde et al. 2021,
    // Eqs. 19--20).
    bool amd_wall_model_gradients = false;
    // Use the effective horizontal filter width of the sharp 2/3 dealiasing
    // (1.5 dx, matching two_thirds_filter_width) as the AMD horizontal
    // Poincare length instead of the raw grid spacing.  The minimum-
    // dissipation derivation is stated for the resolved cutoff of the
    // discrete operator; with 2/3 truncation active on every state field the
    // resolved cutoff is pi/(1.5 dx), so dx underestimates the horizontal
    // numerator terms by (1.5)^2 = 2.25.  Applies to momentum and scalar AMD
    // through the shared scaled cell width.
    bool amd_dealiased_cell_width = false;
    bool scalar_amd_invariant_averaging = false;
    // Form every vertical/w quadratic product of the scalar AMD numerator on
    // its natural w-face location before center interpolation (the same
    // staggering principle already used by the momentum AMD).  The default
    // false keeps the partially staggered numerator of the archived baseline.
    bool scalar_amd_face_products = false;
    // Moeng (1984) / Deardorff (1980) prognostic SGS-TKE closure constants.
    double tke_ck = 0.10;
    double tke_length_coefficient = 0.76;
    double tke_dissipation_base = 0.19;
    double tke_dissipation_slope = 0.74;
    double tke_floor = 1.0e-10;
    std::string scalar_sgs_model = "fixed_prandtl";
    double prandtl_t = 0.74;
    double schmidt_t = 0.74;
    double scalar_lasd_min = 0.0;
    double scalar_lasd_max = 1.0;
    bool scalar_stability_correction = false;
    double scalar_stability_beta = 10.0;
    double scalar_stability_power = 2.0;
    bool thermo_enabled = false;
    double theta0 = 300.0;
    double theta_initial_gradient = 0.0;
    double largeeddy_initial_zi1_fraction = 0.844;
    double surface_theta_flux = 0.0;
    bool moisture_enabled = false;
    double qv0 = 0.0;
    double qv_initial_gradient = 0.0;
    double surface_qv_flux = 0.0;
    double moisture_diffusivity = 0.0;
    double surface_pressure = 100000.0;
    double g = 9.81;
    bool sponge_enabled = false;
    double sponge_start_height = 0.0;
    double sponge_timescale = 0.0;
    double sponge_power = 2.0;
    int random_seed = 0;
    double initial_velocity_perturbation = 0.0;
    double initial_perturbation_height = 0.0;
    double bomex_theta_perturbation = 0.1;
    double bomex_qt_perturbation = 2.5e-5;
    bool horizontal_dealias = false;
    std::string dealiasing = "sharp";
    std::string momentum_advection_form = "advective";
    std::string spectral_filter = "sharp";
    double spectral_filter_alpha = 36.0;
    int spectral_filter_order = 16;
    bool benchmark_enabled = false;
    int benchmark_sample_every = 40;
    double benchmark_average_start_tstar = 10.0;
    double benchmark_average_end_tstar = 11.0;
    std::string benchmark_output_dir;
    bool bomex_diagnostics_enabled = false;
    int bomex_sample_every = 60;
    double bomex_average_start_seconds = 10800.0;
    std::string bomex_output_dir;
    bool frame_dump_enabled = false;
    int frame_dump_every = 0;
    int frame_dump_start_step = 0;
    int frame_dump_end_step = -1;
    int frame_dump_y_index = -1;
    double frame_dump_z_height = -1.0;
    bool frame_dump_slice_only = false;
    std::string frame_dump_output_dir;
    std::string frame_dump_component = "w";
    bool mpi_slab = false;
    bool mpi_profile_enabled = false;
    int mpi_profile_warmup_steps = 0;
    bool cuda_enabled = false;

    double dx() const { return lx / static_cast<double>(nx); }
    double dy() const { return ly / static_cast<double>(ny); }
    double dz() const { return nz > 1 ? lz / static_cast<double>(nz - 1) : lz; }
    double wall_ref_height() const { return 0.5 * dz(); }
    double sgs_delta() const { return sgs_delta_scale * std::cbrt(dx() * dy() * dz()); }
    int nkx() const { return nx / 2 + 1; }
    std::size_t real_size() const {
        return static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
    }
    std::size_t z_face_size() const {
        return static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz + 1);
    }
    std::size_t spectral_size() const {
        return static_cast<std::size_t>(nkx()) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
    }
};

inline std::size_t idx(const Params& p, int i, int j, int k) {
    return (static_cast<std::size_t>(k) * static_cast<std::size_t>(p.ny) + static_cast<std::size_t>(j))
        * static_cast<std::size_t>(p.nx)
        + static_cast<std::size_t>(i);
}

inline std::size_t sidx(const Params& p, int ih, int j, int k) {
    return (static_cast<std::size_t>(k) * static_cast<std::size_t>(p.ny) + static_cast<std::size_t>(j))
        * static_cast<std::size_t>(p.nkx())
        + static_cast<std::size_t>(ih);
}

inline double kx_value(const Params& p, int ih) {
    return 2.0 * pi * static_cast<double>(ih) / p.lx;
}

inline double ky_value(const Params& p, int j) {
    const int signed_j = (j <= p.ny / 2) ? j : j - p.ny;
    return 2.0 * pi * static_cast<double>(signed_j) / p.ly;
}

inline double kx_derivative_value(const Params& p, int ih) {
    if (p.nx % 2 == 0 && ih == p.nx / 2) {
        return 0.0;
    }
    return kx_value(p, ih);
}

inline double ky_derivative_value(const Params& p, int j) {
    if (p.ny % 2 == 0 && j == p.ny / 2) {
        return 0.0;
    }
    return ky_value(p, j);
}

}  // namespace wireles

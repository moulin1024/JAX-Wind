#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cerrno>
#include <cctype>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

#include "wireles/config.hpp"
#include "wireles/cuda_mpi_slab.hpp"
#ifdef WIRELES_HAVE_CPU
#include "wireles/bomex.hpp"
#include "wireles/cuda_solver.hpp"
#include "wireles/fft.hpp"
#include "wireles/field.hpp"
#endif
#ifdef WIRELES_HAVE_MPI
#ifdef WIRELES_HAVE_CPU
#include "wireles/mpi_slab.hpp"
#endif
#endif
#ifdef WIRELES_HAVE_CPU
#include "wireles/operators.hpp"
#include "wireles/pressure.hpp"
#include "wireles/scalar.hpp"
#include "wireles/sgs.hpp"
#include "wireles/timestep.hpp"
#include "wireles/thermodynamics.hpp"
#endif

namespace wireles {
namespace {

int parse_int(const std::string& value, const std::string& name) {
    try {
        return std::stoi(value);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid integer for " + name + ": " + value);
    }
}

double parse_double(const std::string& value, const std::string& name) {
    try {
        return std::stod(value);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid floating-point value for " + name + ": " + value);
    }
}

void print_help(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n\n"
        << "Options:\n"
        << "  --config FILE              Load case config before CLI overrides\n"
        << "  --nx N --ny N --nz N        Grid size, default 32^3\n"
        << "  --lx L --ly L --lz L        Domain lengths, default 2*pi\n"
        << "  --z-i ZI                   Reference inversion height for CBL init\n"
        << "  --time-scheme euler|ab2    Time integration scheme\n"
        << "  --steps N                  Number of Euler/projection steps\n"
        << "  --log-every N              Diagnostics interval\n"
        << "  --dt DT                    Time step\n"
        << "  --nu NU                    Molecular viscosity\n"
        << "  --sgs none|smagorinsky|lasd|amd|amd_plane_dissipation SGS model\n"
        << "  --cs-count N               LASD coefficient update interval\n"
        << "  --smag-cs CS               Classic Smagorinsky coefficient\n"
        << "  --wall none|abl            Momentum wall model\n"
        << "  --wall-stress MODEL        dynamic_neutral or prescribed_ustar\n"
        << "  --coriolis-f F             Coriolis parameter, disabled when zero\n"
        << "  --geostrophic-u U          Geostrophic x velocity\n"
        << "  --geostrophic-v V          Geostrophic y velocity\n"
        << "  --u-fric USTAR             Prescribed friction velocity\n"
        << "  --zo Z0                    Roughness length\n"
        << "  --vonk KAPPA               von Karman constant\n"
        << "  --fgr F --tfr T            Grid/test filter ratios for wall filtering\n"
        << "  --thermo                   Enable potential-temperature transport\n"
        << "  --theta0 THETA             Reference potential temperature\n"
        << "  --theta-gradient G         Initial theta gradient\n"
        << "  --surface-theta-flux Q     Positive upward kinematic heat flux\n"
        << "  --moisture                 Enable water-vapor transport and virtual-temperature buoyancy\n"
        << "  --qv0 Q                    Reference water-vapor mixing ratio [kg/kg]\n"
        << "  --qt0 Q                    Initial total-water mixing ratio [kg/kg]\n"
        << "  --qv-gradient G            Initial vertical qv gradient [(kg/kg)/m]\n"
        << "  --surface-qv-flux Q        Positive upward kinematic qv flux [(kg/kg)m/s]\n"
        << "  --surface-qt-flux Q        Positive upward kinematic total-water flux\n"
        << "  --moisture-diffusivity K   Molecular water-vapor diffusivity\n"
        << "  --schmidt-t Sc             SGS turbulent Schmidt number\n"
        << "  --surface-pressure P       Hydrostatic base-state surface pressure [Pa]\n"
        << "  --scalar-diffusivity K     Molecular scalar diffusivity\n"
        << "  --scalar-sgs fixed_prandtl|lasd\n"
        << "  --prandtl-t Pr             SGS turbulent Prandtl number\n"
        << "  --scalar-lasd-min C        Minimum dynamic scalar coefficient\n"
        << "  --scalar-lasd-max C        Maximum dynamic scalar coefficient\n"
        << "  --scalar-stability-correction / --no-scalar-stability-correction\n"
        << "  --scalar-stability-beta B  Stable-layer scalar diffusivity damping strength\n"
        << "  --scalar-stability-power P Stable-layer scalar diffusivity damping exponent\n"
        << "  --smag-min-shear-fraction F  Minimum fraction of shear-only Smagorinsky viscosity\n"
        << "  --sgs-delta-scale F       Multiplier for the SGS filter length\n"
        << "  --dealiasing sharp|padding_3_2\n"
        << "  --spectral-filter sharp|floor_sharp|exponential\n"
        << "  --spectral-filter-alpha A  Exponential spectral filter strength\n"
        << "  --spectral-filter-order N  Exponential spectral filter order\n"
        << "  --sponge-start-height Z    Rayleigh sponge start height [m]\n"
        << "  --sponge-timescale T       Rayleigh sponge top-layer timescale [s]\n"
        << "  --sponge-power P           Rayleigh sponge vertical ramp power\n"
        << "  --g G                      Gravitational acceleration for buoyancy\n"
        << "  --largeeddy-zi1-fraction F Initial zi1/z_i for largeeddy1993\n"
        << "  --benchmark-sample-every N Benchmark profile sampling interval\n"
        << "  --benchmark-average-start-tstar T  Benchmark averaging window start in t*\n"
        << "  --benchmark-average-end-tstar T    Benchmark averaging window end in t*\n"
        << "  --benchmark-output-dir DIR Write benchmark profile CSV diagnostics\n"
        << "  --bomex-sample-every N    BOMEX profile sampling interval\n"
        << "  --bomex-average-start-seconds T  BOMEX averaging-window start [s]\n"
        << "  --bomex-output-dir DIR    Write BOMEX comparison CSV diagnostics\n"
        << "  --bomex-theta-perturbation A  BOMEX initial theta_l perturbation amplitude [K]\n"
        << "  --bomex-qt-perturbation A  BOMEX initial q_t perturbation amplitude [kg/kg]\n"
        << "  --initial-velocity-perturbation A  Initial velocity perturbation amplitude\n"
        << "  --initial-perturbation-height H    Height over which perturbations decay\n"
        << "  --dump-frames              Write transient field dumps\n"
        << "  --no-dump-frames           Disable transient field dumps from a config\n"
        << "  --frame-every N            Frame dump interval in steps\n"
        << "  --frame-start-step N       First step eligible for frame dump\n"
        << "  --frame-end-step N         Last step eligible for frame dump\n"
        << "  --frame-output-dir DIR     Directory for transient field dumps\n"
        << "  --frame-component NAME     u, v, w, p, theta, or theta_prime\n"
        << "  --frame-y-index J          y index for x-z cross-section, default ny/2\n"
        << "  --frame-z-height Z         Also write x-y cross-sections near height Z [m]\n"
        << "  --frame-slice-only         Write only one y-slice per transient field dump\n"
        << "  --random-seed N            Deterministic random seed for initial perturbations\n"
        << "  --initial NAME             taylor_green, largeeddy1993, or neutral_ekman\n"
        << "  --cuda, --gpu              Enable CUDA kernels; combine with --mpi-slab for CUDA-MPI pressure path\n"
        << "  --mpi-slab                 Run the non-blocking MPI z-slab path\n"
        << "  --mpi-profile              Print internal MPI slab timing report\n"
        << "  --mpi-profile-warmup N     Exclude first N steps from MPI slab timing\n"
        << "  --help                     Show this message\n";
}

Params parse_args(int argc, char** argv) {
    Params params;
    for (int arg = 1; arg < argc; ++arg) {
        const std::string key = argv[arg];
        if (key == "--config") {
            if (arg + 1 >= argc) {
                throw std::runtime_error("missing value after --config");
            }
            ++arg;
            apply_config_file(params, argv[arg]);
        }
    }

    for (int arg = 1; arg < argc; ++arg) {
        const std::string key = argv[arg];
        auto require_value = [&](const std::string& option) -> std::string {
            if (arg + 1 >= argc) {
                throw std::runtime_error("missing value after " + option);
            }
            ++arg;
            return argv[arg];
        };

        if (key == "--help" || key == "-h") {
            print_help(argv[0]);
            std::exit(0);
        } else if (key == "--config") {
            require_value(key);
        } else if (key == "--nx") {
            params.nx = parse_int(require_value(key), key);
        } else if (key == "--ny") {
            params.ny = parse_int(require_value(key), key);
        } else if (key == "--nz") {
            params.nz = parse_int(require_value(key), key);
        } else if (key == "--lx") {
            params.lx = parse_double(require_value(key), key);
        } else if (key == "--ly") {
            params.ly = parse_double(require_value(key), key);
        } else if (key == "--lz") {
            params.lz = parse_double(require_value(key), key);
        } else if (key == "--z-i") {
            params.z_i = parse_double(require_value(key), key);
        } else if (key == "--time-scheme") {
            params.time_scheme = require_value(key);
        } else if (key == "--steps") {
            params.steps = parse_int(require_value(key), key);
        } else if (key == "--log-every") {
            params.log_every = parse_int(require_value(key), key);
        } else if (key == "--dt") {
            params.dt = parse_double(require_value(key), key);
        } else if (key == "--nu") {
            params.nu = parse_double(require_value(key), key);
        } else if (key == "--sgs") {
            params.sgs_model = require_value(key);
        } else if (key == "--cs-count") {
            params.cs_count = parse_int(require_value(key), key);
        } else if (key == "--smag-cs") {
            params.smagorinsky_cs = parse_double(require_value(key), key);
        } else if (key == "--wall") {
            params.momentum_wall_model = require_value(key);
        } else if (key == "--wall-stress") {
            params.wall_stress_model = require_value(key);
        } else if (key == "--coriolis-f") {
            params.coriolis_f = parse_double(require_value(key), key);
        } else if (key == "--geostrophic-u") {
            params.geostrophic_u = parse_double(require_value(key), key);
        } else if (key == "--geostrophic-v") {
            params.geostrophic_v = parse_double(require_value(key), key);
        } else if (key == "--u-fric") {
            params.u_fric = parse_double(require_value(key), key);
        } else if (key == "--zo") {
            params.zo = parse_double(require_value(key), key);
        } else if (key == "--vonk") {
            params.vonk = parse_double(require_value(key), key);
        } else if (key == "--fgr") {
            params.fgr = parse_double(require_value(key), key);
        } else if (key == "--tfr") {
            params.tfr = parse_double(require_value(key), key);
        } else if (key == "--thermo") {
            params.thermo_enabled = true;
        } else if (key == "--theta0") {
            params.theta0 = parse_double(require_value(key), key);
        } else if (key == "--theta-gradient") {
            params.theta_initial_gradient = parse_double(require_value(key), key);
        } else if (key == "--surface-theta-flux") {
            params.surface_theta_flux = parse_double(require_value(key), key);
        } else if (key == "--moisture") {
            params.moisture_enabled = true;
        } else if (key == "--qv0" || key == "--qt0") {
            params.qv0 = parse_double(require_value(key), key);
        } else if (key == "--qv-gradient" || key == "--qt-gradient") {
            params.qv_initial_gradient = parse_double(require_value(key), key);
        } else if (key == "--surface-qv-flux" || key == "--surface-qt-flux") {
            params.surface_qv_flux = parse_double(require_value(key), key);
        } else if (key == "--moisture-diffusivity") {
            params.moisture_diffusivity = parse_double(require_value(key), key);
        } else if (key == "--largeeddy-zi1-fraction") {
            params.largeeddy_initial_zi1_fraction = parse_double(require_value(key), key);
        } else if (key == "--benchmark-sample-every") {
            params.benchmark_sample_every = parse_int(require_value(key), key);
        } else if (key == "--benchmark-average-start-tstar") {
            params.benchmark_average_start_tstar = parse_double(require_value(key), key);
        } else if (key == "--benchmark-average-end-tstar") {
            params.benchmark_average_end_tstar = parse_double(require_value(key), key);
        } else if (key == "--benchmark-output-dir") {
            params.benchmark_output_dir = require_value(key);
        } else if (key == "--bomex-sample-every") {
            params.bomex_sample_every = parse_int(require_value(key), key);
        } else if (key == "--bomex-average-start-seconds") {
            params.bomex_average_start_seconds = parse_double(require_value(key), key);
        } else if (key == "--bomex-output-dir") {
            params.bomex_output_dir = require_value(key);
        } else if (key == "--bomex-theta-perturbation") {
            params.bomex_theta_perturbation = parse_double(require_value(key), key);
        } else if (key == "--bomex-qt-perturbation") {
            params.bomex_qt_perturbation = parse_double(require_value(key), key);
        } else if (key == "--initial-velocity-perturbation") {
            params.initial_velocity_perturbation = parse_double(require_value(key), key);
        } else if (key == "--initial-perturbation-height") {
            params.initial_perturbation_height = parse_double(require_value(key), key);
        } else if (key == "--dump-frames") {
            params.frame_dump_enabled = true;
        } else if (key == "--no-dump-frames") {
            params.frame_dump_enabled = false;
        } else if (key == "--frame-every") {
            params.frame_dump_every = parse_int(require_value(key), key);
        } else if (key == "--frame-start-step") {
            params.frame_dump_start_step = parse_int(require_value(key), key);
        } else if (key == "--frame-end-step") {
            params.frame_dump_end_step = parse_int(require_value(key), key);
        } else if (key == "--frame-output-dir") {
            params.frame_dump_output_dir = require_value(key);
        } else if (key == "--frame-component") {
            params.frame_dump_component = require_value(key);
        } else if (key == "--frame-y-index") {
            params.frame_dump_y_index = parse_int(require_value(key), key);
        } else if (key == "--frame-z-height") {
            params.frame_dump_z_height = parse_double(require_value(key), key);
        } else if (key == "--frame-slice-only") {
            params.frame_dump_slice_only = true;
        } else if (key == "--scalar-diffusivity") {
            params.scalar_diffusivity = parse_double(require_value(key), key);
        } else if (key == "--scalar-sgs") {
            params.scalar_sgs_model = require_value(key);
        } else if (key == "--prandtl-t") {
            params.prandtl_t = parse_double(require_value(key), key);
        } else if (key == "--schmidt-t") {
            params.schmidt_t = parse_double(require_value(key), key);
        } else if (key == "--surface-pressure") {
            params.surface_pressure = parse_double(require_value(key), key);
        } else if (key == "--scalar-lasd-min") {
            params.scalar_lasd_min = parse_double(require_value(key), key);
        } else if (key == "--scalar-lasd-max") {
            params.scalar_lasd_max = parse_double(require_value(key), key);
        } else if (key == "--scalar-stability-correction") {
            params.scalar_stability_correction = true;
        } else if (key == "--no-scalar-stability-correction") {
            params.scalar_stability_correction = false;
        } else if (key == "--scalar-stability-beta") {
            params.scalar_stability_beta = parse_double(require_value(key), key);
        } else if (key == "--scalar-stability-power") {
            params.scalar_stability_power = parse_double(require_value(key), key);
        } else if (key == "--smag-min-shear-fraction") {
            params.smagorinsky_min_shear_fraction = parse_double(require_value(key), key);
        } else if (key == "--sgs-delta-scale") {
            params.sgs_delta_scale = parse_double(require_value(key), key);
        } else if (key == "--dealiasing") {
            params.dealiasing = require_value(key);
        } else if (key == "--momentum-advection-form") {
            params.momentum_advection_form = require_value(key);
        } else if (key == "--spectral-filter") {
            params.spectral_filter = require_value(key);
        } else if (key == "--spectral-filter-alpha") {
            params.spectral_filter_alpha = parse_double(require_value(key), key);
        } else if (key == "--spectral-filter-order") {
            params.spectral_filter_order = parse_int(require_value(key), key);
        } else if (key == "--sponge-start-height") {
            params.sponge_enabled = true;
            params.sponge_start_height = parse_double(require_value(key), key);
        } else if (key == "--sponge-timescale") {
            params.sponge_enabled = true;
            params.sponge_timescale = parse_double(require_value(key), key);
        } else if (key == "--sponge-power") {
            params.sponge_power = parse_double(require_value(key), key);
        } else if (key == "--g") {
            params.g = parse_double(require_value(key), key);
        } else if (key == "--random-seed") {
            params.random_seed = parse_int(require_value(key), key);
        } else if (key == "--initial") {
            params.initial_condition = require_value(key);
            std::transform(params.initial_condition.begin(), params.initial_condition.end(), params.initial_condition.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
        } else if (key == "--cuda" || key == "--gpu") {
            params.cuda_enabled = true;
        } else if (key == "--mpi-slab") {
            params.mpi_slab = true;
        } else if (key == "--mpi-profile" || key == "--profile") {
            params.mpi_profile_enabled = true;
        } else if (key == "--mpi-profile-warmup" || key == "--profile-warmup") {
            params.mpi_profile_warmup_steps = parse_int(require_value(key), key);
        } else {
            throw std::runtime_error("unknown option: " + key);
        }
    }
    std::transform(params.frame_dump_component.begin(), params.frame_dump_component.end(), params.frame_dump_component.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    std::transform(params.spectral_filter.begin(), params.spectral_filter.end(), params.spectral_filter.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    std::transform(params.dealiasing.begin(), params.dealiasing.end(), params.dealiasing.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    if (params.nx < 4 || params.ny < 4 || params.nz < 4) {
        throw std::runtime_error("nx, ny, and nz must all be at least 4");
    }
    if (params.nx % 2 != 0) {
        throw std::runtime_error("nx must be even for rfft layout");
    }
    if (params.steps < 0 || params.log_every <= 0) {
        throw std::runtime_error("steps must be non-negative and log_every must be positive");
    }
    if (params.mpi_profile_warmup_steps < 0) {
        throw std::runtime_error("mpi_profile_warmup_steps must be non-negative");
    }
    if (params.lx <= 0.0 || params.ly <= 0.0 || params.lz <= 0.0 || params.z_i <= 0.0 || params.dt <= 0.0 || params.nu < 0.0) {
        throw std::runtime_error("lx, ly, lz, z_i, dt must be positive and nu must be non-negative");
    }
    if (params.time_scheme != "euler" && params.time_scheme != "ab2" && params.time_scheme != "ab3") {
        throw std::runtime_error("time_scheme must be euler, ab2, or ab3");
    }
    if (params.time_scheme == "ab3" && !params.mpi_slab) {
        throw std::runtime_error("AB3 currently requires the CPU MPI slab path");
    }
    if (params.scalar_diffusivity < 0.0 || params.moisture_diffusivity < 0.0) {
        throw std::runtime_error("scalar and moisture diffusivities must be non-negative");
    }
    if (params.momentum_wall_model != "none" && params.momentum_wall_model != "abl") {
        throw std::runtime_error("wall model must be none or abl");
    }
    if (params.wall_stress_model != "dynamic_neutral" && params.wall_stress_model != "prescribed_ustar"
        && params.wall_stress_model != "prescribed_ustar_local") {
        throw std::runtime_error(
            "wall stress model must be dynamic_neutral, prescribed_ustar, or prescribed_ustar_local");
    }
    if (params.cuda_enabled && params.wall_stress_model == "prescribed_ustar_local") {
        throw std::runtime_error(
            "prescribed_ustar_local is not implemented in the CUDA backend yet; "
            "run the CPU MPI slab path");
    }
    if (params.sgs_model != "none" && params.sgs_model != "smagorinsky" && params.sgs_model != "lasd"
        && params.sgs_model != "amd" && params.sgs_model != "amd_plane_dissipation"
        && params.sgs_model != "tke" && params.sgs_model != "moeng_tke") {
        throw std::runtime_error(
            "sgs model must be none, smagorinsky, lasd, amd, amd_plane_dissipation, tke, or moeng_tke");
    }
    if (params.scalar_sgs_model != "fixed_prandtl"
        && params.scalar_sgs_model != "fixed_smagorinsky"
        && params.scalar_sgs_model != "lasd"
        && params.scalar_sgs_model != "amd"
        && params.scalar_sgs_model != "amd_shared"
        && params.scalar_sgs_model != "amd_plane_dissipation") {
        throw std::runtime_error(
            "scalar_sgs_model must be fixed_prandtl, fixed_smagorinsky, lasd, amd, amd_shared, or amd_plane_dissipation");
    }
    if (params.scalar_sgs_model == "lasd" && params.sgs_model != "lasd") {
        throw std::runtime_error("scalar_sgs_model=lasd requires sgs.model=lasd in the C++ path");
    }
    if ((params.scalar_sgs_model == "amd"
            || params.scalar_sgs_model == "amd_shared"
            || params.scalar_sgs_model == "amd_plane_dissipation")
        && params.sgs_model != "amd" && params.sgs_model != "amd_plane_dissipation") {
        throw std::runtime_error("scalar AMD models require sgs.model=amd in the C++ path");
    }
    if (params.scalar_sgs_model == "amd_shared" && !params.moisture_enabled) {
        throw std::runtime_error("scalar_sgs_model=amd_shared requires moisture coupling");
    }
    if (params.scalar_amd_face_products && params.scalar_sgs_model != "amd") {
        throw std::runtime_error(
            "scalar_amd_face_products currently supports only the independent "
            "scalar_sgs_model=amd closure");
    }
    if (params.amd_wall_model_gradients
        && params.sgs_model != "amd" && params.sgs_model != "amd_plane_dissipation") {
        throw std::runtime_error("amd_wall_model_gradients requires sgs.model=amd");
    }
    if (params.amd_wall_model_gradients && params.momentum_wall_model != "abl") {
        throw std::runtime_error("amd_wall_model_gradients requires physics.momentum_wall_model=abl");
    }
    if ((params.amd_dissipation_averaging || params.amd_multiscale_averaging)
        && params.sgs_model != "amd") {
        throw std::runtime_error(
            "AMD local dissipation/multiscale averaging requires sgs.model=amd");
    }
    const int amd_averaging_options = static_cast<int>(params.amd_invariant_averaging)
        + static_cast<int>(params.amd_dissipation_averaging)
        + static_cast<int>(params.amd_multiscale_averaging);
    if (amd_averaging_options > 1) {
        throw std::runtime_error(
            "AMD invariant, dissipation, and multiscale averaging are separate experiments; enable only one");
    }
    if (params.cuda_enabled
        && (params.amd_invariant_averaging || params.amd_dissipation_averaging
            || params.amd_multiscale_averaging
            || params.scalar_amd_invariant_averaging
            || params.scalar_amd_face_products || params.amd_wall_model_gradients
            || params.amd_dealiased_cell_width
            || !params.amd_buoyancy_correction)) {
        throw std::runtime_error(
            "AMD invariant/dissipation averaging, wall-model gradients, scalar face products, "
            "the dealiased cell width, and the AMD buoyancy-correction toggle are not implemented "
            "in the CUDA backend yet; run the CPU MPI slab path");
    }
    if ((params.scalar_sgs_model == "amd"
            || params.scalar_sgs_model == "amd_shared"
            || params.scalar_sgs_model == "amd_plane_dissipation")
        && params.scalar_stability_correction) {
        throw std::runtime_error("scalar AMD models already include the minimum-dissipation scalar physics; disable scalar_stability_correction");
    }
    if ((params.sgs_model == "tke" || params.sgs_model == "moeng_tke") && !params.mpi_slab) {
        throw std::runtime_error("the Moeng SGS-TKE closure currently requires the CPU MPI slab path");
    }
    if (params.moisture_enabled && !params.thermo_enabled) {
        throw std::runtime_error("moisture coupling requires --thermo");
    }
    if (params.moisture_enabled && params.sgs_model != "smagorinsky"
        && params.sgs_model != "lasd"
        && params.sgs_model != "amd" && params.sgs_model != "amd_plane_dissipation"
        && params.sgs_model != "tke" && params.sgs_model != "moeng_tke") {
        throw std::runtime_error("the CPU moisture path requires --sgs smagorinsky, --sgs lasd, --sgs amd, or --sgs tke");
    }
    if (params.zo <= 0.0 || params.vonk <= 0.0 || params.fgr <= 0.0 || params.tfr <= 0.0) {
        throw std::runtime_error("zo, vonk, fgr, and tfr must be positive");
    }
    if (params.momentum_wall_model == "abl" && params.wall_ref_height() <= params.zo) {
        throw std::runtime_error("ABL wall model requires 0.5*dz > zo");
    }
    if (params.theta0 <= 0.0 || params.prandtl_t <= 0.0 || params.schmidt_t <= 0.0
        || params.surface_pressure <= 0.0 || params.g <= 0.0) {
        throw std::runtime_error("theta0, prandtl_t, schmidt_t, surface_pressure, and g must be positive");
    }
    const double qt_at_top_center = params.qv0
        + params.qv_initial_gradient * (static_cast<double>(params.nz) - 0.5) * params.dz();
    if (params.moisture_enabled && (params.qv0 < 0.0 || qt_at_top_center < 0.0)) {
        throw std::runtime_error("initial qt profile must be non-negative throughout the domain");
    }
    if (params.cs_count <= 0) {
        throw std::runtime_error("cs_count must be positive");
    }
    if (params.scalar_lasd_min < 0.0 || params.scalar_lasd_max < params.scalar_lasd_min) {
        throw std::runtime_error("scalar_lasd_min/max are invalid");
    }
    if (params.scalar_stability_beta < 0.0 || params.scalar_stability_power <= 0.0) {
        throw std::runtime_error("scalar_stability_beta must be nonnegative and scalar_stability_power must be positive");
    }
    if (params.spectral_filter != "sharp" && params.spectral_filter != "floor_sharp"
        && params.spectral_filter != "exponential") {
        throw std::runtime_error("spectral_filter must be sharp, floor_sharp, or exponential");
    }
    if (params.dealiasing != "sharp" && params.dealiasing != "padding_3_2") {
        throw std::runtime_error("dealiasing must be sharp or padding_3_2");
    }
    if (params.dealiasing == "padding_3_2" && (params.nx % 2 != 0 || params.ny % 2 != 0)) {
        throw std::runtime_error("padding_3_2 dealiasing requires even nx and ny");
    }
    if (params.momentum_advection_form != "advective"
        && params.momentum_advection_form != "skew_symmetric"
        && params.momentum_advection_form != "rotational") {
        throw std::runtime_error("momentum_advection_form must be advective, skew_symmetric, or rotational");
    }
    if ((params.momentum_advection_form == "skew_symmetric"
            || params.momentum_advection_form == "rotational")
        && !params.horizontal_dealias) {
        throw std::runtime_error("energy-conserving momentum advection requires horizontal dealiasing");
    }
    if (params.momentum_advection_form == "rotational"
        && !params.mpi_slab) {
        throw std::runtime_error(
            "rotational momentum advection currently requires an MPI slab backend");
    }
    if (params.spectral_filter_alpha < 0.0 || params.spectral_filter_order <= 0) {
        throw std::runtime_error("spectral_filter_alpha must be nonnegative and spectral_filter_order must be positive");
    }
    if (params.initial_condition != "taylor_green"
        && params.initial_condition != "largeeddy1993"
        && params.initial_condition != "neutral_ekman"
        && params.initial_condition != "bomex") {
        throw std::runtime_error("initial condition must be taylor_green, largeeddy1993, neutral_ekman, or bomex");
    }
    if (params.initial_condition == "largeeddy1993" && !params.thermo_enabled) {
        throw std::runtime_error("largeeddy1993 initial condition requires thermo enabled");
    }
    if (params.initial_condition == "neutral_ekman"
        && params.geostrophic_u == 0.0
        && params.geostrophic_v == 0.0) {
        throw std::runtime_error("neutral_ekman initial condition requires a nonzero geostrophic wind");
    }
    if (params.initial_condition == "bomex" && (!params.thermo_enabled || !params.moisture_enabled)) {
        throw std::runtime_error("bomex initial condition requires thermo and moisture enabled");
    }
    if (params.largeeddy_initial_zi1_fraction <= 0.0 || params.largeeddy_initial_zi1_fraction >= 1.0) {
        throw std::runtime_error("largeeddy_initial_zi1_fraction must lie between 0 and 1");
    }
    if (params.benchmark_sample_every <= 0
        || params.benchmark_average_start_tstar < 0.0
        || params.benchmark_average_end_tstar < params.benchmark_average_start_tstar) {
        throw std::runtime_error("benchmark sampling interval/window is invalid");
    }
    if (params.bomex_sample_every <= 0 || params.bomex_average_start_seconds < 0.0) {
        throw std::runtime_error("BOMEX sampling interval/window is invalid");
    }
    if (params.smagorinsky_cs < 0.0 || params.sgs_delta_scale <= 0.0) {
        throw std::runtime_error("smag_cs must be non-negative and sgs_delta_scale must be positive");
    }
    if (params.tke_ck <= 0.0 || params.tke_length_coefficient <= 0.0
        || params.tke_dissipation_base < 0.0 || params.tke_dissipation_slope < 0.0
        || params.tke_floor <= 0.0) {
        throw std::runtime_error("Moeng SGS-TKE constants require ck, length coefficient, and floor > 0 and dissipation constants >= 0");
    }
    if (params.smagorinsky_min_shear_fraction < 0.0 || params.smagorinsky_min_shear_fraction > 1.0) {
        throw std::runtime_error("smagorinsky_min_shear_fraction must lie between 0 and 1");
    }
    if (params.sponge_enabled) {
        if (params.sponge_start_height < 0.0 || params.sponge_start_height >= params.lz) {
            throw std::runtime_error("sponge start_height must lie in [0, lz)");
        }
        if (params.sponge_timescale <= 0.0 || params.sponge_power <= 0.0) {
            throw std::runtime_error("sponge timescale and power must be positive");
        }
    }
    if (params.initial_velocity_perturbation < 0.0 || params.initial_perturbation_height < 0.0) {
        throw std::runtime_error("initial velocity perturbation amplitude and height must be non-negative");
    }
    if (params.bomex_theta_perturbation < 0.0 || params.bomex_qt_perturbation < 0.0) {
        throw std::runtime_error("BOMEX perturbation amplitudes must be non-negative");
    }
    if (params.sgs_delta_scale <= 0.0) {
        throw std::runtime_error("SGS delta scale must be positive");
    }
    if (params.frame_dump_enabled) {
        if (params.frame_dump_every <= 0) {
            throw std::runtime_error("frame dump requires frame_every > 0");
        }
        if (params.frame_dump_start_step < 0 || params.frame_dump_end_step < -1) {
            throw std::runtime_error("frame dump step range is invalid");
        }
        if (params.frame_dump_end_step >= 0 && params.frame_dump_end_step < params.frame_dump_start_step) {
            throw std::runtime_error("frame_end_step must be >= frame_start_step");
        }
        if (params.frame_dump_y_index < -1) {
            throw std::runtime_error("frame_y_index must be -1 or a valid y index");
        }
        if (params.frame_dump_y_index >= params.ny) {
            throw std::runtime_error("frame_y_index must be less than ny");
        }
        if (params.frame_dump_z_height >= params.lz) {
            throw std::runtime_error("frame_z_height must be less than lz");
        }
        if (params.frame_dump_output_dir.empty()) {
            params.frame_dump_output_dir = "outputs/cross_section_frames";
        }
        if (params.frame_dump_component != "u" && params.frame_dump_component != "v"
            && params.frame_dump_component != "w" && params.frame_dump_component != "p"
            && params.frame_dump_component != "theta" && params.frame_dump_component != "theta_prime") {
            throw std::runtime_error("frame_component must be u, v, w, p, theta, or theta_prime");
        }
    }
    return params;
}

#ifdef WIRELES_HAVE_CPU
void print_diagnostics(int step_number, const Diagnostics& diag, const Params& params) {
    std::cout << std::setw(6) << step_number
              << "  " << std::scientific << std::setprecision(6) << diag.ke_max
              << "  " << std::scientific << std::setprecision(6) << diag.div_max
              << "  " << std::fixed << std::setprecision(6) << diag.cfl;
    if (params.moisture_enabled) {
        std::cout << "  " << std::scientific << std::setprecision(6) << diag.qv_min
                  << "  " << diag.qv_max
                  << "  " << diag.ql_max
                  << "  " << diag.column_water
                  << "  " << std::fixed << std::setprecision(6) << diag.moisture_diffusion_number;
    }
    std::cout << '\n';
}

struct BenchmarkSnapshot {
    std::vector<double> heat_flux_total;
    std::vector<double> heat_flux_resolved;
    std::vector<double> heat_flux_sgs;
    std::vector<double> heat_flux_face_total;
    std::vector<double> heat_flux_face_resolved;
    std::vector<double> heat_flux_face_sgs;
    std::vector<double> moisture_flux_total;
    std::vector<double> moisture_flux_resolved;
    std::vector<double> moisture_flux_sgs;
    std::vector<double> moisture_flux_face_total;
    std::vector<double> moisture_flux_face_resolved;
    std::vector<double> moisture_flux_face_sgs;
    std::vector<double> virtual_heat_flux_total;
    std::vector<double> virtual_heat_flux_resolved;
    std::vector<double> virtual_heat_flux_sgs;
    std::vector<double> virtual_heat_flux_face_total;
    std::vector<double> virtual_heat_flux_face_resolved;
    std::vector<double> virtual_heat_flux_face_sgs;
    std::vector<double> u_mean;
    std::vector<double> v_mean;
    std::vector<double> w_mean;
    std::vector<double> p_mean;
    std::vector<double> theta_mean;
    std::vector<double> qv_mean;
    std::vector<double> qt_mean;
    std::vector<double> ql_mean;
    std::vector<double> theta_v_mean;
    std::vector<double> u_var;
    std::vector<double> v_var;
    std::vector<double> w_var;
    std::vector<double> theta_var;
    std::vector<double> qv_var;
    std::vector<double> qt_var;
    std::vector<double> ql_var;
    std::vector<double> theta_v_var;
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
    std::vector<double> moisture_kappa_mean;
    double zi = 0.0;
    double wstar = 0.0;
};

struct BenchmarkAccumulator {
    int sample_count = 0;
    double zi_sum = 0.0;
    double wstar_sum = 0.0;
    std::vector<double> heat_flux_sum;
    std::vector<double> heat_flux_resolved_sum;
    std::vector<double> heat_flux_sgs_sum;
    std::vector<double> heat_flux_face_sum;
    std::vector<double> heat_flux_face_resolved_sum;
    std::vector<double> heat_flux_face_sgs_sum;
    std::vector<double> moisture_flux_sum;
    std::vector<double> moisture_flux_resolved_sum;
    std::vector<double> moisture_flux_sgs_sum;
    std::vector<double> moisture_flux_face_sum;
    std::vector<double> moisture_flux_face_resolved_sum;
    std::vector<double> moisture_flux_face_sgs_sum;
    std::vector<double> virtual_heat_flux_sum;
    std::vector<double> virtual_heat_flux_resolved_sum;
    std::vector<double> virtual_heat_flux_sgs_sum;
    std::vector<double> virtual_heat_flux_face_sum;
    std::vector<double> virtual_heat_flux_face_resolved_sum;
    std::vector<double> virtual_heat_flux_face_sgs_sum;
    std::vector<double> u_mean_sum;
    std::vector<double> v_mean_sum;
    std::vector<double> w_mean_sum;
    std::vector<double> p_mean_sum;
    std::vector<double> theta_mean_sum;
    std::vector<double> qv_mean_sum;
    std::vector<double> qt_mean_sum;
    std::vector<double> ql_mean_sum;
    std::vector<double> theta_v_mean_sum;
    std::vector<double> u_var_sum;
    std::vector<double> v_var_sum;
    std::vector<double> w_var_sum;
    std::vector<double> theta_var_sum;
    std::vector<double> qv_var_sum;
    std::vector<double> qt_var_sum;
    std::vector<double> ql_var_sum;
    std::vector<double> theta_v_var_sum;
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
    std::vector<double> moisture_kappa_mean_sum;
    double initial_column_water = 0.0;
    double final_column_water = 0.0;
    double expected_column_water = 0.0;
    double max_abs_column_water_error = 0.0;
    double qv_min = 0.0;
    double qv_max = 0.0;
    double qt_min = 0.0;
    double qt_max = 0.0;
    double ql_max = 0.0;
    std::size_t moisture_limiter_activations = 0;
    double moisture_limiter_column_correction = 0.0;
};

double convective_wstar(const Params& params, double zi) {
    return std::cbrt((params.g / params.theta0) * params.surface_theta_flux * zi);
}

BenchmarkSnapshot benchmark_snapshot(const FlowState& state, const Params& params, FftwXY& fft) {
    BenchmarkSnapshot snapshot;
    snapshot.heat_flux_total.assign(static_cast<std::size_t>(params.nz), 0.0);
    snapshot.heat_flux_resolved.assign(static_cast<std::size_t>(params.nz), 0.0);
    snapshot.heat_flux_sgs.assign(static_cast<std::size_t>(params.nz), 0.0);
    snapshot.heat_flux_face_total.assign(static_cast<std::size_t>(params.nz + 1), 0.0);
    snapshot.heat_flux_face_resolved.assign(static_cast<std::size_t>(params.nz + 1), 0.0);
    snapshot.heat_flux_face_sgs.assign(static_cast<std::size_t>(params.nz + 1), 0.0);
    snapshot.moisture_flux_total.assign(static_cast<std::size_t>(params.nz), 0.0);
    snapshot.moisture_flux_resolved.assign(static_cast<std::size_t>(params.nz), 0.0);
    snapshot.moisture_flux_sgs.assign(static_cast<std::size_t>(params.nz), 0.0);
    snapshot.moisture_flux_face_total.assign(static_cast<std::size_t>(params.nz + 1), 0.0);
    snapshot.moisture_flux_face_resolved.assign(static_cast<std::size_t>(params.nz + 1), 0.0);
    snapshot.moisture_flux_face_sgs.assign(static_cast<std::size_t>(params.nz + 1), 0.0);
    snapshot.virtual_heat_flux_total.assign(static_cast<std::size_t>(params.nz), 0.0);
    snapshot.virtual_heat_flux_resolved.assign(static_cast<std::size_t>(params.nz), 0.0);
    snapshot.virtual_heat_flux_sgs.assign(static_cast<std::size_t>(params.nz), 0.0);
    snapshot.virtual_heat_flux_face_total.assign(static_cast<std::size_t>(params.nz + 1), 0.0);
    snapshot.virtual_heat_flux_face_resolved.assign(static_cast<std::size_t>(params.nz + 1), 0.0);
    snapshot.virtual_heat_flux_face_sgs.assign(static_cast<std::size_t>(params.nz + 1), 0.0);
    snapshot.u_mean.assign(static_cast<std::size_t>(params.nz), 0.0);
    snapshot.v_mean.assign(static_cast<std::size_t>(params.nz), 0.0);
    snapshot.w_mean.assign(static_cast<std::size_t>(params.nz), 0.0);
    snapshot.p_mean.assign(static_cast<std::size_t>(params.nz), 0.0);
    snapshot.theta_mean.assign(static_cast<std::size_t>(params.nz), 0.0);
    snapshot.qv_mean.assign(static_cast<std::size_t>(params.nz), 0.0);
    snapshot.qt_mean.assign(static_cast<std::size_t>(params.nz), 0.0);
    snapshot.ql_mean.assign(static_cast<std::size_t>(params.nz), 0.0);
    snapshot.theta_v_mean.assign(static_cast<std::size_t>(params.nz), 0.0);
    snapshot.u_var.assign(static_cast<std::size_t>(params.nz), 0.0);
    snapshot.v_var.assign(static_cast<std::size_t>(params.nz), 0.0);
    snapshot.w_var.assign(static_cast<std::size_t>(params.nz), 0.0);
    snapshot.theta_var.assign(static_cast<std::size_t>(params.nz), 0.0);
    snapshot.qv_var.assign(static_cast<std::size_t>(params.nz), 0.0);
    snapshot.qt_var.assign(static_cast<std::size_t>(params.nz), 0.0);
    snapshot.ql_var.assign(static_cast<std::size_t>(params.nz), 0.0);
    snapshot.theta_v_var.assign(static_cast<std::size_t>(params.nz), 0.0);
    snapshot.p_var.assign(static_cast<std::size_t>(params.nz), 0.0);
    snapshot.w3.assign(static_cast<std::size_t>(params.nz), 0.0);
    snapshot.w_transport.assign(static_cast<std::size_t>(params.nz), 0.0);
    snapshot.p_transport.assign(static_cast<std::size_t>(params.nz), 0.0);
    snapshot.alpha_u.assign(static_cast<std::size_t>(params.nz), 0.0);
    snapshot.w_u.assign(static_cast<std::size_t>(params.nz), 0.0);
    snapshot.theta_u_excess.assign(static_cast<std::size_t>(params.nz), 0.0);
    snapshot.epsilon.assign(static_cast<std::size_t>(params.nz), 0.0);
    snapshot.cs2_mean.assign(static_cast<std::size_t>(params.nz), 0.0);
    snapshot.scalar_c_mean.assign(static_cast<std::size_t>(params.nz), 0.0);
    snapshot.kappa_mean.assign(static_cast<std::size_t>(params.nz), 0.0);
    snapshot.moisture_kappa_mean.assign(static_cast<std::size_t>(params.nz), 0.0);

    const double inv_plane = 1.0 / static_cast<double>(params.nx * params.ny);
    const Field w_center = w_to_center(state.w, params);
    const Field theta_v = virtual_potential_temperature(state, params);

    for (int k = 0; k < params.nz; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                snapshot.u_mean[static_cast<std::size_t>(k)] += state.u[n];
                snapshot.v_mean[static_cast<std::size_t>(k)] += state.v[n];
                snapshot.w_mean[static_cast<std::size_t>(k)] += w_center[n];
                snapshot.p_mean[static_cast<std::size_t>(k)] += state.p[n];
                snapshot.theta_mean[static_cast<std::size_t>(k)] += state.theta[n];
                snapshot.qv_mean[static_cast<std::size_t>(k)] += state.qv[n];
                snapshot.qt_mean[static_cast<std::size_t>(k)] += state.qt[n];
                snapshot.ql_mean[static_cast<std::size_t>(k)] += state.ql[n];
                snapshot.theta_v_mean[static_cast<std::size_t>(k)] += theta_v[n];
                snapshot.cs2_mean[static_cast<std::size_t>(k)] += state.cs2[n];
                snapshot.scalar_c_mean[static_cast<std::size_t>(k)] += state.scalar_c[n];
            }
        }
        snapshot.u_mean[static_cast<std::size_t>(k)] *= inv_plane;
        snapshot.v_mean[static_cast<std::size_t>(k)] *= inv_plane;
        snapshot.w_mean[static_cast<std::size_t>(k)] *= inv_plane;
        snapshot.p_mean[static_cast<std::size_t>(k)] *= inv_plane;
        snapshot.theta_mean[static_cast<std::size_t>(k)] *= inv_plane;
        snapshot.qv_mean[static_cast<std::size_t>(k)] *= inv_plane;
        snapshot.qt_mean[static_cast<std::size_t>(k)] *= inv_plane;
        snapshot.ql_mean[static_cast<std::size_t>(k)] *= inv_plane;
        snapshot.theta_v_mean[static_cast<std::size_t>(k)] *= inv_plane;
        snapshot.cs2_mean[static_cast<std::size_t>(k)] *= inv_plane;
        snapshot.scalar_c_mean[static_cast<std::size_t>(k)] *= inv_plane;
    }

    for (int k = 0; k < params.nz; ++k) {
        int updraft_count = 0;
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                const double up = state.u[n] - snapshot.u_mean[static_cast<std::size_t>(k)];
                const double vp = state.v[n] - snapshot.v_mean[static_cast<std::size_t>(k)];
                const double wp = w_center[n] - snapshot.w_mean[static_cast<std::size_t>(k)];
                const double pp = state.p[n] - snapshot.p_mean[static_cast<std::size_t>(k)];
                const double thetap = state.theta[n] - snapshot.theta_mean[static_cast<std::size_t>(k)];
                const double qvp = state.qv[n] - snapshot.qv_mean[static_cast<std::size_t>(k)];
                const double qtp = state.qt[n] - snapshot.qt_mean[static_cast<std::size_t>(k)];
                const double qlp = state.ql[n] - snapshot.ql_mean[static_cast<std::size_t>(k)];
                const double theta_vp = theta_v[n] - snapshot.theta_v_mean[static_cast<std::size_t>(k)];
                const double energy = 0.5 * (up * up + vp * vp + wp * wp);
                snapshot.heat_flux_resolved[static_cast<std::size_t>(k)] +=
                    wp * thetap;
                snapshot.moisture_flux_resolved[static_cast<std::size_t>(k)] += wp * qtp;
                snapshot.virtual_heat_flux_resolved[static_cast<std::size_t>(k)] += wp * theta_vp;
                snapshot.u_var[static_cast<std::size_t>(k)] += up * up;
                snapshot.v_var[static_cast<std::size_t>(k)] += vp * vp;
                snapshot.w_var[static_cast<std::size_t>(k)] += wp * wp;
                snapshot.theta_var[static_cast<std::size_t>(k)] += thetap * thetap;
                snapshot.qv_var[static_cast<std::size_t>(k)] += qvp * qvp;
                snapshot.qt_var[static_cast<std::size_t>(k)] += qtp * qtp;
                snapshot.ql_var[static_cast<std::size_t>(k)] += qlp * qlp;
                snapshot.theta_v_var[static_cast<std::size_t>(k)] += theta_vp * theta_vp;
                snapshot.p_var[static_cast<std::size_t>(k)] += pp * pp;
                snapshot.w3[static_cast<std::size_t>(k)] += wp * wp * wp;
                snapshot.w_transport[static_cast<std::size_t>(k)] += wp * energy;
                snapshot.p_transport[static_cast<std::size_t>(k)] += pp * wp;
                if (wp > 0.0) {
                    ++updraft_count;
                    snapshot.w_u[static_cast<std::size_t>(k)] += w_center[n];
                    snapshot.theta_u_excess[static_cast<std::size_t>(k)] += thetap;
                }
            }
        }
        snapshot.heat_flux_resolved[static_cast<std::size_t>(k)] *= inv_plane;
        snapshot.moisture_flux_resolved[static_cast<std::size_t>(k)] *= inv_plane;
        snapshot.virtual_heat_flux_resolved[static_cast<std::size_t>(k)] *= inv_plane;
        snapshot.u_var[static_cast<std::size_t>(k)] *= inv_plane;
        snapshot.v_var[static_cast<std::size_t>(k)] *= inv_plane;
        snapshot.w_var[static_cast<std::size_t>(k)] *= inv_plane;
        snapshot.theta_var[static_cast<std::size_t>(k)] *= inv_plane;
        snapshot.qv_var[static_cast<std::size_t>(k)] *= inv_plane;
        snapshot.qt_var[static_cast<std::size_t>(k)] *= inv_plane;
        snapshot.ql_var[static_cast<std::size_t>(k)] *= inv_plane;
        snapshot.theta_v_var[static_cast<std::size_t>(k)] *= inv_plane;
        snapshot.p_var[static_cast<std::size_t>(k)] *= inv_plane;
        snapshot.w3[static_cast<std::size_t>(k)] *= inv_plane;
        snapshot.w_transport[static_cast<std::size_t>(k)] *= inv_plane;
        snapshot.p_transport[static_cast<std::size_t>(k)] *= inv_plane;
        snapshot.alpha_u[static_cast<std::size_t>(k)] = static_cast<double>(updraft_count) * inv_plane;
        if (updraft_count > 0) {
            snapshot.w_u[static_cast<std::size_t>(k)] /= static_cast<double>(updraft_count);
            snapshot.theta_u_excess[static_cast<std::size_t>(k)] /= static_cast<double>(updraft_count);
        }
    }

    const VelocityGradients grad = velocity_gradients(state, params, fft);
    const Field strain = strain_magnitude(grad, params);
    const Field nu_t = current_sgs_eddy_viscosity(state, grad, params, fft);
    const Field kappa_center = scalar_eddy_diffusivity(state, nu_t, strain, params, fft);
    const Field kappa_face = center_to_w(kappa_center, params);
    const Field moisture_kappa_center = moisture_eddy_diffusivity(state, nu_t, strain, params, fft);
    const Field moisture_kappa_face = center_to_w(moisture_kappa_center, params);
    const Field& transported_theta = params.moisture_enabled ? state.theta_l : state.theta;
    const Field dtheta_dz_face = ddz_center_to_w(transported_theta, params);
    const Field dqt_dz_face = ddz_center_to_w(state.qt, params);
    Field qz(params.z_face_size(), 0.0);
    Field qvz(params.z_face_size(), 0.0);
    for (int k = 1; k < params.nz; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t face = z_face_idx(params, i, j, k);
                qz[face] = -kappa_face[face] * dtheta_dz_face[face];
                qvz[face] = -moisture_kappa_face[face] * dqt_dz_face[face];
            }
        }
    }
    for (int j = 0; j < params.ny; ++j) {
        for (int i = 0; i < params.nx; ++i) {
            qz[z_face_idx(params, i, j, 0)] = params.surface_theta_flux;
            qz[z_face_idx(params, i, j, params.nz)] = 0.0;
            qvz[z_face_idx(params, i, j, 0)] = params.surface_qv_flux;
            qvz[z_face_idx(params, i, j, params.nz)] = 0.0;
        }
    }

    Field theta_face = center_to_w(state.theta, params);
    Field qv_face = center_to_w(state.qv, params);
    Field qt_face = center_to_w(state.qt, params);
    Field theta_v_face = center_to_w(theta_v, params);
    for (int j = 0; j < params.ny; ++j) {
        for (int i = 0; i < params.nx; ++i) {
            for (int face_k : {0, params.nz}) {
                const int center_k = face_k == 0 ? 0 : params.nz - 1;
                const std::size_t face = z_face_idx(params, i, j, face_k);
                const std::size_t center = idx(params, i, j, center_k);
                theta_face[face] = state.theta[center];
                qv_face[face] = state.qv[center];
                qt_face[face] = state.qt[center];
                theta_v_face[face] = theta_v[center];
            }
        }
    }
    Field virtual_qz(params.z_face_size(), 0.0);
    for (std::size_t face = 0; face < virtual_qz.size(); ++face) {
        virtual_qz[face] = (1.0 + 0.61 * qv_face[face]) * qz[face]
            + 0.61 * theta_face[face] * qvz[face];
    }
    std::vector<double> w_face_mean(static_cast<std::size_t>(params.nz + 1), 0.0);
    std::vector<double> theta_face_mean(static_cast<std::size_t>(params.nz + 1), 0.0);
    std::vector<double> qt_face_mean(static_cast<std::size_t>(params.nz + 1), 0.0);
    std::vector<double> theta_v_face_mean(static_cast<std::size_t>(params.nz + 1), 0.0);
    for (int k = 0; k <= params.nz; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t face = z_face_idx(params, i, j, k);
                w_face_mean[static_cast<std::size_t>(k)] += state.w[face];
                theta_face_mean[static_cast<std::size_t>(k)] += theta_face[face];
                qt_face_mean[static_cast<std::size_t>(k)] += qt_face[face];
                theta_v_face_mean[static_cast<std::size_t>(k)] += theta_v_face[face];
            }
        }
        w_face_mean[static_cast<std::size_t>(k)] *= inv_plane;
        theta_face_mean[static_cast<std::size_t>(k)] *= inv_plane;
        qt_face_mean[static_cast<std::size_t>(k)] *= inv_plane;
        theta_v_face_mean[static_cast<std::size_t>(k)] *= inv_plane;
    }
    for (int k = 0; k <= params.nz; ++k) {
        double resolved = 0.0;
        double sgs = 0.0;
        double moisture_resolved = 0.0;
        double moisture_sgs = 0.0;
        double virtual_resolved = 0.0;
        double virtual_sgs = 0.0;
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t face = z_face_idx(params, i, j, k);
                resolved +=
                    (state.w[face] - w_face_mean[static_cast<std::size_t>(k)])
                    * (theta_face[face] - theta_face_mean[static_cast<std::size_t>(k)]);
                sgs += qz[face];
                moisture_resolved +=
                    (state.w[face] - w_face_mean[static_cast<std::size_t>(k)])
                    * (qt_face[face] - qt_face_mean[static_cast<std::size_t>(k)]);
                moisture_sgs += qvz[face];
                virtual_resolved +=
                    (state.w[face] - w_face_mean[static_cast<std::size_t>(k)])
                    * (theta_v_face[face] - theta_v_face_mean[static_cast<std::size_t>(k)]);
                virtual_sgs += virtual_qz[face];
            }
        }
        resolved *= inv_plane;
        sgs *= inv_plane;
        moisture_resolved *= inv_plane;
        moisture_sgs *= inv_plane;
        virtual_resolved *= inv_plane;
        virtual_sgs *= inv_plane;
        snapshot.heat_flux_face_resolved[static_cast<std::size_t>(k)] = resolved;
        snapshot.heat_flux_face_sgs[static_cast<std::size_t>(k)] = sgs;
        snapshot.heat_flux_face_total[static_cast<std::size_t>(k)] = resolved + sgs;
        snapshot.moisture_flux_face_resolved[static_cast<std::size_t>(k)] = moisture_resolved;
        snapshot.moisture_flux_face_sgs[static_cast<std::size_t>(k)] = moisture_sgs;
        snapshot.moisture_flux_face_total[static_cast<std::size_t>(k)] = moisture_resolved + moisture_sgs;
        snapshot.virtual_heat_flux_face_resolved[static_cast<std::size_t>(k)] = virtual_resolved;
        snapshot.virtual_heat_flux_face_sgs[static_cast<std::size_t>(k)] = virtual_sgs;
        snapshot.virtual_heat_flux_face_total[static_cast<std::size_t>(k)] = virtual_resolved + virtual_sgs;
    }

    double min_heat_flux = std::numeric_limits<double>::infinity();
    int min_k = 0;
    for (int k = 0; k < params.nz; ++k) {
        double qz_center_mean = 0.0;
        double kappa_mean = 0.0;
        double moisture_kappa_mean = 0.0;
        double epsilon_mean = 0.0;
        double moisture_qz_center_mean = 0.0;
        double virtual_qz_center_mean = 0.0;
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                qz_center_mean += 0.5 * (qz[z_face_idx(params, i, j, k)] + qz[z_face_idx(params, i, j, k + 1)]);
                kappa_mean += kappa_center[n];
                moisture_kappa_mean += moisture_kappa_center[n];
                epsilon_mean += nu_t[n] * strain[n] * strain[n];
                moisture_qz_center_mean += 0.5 * (
                    qvz[z_face_idx(params, i, j, k)] + qvz[z_face_idx(params, i, j, k + 1)]);
                virtual_qz_center_mean += 0.5 * (
                    virtual_qz[z_face_idx(params, i, j, k)] + virtual_qz[z_face_idx(params, i, j, k + 1)]);
            }
        }
        qz_center_mean *= inv_plane;
        kappa_mean *= inv_plane;
        moisture_kappa_mean *= inv_plane;
        epsilon_mean *= inv_plane;
        moisture_qz_center_mean *= inv_plane;
        virtual_qz_center_mean *= inv_plane;
        snapshot.heat_flux_sgs[static_cast<std::size_t>(k)] = qz_center_mean;
        snapshot.kappa_mean[static_cast<std::size_t>(k)] = kappa_mean;
        snapshot.moisture_kappa_mean[static_cast<std::size_t>(k)] = moisture_kappa_mean;
        snapshot.epsilon[static_cast<std::size_t>(k)] = epsilon_mean;
        snapshot.heat_flux_total[static_cast<std::size_t>(k)] =
            snapshot.heat_flux_resolved[static_cast<std::size_t>(k)] + qz_center_mean;
        snapshot.moisture_flux_sgs[static_cast<std::size_t>(k)] = moisture_qz_center_mean;
        snapshot.moisture_flux_total[static_cast<std::size_t>(k)] =
            snapshot.moisture_flux_resolved[static_cast<std::size_t>(k)] + moisture_qz_center_mean;
        snapshot.virtual_heat_flux_sgs[static_cast<std::size_t>(k)] = virtual_qz_center_mean;
        snapshot.virtual_heat_flux_total[static_cast<std::size_t>(k)] =
            snapshot.virtual_heat_flux_resolved[static_cast<std::size_t>(k)] + virtual_qz_center_mean;
        if (snapshot.heat_flux_total[static_cast<std::size_t>(k)] < min_heat_flux) {
            min_heat_flux = snapshot.heat_flux_total[static_cast<std::size_t>(k)];
            min_k = k;
        }
    }

    snapshot.zi = (static_cast<double>(min_k) + 0.5) * params.dz();
    snapshot.wstar = convective_wstar(params, snapshot.zi);
    return snapshot;
}

void add_benchmark_sample(BenchmarkAccumulator& accumulator, const BenchmarkSnapshot& snapshot) {
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
        init(accumulator.moisture_flux_sum, snapshot.moisture_flux_total);
        init(accumulator.moisture_flux_resolved_sum, snapshot.moisture_flux_resolved);
        init(accumulator.moisture_flux_sgs_sum, snapshot.moisture_flux_sgs);
        init(accumulator.moisture_flux_face_sum, snapshot.moisture_flux_face_total);
        init(accumulator.moisture_flux_face_resolved_sum, snapshot.moisture_flux_face_resolved);
        init(accumulator.moisture_flux_face_sgs_sum, snapshot.moisture_flux_face_sgs);
        init(accumulator.virtual_heat_flux_sum, snapshot.virtual_heat_flux_total);
        init(accumulator.virtual_heat_flux_resolved_sum, snapshot.virtual_heat_flux_resolved);
        init(accumulator.virtual_heat_flux_sgs_sum, snapshot.virtual_heat_flux_sgs);
        init(accumulator.virtual_heat_flux_face_sum, snapshot.virtual_heat_flux_face_total);
        init(accumulator.virtual_heat_flux_face_resolved_sum, snapshot.virtual_heat_flux_face_resolved);
        init(accumulator.virtual_heat_flux_face_sgs_sum, snapshot.virtual_heat_flux_face_sgs);
        init(accumulator.u_mean_sum, snapshot.u_mean);
        init(accumulator.v_mean_sum, snapshot.v_mean);
        init(accumulator.w_mean_sum, snapshot.w_mean);
        init(accumulator.p_mean_sum, snapshot.p_mean);
        init(accumulator.theta_mean_sum, snapshot.theta_mean);
        init(accumulator.qv_mean_sum, snapshot.qv_mean);
        init(accumulator.qt_mean_sum, snapshot.qt_mean);
        init(accumulator.ql_mean_sum, snapshot.ql_mean);
        init(accumulator.theta_v_mean_sum, snapshot.theta_v_mean);
        init(accumulator.u_var_sum, snapshot.u_var);
        init(accumulator.v_var_sum, snapshot.v_var);
        init(accumulator.w_var_sum, snapshot.w_var);
        init(accumulator.theta_var_sum, snapshot.theta_var);
        init(accumulator.qv_var_sum, snapshot.qv_var);
        init(accumulator.qt_var_sum, snapshot.qt_var);
        init(accumulator.ql_var_sum, snapshot.ql_var);
        init(accumulator.theta_v_var_sum, snapshot.theta_v_var);
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
        init(accumulator.moisture_kappa_mean_sum, snapshot.moisture_kappa_mean);
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
    add(accumulator.moisture_flux_sum, snapshot.moisture_flux_total);
    add(accumulator.moisture_flux_resolved_sum, snapshot.moisture_flux_resolved);
    add(accumulator.moisture_flux_sgs_sum, snapshot.moisture_flux_sgs);
    add(accumulator.moisture_flux_face_sum, snapshot.moisture_flux_face_total);
    add(accumulator.moisture_flux_face_resolved_sum, snapshot.moisture_flux_face_resolved);
    add(accumulator.moisture_flux_face_sgs_sum, snapshot.moisture_flux_face_sgs);
    add(accumulator.virtual_heat_flux_sum, snapshot.virtual_heat_flux_total);
    add(accumulator.virtual_heat_flux_resolved_sum, snapshot.virtual_heat_flux_resolved);
    add(accumulator.virtual_heat_flux_sgs_sum, snapshot.virtual_heat_flux_sgs);
    add(accumulator.virtual_heat_flux_face_sum, snapshot.virtual_heat_flux_face_total);
    add(accumulator.virtual_heat_flux_face_resolved_sum, snapshot.virtual_heat_flux_face_resolved);
    add(accumulator.virtual_heat_flux_face_sgs_sum, snapshot.virtual_heat_flux_face_sgs);
    add(accumulator.u_mean_sum, snapshot.u_mean);
    add(accumulator.v_mean_sum, snapshot.v_mean);
    add(accumulator.w_mean_sum, snapshot.w_mean);
    add(accumulator.p_mean_sum, snapshot.p_mean);
    add(accumulator.theta_mean_sum, snapshot.theta_mean);
    add(accumulator.qv_mean_sum, snapshot.qv_mean);
    add(accumulator.qt_mean_sum, snapshot.qt_mean);
    add(accumulator.ql_mean_sum, snapshot.ql_mean);
    add(accumulator.theta_v_mean_sum, snapshot.theta_v_mean);
    add(accumulator.u_var_sum, snapshot.u_var);
    add(accumulator.v_var_sum, snapshot.v_var);
    add(accumulator.w_var_sum, snapshot.w_var);
    add(accumulator.theta_var_sum, snapshot.theta_var);
    add(accumulator.qv_var_sum, snapshot.qv_var);
    add(accumulator.qt_var_sum, snapshot.qt_var);
    add(accumulator.ql_var_sum, snapshot.ql_var);
    add(accumulator.theta_v_var_sum, snapshot.theta_v_var);
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
    add(accumulator.moisture_kappa_mean_sum, snapshot.moisture_kappa_mean);
}

void initialize_moisture_budget(
    BenchmarkAccumulator& accumulator,
    const FlowState& state,
    const Params& params) {
    if (!params.moisture_enabled) {
        return;
    }
    accumulator.initial_column_water = column_integrated_water(state.qt, params);
    accumulator.final_column_water = accumulator.initial_column_water;
    accumulator.expected_column_water = accumulator.initial_column_water;
    const auto [qv_min, qv_max] = std::minmax_element(state.qv.begin(), state.qv.end());
    accumulator.qv_min = *qv_min;
    accumulator.qv_max = *qv_max;
    const auto [qt_min, qt_max] = std::minmax_element(state.qt.begin(), state.qt.end());
    accumulator.qt_min = *qt_min;
    accumulator.qt_max = *qt_max;
    accumulator.ql_max = *std::max_element(state.ql.begin(), state.ql.end());
}

void update_moisture_budget(
    BenchmarkAccumulator& accumulator,
    const FlowState& state,
    const Params& params) {
    if (!params.moisture_enabled) {
        return;
    }
    accumulator.final_column_water = column_integrated_water(state.qt, params);
    if (params.initial_condition == "bomex") {
        accumulator.expected_column_water += params.dt * (
            bomex_surface_qt_mixing_ratio_flux(params.surface_qv_flux)
                + bomex_column_water_large_scale_tendency(state, params));
    } else {
        accumulator.expected_column_water = accumulator.initial_column_water
            + static_cast<double>(state.step_count) * params.dt * params.surface_qv_flux;
    }
    accumulator.max_abs_column_water_error = std::max(
        accumulator.max_abs_column_water_error,
        std::abs(accumulator.final_column_water - accumulator.expected_column_water));
    const auto [qv_min, qv_max] = std::minmax_element(state.qv.begin(), state.qv.end());
    accumulator.qv_min = *qv_min;
    accumulator.qv_max = *qv_max;
    const auto [qt_min, qt_max] = std::minmax_element(state.qt.begin(), state.qt.end());
    accumulator.qt_min = *qt_min;
    accumulator.qt_max = *qt_max;
    accumulator.ql_max = *std::max_element(state.ql.begin(), state.ql.end());
    accumulator.moisture_limiter_activations = state.moisture_limiter_activations;
    accumulator.moisture_limiter_column_correction = state.moisture_limiter_column_correction;
}

void print_benchmark_summary(BenchmarkAccumulator& accumulator, const Params& params) {
    auto print_moisture_budget = [&]() {
        if (!params.moisture_enabled) {
            return;
        }
        std::cout << "[moisture] column-water budget\n"
                  << "  initial: " << std::setprecision(10) << accumulator.initial_column_water << '\n'
                  << "  final: " << accumulator.final_column_water << '\n'
                  << "  expected: " << accumulator.expected_column_water << '\n'
                  << "  error: " << accumulator.final_column_water - accumulator.expected_column_water << '\n'
                  << "  max_abs_error: " << accumulator.max_abs_column_water_error << '\n'
                  << "  limiter_activations: " << accumulator.moisture_limiter_activations << '\n';
    };
    if (!params.benchmark_enabled) {
        print_moisture_budget();
        return;
    }
    if (accumulator.sample_count == 0) {
        std::cout << "[benchmark] no samples in configured averaging window\n";
        print_moisture_budget();
        return;
    }

    const double inv_count = 1.0 / static_cast<double>(accumulator.sample_count);
    const double zi_instantaneous_mean = accumulator.zi_sum * inv_count;
    const double wstar_instantaneous_mean = accumulator.wstar_sum * inv_count;
    const double wstar0 = convective_wstar(params, params.z_i);
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
    const double wstar_mean = convective_wstar(params, zi_mean);
    double theta_mixed_sum = 0.0;
    int theta_mixed_count = 0;
    for (int k = 0; k < params.nz; ++k) {
        const double z = (static_cast<double>(k) + 0.5) * params.dz();
        if (z < 0.8 * zi_mean) {
            theta_mixed_sum += accumulator.theta_mean_sum[static_cast<std::size_t>(k)] * inv_count;
            ++theta_mixed_count;
        }
    }
    const double theta_mixed_mean = theta_mixed_count > 0 ? theta_mixed_sum / static_cast<double>(theta_mixed_count) : 0.0;
    const double zi_over_zi0 = zi_mean / params.z_i;
    const double wstar_over_wstar0 = wstar_mean / wstar0;
    const double entrainment_ratio = -min_heat_flux / params.surface_theta_flux;

    if (params.moisture_enabled) {
        std::cout << "[diagnostics] moist CBL profile summary\n"
                  << "  sample_count: " << accumulator.sample_count << '\n'
                  << "  zi_over_zi0: " << std::setprecision(6) << zi_over_zi0 << '\n'
                  << "  wstar_over_wstar0: " << wstar_over_wstar0 << '\n'
                  << "  theta_mixed_layer_mean: " << theta_mixed_mean << '\n';
        print_moisture_budget();
        return;
    }

    auto print_compare = [](const char* name, double value, double target, double tolerance) {
        const bool pass = std::abs(value - target) <= tolerance;
        std::cout << "  " << name << ": " << std::setprecision(6) << value
                  << "  target=" << target << " +/- " << tolerance
                  << "  " << (pass ? "PASS" : "FAIL") << '\n';
    };

    std::cout << "[benchmark] Moeng Table 3 comparison\n";
    std::cout << "  sample_count: " << accumulator.sample_count << '\n';
    print_compare("zi_over_zi0", zi_over_zi0, 1.0312, 0.02);
    print_compare("wstar_over_wstar0", wstar_over_wstar0, 1.010, 0.015);
    print_compare("entrainment_ratio", entrainment_ratio, 0.106, 0.03);
    std::cout << "  theta_mixed_layer_mean: " << std::setprecision(6) << theta_mixed_mean << '\n';
    std::cout << "  instantaneous_zi_mean_over_zi0: "
              << std::setprecision(6) << zi_instantaneous_mean / params.z_i << '\n';
    std::cout << "  instantaneous_wstar_mean_over_wstar0: "
              << std::setprecision(6) << wstar_instantaneous_mean / wstar0 << '\n';
    const std::size_t ek = static_cast<std::size_t>(min_heat_flux_k);
    const double z_entrainment = (static_cast<double>(min_heat_flux_k) + 0.5) * params.dz();
    std::cout << "  entrainment_layer_z_over_zi0: " << std::setprecision(6) << z_entrainment / params.z_i << '\n';
    std::cout << "  entrainment_heat_flux_resolved_over_qs: "
              << std::setprecision(6) << (accumulator.heat_flux_resolved_sum[ek] * inv_count) / params.surface_theta_flux << '\n';
    std::cout << "  entrainment_heat_flux_sgs_over_qs: "
              << std::setprecision(6) << (accumulator.heat_flux_sgs_sum[ek] * inv_count) / params.surface_theta_flux << '\n';
    std::cout << "  entrainment_cs2_mean: "
              << std::setprecision(6) << accumulator.cs2_mean_sum[ek] * inv_count << '\n';
    std::cout << "  entrainment_scalar_c_mean: "
              << std::setprecision(6) << accumulator.scalar_c_mean_sum[ek] * inv_count << '\n';
    std::cout << "  entrainment_kappa_mean: "
              << std::setprecision(6) << accumulator.kappa_mean_sum[ek] * inv_count << '\n';
    print_moisture_budget();
}

void ensure_directory(const std::string& path) {
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

std::string join_path(const std::string& directory, const std::string& name) {
    if (directory.empty()) {
        return name;
    }
    return directory.back() == '/' ? directory + name : directory + "/" + name;
}

double averaged_at(const std::vector<double>& values, int k, double inv_count) {
    return values[static_cast<std::size_t>(k)] * inv_count;
}

double gradient_at(const std::vector<double>& values, int k, const Params& params, double inv_count) {
    if (params.nz <= 1) {
        return 0.0;
    }
    if (k == 0) {
        return (averaged_at(values, 1, inv_count) - averaged_at(values, 0, inv_count)) / params.dz();
    }
    if (k == params.nz - 1) {
        return (averaged_at(values, params.nz - 1, inv_count) - averaged_at(values, params.nz - 2, inv_count)) / params.dz();
    }
    return (averaged_at(values, k + 1, inv_count) - averaged_at(values, k - 1, inv_count)) / (2.0 * params.dz());
}

void write_benchmark_outputs(const BenchmarkAccumulator& accumulator, const Params& params) {
    if (!params.benchmark_enabled || params.benchmark_output_dir.empty()) {
        return;
    }
    if (accumulator.sample_count == 0 || accumulator.heat_flux_sum.empty()) {
        std::cout << "[benchmark] no profile CSV written because no samples were collected\n";
        return;
    }

    ensure_directory(params.benchmark_output_dir);
    const double inv_count = 1.0 / static_cast<double>(accumulator.sample_count);
    const double wstar0 = convective_wstar(params, params.z_i);
    double min_heat_flux = std::numeric_limits<double>::infinity();
    int min_heat_flux_k = 0;
    for (int k = 0; k < params.nz; ++k) {
        const double value = averaged_at(accumulator.heat_flux_sum, k, inv_count);
        if (value < min_heat_flux) {
            min_heat_flux = value;
            min_heat_flux_k = k;
        }
    }

    const double zi_mean = (static_cast<double>(min_heat_flux_k) + 0.5) * params.dz();
    const double wstar = convective_wstar(params, zi_mean);
    const double theta_star = params.surface_theta_flux / wstar;
    const double transport_norm = wstar * wstar * wstar / zi_mean;
    const double instantaneous_zi_mean = accumulator.zi_sum * inv_count;
    const double instantaneous_wstar_mean = accumulator.wstar_sum * inv_count;

    double theta_mixed_sum = 0.0;
    int theta_mixed_count = 0;
    for (int k = 0; k < params.nz; ++k) {
        const double z = (static_cast<double>(k) + 0.5) * params.dz();
        if (z < 0.8 * zi_mean) {
            theta_mixed_sum += averaged_at(accumulator.theta_mean_sum, k, inv_count);
            ++theta_mixed_count;
        }
    }
    const double theta_mixed_mean = theta_mixed_count > 0 ? theta_mixed_sum / static_cast<double>(theta_mixed_count) : 0.0;

    {
        std::ofstream out(join_path(params.benchmark_output_dir, "summary.csv"));
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
        std::ofstream out(join_path(params.benchmark_output_dir, "profiles.csv"));
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
            const double heat_flux = averaged_at(accumulator.heat_flux_sum, k, inv_count);
            const double heat_flux_resolved = averaged_at(accumulator.heat_flux_resolved_sum, k, inv_count);
            const double heat_flux_sgs = averaged_at(accumulator.heat_flux_sgs_sum, k, inv_count);
            const double u_var = averaged_at(accumulator.u_var_sum, k, inv_count);
            const double v_var = averaged_at(accumulator.v_var_sum, k, inv_count);
            const double w_var = averaged_at(accumulator.w_var_sum, k, inv_count);
            const double theta_var = averaged_at(accumulator.theta_var_sum, k, inv_count);
            const double p_var = averaged_at(accumulator.p_var_sum, k, inv_count);
            const double w3 = averaged_at(accumulator.w3_sum, k, inv_count);
            const double skewness = w_var > 0.0 ? w3 / std::pow(w_var, 1.5) : 0.0;
            const double buoyancy = (params.g / params.theta0) * heat_flux / transport_norm;
            const double d_w_transport = gradient_at(accumulator.w_transport_sum, k, params, inv_count) / transport_norm;
            const double d_p_transport = gradient_at(accumulator.p_transport_sum, k, params, inv_count) / transport_norm;

            out << z << ',' << z / zi_mean << ',' << z / params.z_i << ','
                << averaged_at(accumulator.u_mean_sum, k, inv_count) << ','
                << averaged_at(accumulator.v_mean_sum, k, inv_count) << ','
                << averaged_at(accumulator.w_mean_sum, k, inv_count) << ','
                << averaged_at(accumulator.p_mean_sum, k, inv_count) << ','
                << averaged_at(accumulator.theta_mean_sum, k, inv_count) << ','
                << heat_flux / params.surface_theta_flux << ','
                << heat_flux_resolved / params.surface_theta_flux << ','
                << heat_flux_sgs / params.surface_theta_flux << ','
                << heat_flux / params.surface_theta_flux << ','
                << averaged_at(accumulator.epsilon_sum, k, inv_count) * zi_mean / (wstar * wstar * wstar) << ','
                << u_var / (wstar * wstar) << ','
                << v_var / (wstar * wstar) << ','
                << 0.5 * (u_var + v_var) / (wstar * wstar) << ','
                << w_var / (wstar * wstar) << ','
                << theta_var / (theta_star * theta_star) << ','
                << p_var / std::pow(wstar, 4.0) << ','
                << w3 / (wstar * wstar * wstar) << ','
                << skewness << ','
                << averaged_at(accumulator.alpha_u_sum, k, inv_count) << ','
                << averaged_at(accumulator.w_u_sum, k, inv_count) / wstar << ','
                << averaged_at(accumulator.theta_u_excess_sum, k, inv_count) / theta_star << ','
                << averaged_at(accumulator.cs2_mean_sum, k, inv_count) << ','
                << averaged_at(accumulator.scalar_c_mean_sum, k, inv_count) << ','
                << averaged_at(accumulator.kappa_mean_sum, k, inv_count) << ','
                << buoyancy << ',' << d_w_transport << ',' << d_p_transport << '\n';
        }
    }

    {
        std::ofstream out(join_path(params.benchmark_output_dir, "heat_flux_faces.csv"));
        if (!out) {
            throw std::runtime_error("failed to open benchmark face heat-flux output");
        }
        out << std::setprecision(17);
        out << "z,z_over_zi,z_over_zi0,heat_flux_over_qs,heat_flux_resolved_over_qs,heat_flux_sgs_over_qs\n";
        for (int k = 0; k <= params.nz; ++k) {
            const double z = static_cast<double>(k) * params.dz();
            const double heat_flux = averaged_at(accumulator.heat_flux_face_sum, k, inv_count);
            const double resolved = averaged_at(accumulator.heat_flux_face_resolved_sum, k, inv_count);
            const double sgs = averaged_at(accumulator.heat_flux_face_sgs_sum, k, inv_count);
            out << z << ',' << z / zi_mean << ',' << z / params.z_i << ','
                << heat_flux / params.surface_theta_flux << ','
                << resolved / params.surface_theta_flux << ','
                << sgs / params.surface_theta_flux << '\n';
        }
    }

    if (params.moisture_enabled) {
        {
            std::ofstream out(join_path(params.benchmark_output_dir, "moist_summary.csv"));
            if (!out) {
                throw std::runtime_error("failed to open moisture summary output");
            }
            const double water_error = accumulator.final_column_water - accumulator.expected_column_water;
            const double relative_error = water_error
                / std::max(std::abs(accumulator.expected_column_water), 1.0e-30);
            out << std::setprecision(17);
            out << "quantity,value\n";
            out << "initial_column_water," << accumulator.initial_column_water << '\n';
            out << "final_column_water," << accumulator.final_column_water << '\n';
            out << "expected_column_water," << accumulator.expected_column_water << '\n';
            out << "column_water_error," << water_error << '\n';
            out << "relative_column_water_error," << relative_error << '\n';
            out << "max_abs_column_water_error," << accumulator.max_abs_column_water_error << '\n';
            out << "qv_min," << accumulator.qv_min << '\n';
            out << "qv_max," << accumulator.qv_max << '\n';
            out << "qt_min," << accumulator.qt_min << '\n';
            out << "qt_max," << accumulator.qt_max << '\n';
            out << "ql_max," << accumulator.ql_max << '\n';
            out << "surface_qv_flux," << params.surface_qv_flux << '\n';
            out << "surface_qt_flux," << params.surface_qv_flux << '\n';
            out << "surface_pressure," << params.surface_pressure << '\n';
            out << "schmidt_t," << params.schmidt_t << '\n';
            out << "limiter_activations," << accumulator.moisture_limiter_activations << '\n';
            out << "limiter_column_correction," << accumulator.moisture_limiter_column_correction << '\n';
        }

        {
            std::ofstream out(join_path(params.benchmark_output_dir, "moist_profiles.csv"));
            if (!out) {
                throw std::runtime_error("failed to open moisture profile output");
            }
            const double theta_v0 = params.theta0 * (1.0 + 0.61 * params.qv0);
            out << std::setprecision(17);
            out << "z,z_over_zi,z_over_zi0,base_pressure,qt_mean,qv_mean,ql_mean,qt_variance,qv_variance,ql_variance,"
                << "theta_v_mean,theta_v_variance,"
                << "qt_flux_total,qt_flux_resolved,qt_flux_sgs,"
                << "theta_v_flux_total,theta_v_flux_resolved,theta_v_flux_sgs,"
                << "buoyancy_flux,moisture_kappa_mean\n";
            for (int k = 0; k < params.nz; ++k) {
                const double z = (static_cast<double>(k) + 0.5) * params.dz();
                const double virtual_flux = averaged_at(accumulator.virtual_heat_flux_sum, k, inv_count);
                out << z << ',' << z / zi_mean << ',' << z / params.z_i << ','
                    << hydrostatic_base_pressure(z, params.surface_pressure, params.theta0, params.g) << ','
                    << averaged_at(accumulator.qt_mean_sum, k, inv_count) << ','
                    << averaged_at(accumulator.qv_mean_sum, k, inv_count) << ','
                    << averaged_at(accumulator.ql_mean_sum, k, inv_count) << ','
                    << averaged_at(accumulator.qt_var_sum, k, inv_count) << ','
                    << averaged_at(accumulator.qv_var_sum, k, inv_count) << ','
                    << averaged_at(accumulator.ql_var_sum, k, inv_count) << ','
                    << averaged_at(accumulator.theta_v_mean_sum, k, inv_count) << ','
                    << averaged_at(accumulator.theta_v_var_sum, k, inv_count) << ','
                    << averaged_at(accumulator.moisture_flux_sum, k, inv_count) << ','
                    << averaged_at(accumulator.moisture_flux_resolved_sum, k, inv_count) << ','
                    << averaged_at(accumulator.moisture_flux_sgs_sum, k, inv_count) << ','
                    << virtual_flux << ','
                    << averaged_at(accumulator.virtual_heat_flux_resolved_sum, k, inv_count) << ','
                    << averaged_at(accumulator.virtual_heat_flux_sgs_sum, k, inv_count) << ','
                    << (params.g / theta_v0) * virtual_flux << ','
                    << averaged_at(accumulator.moisture_kappa_mean_sum, k, inv_count) << '\n';
            }
        }

        {
            std::ofstream out(join_path(params.benchmark_output_dir, "moist_flux_faces.csv"));
            if (!out) {
                throw std::runtime_error("failed to open moisture face-flux output");
            }
            out << std::setprecision(17);
            out << "z,z_over_zi,z_over_zi0,qt_flux_total,qt_flux_resolved,qt_flux_sgs,"
                << "theta_v_flux_total,theta_v_flux_resolved,theta_v_flux_sgs\n";
            for (int k = 0; k <= params.nz; ++k) {
                const double z = static_cast<double>(k) * params.dz();
                out << z << ',' << z / zi_mean << ',' << z / params.z_i << ','
                    << averaged_at(accumulator.moisture_flux_face_sum, k, inv_count) << ','
                    << averaged_at(accumulator.moisture_flux_face_resolved_sum, k, inv_count) << ','
                    << averaged_at(accumulator.moisture_flux_face_sgs_sum, k, inv_count) << ','
                    << averaged_at(accumulator.virtual_heat_flux_face_sum, k, inv_count) << ','
                    << averaged_at(accumulator.virtual_heat_flux_face_resolved_sum, k, inv_count) << ','
                    << averaged_at(accumulator.virtual_heat_flux_face_sgs_sum, k, inv_count) << '\n';
            }
        }
    }

    std::cout << "[benchmark] wrote profile diagnostics to " << params.benchmark_output_dir << '\n';
}
#endif

}  // namespace
}  // namespace wireles

int main(int argc, char** argv) {
    try {
        wireles::Params params = wireles::parse_args(argc, argv);
#ifndef WIRELES_HAVE_CPU
        if (!params.cuda_enabled || !params.mpi_slab) {
            throw std::runtime_error("this build was configured with WIRELES_ENABLE_CPU=OFF and supports only --cuda --mpi-slab");
        }
#endif
        if (params.cuda_enabled) {
#ifdef WIRELES_HAVE_CUDA
#ifdef WIRELES_HAVE_CPU
            if (!wireles::cuda_available()) {
                throw std::runtime_error("CUDA was requested, but no CUDA device is available");
            }
#endif
#else
            throw std::runtime_error("CUDA was requested, but this build was configured without WIRELES_ENABLE_CUDA=ON");
#endif
#ifdef WIRELES_HAVE_CPU
            if (!params.mpi_slab && params.sgs_model != "none" && params.sgs_model != "smagorinsky") {
                throw std::runtime_error("fully device-resident CUDA currently supports only --sgs none or --sgs smagorinsky");
            }
            if (!params.mpi_slab && params.thermo_enabled) {
                throw std::runtime_error("fully device-resident CUDA currently does not support --thermo");
            }
            if (!params.mpi_slab && params.momentum_wall_model != "none") {
                throw std::runtime_error("fully device-resident CUDA currently supports only --wall none");
            }
            if (!params.mpi_slab && params.horizontal_dealias) {
                throw std::runtime_error("fully device-resident CUDA currently does not support horizontal dealiasing");
            }
            if (params.dealiasing == "padding_3_2") {
                throw std::runtime_error("padding_3_2 dealiasing is currently implemented only for CPU paths");
            }
#endif
        }
        if (params.mpi_slab) {
#ifdef WIRELES_HAVE_MPI
            if (params.cuda_enabled) {
#ifdef WIRELES_HAVE_CUDA
                return wireles::run_cuda_mpi_slab(params, argc, argv);
#else
                throw std::runtime_error("CUDA-MPI slab mode requires WIRELES_ENABLE_CUDA=ON");
#endif
            }
#ifdef WIRELES_HAVE_CPU
            return wireles::run_mpi_slab(params, argc, argv);
#else
            throw std::runtime_error("CPU MPI slab mode is unavailable because this build used WIRELES_ENABLE_CPU=OFF");
#endif
#else
            throw std::runtime_error("this build was not compiled with MPI support");
#endif
        }
#ifndef WIRELES_HAVE_CPU
        if (params.frame_dump_enabled) {
            throw std::runtime_error("transient field output is MPI-HDF5 only; run with --cuda --mpi-slab and build with -DWIRELES_ENABLE_HDF5=ON");
        }
#endif
#ifndef WIRELES_HAVE_CPU
        throw std::runtime_error("single-process CPU/CUDA paths are unavailable because this build used WIRELES_ENABLE_CPU=OFF");
#else
        if (params.frame_dump_enabled) {
            throw std::runtime_error("transient field output is MPI-HDF5 only; run with --cuda --mpi-slab and build with -DWIRELES_ENABLE_HDF5=ON");
        }
        wireles::FftwXY fft(params);
        wireles::FlowState state(params);
        wireles::TimestepWorkspace timestep_workspace(params);

        wireles::initialize(state, params);
        if (params.cuda_enabled) {
            wireles::cuda_upload_flow_state(*timestep_workspace.cuda, state, params);
            wireles::cuda_enforce_walls(timestep_workspace, params);
            wireles::cuda_project(state, timestep_workspace, params);
            wireles::cuda_download_flow_state(*timestep_workspace.cuda, state, params);
        } else {
            wireles::project(state, params, fft);
        }

        std::cout << "# wireles single-process C++ solver\n";
        std::cout << "# grid " << params.nx << "x" << params.ny << "x" << params.nz
                  << ", dt=" << params.dt << ", scheme=" << params.time_scheme
                  << ", nu=" << params.nu << '\n';
        std::cout << "# wall=" << params.momentum_wall_model
                  << ", sgs=" << params.sgs_model
                  << ", thermo=" << (params.thermo_enabled ? "on" : "off")
                  << ", moisture=" << (params.moisture_enabled ? "on" : "off")
                  << ", cuda=" << (params.cuda_enabled ? "on" : "off") << '\n';
        std::cout << "#  step        ke_max         div_max       cfl";
        if (params.moisture_enabled) {
            std::cout << "        qv_min         qv_max         ql_max    column_water    qt_diff";
        }
        std::cout << '\n';
        wireles::print_diagnostics(0, wireles::diagnostics(state, params, fft), params);

        wireles::BenchmarkAccumulator benchmark;
        wireles::BomexAccumulator bomex;
        wireles::initialize_moisture_budget(benchmark, state, params);
        const double wstar0 = wireles::convective_wstar(params, params.z_i);
        const double tstar0 = params.z_i / wstar0;
        const int average_start_step = static_cast<int>(std::ceil(params.benchmark_average_start_tstar * tstar0 / params.dt));
        const int average_end_step = std::min(
            params.steps,
            static_cast<int>(std::floor(params.benchmark_average_end_tstar * tstar0 / params.dt)));

        for (int step_number = 1; step_number <= params.steps; ++step_number) {
            wireles::step(state, params, fft, timestep_workspace);
            wireles::update_moisture_budget(benchmark, state, params);
            const bool needs_diagnostics = step_number % params.log_every == 0 || step_number == params.steps;
            const bool needs_benchmark = params.benchmark_enabled
                && (step_number % params.benchmark_sample_every == 0 || step_number == params.steps)
                && step_number >= average_start_step
                && step_number <= average_end_step;
            const bool needs_bomex = params.bomex_diagnostics_enabled
                && (step_number % params.bomex_sample_every == 0 || step_number == params.steps)
                && static_cast<double>(step_number) * params.dt >= params.bomex_average_start_seconds;
            if (params.cuda_enabled && (needs_diagnostics || needs_benchmark || needs_bomex)) {
                wireles::cuda_download_flow_state(*timestep_workspace.cuda, state, params);
            }
            if (needs_diagnostics) {
                const wireles::Diagnostics diag = wireles::diagnostics(state, params, fft);
                wireles::print_diagnostics(step_number, diag, params);
                if (!std::isfinite(diag.ke_max) || !std::isfinite(diag.div_max) || !std::isfinite(diag.cfl)) {
                    throw std::runtime_error("non-finite diagnostics encountered; stopping run");
                }
            }
            if (needs_benchmark) {
                wireles::add_benchmark_sample(benchmark, wireles::benchmark_snapshot(state, params, fft));
            }
            if (needs_bomex) {
                wireles::add_bomex_sample(bomex, state, params);
            }
        }
        wireles::print_benchmark_summary(benchmark, params);
        wireles::write_benchmark_outputs(benchmark, params);
        wireles::print_bomex_summary(bomex, params);
        wireles::write_bomex_outputs(bomex, state, params);
#endif
    } catch (const std::exception& exc) {
        std::cerr << "ERROR: " << exc.what() << '\n';
        return 1;
    }
}

#include "wireles/config.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>
#include <string>

namespace wireles {
namespace {

std::string trim(std::string value) {
    auto not_space = [](unsigned char c) { return !std::isspace(c); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

std::string lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

std::string strip_comment(const std::string& line) {
    bool in_quote = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        if (line[i] == '"') {
            in_quote = !in_quote;
        } else if (line[i] == '#' && !in_quote) {
            return line.substr(0, i);
        }
    }
    return line;
}

std::string parse_string_value(std::string value) {
    value = trim(value);
    if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
        return value.substr(1, value.size() - 2);
    }
    return value;
}

int parse_int_value(const std::string& value, const std::string& name) {
    try {
        return std::stoi(value);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid integer for " + name + ": " + value);
    }
}

double parse_double_value(const std::string& value, const std::string& name) {
    try {
        return std::stod(value);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid floating-point value for " + name + ": " + value);
    }
}

bool parse_bool_value(const std::string& value, const std::string& name) {
    const std::string normalized = lower(parse_string_value(value));
    if (normalized == "true" || normalized == "1") {
        return true;
    }
    if (normalized == "false" || normalized == "0") {
        return false;
    }
    throw std::runtime_error("invalid boolean for " + name + ": " + value);
}

void apply_value(Params& params, const std::string& section, const std::string& key, const std::string& value) {
    const std::string name = section.empty() ? key : section + "." + key;
    const std::string string_value = parse_string_value(value);

    if (section == "grid") {
        if (key == "nx") params.nx = parse_int_value(value, name);
        else if (key == "ny") params.ny = parse_int_value(value, name);
        else if (key == "nz") params.nz = parse_int_value(value, name);
        else if (key == "lx") params.lx = parse_double_value(value, name);
        else if (key == "ly") params.ly = parse_double_value(value, name);
        else if (key == "lz") params.lz = parse_double_value(value, name);
        else if (key == "z_i") params.z_i = parse_double_value(value, name);
        return;
    }

    if (section == "time") {
        if (key == "scheme") params.time_scheme = lower(string_value);
        else if (key == "steps") params.steps = parse_int_value(value, name);
        else if (key == "log_every") params.log_every = parse_int_value(value, name);
        else if (key == "dt") params.dt = parse_double_value(value, name);
        return;
    }

    if (section == "physics") {
        if (key == "u_fric") params.u_fric = parse_double_value(value, name);
        else if (key == "zo") params.zo = parse_double_value(value, name);
        else if (key == "vonk") params.vonk = parse_double_value(value, name);
        else if (key == "nu") params.nu = parse_double_value(value, name);
        else if (key == "molecular_viscosity") params.nu = parse_double_value(value, name);
        else if (key == "initial_condition") params.initial_condition = lower(string_value);
        else if (key == "momentum_wall_model") params.momentum_wall_model = lower(string_value);
        else if (key == "wall_stress_model") params.wall_stress_model = lower(string_value);
        else if (key == "bomex_theta_perturbation") params.bomex_theta_perturbation = parse_double_value(value, name);
        else if (key == "bomex_qt_perturbation") params.bomex_qt_perturbation = parse_double_value(value, name);
        else if (key == "coriolis_f" || key == "f_coriolis") params.coriolis_f = parse_double_value(value, name);
        return;
    }

    if (section == "forcing") {
        if (key == "coriolis_f" || key == "f_coriolis") params.coriolis_f = parse_double_value(value, name);
        else if (key == "geostrophic_u" || key == "u_geostrophic") {
            params.geostrophic_u = parse_double_value(value, name);
        } else if (key == "geostrophic_v" || key == "v_geostrophic") {
            params.geostrophic_v = parse_double_value(value, name);
        }
        return;
    }

    if (section == "wall_filter") {
        if (key == "fgr") params.fgr = parse_double_value(value, name);
        else if (key == "tfr") params.tfr = parse_double_value(value, name);
        return;
    }

    if (section == "sgs") {
        if (key == "model") params.sgs_model = lower(string_value);
        else if (key == "cs_count") params.cs_count = parse_int_value(value, name);
        else if (key == "smag_cs") params.smagorinsky_cs = parse_double_value(value, name);
        else if (key == "smagorinsky_cs") params.smagorinsky_cs = parse_double_value(value, name);
        else if (key == "buoyancy_correction" || key == "smagorinsky_buoyancy_correction") {
            params.smagorinsky_buoyancy_correction = parse_bool_value(value, name);
        }
        else if (key == "min_shear_fraction" || key == "smagorinsky_min_shear_fraction") {
            params.smagorinsky_min_shear_fraction = parse_double_value(value, name);
        }
        else if (key == "sgs_delta_scale") params.sgs_delta_scale = parse_double_value(value, name);
        else if (key == "amd_buoyancy_correction") {
            params.amd_buoyancy_correction = parse_bool_value(value, name);
        }
        else if (key == "amd_invariant_averaging" || key == "invariant_averaging") {
            params.amd_invariant_averaging = parse_bool_value(value, name);
        }
        else if (key == "amd_dissipation_averaging" || key == "dissipation_averaging") {
            params.amd_dissipation_averaging = parse_bool_value(value, name);
        }
        else if (key == "amd_multiscale_averaging" || key == "multiscale_averaging") {
            params.amd_multiscale_averaging = parse_bool_value(value, name);
        }
        else if (key == "amd_wall_model_gradients" || key == "wall_model_gradients") {
            params.amd_wall_model_gradients = parse_bool_value(value, name);
        }
        else if (key == "amd_dealiased_cell_width" || key == "dealiased_cell_width") {
            params.amd_dealiased_cell_width = parse_bool_value(value, name);
        }
        else if (key == "tke_ck") params.tke_ck = parse_double_value(value, name);
        else if (key == "tke_length_coefficient") params.tke_length_coefficient = parse_double_value(value, name);
        else if (key == "tke_dissipation_base") params.tke_dissipation_base = parse_double_value(value, name);
        else if (key == "tke_dissipation_slope") params.tke_dissipation_slope = parse_double_value(value, name);
        else if (key == "tke_floor") params.tke_floor = parse_double_value(value, name);
        return;
    }

    if (section == "sponge") {
        if (key == "enabled") params.sponge_enabled = parse_bool_value(value, name);
        else if (key == "start_height" || key == "start_m") {
            params.sponge_start_height = parse_double_value(value, name);
        } else if (key == "timescale" || key == "timescale_seconds" || key == "tau") {
            params.sponge_timescale = parse_double_value(value, name);
        } else if (key == "power") {
            params.sponge_power = parse_double_value(value, name);
        }
        return;
    }

    if (section == "thermo") {
        if (key == "enabled") params.thermo_enabled = parse_bool_value(value, name);
        else if (key == "theta0") params.theta0 = parse_double_value(value, name);
        else if (key == "g") params.g = parse_double_value(value, name);
        else if (key == "theta_initial_gradient") params.theta_initial_gradient = parse_double_value(value, name);
        else if (key == "theta_top_gradient" && params.theta_initial_gradient == 0.0) {
            params.theta_initial_gradient = parse_double_value(value, name);
        } else if (key == "surface_theta_flux") params.surface_theta_flux = parse_double_value(value, name);
        else if (key == "scalar_diffusivity") params.scalar_diffusivity = parse_double_value(value, name);
        else if (key == "molecular_diffusivity") params.scalar_diffusivity = parse_double_value(value, name);
        else if (key == "largeeddy_initial_zi1_fraction") {
            params.largeeddy_initial_zi1_fraction = parse_double_value(value, name);
        } else if (key == "scalar_sgs_model") params.scalar_sgs_model = lower(string_value);
        else if (key == "scalar_amd_invariant_averaging" || key == "amd_invariant_averaging") {
            params.scalar_amd_invariant_averaging = parse_bool_value(value, name);
        }
        else if (key == "scalar_amd_face_products") {
            params.scalar_amd_face_products = parse_bool_value(value, name);
        }
        else if (key == "prandtl_t") params.prandtl_t = parse_double_value(value, name);
        else if (key == "schmidt_t") params.schmidt_t = parse_double_value(value, name);
        else if (key == "scalar_lasd_min") params.scalar_lasd_min = parse_double_value(value, name);
        else if (key == "scalar_lasd_max") params.scalar_lasd_max = parse_double_value(value, name);
        else if (key == "scalar_stability_correction") {
            params.scalar_stability_correction = parse_bool_value(value, name);
        } else if (key == "scalar_stability_beta") {
            params.scalar_stability_beta = parse_double_value(value, name);
        } else if (key == "scalar_stability_power") {
            params.scalar_stability_power = parse_double_value(value, name);
        }
        return;
    }

    if (section == "moisture") {
        if (key == "enabled") params.moisture_enabled = parse_bool_value(value, name);
        else if (key == "qt0" || key == "qv0") params.qv0 = parse_double_value(value, name);
        else if (key == "qt_initial_gradient" || key == "qv_initial_gradient") {
            params.qv_initial_gradient = parse_double_value(value, name);
        } else if (key == "surface_qt_flux" || key == "surface_qv_flux") {
            params.surface_qv_flux = parse_double_value(value, name);
        } else if (key == "molecular_diffusivity" || key == "moisture_diffusivity") {
            params.moisture_diffusivity = parse_double_value(value, name);
        } else if (key == "schmidt_t") params.schmidt_t = parse_double_value(value, name);
        else if (key == "surface_pressure") params.surface_pressure = parse_double_value(value, name);
        return;
    }

    if (section == "runtime") {
        if (key == "random_seed") params.random_seed = parse_int_value(value, name);
        else if (key == "cuda_enabled" || key == "cuda" || key == "gpu") {
            params.cuda_enabled = parse_bool_value(value, name);
        } else if (key == "mpi_slab" || key == "cuda_slab_backend") {
            params.mpi_slab = parse_bool_value(value, name);
        } else if (key == "initial_velocity_perturbation") {
            params.initial_velocity_perturbation = parse_double_value(value, name);
        } else if (key == "initial_perturbation_height") {
            params.initial_perturbation_height = parse_double_value(value, name);
        }
        return;
    }

    if (section == "benchmark" || section == "diagnostics") {
        if (key == "enabled") params.benchmark_enabled = parse_bool_value(value, name);
        else if (key == "sample_every") params.benchmark_sample_every = parse_int_value(value, name);
        else if (key == "average_start_tstar") {
            params.benchmark_average_start_tstar = parse_double_value(value, name);
        } else if (key == "average_end_tstar") {
            params.benchmark_average_end_tstar = parse_double_value(value, name);
        } else if (key == "output_dir") params.benchmark_output_dir = string_value;
        return;
    }

    if (section == "bomex") {
        if (key == "diagnostics_enabled" || key == "enabled") {
            params.bomex_diagnostics_enabled = parse_bool_value(value, name);
        } else if (key == "sample_every") params.bomex_sample_every = parse_int_value(value, name);
        else if (key == "average_start_seconds") {
            params.bomex_average_start_seconds = parse_double_value(value, name);
        } else if (key == "output_dir") params.bomex_output_dir = string_value;
        return;
    }

    if (section == "numerics") {
        if (key == "horizontal_dealias") params.horizontal_dealias = parse_bool_value(value, name);
        else if (key == "dealiasing" || key == "dealiasing_method") params.dealiasing = lower(string_value);
        else if (key == "momentum_advection_form") params.momentum_advection_form = lower(string_value);
        else if (key == "spectral_filter") params.spectral_filter = lower(string_value);
        else if (key == "spectral_filter_alpha") params.spectral_filter_alpha = parse_double_value(value, name);
        else if (key == "spectral_filter_order") params.spectral_filter_order = parse_int_value(value, name);
        return;
    }

    if (section == "postprocess") {
        if (key == "dump_frames" || key == "frame_dump_enabled") {
            params.frame_dump_enabled = parse_bool_value(value, name);
        } else if (key == "frame_every" || key == "frame_dump_every") {
            params.frame_dump_every = parse_int_value(value, name);
        } else if (key == "frame_start_step" || key == "frame_dump_start_step") {
            params.frame_dump_start_step = parse_int_value(value, name);
        } else if (key == "frame_end_step" || key == "frame_dump_end_step") {
            params.frame_dump_end_step = parse_int_value(value, name);
        } else if (key == "frame_y_index" || key == "frame_dump_y_index") {
            params.frame_dump_y_index = parse_int_value(value, name);
        } else if (key == "frame_z_height" || key == "frame_dump_z_height") {
            params.frame_dump_z_height = parse_double_value(value, name);
        } else if (key == "frame_slice_only" || key == "frame_dump_slice_only" || key == "slice_only") {
            params.frame_dump_slice_only = parse_bool_value(value, name);
        } else if (key == "frame_output_dir" || key == "frame_dump_output_dir") {
            params.frame_dump_output_dir = string_value;
        } else if (key == "frame_component" || key == "frame_dump_component") {
            params.frame_dump_component = lower(string_value);
        }
        return;
    }

    if (section == "profiling") {
        if (key == "enabled" || key == "mpi_enabled") {
            params.mpi_profile_enabled = parse_bool_value(value, name);
        } else if (key == "warmup_steps" || key == "mpi_warmup_steps") {
            params.mpi_profile_warmup_steps = parse_int_value(value, name);
        }
        return;
    }
}

}  // namespace

void apply_config_file(Params& params, const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open config file: " + path);
    }

    std::string section;
    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        line = trim(strip_comment(line));
        if (line.empty()) {
            continue;
        }
        if (line.front() == '[' && line.back() == ']') {
            section = lower(trim(line.substr(1, line.size() - 2)));
            continue;
        }
        const std::size_t equals = line.find('=');
        if (equals == std::string::npos) {
            throw std::runtime_error(path + ":" + std::to_string(line_number) + ": expected key = value");
        }
        const std::string key = lower(trim(line.substr(0, equals)));
        const std::string value = trim(line.substr(equals + 1));
        apply_value(params, section, key, value);
    }
}

}  // namespace wireles

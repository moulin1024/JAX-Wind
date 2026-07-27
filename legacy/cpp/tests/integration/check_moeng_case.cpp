#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    std::istringstream input(line);
    while (std::getline(input, field, ',')) {
        fields.push_back(field);
    }
    if (!line.empty() && line.back() == ',') {
        fields.emplace_back();
    }
    return fields;
}

double parse_double(const std::string& value, const std::string& name) {
    char* end = nullptr;
    const double parsed = std::strtod(value.c_str(), &end);
    if (end == value.c_str() || (end != nullptr && *end != '\0') || !std::isfinite(parsed)) {
        throw std::runtime_error("invalid finite number for " + name + ": " + value);
    }
    return parsed;
}

std::string join_path(const std::string& directory, const std::string& file) {
    if (directory.empty()) {
        return file;
    }
    return directory.back() == '/' ? directory + file : directory + "/" + file;
}

std::map<std::string, double> read_summary(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open " + path);
    }
    std::string line;
    if (!std::getline(input, line) || line != "quantity,value") {
        throw std::runtime_error("summary.csv has an unexpected header");
    }
    std::map<std::string, double> values;
    int row = 1;
    while (std::getline(input, line)) {
        ++row;
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> fields = split_csv_line(line);
        if (fields.size() != 2) {
            throw std::runtime_error("summary.csv row " + std::to_string(row) + " must have two columns");
        }
        values[fields[0]] = parse_double(fields[1], "summary." + fields[0]);
    }
    return values;
}

struct Table {
    std::vector<std::string> header;
    std::vector<std::vector<double>> rows;
};

Table read_table(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open " + path);
    }
    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error(path + " is empty");
    }
    Table table;
    table.header = split_csv_line(line);
    if (table.header.empty()) {
        throw std::runtime_error(path + " has an empty header");
    }
    int row_number = 1;
    while (std::getline(input, line)) {
        ++row_number;
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> fields = split_csv_line(line);
        if (fields.size() != table.header.size()) {
            throw std::runtime_error(path + " row " + std::to_string(row_number) + " has the wrong column count");
        }
        std::vector<double> row;
        row.reserve(fields.size());
        for (std::size_t i = 0; i < fields.size(); ++i) {
            row.push_back(parse_double(fields[i], path + "." + table.header[i]));
        }
        table.rows.push_back(std::move(row));
    }
    return table;
}

std::size_t column_index(const Table& table, const std::string& name) {
    for (std::size_t i = 0; i < table.header.size(); ++i) {
        if (table.header[i] == name) {
            return i;
        }
    }
    throw std::runtime_error("missing column: " + name);
}

void require_summary_keys(const std::map<std::string, double>& summary) {
    const std::set<std::string> required = {
        "sample_count",
        "zi_mean",
        "zi_over_zi0",
        "wstar_mean",
        "wstar_over_wstar0",
        "theta_mixed_layer_mean",
        "surface_theta_flux",
        "entrainment_ratio",
        "entrainment_layer_z",
        "entrainment_layer_z_over_zi0",
        "z_i0",
        "wstar0",
        "theta0",
        "g",
    };
    for (const std::string& key : required) {
        if (summary.find(key) == summary.end()) {
            throw std::runtime_error("summary.csv missing quantity: " + key);
        }
    }
}

void check_monotone_z(const Table& table) {
    const std::size_t z_col = column_index(table, "z");
    double previous = -std::numeric_limits<double>::infinity();
    for (const auto& row : table.rows) {
        if (row[z_col] <= previous) {
            throw std::runtime_error("z coordinate is not strictly increasing");
        }
        previous = row[z_col];
    }
}

void check_profile_columns(const Table& profiles) {
    const std::set<std::string> required = {
        "z",
        "z_over_zi",
        "z_over_zi0",
        "theta_mean",
        "heat_flux_over_qs",
        "heat_flux_resolved_over_qs",
        "heat_flux_sgs_over_qs",
        "epsilon_zi_over_wstar3",
        "w_var_over_wstar_sq",
        "theta_var_over_thetastar_sq",
        "cs2_mean",
        "scalar_c_mean",
        "kappa_mean",
    };
    for (const std::string& key : required) {
        column_index(profiles, key);
    }
}

void check_face_columns(const Table& faces) {
    const std::set<std::string> required = {
        "z",
        "z_over_zi",
        "z_over_zi0",
        "heat_flux_over_qs",
        "heat_flux_resolved_over_qs",
        "heat_flux_sgs_over_qs",
    };
    for (const std::string& key : required) {
        column_index(faces, key);
    }
}

void require_near(
    const std::map<std::string, double>& summary,
    const std::string& key,
    double target,
    double tolerance) {
    const auto found = summary.find(key);
    if (found == summary.end()) {
        throw std::runtime_error("summary.csv missing quantity: " + key);
    }
    const double value = found->second;
    if (std::abs(value - target) > tolerance) {
        std::ostringstream message;
        message << key << "=" << value << " outside target " << target << " +/- " << tolerance;
        throw std::runtime_error(message.str());
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            throw std::runtime_error("usage: check_moeng_case OUTPUT_DIR EXPECTED_NZ");
        }
        const std::string output_dir = argv[1];
        const int expected_nz = std::atoi(argv[2]);
        if (expected_nz <= 0) {
            throw std::runtime_error("EXPECTED_NZ must be positive");
        }

        const std::map<std::string, double> summary = read_summary(join_path(output_dir, "summary.csv"));
        require_summary_keys(summary);

        const double sample_count = summary.at("sample_count");
        if (sample_count < 2.0) {
            throw std::runtime_error("Moeng integration must collect at least two benchmark samples");
        }
        if (summary.at("surface_theta_flux") <= 0.0) {
            throw std::runtime_error("surface heat flux must be positive for Moeng CBL integration");
        }
        if (summary.at("z_i0") <= 0.0 || summary.at("theta0") <= 0.0 || summary.at("g") <= 0.0) {
            throw std::runtime_error("reference thermodynamic constants must be positive");
        }
        if (summary.at("zi_over_zi0") <= 0.0 || summary.at("zi_over_zi0") > 2.0) {
            throw std::runtime_error("zi_over_zi0 is outside a broad physical sanity range");
        }
        if (summary.at("wstar_mean") <= 0.0 || summary.at("wstar0") <= 0.0) {
            throw std::runtime_error("convective velocity scale must be positive");
        }
        if (std::abs(summary.at("entrainment_ratio")) > 10.0) {
            throw std::runtime_error("entrainment_ratio is outside a broad numerical sanity range");
        }
        require_near(summary, "zi_over_zi0", 1.0312, 0.02);
        require_near(summary, "wstar_over_wstar0", 1.010, 0.015);
        require_near(summary, "entrainment_ratio", 0.106, 0.03);

        const Table profiles = read_table(join_path(output_dir, "profiles.csv"));
        check_profile_columns(profiles);
        if (profiles.rows.size() != static_cast<std::size_t>(expected_nz)) {
            throw std::runtime_error("profiles.csv row count does not match expected nz");
        }
        check_monotone_z(profiles);

        const Table faces = read_table(join_path(output_dir, "heat_flux_faces.csv"));
        check_face_columns(faces);
        if (faces.rows.size() != static_cast<std::size_t>(expected_nz + 1)) {
            throw std::runtime_error("heat_flux_faces.csv row count does not match expected nz + 1");
        }
        check_monotone_z(faces);

        const std::size_t face_total_flux = column_index(faces, "heat_flux_over_qs");
        const std::size_t face_sgs_flux = column_index(faces, "heat_flux_sgs_over_qs");
        if (std::abs(faces.rows.front()[face_total_flux] - 1.0) > 1.0e-10
            || std::abs(faces.rows.front()[face_sgs_flux] - 1.0) > 1.0e-10) {
            throw std::runtime_error("bottom face heat flux is not equal to the prescribed surface flux");
        }

        const std::size_t cs2_col = column_index(profiles, "cs2_mean");
        const std::size_t scalar_c_col = column_index(profiles, "scalar_c_mean");
        const std::size_t kappa_col = column_index(profiles, "kappa_mean");
        for (const auto& row : profiles.rows) {
            if (row[cs2_col] < -1.0e-14 || row[scalar_c_col] < -1.0e-14 || row[kappa_col] < -1.0e-14) {
                throw std::runtime_error("dynamic SGS/scalar coefficients must be non-negative within roundoff");
            }
        }

        std::cout << "Moeng integration diagnostics validated in " << output_dir << '\n';
    } catch (const std::exception& exc) {
        std::cerr << "ERROR: " << exc.what() << '\n';
        return 1;
    }
    return 0;
}

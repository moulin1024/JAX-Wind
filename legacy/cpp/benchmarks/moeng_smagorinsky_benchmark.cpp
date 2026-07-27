#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef WIRELES_SOURCE_DIR
#define WIRELES_SOURCE_DIR "."
#endif

#ifndef WIRELES_BINARY_DIR
#define WIRELES_BINARY_DIR "."
#endif

#ifndef WIRELES_MPIEXEC_EXECUTABLE
#define WIRELES_MPIEXEC_EXECUTABLE "mpiexec"
#endif

#ifndef WIRELES_MPIEXEC_NUMPROC_FLAG
#define WIRELES_MPIEXEC_NUMPROC_FLAG "-n"
#endif

namespace {

std::string dirname_of(const std::string& path) {
    const std::size_t slash = path.find_last_of("/\\");
    return slash == std::string::npos ? "." : path.substr(0, slash);
}

std::string join_path(const std::string& directory, const std::string& name) {
    if (directory.empty()) {
        return name;
    }
    const char last = directory.back();
    return (last == '/' || last == '\\') ? directory + name : directory + "/" + name;
}

std::string shell_quote(const std::string& value) {
    std::string out = "'";
    for (const char c : value) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }
    out += "'";
    return out;
}

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    std::istringstream input(line);
    while (std::getline(input, field, ',')) {
        fields.push_back(field);
    }
    return fields;
}

std::map<std::string, std::string> read_summary(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open benchmark summary: " + path);
    }
    std::string line;
    if (!std::getline(input, line) || line != "quantity,value") {
        throw std::runtime_error("unexpected summary.csv header");
    }
    std::map<std::string, std::string> values;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> fields = split_csv_line(line);
        if (fields.size() == 2) {
            values[fields[0]] = fields[1];
        }
    }
    return values;
}

void print_usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " [options]\n\n"
        << "Runs the Moeng/Nieuwstadt dry CBL benchmark with fixed Smagorinsky SGS.\n\n"
        << "Options:\n"
        << "  --output-dir DIR       Benchmark CSV directory\n"
        << "  --config FILE          Override case config\n"
        << "  --wireles PATH         Solver executable, default is sibling ./wireles\n"
        << "  --steps N              Override step count\n"
        << "  --sample-every N       Override benchmark sampling interval\n"
        << "  --average-start T      Override average window start in t*\n"
        << "  --average-end T        Override average window end in t*\n"
        << "  --mpi-ranks N          Run distributed MPI z-slab benchmark\n"
        << "  --mpi-profile          Print MPI slab timing profile\n"
        << "  --keep-frames          Keep frame dump settings from config\n"
        << "  --help                 Show this help\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string output_dir = join_path(WIRELES_BINARY_DIR, "benchmark_outputs/moeng_smagorinsky");
        std::string config = join_path(WIRELES_SOURCE_DIR, "configs/largeeddy1993_moeng.toml");
        std::string solver = join_path(dirname_of(argv[0]), "wireles");
        std::string steps;
        std::string sample_every;
        std::string average_start;
        std::string average_end;
        std::string mpi_ranks;
        bool keep_frames = false;
        bool mpi_profile = false;

        for (int arg = 1; arg < argc; ++arg) {
            const std::string key = argv[arg];
            auto require_value = [&](const std::string& option) -> std::string {
                if (arg + 1 >= argc) {
                    throw std::runtime_error("missing value after " + option);
                }
                return argv[++arg];
            };
            if (key == "--help" || key == "-h") {
                print_usage(argv[0]);
                return 0;
            } else if (key == "--output-dir") {
                output_dir = require_value(key);
            } else if (key == "--config") {
                config = require_value(key);
            } else if (key == "--wireles") {
                solver = require_value(key);
            } else if (key == "--steps") {
                steps = require_value(key);
            } else if (key == "--sample-every") {
                sample_every = require_value(key);
            } else if (key == "--average-start") {
                average_start = require_value(key);
            } else if (key == "--average-end") {
                average_end = require_value(key);
            } else if (key == "--mpi-ranks") {
                mpi_ranks = require_value(key);
            } else if (key == "--mpi-profile" || key == "--profile") {
                mpi_profile = true;
            } else if (key == "--keep-frames") {
                keep_frames = true;
            } else {
                throw std::runtime_error("unknown option: " + key);
            }
        }

        std::ostringstream command;
        if (!mpi_ranks.empty()) {
            command << shell_quote(WIRELES_MPIEXEC_EXECUTABLE)
                    << " " << shell_quote(WIRELES_MPIEXEC_NUMPROC_FLAG)
                    << " " << shell_quote(mpi_ranks)
                    << " ";
        }
        command << shell_quote(solver)
                << " --config " << shell_quote(config)
                << " --sgs smagorinsky"
                << " --scalar-sgs fixed_prandtl"
                << " --benchmark-output-dir " << shell_quote(output_dir);
        if (!mpi_ranks.empty()) {
            command << " --mpi-slab";
        }
        if (mpi_profile) {
            command << " --mpi-profile";
        }
        if (!steps.empty()) {
            command << " --steps " << shell_quote(steps);
        }
        if (!sample_every.empty()) {
            command << " --benchmark-sample-every " << shell_quote(sample_every);
        }
        if (!average_start.empty()) {
            command << " --benchmark-average-start-tstar " << shell_quote(average_start);
        }
        if (!average_end.empty()) {
            command << " --benchmark-average-end-tstar " << shell_quote(average_end);
        }
        if (!keep_frames) {
            command << " --no-dump-frames";
        }

        std::cout << "[moeng-smagorinsky] running:\n  " << command.str() << '\n';
        const int status = std::system(command.str().c_str());
        if (status != 0) {
            return 1;
        }

        const auto summary = read_summary(join_path(output_dir, "summary.csv"));
        auto get = [&](const std::string& key) -> std::string {
            const auto found = summary.find(key);
            return found == summary.end() ? "missing" : found->second;
        };

        std::cout << "\n[moeng-smagorinsky] output: " << output_dir << '\n';
        std::cout << "  sample_count: " << get("sample_count") << '\n';
        std::cout << "  zi_over_zi0: " << get("zi_over_zi0") << '\n';
        std::cout << "  wstar_over_wstar0: " << get("wstar_over_wstar0") << '\n';
        std::cout << "  entrainment_ratio: " << get("entrainment_ratio") << '\n';
        std::cout << "  profiles: " << join_path(output_dir, "profiles.csv") << '\n';
        std::cout << "  heat_flux_faces: " << join_path(output_dir, "heat_flux_faces.csv") << '\n';
    } catch (const std::exception& exc) {
        std::cerr << "ERROR: " << exc.what() << '\n';
        return 1;
    }
    return 0;
}

#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: tools/package_rerun.sh [--with-tests] [OUTPUT.tar.gz]

Create a compact, self-contained source archive for rerunning JAX-Wind
benchmarks on another machine.  The default archive is written next to the
repository.  Tests are omitted unless --with-tests is supplied.

The archive intentionally excludes Git metadata, hidden files/directories,
caches, legacy sources, external submodules, benchmark_results, checkpoints,
logs, profiles, and generated benchmark overlay images.
EOF
}

with_tests=0
output_path=""
while (($#)); do
    case "$1" in
        --with-tests)
            with_tests=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -* )
            echo "error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
        *)
            if [[ -n "$output_path" ]]; then
                echo "error: only one output archive may be specified" >&2
                exit 2
            fi
            output_path=$1
            ;;
    esac
    shift
done

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd "$script_dir/.." && pwd)
project_name=$(basename "$project_root")
package_name="${project_name}-rerun"

if [[ -z "$output_path" ]]; then
    timestamp=$(date +%Y%m%d-%H%M%S)
    output_path="$(dirname "$project_root")/${package_name}-${timestamp}.tar.gz"
elif [[ "$output_path" != /* ]]; then
    output_path="$(pwd)/$output_path"
fi

output_dir=$(dirname "$output_path")
mkdir -p "$output_dir"
output_dir=$(cd "$output_dir" && pwd)
output_path="$output_dir/$(basename "$output_path")"

staging_parent=$(mktemp -d "${TMPDIR:-/tmp}/jaxwind-rerun.XXXXXX")
temporary_archive=""
cleanup() {
    if [[ -n "$temporary_archive" && -e "$temporary_archive" ]]; then
        rm -f "$temporary_archive"
    fi
    rm -rf "$staging_parent"
}
trap cleanup EXIT INT TERM

package_root="$staging_parent/$package_name"
mkdir -p "$package_root"

is_generated_path() {
    local relative=$1
    if [[ "$relative" == benchmark/* ]]; then
        local benchmark_parent=${relative%/*}
        local path_below_benchmark=${benchmark_parent#benchmark/}
        if [[ "$path_below_benchmark" != */* ]]; then
            case "$relative" in
                *.png|*.gif|*.npz|*.h5)
                    # Generated case-level overlays.  Nested reference/ and
                    # inflow/ assets remain packaged.
                    return 0
                    ;;
            esac
        fi
    fi
    case "$relative" in
        .*|*/.*)
            return 0
            ;;
        */__pycache__/*|*.pyc|*.pyo)
            return 0
            ;;
        benchmark_results/*|*/benchmark_results/*)
            return 0
            ;;
        */results/*|*/output/*|*/outputs/*|*/job/*|*/frames/*)
            return 0
            ;;
        */checkpoint.npz|*/checkpoint_*.npz|*.log|*.prof|*.trace|*.tmp|*.swp)
            return 0
            ;;
    esac
    return 1
}

copy_candidate() {
    local source=$1
    local relative=${source#"$project_root/"}
    if is_generated_path "$relative"; then
        return
    fi
    mkdir -p "$package_root/$(dirname "$relative")"
    cp -pP "$source" "$package_root/$relative"
}

copy_tree() {
    local relative_root=$1
    local source_root="$project_root/$relative_root"
    [[ -d "$source_root" ]] || return
    while IFS= read -r -d '' source; do
        copy_candidate "$source"
    done < <(
        find "$source_root" \
            \( -name '.*' -o -name '__pycache__' -o \
               -name '.pytest_cache' -o -name '.ruff_cache' -o \
               -name 'benchmark_results' -o -name 'results' -o \
               -name 'outputs' -o -name 'output' -o -name 'job' \) \
            -prune -o \( -type f -o -type l \) -print0
    )
}

for tree in src benchmark tools runners; do
    copy_tree "$tree"
done
if ((with_tests)); then
    copy_tree tests
fi

for metadata in pyproject.toml README.md LICENSE; do
    if [[ -f "$project_root/$metadata" ]]; then
        copy_candidate "$project_root/$metadata"
    fi
done

while IFS= read -r -d '' source; do
    copy_candidate "$source"
done < <(
    find "$project_root" -maxdepth 1 -type f \
        \( -name '*.py' -o -name '*.sh' \) -print0
)

cat > "$package_root/TRANSFER_README.txt" <<'EOF'
JAX-Wind minimal rerun package
==============================

This archive contains the current source tree, benchmark runners, reference
data, tools, and runner configurations.  Runtime results and Git metadata are
intentionally absent.

Suggested environment:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -e .
    python -m pip install numpy matplotlib

For MPI y-slab runs, also install mpi4py and an MPI implementation.

On the ROCm/DCU Slurm cluster, create an experiment directory and invoke the
bundled single-GPU 64-cubed job by absolute path.  The script submits itself;
Slurm logs and benchmark results are written below the experiment directory:

    mkdir -p /scratch/$USER/gabls1-run
    cd /scratch/$USER/gabls1-run
    /path/to/JAX-Wind-rerun/poisson_scaling.sh

Direct sbatch is also supported:

    sbatch /path/to/JAX-Wind-rerun/poisson_scaling.sh

Quick single-process GABLS1 verification:

    python benchmark/GABLS1/run.py \
        --quick \
        --projection-method fpj2 \
        --coupling-integrator coupled-ssprk3 \
        --output-dir benchmark_results/gabls1_quick

Current optimized 64-cubed GABLS1 configuration from a fresh initial state:

    python benchmark/GABLS1/run.py \
        --nx 64 --ny 64 --nz 64 \
        --end-hours 9 \
        --target-cfl 0.9 \
        --projection-method fpj2 \
        --coupling-integrator coupled-ssprk3 \
        --pressure-rtol 1e-5 \
        --pressure-smooth 1 \
        --output-dir benchmark_results/gabls1_amd_64cubed

The benchmark_results directory will be created by the runner.
EOF

(
    cd "$package_root"
    find . -type f -print | LC_ALL=C sort > PACKAGE_MANIFEST.txt
)

required_files="
src/jaxwind/pressure/matrix_free_gmg.py
src/jaxwind/pressure/distributed_gmg.py
benchmark/GABLS1/run.py
benchmark/GABLS1/run_mpi.py
benchmark/GABLS1/reference/official_12p5m/SOURCE.json
pyproject.toml
TRANSFER_README.txt
"
while IFS= read -r required; do
    [[ -z "$required" ]] && continue
    if [[ ! -e "$package_root/$required" ]]; then
        echo "error: required rerun file was not packaged: $required" >&2
        exit 1
    fi
done <<< "$required_files"

temporary_archive=$(mktemp "$output_dir/.${package_name}.XXXXXX")
rm -f "$temporary_archive"
(
    cd "$staging_parent"
    COPYFILE_DISABLE=1 tar -czf "$temporary_archive" "$package_name"
)

archive_listing="$staging_parent/archive.list"
tar -tzf "$temporary_archive" > "$archive_listing"
while IFS= read -r entry; do
    relative=${entry#"$package_name/"}
    if is_generated_path "$relative"; then
        echo "error: excluded path leaked into archive: $entry" >&2
        exit 1
    fi
done < "$archive_listing"

mv "$temporary_archive" "$output_path"
temporary_archive=""

file_count=$(wc -l < "$archive_listing" | tr -d ' ')
archive_size=$(du -h "$output_path" | awk '{print $1}')
echo "Created: $output_path"
echo "Archive size: $archive_size"
echo "Archive entries: $file_count"
if command -v shasum >/dev/null 2>&1; then
    echo "SHA-256: $(shasum -a 256 "$output_path" | awk '{print $1}')"
elif command -v sha256sum >/dev/null 2>&1; then
    echo "SHA-256: $(sha256sum "$output_path" | awk '{print $1}')"
fi

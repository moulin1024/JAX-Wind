#!/usr/bin/env bash
#SBATCH --job-name=gabls1-amd-gpu
#SBATCH --partition=hx1hdnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=dcu:1
# This cluster ties host memory to requested CPUs.  Eight CPU shares leave
# enough host memory for ROCm/XLA compilation while allocating exactly one DCU.
#SBATCH --time=12:00:00
#SBATCH --output=gabls1-amd-gpu-%j.out
#SBATCH --error=gabls1-amd-gpu-%j.err

# Single-GPU GABLS1 stable-boundary-layer run using the non-spectral MAC
# pressure solver and AMD SGS model.
#
# Submit from any output directory.  The first form records the repository
# location explicitly; direct sbatch is also supported on Slurm installations
# whose `scontrol show job` reports the original script path:
#   cd /scratch/my-gabls1-run
#   /path/to/JAX-Wind/poisson_scaling.sh
#   sbatch /path/to/JAX-Wind/poisson_scaling.sh
#
# Resume a separately transferred checkpoint into the same output directory:
#   RESTART=/path/to/checkpoint.npz RESULT_DIR=results \
#       /path/to/JAX-Wind/poisson_scaling.sh
#
# Common overrides:
#   NX=32 NY=32 NZ=32 /path/to/JAX-Wind/poisson_scaling.sh
#   END_HOURS=1 SAMPLE_START_HOURS=0 /path/to/JAX-Wind/poisson_scaling.sh
#   CONDA_ENV=my-rocm-jax-env /path/to/JAX-Wind/poisson_scaling.sh

set -euo pipefail

# Slurm normally executes a private spool copy of a batch script, so its
# BASH_SOURCE no longer points at the repository.  When this file is invoked
# directly, submit it once and export both the real repository path and the
# caller's current directory to the batch job.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    script_path="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)/$(basename -- "${BASH_SOURCE[0]}")"
    project_root="$(dirname -- "$script_path")"
    submit_dir="$(pwd -P)"
    sbatch \
        --export="ALL,GABLS1_PROJECT_ROOT=${project_root},GABLS1_SUBMIT_DIR=${submit_dir}" \
        "$script_path" "$@"
    exit $?
fi

SUBMIT_DIR="${GABLS1_SUBMIT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd -P)}}"
if [[ "$SUBMIT_DIR" != /* ]]; then
    SUBMIT_DIR="$(pwd -P)/$SUBMIT_DIR"
fi
if [[ ! -d "$SUBMIT_DIR" ]]; then
    echo "ERROR: submission/output directory not found: $SUBMIT_DIR" >&2
    exit 2
fi
SUBMIT_DIR="$(cd -- "$SUBMIT_DIR" && pwd -P)"

# PROJECT_ROOT may be supplied explicitly.  The direct-launch path above does
# that automatically.  For `sbatch /path/to/script`, recover Slurm's recorded
# original script path; finally retain compatibility with submission in repo.
PROJECT_ROOT="${PROJECT_ROOT:-${GABLS1_PROJECT_ROOT:-}}"
if [[ -z "$PROJECT_ROOT" ]]; then
    job_record="$(scontrol show job -o "$SLURM_JOB_ID" 2>/dev/null || true)"
    if [[ "$job_record" =~ (^|[[:space:]])Command=([^[:space:]]+) ]]; then
        submitted_script="${BASH_REMATCH[2]}"
        if [[ -f "$submitted_script" ]]; then
            PROJECT_ROOT="$(dirname -- "$submitted_script")"
        fi
    fi
fi
if [[ -z "$PROJECT_ROOT" && -f "$SUBMIT_DIR/benchmark/GABLS1/run.py" ]]; then
    PROJECT_ROOT="$SUBMIT_DIR"
fi
if [[ -z "$PROJECT_ROOT" ]]; then
    echo "ERROR: cannot locate the JAX-Wind repository." >&2
    echo "Submit with /path/to/JAX-Wind/poisson_scaling.sh, or set PROJECT_ROOT." >&2
    exit 2
fi
if [[ "$PROJECT_ROOT" != /* ]]; then
    PROJECT_ROOT="$SUBMIT_DIR/$PROJECT_ROOT"
fi
PROJECT_ROOT="$(cd -- "$PROJECT_ROOT" && pwd -P)"

BENCHMARK="${BENCHMARK:-${PROJECT_ROOT}/benchmark/GABLS1/run.py}"
if [[ "$BENCHMARK" != /* ]]; then
    BENCHMARK="$PROJECT_ROOT/$BENCHMARK"
fi
CONDA_ENV="${CONDA_ENV:-jax060}"
DTK_MODULE="${DTK_MODULE:-compiler/dtk/26.04}"
GCC_MODULE="${GCC_MODULE:-compiler/gcc/9.3.0}"
MPI_MODULE="${MPI_MODULE:-mpi/openmpi/openmpi-4.1.5-gcc9.3.0}"
DTK_ROOT="${DTK_ROOT:-/public/software/compiler/dtk-26.04}"

NX="${NX:-64}"
NY="${NY:-64}"
NZ="${NZ:-64}"
END_HOURS="${END_HOURS:-9}"
SAMPLE_START_HOURS="${SAMPLE_START_HOURS:-8}"
SAMPLE_INTERVAL_SECONDS="${SAMPLE_INTERVAL_SECONDS:-60}"
DT_MAX="${DT_MAX:-1}"
TARGET_CFL="${TARGET_CFL:-0.9}"
TARGET_DIFFUSIVE_CFL="${TARGET_DIFFUSIVE_CFL:-0.5}"
PRESSURE_RTOL="${PRESSURE_RTOL:-1e-5}"
PRESSURE_MAX_ITERATIONS="${PRESSURE_MAX_ITERATIONS:-40}"
PRESSURE_SMOOTH="${PRESSURE_SMOOTH:-1}"
PRESSURE_COARSE_SMOOTH="${PRESSURE_COARSE_SMOOTH:-20}"
AMD_COEFFICIENT="${AMD_COEFFICIENT:-0.212}"
SCALAR_AMD_COEFFICIENT="${SCALAR_AMD_COEFFICIENT:-0.212}"
ADVECTION_DISSIPATION_STRENGTH="${ADVECTION_DISSIPATION_STRENGTH:-${MP5_STRENGTH:-1}}"
ADVECTION_LIMITER="${ADVECTION_LIMITER:-mp5}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-1000}"
LOG_EVERY="${LOG_EVERY:-300}"
METRICS_EVERY="${METRICS_EVERY:-300}"
RESTART="${RESTART:-}"
RESULT_DIR="${RESULT_DIR:-${SUBMIT_DIR}/gabls1_amd_${NX}cubed_gpu_${SLURM_JOB_ID}}"
if [[ "$RESULT_DIR" != /* ]]; then
    RESULT_DIR="$SUBMIT_DIR/$RESULT_DIR"
fi
if [[ -n "$RESTART" && "$RESTART" != /* ]]; then
    RESTART="$SUBMIT_DIR/$RESTART"
fi

if [[ ! -f "$BENCHMARK" ]]; then
    echo "ERROR: GABLS1 runner not found: $BENCHMARK" >&2
    exit 2
fi
if [[ -n "$RESTART" && ! -f "$RESTART" ]]; then
    echo "ERROR: restart checkpoint not found: $RESTART" >&2
    exit 2
fi

module purge
module load "$DTK_MODULE"
module load "$GCC_MODULE"
module load "$MPI_MODULE"

if ! MPI_MPICC="$(command -v mpicc)"; then
    echo "ERROR: mpicc is unavailable after loading $MPI_MODULE" >&2
    exit 2
fi

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Use the module's compiler wrapper captured before Conda activation, even if
# the environment happens to contain another MPI implementation.
read -r -a openmpi_libdirs <<< "$("$MPI_MPICC" --showme:libdirs)"
OPENMPI_LIBRARY=""
for library_dir in "${openmpi_libdirs[@]}"; do
    if [[ -r "$library_dir/libmpi.so" ]]; then
        OPENMPI_LIBRARY="$library_dir/libmpi.so"
        break
    fi
done
if [[ -z "$OPENMPI_LIBRARY" ]]; then
    echo "ERROR: cannot find libmpi.so from: ${openmpi_libdirs[*]}" >&2
    exit 2
fi
if ! nm -D --defined-only "$OPENMPI_LIBRARY" 2>/dev/null \
    | grep -E '[[:space:]]ompi_mpi_int$' >/dev/null; then
    echo "ERROR: $OPENMPI_LIBRARY does not provide ompi_mpi_int" >&2
    exit 2
fi
OPENMPI_LIBRARY_PATH="$(IFS=:; echo "${openmpi_libdirs[*]}")"

# Keep the process working directory at the directory from which the job was
# submitted.  All source imports use absolute repository paths.
cd "$SUBMIT_DIR"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${OPENMPI_LIBRARY_PATH}:${DTK_ROOT}/lib:${DTK_ROOT}/dcc/gcvm/lib:${LD_LIBRARY_PATH:-}"
# DTK's libhipfftMp is loaded dynamically by the ROCm PJRT plugin.  Preloading
# the matching OpenMPI library makes its predefined datatype symbols (including
# ompi_mpi_int) globally visible during that dlopen.
JAX_LD_PRELOAD="${OPENMPI_LIBRARY}${LD_PRELOAD:+:${LD_PRELOAD}}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-rocm}"
export JAX_ENABLE_X64="0"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export PYTHONUNBUFFERED="1"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SLURM_TMPDIR:-/tmp}/jaxwind-matplotlib-${SLURM_JOB_ID:-manual}}"

mkdir -p "$RESULT_DIR" "$MPLCONFIGDIR"

run_args=(
    --nx "$NX"
    --ny "$NY"
    --nz "$NZ"
    --end-hours "$END_HOURS"
    --sample-start-hours "$SAMPLE_START_HOURS"
    --sample-interval-seconds "$SAMPLE_INTERVAL_SECONDS"
    --dt-max "$DT_MAX"
    --target-cfl "$TARGET_CFL"
    --target-diffusive-cfl "$TARGET_DIFFUSIVE_CFL"
    --amd-coefficient "$AMD_COEFFICIENT"
    --scalar-amd-coefficient "$SCALAR_AMD_COEFFICIENT"
    --advection-dissipation-strength "$ADVECTION_DISSIPATION_STRENGTH"
    --advection-limiter "$ADVECTION_LIMITER"
    --projection-method fpj2
    --coupling-integrator coupled-ssprk3
    --pressure-rtol "$PRESSURE_RTOL"
    --pressure-max-iterations "$PRESSURE_MAX_ITERATIONS"
    --pressure-smooth "$PRESSURE_SMOOTH"
    --pressure-coarse-smooth "$PRESSURE_COARSE_SMOOTH"
    --checkpoint-every "$CHECKPOINT_EVERY"
    --log-every "$LOG_EVERY"
    --metrics-every "$METRICS_EVERY"
    --output-dir "$RESULT_DIR"
)
if [[ -n "$RESTART" ]]; then
    run_args+=(--restart "$RESTART")
fi
run_args+=("$@")

step_flags=(
    --ntasks=1
    --cpus-per-task="${SLURM_CPUS_PER_TASK:-8}"
    --tres-per-task=gres/dcu:1
    --kill-on-bad-exit=1
)

echo "Job ID         : ${SLURM_JOB_ID:-manual}"
echo "Node           : $(hostname)"
echo "Submit dir     : $SUBMIT_DIR"
echo "Project root   : $PROJECT_ROOT"
echo "Runner         : $BENCHMARK"
echo "Python         : $(command -v python)"
echo "Conda env      : $CONDA_ENV"
echo "DTK module     : $DTK_MODULE"
echo "Compiler       : $GCC_MODULE"
echo "MPI module     : $MPI_MODULE"
echo "MPI compiler   : $MPI_MPICC"
echo "MPI version    : $("$MPI_MPICC" --showme:version)"
echo "MPI library    : $OPENMPI_LIBRARY"
echo "JAX preload    : $JAX_LD_PRELOAD"
echo "Grid           : ${NX}x${NY}x${NZ}"
echo "Integrator     : coupled-ssprk3 + FPJ2"
echo "Advection      : limiter=$ADVECTION_LIMITER strength=$ADVECTION_DISSIPATION_STRENGTH"
echo "CFL targets    : advective=$TARGET_CFL diffusive=$TARGET_DIFFUSIVE_CFL"
echo "Metrics cadence: every $METRICS_EVERY steps"
echo "Pressure       : rtol=$PRESSURE_RTOL smooth=$PRESSURE_SMOOTH coarse=$PRESSURE_COARSE_SMOOTH"
echo "Time window    : 0 to ${END_HOURS} h; samples from ${SAMPLE_START_HOURS} h"
echo "Restart        : ${RESTART:-fresh initial state}"
echo "Results        : $RESULT_DIR"
echo "Visible DCUs   : ${ROCR_VISIBLE_DEVICES:-assigned by Slurm}"
echo
echo "Loaded modules:"
module -t list 2>&1
echo

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    scontrol show job "$SLURM_JOB_ID" > "$RESULT_DIR/slurm_job.txt"
fi
(rocm-smi --showtopo || hy-smi --showtopo || echo "no SMI topology tool found") \
    > "$RESULT_DIR/topology.txt" 2>&1 || true

echo "===== GPU preflight: $(date --iso-8601=seconds) ====="
srun "${step_flags[@]}" env LD_PRELOAD="$JAX_LD_PRELOAD" python -c \
    'import jax; d=jax.devices(); print("backend:", jax.default_backend()); print("devices:", d); assert len(d) == 1, f"expected exactly one JAX device, got {len(d)}"'

echo "===== Starting GABLS1: $(date --iso-8601=seconds) ====="
srun "${step_flags[@]}" env LD_PRELOAD="$JAX_LD_PRELOAD" \
    python "$BENCHMARK" "${run_args[@]}" \
    2>&1 | tee "$RESULT_DIR/run.log"
echo "===== Finished GABLS1: $(date --iso-8601=seconds) ====="
echo "Results: $RESULT_DIR"

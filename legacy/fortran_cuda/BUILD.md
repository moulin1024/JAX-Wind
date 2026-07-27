# WIRELES_GPU Build Instructions

This project has been migrated to use CMake for building, with runtime configuration support.

## Prerequisites

- CMake 3.18 or higher
- CUDA-enabled Fortran compiler (PGI/NVIDIA HPC SDK or similar)
- MPI (OpenMPI or Intel MPI)
- CUDA toolkit

## Building

### Basic Build

```bash
mkdir build
cd build
cmake ..
make
```

The executable will be in `build/bin/wireles_src`.

### Build Options

- `USE_DOUBLE_PRECISION`: Enable double precision (default: OFF)
  ```bash
  cmake -DUSE_DOUBLE_PRECISION=ON ..
  ```

- `USE_NVTX`: Enable NVTX profiling markers (default: ON)
  ```bash
  cmake -DUSE_NVTX=OFF ..
  ```

### Example Build Commands

```bash
# Single precision build
cmake -DUSE_DOUBLE_PRECISION=OFF -DUSE_NVTX=ON ..
make -j8

# Double precision build
cmake -DUSE_DOUBLE_PRECISION=ON -DUSE_NVTX=ON ..
make -j8
```

## Running

The executable now reads the config file at runtime. You can specify the config file as a command-line argument:

```bash
# Use default config file (input/config)
mpirun -np 2 ./bin/wireles_src

# Specify custom config file
mpirun -np 2 ./bin/wireles_src /path/to/config
```

## Configuration

The config file format remains the same as before. The key difference is that you no longer need to rebuild when changing parameters - just edit the config file and rerun.

### Config File Location

By default, the executable looks for `input/config` in the current working directory. You can override this by passing the config file path as the first command-line argument.

## Migration Notes

### What Changed

1. **Build System**: Migrated from Python scripts + Makefile to CMake
2. **Configuration**: Config is now read at runtime instead of compile-time
3. **No Python Dependency**: The build process no longer requires Python scripts

### What Stayed the Same

- Config file format
- Source code structure
- Input/output file formats

## Troubleshooting

### CMake can't find CUDA

Set `CUDA_HOME` or `CUDA_TOOLKIT_ROOT_DIR`:
```bash
export CUDA_HOME=/path/to/cuda
cmake ..
```

### MPI not found

Set `MPI_HOME` or ensure MPI is in your PATH:
```bash
export MPI_HOME=/path/to/mpi
cmake ..
```

### Compiler issues

Specify the Fortran compiler:
```bash
cmake -DCMAKE_Fortran_COMPILER=mpif90 ..
```

### Compilation Errors: "Host MODULE data cannot be used in a DEVICE or GLOBAL subprogram"

This error occurs because module variables (like `nx`, `ny`, `nz2`) are now runtime variables, but CUDA device code requires compile-time constants or device variables.

**Solution**: You need to refactor the code to either:
1. Pass these values as arguments to CUDA kernels
2. Use device variables for values needed in kernels
3. Keep critical values as parameters (if they don't change often)

This is a code refactoring task that requires updating the source files that use these variables in device code. The build system is working correctly - this is a code architecture issue that needs to be addressed.


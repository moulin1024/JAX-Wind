cmake_minimum_required(VERSION 3.18)

# Set Fortran compiler to mpif90 (NVIDIA HPC SDK)
if(NOT DEFINED CMAKE_Fortran_COMPILER)
    set(CMAKE_Fortran_COMPILER mpif90)
endif()

project(WIRELES_GPU LANGUAGES C Fortran)

# Set Fortran standard
set(CMAKE_Fortran_STANDARD 2008)
set(CMAKE_Fortran_STANDARD_REQUIRED ON)

# Tell CMake that .cuf files are Fortran files (CUDA Fortran)
set(CMAKE_Fortran_SOURCE_FILE_EXTENSIONS f;F;f90;F90;f95;F95;f03;F03;f08;F08;cuf)

# Set output directories to build directory (keep source files in source directory)
# Object files (.o) and module files (.mod) go to build directory
set(CMAKE_Fortran_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_BINARY_DIR})

# Find MPI
find_package(MPI REQUIRED COMPONENTS Fortran)

# Find CUDA libraries without the deprecated FindCUDA module. CUDAToolkit_ROOT,
# CUDA_HOME, or a CUDA-enabled compiler/toolkit in PATH can be used to locate it.
find_package(CUDAToolkit REQUIRED)

# dimen.cuf is a compile-time template. Generate a build-local copy from the
# case config so constants such as nx, ny, nzb, and turbine parameters are real
# Fortran parameters before the compiler sees the file.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(WIRELES_CONFIG_FILE "" CACHE FILEPATH "WiRE-LES case config used to generate dimen.cuf")
if(NOT WIRELES_CONFIG_FILE)
    set(WIRELES_CONFIG_CANDIDATES
        "${CMAKE_BINARY_DIR}/../input/config"
        "${CMAKE_SOURCE_DIR}/config"
    )
    foreach(WIRELES_CONFIG_CANDIDATE IN LISTS WIRELES_CONFIG_CANDIDATES)
        if(EXISTS "${WIRELES_CONFIG_CANDIDATE}")
            set(WIRELES_CONFIG_FILE "${WIRELES_CONFIG_CANDIDATE}")
            break()
        endif()
    endforeach()
endif()
if(NOT WIRELES_CONFIG_FILE OR NOT EXISTS "${WIRELES_CONFIG_FILE}")
    message(FATAL_ERROR "No config file found for dimen.cuf generation. Set -DWIRELES_CONFIG_FILE=/path/to/input/config.")
endif()

set(WIRELES_DIMEN_TEMPLATE "${CMAKE_SOURCE_DIR}/src/dimen.cuf")
set(WIRELES_DIMEN_SOURCE "${CMAKE_BINARY_DIR}/generated/dimen.cuf")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${WIRELES_CONFIG_FILE}"
    "${WIRELES_DIMEN_TEMPLATE}"
)
execute_process(
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_SOURCE_DIR}/prc/generate_dimen.py"
            --template "${WIRELES_DIMEN_TEMPLATE}"
            --config "${WIRELES_CONFIG_FILE}"
            --output "${WIRELES_DIMEN_SOURCE}"
    RESULT_VARIABLE WIRELES_DIMEN_RESULT
    OUTPUT_VARIABLE WIRELES_DIMEN_STDOUT
    ERROR_VARIABLE WIRELES_DIMEN_STDERR
)
if(NOT WIRELES_DIMEN_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to generate dimen.cuf:\n${WIRELES_DIMEN_STDOUT}\n${WIRELES_DIMEN_STDERR}")
endif()
message(STATUS "Generated dimen.cuf from ${WIRELES_CONFIG_FILE}")

# Compiler flags
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -O3 -Mpreprocess")

# CUDA Fortran flags (updated to use new flags: -cuda -gpu=ccXX -acc instead of deprecated -Mcuda -ta=nvidia)
set(GPU_ARCH "86" CACHE STRING "GPU architecture, e.g. 80, sm_80, compute_80, or cc80")
string(REGEX REPLACE "^(sm_|compute_|cc)" "" GPU_ARCH_CC "${GPU_ARCH}")
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -cuda -gpu=cc${GPU_ARCH_CC} -acc")

# Optional: NVTX support
option(USE_NVTX "Enable NVTX profiling markers" ON)
if(USE_NVTX)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -DUSE_NVTX")
endif()

# Select the legacy turbine kernel at configure time.  The implementations
# export the same module name but not the same turbine_force signature.
set(WIRELES_TURBINE_MODEL "alm_nrel_new" CACHE STRING
    "Turbine implementation: alm_nrel_new or admnr")
set_property(CACHE WIRELES_TURBINE_MODEL PROPERTY STRINGS alm_nrel_new admnr)

# Optional: Double precision
option(USE_DOUBLE_PRECISION "Use double precision" OFF)
if(DEFINED PRECISION_DOUBLE)
    set(USE_DOUBLE_PRECISION "${PRECISION_DOUBLE}" CACHE BOOL "Use double precision" FORCE)
endif()
if(USE_DOUBLE_PRECISION)
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -DDOUBLE")
endif()

# Source files (matching makefile dependencies)
# Note: Root CMakeLists.txt builds from root, so paths include src/ prefix
# DEP0: precision, config_reader, nvtx (if USE_NVTX)
set(SOURCES_DEP0
    src/precision.cuf
    src/config_reader.cuf
)

# DEP1: dimen, cufft_m (depends on precision, config_reader)
set(SOURCES_DEP1
    ${WIRELES_DIMEN_SOURCE}
    src/cufft_m.cuf
)

# DEP2: update, convection, derivative, filter, io, pressure, rhs, time, control, structural, rotation
# (depends on precision, dimen)
set(SOURCES_DEP2
    src/update.cuf
    src/convection.cuf
    src/derivative.cuf
    src/filter.cuf
    src/io.cuf
    src/pressure.cuf
    src/rhs.cuf
    src/time.cuf
    src/control.cuf
    src/structural.cuf
    src/rotation.cuf
)

# DEP3: divergence and selected turbine implementation.
if(WIRELES_TURBINE_MODEL STREQUAL "admnr")
    set(WIRELES_TURBINE_SOURCE src/turbine_admnr.cuf)
    add_compile_definitions(WIRELES_TURBINE_ADMNR)
elseif(WIRELES_TURBINE_MODEL STREQUAL "alm_nrel_new")
    set(WIRELES_TURBINE_SOURCE src/turbine_alm_nrel_new.cuf)
else()
    message(FATAL_ERROR "Unsupported WIRELES_TURBINE_MODEL=${WIRELES_TURBINE_MODEL}")
endif()

set(SOURCES_DEP3
    src/divergence.cuf
    ${WIRELES_TURBINE_SOURCE}
)

# DEP4: sgs, sgs_mgm, wall (depends on precision, dimen, cufft_m, filter)
set(SOURCES_DEP4
    src/sgs.cuf
    src/sgs_mgm.cuf
    src/wall.cuf
)

# Add nvtx if USE_NVTX is enabled
if(USE_NVTX)
    list(APPEND SOURCES_DEP0 src/nvtx.cuf)
endif()

# Combine all sources
set(SOURCES
    ${SOURCES_DEP0}
    ${SOURCES_DEP1}
    ${SOURCES_DEP2}
    ${SOURCES_DEP3}
    ${SOURCES_DEP4}
    src/main.cuf
)

# Add executable
# .cuf files are treated as Fortran files with CUDA support
add_executable(wireles_src ${SOURCES})

# Explicitly set language for .cuf files and linker language
set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE Fortran)
set_target_properties(wireles_src PROPERTIES LINKER_LANGUAGE Fortran)

# Link MPI
target_link_libraries(wireles_src PUBLIC MPI::MPI_Fortran)

# Link CUDA toolkit libraries
target_link_libraries(wireles_src PUBLIC CUDA::cufft)
if(USE_NVTX)
    if(TARGET CUDA::nvToolsExt)
        target_link_libraries(wireles_src PUBLIC CUDA::nvToolsExt)
    elseif(TARGET CUDA::nvtx3_interop)
        target_link_libraries(wireles_src PUBLIC CUDA::nvtx3_interop)
    elseif(TARGET CUDA::nvtx3interop)
        target_link_libraries(wireles_src PUBLIC CUDA::nvtx3interop)
    else()
        find_library(NVTX_LIBRARY
            NAMES nvToolsExt nvtx3interop
            HINTS "${CUDAToolkit_LIBRARY_DIR}" "$ENV{CUDA_HOME}/lib64"
        )
        if(NVTX_LIBRARY)
            target_link_libraries(wireles_src PUBLIC "${NVTX_LIBRARY}")
        else()
            message(WARNING "USE_NVTX=ON, but no NVTX library target or library was found")
        endif()
    endif()
endif()

# Set output directory to build/bin directory
set_target_properties(wireles_src PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Install target
install(TARGETS wireles_src
    RUNTIME DESTINATION bin
)

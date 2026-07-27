cmake_minimum_required(VERSION 3.18)

project(wireles LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(WIRELES_ENABLE_CPU "Enable CPU/FFTW solver paths" ON)
option(WIRELES_ENABLE_MPI "Enable optional MPI z-slab execution path" ON)
option(WIRELES_ENABLE_CUDA "Enable optional CUDA acceleration for selected serial kernels" OFF)
option(WIRELES_ENABLE_HDF5 "Enable parallel HDF5 transient field output" OFF)

if(NOT WIRELES_ENABLE_CPU AND (NOT WIRELES_ENABLE_CUDA OR NOT WIRELES_ENABLE_MPI))
    message(FATAL_ERROR "WIRELES_ENABLE_CPU=OFF requires WIRELES_ENABLE_CUDA=ON and WIRELES_ENABLE_MPI=ON")
endif()

if(WIRELES_ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
endif()

if(WIRELES_ENABLE_CPU)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(FFTW3 QUIET IMPORTED_TARGET fftw3)
    endif()

    if(NOT TARGET PkgConfig::FFTW3)
        find_path(FFTW3_INCLUDE_DIR fftw3.h)
        find_library(FFTW3_LIBRARY NAMES fftw3)
        if(NOT FFTW3_INCLUDE_DIR OR NOT FFTW3_LIBRARY)
            message(FATAL_ERROR "FFTW3 was not found. Install fftw3 or provide FFTW3_INCLUDE_DIR/FFTW3_LIBRARY.")
        endif()
        add_library(FFTW3::fftw3 UNKNOWN IMPORTED)
        set_target_properties(FFTW3::fftw3 PROPERTIES
            IMPORTED_LOCATION "${FFTW3_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIR}"
        )
    endif()
endif()

if(WIRELES_ENABLE_MPI)
    find_package(MPI QUIET COMPONENTS CXX)
endif()

if(WIRELES_ENABLE_HDF5)
    if(NOT WIRELES_ENABLE_MPI)
        message(FATAL_ERROR "WIRELES_ENABLE_HDF5=ON requires WIRELES_ENABLE_MPI=ON")
    endif()
    enable_language(C)
    find_package(HDF5 REQUIRED COMPONENTS C)
    if(DEFINED HDF5_IS_PARALLEL AND NOT HDF5_IS_PARALLEL)
        message(FATAL_ERROR "WIRELES_ENABLE_HDF5=ON requires a parallel/MPI HDF5 build")
    endif()
endif()

if(NOT WIRELES_ENABLE_CPU AND NOT MPI_CXX_FOUND)
    message(FATAL_ERROR "WIRELES_ENABLE_CPU=OFF requires a working MPI CXX compiler/target")
endif()

set(WIRELES_MPI_CUDA_COMPILE_OPTIONS)
if(WIRELES_ENABLE_MPI AND MPI_CXX_FOUND AND MPI_CXX_COMPILER)
    execute_process(
        COMMAND "${MPI_CXX_COMPILER}" --showme:compile
        OUTPUT_VARIABLE WIRELES_MPI_WRAPPER_COMPILE_FLAGS
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(WIRELES_MPI_WRAPPER_COMPILE_FLAGS)
        separate_arguments(WIRELES_MPI_CUDA_COMPILE_OPTIONS UNIX_COMMAND "${WIRELES_MPI_WRAPPER_COMPILE_FLAGS}")
    endif()
endif()

add_executable(wireles
    src/bomex.cpp
    src/config.cpp
    src/field.cpp
    src/main.cpp
    src/cuda_mpi_slab.cpp
    src/thermodynamics.cpp
)

if(WIRELES_ENABLE_CPU)
    target_sources(wireles PRIVATE
        src/fft.cpp
        src/operators.cpp
        src/pressure.cpp
        src/scalar.cpp
        src/sgs.cpp
        src/cuda_solver.cpp
        src/timestep.cpp
        src/wall.cpp
    )
    target_compile_definitions(wireles PRIVATE WIRELES_HAVE_CPU=1)
endif()

if(WIRELES_ENABLE_CUDA)
    if(WIRELES_ENABLE_CPU)
        target_sources(wireles PRIVATE src/cuda_solver.cu)
    endif()
    target_compile_definitions(wireles PRIVATE WIRELES_HAVE_CUDA=1)
    target_link_libraries(wireles PRIVATE CUDA::cudart CUDA::cufft)
    set_target_properties(wireles PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

if(WIRELES_ENABLE_CPU AND WIRELES_ENABLE_MPI AND MPI_CXX_FOUND)
    target_sources(wireles PRIVATE src/mpi_slab.cpp)
    target_compile_definitions(wireles PRIVATE WIRELES_HAVE_MPI=1)
    target_link_libraries(wireles PRIVATE MPI::MPI_CXX)
endif()

if(WIRELES_ENABLE_CUDA AND WIRELES_ENABLE_MPI AND MPI_CXX_FOUND)
    target_sources(wireles PRIVATE src/cuda_mpi_slab.cu)
    target_compile_definitions(wireles PRIVATE WIRELES_HAVE_MPI=1)
    target_link_libraries(wireles PRIVATE MPI::MPI_CXX)
endif()

if(WIRELES_ENABLE_HDF5)
    target_compile_definitions(wireles PRIVATE WIRELES_HAVE_HDF5=1)
    target_include_directories(wireles PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(wireles PRIVATE ${HDF5_LIBRARIES})
endif()

target_include_directories(wireles PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(WIRELES_ENABLE_MPI AND MPI_CXX_FOUND)
    target_include_directories(wireles PRIVATE ${MPI_CXX_INCLUDE_DIRS})
    foreach(flag IN LISTS WIRELES_MPI_CUDA_COMPILE_OPTIONS)
        target_compile_options(wireles PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${flag}>)
    endforeach()
endif()

if(WIRELES_ENABLE_CPU)
    if(TARGET PkgConfig::FFTW3)
        target_link_libraries(wireles PRIVATE PkgConfig::FFTW3)
    else()
        target_link_libraries(wireles PRIVATE FFTW3::fftw3)
    endif()
endif()

target_compile_options(wireles PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic -O3>
)

if(WIRELES_ENABLE_CPU)
    add_executable(wireles_benchmark_moeng_smagorinsky
        benchmarks/moeng_smagorinsky_benchmark.cpp
    )
    target_compile_definitions(wireles_benchmark_moeng_smagorinsky PRIVATE
        WIRELES_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
        WIRELES_BINARY_DIR="${CMAKE_CURRENT_BINARY_DIR}"
    )
    if(MPIEXEC_EXECUTABLE)
        target_compile_definitions(wireles_benchmark_moeng_smagorinsky PRIVATE
            WIRELES_MPIEXEC_EXECUTABLE="${MPIEXEC_EXECUTABLE}"
            WIRELES_MPIEXEC_NUMPROC_FLAG="${MPIEXEC_NUMPROC_FLAG}"
        )
    endif()
    target_compile_options(wireles_benchmark_moeng_smagorinsky PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic -O2>
    )
endif()

include(CTest)

if(BUILD_TESTING AND WIRELES_ENABLE_CPU)
    add_executable(wireles_check_moeng_case
        tests/integration/check_moeng_case.cpp
    )
    target_compile_options(wireles_check_moeng_case PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic -O2>
    )

    set(WIRELES_MOENG_TEST_OUTPUT_DIR
        "${CMAKE_CURRENT_BINARY_DIR}/test_outputs/moeng_integration")
    set(WIRELES_MOENG_TEST_FRAME_DIR
        "${CMAKE_CURRENT_BINARY_DIR}/test_outputs/moeng_integration_frames")

    add_test(NAME wireles_moeng_case_run
        COMMAND wireles
            --config "${CMAKE_CURRENT_SOURCE_DIR}/configs/largeeddy1993_moeng_lasd.toml"
            --benchmark-output-dir "${WIRELES_MOENG_TEST_OUTPUT_DIR}"
            --frame-output-dir "${WIRELES_MOENG_TEST_FRAME_DIR}"
    )
    set_tests_properties(wireles_moeng_case_run PROPERTIES
        FIXTURES_SETUP wireles_moeng_case
        LABELS "integration;moeng;full"
        TIMEOUT 1800
    )

    add_test(NAME wireles_moeng_case_validate
        COMMAND wireles_check_moeng_case "${WIRELES_MOENG_TEST_OUTPUT_DIR}" 48
    )
    set_tests_properties(wireles_moeng_case_validate PROPERTIES
        FIXTURES_REQUIRED wireles_moeng_case
        LABELS "integration;moeng;full"
        TIMEOUT 30
    )

    add_test(NAME wireles_benchmark_moeng_smagorinsky_smoke
        COMMAND wireles_benchmark_moeng_smagorinsky
            --steps 2
            --sample-every 1
            --average-start 0
            --average-end 999
            --output-dir "${CMAKE_CURRENT_BINARY_DIR}/test_outputs/moeng_smagorinsky_benchmark_smoke"
    )
    set_tests_properties(wireles_benchmark_moeng_smagorinsky_smoke PROPERTIES
        LABELS "benchmark;moeng;smagorinsky;smoke"
        TIMEOUT 60
    )

    if(WIRELES_ENABLE_MPI AND MPI_CXX_FOUND AND MPIEXEC_EXECUTABLE)
        add_test(NAME wireles_benchmark_moeng_smagorinsky_mpi_smoke
            COMMAND wireles_benchmark_moeng_smagorinsky
                --mpi-ranks 2
                --steps 2
                --sample-every 1
                --average-start 0
                --average-end 999
                --output-dir "${CMAKE_CURRENT_BINARY_DIR}/test_outputs/moeng_smagorinsky_benchmark_mpi_smoke"
        )
        set_tests_properties(wireles_benchmark_moeng_smagorinsky_mpi_smoke PROPERTIES
            LABELS "benchmark;moeng;smagorinsky;smoke;mpi"
            TIMEOUT 60
        )

        add_test(NAME wireles_mpi_slab_smoke
            COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 2
                "$<TARGET_FILE:wireles>"
                --mpi-slab
                --nx 8
                --ny 8
                --nz 15
                --initial neutral_ekman
                --geostrophic-u 1.0
                --initial-velocity-perturbation 0.1
                --initial-perturbation-height 1.0
                --steps 2
                --log-every 1
                --dt 0.001
                --nu 0.001
        )
        set_tests_properties(wireles_mpi_slab_smoke PROPERTIES
            LABELS "mpi;slab"
            TIMEOUT 60
        )

        add_test(NAME wireles_mpi_slab_smagorinsky_dealias_smoke
            COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 2
                "$<TARGET_FILE:wireles>"
                --config "${CMAKE_CURRENT_SOURCE_DIR}/configs/neutral_ekman_64.toml"
                --mpi-slab
                --nx 8
                --ny 8
                --nz 8
                --steps 2
                --log-every 1
                --sgs smagorinsky
                --thermo
                --theta-gradient 0.001
                --surface-theta-flux 0.001
                --scalar-diffusivity 0.001
                --frame-start-step 100000
                --frame-end-step -1
        )
        set_tests_properties(wireles_mpi_slab_smagorinsky_dealias_smoke PROPERTIES
            LABELS "mpi;slab"
            TIMEOUT 60
        )

        add_test(NAME wireles_mpi_slab_rotational_smoke
            COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 2
                "$<TARGET_FILE:wireles>"
                --config "${CMAKE_CURRENT_SOURCE_DIR}/configs/neutral_ekman_64.toml"
                --mpi-slab
                --nx 8
                --ny 8
                --nz 15
                --steps 4
                --log-every 4
                --momentum-advection-form rotational
                --frame-start-step 100000
                --frame-end-step -1
        )
        set_tests_properties(wireles_mpi_slab_rotational_smoke PROPERTIES
            LABELS "mpi;slab;rotational"
            PASS_REGULAR_EXPRESSION
                "\\[advection-energy\\] mean_power_per_cell="
            TIMEOUT 60
        )

        add_test(NAME wireles_mpi_slab_lasd_smoke
            COMMAND "${MPIEXEC_EXECUTABLE}" "${MPIEXEC_NUMPROC_FLAG}" 2
                "$<TARGET_FILE:wireles>"
                --config "${CMAKE_CURRENT_SOURCE_DIR}/configs/neutral_ekman_64.toml"
                --mpi-slab
                --nx 8
                --ny 8
                --nz 8
                --steps 4
                --log-every 1
                --sgs lasd
                --scalar-sgs lasd
                --thermo
                --theta-gradient 0.001
                --surface-theta-flux 0.001
                --scalar-diffusivity 0.001
                --cs-count 2
                --frame-start-step 100000
                --frame-end-step -1
        )
        set_tests_properties(wireles_mpi_slab_lasd_smoke PROPERTIES
            LABELS "mpi;slab;lasd"
            TIMEOUT 60
        )
    endif()
endif()

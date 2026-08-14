cmake_minimum_required(VERSION 3.24)

project(jaxwind_lasd_cufft LANGUAGES CXX CUDA)

if(NOT DEFINED JAX_FFI_INCLUDE_DIR)
  message(FATAL_ERROR "JAX_FFI_INCLUDE_DIR must point to jaxlib/include")
endif()

find_package(CUDAToolkit REQUIRED)

add_library(jaxwind_lasd_cufft SHARED lasd_cufft.cu)
target_compile_features(jaxwind_lasd_cufft PRIVATE cxx_std_17)
target_include_directories(jaxwind_lasd_cufft PRIVATE "${JAX_FFI_INCLUDE_DIR}")
target_link_libraries(
  jaxwind_lasd_cufft
  PRIVATE
    CUDA::cudart
    CUDA::cufft
)
set_target_properties(
  jaxwind_lasd_cufft
  PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

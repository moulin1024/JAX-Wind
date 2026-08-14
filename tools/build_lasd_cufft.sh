#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_executable="${JAXWIND_PYTHON:-/home/moulin/anaconda3/envs/jaxcuda/bin/python}"
build_directory="${repository_root}/build/native/lasd_cufft"
cuda_architectures="${JAXWIND_CUDA_ARCHITECTURES:-86}"
jax_include="$(${python_executable} -c 'import pathlib, jaxlib; print(pathlib.Path(jaxlib.__file__).parent / "include")')"

cmake \
  -S "${repository_root}/native/lasd_cufft" \
  -B "${build_directory}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="${cuda_architectures}" \
  -DJAX_FFI_INCLUDE_DIR="${jax_include}"
cmake --build "${build_directory}" --parallel

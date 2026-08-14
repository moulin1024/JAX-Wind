#include <cuda_runtime_api.h>
#include <cufft.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

struct TransformPlans {
  cufftHandle forward = 0;
  cufftHandle inverse = 0;
};

struct CacheEntry {
  int device = 0;
  cudaStream_t stream = nullptr;
  int components = 0;
  int levels = 0;
  int ny = 0;
  int nx = 0;
  int maximum_group = 0;
  cufftComplex* spectrum = nullptr;
  cufftComplex* filtered = nullptr;
  std::unordered_map<int, TransformPlans> plans;
};

std::mutex& CacheMutex() {
  static auto* mutex = new std::mutex;
  return *mutex;
}

std::vector<std::unique_ptr<CacheEntry>>& Cache() {
  // CUDA resources intentionally live until process termination. Destroying
  // them from static destructors is unsafe after the CUDA runtime is unloaded.
  static auto* cache = new std::vector<std::unique_ptr<CacheEntry>>;
  return *cache;
}

std::string CufftMessage(const char* operation, cufftResult result) {
  std::ostringstream message;
  message << operation << " failed with cuFFT status " << static_cast<int>(result);
  return message.str();
}

std::string CudaMessage(const char* operation, cudaError_t result) {
  std::ostringstream message;
  message << operation << " failed: " << cudaGetErrorString(result);
  return message.str();
}

bool SameKey(const CacheEntry& entry, int device, cudaStream_t stream,
             int components, int levels, int ny, int nx) {
  return entry.device == device && entry.stream == stream &&
         entry.components == components && entry.levels == levels &&
         entry.ny == ny && entry.nx == nx;
}

std::vector<int> ComponentGroups(int components) {
  std::vector<int> groups;
  int remaining = components;
  // Momentum LASD is laid out as 3 + 6 + 6 + 6. Keep the leading vector
  // batch and reuse the six-component tensor plan for all following tensors.
  if (remaining >= 3) {
    groups.push_back(3);
    remaining -= 3;
  }
  while (remaining > 0) {
    const int group = std::min(6, remaining);
    groups.push_back(group);
    remaining -= group;
  }
  return groups;
}

cufftResult MakePlans(CacheEntry* entry, int group) {
  const int rank = 2;
  int dimensions[rank] = {entry->ny, entry->nx};
  int real_embed[rank] = {entry->ny, entry->nx};
  int complex_embed[rank] = {entry->ny, entry->nx / 2 + 1};
  const int real_distance = entry->ny * entry->nx;
  const int complex_distance = entry->ny * (entry->nx / 2 + 1);
  const int batch = entry->levels * group;

  TransformPlans plans;
  cufftResult result = cufftPlanMany(
      &plans.forward, rank, dimensions, real_embed, 1, real_distance,
      complex_embed, 1, complex_distance, CUFFT_R2C, batch);
  if (result != CUFFT_SUCCESS) return result;
  result = cufftPlanMany(
      &plans.inverse, rank, dimensions, complex_embed, 1, complex_distance,
      real_embed, 1, real_distance, CUFFT_C2R, batch);
  if (result != CUFFT_SUCCESS) return result;
  entry->plans.emplace(group, plans);
  return CUFFT_SUCCESS;
}

ffi::ErrorOr<CacheEntry*> GetCacheEntry(int device, cudaStream_t stream,
                                        int components, int levels, int ny,
                                        int nx) {
  std::lock_guard<std::mutex> lock(CacheMutex());
  for (const auto& candidate : Cache()) {
    if (SameKey(*candidate, device, stream, components, levels, ny, nx)) {
      return candidate.get();
    }
  }

  auto entry = std::make_unique<CacheEntry>();
  entry->device = device;
  entry->stream = stream;
  entry->components = components;
  entry->levels = levels;
  entry->ny = ny;
  entry->nx = nx;
  const std::vector<int> groups = ComponentGroups(components);
  entry->maximum_group = *std::max_element(groups.begin(), groups.end());

  const std::size_t spectral_values =
      static_cast<std::size_t>(entry->maximum_group) * levels * ny *
      (nx / 2 + 1);
  cudaError_t cuda_result = cudaMalloc(
      reinterpret_cast<void**>(&entry->spectrum),
      spectral_values * sizeof(cufftComplex));
  if (cuda_result != cudaSuccess) {
    return ffi::Unexpected(
        ffi::Error::Internal(CudaMessage("cudaMalloc(spectrum)", cuda_result)));
  }
  cuda_result = cudaMalloc(
      reinterpret_cast<void**>(&entry->filtered),
      spectral_values * sizeof(cufftComplex));
  if (cuda_result != cudaSuccess) {
    return ffi::Unexpected(
        ffi::Error::Internal(CudaMessage("cudaMalloc(filtered)", cuda_result)));
  }

  for (const int group : groups) {
    if (entry->plans.count(group) != 0) continue;
    const cufftResult plan_result = MakePlans(entry.get(), group);
    if (plan_result != CUFFT_SUCCESS) {
      return ffi::Unexpected(ffi::Error::Internal(
          CufftMessage("cufftPlanMany", plan_result)));
    }
  }

  CacheEntry* result = entry.get();
  Cache().push_back(std::move(entry));
  return result;
}

__global__ void ApplySharpCutoff(const cufftComplex* source,
                                 cufftComplex* destination,
                                 std::int64_t count, int ny, int nx,
                                 const float* filter_width) {
  const std::int64_t index =
      static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;

  const int spectral_nx = nx / 2 + 1;
  const int x_mode = static_cast<int>(index % spectral_nx);
  const int y_index = static_cast<int>((index / spectral_nx) % ny);
  const int y_mode = y_index >= (ny + 1) / 2 ? y_index - ny : y_index;
  const float width = *filter_width;
  const int cutoff_x = static_cast<int>(floorf(nx / (2.0f * width) + 0.5f));
  const int cutoff_y = static_cast<int>(floorf(ny / (2.0f * width) + 0.5f));
  const float normalization = 1.0f / static_cast<float>(nx * ny);

  if (x_mode < cutoff_x && abs(y_mode) < cutoff_y) {
    const cufftComplex value = source[index];
    destination[index] =
        make_cuFloatComplex(value.x * normalization, value.y * normalization);
  } else {
    destination[index] = make_cuFloatComplex(0.0f, 0.0f);
  }
}

ffi::Error FilterTwoScales(
    cudaStream_t stream, ffi::BufferR4<ffi::F32> values,
    ffi::BufferR0<ffi::F32> first_filter_width,
    ffi::BufferR0<ffi::F32> second_filter_width,
    ffi::ResultBufferR4<ffi::F32> output) {
  const auto input_dimensions = values.dimensions();
  const auto output_dimensions = output->dimensions();
  const int components = static_cast<int>(input_dimensions[0]);
  const int levels = static_cast<int>(input_dimensions[1]);
  const int ny = static_cast<int>(input_dimensions[2]);
  const int nx = static_cast<int>(input_dimensions[3]);
  if (components <= 0 || levels <= 0 || ny <= 0 || nx <= 1) {
    return ffi::Error::InvalidArgument("LASD filter dimensions must be positive");
  }
  if (output_dimensions[0] != 2 * input_dimensions[0] ||
      output_dimensions[1] != input_dimensions[1] ||
      output_dimensions[2] != input_dimensions[2] ||
      output_dimensions[3] != input_dimensions[3]) {
    return ffi::Error::InvalidArgument(
        "LASD filter output must have shape (2*components, z, y, x)");
  }

  int device = 0;
  cudaError_t cuda_result = cudaGetDevice(&device);
  if (cuda_result != cudaSuccess) {
    return ffi::Error::Internal(CudaMessage("cudaGetDevice", cuda_result));
  }
  auto entry_or =
      GetCacheEntry(device, stream, components, levels, ny, nx);
  if (!entry_or.has_value()) return entry_or.error();
  CacheEntry* entry = *entry_or;

  const std::int64_t real_plane = static_cast<std::int64_t>(levels) * ny * nx;
  const std::int64_t spectral_plane =
      static_cast<std::int64_t>(levels) * ny * (nx / 2 + 1);
  const float* input = values.typed_data();
  float* result = output->typed_data();
  const std::vector<int> groups = ComponentGroups(components);
  int component_offset = 0;
  for (const int group : groups) {
    TransformPlans& plans = entry->plans.at(group);
    cufftResult cufft_result = cufftSetStream(plans.forward, stream);
    if (cufft_result != CUFFT_SUCCESS) {
      return ffi::Error::Internal(
          CufftMessage("cufftSetStream(forward)", cufft_result));
    }
    cufft_result = cufftSetStream(plans.inverse, stream);
    if (cufft_result != CUFFT_SUCCESS) {
      return ffi::Error::Internal(
          CufftMessage("cufftSetStream(inverse)", cufft_result));
    }
    cufft_result = cufftExecR2C(
        plans.forward,
        const_cast<cufftReal*>(input + component_offset * real_plane),
        entry->spectrum);
    if (cufft_result != CUFFT_SUCCESS) {
      return ffi::Error::Internal(
          CufftMessage("cufftExecR2C", cufft_result));
    }

    const std::int64_t spectrum_count = group * spectral_plane;
    constexpr int threads = 256;
    const int blocks = static_cast<int>((spectrum_count + threads - 1) / threads);
    ApplySharpCutoff<<<blocks, threads, 0, stream>>>(
        entry->spectrum, entry->filtered, spectrum_count, ny, nx,
        first_filter_width.typed_data());
    cufft_result = cufftExecC2R(
        plans.inverse, entry->filtered,
        result + component_offset * real_plane);
    if (cufft_result != CUFFT_SUCCESS) {
      return ffi::Error::Internal(
          CufftMessage("cufftExecC2R(first)", cufft_result));
    }

    ApplySharpCutoff<<<blocks, threads, 0, stream>>>(
        entry->spectrum, entry->filtered, spectrum_count, ny, nx,
        second_filter_width.typed_data());
    cufft_result = cufftExecC2R(
        plans.inverse, entry->filtered,
        result + (components + component_offset) * real_plane);
    if (cufft_result != CUFFT_SUCCESS) {
      return ffi::Error::Internal(
          CufftMessage("cufftExecC2R(second)", cufft_result));
    }
    component_offset += group;
  }

  cuda_result = cudaPeekAtLastError();
  if (cuda_result != cudaSuccess) {
    return ffi::Error::Internal(
        CudaMessage("ApplySharpCutoff", cuda_result));
  }
  return ffi::Error::Success();
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    JaxwindLasdFilterTwoScalesF32, FilterTwoScales,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::BufferR4<ffi::F32>>()
        .Arg<ffi::BufferR0<ffi::F32>>()
        .Arg<ffi::BufferR0<ffi::F32>>()
        .Ret<ffi::BufferR4<ffi::F32>>());

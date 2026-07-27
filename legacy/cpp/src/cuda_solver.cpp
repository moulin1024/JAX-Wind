#include "wireles/cuda_solver.hpp"

#ifndef WIRELES_HAVE_CUDA

#include <stdexcept>

namespace wireles {

struct CudaFlowState::Impl {};

CudaFlowState::CudaFlowState(const Params&) {
    throw std::runtime_error("CUDA support was requested, but this build was configured without WIRELES_ENABLE_CUDA=ON");
}

CudaFlowState::~CudaFlowState() = default;

bool cuda_available() {
    return false;
}

void cuda_upload_flow_state(CudaFlowState&, const FlowState&, const Params&) {
    throw std::runtime_error("CUDA support was requested, but this build was configured without WIRELES_ENABLE_CUDA=ON");
}

void cuda_download_flow_state(const CudaFlowState&, FlowState&, const Params&) {
    throw std::runtime_error("CUDA support was requested, but this build was configured without WIRELES_ENABLE_CUDA=ON");
}

void cuda_enforce_walls(TimestepWorkspace&, const Params&) {
    throw std::runtime_error("CUDA support was requested, but this build was configured without WIRELES_ENABLE_CUDA=ON");
}

void cuda_project(FlowState&, TimestepWorkspace&, const Params&) {
    throw std::runtime_error("CUDA support was requested, but this build was configured without WIRELES_ENABLE_CUDA=ON");
}

void cuda_step_device_resident(FlowState&, TimestepWorkspace&, bool, const Params&) {
    throw std::runtime_error("CUDA support was requested, but this build was configured without WIRELES_ENABLE_CUDA=ON");
}

}  // namespace wireles

#endif

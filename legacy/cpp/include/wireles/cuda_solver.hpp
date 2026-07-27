#pragma once

#include <memory>

#include "wireles/field.hpp"
#include "wireles/params.hpp"

namespace wireles {

struct TimestepWorkspace;

class CudaFlowState {
public:
    explicit CudaFlowState(const Params& params);
    CudaFlowState(const CudaFlowState&) = delete;
    CudaFlowState& operator=(const CudaFlowState&) = delete;
    ~CudaFlowState();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    friend bool cuda_available();
    friend void cuda_upload_flow_state(CudaFlowState& device, const FlowState& state, const Params& params);
    friend void cuda_download_flow_state(const CudaFlowState& device, FlowState& state, const Params& params);
    friend void cuda_enforce_walls(TimestepWorkspace& workspace, const Params& params);
    friend void cuda_project(FlowState& state, TimestepWorkspace& workspace, const Params& params);
    friend void cuda_step_device_resident(FlowState& state, TimestepWorkspace& workspace, bool use_ab2, const Params& params);
};

bool cuda_available();
void cuda_upload_flow_state(CudaFlowState& device, const FlowState& state, const Params& params);
void cuda_download_flow_state(const CudaFlowState& device, FlowState& state, const Params& params);
void cuda_enforce_walls(TimestepWorkspace& workspace, const Params& params);
void cuda_project(FlowState& state, TimestepWorkspace& workspace, const Params& params);
void cuda_step_device_resident(FlowState& state, TimestepWorkspace& workspace, bool use_ab2, const Params& params);

}  // namespace wireles

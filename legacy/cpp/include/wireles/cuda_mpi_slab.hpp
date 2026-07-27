#pragma once

#include "wireles/params.hpp"

namespace wireles {

int run_cuda_mpi_slab(const Params& params, int argc, char** argv);

}  // namespace wireles

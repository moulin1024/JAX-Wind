#include "wireles/cuda_mpi_slab.hpp"

#include <stdexcept>

namespace wireles {

#if !defined(WIRELES_HAVE_CUDA) || !defined(WIRELES_HAVE_MPI)
int run_cuda_mpi_slab(const Params&, int, char**) {
    throw std::runtime_error("CUDA-MPI slab support requires WIRELES_ENABLE_CUDA=ON and WIRELES_ENABLE_MPI=ON");
}
#endif

}  // namespace wireles

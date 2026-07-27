#include "wireles/cuda_mpi_slab.hpp"

#include <mpi.h>

#include <cuda_runtime.h>
#include <cufft.h>

#ifdef WIRELES_HAVE_HDF5
#include <hdf5.h>
#endif

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <sys/stat.h>

#include "wireles/bomex.hpp"
#include "wireles/field.hpp"
#include "wireles/thermodynamics.hpp"

namespace wireles {
namespace {

void check_cuda(cudaError_t result, const char* action) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA ") + action + " failed: " + cudaGetErrorString(result));
    }
}

void check_cufft(cufftResult result, const char* action) {
    if (result != CUFFT_SUCCESS) {
        throw std::runtime_error(std::string("cuFFT ") + action + " failed with code " + std::to_string(static_cast<int>(result)));
    }
}

void check_mpi(int result, const char* action) {
    if (result != MPI_SUCCESS) {
        throw std::runtime_error(std::string("MPI ") + action + " failed");
    }
}

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) { resize(count); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    void resize(std::size_t count) {
        if (count == count_) {
            return;
        }
        if (data_ != nullptr) {
            cudaFree(data_);
            data_ = nullptr;
        }
        count_ = count;
        if (count_ > 0) {
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)), "allocation");
        }
    }

    void zero() {
        if (count_ > 0) {
            check_cuda(cudaMemset(data_, 0, count_ * sizeof(T)), "memset");
        }
    }

    void copy_from(const std::vector<T>& values, const char* name) {
        if (values.size() != count_) {
            throw std::runtime_error(std::string("CUDA-MPI input size mismatch for ") + name);
        }
        if (count_ > 0) {
            check_cuda(cudaMemcpy(data_, values.data(), count_ * sizeof(T), cudaMemcpyHostToDevice), "host-to-device copy");
        }
    }

    void copy_to(std::vector<T>& values, const char* name) const {
        values.resize(count_);
        if (count_ > 0) {
            check_cuda(cudaMemcpy(values.data(), data_, count_ * sizeof(T), cudaMemcpyDeviceToHost), "device-to-host copy");
        }
    }

    T* data() { return data_; }
    const T* data() const { return data_; }
    std::size_t size() const { return count_; }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

struct CufftPlan {
    cufftHandle handle = 0;
    CufftPlan() = default;
    CufftPlan(const CufftPlan&) = delete;
    CufftPlan& operator=(const CufftPlan&) = delete;
    ~CufftPlan() {
        if (handle != 0) {
            cufftDestroy(handle);
        }
    }
};

struct MpiRuntime {
    MpiRuntime(int argc, char** argv) {
        MPI_Init(&argc, &argv);
    }
    MpiRuntime(const MpiRuntime&) = delete;
    MpiRuntime& operator=(const MpiRuntime&) = delete;
    ~MpiRuntime() {
        MPI_Finalize();
    }
};

struct Slab {
    int rank = 0;
    int size = 1;
    int k_begin = 0;
    int k_count = 0;
    int face_begin = 0;
    int face_count = 0;
};

std::size_t plane_size(const Params& params) {
    return static_cast<std::size_t>(params.nx) * static_cast<std::size_t>(params.ny);
}

Slab make_slab(const Params& params, MPI_Comm comm) {
    Slab slab;
    MPI_Comm_rank(comm, &slab.rank);
    MPI_Comm_size(comm, &slab.size);
    if (slab.size > params.nz) {
        throw std::runtime_error("CUDA-MPI slab path requires number of ranks <= nz");
    }
    if (params.nz % slab.size != 0) {
        throw std::runtime_error("CUDA-MPI distributed pressure path requires nz divisible by number of ranks");
    }
    if (params.ny % slab.size != 0) {
        throw std::runtime_error("CUDA-MPI distributed pressure path requires ny divisible by number of ranks");
    }
    const int base = params.nz / slab.size;
    slab.k_begin = slab.rank * base;
    slab.k_count = base;
    slab.face_begin = slab.k_begin;
    slab.face_count = base + (slab.rank == slab.size - 1 ? 1 : 0);
    return slab;
}

bool owns_center_plane(const Slab& slab, int k) {
    return k >= slab.k_begin && k < slab.k_begin + slab.k_count;
}

bool owns_face_plane(const Slab& slab, int k) {
    return k >= slab.face_begin && k < slab.face_begin + slab.face_count;
}

struct DeviceLocalField {
    DeviceBuffer<double> values;
    int plane_begin = 0;
    int plane_count = 0;
    int owned_begin = 0;
    int owned_count = 0;
    int total_planes = 0;
    std::size_t plane_stride = 0;

    void resize(int begin, int count, int total, const Params& params, bool with_halo = true) {
        owned_begin = begin;
        owned_count = count;
        total_planes = total;
        plane_stride = plane_size(params);
        const int lower_halo = with_halo && begin > 0 ? 1 : 0;
        const int upper_halo = with_halo && begin + count < total ? 1 : 0;
        plane_begin = begin - lower_halo;
        plane_count = count + lower_halo + upper_halo;
        values.resize(static_cast<std::size_t>(plane_count) * plane_stride);
        values.zero();
    }
};

struct CudaMpiState {
    explicit CudaMpiState(const Params& params, const Slab& slab) {
        u.resize(slab.k_begin, slab.k_count, params.nz, params);
        v.resize(slab.k_begin, slab.k_count, params.nz, params);
        p.resize(slab.k_begin, slab.k_count, params.nz, params);
        theta.resize(slab.k_begin, slab.k_count, params.nz, params);
        theta_l.resize(slab.k_begin, slab.k_count, params.nz, params);
        qt.resize(slab.k_begin, slab.k_count, params.nz, params);
        qv.resize(slab.k_begin, slab.k_count, params.nz, params);
        ql.resize(slab.k_begin, slab.k_count, params.nz, params);
        w.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        base_pressure.resize(static_cast<std::size_t>(params.nz));
    }

    DeviceLocalField u;
    DeviceLocalField v;
    DeviceLocalField w;
    DeviceLocalField p;
    DeviceLocalField theta;
    DeviceLocalField theta_l;
    DeviceLocalField qt;
    DeviceLocalField qv;
    DeviceLocalField ql;
    DeviceBuffer<double> base_pressure;
    int step_count = 0;
    bool has_rhs_prev = false;
};

struct HaloScratch {
    DeviceBuffer<double> send_lower;
    DeviceBuffer<double> send_upper;
    DeviceBuffer<double> recv_lower;
    DeviceBuffer<double> recv_upper;

    explicit HaloScratch(std::size_t plane) : send_lower(plane), send_upper(plane), recv_lower(plane), recv_upper(plane) {}
};

struct PressureWorkspace {
    explicit PressureWorkspace(const Params& params, const Slab& slab)
        : owned_real_count(static_cast<std::size_t>(slab.k_count) * plane_size(params)),
          spectral_count(static_cast<std::size_t>(slab.k_count) * static_cast<std::size_t>(params.ny) * static_cast<std::size_t>(params.nkx())),
          nj(params.ny / slab.size),
          chunk(static_cast<std::size_t>(params.nkx()) * static_cast<std::size_t>(nj) * static_cast<std::size_t>(slab.k_count)),
          pencil_count(static_cast<std::size_t>(params.nkx()) * static_cast<std::size_t>(nj) * static_cast<std::size_t>(params.nz)),
          u_owned(owned_real_count),
          v_owned(owned_real_count),
          dwdz_owned(owned_real_count),
          p_owned(owned_real_count),
          dpdx_owned(owned_real_count),
          dpdy_owned(owned_real_count),
          u_hat(spectral_count),
          v_hat(spectral_count),
          dwdz_hat(spectral_count),
          div_hat(spectral_count),
          p_hat(spectral_count),
          dpdx_hat(spectral_count),
          dpdy_hat(spectral_count),
          transpose_send(chunk * static_cast<std::size_t>(slab.size)),
          transpose_recv(chunk * static_cast<std::size_t>(slab.size)),
          y_pencil(pencil_count),
          thomas_cp(pencil_count),
          thomas_dp(pencil_count) {
        int n[2] = {params.ny, params.nx};
        check_cufft(cufftPlanMany(&r2c.handle, 2, n, nullptr, 1, params.nx * params.ny,
                        nullptr, 1, params.ny * params.nkx(), CUFFT_D2Z, slab.k_count),
            "CUDA-MPI D2Z plan creation");
        check_cufft(cufftPlanMany(&c2r.handle, 2, n, nullptr, 1, params.ny * params.nkx(),
                        nullptr, 1, params.nx * params.ny, CUFFT_Z2D, slab.k_count),
            "CUDA-MPI Z2D plan creation");
    }

    std::size_t owned_real_count;
    std::size_t spectral_count;
    int nj;
    std::size_t chunk;
    std::size_t pencil_count;
    DeviceBuffer<double> u_owned;
    DeviceBuffer<double> v_owned;
    DeviceBuffer<double> dwdz_owned;
    DeviceBuffer<double> p_owned;
    DeviceBuffer<double> dpdx_owned;
    DeviceBuffer<double> dpdy_owned;
    DeviceBuffer<cufftDoubleComplex> u_hat;
    DeviceBuffer<cufftDoubleComplex> v_hat;
    DeviceBuffer<cufftDoubleComplex> dwdz_hat;
    DeviceBuffer<cufftDoubleComplex> div_hat;
    DeviceBuffer<cufftDoubleComplex> p_hat;
    DeviceBuffer<cufftDoubleComplex> dpdx_hat;
    DeviceBuffer<cufftDoubleComplex> dpdy_hat;
    DeviceBuffer<cufftDoubleComplex> transpose_send;
    DeviceBuffer<cufftDoubleComplex> transpose_recv;
    DeviceBuffer<cufftDoubleComplex> y_pencil;
    DeviceBuffer<cufftDoubleComplex> thomas_cp;
    DeviceBuffer<cufftDoubleComplex> thomas_dp;
    CufftPlan r2c;
    CufftPlan c2r;
};

struct RhsWorkspace {
    explicit RhsWorkspace(const Params& params, const Slab& slab)
        : center_count(static_cast<std::size_t>(slab.k_count) * plane_size(params)),
          face_count(static_cast<std::size_t>(slab.face_count) * plane_size(params)),
          center_spectral_count(static_cast<std::size_t>(slab.k_count) * static_cast<std::size_t>(params.ny) * static_cast<std::size_t>(params.nkx())),
          face_spectral_count(static_cast<std::size_t>(slab.face_count) * static_cast<std::size_t>(params.ny) * static_cast<std::size_t>(params.nkx())),
          u(center_count),
          v(center_count),
          w_center(center_count),
          dudx(center_count),
          dudy(center_count),
          dudz(center_count),
          dvdx(center_count),
          dvdy(center_count),
          dvdz(center_count),
          lap_u(center_count),
          lap_v(center_count),
          rhs_u(center_count),
          rhs_v(center_count),
          rhs_theta(center_count),
          rhs_qt(center_count),
          rhs_u_prev(center_count),
          rhs_v_prev(center_count),
          rhs_theta_prev(center_count),
          rhs_qt_prev(center_count),
          rhs_u_prev2(center_count),
          rhs_v_prev2(center_count),
          rhs_theta_prev2(center_count),
          rhs_qt_prev2(center_count),
          theta(center_count),
          qt(center_count),
          dwdx_center(center_count),
          dwdy_center(center_count),
          dwdz_center(center_count),
          strain(center_count),
          nu_t(center_count),
          txx(center_count),
          txy(center_count),
          tyy(center_count),
          tzz(center_count),
          dtxx_dx(center_count),
          dtxy_dy(center_count),
          dtxy_dx(center_count),
          dtyy_dy(center_count),
          dtxz_dz(center_count),
          dtyz_dz(center_count),
          w_face(face_count),
          u_on_w(face_count),
          v_on_w(face_count),
          dwdx_face(face_count),
          dwdy_face(face_count),
          dwdz_face(face_count),
          lap_w(face_count),
          rhs_w(face_count),
          rhs_w_prev(face_count),
          rhs_w_prev2(face_count),
          nu_t_face(face_count),
          dudz_face(face_count),
          dvdz_face(face_count),
          txz(face_count),
          tyz(face_count),
          dtxz_dx(face_count),
          dtyz_dy(face_count),
          dtzz_dz(face_count),
          theta_on_w(face_count),
          theta_flux_x(center_count),
          theta_flux_y(center_count),
          theta_flux_z(face_count),
          dtheta_dx(center_count),
          dtheta_dy(center_count),
          dtheta_dz_center(center_count),
          dtheta_dz_w(face_count),
          kappa_center(center_count),
          theta_kappa(center_count),
          qt_kappa(center_count),
          kappa_w(face_count),
          qx(center_count),
          qy(center_count),
          qz(face_count),
          div_xy(center_count),
          div_z(center_count),
          theta_plane_mean(static_cast<std::size_t>(slab.k_count + (slab.k_begin > 0 ? 1 : 0)
              + (slab.k_begin + slab.k_count < params.nz ? 1 : 0))),
          u_plane_mean(static_cast<std::size_t>(params.nz)),
          v_plane_mean(static_cast<std::size_t>(params.nz)),
          qt_plane_mean(static_cast<std::size_t>(params.nz)),
          moisture_sums(2),
          center_hat(center_spectral_count),
          center_hat_scratch(center_spectral_count),
          face_hat(face_spectral_count),
          face_hat_scratch(face_spectral_count) {
        rhs_u_prev.zero();
        rhs_v_prev.zero();
        rhs_w_prev.zero();
        rhs_w_prev2.zero();
        rhs_theta_prev.zero();
        rhs_qt_prev.zero();
        rhs_u_prev2.zero();
        rhs_v_prev2.zero();
        rhs_theta_prev2.zero();
        rhs_qt_prev2.zero();
        nu_t_field.resize(slab.k_begin, slab.k_count, params.nz, params);
        tzz_field.resize(slab.k_begin, slab.k_count, params.nz, params);
        txz_field.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        tyz_field.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        scalar_flux_z_field.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        scalar_qz_field.resize(slab.face_begin, slab.face_count, params.nz + 1, params);
        int n[2] = {params.ny, params.nx};
        check_cufft(cufftPlanMany(&r2c_center.handle, 2, n, nullptr, 1, params.nx * params.ny,
                        nullptr, 1, params.ny * params.nkx(), CUFFT_D2Z, slab.k_count),
            "CUDA-MPI RHS center D2Z plan creation");
        check_cufft(cufftPlanMany(&c2r_center.handle, 2, n, nullptr, 1, params.ny * params.nkx(),
                        nullptr, 1, params.nx * params.ny, CUFFT_Z2D, slab.k_count),
            "CUDA-MPI RHS center Z2D plan creation");
        check_cufft(cufftPlanMany(&r2c_face.handle, 2, n, nullptr, 1, params.nx * params.ny,
                        nullptr, 1, params.ny * params.nkx(), CUFFT_D2Z, slab.face_count),
            "CUDA-MPI RHS face D2Z plan creation");
        check_cufft(cufftPlanMany(&c2r_face.handle, 2, n, nullptr, 1, params.ny * params.nkx(),
                        nullptr, 1, params.nx * params.ny, CUFFT_Z2D, slab.face_count),
            "CUDA-MPI RHS face Z2D plan creation");
    }

    std::size_t center_count;
    std::size_t face_count;
    std::size_t center_spectral_count;
    std::size_t face_spectral_count;
    DeviceBuffer<double> u;
    DeviceBuffer<double> v;
    DeviceBuffer<double> w_center;
    DeviceBuffer<double> dudx;
    DeviceBuffer<double> dudy;
    DeviceBuffer<double> dudz;
    DeviceBuffer<double> dvdx;
    DeviceBuffer<double> dvdy;
    DeviceBuffer<double> dvdz;
    DeviceBuffer<double> lap_u;
    DeviceBuffer<double> lap_v;
    DeviceBuffer<double> rhs_u;
    DeviceBuffer<double> rhs_v;
    DeviceBuffer<double> rhs_theta;
    DeviceBuffer<double> rhs_qt;
    DeviceBuffer<double> rhs_u_prev;
    DeviceBuffer<double> rhs_v_prev;
    DeviceBuffer<double> rhs_theta_prev;
    DeviceBuffer<double> rhs_qt_prev;
    DeviceBuffer<double> rhs_u_prev2;
    DeviceBuffer<double> rhs_v_prev2;
    DeviceBuffer<double> rhs_theta_prev2;
    DeviceBuffer<double> rhs_qt_prev2;
    DeviceBuffer<double> theta;
    DeviceBuffer<double> qt;
    DeviceBuffer<double> dwdx_center;
    DeviceBuffer<double> dwdy_center;
    DeviceBuffer<double> dwdz_center;
    DeviceBuffer<double> strain;
    DeviceBuffer<double> nu_t;
    DeviceBuffer<double> txx;
    DeviceBuffer<double> txy;
    DeviceBuffer<double> tyy;
    DeviceBuffer<double> tzz;
    DeviceBuffer<double> dtxx_dx;
    DeviceBuffer<double> dtxy_dy;
    DeviceBuffer<double> dtxy_dx;
    DeviceBuffer<double> dtyy_dy;
    DeviceBuffer<double> dtxz_dz;
    DeviceBuffer<double> dtyz_dz;
    DeviceBuffer<double> w_face;
    DeviceBuffer<double> u_on_w;
    DeviceBuffer<double> v_on_w;
    DeviceBuffer<double> dwdx_face;
    DeviceBuffer<double> dwdy_face;
    DeviceBuffer<double> dwdz_face;
    DeviceBuffer<double> lap_w;
    DeviceBuffer<double> rhs_w;
    DeviceBuffer<double> rhs_w_prev;
    DeviceBuffer<double> rhs_w_prev2;
    DeviceBuffer<double> nu_t_face;
    DeviceBuffer<double> dudz_face;
    DeviceBuffer<double> dvdz_face;
    DeviceBuffer<double> txz;
    DeviceBuffer<double> tyz;
    DeviceBuffer<double> dtxz_dx;
    DeviceBuffer<double> dtyz_dy;
    DeviceBuffer<double> dtzz_dz;
    DeviceBuffer<double> theta_on_w;
    DeviceBuffer<double> theta_flux_x;
    DeviceBuffer<double> theta_flux_y;
    DeviceBuffer<double> theta_flux_z;
    DeviceBuffer<double> dtheta_dx;
    DeviceBuffer<double> dtheta_dy;
    DeviceBuffer<double> dtheta_dz_center;
    DeviceBuffer<double> dtheta_dz_w;
    DeviceBuffer<double> kappa_center;
    DeviceBuffer<double> theta_kappa;
    DeviceBuffer<double> qt_kappa;
    DeviceBuffer<double> kappa_w;
    DeviceBuffer<double> qx;
    DeviceBuffer<double> qy;
    DeviceBuffer<double> qz;
    DeviceBuffer<double> div_xy;
    DeviceBuffer<double> div_z;
    DeviceBuffer<double> theta_plane_mean;
    DeviceBuffer<double> u_plane_mean;
    DeviceBuffer<double> v_plane_mean;
    DeviceBuffer<double> qt_plane_mean;
    DeviceBuffer<double> moisture_sums;
    DeviceBuffer<cufftDoubleComplex> center_hat;
    DeviceBuffer<cufftDoubleComplex> center_hat_scratch;
    DeviceBuffer<cufftDoubleComplex> face_hat;
    DeviceBuffer<cufftDoubleComplex> face_hat_scratch;
    DeviceLocalField nu_t_field;
    DeviceLocalField tzz_field;
    DeviceLocalField txz_field;
    DeviceLocalField tyz_field;
    DeviceLocalField scalar_flux_z_field;
    DeviceLocalField scalar_qz_field;
    CufftPlan r2c_center;
    CufftPlan c2r_center;
    CufftPlan r2c_face;
    CufftPlan c2r_face;
};

int blocks_for(std::size_t count, int threads) {
    return static_cast<int>((count + static_cast<std::size_t>(threads) - 1) / static_cast<std::size_t>(threads));
}

__device__ cufftDoubleComplex cadd(cufftDoubleComplex a, cufftDoubleComplex b) {
    return make_cuDoubleComplex(a.x + b.x, a.y + b.y);
}

__device__ cufftDoubleComplex csub(cufftDoubleComplex a, cufftDoubleComplex b) {
    return make_cuDoubleComplex(a.x - b.x, a.y - b.y);
}

__device__ cufftDoubleComplex cmul(cufftDoubleComplex a, cufftDoubleComplex b) {
    return make_cuDoubleComplex(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ cufftDoubleComplex cscale(cufftDoubleComplex a, double s) {
    return make_cuDoubleComplex(a.x * s, a.y * s);
}

__device__ cufftDoubleComplex cdiv(cufftDoubleComplex a, cufftDoubleComplex b) {
    const double denom = b.x * b.x + b.y * b.y;
    return make_cuDoubleComplex((a.x * b.x + a.y * b.y) / denom, (a.y * b.x - a.x * b.y) / denom);
}

__device__ double kx_derivative_device(int nx, double lx, int ih) {
    if ((nx % 2) == 0 && ih == nx / 2) {
        return 0.0;
    }
    return 2.0 * 3.141592653589793238462643383279502884 * static_cast<double>(ih) / lx;
}

__device__ double ky_derivative_device(int ny, double ly, int j) {
    if ((ny % 2) == 0 && j == ny / 2) {
        return 0.0;
    }
    const int signed_j = (j <= ny / 2) ? j : j - ny;
    return 2.0 * 3.141592653589793238462643383279502884 * static_cast<double>(signed_j) / ly;
}

__device__ double kx_value_device(int, double lx, int ih) {
    return 2.0 * 3.141592653589793238462643383279502884 * static_cast<double>(ih) / lx;
}

__device__ double ky_value_device(int ny, double ly, int j) {
    const int signed_j = (j <= ny / 2) ? j : j - ny;
    return 2.0 * 3.141592653589793238462643383279502884 * static_cast<double>(signed_j) / ly;
}

__global__ void scale_real_kernel(double* q, std::size_t count, double scale) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n < count) {
        q[n] *= scale;
    }
}

__global__ void copy_owned_from_local_kernel(
    const double* field,
    double* owned,
    std::size_t count,
    int nx,
    int ny,
    int owned_begin,
    int plane_begin) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const int k_local = static_cast<int>(n / plane);
    const std::size_t in_plane = n - static_cast<std::size_t>(k_local) * plane;
    const int k_global = owned_begin + k_local;
    owned[n] = field[static_cast<std::size_t>(k_global - plane_begin) * plane + in_plane];
}

__global__ void scatter_owned_to_local_kernel(
    const double* owned,
    double* field,
    std::size_t count,
    int nx,
    int ny,
    int owned_begin,
    int plane_begin) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const int k_local = static_cast<int>(n / plane);
    const std::size_t in_plane = n - static_cast<std::size_t>(k_local) * plane;
    const int k_global = owned_begin + k_local;
    field[static_cast<std::size_t>(k_global - plane_begin) * plane + in_plane] = owned[n];
}

__global__ void spectral_derivative_x_kernel(
    const cufftDoubleComplex* in,
    cufftDoubleComplex* out,
    std::size_t count,
    int nkx,
    int ny,
    int nx,
    double lx) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    const int plane_index = static_cast<int>(n % static_cast<std::size_t>(nkx * ny));
    const int ih = plane_index % nkx;
    const double kx = kx_derivative_device(nx, lx, ih);
    out[n] = make_cuDoubleComplex(-kx * in[n].y, kx * in[n].x);
}

__global__ void spectral_derivative_y_kernel(
    const cufftDoubleComplex* in,
    cufftDoubleComplex* out,
    std::size_t count,
    int nkx,
    int ny,
    double ly) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    const int plane_index = static_cast<int>(n % static_cast<std::size_t>(nkx * ny));
    const int j = plane_index / nkx;
    const double ky = ky_derivative_device(ny, ly, j);
    out[n] = make_cuDoubleComplex(-ky * in[n].y, ky * in[n].x);
}

__global__ void spectral_laplacian_kernel(
    const cufftDoubleComplex* in,
    cufftDoubleComplex* out,
    std::size_t count,
    int nkx,
    int ny,
    int nx,
    double lx,
    double ly) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    const int plane_index = static_cast<int>(n % static_cast<std::size_t>(nkx * ny));
    const int j = plane_index / nkx;
    const int ih = plane_index - j * nkx;
    const double kx = kx_value_device(nx, lx, ih);
    const double ky = ky_value_device(ny, ly, j);
    out[n] = cscale(in[n], -(kx * kx + ky * ky));
}

__global__ void spectral_fortran_sharp_filter_kernel(
    cufftDoubleComplex* data,
    std::size_t count,
    int nkx,
    int ny,
    int nx,
    double lx,
    double ly,
    double filter_width) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    const int plane_index = static_cast<int>(n % static_cast<std::size_t>(nkx * ny));
    const int j = plane_index / nkx;
    const int ih = plane_index - j * nkx;
    const int signed_j = (j < static_cast<int>(round(0.5 * static_cast<double>(ny)))) ? j : j - ny;
    const double length_ratio = lx / ly;
    const int cutoff_x = static_cast<int>(round(static_cast<double>(nx) / (2.0 * filter_width)));
    const double cutoff_y = round(fabs(length_ratio) * static_cast<double>(ny) / (2.0 * filter_width));
    const bool keep = (ih < cutoff_x)
        && (fabs(static_cast<double>(signed_j) * length_ratio) < cutoff_y);
    if (!keep) {
        data[n] = make_cuDoubleComplex(0.0, 0.0);
    }
}

__global__ void w_to_center_local_kernel(
    const double* w,
    double* w_center,
    std::size_t count,
    int nx,
    int ny,
    int k_begin,
    int w_plane_begin) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const int k_local = static_cast<int>(n / plane);
    const std::size_t in_plane = n - static_cast<std::size_t>(k_local) * plane;
    const int k_global = k_begin + k_local;
    const std::size_t lower = static_cast<std::size_t>(k_global - w_plane_begin) * plane + in_plane;
    w_center[n] = 0.5 * (w[lower] + w[lower + plane]);
}

__global__ void extract_center_hdf5_layout_kernel(
    const double* field,
    double* out,
    std::size_t count,
    int nx,
    int ny,
    int k_begin,
    int plane_begin) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    const std::size_t nz_local = count / (static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny));
    const int k_local = static_cast<int>(n % nz_local);
    const std::size_t xy = n / nz_local;
    const int j = static_cast<int>(xy % static_cast<std::size_t>(ny));
    const int i = static_cast<int>(xy / static_cast<std::size_t>(ny));
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const int k_global = k_begin + k_local;
    const std::size_t source = static_cast<std::size_t>(k_global - plane_begin) * plane
        + static_cast<std::size_t>(j) * static_cast<std::size_t>(nx)
        + static_cast<std::size_t>(i);
    out[n] = field[source];
}

__global__ void extract_w_center_hdf5_layout_kernel(
    const double* w,
    double* out,
    std::size_t count,
    int nx,
    int ny,
    int k_begin,
    int w_plane_begin) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    const std::size_t nz_local = count / (static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny));
    const int k_local = static_cast<int>(n % nz_local);
    const std::size_t xy = n / nz_local;
    const int j = static_cast<int>(xy % static_cast<std::size_t>(ny));
    const int i = static_cast<int>(xy / static_cast<std::size_t>(ny));
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const int k_global = k_begin + k_local;
    const std::size_t lower = static_cast<std::size_t>(k_global - w_plane_begin) * plane
        + static_cast<std::size_t>(j) * static_cast<std::size_t>(nx)
        + static_cast<std::size_t>(i);
    out[n] = 0.5 * (w[lower] + w[lower + plane]);
}

__global__ void extract_center_hdf5_slice_kernel(
    const double* field,
    double* out,
    std::size_t count,
    int nx,
    int ny,
    int k_begin,
    int plane_begin,
    int y_index) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    const std::size_t nz_local = count / static_cast<std::size_t>(nx);
    const int k_local = static_cast<int>(n % nz_local);
    const int i = static_cast<int>(n / nz_local);
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const int k_global = k_begin + k_local;
    const std::size_t source = static_cast<std::size_t>(k_global - plane_begin) * plane
        + static_cast<std::size_t>(y_index) * static_cast<std::size_t>(nx)
        + static_cast<std::size_t>(i);
    out[n] = field[source];
}

__global__ void extract_w_center_hdf5_slice_kernel(
    const double* w,
    double* out,
    std::size_t count,
    int nx,
    int ny,
    int k_begin,
    int w_plane_begin,
    int y_index) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    const std::size_t nz_local = count / static_cast<std::size_t>(nx);
    const int k_local = static_cast<int>(n % nz_local);
    const int i = static_cast<int>(n / nz_local);
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const int k_global = k_begin + k_local;
    const std::size_t lower = static_cast<std::size_t>(k_global - w_plane_begin) * plane
        + static_cast<std::size_t>(y_index) * static_cast<std::size_t>(nx)
        + static_cast<std::size_t>(i);
    out[n] = 0.5 * (w[lower] + w[lower + plane]);
}

__global__ void center_to_face_local_kernel(
    const double* q,
    double* out,
    std::size_t count,
    int nx,
    int ny,
    int face_begin,
    int center_plane_begin,
    int nz) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const int face_local = static_cast<int>(n / plane);
    const std::size_t in_plane = n - static_cast<std::size_t>(face_local) * plane;
    const int k_global = face_begin + face_local;
    if (k_global == 0) {
        out[n] = q[static_cast<std::size_t>(0 - center_plane_begin) * plane + in_plane];
    } else if (k_global == nz) {
        out[n] = q[static_cast<std::size_t>(nz - 1 - center_plane_begin) * plane + in_plane];
    } else {
        const std::size_t upper = static_cast<std::size_t>(k_global - center_plane_begin) * plane + in_plane;
        out[n] = 0.5 * (q[upper - plane] + q[upper]);
    }
}

__global__ void ddz_center_local_kernel(
    const double* q,
    double* out,
    std::size_t count,
    int nx,
    int ny,
    int k_begin,
    int plane_begin,
    int nz,
    double inv_dz) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const int k_local = static_cast<int>(n / plane);
    const std::size_t in_plane = n - static_cast<std::size_t>(k_local) * plane;
    const int k_global = k_begin + k_local;
    const std::size_t current = static_cast<std::size_t>(k_global - plane_begin) * plane + in_plane;
    if (k_global == 0) {
        out[n] = (q[current + plane] - q[current]) * inv_dz;
    } else if (k_global == nz - 1) {
        out[n] = (q[current] - q[current - plane]) * inv_dz;
    } else {
        out[n] = (q[current + plane] - q[current - plane]) * (0.5 * inv_dz);
    }
}

__global__ void add_vertical_laplacian_center_local_kernel(
    double* lap,
    const double* q,
    std::size_t count,
    int nx,
    int ny,
    int k_begin,
    int plane_begin,
    int nz,
    double inv_dz2) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const int k_local = static_cast<int>(n / plane);
    const std::size_t in_plane = n - static_cast<std::size_t>(k_local) * plane;
    const int k_global = k_begin + k_local;
    const std::size_t current = static_cast<std::size_t>(k_global - plane_begin) * plane + in_plane;
    if (k_global == 0) {
        lap[n] += (q[current + plane] - q[current]) * inv_dz2;
    } else if (k_global == nz - 1) {
        lap[n] += (q[current - plane] - q[current]) * inv_dz2;
    } else {
        lap[n] += (q[current - plane] - 2.0 * q[current] + q[current + plane]) * inv_dz2;
    }
}

__global__ void ddz_center_to_face_local_kernel(
    const double* q,
    double* out,
    std::size_t count,
    int nx,
    int ny,
    int face_begin,
    int center_plane_begin,
    int nz,
    double inv_dz) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const int face_local = static_cast<int>(n / plane);
    const std::size_t in_plane = n - static_cast<std::size_t>(face_local) * plane;
    const int k_global = face_begin + face_local;
    if (k_global <= 0 || k_global >= nz) {
        out[n] = 0.0;
        return;
    }
    const std::size_t upper = static_cast<std::size_t>(k_global - center_plane_begin) * plane + in_plane;
    out[n] = (q[upper] - q[upper - plane]) * inv_dz;
}

__global__ void ddz_face_to_center_local_kernel(
    const double* q,
    double* out,
    std::size_t count,
    int nx,
    int ny,
    int k_begin,
    int face_plane_begin,
    double inv_dz) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const int k_local = static_cast<int>(n / plane);
    const std::size_t in_plane = n - static_cast<std::size_t>(k_local) * plane;
    const int k_global = k_begin + k_local;
    const std::size_t lower = static_cast<std::size_t>(k_global - face_plane_begin) * plane + in_plane;
    out[n] = (q[lower + plane] - q[lower]) * inv_dz;
}

__global__ void ddz_w_face_local_kernel(
    const double* w,
    double* out,
    std::size_t count,
    int nx,
    int ny,
    int face_begin,
    int w_plane_begin,
    int nz,
    double inv_dz) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const int face_local = static_cast<int>(n / plane);
    const std::size_t in_plane = n - static_cast<std::size_t>(face_local) * plane;
    const int k_global = face_begin + face_local;
    const std::size_t current = static_cast<std::size_t>(k_global - w_plane_begin) * plane + in_plane;
    if (k_global <= 0 || k_global >= nz) {
        out[n] = 0.0;
    } else {
        out[n] = (w[current + plane] - w[current - plane]) * (0.5 * inv_dz);
    }
}

__global__ void add_vertical_laplacian_face_local_kernel(
    double* lap,
    const double* w,
    std::size_t count,
    int nx,
    int ny,
    int face_begin,
    int w_plane_begin,
    int nz,
    double inv_dz2) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const int face_local = static_cast<int>(n / plane);
    const std::size_t in_plane = n - static_cast<std::size_t>(face_local) * plane;
    const int k_global = face_begin + face_local;
    const std::size_t current = static_cast<std::size_t>(k_global - w_plane_begin) * plane + in_plane;
    if (k_global <= 0 || k_global >= nz) {
        lap[n] = 0.0;
    } else {
        lap[n] += (w[current - plane] - 2.0 * w[current] + w[current + plane]) * inv_dz2;
    }
}

__global__ void build_center_rhs_kernel(
    const double* u,
    const double* v,
    const double* w_center,
    const double* dudx,
    const double* dudy,
    const double* dudz,
    const double* dvdx,
    const double* dvdy,
    const double* dvdz,
    const double* lap_u,
    const double* lap_v,
    double* rhs_u,
    double* rhs_v,
    std::size_t count,
    double nu,
    double coriolis_f,
    double geostrophic_u,
    double geostrophic_v,
    double fft_scale) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    const double dudx_real = dudx[n] * fft_scale;
    const double dudy_real = dudy[n] * fft_scale;
    const double dvdx_real = dvdx[n] * fft_scale;
    const double dvdy_real = dvdy[n] * fft_scale;
    const double lap_u_real = lap_u[n] * fft_scale;
    const double lap_v_real = lap_v[n] * fft_scale;
    double ru = -(u[n] * dudx_real + v[n] * dudy_real + w_center[n] * dudz[n]) + nu * lap_u_real;
    double rv = -(u[n] * dvdx_real + v[n] * dvdy_real + w_center[n] * dvdz[n]) + nu * lap_v_real;
    if (coriolis_f != 0.0) {
        ru += coriolis_f * (v[n] - geostrophic_v);
        rv += -coriolis_f * (u[n] - geostrophic_u);
    }
    rhs_u[n] = ru;
    rhs_v[n] = rv;
}

__global__ void build_w_rhs_kernel(
    const double* w,
    const double* u_on_w,
    const double* v_on_w,
    const double* dwdx_face,
    const double* dwdy_face,
    const double* dwdz_face,
    const double* lap_w,
    double* rhs_w,
    std::size_t count,
    int nx,
    int ny,
    int face_begin,
    int nz,
    double nu,
    double fft_scale) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const int k = face_begin + static_cast<int>(n / plane);
    if (k <= 0 || k >= nz) {
        rhs_w[n] = 0.0;
        return;
    }
    rhs_w[n] = -(u_on_w[n] * dwdx_face[n] * fft_scale + v_on_w[n] * dwdy_face[n] * fft_scale + w[n] * dwdz_face[n])
        + nu * lap_w[n] * fft_scale;
}

__global__ void build_rotational_center_rhs_kernel(
    const double* u, const double* v, const double* w,
    const double* dudy, const double* dvdx,
    const double* dudz_face, const double* dvdz_face,
    const double* dwdx_face, const double* dwdy_face,
    const double* lap_u, const double* lap_v,
    double* rhs_u, double* rhs_v, std::size_t count,
    int nx, int ny, int center_begin, int face_begin,
    double nu) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= count) return;
    const std::size_t plane = static_cast<std::size_t>(nx) * ny;
    const int k = center_begin + static_cast<int>(n / plane);
    const std::size_t in_plane = n % plane;
    const std::size_t lower = static_cast<std::size_t>(k - face_begin) * plane + in_plane;
    const std::size_t upper = lower + plane;
    const double omega_z = dvdx[n] - dudy[n];
    const double omega_y_lower = dudz_face[lower] - dwdx_face[lower];
    const double omega_y_upper = dudz_face[upper] - dwdx_face[upper];
    const double omega_x_lower = dwdy_face[lower] - dvdz_face[lower];
    const double omega_x_upper = dwdy_face[upper] - dvdz_face[upper];
    rhs_u[n] = v[n] * omega_z
        - 0.5 * (w[lower] * omega_y_lower + w[upper] * omega_y_upper)
        + nu * lap_u[n];
    rhs_v[n] = 0.5 * (w[lower] * omega_x_lower + w[upper] * omega_x_upper)
        - u[n] * omega_z + nu * lap_v[n];
}

__global__ void build_rotational_face_rhs_kernel(
    const double* u_on_w, const double* v_on_w,
    const double* dudz_face, const double* dvdz_face,
    const double* dwdx_face, const double* dwdy_face,
    const double* lap_w, double* rhs_w, std::size_t count,
    int nx, int ny, int face_begin, int nz, double nu) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= count) return;
    const std::size_t plane = static_cast<std::size_t>(nx) * ny;
    const int k = face_begin + static_cast<int>(n / plane);
    if (k <= 0 || k >= nz) {
        rhs_w[n] = 0.0;
        return;
    }
    const double omega_x = dwdy_face[n] - dvdz_face[n];
    const double omega_y = dudz_face[n] - dwdx_face[n];
    rhs_w[n] = u_on_w[n] * omega_y - v_on_w[n] * omega_x + nu * lap_w[n];
}

__global__ void build_smagorinsky_center_kernel(
    const double* dudx,
    const double* dudy,
    const double* dudz,
    const double* dvdx,
    const double* dvdy,
    const double* dvdz,
    const double* dwdx,
    const double* dwdy,
    const double* dwdz,
    double* strain,
    double* nu_t,
    double* txx,
    double* txy,
    double* tyy,
    double* tzz,
    std::size_t count,
    double coeff,
    double fft_scale) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    const double s11 = dudx[n] * fft_scale;
    const double s22 = dvdy[n] * fft_scale;
    const double s33 = dwdz[n];
    const double s12 = 0.5 * (dudy[n] * fft_scale + dvdx[n] * fft_scale);
    const double s13 = 0.5 * (dudz[n] + dwdx[n] * fft_scale);
    const double s23 = 0.5 * (dvdz[n] + dwdy[n] * fft_scale);
    const double sij_sij = s11 * s11 + s22 * s22 + s33 * s33 + 2.0 * (s12 * s12 + s13 * s13 + s23 * s23);
    const double mag = sqrt(fmax(2.0 * sij_sij, 0.0));
    const double nt = coeff * mag;
    strain[n] = mag;
    nu_t[n] = nt;
    txx[n] = 2.0 * nt * s11;
    txy[n] = 2.0 * nt * s12;
    tyy[n] = 2.0 * nt * s22;
    tzz[n] = 2.0 * nt * s33;
}

__global__ void build_smagorinsky_face_kernel(
    const double* nu_t_face,
    const double* dudz_face,
    const double* dvdz_face,
    const double* dwdx_face,
    const double* dwdy_face,
    double* txz,
    double* tyz,
    std::size_t count,
    int nx,
    int ny,
    int face_begin,
    int nz,
    double fft_scale) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const int k = face_begin + static_cast<int>(n / plane);
    if (k <= 0 || k >= nz) {
        txz[n] = 0.0;
        tyz[n] = 0.0;
        return;
    }
    txz[n] = nu_t_face[n] * (dudz_face[n] + dwdx_face[n] * fft_scale);
    tyz[n] = nu_t_face[n] * (dvdz_face[n] + dwdy_face[n] * fft_scale);
}

__global__ void add_center_sgs_divergence_kernel(
    double* rhs_u,
    double* rhs_v,
    const double* dtxx_dx,
    const double* dtxy_dy,
    const double* dtxy_dx,
    const double* dtyy_dy,
    const double* dtxz_dz,
    const double* dtyz_dz,
    std::size_t count,
    double fft_scale) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    rhs_u[n] += fft_scale * (dtxx_dx[n] + dtxy_dy[n]) + dtxz_dz[n];
    rhs_v[n] += fft_scale * (dtxy_dx[n] + dtyy_dy[n]) + dtyz_dz[n];
}

__global__ void add_face_sgs_divergence_kernel(
    double* rhs_w,
    const double* dtxz_dx,
    const double* dtyz_dy,
    const double* dtzz_dz,
    std::size_t count,
    int nx,
    int ny,
    int face_begin,
    int nz,
    double fft_scale) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const int k = face_begin + static_cast<int>(n / plane);
    if (k <= 0 || k >= nz) {
        return;
    }
    rhs_w[n] += fft_scale * (dtxz_dx[n] + dtyz_dy[n]) + dtzz_dz[n];
}

__global__ void apply_wall_stress_kernel(
    double* rhs_u,
    double* rhs_v,
    const double* u_filtered,
    const double* v_filtered,
    int nx,
    int ny,
    double inv_dz,
    double u_fric,
    double vonk,
    double denom,
    int dynamic_neutral) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    if (n >= plane) {
        return;
    }
    constexpr double eps = 1.0e-12;
    const double u0 = u_filtered[n];
    const double v0 = v_filtered[n];
    const double speed = sqrt(u0 * u0 + v0 * v0);
    if (speed <= eps || fabs(denom) <= eps) {
        return;
    }
    double ustar = u_fric;
    if (dynamic_neutral != 0) {
        ustar = speed * vonk / denom;
    }
    const double tau = -(ustar * ustar);
    rhs_u[n] += tau * u0 / speed * inv_dz;
    rhs_v[n] += tau * v0 / speed * inv_dz;
}

__global__ void build_scalar_advective_flux_kernel(
    const double* u,
    const double* v,
    const double* theta,
    double* flux_x,
    double* flux_y,
    std::size_t count) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    flux_x[n] = u[n] * theta[n];
    flux_y[n] = v[n] * theta[n];
}

__global__ void build_scalar_face_flux_kernel(
    const double* w,
    const double* theta_on_w,
    double* flux_z,
    std::size_t count) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    flux_z[n] = w[n] * theta_on_w[n];
}

__global__ void horizontal_divergence_kernel(
    const double* fx_dx,
    const double* fy_dy,
    double* div_xy,
    std::size_t count) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n < count) {
        div_xy[n] = fx_dx[n] + fy_dy[n];
    }
}

__global__ void scalar_rhs_advective_kernel(
    double* rhs,
    const double* div_xy,
    const double* div_z,
    std::size_t count) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n < count) {
        rhs[n] = -(div_xy[n] + div_z[n]);
    }
}

__global__ void scalar_kappa_kernel(
    const double* nu_t,
    const double* strain,
    const double* dtheta_dz,
    double* kappa,
    std::size_t count,
    double scalar_diffusivity,
    double prandtl_t,
    int stability_correction,
    double g,
    double theta0,
    double smag_delta2,
    double beta,
    double power) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    double diffusivity = scalar_diffusivity + nu_t[n] / prandtl_t;
    if (stability_correction != 0) {
        const double local_strain = smag_delta2 > 0.0 ? nu_t[n] / smag_delta2 : strain[n];
        const double n2 = (g / theta0) * dtheta_dz[n];
        const double ri = fmax(n2, 0.0) / fmax(local_strain * local_strain, 1.0e-24);
        diffusivity *= pow(1.0 + beta * ri, -power);
    }
    kappa[n] = diffusivity;
}

__global__ void build_scalar_diffusive_flux_kernel(
    const double* kappa,
    const double* dtheta_dx,
    const double* dtheta_dy,
    double* qx,
    double* qy,
    std::size_t count) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    qx[n] = -kappa[n] * dtheta_dx[n];
    qy[n] = -kappa[n] * dtheta_dy[n];
}

__global__ void build_scalar_diffusive_face_flux_kernel(
    const double* kappa_w,
    const double* dtheta_dz_w,
    double* qz,
    std::size_t count,
    int nx,
    int ny,
    int face_begin,
    int nz,
    double surface_theta_flux) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const int k = face_begin + static_cast<int>(n / plane);
    if (k == 0) {
        qz[n] = surface_theta_flux;
    } else if (k >= nz) {
        qz[n] = 0.0;
    } else {
        qz[n] = -kappa_w[n] * dtheta_dz_w[n];
    }
}

__global__ void add_scalar_diffusion_kernel(
    double* rhs,
    const double* div_xy,
    const double* div_z,
    std::size_t count) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n < count) {
        rhs[n] -= div_xy[n] + div_z[n];
    }
}

__global__ void add_buoyancy_kernel(
    double* rhs_w,
    const double* theta,
    const double* theta_plane_mean,
    std::size_t count,
    int nx,
    int ny,
    int face_begin,
    int theta_plane_begin,
    int nz,
    double coeff) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const int k = face_begin + static_cast<int>(n / plane);
    const std::size_t in_plane = n % plane;
    if (k <= 0 || k >= nz) {
        return;
    }
    const std::size_t upper = static_cast<std::size_t>(k - theta_plane_begin) * plane + in_plane;
    const int upper_plane = k - theta_plane_begin;
    const int lower_plane = upper_plane - 1;
    const double theta_lower = theta[upper - plane] - theta_plane_mean[lower_plane];
    const double theta_upper = theta[upper] - theta_plane_mean[upper_plane];
    rhs_w[n] += 0.5 * coeff * (theta_lower + theta_upper);
}

__global__ void plane_mean_kernel(
    const double* q,
    double* plane_mean,
    int nx,
    int ny,
    int plane_count) {
    extern __shared__ double shared[];
    const int plane_index = blockIdx.x;
    const int tid = threadIdx.x;
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    double sum = 0.0;
    for (std::size_t n = static_cast<std::size_t>(tid); n < plane; n += static_cast<std::size_t>(blockDim.x)) {
        sum += q[static_cast<std::size_t>(plane_index) * plane + n];
    }
    shared[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0 && plane_index < plane_count) {
        plane_mean[plane_index] = shared[0] / static_cast<double>(plane);
    }
}

__global__ void build_virtual_potential_temperature_kernel(
    const double* theta,
    const double* qv,
    const double* ql,
    double* theta_v,
    std::size_t count) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n < count) {
        theta_v[n] = theta[n] * (1.0 + 0.61 * qv[n] - ql[n]);
    }
}

__global__ void remove_plane_mean_and_scale_kernel(
    double* q,
    const double* plane_mean,
    std::size_t count,
    std::size_t plane,
    double scale) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n < count) {
        q[n] = scale * (q[n] - plane_mean[n / plane]);
    }
}

__device__ double amd_staggered_viscosity_device(
    const double g00, const double g01, const double g02,
    const double g10, const double g11, const double g12,
    const double g20, const double g21, const double g22,
    const double luz, const double lvz, const double lwx, const double lwy,
    const double uuz, const double uvz, const double uwx, const double uwy,
    const double dbx, const double dby, const double dbz,
    const double lx, const double ly, const double lz) {
    const double s11 = g00;
    const double s22 = g11;
    const double s33 = g22;
    const double s12 = 0.5 * (g01 + g10);
    const double ls13 = 0.5 * (luz + lwx);
    const double us13 = 0.5 * (uuz + uwx);
    const double ls23 = 0.5 * (lvz + lwy);
    const double us23 = 0.5 * (uvz + uwy);
    const double wxs13 = 0.5 * (lwx * ls13 + uwx * us13);
    const double wxs23 = 0.5 * (lwx * ls23 + uwx * us23);
    const double wys13 = 0.5 * (lwy * ls13 + uwy * us13);
    const double wys23 = 0.5 * (lwy * ls23 + uwy * us23);
    const double uzs13 = 0.5 * (luz * ls13 + uuz * us13);
    const double vzs23 = 0.5 * (lvz * ls23 + uvz * us23);
    const double wx2 = 0.5 * (lwx * lwx + uwx * uwx);
    const double wy2 = 0.5 * (lwy * lwy + uwy * uwy);
    const double uz2 = 0.5 * (luz * luz + uuz * uuz);
    const double vz2 = 0.5 * (lvz * lvz + uvz * uvz);
    const double uzvz = 0.5 * (luz * lvz + uuz * uvz);
    const double cx = s11 * g00 * g00 + s22 * g10 * g10 + s33 * wx2
        + 2.0 * s12 * g00 * g10 + 2.0 * g00 * wxs13 + 2.0 * g10 * wxs23;
    const double cy = s11 * g01 * g01 + s22 * g11 * g11 + s33 * wy2
        + 2.0 * s12 * g01 * g11 + 2.0 * g01 * wys13 + 2.0 * g11 * wys23;
    const double cz = s11 * uz2 + s22 * vz2 + s33 * g22 * g22
        + 2.0 * s12 * uzvz + 2.0 * g22 * uzs13 + 2.0 * g22 * vzs23;
    const double numerator = -lx * lx * cx - ly * ly * cy - lz * lz * cz
        + lx * lx * g20 * dbx + ly * ly * g21 * dby + lz * lz * g22 * dbz;
    const double denominator = g00 * g00 + g10 * g10 + wx2
        + g01 * g01 + g11 * g11 + wy2 + uz2 + vz2 + g22 * g22;
    if (!(denominator > 0.0) || !isfinite(numerator) || !isfinite(denominator)) {
        return 0.0;
    }
    return fmax(numerator, 0.0) / denominator;
}

__global__ void build_amd_center_stress_kernel(
    const double* dudx, const double* dudy, const double* dudz,
    const double* dvdx, const double* dvdy, const double* dvdz,
    const double* dwdx, const double* dwdy, const double* dwdz,
    const double* dudz_face, const double* dvdz_face,
    const double* dwdx_face, const double* dwdy_face,
    const double* dbdx, const double* dbdy, const double* dbdz,
    double* strain, double* nu_t,
    double* txx, double* txy, double* tyy, double* tzz,
    std::size_t count, int nx, int ny, int center_begin, int face_begin,
    double lx, double ly, double lz) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= count) return;
    const std::size_t plane = static_cast<std::size_t>(nx) * ny;
    const int k = center_begin + static_cast<int>(n / plane);
    const std::size_t in_plane = n % plane;
    const std::size_t lower = static_cast<std::size_t>(k - face_begin) * plane + in_plane;
    const std::size_t upper = lower + plane;
    const double nt = amd_staggered_viscosity_device(
        dudx[n], dudy[n], dudz[n], dvdx[n], dvdy[n], dvdz[n],
        dwdx[n], dwdy[n], dwdz[n],
        dudz_face[lower], dvdz_face[lower], dwdx_face[lower], dwdy_face[lower],
        dudz_face[upper], dvdz_face[upper], dwdx_face[upper], dwdy_face[upper],
        dbdx[n], dbdy[n], dbdz[n], lx, ly, lz);
    const double s11 = dudx[n];
    const double s22 = dvdy[n];
    const double s33 = dwdz[n];
    const double s12 = 0.5 * (dudy[n] + dvdx[n]);
    const double s13 = 0.5 * (dudz[n] + dwdx[n]);
    const double s23 = 0.5 * (dvdz[n] + dwdy[n]);
    const double sij2 = s11*s11 + s22*s22 + s33*s33
        + 2.0 * (s12*s12 + s13*s13 + s23*s23);
    strain[n] = sqrt(fmax(2.0 * sij2, 0.0));
    nu_t[n] = nt;
    txx[n] = 2.0 * nt * s11;
    txy[n] = 2.0 * nt * s12;
    tyy[n] = 2.0 * nt * s22;
    tzz[n] = 2.0 * nt * s33;
}

__global__ void amd_scalar_kappa_kernel(
    const double* dudx, const double* dudy, const double* dudz,
    const double* dvdx, const double* dvdy, const double* dvdz,
    const double* dwdx, const double* dwdy, const double* dwdz,
    const double* ds_dx, const double* ds_dy, const double* ds_dz,
    const double* ds_dz_face, double* kappa, std::size_t count,
    int nx, int ny, int center_begin, int face_begin,
    double lx, double ly, double lz, double molecular) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= count) return;
    const std::size_t plane = static_cast<std::size_t>(nx) * ny;
    const int k = center_begin + static_cast<int>(n / plane);
    const std::size_t in_plane = n % plane;
    const std::size_t lower = static_cast<std::size_t>(k - face_begin) * plane + in_plane;
    const std::size_t upper = lower + plane;
    const double sx = ds_dx[n];
    const double sy = ds_dy[n];
    const double sz = ds_dz[n];
    double numerator = 0.0;
    numerator -= lx*lx * (dudx[n]*sx*sx + dvdx[n]*sx*sy + dwdx[n]*sx*sz);
    numerator -= ly*ly * (dudy[n]*sy*sx + dvdy[n]*sy*sy + dwdy[n]*sy*sz);
    numerator -= lz*lz * (dudz[n]*sz*sx + dvdz[n]*sz*sy + dwdz[n]*sz*sz);
    const double denominator = sx*sx + sy*sy
        + 0.5 * (ds_dz_face[lower]*ds_dz_face[lower]
            + ds_dz_face[upper]*ds_dz_face[upper]);
    const double amd = denominator > 0.0 && isfinite(numerator)
        ? fmax(numerator, 0.0) / denominator : 0.0;
    kappa[n] = molecular + amd;
}

__device__ double saturation_mixing_ratio_device(double temperature, double pressure) {
    constexpr double rd = 287.04;
    constexpr double rv = 461.5;
    constexpr double t0 = 273.15;
    constexpr double e0 = 611.2;
    const double tc = temperature - t0;
    const double vapor_pressure = e0 * exp(17.67 * tc / (tc + 243.5));
    return (rd / rv) * vapor_pressure / fmax(pressure - vapor_pressure, 1.0);
}

__global__ void saturation_adjustment_kernel(
    const double* theta_l,
    const double* qt,
    const double* base_pressure,
    double* theta,
    double* qv,
    double* ql,
    std::size_t count,
    std::size_t plane,
    int center_begin) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= count) return;
    constexpr double p_ref = 100000.0;
    constexpr double rd = 287.04;
    constexpr double cp = 1004.0;
    constexpr double lv = 2.5e6;
    const int k = center_begin + static_cast<int>(n / plane);
    if (!isfinite(theta_l[n]) || !isfinite(qt[n])) {
        theta[n] = CUDART_NAN;
        qv[n] = CUDART_NAN;
        ql[n] = CUDART_NAN;
        return;
    }
    const double pressure = base_pressure[k];
    const double exner = pow(pressure / p_ref, rd / cp);
    const double dry_temperature = exner * theta_l[n];
    const double total = fmax(qt[n], 0.0);
    const double dry_qsat = saturation_mixing_ratio_device(dry_temperature, pressure);
    if (total <= dry_qsat) {
        theta[n] = theta_l[n];
        qv[n] = total;
        ql[n] = 0.0;
        return;
    }
    const double latent_over_cp = lv / cp;
    double lower = dry_temperature;
    double upper = dry_temperature + latent_over_cp * total;
    for (int iteration = 0; iteration < 48; ++iteration) {
        const double middle = 0.5 * (lower + upper);
        const double residual = middle - dry_temperature
            - latent_over_cp * (total - saturation_mixing_ratio_device(middle, pressure));
        if (residual > 0.0) upper = middle;
        else lower = middle;
    }
    const double temperature = 0.5 * (lower + upper);
    const double vapor = fmin(total, saturation_mixing_ratio_device(temperature, pressure));
    theta[n] = temperature / exner;
    qv[n] = vapor;
    ql[n] = fmax(total - vapor, 0.0);
}

__global__ void accumulate_moisture_sign_kernel(
    const double* qt, double* sums, std::size_t count) {
    extern __shared__ double shared[];
    double* negative = shared;
    double* positive = shared + blockDim.x;
    double local_negative = 0.0;
    double local_positive = 0.0;
    for (std::size_t n = static_cast<std::size_t>(threadIdx.x);
         n < count; n += static_cast<std::size_t>(blockDim.x)) {
        const double value = qt[n];
        if (!isfinite(value)) local_negative = CUDART_NAN;
        else if (value < 0.0) local_negative -= value;
        else local_positive += value;
    }
    negative[threadIdx.x] = local_negative;
    positive[threadIdx.x] = local_positive;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            negative[threadIdx.x] += negative[threadIdx.x + stride];
            positive[threadIdx.x] += positive[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        sums[0] = negative[0];
        sums[1] = positive[0];
    }
}

__global__ void apply_conservative_moisture_limiter_kernel(
    double* qt, const double* sums, std::size_t count) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= count) return;
    const double negative = sums[0];
    const double positive = sums[1];
    if (!isfinite(negative) || !isfinite(positive) || positive < negative) {
        qt[n] = CUDART_NAN;
        return;
    }
    const double scale = positive > negative && positive > 0.0
        ? (positive - negative) / positive : 0.0;
    qt[n] = qt[n] > 0.0 ? qt[n] * scale : 0.0;
}

__device__ double bomex_subsidence_device(double z) {
    if (z <= 1500.0) return -0.0065 * z / 1500.0;
    if (z <= 2100.0) return -0.0065 * (2100.0 - z) / 600.0;
    return 0.0;
}

__device__ double bomex_radiation_device(double z) {
    constexpr double seconds_per_day = 86400.0;
    if (z <= 1500.0) return -2.0 / seconds_per_day;
    if (z <= 3000.0) return (-2.0 / seconds_per_day) * (3000.0 - z) / 1500.0;
    return 0.0;
}

__device__ double bomex_moisture_advection_device(double z) {
    if (z <= 300.0) return -1.2e-8;
    if (z <= 500.0) return -1.2e-8 * (500.0 - z) / 200.0;
    return 0.0;
}

__device__ double profile_vertical_derivative_device(
    const double* profile, int k, int nz, double inv_dz) {
    if (k == 0) return (profile[1] - profile[0]) * inv_dz;
    if (k == nz - 1) return (profile[k] - profile[k - 1]) * inv_dz;
    return (profile[k + 1] - profile[k - 1]) * (0.5 * inv_dz);
}

__global__ void add_bomex_large_scale_forcing_kernel(
    double* rhs_u, double* rhs_v, double* rhs_theta_l, double* rhs_qt,
    const double* mean_u, const double* mean_v,
    const double* mean_theta_l, const double* mean_qt,
    std::size_t count, std::size_t plane, int center_begin,
    int nz, double dz) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= count) return;
    const int k = center_begin + static_cast<int>(n / plane);
    const double z = (static_cast<double>(k) + 0.5) * dz;
    const double subsidence = bomex_subsidence_device(z);
    const double inv_dz = 1.0 / dz;
    rhs_u[n] -= subsidence * profile_vertical_derivative_device(mean_u, k, nz, inv_dz);
    rhs_v[n] -= subsidence * profile_vertical_derivative_device(mean_v, k, nz, inv_dz);
    rhs_theta_l[n] += -subsidence
        * profile_vertical_derivative_device(mean_theta_l, k, nz, inv_dz)
        + bomex_radiation_device(z);
    const double mixing_ratio = fmax(mean_qt[k], 0.0);
    const double specific = mixing_ratio / (1.0 + mixing_ratio);
    const double jacobian = 1.0 / ((1.0 - specific) * (1.0 - specific));
    rhs_qt[n] += -subsidence
        * profile_vertical_derivative_device(mean_qt, k, nz, inv_dz)
        + bomex_moisture_advection_device(z) * jacobian;
}

__global__ void add_coriolis_kernel(
    double* rhs_u, double* rhs_v, const double* u, const double* v,
    std::size_t count, std::size_t plane, int center_begin,
    double dz, double coriolis_f, double geostrophic_u,
    double geostrophic_v, int bomex) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= count || coriolis_f == 0.0) return;
    const int k = center_begin + static_cast<int>(n / plane);
    const double z = (static_cast<double>(k) + 0.5) * dz;
    const double ug = bomex != 0 ? -10.0 + 1.8e-3 * z : geostrophic_u;
    const double vg = bomex != 0 ? 0.0 : geostrophic_v;
    rhs_u[n] += coriolis_f * (v[n] - vg);
    rhs_v[n] += -coriolis_f * (u[n] - ug);
}

__global__ void apply_rayleigh_sponge_center_kernel(
    double* u, double* v,
    std::size_t count, std::size_t plane, int center_begin, int plane_begin,
    double dz, double lz, double start_height, double timescale,
    double power, double dt, double geostrophic_u, double geostrophic_v,
    int bomex) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= count) return;
    const int k = center_begin + static_cast<int>(n / plane);
    const std::size_t storage = static_cast<std::size_t>(k - plane_begin) * plane + n % plane;
    const double z = (static_cast<double>(k) + 0.5) * dz;
    if (z <= start_height) return;
    const double depth = fmax(lz - start_height, dz);
    const double eta = fmin(fmax((z - start_height) / depth, 0.0), 1.0);
    const double factor = exp(-pow(eta, power) * dt / timescale);
    const double target_u = bomex != 0 ? -10.0 + 1.8e-3 * z : geostrophic_u;
    const double target_v = bomex != 0 ? 0.0 : geostrophic_v;
    u[storage] = target_u + (u[storage] - target_u) * factor;
    v[storage] = target_v + (v[storage] - target_v) * factor;
}

__global__ void apply_rayleigh_sponge_face_kernel(
    double* w, std::size_t count, std::size_t plane, int face_begin,
    int w_plane_begin,
    int nz, double dz, double lz, double start_height,
    double timescale, double power, double dt) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= count) return;
    const int k = face_begin + static_cast<int>(n / plane);
    if (k <= 0 || k >= nz) return;
    const double z = static_cast<double>(k) * dz;
    if (z <= start_height) return;
    const double depth = fmax(lz - start_height, dz);
    const double eta = fmin(fmax((z - start_height) / depth, 0.0), 1.0);
    const std::size_t storage = static_cast<std::size_t>(k - w_plane_begin) * plane + n % plane;
    w[storage] *= exp(-pow(eta, power) * dt / timescale);
}

__global__ void advance_local_kernel(
    double* field,
    const double* rhs,
    double* rhs_prev,
    double* rhs_prev2,
    std::size_t count,
    int nx,
    int ny,
    int owned_begin,
    int plane_begin,
    int time_scheme,
    int step_count,
    double dt) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    double tendency = rhs[n];
    if (time_scheme == 3 && step_count >= 2) {
        tendency = (23.0 * rhs[n] - 16.0 * rhs_prev[n] + 5.0 * rhs_prev2[n]) / 12.0;
    } else if ((time_scheme == 2 && step_count >= 1)
        || (time_scheme == 3 && step_count == 1)) {
        tendency = 1.5 * rhs[n] - 0.5 * rhs_prev[n];
    }
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const int k_local = static_cast<int>(n / plane);
    const std::size_t in_plane = n - static_cast<std::size_t>(k_local) * plane;
    const int k_global = owned_begin + k_local;
    field[static_cast<std::size_t>(k_global - plane_begin) * plane + in_plane] += dt * tendency;
    rhs_prev2[n] = rhs_prev[n];
    rhs_prev[n] = rhs[n];
}

__global__ void enforce_walls_local_kernel(double* w, int nx, int ny, int nz, int plane_begin, int plane_count) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    if (n >= plane) {
        return;
    }
    if (plane_begin == 0) {
        w[n] = 0.0;
    }
    if (plane_begin + plane_count == nz + 1) {
        w[static_cast<std::size_t>(nz - plane_begin) * plane + n] = 0.0;
    }
}

__global__ void pack_plane_kernel(const double* field, double* plane, std::size_t plane_stride, int local_plane) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n < plane_stride) {
        plane[n] = field[static_cast<std::size_t>(local_plane) * plane_stride + n];
    }
}

__global__ void unpack_plane_kernel(double* field, const double* plane, std::size_t plane_stride, int local_plane) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n < plane_stride) {
        field[static_cast<std::size_t>(local_plane) * plane_stride + n] = plane[n];
    }
}

__global__ void extract_pressure_inputs_kernel(
    const double* u,
    const double* v,
    const double* w,
    double* u_owned,
    double* v_owned,
    double* dwdz_owned,
    std::size_t count,
    int nx,
    int ny,
    int k_begin,
    int u_plane_begin,
    int w_plane_begin,
    double inv_dz) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const int k_local = static_cast<int>(n / plane);
    const std::size_t in_plane = n - static_cast<std::size_t>(k_local) * plane;
    const int k_global = k_begin + k_local;
    const std::size_t center_index = static_cast<std::size_t>(k_global - u_plane_begin) * plane + in_plane;
    const std::size_t face_lower = static_cast<std::size_t>(k_global - w_plane_begin) * plane + in_plane;
    const std::size_t face_upper = static_cast<std::size_t>(k_global + 1 - w_plane_begin) * plane + in_plane;
    u_owned[n] = u[center_index];
    v_owned[n] = v[center_index];
    dwdz_owned[n] = (w[face_upper] - w[face_lower]) * inv_dz;
}

__global__ void spectral_divergence_kernel(
    const cufftDoubleComplex* u_hat,
    const cufftDoubleComplex* v_hat,
    const cufftDoubleComplex* dwdz_hat,
    cufftDoubleComplex* div_hat,
    std::size_t count,
    int nkx,
    int ny,
    int nx,
    double lx,
    double ly) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    const int plane_index = static_cast<int>(n % static_cast<std::size_t>(nkx * ny));
    const int j = plane_index / nkx;
    const int ih = plane_index - j * nkx;
    const double kx = kx_derivative_device(nx, lx, ih);
    const double ky = ky_derivative_device(ny, ly, j);
    const cufftDoubleComplex ux = make_cuDoubleComplex(-kx * u_hat[n].y, kx * u_hat[n].x);
    const cufftDoubleComplex vy = make_cuDoubleComplex(-ky * v_hat[n].y, ky * v_hat[n].x);
    div_hat[n] = make_cuDoubleComplex(ux.x + vy.x + dwdz_hat[n].x, ux.y + vy.y + dwdz_hat[n].y);
}

__global__ void pack_z_slab_to_y_pencil_kernel(
    const cufftDoubleComplex* local_hat,
    cufftDoubleComplex* send,
    std::size_t total,
    int size,
    int k_count,
    int ny,
    int nkx,
    int nj) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= total) {
        return;
    }
    const std::size_t chunk = static_cast<std::size_t>(k_count) * static_cast<std::size_t>(nj) * static_cast<std::size_t>(nkx);
    const int dest = static_cast<int>(n / chunk);
    if (dest >= size) {
        return;
    }
    const std::size_t rem = n - static_cast<std::size_t>(dest) * chunk;
    const int ih = static_cast<int>(rem % static_cast<std::size_t>(nkx));
    const int j_local = static_cast<int>((rem / static_cast<std::size_t>(nkx)) % static_cast<std::size_t>(nj));
    const int k_local = static_cast<int>(rem / static_cast<std::size_t>(nkx * nj));
    const int j_global = dest * nj + j_local;
    send[n] = local_hat[(static_cast<std::size_t>(k_local) * static_cast<std::size_t>(ny) + static_cast<std::size_t>(j_global))
        * static_cast<std::size_t>(nkx) + static_cast<std::size_t>(ih)];
}

__global__ void unpack_y_pencil_kernel(
    const cufftDoubleComplex* recv,
    cufftDoubleComplex* y_pencil,
    std::size_t total,
    int k_count,
    int nkx,
    int nj) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= total) {
        return;
    }
    const std::size_t chunk = static_cast<std::size_t>(k_count) * static_cast<std::size_t>(nj) * static_cast<std::size_t>(nkx);
    const int source = static_cast<int>(n / chunk);
    const std::size_t rem = n - static_cast<std::size_t>(source) * chunk;
    const int ih = static_cast<int>(rem % static_cast<std::size_t>(nkx));
    const int j_local = static_cast<int>((rem / static_cast<std::size_t>(nkx)) % static_cast<std::size_t>(nj));
    const int k_local = static_cast<int>(rem / static_cast<std::size_t>(nkx * nj));
    const int k_global = source * k_count + k_local;
    y_pencil[(static_cast<std::size_t>(k_global) * static_cast<std::size_t>(nj) + static_cast<std::size_t>(j_local))
        * static_cast<std::size_t>(nkx) + static_cast<std::size_t>(ih)] = recv[n];
}

__global__ void pressure_solve_y_pencil_kernel(
    cufftDoubleComplex* y_pencil,
    cufftDoubleComplex* cp_store,
    cufftDoubleComplex* dp_store,
    int columns,
    int nkx,
    int nj,
    int ny,
    int nz,
    int nx,
    int rank,
    double lx,
    double ly,
    double inv_dz2,
    double inv_dt) {
    const int col = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= columns) {
        return;
    }
    const int ih = col % nkx;
    const int j_local = col / nkx;
    const int j_global = rank * nj + j_local;
    const double kx = kx_derivative_device(nx, lx, ih);
    const double ky = ky_derivative_device(ny, ly, j_global);
    const double kh2 = kx * kx + ky * ky;
    const bool zero_pressure_mode = kh2 == 0.0;

    cufftDoubleComplex cp_prev = make_cuDoubleComplex(0.0, 0.0);
    cufftDoubleComplex dp_prev = make_cuDoubleComplex(0.0, 0.0);
    for (int k = 0; k < nz; ++k) {
        cufftDoubleComplex lower = make_cuDoubleComplex(0.0, 0.0);
        cufftDoubleComplex diag = make_cuDoubleComplex(0.0, 0.0);
        cufftDoubleComplex upper = make_cuDoubleComplex(0.0, 0.0);
        if (zero_pressure_mode) {
            if (k == 0) {
                diag = make_cuDoubleComplex(1.0, 0.0);
            } else if (k == nz - 1) {
                lower = make_cuDoubleComplex(inv_dz2, 0.0);
                diag = make_cuDoubleComplex(-inv_dz2, 0.0);
            } else {
                lower = make_cuDoubleComplex(inv_dz2, 0.0);
                diag = make_cuDoubleComplex(-2.0 * inv_dz2, 0.0);
                upper = make_cuDoubleComplex(inv_dz2, 0.0);
            }
        } else {
            if (k == 0) {
                diag = make_cuDoubleComplex(-inv_dz2 - kh2, 0.0);
                upper = make_cuDoubleComplex(inv_dz2, 0.0);
            } else if (k == nz - 1) {
                lower = make_cuDoubleComplex(inv_dz2, 0.0);
                diag = make_cuDoubleComplex(-inv_dz2 - kh2, 0.0);
            } else {
                lower = make_cuDoubleComplex(inv_dz2, 0.0);
                diag = make_cuDoubleComplex(-2.0 * inv_dz2 - kh2, 0.0);
                upper = make_cuDoubleComplex(inv_dz2, 0.0);
            }
        }
        const std::size_t id = (static_cast<std::size_t>(k) * static_cast<std::size_t>(nj) + static_cast<std::size_t>(j_local))
            * static_cast<std::size_t>(nkx) + static_cast<std::size_t>(ih);
        const cufftDoubleComplex denom = k == 0 ? diag : csub(diag, cmul(lower, cp_prev));
        cufftDoubleComplex rhs = cscale(y_pencil[id], inv_dt);
        if (zero_pressure_mode && k == 0) {
            rhs = make_cuDoubleComplex(0.0, 0.0);
        }
        const cufftDoubleComplex cp = (k == nz - 1) ? make_cuDoubleComplex(0.0, 0.0) : cdiv(upper, denom);
        const cufftDoubleComplex dp = k == 0 ? cdiv(rhs, denom) : cdiv(csub(rhs, cmul(lower, dp_prev)), denom);
        cp_store[id] = cp;
        dp_store[id] = dp;
        cp_prev = cp;
        dp_prev = dp;
    }

    cufftDoubleComplex solution = dp_prev;
    for (int k = nz - 1; k >= 0; --k) {
        const std::size_t id = (static_cast<std::size_t>(k) * static_cast<std::size_t>(nj) + static_cast<std::size_t>(j_local))
            * static_cast<std::size_t>(nkx) + static_cast<std::size_t>(ih);
        if (k != nz - 1) {
            solution = csub(dp_store[id], cmul(cp_store[id], solution));
        }
        y_pencil[id] = solution;
    }
}

__global__ void pack_y_pencil_to_z_slab_kernel(
    const cufftDoubleComplex* y_pencil,
    cufftDoubleComplex* send,
    std::size_t total,
    int k_count,
    int nkx,
    int nj) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= total) {
        return;
    }
    const std::size_t chunk = static_cast<std::size_t>(k_count) * static_cast<std::size_t>(nj) * static_cast<std::size_t>(nkx);
    const int dest = static_cast<int>(n / chunk);
    const std::size_t rem = n - static_cast<std::size_t>(dest) * chunk;
    const int ih = static_cast<int>(rem % static_cast<std::size_t>(nkx));
    const int j_local = static_cast<int>((rem / static_cast<std::size_t>(nkx)) % static_cast<std::size_t>(nj));
    const int k_local = static_cast<int>(rem / static_cast<std::size_t>(nkx * nj));
    const int k_global = dest * k_count + k_local;
    send[n] = y_pencil[(static_cast<std::size_t>(k_global) * static_cast<std::size_t>(nj) + static_cast<std::size_t>(j_local))
        * static_cast<std::size_t>(nkx) + static_cast<std::size_t>(ih)];
}

__global__ void unpack_z_slab_kernel(
    const cufftDoubleComplex* recv,
    cufftDoubleComplex* local_hat,
    std::size_t total,
    int k_count,
    int ny,
    int nkx,
    int nj) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= total) {
        return;
    }
    const std::size_t chunk = static_cast<std::size_t>(k_count) * static_cast<std::size_t>(nj) * static_cast<std::size_t>(nkx);
    const int source = static_cast<int>(n / chunk);
    const std::size_t rem = n - static_cast<std::size_t>(source) * chunk;
    const int ih = static_cast<int>(rem % static_cast<std::size_t>(nkx));
    const int j_local = static_cast<int>((rem / static_cast<std::size_t>(nkx)) % static_cast<std::size_t>(nj));
    const int k_local = static_cast<int>(rem / static_cast<std::size_t>(nkx * nj));
    const int j_global = source * nj + j_local;
    local_hat[(static_cast<std::size_t>(k_local) * static_cast<std::size_t>(ny) + static_cast<std::size_t>(j_global))
        * static_cast<std::size_t>(nkx) + static_cast<std::size_t>(ih)] = recv[n];
}

__global__ void spectral_pressure_gradient_kernel(
    const cufftDoubleComplex* p_hat,
    cufftDoubleComplex* dpdx_hat,
    cufftDoubleComplex* dpdy_hat,
    std::size_t count,
    int nkx,
    int ny,
    int nx,
    double lx,
    double ly) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    const int plane_index = static_cast<int>(n % static_cast<std::size_t>(nkx * ny));
    const int j = plane_index / nkx;
    const int ih = plane_index - j * nkx;
    const double kx = kx_derivative_device(nx, lx, ih);
    const double ky = ky_derivative_device(ny, ly, j);
    dpdx_hat[n] = make_cuDoubleComplex(-kx * p_hat[n].y, kx * p_hat[n].x);
    dpdy_hat[n] = make_cuDoubleComplex(-ky * p_hat[n].y, ky * p_hat[n].x);
}

__global__ void apply_center_projection_kernel(
    double* u,
    double* v,
    double* p,
    const double* p_owned,
    const double* dpdx,
    const double* dpdy,
    std::size_t count,
    int nx,
    int ny,
    int k_begin,
    int plane_begin,
    double dt,
    double fft_scale) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const int k_local = static_cast<int>(n / plane);
    const std::size_t in_plane = n - static_cast<std::size_t>(k_local) * plane;
    const int k_global = k_begin + k_local;
    const std::size_t local = static_cast<std::size_t>(k_global - plane_begin) * plane + in_plane;
    p[local] = p_owned[n] * fft_scale;
    u[local] -= dt * dpdx[n] * fft_scale;
    v[local] -= dt * dpdy[n] * fft_scale;
}

__global__ void apply_vertical_projection_kernel(
    double* w,
    const double* p,
    std::size_t count,
    int nx,
    int ny,
    int face_begin,
    int w_plane_begin,
    int p_plane_begin,
    int nz,
    double dt,
    double inv_dz) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const int face_local = static_cast<int>(n / plane);
    const std::size_t in_plane = n - static_cast<std::size_t>(face_local) * plane;
    const int k_global = face_begin + face_local;
    const std::size_t w_index = static_cast<std::size_t>(k_global - w_plane_begin) * plane + in_plane;
    if (k_global <= 0 || k_global >= nz) {
        w[w_index] = 0.0;
        return;
    }
    const std::size_t p_upper = static_cast<std::size_t>(k_global - p_plane_begin) * plane + in_plane;
    const std::size_t p_lower = static_cast<std::size_t>(k_global - 1 - p_plane_begin) * plane + in_plane;
    w[w_index] -= dt * (p[p_upper] - p[p_lower]) * inv_dz;
}

void mpi_alltoall_device_complex(DeviceBuffer<cufftDoubleComplex>& send, DeviceBuffer<cufftDoubleComplex>& recv, std::size_t chunk, MPI_Comm comm) {
    if (chunk > static_cast<std::size_t>(std::numeric_limits<int>::max() / 2)) {
        throw std::runtime_error("CUDA-MPI all-to-all chunk exceeds MPI int count limit");
    }
    MPI_Request request = MPI_REQUEST_NULL;
    check_mpi(MPI_Ialltoall(
                  reinterpret_cast<const double*>(send.data()),
                  static_cast<int>(2 * chunk),
                  MPI_DOUBLE,
                  reinterpret_cast<double*>(recv.data()),
                  static_cast<int>(2 * chunk),
                  MPI_DOUBLE,
                  comm,
                  &request),
        "Ialltoall");
    check_mpi(MPI_Wait(&request, MPI_STATUS_IGNORE), "Wait");
}

void exchange_field_halo(DeviceLocalField& field, int tag_base, const Params& params, const Slab& slab, MPI_Comm comm, HaloScratch& scratch) {
    constexpr int threads = 256;
    const int lower_rank = slab.rank > 0 ? slab.rank - 1 : MPI_PROC_NULL;
    const int upper_rank = slab.rank + 1 < slab.size ? slab.rank + 1 : MPI_PROC_NULL;
    const bool recv_lower = lower_rank != MPI_PROC_NULL && field.owned_begin > 0;
    const bool recv_upper = upper_rank != MPI_PROC_NULL && field.owned_begin + field.owned_count < field.total_planes;
    const bool send_lower = lower_rank != MPI_PROC_NULL && field.owned_count > 0;
    const bool send_upper = upper_rank != MPI_PROC_NULL && field.owned_count > 0;

    std::vector<MPI_Request> requests;
    requests.reserve(4);
    MPI_Request request = MPI_REQUEST_NULL;
    if (recv_lower) {
        check_mpi(MPI_Irecv(scratch.recv_lower.data(), static_cast<int>(field.plane_stride), MPI_DOUBLE, lower_rank, tag_base + 1, comm, &request), "Irecv lower halo");
        requests.push_back(request);
    }
    if (recv_upper) {
        check_mpi(MPI_Irecv(scratch.recv_upper.data(), static_cast<int>(field.plane_stride), MPI_DOUBLE, upper_rank, tag_base + 0, comm, &request), "Irecv upper halo");
        requests.push_back(request);
    }
    if (send_lower) {
        const int send_plane = field.owned_begin - field.plane_begin;
        pack_plane_kernel<<<blocks_for(field.plane_stride, threads), threads>>>(field.values.data(), scratch.send_lower.data(), field.plane_stride, send_plane);
        check_cuda(cudaGetLastError(), "lower halo pack kernel launch");
    }
    if (send_upper) {
        const int send_plane = field.owned_begin + field.owned_count - 1 - field.plane_begin;
        pack_plane_kernel<<<blocks_for(field.plane_stride, threads), threads>>>(field.values.data(), scratch.send_upper.data(), field.plane_stride, send_plane);
        check_cuda(cudaGetLastError(), "upper halo pack kernel launch");
    }
    check_cuda(cudaDeviceSynchronize(), "halo pack completion");
    if (send_lower) {
        check_mpi(MPI_Isend(scratch.send_lower.data(), static_cast<int>(field.plane_stride), MPI_DOUBLE, lower_rank, tag_base + 0, comm, &request), "Isend lower halo");
        requests.push_back(request);
    }
    if (send_upper) {
        check_mpi(MPI_Isend(scratch.send_upper.data(), static_cast<int>(field.plane_stride), MPI_DOUBLE, upper_rank, tag_base + 1, comm, &request), "Isend upper halo");
        requests.push_back(request);
    }
    if (!requests.empty()) {
        check_mpi(MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE), "Waitall halo exchange");
    }
    if (recv_lower) {
        const int target_plane = field.owned_begin - 1 - field.plane_begin;
        unpack_plane_kernel<<<blocks_for(field.plane_stride, threads), threads>>>(field.values.data(), scratch.recv_lower.data(), field.plane_stride, target_plane);
        check_cuda(cudaGetLastError(), "lower halo unpack kernel launch");
    }
    if (recv_upper) {
        const int target_plane = field.owned_begin + field.owned_count - field.plane_begin;
        unpack_plane_kernel<<<blocks_for(field.plane_stride, threads), threads>>>(field.values.data(), scratch.recv_upper.data(), field.plane_stride, target_plane);
        check_cuda(cudaGetLastError(), "upper halo unpack kernel launch");
    }
    check_cuda(cudaDeviceSynchronize(), "halo unpack completion");
}

void exchange_state_halos(CudaMpiState& state, const Params& params, const Slab& slab, MPI_Comm comm, HaloScratch& scratch) {
    exchange_field_halo(state.u, 100, params, slab, comm, scratch);
    exchange_field_halo(state.v, 110, params, slab, comm, scratch);
    exchange_field_halo(state.w, 120, params, slab, comm, scratch);
    if (params.thermo_enabled) {
        exchange_field_halo(state.theta, 130, params, slab, comm, scratch);
        if (params.moisture_enabled) {
            exchange_field_halo(state.theta_l, 140, params, slab, comm, scratch);
            exchange_field_halo(state.qt, 150, params, slab, comm, scratch);
            exchange_field_halo(state.qv, 160, params, slab, comm, scratch);
            exchange_field_halo(state.ql, 170, params, slab, comm, scratch);
        }
    }
}

void cuda_mpi_project(CudaMpiState& state, const Params& params, const Slab& slab, MPI_Comm comm, HaloScratch& halo, PressureWorkspace& pressure);
void apply_wall_stress(CudaMpiState& state, RhsWorkspace& work, const Params& params, const Slab& slab);
void compute_scalar_rhs(CudaMpiState& state, RhsWorkspace& work, const Params& params, const Slab& slab, MPI_Comm comm, HaloScratch& halo);

void inverse_center_scaled(RhsWorkspace& work, DeviceBuffer<cufftDoubleComplex>& spec, DeviceBuffer<double>& out, const Params& params) {
    constexpr int threads = 256;
    check_cufft(cufftExecZ2D(work.c2r_center.handle, spec.data(), out.data()), "RHS center inverse transform");
    scale_real_kernel<<<blocks_for(work.center_count, threads), threads>>>(out.data(), work.center_count, 1.0 / static_cast<double>(params.nx * params.ny));
    check_cuda(cudaGetLastError(), "RHS center inverse scaling kernel launch");
}

void inverse_face_scaled(RhsWorkspace& work, DeviceBuffer<cufftDoubleComplex>& spec, DeviceBuffer<double>& out, const Params& params) {
    constexpr int threads = 256;
    check_cufft(cufftExecZ2D(work.c2r_face.handle, spec.data(), out.data()), "RHS face inverse transform");
    scale_real_kernel<<<blocks_for(work.face_count, threads), threads>>>(out.data(), work.face_count, 1.0 / static_cast<double>(params.nx * params.ny));
    check_cuda(cudaGetLastError(), "RHS face inverse scaling kernel launch");
}

void center_horizontal_derivatives(
    RhsWorkspace& work,
    DeviceBuffer<double>& q,
    DeviceBuffer<double>& dx,
    DeviceBuffer<double>& dy,
    const Params& params) {
    constexpr int threads = 256;
    check_cufft(cufftExecD2Z(work.r2c_center.handle, q.data(), work.center_hat.data()), "RHS center forward transform");
    spectral_derivative_x_kernel<<<blocks_for(work.center_spectral_count, threads), threads>>>(
        work.center_hat.data(), work.center_hat_scratch.data(), work.center_spectral_count, params.nkx(), params.ny, params.nx, params.lx);
    check_cuda(cudaGetLastError(), "RHS center spectral dx kernel launch");
    inverse_center_scaled(work, work.center_hat_scratch, dx, params);
    spectral_derivative_y_kernel<<<blocks_for(work.center_spectral_count, threads), threads>>>(
        work.center_hat.data(), work.center_hat_scratch.data(), work.center_spectral_count, params.nkx(), params.ny, params.ly);
    check_cuda(cudaGetLastError(), "RHS center spectral dy kernel launch");
    inverse_center_scaled(work, work.center_hat_scratch, dy, params);
}

void center_horizontal_derivatives_and_laplacian(
    RhsWorkspace& work,
    DeviceBuffer<double>& q,
    DeviceBuffer<double>& dx,
    DeviceBuffer<double>& dy,
    DeviceBuffer<double>& lap,
    const Params& params) {
    constexpr int threads = 256;
    check_cufft(cufftExecD2Z(work.r2c_center.handle, q.data(), work.center_hat.data()), "RHS center forward transform");
    spectral_derivative_x_kernel<<<blocks_for(work.center_spectral_count, threads), threads>>>(
        work.center_hat.data(), work.center_hat_scratch.data(), work.center_spectral_count, params.nkx(), params.ny, params.nx, params.lx);
    check_cuda(cudaGetLastError(), "RHS center spectral dx kernel launch");
    inverse_center_scaled(work, work.center_hat_scratch, dx, params);
    spectral_derivative_y_kernel<<<blocks_for(work.center_spectral_count, threads), threads>>>(
        work.center_hat.data(), work.center_hat_scratch.data(), work.center_spectral_count, params.nkx(), params.ny, params.ly);
    check_cuda(cudaGetLastError(), "RHS center spectral dy kernel launch");
    inverse_center_scaled(work, work.center_hat_scratch, dy, params);
    spectral_laplacian_kernel<<<blocks_for(work.center_spectral_count, threads), threads>>>(
        work.center_hat.data(), work.center_hat_scratch.data(), work.center_spectral_count, params.nkx(), params.ny, params.nx, params.lx, params.ly);
    check_cuda(cudaGetLastError(), "RHS center spectral laplacian kernel launch");
    inverse_center_scaled(work, work.center_hat_scratch, lap, params);
}

void center_horizontal_derivative_x(RhsWorkspace& work, DeviceBuffer<double>& q, DeviceBuffer<double>& dx, const Params& params) {
    constexpr int threads = 256;
    check_cufft(cufftExecD2Z(work.r2c_center.handle, q.data(), work.center_hat.data()), "RHS center forward transform");
    spectral_derivative_x_kernel<<<blocks_for(work.center_spectral_count, threads), threads>>>(
        work.center_hat.data(), work.center_hat_scratch.data(), work.center_spectral_count, params.nkx(), params.ny, params.nx, params.lx);
    check_cuda(cudaGetLastError(), "RHS center spectral dx kernel launch");
    inverse_center_scaled(work, work.center_hat_scratch, dx, params);
}

void center_horizontal_derivative_y(RhsWorkspace& work, DeviceBuffer<double>& q, DeviceBuffer<double>& dy, const Params& params) {
    constexpr int threads = 256;
    check_cufft(cufftExecD2Z(work.r2c_center.handle, q.data(), work.center_hat.data()), "RHS center forward transform");
    spectral_derivative_y_kernel<<<blocks_for(work.center_spectral_count, threads), threads>>>(
        work.center_hat.data(), work.center_hat_scratch.data(), work.center_spectral_count, params.nkx(), params.ny, params.ly);
    check_cuda(cudaGetLastError(), "RHS center spectral dy kernel launch");
    inverse_center_scaled(work, work.center_hat_scratch, dy, params);
}

void face_horizontal_derivatives_and_laplacian(
    RhsWorkspace& work,
    DeviceBuffer<double>& q,
    DeviceBuffer<double>& dx,
    DeviceBuffer<double>& dy,
    DeviceBuffer<double>& lap,
    const Params& params) {
    constexpr int threads = 256;
    check_cufft(cufftExecD2Z(work.r2c_face.handle, q.data(), work.face_hat.data()), "RHS face forward transform");
    spectral_derivative_x_kernel<<<blocks_for(work.face_spectral_count, threads), threads>>>(
        work.face_hat.data(), work.face_hat_scratch.data(), work.face_spectral_count, params.nkx(), params.ny, params.nx, params.lx);
    check_cuda(cudaGetLastError(), "RHS face spectral dx kernel launch");
    inverse_face_scaled(work, work.face_hat_scratch, dx, params);
    spectral_derivative_y_kernel<<<blocks_for(work.face_spectral_count, threads), threads>>>(
        work.face_hat.data(), work.face_hat_scratch.data(), work.face_spectral_count, params.nkx(), params.ny, params.ly);
    check_cuda(cudaGetLastError(), "RHS face spectral dy kernel launch");
    inverse_face_scaled(work, work.face_hat_scratch, dy, params);
    spectral_laplacian_kernel<<<blocks_for(work.face_spectral_count, threads), threads>>>(
        work.face_hat.data(), work.face_hat_scratch.data(), work.face_spectral_count, params.nkx(), params.ny, params.nx, params.lx, params.ly);
    check_cuda(cudaGetLastError(), "RHS face spectral laplacian kernel launch");
    inverse_face_scaled(work, work.face_hat_scratch, lap, params);
}

void face_horizontal_derivative_x(RhsWorkspace& work, DeviceBuffer<double>& q, DeviceBuffer<double>& dx, const Params& params) {
    constexpr int threads = 256;
    check_cufft(cufftExecD2Z(work.r2c_face.handle, q.data(), work.face_hat.data()), "RHS face forward transform");
    spectral_derivative_x_kernel<<<blocks_for(work.face_spectral_count, threads), threads>>>(
        work.face_hat.data(), work.face_hat_scratch.data(), work.face_spectral_count, params.nkx(), params.ny, params.nx, params.lx);
    check_cuda(cudaGetLastError(), "RHS face spectral dx kernel launch");
    inverse_face_scaled(work, work.face_hat_scratch, dx, params);
}

void face_horizontal_derivative_y(RhsWorkspace& work, DeviceBuffer<double>& q, DeviceBuffer<double>& dy, const Params& params) {
    constexpr int threads = 256;
    check_cufft(cufftExecD2Z(work.r2c_face.handle, q.data(), work.face_hat.data()), "RHS face forward transform");
    spectral_derivative_y_kernel<<<blocks_for(work.face_spectral_count, threads), threads>>>(
        work.face_hat.data(), work.face_hat_scratch.data(), work.face_spectral_count, params.nkx(), params.ny, params.ly);
    check_cuda(cudaGetLastError(), "RHS face spectral dy kernel launch");
    inverse_face_scaled(work, work.face_hat_scratch, dy, params);
}

void filter_center_owned(RhsWorkspace& work, DeviceBuffer<double>& q, DeviceBuffer<double>& out, const Params& params, double filter_width) {
    constexpr int threads = 256;
    check_cufft(cufftExecD2Z(work.r2c_center.handle, q.data(), work.center_hat.data()), "RHS center filter forward transform");
    spectral_fortran_sharp_filter_kernel<<<blocks_for(work.center_spectral_count, threads), threads>>>(
        work.center_hat.data(), work.center_spectral_count, params.nkx(), params.ny, params.nx,
        params.lx, params.ly, filter_width);
    check_cuda(cudaGetLastError(), "RHS center spectral filter kernel launch");
    inverse_center_scaled(work, work.center_hat, out, params);
}

void filter_face_owned(RhsWorkspace& work, DeviceBuffer<double>& q, DeviceBuffer<double>& out, const Params& params, double filter_width) {
    constexpr int threads = 256;
    check_cufft(cufftExecD2Z(work.r2c_face.handle, q.data(), work.face_hat.data()), "RHS face filter forward transform");
    spectral_fortran_sharp_filter_kernel<<<blocks_for(work.face_spectral_count, threads), threads>>>(
        work.face_hat.data(), work.face_spectral_count, params.nkx(), params.ny, params.nx,
        params.lx, params.ly, filter_width);
    check_cuda(cudaGetLastError(), "RHS face spectral filter kernel launch");
    inverse_face_scaled(work, work.face_hat, out, params);
}

void scatter_center_owned(DeviceBuffer<double>& owned, DeviceLocalField& field, const Params& params) {
    constexpr int threads = 256;
    scatter_owned_to_local_kernel<<<blocks_for(owned.size(), threads), threads>>>(
        owned.data(), field.values.data(), owned.size(), params.nx, params.ny, field.owned_begin, field.plane_begin);
    check_cuda(cudaGetLastError(), "scatter center owned kernel launch");
}

void scatter_face_owned(DeviceBuffer<double>& owned, DeviceLocalField& field, const Params& params) {
    constexpr int threads = 256;
    scatter_owned_to_local_kernel<<<blocks_for(owned.size(), threads), threads>>>(
        owned.data(), field.values.data(), owned.size(), params.nx, params.ny, field.owned_begin, field.plane_begin);
    check_cuda(cudaGetLastError(), "scatter face owned kernel launch");
}

void apply_viscosity_sgs(CudaMpiState& state, RhsWorkspace& work, const Params& params, const Slab& slab, MPI_Comm comm, HaloScratch& halo) {
    constexpr int threads = 256;
    const double inv_dz = 1.0 / params.dz();
    const double length = params.smagorinsky_cs * params.sgs_delta();
    const double coeff = length * length;

    center_horizontal_derivatives(work, work.w_center, work.dwdx_center, work.dwdy_center, params);
    ddz_face_to_center_local_kernel<<<blocks_for(work.center_count, threads), threads>>>(
        state.w.values.data(), work.dwdz_center.data(), work.center_count, params.nx, params.ny, slab.k_begin, state.w.plane_begin, inv_dz);
    check_cuda(cudaGetLastError(), "SGS dwdz center kernel launch");
    ddz_center_to_face_local_kernel<<<blocks_for(work.face_count, threads), threads>>>(
        state.u.values.data(), work.dudz_face.data(), work.face_count, params.nx, params.ny, slab.face_begin, state.u.plane_begin, params.nz, inv_dz);
    check_cuda(cudaGetLastError(), "SGS dudz face kernel launch");
    ddz_center_to_face_local_kernel<<<blocks_for(work.face_count, threads), threads>>>(
        state.v.values.data(), work.dvdz_face.data(), work.face_count, params.nx, params.ny, slab.face_begin, state.v.plane_begin, params.nz, inv_dz);
    check_cuda(cudaGetLastError(), "SGS dvdz face kernel launch");
    if (params.sgs_model == "amd" || params.sgs_model == "amd_plane_dissipation") {
        if (params.thermo_enabled) {
            if (params.moisture_enabled) {
                build_virtual_potential_temperature_kernel<<<blocks_for(work.center_count, threads), threads>>>(
                    state.theta.values.data(), state.qv.values.data(), state.ql.values.data(),
                    work.theta.data(), work.center_count);
            } else {
                copy_owned_from_local_kernel<<<blocks_for(work.center_count, threads), threads>>>(
                    state.theta.values.data(), work.theta.data(), work.center_count,
                    params.nx, params.ny, slab.k_begin, state.theta.plane_begin);
            }
            check_cuda(cudaGetLastError(), "AMD virtual potential temperature kernel launch");
            plane_mean_kernel<<<slab.k_count, threads, static_cast<std::size_t>(threads) * sizeof(double)>>>(
                work.theta.data(), work.theta_plane_mean.data(), params.nx, params.ny, slab.k_count);
            remove_plane_mean_and_scale_kernel<<<blocks_for(work.center_count, threads), threads>>>(
                work.theta.data(), work.theta_plane_mean.data(), work.center_count,
                plane_size(params), params.g / (params.theta0 * (params.moisture_enabled ? 1.0 + 0.61 * params.qv0 : 1.0)));
            center_horizontal_derivatives(work, work.theta, work.qx, work.qy, params);
            ddz_center_local_kernel<<<blocks_for(work.center_count, threads), threads>>>(
                work.theta.data(), work.dtheta_dz_center.data(), work.center_count,
                params.nx, params.ny, slab.k_begin, slab.k_begin, params.nz, inv_dz);
            check_cuda(cudaGetLastError(), "AMD buoyancy-gradient kernel launch");
        } else {
            work.qx.zero();
            work.qy.zero();
            work.dtheta_dz_center.zero();
        }
        build_amd_center_stress_kernel<<<blocks_for(work.center_count, threads), threads>>>(
            work.dudx.data(), work.dudy.data(), work.dudz.data(),
            work.dvdx.data(), work.dvdy.data(), work.dvdz.data(),
            work.dwdx_center.data(), work.dwdy_center.data(), work.dwdz_center.data(),
            work.dudz_face.data(), work.dvdz_face.data(),
            work.dwdx_face.data(), work.dwdy_face.data(),
            work.qx.data(), work.qy.data(), work.dtheta_dz_center.data(),
            work.strain.data(), work.nu_t.data(), work.txx.data(), work.txy.data(),
            work.tyy.data(), work.tzz.data(), work.center_count,
            params.nx, params.ny, slab.k_begin, slab.face_begin,
            params.dx() / std::sqrt(12.0), params.dy() / std::sqrt(12.0),
            params.dz() / std::sqrt(3.0));
        check_cuda(cudaGetLastError(), "AMD center stress kernel launch");
    } else {
        build_smagorinsky_center_kernel<<<blocks_for(work.center_count, threads), threads>>>(
            work.dudx.data(), work.dudy.data(), work.dudz.data(),
            work.dvdx.data(), work.dvdy.data(), work.dvdz.data(),
            work.dwdx_center.data(), work.dwdy_center.data(), work.dwdz_center.data(),
            work.strain.data(), work.nu_t.data(), work.txx.data(), work.txy.data(),
            work.tyy.data(), work.tzz.data(), work.center_count, coeff, 1.0);
        check_cuda(cudaGetLastError(), "Smagorinsky center stress kernel launch");
    }

    scatter_center_owned(work.nu_t, work.nu_t_field, params);
    scatter_center_owned(work.tzz, work.tzz_field, params);
    check_cuda(cudaDeviceSynchronize(), "SGS center stress scatter completion");
    exchange_field_halo(work.nu_t_field, 700, params, slab, comm, halo);
    exchange_field_halo(work.tzz_field, 710, params, slab, comm, halo);

    center_to_face_local_kernel<<<blocks_for(work.face_count, threads), threads>>>(
        work.nu_t_field.values.data(), work.nu_t_face.data(), work.face_count, params.nx, params.ny, slab.face_begin, work.nu_t_field.plane_begin, params.nz);
    check_cuda(cudaGetLastError(), "SGS nu_t face interpolation kernel launch");
    build_smagorinsky_face_kernel<<<blocks_for(work.face_count, threads), threads>>>(
        work.nu_t_face.data(),
        work.dudz_face.data(),
        work.dvdz_face.data(),
        work.dwdx_face.data(),
        work.dwdy_face.data(),
        work.txz.data(),
        work.tyz.data(),
        work.face_count,
        params.nx,
        params.ny,
        slab.face_begin,
        params.nz,
        1.0);
    check_cuda(cudaGetLastError(), "SGS face stress kernel launch");
    scatter_face_owned(work.txz, work.txz_field, params);
    scatter_face_owned(work.tyz, work.tyz_field, params);
    check_cuda(cudaDeviceSynchronize(), "SGS face stress scatter completion");
    exchange_field_halo(work.txz_field, 720, params, slab, comm, halo);
    exchange_field_halo(work.tyz_field, 730, params, slab, comm, halo);

    center_horizontal_derivative_x(work, work.txx, work.dtxx_dx, params);
    center_horizontal_derivative_y(work, work.txy, work.dtxy_dy, params);
    center_horizontal_derivative_x(work, work.txy, work.dtxy_dx, params);
    center_horizontal_derivative_y(work, work.tyy, work.dtyy_dy, params);
    ddz_face_to_center_local_kernel<<<blocks_for(work.center_count, threads), threads>>>(
        work.txz_field.values.data(), work.dtxz_dz.data(), work.center_count, params.nx, params.ny, slab.k_begin, work.txz_field.plane_begin, inv_dz);
    check_cuda(cudaGetLastError(), "SGS dtxz dz kernel launch");
    ddz_face_to_center_local_kernel<<<blocks_for(work.center_count, threads), threads>>>(
        work.tyz_field.values.data(), work.dtyz_dz.data(), work.center_count, params.nx, params.ny, slab.k_begin, work.tyz_field.plane_begin, inv_dz);
    check_cuda(cudaGetLastError(), "SGS dtyz dz kernel launch");
    add_center_sgs_divergence_kernel<<<blocks_for(work.center_count, threads), threads>>>(
        work.rhs_u.data(),
        work.rhs_v.data(),
        work.dtxx_dx.data(),
        work.dtxy_dy.data(),
        work.dtxy_dx.data(),
        work.dtyy_dy.data(),
        work.dtxz_dz.data(),
        work.dtyz_dz.data(),
        work.center_count,
        1.0);
    check_cuda(cudaGetLastError(), "SGS center divergence kernel launch");

    face_horizontal_derivative_x(work, work.txz, work.dtxz_dx, params);
    face_horizontal_derivative_y(work, work.tyz, work.dtyz_dy, params);
    ddz_center_to_face_local_kernel<<<blocks_for(work.face_count, threads), threads>>>(
        work.tzz_field.values.data(), work.dtzz_dz.data(), work.face_count, params.nx, params.ny, slab.face_begin, work.tzz_field.plane_begin, params.nz, inv_dz);
    check_cuda(cudaGetLastError(), "SGS dtzz dz kernel launch");
    add_face_sgs_divergence_kernel<<<blocks_for(work.face_count, threads), threads>>>(
        work.rhs_w.data(),
        work.dtxz_dx.data(),
        work.dtyz_dy.data(),
        work.dtzz_dz.data(),
        work.face_count,
        params.nx,
        params.ny,
        slab.face_begin,
        params.nz,
        1.0);
    check_cuda(cudaGetLastError(), "SGS face divergence kernel launch");
}

void compute_momentum_rhs(CudaMpiState& state, RhsWorkspace& work, const Params& params, const Slab& slab, MPI_Comm comm, HaloScratch& halo) {
    constexpr int threads = 256;
    const double inv_dz = 1.0 / params.dz();
    const double inv_dz2 = 1.0 / (params.dz() * params.dz());

    copy_owned_from_local_kernel<<<blocks_for(work.center_count, threads), threads>>>(
        state.u.values.data(), work.u.data(), work.center_count, params.nx, params.ny, slab.k_begin, state.u.plane_begin);
    check_cuda(cudaGetLastError(), "copy local u to RHS workspace kernel launch");
    copy_owned_from_local_kernel<<<blocks_for(work.center_count, threads), threads>>>(
        state.v.values.data(), work.v.data(), work.center_count, params.nx, params.ny, slab.k_begin, state.v.plane_begin);
    check_cuda(cudaGetLastError(), "copy local v to RHS workspace kernel launch");
    copy_owned_from_local_kernel<<<blocks_for(work.face_count, threads), threads>>>(
        state.w.values.data(), work.w_face.data(), work.face_count, params.nx, params.ny, slab.face_begin, state.w.plane_begin);
    check_cuda(cudaGetLastError(), "copy local w to RHS workspace kernel launch");
    if (params.thermo_enabled) {
        copy_owned_from_local_kernel<<<blocks_for(work.center_count, threads), threads>>>(
            state.theta.values.data(), work.theta.data(), work.center_count, params.nx, params.ny, slab.k_begin, state.theta.plane_begin);
        check_cuda(cudaGetLastError(), "copy local theta to RHS workspace kernel launch");
    }
    work.nu_t.zero();

    w_to_center_local_kernel<<<blocks_for(work.center_count, threads), threads>>>(
        state.w.values.data(), work.w_center.data(), work.center_count, params.nx, params.ny, slab.k_begin, state.w.plane_begin);
    check_cuda(cudaGetLastError(), "w center interpolation kernel launch");
    center_to_face_local_kernel<<<blocks_for(work.face_count, threads), threads>>>(
        state.u.values.data(), work.u_on_w.data(), work.face_count, params.nx, params.ny, slab.face_begin, state.u.plane_begin, params.nz);
    check_cuda(cudaGetLastError(), "u face interpolation kernel launch");
    center_to_face_local_kernel<<<blocks_for(work.face_count, threads), threads>>>(
        state.v.values.data(), work.v_on_w.data(), work.face_count, params.nx, params.ny, slab.face_begin, state.v.plane_begin, params.nz);
    check_cuda(cudaGetLastError(), "v face interpolation kernel launch");

    ddz_center_local_kernel<<<blocks_for(work.center_count, threads), threads>>>(
        state.u.values.data(), work.dudz.data(), work.center_count, params.nx, params.ny, slab.k_begin, state.u.plane_begin, params.nz, inv_dz);
    check_cuda(cudaGetLastError(), "dudz center kernel launch");
    ddz_center_local_kernel<<<blocks_for(work.center_count, threads), threads>>>(
        state.v.values.data(), work.dvdz.data(), work.center_count, params.nx, params.ny, slab.k_begin, state.v.plane_begin, params.nz, inv_dz);
    check_cuda(cudaGetLastError(), "dvdz center kernel launch");
    ddz_w_face_local_kernel<<<blocks_for(work.face_count, threads), threads>>>(
        state.w.values.data(), work.dwdz_face.data(), work.face_count, params.nx, params.ny, slab.face_begin, state.w.plane_begin, params.nz, inv_dz);
    check_cuda(cudaGetLastError(), "dwdz face kernel launch");

    center_horizontal_derivatives_and_laplacian(work, work.u, work.dudx, work.dudy, work.lap_u, params);
    center_horizontal_derivatives_and_laplacian(work, work.v, work.dvdx, work.dvdy, work.lap_v, params);
    face_horizontal_derivatives_and_laplacian(work, work.w_face, work.dwdx_face, work.dwdy_face, work.lap_w, params);
    add_vertical_laplacian_center_local_kernel<<<blocks_for(work.center_count, threads), threads>>>(
        work.lap_u.data(), state.u.values.data(), work.center_count, params.nx, params.ny, slab.k_begin, state.u.plane_begin, params.nz, inv_dz2);
    check_cuda(cudaGetLastError(), "u vertical laplacian kernel launch");
    add_vertical_laplacian_center_local_kernel<<<blocks_for(work.center_count, threads), threads>>>(
        work.lap_v.data(), state.v.values.data(), work.center_count, params.nx, params.ny, slab.k_begin, state.v.plane_begin, params.nz, inv_dz2);
    check_cuda(cudaGetLastError(), "v vertical laplacian kernel launch");
    add_vertical_laplacian_face_local_kernel<<<blocks_for(work.face_count, threads), threads>>>(
        work.lap_w.data(), state.w.values.data(), work.face_count, params.nx, params.ny, slab.face_begin, state.w.plane_begin, params.nz, inv_dz2);
    check_cuda(cudaGetLastError(), "w vertical laplacian kernel launch");

    if (params.momentum_advection_form == "rotational") {
        ddz_center_to_face_local_kernel<<<blocks_for(work.face_count, threads), threads>>>(
            state.u.values.data(), work.dudz_face.data(), work.face_count, params.nx, params.ny,
            slab.face_begin, state.u.plane_begin, params.nz, inv_dz);
        ddz_center_to_face_local_kernel<<<blocks_for(work.face_count, threads), threads>>>(
            state.v.values.data(), work.dvdz_face.data(), work.face_count, params.nx, params.ny,
            slab.face_begin, state.v.plane_begin, params.nz, inv_dz);
        check_cuda(cudaGetLastError(), "rotational face-gradient kernels launch");
    }

    build_center_rhs_kernel<<<blocks_for(work.center_count, threads), threads>>>(
        work.u.data(),
        work.v.data(),
        work.w_center.data(),
        work.dudx.data(),
        work.dudy.data(),
        work.dudz.data(),
        work.dvdx.data(),
        work.dvdy.data(),
        work.dvdz.data(),
        work.lap_u.data(),
        work.lap_v.data(),
        work.rhs_u.data(),
        work.rhs_v.data(),
        work.center_count,
        params.nu,
        0.0,
        params.geostrophic_u,
        params.geostrophic_v,
        1.0);
    check_cuda(cudaGetLastError(), "center momentum RHS kernel launch");
    build_w_rhs_kernel<<<blocks_for(work.face_count, threads), threads>>>(
        work.w_face.data(),
        work.u_on_w.data(),
        work.v_on_w.data(),
        work.dwdx_face.data(),
        work.dwdy_face.data(),
        work.dwdz_face.data(),
        work.lap_w.data(),
        work.rhs_w.data(),
        work.face_count,
        params.nx,
        params.ny,
        slab.face_begin,
        params.nz,
        params.nu,
        1.0);
    check_cuda(cudaGetLastError(), "w momentum RHS kernel launch");

    if (params.momentum_advection_form == "rotational") {
        build_rotational_center_rhs_kernel<<<blocks_for(work.center_count, threads), threads>>>(
            work.u.data(), work.v.data(), work.w_face.data(),
            work.dudy.data(), work.dvdx.data(), work.dudz_face.data(), work.dvdz_face.data(),
            work.dwdx_face.data(), work.dwdy_face.data(), work.lap_u.data(), work.lap_v.data(),
            work.rhs_u.data(), work.rhs_v.data(), work.center_count,
            params.nx, params.ny, slab.k_begin, slab.face_begin, params.nu);
        build_rotational_face_rhs_kernel<<<blocks_for(work.face_count, threads), threads>>>(
            work.u_on_w.data(), work.v_on_w.data(), work.dudz_face.data(), work.dvdz_face.data(),
            work.dwdx_face.data(), work.dwdy_face.data(), work.lap_w.data(), work.rhs_w.data(),
            work.face_count, params.nx, params.ny, slab.face_begin, params.nz, params.nu);
        check_cuda(cudaGetLastError(), "rotational momentum RHS kernels launch");
    }

    add_coriolis_kernel<<<blocks_for(work.center_count, threads), threads>>>(
        work.rhs_u.data(), work.rhs_v.data(), work.u.data(), work.v.data(),
        work.center_count, plane_size(params), slab.k_begin, params.dz(),
        params.coriolis_f, params.geostrophic_u, params.geostrophic_v,
        params.initial_condition == "bomex" ? 1 : 0);
    check_cuda(cudaGetLastError(), "Coriolis forcing kernel launch");

    if (params.sgs_model == "smagorinsky" || params.sgs_model == "amd") {
        apply_viscosity_sgs(state, work, params, slab, comm, halo);
    }
    apply_wall_stress(state, work, params, slab);
    compute_scalar_rhs(state, work, params, slab, comm, halo);
}

void apply_wall_stress(CudaMpiState&, RhsWorkspace& work, const Params& params, const Slab& slab) {
    if (params.momentum_wall_model != "abl" || slab.rank != 0) {
        return;
    }
    if (params.wall_ref_height() <= params.zo) {
        throw std::runtime_error("ABL wall stress requires wall_ref_height > zo");
    }
    constexpr int threads = 256;
    const double* wall_u = work.u.data();
    const double* wall_v = work.v.data();
    if (params.wall_stress_model == "dynamic_neutral") {
        const double filter_width = params.fgr * params.tfr;
        filter_center_owned(work, work.u, work.dtxx_dx, params, filter_width);
        filter_center_owned(work, work.v, work.dtxy_dy, params, filter_width);
        wall_u = work.dtxx_dx.data();
        wall_v = work.dtxy_dy.data();
    }
    apply_wall_stress_kernel<<<blocks_for(plane_size(params), threads), threads>>>(
        work.rhs_u.data(),
        work.rhs_v.data(),
        wall_u,
        wall_v,
        params.nx,
        params.ny,
        1.0 / params.dz(),
        params.u_fric,
        params.vonk,
        std::log(params.wall_ref_height() / params.zo),
        params.wall_stress_model == "dynamic_neutral" ? 1 : 0);
    check_cuda(cudaGetLastError(), "wall stress kernel launch");
}

void compute_one_scalar_rhs_single_gpu(
    CudaMpiState& state,
    DeviceLocalField& scalar,
    DeviceBuffer<double>& scalar_owned,
    DeviceBuffer<double>& rhs,
    double molecular_diffusivity,
    double turbulent_ratio,
    double surface_flux,
    RhsWorkspace& work,
    const Params& params,
    const Slab& slab,
    MPI_Comm comm,
    HaloScratch& halo) {
    constexpr int threads = 256;
    constexpr double two_thirds_filter_width = 1.5;
    const double inv_dz = 1.0 / params.dz();
    copy_owned_from_local_kernel<<<blocks_for(work.center_count, threads), threads>>>(
        scalar.values.data(), scalar_owned.data(), work.center_count,
        params.nx, params.ny, slab.k_begin, scalar.plane_begin);
    center_to_face_local_kernel<<<blocks_for(work.face_count, threads), threads>>>(
        scalar.values.data(), work.theta_on_w.data(), work.face_count,
        params.nx, params.ny, slab.face_begin, scalar.plane_begin, params.nz);
    build_scalar_advective_flux_kernel<<<blocks_for(work.center_count, threads), threads>>>(
        work.u.data(), work.v.data(), scalar_owned.data(),
        work.theta_flux_x.data(), work.theta_flux_y.data(), work.center_count);
    build_scalar_face_flux_kernel<<<blocks_for(work.face_count, threads), threads>>>(
        work.w_face.data(), work.theta_on_w.data(), work.theta_flux_z.data(), work.face_count);
    check_cuda(cudaGetLastError(), "single-GPU scalar advective flux kernels launch");
    if (params.horizontal_dealias && params.dealiasing == "sharp") {
        filter_center_owned(work, work.theta_flux_x, work.theta_flux_x, params, two_thirds_filter_width);
        filter_center_owned(work, work.theta_flux_y, work.theta_flux_y, params, two_thirds_filter_width);
    }
    scatter_face_owned(work.theta_flux_z, work.scalar_flux_z_field, params);
    check_cuda(cudaDeviceSynchronize(), "single-GPU scalar advective face scatter completion");
    exchange_field_halo(work.scalar_flux_z_field, 550, params, slab, comm, halo);
    center_horizontal_derivative_x(work, work.theta_flux_x, work.dtheta_dx, params);
    center_horizontal_derivative_y(work, work.theta_flux_y, work.dtheta_dy, params);
    horizontal_divergence_kernel<<<blocks_for(work.center_count, threads), threads>>>(
        work.dtheta_dx.data(), work.dtheta_dy.data(), work.div_xy.data(), work.center_count);
    ddz_face_to_center_local_kernel<<<blocks_for(work.center_count, threads), threads>>>(
        work.scalar_flux_z_field.values.data(), work.div_z.data(), work.center_count,
        params.nx, params.ny, slab.k_begin, work.scalar_flux_z_field.plane_begin, inv_dz);
    scalar_rhs_advective_kernel<<<blocks_for(work.center_count, threads), threads>>>(
        rhs.data(), work.div_xy.data(), work.div_z.data(), work.center_count);
    check_cuda(cudaGetLastError(), "single-GPU scalar advection divergence kernels launch");

    center_horizontal_derivatives(work, scalar_owned, work.dtheta_dx, work.dtheta_dy, params);
    ddz_center_local_kernel<<<blocks_for(work.center_count, threads), threads>>>(
        scalar.values.data(), work.dtheta_dz_center.data(), work.center_count,
        params.nx, params.ny, slab.k_begin, scalar.plane_begin, params.nz, inv_dz);
    ddz_center_to_face_local_kernel<<<blocks_for(work.face_count, threads), threads>>>(
        scalar.values.data(), work.dtheta_dz_w.data(), work.face_count,
        params.nx, params.ny, slab.face_begin, scalar.plane_begin, params.nz, inv_dz);
    check_cuda(cudaGetLastError(), "single-GPU scalar gradient kernels launch");
    if (params.scalar_sgs_model == "amd") {
        amd_scalar_kappa_kernel<<<blocks_for(work.center_count, threads), threads>>>(
            work.dudx.data(), work.dudy.data(), work.dudz.data(),
            work.dvdx.data(), work.dvdy.data(), work.dvdz.data(),
            work.dwdx_center.data(), work.dwdy_center.data(), work.dwdz_center.data(),
            work.dtheta_dx.data(), work.dtheta_dy.data(), work.dtheta_dz_center.data(),
            work.dtheta_dz_w.data(), work.kappa_center.data(), work.center_count,
            params.nx, params.ny, slab.k_begin, slab.face_begin,
            params.dx() / std::sqrt(12.0), params.dy() / std::sqrt(12.0),
            params.dz() / std::sqrt(3.0), molecular_diffusivity);
    } else {
        scalar_kappa_kernel<<<blocks_for(work.center_count, threads), threads>>>(
            work.nu_t.data(), work.strain.data(), work.dtheta_dz_center.data(),
            work.kappa_center.data(), work.center_count, molecular_diffusivity,
            turbulent_ratio, params.scalar_stability_correction ? 1 : 0,
            params.g, params.theta0,
            std::pow(params.smagorinsky_cs * params.sgs_delta(), 2.0),
            params.scalar_stability_beta, params.scalar_stability_power);
    }
    check_cuda(cudaGetLastError(), "single-GPU scalar diffusivity kernel launch");
    scatter_center_owned(work.kappa_center, work.nu_t_field, params);
    check_cuda(cudaDeviceSynchronize(), "single-GPU scalar diffusivity scatter completion");
    exchange_field_halo(work.nu_t_field, 560, params, slab, comm, halo);
    center_to_face_local_kernel<<<blocks_for(work.face_count, threads), threads>>>(
        work.nu_t_field.values.data(), work.kappa_w.data(), work.face_count,
        params.nx, params.ny, slab.face_begin, work.nu_t_field.plane_begin, params.nz);
    build_scalar_diffusive_flux_kernel<<<blocks_for(work.center_count, threads), threads>>>(
        work.kappa_center.data(), work.dtheta_dx.data(), work.dtheta_dy.data(),
        work.qx.data(), work.qy.data(), work.center_count);
    build_scalar_diffusive_face_flux_kernel<<<blocks_for(work.face_count, threads), threads>>>(
        work.kappa_w.data(), work.dtheta_dz_w.data(), work.qz.data(), work.face_count,
        params.nx, params.ny, slab.face_begin, params.nz,
        slab.rank == 0 ? surface_flux : 0.0);
    check_cuda(cudaGetLastError(), "single-GPU scalar diffusive flux kernels launch");
    scatter_face_owned(work.qz, work.scalar_qz_field, params);
    check_cuda(cudaDeviceSynchronize(), "single-GPU scalar diffusive face scatter completion");
    exchange_field_halo(work.scalar_qz_field, 570, params, slab, comm, halo);
    center_horizontal_derivative_x(work, work.qx, work.dtheta_dx, params);
    center_horizontal_derivative_y(work, work.qy, work.dtheta_dy, params);
    horizontal_divergence_kernel<<<blocks_for(work.center_count, threads), threads>>>(
        work.dtheta_dx.data(), work.dtheta_dy.data(), work.div_xy.data(), work.center_count);
    ddz_face_to_center_local_kernel<<<blocks_for(work.center_count, threads), threads>>>(
        work.scalar_qz_field.values.data(), work.div_z.data(), work.center_count,
        params.nx, params.ny, slab.k_begin, work.scalar_qz_field.plane_begin, inv_dz);
    add_scalar_diffusion_kernel<<<blocks_for(work.center_count, threads), threads>>>(
        rhs.data(), work.div_xy.data(), work.div_z.data(), work.center_count);
    check_cuda(cudaGetLastError(), "single-GPU scalar diffusion divergence kernels launch");
}

void compute_moist_scalar_rhs_single_gpu(
    CudaMpiState& state,
    RhsWorkspace& work,
    const Params& params,
    const Slab& slab,
    MPI_Comm comm,
    HaloScratch& halo) {
    constexpr int threads = 256;
    compute_one_scalar_rhs_single_gpu(
        state, state.theta_l, work.theta, work.rhs_theta,
        params.scalar_diffusivity, params.prandtl_t, params.surface_theta_flux,
        work, params, slab, comm, halo);
    check_cuda(cudaMemcpyAsync(
        work.theta_kappa.data(), work.kappa_center.data(),
        work.center_count * sizeof(double), cudaMemcpyDeviceToDevice),
        "theta_l diffusivity diagnostic copy");
    const double surface_qt_flux = params.initial_condition == "bomex"
        ? params.surface_qv_flux / std::pow(1.0 - 0.017, 2.0)
        : params.surface_qv_flux;
    compute_one_scalar_rhs_single_gpu(
        state, state.qt, work.qt, work.rhs_qt,
        params.moisture_diffusivity, params.schmidt_t, surface_qt_flux,
        work, params, slab, comm, halo);
    check_cuda(cudaMemcpyAsync(
        work.qt_kappa.data(), work.kappa_center.data(),
        work.center_count * sizeof(double), cudaMemcpyDeviceToDevice),
        "q_t diffusivity diagnostic copy");

    if (params.initial_condition == "bomex") {
        plane_mean_kernel<<<slab.k_count, threads, static_cast<std::size_t>(threads) * sizeof(double)>>>(
            work.u.data(), work.u_plane_mean.data(), params.nx, params.ny, slab.k_count);
        plane_mean_kernel<<<slab.k_count, threads, static_cast<std::size_t>(threads) * sizeof(double)>>>(
            work.v.data(), work.v_plane_mean.data(), params.nx, params.ny, slab.k_count);
        plane_mean_kernel<<<slab.k_count, threads, static_cast<std::size_t>(threads) * sizeof(double)>>>(
            work.theta.data(), work.theta_plane_mean.data(), params.nx, params.ny, slab.k_count);
        plane_mean_kernel<<<slab.k_count, threads, static_cast<std::size_t>(threads) * sizeof(double)>>>(
            work.qt.data(), work.qt_plane_mean.data(), params.nx, params.ny, slab.k_count);
        add_bomex_large_scale_forcing_kernel<<<blocks_for(work.center_count, threads), threads>>>(
            work.rhs_u.data(), work.rhs_v.data(), work.rhs_theta.data(), work.rhs_qt.data(),
            work.u_plane_mean.data(), work.v_plane_mean.data(),
            work.theta_plane_mean.data(), work.qt_plane_mean.data(),
            work.center_count, plane_size(params), slab.k_begin, params.nz, params.dz());
        check_cuda(cudaGetLastError(), "BOMEX large-scale forcing kernel launch");
    }

    build_virtual_potential_temperature_kernel<<<blocks_for(work.center_count, threads), threads>>>(
        state.theta.values.data(), state.qv.values.data(), state.ql.values.data(),
        work.theta.data(), work.center_count);
    plane_mean_kernel<<<slab.k_count, threads, static_cast<std::size_t>(threads) * sizeof(double)>>>(
        work.theta.data(), work.theta_plane_mean.data(), params.nx, params.ny, slab.k_count);
    add_buoyancy_kernel<<<blocks_for(work.face_count, threads), threads>>>(
        work.rhs_w.data(), work.theta.data(), work.theta_plane_mean.data(),
        work.face_count, params.nx, params.ny, slab.face_begin, slab.k_begin,
        params.nz, params.g / params.theta0);
    check_cuda(cudaGetLastError(), "moist buoyancy kernel launch");
}

void compute_scalar_rhs(CudaMpiState& state, RhsWorkspace& work, const Params& params, const Slab& slab, MPI_Comm comm, HaloScratch& halo) {
    if (!params.thermo_enabled) {
        return;
    }
    if (params.moisture_enabled) {
        compute_moist_scalar_rhs_single_gpu(state, work, params, slab, comm, halo);
        return;
    }
    constexpr int threads = 256;
    const double inv_dz = 1.0 / params.dz();

    center_to_face_local_kernel<<<blocks_for(work.face_count, threads), threads>>>(
        state.theta.values.data(), work.theta_on_w.data(), work.face_count, params.nx, params.ny, slab.face_begin, state.theta.plane_begin, params.nz);
    check_cuda(cudaGetLastError(), "theta face interpolation kernel launch");
    build_scalar_advective_flux_kernel<<<blocks_for(work.center_count, threads), threads>>>(
        work.u.data(), work.v.data(), work.theta.data(), work.theta_flux_x.data(), work.theta_flux_y.data(), work.center_count);
    check_cuda(cudaGetLastError(), "scalar advective center flux kernel launch");
    build_scalar_face_flux_kernel<<<blocks_for(work.face_count, threads), threads>>>(
        work.w_face.data(), work.theta_on_w.data(), work.theta_flux_z.data(), work.face_count);
    check_cuda(cudaGetLastError(), "scalar advective face flux kernel launch");
    scatter_face_owned(work.theta_flux_z, work.scalar_flux_z_field, params);
    check_cuda(cudaDeviceSynchronize(), "scalar advective face flux scatter completion");
    exchange_field_halo(work.scalar_flux_z_field, 550, params, slab, comm, halo);

    center_horizontal_derivative_x(work, work.theta_flux_x, work.dtheta_dx, params);
    center_horizontal_derivative_y(work, work.theta_flux_y, work.dtheta_dy, params);
    horizontal_divergence_kernel<<<blocks_for(work.center_count, threads), threads>>>(
        work.dtheta_dx.data(), work.dtheta_dy.data(), work.div_xy.data(), work.center_count);
    check_cuda(cudaGetLastError(), "scalar advective horizontal divergence kernel launch");
    ddz_face_to_center_local_kernel<<<blocks_for(work.center_count, threads), threads>>>(
        work.scalar_flux_z_field.values.data(), work.div_z.data(), work.center_count, params.nx, params.ny,
        slab.k_begin, work.scalar_flux_z_field.plane_begin, inv_dz);
    check_cuda(cudaGetLastError(), "scalar advective vertical divergence kernel launch");
    scalar_rhs_advective_kernel<<<blocks_for(work.center_count, threads), threads>>>(
        work.rhs_theta.data(), work.div_xy.data(), work.div_z.data(), work.center_count);
    check_cuda(cudaGetLastError(), "scalar RHS advective kernel launch");

    center_horizontal_derivatives(work, work.theta, work.dtheta_dx, work.dtheta_dy, params);
    ddz_center_to_face_local_kernel<<<blocks_for(work.face_count, threads), threads>>>(
        state.theta.values.data(), work.dtheta_dz_w.data(), work.face_count, params.nx, params.ny, slab.face_begin, state.theta.plane_begin, params.nz, inv_dz);
    check_cuda(cudaGetLastError(), "scalar dtheta dz face kernel launch");
    if (params.scalar_stability_correction) {
        ddz_center_local_kernel<<<blocks_for(work.center_count, threads), threads>>>(
            state.theta.values.data(), work.dtheta_dz_center.data(), work.center_count, params.nx, params.ny, slab.k_begin, state.theta.plane_begin, params.nz, inv_dz);
        check_cuda(cudaGetLastError(), "scalar dtheta dz center kernel launch");
    } else {
        work.dtheta_dz_center.zero();
    }
    scalar_kappa_kernel<<<blocks_for(work.center_count, threads), threads>>>(
        work.nu_t.data(),
        work.strain.data(),
        work.dtheta_dz_center.data(),
        work.kappa_center.data(),
        work.center_count,
        params.scalar_diffusivity,
        params.prandtl_t,
        params.scalar_stability_correction ? 1 : 0,
        params.g,
        params.theta0,
        std::pow(params.smagorinsky_cs * params.sgs_delta(), 2.0),
        params.scalar_stability_beta,
        params.scalar_stability_power);
    check_cuda(cudaGetLastError(), "scalar kappa center kernel launch");
    scatter_center_owned(work.kappa_center, work.nu_t_field, params);
    check_cuda(cudaDeviceSynchronize(), "scalar kappa scatter completion");
    exchange_field_halo(work.nu_t_field, 560, params, slab, comm, halo);
    center_to_face_local_kernel<<<blocks_for(work.face_count, threads), threads>>>(
        work.nu_t_field.values.data(), work.kappa_w.data(), work.face_count, params.nx, params.ny, slab.face_begin, work.nu_t_field.plane_begin, params.nz);
    check_cuda(cudaGetLastError(), "scalar kappa face interpolation kernel launch");

    build_scalar_diffusive_flux_kernel<<<blocks_for(work.center_count, threads), threads>>>(
        work.kappa_center.data(), work.dtheta_dx.data(), work.dtheta_dy.data(), work.qx.data(), work.qy.data(), work.center_count);
    check_cuda(cudaGetLastError(), "scalar diffusive center flux kernel launch");
    build_scalar_diffusive_face_flux_kernel<<<blocks_for(work.face_count, threads), threads>>>(
        work.kappa_w.data(), work.dtheta_dz_w.data(), work.qz.data(), work.face_count, params.nx, params.ny, slab.face_begin, params.nz,
        slab.rank == 0 ? params.surface_theta_flux : 0.0);
    check_cuda(cudaGetLastError(), "scalar diffusive face flux kernel launch");
    scatter_face_owned(work.qz, work.scalar_qz_field, params);
    check_cuda(cudaDeviceSynchronize(), "scalar diffusive face flux scatter completion");
    exchange_field_halo(work.scalar_qz_field, 570, params, slab, comm, halo);

    center_horizontal_derivative_x(work, work.qx, work.dtheta_dx, params);
    center_horizontal_derivative_y(work, work.qy, work.dtheta_dy, params);
    horizontal_divergence_kernel<<<blocks_for(work.center_count, threads), threads>>>(
        work.dtheta_dx.data(), work.dtheta_dy.data(), work.div_xy.data(), work.center_count);
    check_cuda(cudaGetLastError(), "scalar diffusive horizontal divergence kernel launch");
    ddz_face_to_center_local_kernel<<<blocks_for(work.center_count, threads), threads>>>(
        work.scalar_qz_field.values.data(), work.div_z.data(), work.center_count, params.nx, params.ny,
        slab.k_begin, work.scalar_qz_field.plane_begin, inv_dz);
    check_cuda(cudaGetLastError(), "scalar diffusive vertical divergence kernel launch");
    add_scalar_diffusion_kernel<<<blocks_for(work.center_count, threads), threads>>>(
        work.rhs_theta.data(), work.div_xy.data(), work.div_z.data(), work.center_count);
    check_cuda(cudaGetLastError(), "scalar RHS diffusive kernel launch");

    plane_mean_kernel<<<state.theta.plane_count, threads, static_cast<std::size_t>(threads) * sizeof(double)>>>(
        state.theta.values.data(),
        work.theta_plane_mean.data(),
        params.nx,
        params.ny,
        state.theta.plane_count);
    check_cuda(cudaGetLastError(), "theta plane mean kernel launch");
    add_buoyancy_kernel<<<blocks_for(work.face_count, threads), threads>>>(
        work.rhs_w.data(),
        state.theta.values.data(),
        work.theta_plane_mean.data(),
        work.face_count,
        params.nx,
        params.ny,
        slab.face_begin,
        state.theta.plane_begin,
        params.nz,
        params.g / params.theta0);
    check_cuda(cudaGetLastError(), "buoyancy kernel launch");
}

void horizontal_dealias_state(CudaMpiState& state, RhsWorkspace& work, const Params& params, const Slab& slab) {
    if (!params.horizontal_dealias) {
        return;
    }
    constexpr double two_thirds_filter_width = 1.5;
    constexpr int threads = 256;
    copy_owned_from_local_kernel<<<blocks_for(work.center_count, threads), threads>>>(
        state.u.values.data(), work.u.data(), work.center_count, params.nx, params.ny, slab.k_begin, state.u.plane_begin);
    copy_owned_from_local_kernel<<<blocks_for(work.center_count, threads), threads>>>(
        state.v.values.data(), work.v.data(), work.center_count, params.nx, params.ny, slab.k_begin, state.v.plane_begin);
    copy_owned_from_local_kernel<<<blocks_for(work.face_count, threads), threads>>>(
        state.w.values.data(), work.w_face.data(), work.face_count, params.nx, params.ny, slab.face_begin, state.w.plane_begin);
    check_cuda(cudaGetLastError(), "dealias copy kernels launch");
    filter_center_owned(work, work.u, work.u, params, two_thirds_filter_width);
    filter_center_owned(work, work.v, work.v, params, two_thirds_filter_width);
    filter_face_owned(work, work.w_face, work.w_face, params, two_thirds_filter_width);
    scatter_center_owned(work.u, state.u, params);
    scatter_center_owned(work.v, state.v, params);
    scatter_face_owned(work.w_face, state.w, params);
    if (params.thermo_enabled) {
        if (params.moisture_enabled) {
            copy_owned_from_local_kernel<<<blocks_for(work.center_count, threads), threads>>>(
                state.theta_l.values.data(), work.theta.data(), work.center_count,
                params.nx, params.ny, slab.k_begin, state.theta_l.plane_begin);
            copy_owned_from_local_kernel<<<blocks_for(work.center_count, threads), threads>>>(
                state.qt.values.data(), work.qt.data(), work.center_count,
                params.nx, params.ny, slab.k_begin, state.qt.plane_begin);
            check_cuda(cudaGetLastError(), "dealias moist scalar copy kernels launch");
            filter_center_owned(work, work.theta, work.theta, params, two_thirds_filter_width);
            filter_center_owned(work, work.qt, work.qt, params, two_thirds_filter_width);
            scatter_center_owned(work.theta, state.theta_l, params);
            scatter_center_owned(work.qt, state.qt, params);
        } else {
            copy_owned_from_local_kernel<<<blocks_for(work.center_count, threads), threads>>>(
                state.theta.values.data(), work.theta.data(), work.center_count,
                params.nx, params.ny, slab.k_begin, state.theta.plane_begin);
            check_cuda(cudaGetLastError(), "dealias theta copy kernel launch");
            filter_center_owned(work, work.theta, work.theta, params, two_thirds_filter_width);
            scatter_center_owned(work.theta, state.theta, params);
        }
    }
    enforce_walls_local_kernel<<<blocks_for(plane_size(params), threads), threads>>>(
        state.w.values.data(), params.nx, params.ny, params.nz, state.w.plane_begin, state.w.plane_count);
    check_cuda(cudaGetLastError(), "dealias wall enforcement kernel launch");
    check_cuda(cudaDeviceSynchronize(), "dealias completion");
}

void cuda_mpi_step(CudaMpiState& state, RhsWorkspace& rhs, PressureWorkspace& pressure, const Params& params, const Slab& slab, MPI_Comm comm, HaloScratch& halo) {
    constexpr int threads = 256;
    const int time_scheme = params.time_scheme == "ab3" ? 3 : (params.time_scheme == "ab2" ? 2 : 1);
    compute_momentum_rhs(state, rhs, params, slab, comm, halo);
    advance_local_kernel<<<blocks_for(rhs.center_count, threads), threads>>>(
        state.u.values.data(), rhs.rhs_u.data(), rhs.rhs_u_prev.data(), rhs.rhs_u_prev2.data(),
        rhs.center_count, params.nx, params.ny, slab.k_begin, state.u.plane_begin,
        time_scheme, state.step_count, params.dt);
    check_cuda(cudaGetLastError(), "u advance kernel launch");
    advance_local_kernel<<<blocks_for(rhs.center_count, threads), threads>>>(
        state.v.values.data(), rhs.rhs_v.data(), rhs.rhs_v_prev.data(), rhs.rhs_v_prev2.data(),
        rhs.center_count, params.nx, params.ny, slab.k_begin, state.v.plane_begin,
        time_scheme, state.step_count, params.dt);
    check_cuda(cudaGetLastError(), "v advance kernel launch");
    advance_local_kernel<<<blocks_for(rhs.face_count, threads), threads>>>(
        state.w.values.data(), rhs.rhs_w.data(), rhs.rhs_w_prev.data(), rhs.rhs_w_prev2.data(),
        rhs.face_count, params.nx, params.ny, slab.face_begin, state.w.plane_begin,
        time_scheme, state.step_count, params.dt);
    check_cuda(cudaGetLastError(), "w advance kernel launch");
    if (params.thermo_enabled) {
        if (params.moisture_enabled) {
            advance_local_kernel<<<blocks_for(rhs.center_count, threads), threads>>>(
                state.theta_l.values.data(), rhs.rhs_theta.data(), rhs.rhs_theta_prev.data(), rhs.rhs_theta_prev2.data(),
                rhs.center_count, params.nx, params.ny, slab.k_begin, state.theta_l.plane_begin,
                time_scheme, state.step_count, params.dt);
            advance_local_kernel<<<blocks_for(rhs.center_count, threads), threads>>>(
                state.qt.values.data(), rhs.rhs_qt.data(), rhs.rhs_qt_prev.data(), rhs.rhs_qt_prev2.data(),
                rhs.center_count, params.nx, params.ny, slab.k_begin, state.qt.plane_begin,
                time_scheme, state.step_count, params.dt);
            check_cuda(cudaGetLastError(), "moist conserved scalar advance kernels launch");
        } else {
            advance_local_kernel<<<blocks_for(rhs.center_count, threads), threads>>>(
                state.theta.values.data(), rhs.rhs_theta.data(), rhs.rhs_theta_prev.data(), rhs.rhs_theta_prev2.data(),
                rhs.center_count, params.nx, params.ny, slab.k_begin, state.theta.plane_begin,
                time_scheme, state.step_count, params.dt);
            check_cuda(cudaGetLastError(), "theta advance kernel launch");
        }
    }
    if (params.sponge_enabled) {
        apply_rayleigh_sponge_center_kernel<<<blocks_for(rhs.center_count, threads), threads>>>(
            state.u.values.data(), state.v.values.data(),
            rhs.center_count, plane_size(params), slab.k_begin, state.u.plane_begin,
            params.dz(), params.lz,
            params.sponge_start_height, params.sponge_timescale, params.sponge_power, params.dt,
            params.geostrophic_u, params.geostrophic_v,
            params.initial_condition == "bomex" ? 1 : 0);
        apply_rayleigh_sponge_face_kernel<<<blocks_for(rhs.face_count, threads), threads>>>(
            state.w.values.data(), rhs.face_count, plane_size(params), slab.face_begin,
            state.w.plane_begin, params.nz, params.dz(), params.lz, params.sponge_start_height,
            params.sponge_timescale, params.sponge_power, params.dt);
        check_cuda(cudaGetLastError(), "Rayleigh sponge kernels launch");
    }
    enforce_walls_local_kernel<<<blocks_for(plane_size(params), threads), threads>>>(
        state.w.values.data(), params.nx, params.ny, params.nz, state.w.plane_begin, state.w.plane_count);
    check_cuda(cudaGetLastError(), "wall enforcement kernel launch");
    check_cuda(cudaDeviceSynchronize(), "advance kernel completion");
    horizontal_dealias_state(state, rhs, params, slab);
    if (params.moisture_enabled) {
        accumulate_moisture_sign_kernel<<<1, threads, 2 * static_cast<std::size_t>(threads) * sizeof(double)>>>(
            state.qt.values.data(), rhs.moisture_sums.data(), rhs.center_count);
        apply_conservative_moisture_limiter_kernel<<<blocks_for(rhs.center_count, threads), threads>>>(
            state.qt.values.data(), rhs.moisture_sums.data(), rhs.center_count);
        saturation_adjustment_kernel<<<blocks_for(rhs.center_count, threads), threads>>>(
            state.theta_l.values.data(), state.qt.values.data(), state.base_pressure.data(),
            state.theta.values.data(), state.qv.values.data(), state.ql.values.data(),
            rhs.center_count, plane_size(params), slab.k_begin);
        check_cuda(cudaGetLastError(), "moisture limiter and saturation-adjustment kernels launch");
    }
    exchange_state_halos(state, params, slab, comm, halo);
    cuda_mpi_project(state, params, slab, comm, halo, pressure);
    state.has_rhs_prev = true;
    ++state.step_count;
}

void cuda_mpi_project(CudaMpiState& state, const Params& params, const Slab& slab, MPI_Comm comm, HaloScratch& halo, PressureWorkspace& pressure) {
    constexpr int threads = 256;
    const double inv_dz = 1.0 / params.dz();
    const double fft_scale = 1.0 / static_cast<double>(params.nx * params.ny);

    extract_pressure_inputs_kernel<<<blocks_for(pressure.owned_real_count, threads), threads>>>(
        state.u.values.data(),
        state.v.values.data(),
        state.w.values.data(),
        pressure.u_owned.data(),
        pressure.v_owned.data(),
        pressure.dwdz_owned.data(),
        pressure.owned_real_count,
        params.nx,
        params.ny,
        slab.k_begin,
        state.u.plane_begin,
        state.w.plane_begin,
        inv_dz);
    check_cuda(cudaGetLastError(), "pressure input extraction kernel launch");

    check_cufft(cufftExecD2Z(pressure.r2c.handle, pressure.u_owned.data(), pressure.u_hat.data()), "u local forward transform");
    check_cufft(cufftExecD2Z(pressure.r2c.handle, pressure.v_owned.data(), pressure.v_hat.data()), "v local forward transform");
    check_cufft(cufftExecD2Z(pressure.r2c.handle, pressure.dwdz_owned.data(), pressure.dwdz_hat.data()), "dwdz local forward transform");

    spectral_divergence_kernel<<<blocks_for(pressure.spectral_count, threads), threads>>>(
        pressure.u_hat.data(),
        pressure.v_hat.data(),
        pressure.dwdz_hat.data(),
        pressure.div_hat.data(),
        pressure.spectral_count,
        params.nkx(),
        params.ny,
        params.nx,
        params.lx,
        params.ly);
    check_cuda(cudaGetLastError(), "distributed divergence kernel launch");

    const std::size_t alltoall_total = pressure.chunk * static_cast<std::size_t>(slab.size);
    if (slab.size == 1) {
        // In the one-rank layout the z slab and y pencil have the same
        // [k][j][kx] ordering.  Keep the projection entirely on the GPU: this
        // avoids two packing passes, two synchronizations, and any dependency
        // on CUDA-aware MPI for the single-GPU solver.
        check_cuda(cudaMemcpyAsync(
            pressure.y_pencil.data(), pressure.div_hat.data(),
            pressure.spectral_count * sizeof(cufftDoubleComplex),
            cudaMemcpyDeviceToDevice),
            "single-GPU divergence-to-pressure copy");
    } else {
        pack_z_slab_to_y_pencil_kernel<<<blocks_for(alltoall_total, threads), threads>>>(
            pressure.div_hat.data(), pressure.transpose_send.data(), alltoall_total,
            slab.size, slab.k_count, params.ny, params.nkx(), pressure.nj);
        check_cuda(cudaGetLastError(), "z-slab to y-pencil pack kernel launch");
        check_cuda(cudaDeviceSynchronize(), "z-slab to y-pencil pack completion");
        mpi_alltoall_device_complex(
            pressure.transpose_send, pressure.transpose_recv, pressure.chunk, comm);

        unpack_y_pencil_kernel<<<blocks_for(alltoall_total, threads), threads>>>(
            pressure.transpose_recv.data(), pressure.y_pencil.data(), alltoall_total,
            slab.k_count, params.nkx(), pressure.nj);
        check_cuda(cudaGetLastError(), "y-pencil unpack kernel launch");
    }

    const int columns = params.nkx() * pressure.nj;
    pressure_solve_y_pencil_kernel<<<blocks_for(static_cast<std::size_t>(columns), 128), 128>>>(
        pressure.y_pencil.data(),
        pressure.thomas_cp.data(),
        pressure.thomas_dp.data(),
        columns,
        params.nkx(),
        pressure.nj,
        params.ny,
        params.nz,
        params.nx,
        slab.rank,
        params.lx,
        params.ly,
        1.0 / (params.dz() * params.dz()),
        1.0 / params.dt);
    check_cuda(cudaGetLastError(), "distributed pressure Thomas solve kernel launch");

    if (slab.size == 1) {
        check_cuda(cudaMemcpyAsync(
            pressure.p_hat.data(), pressure.y_pencil.data(),
            pressure.spectral_count * sizeof(cufftDoubleComplex),
            cudaMemcpyDeviceToDevice),
            "single-GPU pressure-to-spectrum copy");
    } else {
        pack_y_pencil_to_z_slab_kernel<<<blocks_for(alltoall_total, threads), threads>>>(
            pressure.y_pencil.data(), pressure.transpose_send.data(), alltoall_total,
            slab.k_count, params.nkx(), pressure.nj);
        check_cuda(cudaGetLastError(), "y-pencil to z-slab pack kernel launch");
        check_cuda(cudaDeviceSynchronize(), "y-pencil to z-slab pack completion");
        mpi_alltoall_device_complex(
            pressure.transpose_send, pressure.transpose_recv, pressure.chunk, comm);

        unpack_z_slab_kernel<<<blocks_for(alltoall_total, threads), threads>>>(
            pressure.transpose_recv.data(), pressure.p_hat.data(), alltoall_total,
            slab.k_count, params.ny, params.nkx(), pressure.nj);
        check_cuda(cudaGetLastError(), "z-slab pressure unpack kernel launch");
    }

    spectral_pressure_gradient_kernel<<<blocks_for(pressure.spectral_count, threads), threads>>>(
        pressure.p_hat.data(),
        pressure.dpdx_hat.data(),
        pressure.dpdy_hat.data(),
        pressure.spectral_count,
        params.nkx(),
        params.ny,
        params.nx,
        params.lx,
        params.ly);
    check_cuda(cudaGetLastError(), "distributed pressure gradient kernel launch");

    check_cufft(cufftExecZ2D(pressure.c2r.handle, pressure.p_hat.data(), pressure.p_owned.data()), "pressure local inverse transform");
    check_cufft(cufftExecZ2D(pressure.c2r.handle, pressure.dpdx_hat.data(), pressure.dpdx_owned.data()), "dpdx local inverse transform");
    check_cufft(cufftExecZ2D(pressure.c2r.handle, pressure.dpdy_hat.data(), pressure.dpdy_owned.data()), "dpdy local inverse transform");

    apply_center_projection_kernel<<<blocks_for(pressure.owned_real_count, threads), threads>>>(
        state.u.values.data(),
        state.v.values.data(),
        state.p.values.data(),
        pressure.p_owned.data(),
        pressure.dpdx_owned.data(),
        pressure.dpdy_owned.data(),
        pressure.owned_real_count,
        params.nx,
        params.ny,
        slab.k_begin,
        state.u.plane_begin,
        params.dt,
        fft_scale);
    check_cuda(cudaGetLastError(), "distributed horizontal projection kernel launch");
    check_cuda(cudaDeviceSynchronize(), "center projection completion");

    exchange_field_halo(state.p, 600, params, slab, comm, halo);

    const std::size_t face_owned_count = static_cast<std::size_t>(slab.face_count) * plane_size(params);
    apply_vertical_projection_kernel<<<blocks_for(face_owned_count, threads), threads>>>(
        state.w.values.data(),
        state.p.values.data(),
        face_owned_count,
        params.nx,
        params.ny,
        slab.face_begin,
        state.w.plane_begin,
        state.p.plane_begin,
        params.nz,
        params.dt,
        inv_dz);
    check_cuda(cudaGetLastError(), "distributed vertical projection kernel launch");
    check_cuda(cudaDeviceSynchronize(), "vertical projection completion");

    exchange_state_halos(state, params, slab, comm, halo);
}

std::size_t local_offset(const DeviceLocalField& field, const Params& params, int i, int j, int k) {
    return static_cast<std::size_t>(k - field.plane_begin) * plane_size(params)
        + static_cast<std::size_t>(j) * static_cast<std::size_t>(params.nx)
        + static_cast<std::size_t>(i);
}

void upload_initialized_state(CudaMpiState& state, const Params& params, const Slab& slab) {
    std::vector<double> u(state.u.values.size(), 0.0);
    std::vector<double> v(state.v.values.size(), 0.0);
    std::vector<double> w(state.w.values.size(), 0.0);
    std::vector<double> p(state.p.values.size(), 0.0);
    std::vector<double> theta(state.theta.values.size(), 0.0);
    std::vector<double> theta_l(state.theta_l.values.size(), 0.0);
    std::vector<double> qt(state.qt.values.size(), 0.0);
    std::vector<double> qv(state.qv.values.size(), 0.0);
    std::vector<double> ql(state.ql.values.size(), 0.0);
    std::vector<double> base_pressure(static_cast<std::size_t>(params.nz), 0.0);

    if (params.initial_condition == "taylor_green") {
        for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
            const double z = (static_cast<double>(k) + 0.5) * params.dz();
            (void)z;
            for (int j = 0; j < params.ny; ++j) {
                const double y = static_cast<double>(j) * params.dy();
                for (int i = 0; i < params.nx; ++i) {
                    const double x = static_cast<double>(i) * params.dx();
                    const std::size_t n = local_offset(state.u, params, i, j, k);
                    u[n] = std::sin(x) * std::cos(y);
                    v[n] = -std::cos(x) * std::sin(y);
                    theta[local_offset(state.theta, params, i, j, k)] = params.theta0 + params.theta_initial_gradient * z;
                }
            }
        }
    } else if (params.initial_condition == "largeeddy1993") {
        if (params.surface_theta_flux <= 0.0) {
            throw std::runtime_error("largeeddy1993 initial condition requires positive surface_theta_flux");
        }
        const double zi1 = params.largeeddy_initial_zi1_fraction * params.z_i;
        const double wstar = std::cbrt((params.g / params.theta0) * params.surface_theta_flux * params.z_i);
        const double theta_star = params.surface_theta_flux / wstar;
        std::mt19937 rng(static_cast<std::mt19937::result_type>(params.random_seed));
        std::uniform_real_distribution<double> uniform(-0.5, 0.5);
        for (int k = 0; k < params.nz; ++k) {
            const double z = (static_cast<double>(k) + 0.5) * params.dz();
            const double lower_weight = std::max(1.0 - z / zi1, 0.0);
            const bool in_mixed_layer = z < zi1;
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const double perturb = 0.1 * uniform(rng) * lower_weight;
                    if (owns_center_plane(slab, k)) {
                        const std::size_t n = local_offset(state.u, params, i, j, k);
                        u[n] = 0.0;
                        v[n] = 0.0;
                        theta[local_offset(state.theta, params, i, j, k)] = in_mixed_layer
                            ? params.theta0 + perturb * theta_star
                            : params.theta0 + (z - zi1) * params.theta_initial_gradient;
                    }
                }
            }
        }
        for (int k = 1; k < params.nz; ++k) {
            const double z = static_cast<double>(k) * params.dz();
            const double lower_weight = std::max(1.0 - z / zi1, 0.0);
            const bool in_mixed_layer = z < zi1;
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const double perturb = 0.1 * uniform(rng) * lower_weight;
                    if (owns_face_plane(slab, k)) {
                        w[local_offset(state.w, params, i, j, k)] = in_mixed_layer ? perturb * wstar : 0.0;
                    }
                }
            }
        }
    } else if (params.initial_condition == "neutral_ekman") {
        const double perturbation_height = params.initial_perturbation_height > 0.0
            ? params.initial_perturbation_height
            : 0.25 * params.lz;
        std::mt19937 rng(static_cast<std::mt19937::result_type>(params.random_seed));
        std::uniform_real_distribution<double> uniform(-0.5, 0.5);
        for (int k = 0; k < params.nz; ++k) {
            const double z = (static_cast<double>(k) + 0.5) * params.dz();
            const double lower_weight = perturbation_height > 0.0 ? std::max(1.0 - z / perturbation_height, 0.0) : 0.0;
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const double u_perturb = params.initial_velocity_perturbation * lower_weight * uniform(rng);
                    const double v_perturb = params.initial_velocity_perturbation * lower_weight * uniform(rng);
                    if (owns_center_plane(slab, k)) {
                        const std::size_t n = local_offset(state.u, params, i, j, k);
                        u[n] = params.geostrophic_u + u_perturb;
                        v[n] = params.geostrophic_v + v_perturb;
                        theta[local_offset(state.theta, params, i, j, k)] = params.theta0 + params.theta_initial_gradient * z;
                    }
                }
            }
        }
        for (int k = 1; k < params.nz; ++k) {
            const double z = static_cast<double>(k) * params.dz();
            const double lower_weight = perturbation_height > 0.0 ? std::max(1.0 - z / perturbation_height, 0.0) : 0.0;
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const double perturb = params.initial_velocity_perturbation * lower_weight * uniform(rng);
                    if (owns_face_plane(slab, k)) {
                        w[local_offset(state.w, params, i, j, k)] = perturb;
                    }
                }
            }
        }
    } else if (params.initial_condition == "bomex") {
        std::mt19937 rng(static_cast<std::mt19937::result_type>(params.random_seed));
        std::uniform_real_distribution<double> uniform(-1.0, 1.0);
        for (int k = 0; k < params.nz; ++k) {
            const double z = (static_cast<double>(k) + 0.5) * params.dz();
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const bool perturb = params.initial_perturbation_height > 0.0
                        ? z < params.initial_perturbation_height : k < 4;
                    const double theta_perturbation = perturb
                        ? params.bomex_theta_perturbation * uniform(rng) : 0.0;
                    const double qt_perturbation = perturb
                        ? params.bomex_qt_perturbation * uniform(rng) : 0.0;
                    if (owns_center_plane(slab, k)) {
                        const std::size_t n = local_offset(state.u, params, i, j, k);
                        u[n] = bomex_initial_u(z);
                        v[n] = 0.0;
                        const double theta_l_value = bomex_initial_theta_l(z) + theta_perturbation;
                        const double qt_value = std::max(0.0, bomex_initial_qt(z) + qt_perturbation);
                        theta_l[local_offset(state.theta_l, params, i, j, k)] = theta_l_value;
                        qt[local_offset(state.qt, params, i, j, k)] = qt_value;
                    }
                }
            }
        }
    } else {
        throw std::runtime_error("unsupported initial_condition: " + params.initial_condition);
    }

    if (params.moisture_enabled) {
        for (int k = 0; k < params.nz; ++k) {
            const double z = (static_cast<double>(k) + 0.5) * params.dz();
            base_pressure[static_cast<std::size_t>(k)] = hydrostatic_base_pressure(
                z, params.surface_pressure, params.theta0, params.g);
        }
        for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
            for (int j = 0; j < params.ny; ++j) {
                for (int i = 0; i < params.nx; ++i) {
                    const std::size_t nt = local_offset(state.theta_l, params, i, j, k);
                    if (params.initial_condition != "bomex") {
                        theta_l[nt] = theta[local_offset(state.theta, params, i, j, k)];
                        qt[nt] = std::max(0.0, params.qv0
                            + params.qv_initial_gradient * (static_cast<double>(k) + 0.5) * params.dz());
                    }
                    const MoistThermodynamicState moist = saturation_adjustment(
                        theta_l[nt], qt[nt], base_pressure[static_cast<std::size_t>(k)]);
                    theta[local_offset(state.theta, params, i, j, k)] = moist.potential_temperature;
                    qv[local_offset(state.qv, params, i, j, k)] = moist.water_vapor_mixing_ratio;
                    ql[local_offset(state.ql, params, i, j, k)] = moist.liquid_water_mixing_ratio;
                }
            }
        }
    }

    state.u.values.copy_from(u, "local u");
    state.v.values.copy_from(v, "local v");
    state.w.values.copy_from(w, "local w");
    state.p.values.copy_from(p, "local p");
    state.theta.values.copy_from(theta, "local theta");
    state.theta_l.values.copy_from(theta_l, "local theta_l");
    state.qt.values.copy_from(qt, "local qt");
    state.qv.values.copy_from(qv, "local qv");
    state.ql.values.copy_from(ql, "local ql");
    state.base_pressure.copy_from(base_pressure, "base pressure");
}

void download_single_gpu_state(
    const CudaMpiState& device_state,
    FlowState& host_state,
    const Params& params,
    const Slab& slab) {
    if (slab.size != 1 || slab.k_begin != 0 || slab.k_count != params.nz) {
        throw std::runtime_error("full CUDA state download is available only in single-GPU mode");
    }
    device_state.u.values.copy_to(host_state.u, "host diagnostic u");
    device_state.v.values.copy_to(host_state.v, "host diagnostic v");
    device_state.w.values.copy_to(host_state.w, "host diagnostic w");
    device_state.p.values.copy_to(host_state.p, "host diagnostic p");
    device_state.theta.values.copy_to(host_state.theta, "host diagnostic theta");
    device_state.theta_l.values.copy_to(host_state.theta_l, "host diagnostic theta_l");
    device_state.qt.values.copy_to(host_state.qt, "host diagnostic qt");
    device_state.qv.values.copy_to(host_state.qv, "host diagnostic qv");
    device_state.ql.values.copy_to(host_state.ql, "host diagnostic ql");
    device_state.base_pressure.copy_to(host_state.base_pressure, "host base pressure");
    host_state.step_count = device_state.step_count;
    host_state.has_rhs_prev = device_state.has_rhs_prev;
}

void add_single_gpu_sgs_bomex_sample(
    BomexAccumulator& accumulator,
    const FlowState& state,
    const RhsWorkspace& work,
    const Params& params) {
    std::vector<double> nu_t;
    std::vector<double> strain;
    std::vector<double> theta_kappa;
    std::vector<double> qt_kappa;
    std::vector<double> txz;
    std::vector<double> tyz;
    work.nu_t.copy_to(nu_t, "BOMEX diagnostic nu_t");
    work.strain.copy_to(strain, "BOMEX diagnostic strain");
    work.theta_kappa.copy_to(theta_kappa, "BOMEX diagnostic theta_l diffusivity");
    work.qt_kappa.copy_to(qt_kappa, "BOMEX diagnostic qt diffusivity");
    work.txz.copy_to(txz, "BOMEX diagnostic txz");
    work.tyz.copy_to(tyz, "BOMEX diagnostic tyz");
    const double inverse_plane = 1.0 / static_cast<double>(params.nx * params.ny);
    const double surface_qt_flux = params.surface_qv_flux / std::pow(1.0 - 0.017, 2.0);
    const std::size_t plane = plane_size(params);
    double surface_mean_qv = 0.0;
    double surface_mean_theta_l = 0.0;
    for (int j = 0; j < params.ny; ++j) {
        for (int i = 0; i < params.nx; ++i) {
            const std::size_t n = idx(params, i, j, 0);
            surface_mean_qv += state.qv[n] * inverse_plane;
            surface_mean_theta_l += state.theta_l[n] * inverse_plane;
        }
    }
    const double surface_theta_v_flux =
        (1.0 + 0.61 * surface_mean_qv) * params.surface_theta_flux
        + 0.61 * surface_mean_theta_l * surface_qt_flux;

    auto scalar_face_flux = [&](const Field& scalar, const std::vector<double>& kappa,
                                int face_k, int i, int j, double surface_flux) {
        if (face_k <= 0) return surface_flux;
        if (face_k >= params.nz) return 0.0;
        const std::size_t lower = idx(params, i, j, face_k - 1);
        const std::size_t upper = idx(params, i, j, face_k);
        return -0.5 * (kappa[lower] + kappa[upper])
            * (scalar[upper] - scalar[lower]) / params.dz();
    };
    auto moist_diagnostic_flux = [&](int face_k, int i, int j) {
        if (face_k <= 0) {
            return std::pair<double, double>{0.0, surface_theta_v_flux};
        }
        if (face_k >= params.nz) return std::pair<double, double>{0.0, 0.0};
        const std::size_t lower = idx(params, i, j, face_k - 1);
        const std::size_t upper = idx(params, i, j, face_k);
        const double theta_flux = scalar_face_flux(
            state.theta_l, theta_kappa, face_k, i, j, params.surface_theta_flux);
        const double qt_flux = scalar_face_flux(
            state.qt, qt_kappa, face_k, i, j, surface_qt_flux);
        const MoistConservedJacobians lower_jacobian = moist_conserved_jacobians(
            state.theta[lower], state.qv[lower], state.ql[lower],
            state.base_pressure[static_cast<std::size_t>(face_k - 1)]);
        const MoistConservedJacobians upper_jacobian = moist_conserved_jacobians(
            state.theta[upper], state.qv[upper], state.ql[upper],
            state.base_pressure[static_cast<std::size_t>(face_k)]);
        const double dql_dtheta = 0.5 * (
            lower_jacobian.dliquid_water_dtheta_l + upper_jacobian.dliquid_water_dtheta_l);
        const double dql_dqt = 0.5 * (
            lower_jacobian.dliquid_water_dtotal_water + upper_jacobian.dliquid_water_dtotal_water);
        const double dtv_dtheta = 0.5 * (
            lower_jacobian.dvirtual_theta_dtheta_l + upper_jacobian.dvirtual_theta_dtheta_l);
        const double dtv_dqt = 0.5 * (
            lower_jacobian.dvirtual_theta_dtotal_water + upper_jacobian.dvirtual_theta_dtotal_water);
        return std::pair<double, double>{
            dql_dtheta * theta_flux + dql_dqt * qt_flux,
            dtv_dtheta * theta_flux + dtv_dqt * qt_flux};
    };

    for (int k = 0; k < params.nz; ++k) {
        double mean_u = 0.0;
        double mean_v = 0.0;
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                mean_u += state.u[n] * inverse_plane;
                mean_v += state.v[n] * inverse_plane;
            }
        }
        const double speed = std::hypot(mean_u, mean_v);
        const double surface_uw_flux = speed > 0.0
            ? -params.u_fric * params.u_fric * mean_u / speed : 0.0;
        auto uw_face_flux = [&](int face_k, int i, int j) {
            if (face_k <= 0) return surface_uw_flux;
            if (face_k >= params.nz) return 0.0;
            return -txz[static_cast<std::size_t>(face_k) * plane
                + static_cast<std::size_t>(j * params.nx + i)];
        };
        auto vw_face_flux = [&](int face_k, int i, int j) {
            if (face_k <= 0 || face_k >= params.nz) return 0.0;
            return -tyz[static_cast<std::size_t>(face_k) * plane
                + static_cast<std::size_t>(j * params.nx + i)];
        };
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = idx(params, i, j, k);
                const double strain_squared = strain[n] * strain[n];
                accumulator.mean_eddy_viscosity[static_cast<std::size_t>(k)] += nu_t[n] * inverse_plane;
                accumulator.mean_strain_squared[static_cast<std::size_t>(k)] += strain_squared * inverse_plane;
                accumulator.mean_sgs_dissipation[static_cast<std::size_t>(k)] += nu_t[n] * strain_squared * inverse_plane;
                accumulator.zero_eddy_viscosity_fraction[static_cast<std::size_t>(k)] +=
                    nu_t[n] <= 1.0e-14 ? inverse_plane : 0.0;
                accumulator.mean_theta_l_scalar_diffusivity[static_cast<std::size_t>(k)] += theta_kappa[n] * inverse_plane;
                accumulator.mean_qt_scalar_diffusivity[static_cast<std::size_t>(k)] += qt_kappa[n] * inverse_plane;
                accumulator.zero_theta_l_scalar_diffusivity_fraction[static_cast<std::size_t>(k)] +=
                    theta_kappa[n] <= 1.0e-14 ? inverse_plane : 0.0;
                accumulator.zero_qt_scalar_diffusivity_fraction[static_cast<std::size_t>(k)] +=
                    qt_kappa[n] <= 1.0e-14 ? inverse_plane : 0.0;
                accumulator.sgs_theta_l_flux[static_cast<std::size_t>(k)] += 0.5 * inverse_plane * (
                    scalar_face_flux(state.theta_l, theta_kappa, k, i, j, params.surface_theta_flux)
                    + scalar_face_flux(state.theta_l, theta_kappa, k + 1, i, j, params.surface_theta_flux));
                accumulator.sgs_qt_flux[static_cast<std::size_t>(k)] += 0.5 * inverse_plane * (
                    scalar_face_flux(state.qt, qt_kappa, k, i, j, surface_qt_flux)
                    + scalar_face_flux(state.qt, qt_kappa, k + 1, i, j, surface_qt_flux));
                const auto moist_lower = moist_diagnostic_flux(k, i, j);
                const auto moist_upper = moist_diagnostic_flux(k + 1, i, j);
                accumulator.sgs_ql_flux[static_cast<std::size_t>(k)] +=
                    0.5 * inverse_plane * (moist_lower.first + moist_upper.first);
                accumulator.sgs_theta_v_flux[static_cast<std::size_t>(k)] +=
                    0.5 * inverse_plane * (moist_lower.second + moist_upper.second);
                accumulator.sgs_uw_flux[static_cast<std::size_t>(k)] += 0.5 * inverse_plane * (
                    uw_face_flux(k, i, j) + uw_face_flux(k + 1, i, j));
                accumulator.sgs_vw_flux[static_cast<std::size_t>(k)] += 0.5 * inverse_plane * (
                    vw_face_flux(k, i, j) + vw_face_flux(k + 1, i, j));
            }
        }
    }
}

struct DiagnosticsLocal {
    double ke_max = 0.0;
    double div_max = 0.0;
    double cfl = 0.0;
    double qv_min = std::numeric_limits<double>::infinity();
    double qv_max = -std::numeric_limits<double>::infinity();
    double ql_max = 0.0;
    double column_water = 0.0;
};

double spectral_divergence_max_cuda_mpi(
    CudaMpiState& state,
    PressureWorkspace& pressure,
    const Params& params,
    const Slab& slab,
    MPI_Comm comm) {
    constexpr int threads = 256;
    extract_pressure_inputs_kernel<<<blocks_for(pressure.owned_real_count, threads), threads>>>(
        state.u.values.data(),
        state.v.values.data(),
        state.w.values.data(),
        pressure.u_owned.data(),
        pressure.v_owned.data(),
        pressure.dwdz_owned.data(),
        pressure.owned_real_count,
        params.nx,
        params.ny,
        slab.k_begin,
        state.u.plane_begin,
        state.w.plane_begin,
        1.0 / params.dz());
    check_cuda(cudaGetLastError(), "diagnostic pressure input extraction kernel launch");

    check_cufft(cufftExecD2Z(pressure.r2c.handle, pressure.u_owned.data(), pressure.u_hat.data()), "diagnostic u local forward transform");
    check_cufft(cufftExecD2Z(pressure.r2c.handle, pressure.v_owned.data(), pressure.v_hat.data()), "diagnostic v local forward transform");
    check_cufft(cufftExecD2Z(pressure.r2c.handle, pressure.dwdz_owned.data(), pressure.dwdz_hat.data()), "diagnostic dwdz local forward transform");

    spectral_divergence_kernel<<<blocks_for(pressure.spectral_count, threads), threads>>>(
        pressure.u_hat.data(),
        pressure.v_hat.data(),
        pressure.dwdz_hat.data(),
        pressure.div_hat.data(),
        pressure.spectral_count,
        params.nkx(),
        params.ny,
        params.nx,
        params.lx,
        params.ly);
    check_cuda(cudaGetLastError(), "diagnostic spectral divergence kernel launch");

    check_cufft(cufftExecZ2D(pressure.c2r.handle, pressure.div_hat.data(), pressure.p_owned.data()), "diagnostic divergence inverse transform");
    scale_real_kernel<<<blocks_for(pressure.owned_real_count, threads), threads>>>(
        pressure.p_owned.data(), pressure.owned_real_count, 1.0 / static_cast<double>(params.nx * params.ny));
    check_cuda(cudaGetLastError(), "diagnostic divergence scaling kernel launch");

    std::vector<double> div;
    pressure.p_owned.copy_to(div, "diagnostic spectral divergence");
    double local_div_max = 0.0;
    for (double value : div) {
        local_div_max = std::max(local_div_max, std::abs(value));
    }
    double global_div_max = 0.0;
    check_mpi(MPI_Allreduce(&local_div_max, &global_div_max, 1, MPI_DOUBLE, MPI_MAX, comm), "Allreduce spectral div");
    return global_div_max;
}

DiagnosticsLocal diagnostics_cuda_mpi(
    CudaMpiState& state,
    PressureWorkspace& pressure,
    const Params& params,
    const Slab& slab,
    MPI_Comm comm) {
    std::vector<double> u;
    std::vector<double> v;
    std::vector<double> w;
    std::vector<double> qv;
    std::vector<double> ql;
    std::vector<double> qt;
    state.u.values.copy_to(u, "diag u");
    state.v.values.copy_to(v, "diag v");
    state.w.values.copy_to(w, "diag w");
    if (params.moisture_enabled) {
        state.qv.values.copy_to(qv, "diag qv");
        state.ql.values.copy_to(ql, "diag ql");
        state.qt.values.copy_to(qt, "diag qt");
    }
    DiagnosticsLocal local;
    const double inv_dx = 1.0 / params.dx();
    const double inv_dy = 1.0 / params.dy();
    const double inv_dz = 1.0 / params.dz();
    for (int k = slab.k_begin; k < slab.k_begin + slab.k_count; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                const std::size_t n = local_offset(state.u, params, i, j, k);
                const double wc = 0.5 * (w[local_offset(state.w, params, i, j, k)] + w[local_offset(state.w, params, i, j, k + 1)]);
                local.ke_max = std::max(local.ke_max, 0.5 * (u[n] * u[n] + v[n] * v[n] + wc * wc));
                local.cfl = std::max(local.cfl, params.dt * (std::abs(u[n]) * inv_dx + std::abs(v[n]) * inv_dy + std::abs(wc) * inv_dz));
                if (params.moisture_enabled) {
                    if (!std::isfinite(qv[n]) || !std::isfinite(ql[n]) || !std::isfinite(qt[n])) {
                        local.qv_min = -std::numeric_limits<double>::infinity();
                        local.qv_max = std::numeric_limits<double>::infinity();
                        local.ql_max = std::numeric_limits<double>::infinity();
                        local.column_water = std::numeric_limits<double>::infinity();
                    } else {
                        local.qv_min = std::min(local.qv_min, qv[n]);
                        local.qv_max = std::max(local.qv_max, qv[n]);
                        local.ql_max = std::max(local.ql_max, ql[n]);
                        local.column_water += qt[n] * params.dz()
                            / static_cast<double>(params.nx * params.ny);
                    }
                }
            }
        }
    }
    DiagnosticsLocal global;
    check_mpi(MPI_Allreduce(&local.ke_max, &global.ke_max, 1, MPI_DOUBLE, MPI_MAX, comm), "Allreduce ke");
    check_mpi(MPI_Allreduce(&local.cfl, &global.cfl, 1, MPI_DOUBLE, MPI_MAX, comm), "Allreduce cfl");
    if (params.moisture_enabled) {
        check_mpi(MPI_Allreduce(&local.qv_min, &global.qv_min, 1, MPI_DOUBLE, MPI_MIN, comm), "Allreduce qv min");
        check_mpi(MPI_Allreduce(&local.qv_max, &global.qv_max, 1, MPI_DOUBLE, MPI_MAX, comm), "Allreduce qv max");
        check_mpi(MPI_Allreduce(&local.ql_max, &global.ql_max, 1, MPI_DOUBLE, MPI_MAX, comm), "Allreduce ql max");
        check_mpi(MPI_Allreduce(&local.column_water, &global.column_water, 1, MPI_DOUBLE, MPI_SUM, comm), "Allreduce column water");
    }
    global.div_max = spectral_divergence_max_cuda_mpi(state, pressure, params, slab, comm);
    return global;
}

void ensure_directory_cuda_mpi(const std::string& path) {
    if (path.empty()) {
        return;
    }
    std::string current;
    std::size_t start = 0;
    if (!path.empty() && path.front() == '/') {
        current = "/";
        start = 1;
    }
    while (start <= path.size()) {
        const std::size_t slash = path.find('/', start);
        const std::string part = path.substr(start, slash == std::string::npos ? std::string::npos : slash - start);
        if (!part.empty()) {
            if (!current.empty() && current.back() != '/') {
                current += '/';
            }
            current += part;
            if (::mkdir(current.c_str(), 0775) != 0 && errno != EEXIST) {
                throw std::runtime_error("failed to create directory: " + current);
            }
        }
        if (slash == std::string::npos) {
            break;
        }
        start = slash + 1;
    }
}

std::string join_path_cuda_mpi(const std::string& directory, const std::string& name) {
    if (directory.empty()) {
        return name;
    }
    return directory.back() == '/' ? directory + name : directory + "/" + name;
}

bool should_write_frame_cuda_mpi(const Params& params, int step_number) {
    if (!params.frame_dump_enabled) {
        return false;
    }
    if (step_number < params.frame_dump_start_step) {
        return false;
    }
    if (params.frame_dump_end_step >= 0 && step_number > params.frame_dump_end_step) {
        return false;
    }
    return step_number % params.frame_dump_every == 0 || step_number == params.steps;
}

std::string frame_filename_cuda_mpi(const Params& params, int step_number) {
    std::ostringstream name;
    name << "fields_step_" << std::setw(6) << std::setfill('0') << step_number << ".h5";
    return name.str();
}

#ifdef WIRELES_HAVE_HDF5
void check_hdf5(herr_t status, const char* action) {
    if (status < 0) {
        throw std::runtime_error(std::string("HDF5 ") + action + " failed");
    }
}

hid_t check_hdf5_id(hid_t id, const char* action) {
    if (id < 0) {
        throw std::runtime_error(std::string("HDF5 ") + action + " failed");
    }
    return id;
}

void write_hdf5_scalar_attr(hid_t object, const char* name, hid_t type, const void* value) {
    hid_t space = check_hdf5_id(H5Screate(H5S_SCALAR), "create scalar attr dataspace");
    hid_t attr = check_hdf5_id(H5Acreate2(object, name, type, space, H5P_DEFAULT, H5P_DEFAULT), "create scalar attr");
    check_hdf5(H5Awrite(attr, type, value), "write scalar attr");
    check_hdf5(H5Aclose(attr), "close scalar attr");
    check_hdf5(H5Sclose(space), "close scalar attr dataspace");
}

void fill_hdf5_center_buffer(
    const DeviceLocalField& field,
    DeviceBuffer<double>& device_buffer,
    std::vector<double>& host_buffer,
    const Params& params,
    const Slab& slab) {
    const std::size_t count = static_cast<std::size_t>(params.nx)
        * static_cast<std::size_t>(params.ny)
        * static_cast<std::size_t>(slab.k_count);
    constexpr int threads = 256;
    extract_center_hdf5_layout_kernel<<<blocks_for(count, threads), threads>>>(
        field.values.data(),
        device_buffer.data(),
        count,
        params.nx,
        params.ny,
        slab.k_begin,
        field.plane_begin);
    check_cuda(cudaGetLastError(), "CUDA-MPI HDF5 center field packing kernel launch");
    device_buffer.copy_to(host_buffer, "CUDA-MPI HDF5 center field");
}

void fill_hdf5_w_buffer(
    const CudaMpiState& state,
    DeviceBuffer<double>& device_buffer,
    std::vector<double>& host_buffer,
    const Params& params,
    const Slab& slab) {
    const std::size_t count = static_cast<std::size_t>(params.nx)
        * static_cast<std::size_t>(params.ny)
        * static_cast<std::size_t>(slab.k_count);
    constexpr int threads = 256;
    extract_w_center_hdf5_layout_kernel<<<blocks_for(count, threads), threads>>>(
        state.w.values.data(),
        device_buffer.data(),
        count,
        params.nx,
        params.ny,
        slab.k_begin,
        state.w.plane_begin);
    check_cuda(cudaGetLastError(), "CUDA-MPI HDF5 w field packing kernel launch");
    device_buffer.copy_to(host_buffer, "CUDA-MPI HDF5 w field");
}

void fill_hdf5_center_slice_buffer(
    const DeviceLocalField& field,
    DeviceBuffer<double>& device_buffer,
    std::vector<double>& host_buffer,
    const Params& params,
    const Slab& slab,
    int y_index) {
    const std::size_t count = static_cast<std::size_t>(params.nx)
        * static_cast<std::size_t>(slab.k_count);
    constexpr int threads = 256;
    extract_center_hdf5_slice_kernel<<<blocks_for(count, threads), threads>>>(
        field.values.data(),
        device_buffer.data(),
        count,
        params.nx,
        params.ny,
        slab.k_begin,
        field.plane_begin,
        y_index);
    check_cuda(cudaGetLastError(), "CUDA-MPI HDF5 center slice packing kernel launch");
    device_buffer.copy_to(host_buffer, "CUDA-MPI HDF5 center slice");
}

void fill_hdf5_w_slice_buffer(
    const CudaMpiState& state,
    DeviceBuffer<double>& device_buffer,
    std::vector<double>& host_buffer,
    const Params& params,
    const Slab& slab,
    int y_index) {
    const std::size_t count = static_cast<std::size_t>(params.nx)
        * static_cast<std::size_t>(slab.k_count);
    constexpr int threads = 256;
    extract_w_center_hdf5_slice_kernel<<<blocks_for(count, threads), threads>>>(
        state.w.values.data(),
        device_buffer.data(),
        count,
        params.nx,
        params.ny,
        slab.k_begin,
        state.w.plane_begin,
        y_index);
    check_cuda(cudaGetLastError(), "CUDA-MPI HDF5 w slice packing kernel launch");
    device_buffer.copy_to(host_buffer, "CUDA-MPI HDF5 w slice");
}

void write_hdf5_field_dataset(
    hid_t group,
    hid_t dxpl,
    const char* name,
    const std::vector<double>& local_values,
    const Params& params,
    const Slab& slab) {
    hsize_t file_dims[3] = {
        static_cast<hsize_t>(params.nx),
        static_cast<hsize_t>(params.ny),
        static_cast<hsize_t>(params.nz),
    };
    hsize_t mem_dims[3] = {
        static_cast<hsize_t>(params.nx),
        static_cast<hsize_t>(params.ny),
        static_cast<hsize_t>(slab.k_count),
    };
    hsize_t start[3] = {0, 0, static_cast<hsize_t>(slab.k_begin)};
    hsize_t count[3] = {
        static_cast<hsize_t>(params.nx),
        static_cast<hsize_t>(params.ny),
        static_cast<hsize_t>(slab.k_count),
    };

    hid_t file_space = check_hdf5_id(H5Screate_simple(3, file_dims, nullptr), "create field file dataspace");
    hid_t dataset = check_hdf5_id(H5Dcreate2(group, name, H5T_NATIVE_DOUBLE, file_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
        "create field dataset");
    hid_t mem_space = check_hdf5_id(H5Screate_simple(3, mem_dims, nullptr), "create field memory dataspace");
    check_hdf5(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, nullptr, count, nullptr), "select field hyperslab");
    check_hdf5(H5Dwrite(dataset, H5T_NATIVE_DOUBLE, mem_space, file_space, dxpl, local_values.data()), "write field hyperslab");
    check_hdf5(H5Sclose(mem_space), "close field memory dataspace");
    check_hdf5(H5Dclose(dataset), "close field dataset");
    check_hdf5(H5Sclose(file_space), "close field file dataspace");
}

void write_hdf5_field_slice_dataset(
    hid_t group,
    hid_t dxpl,
    const char* name,
    const std::vector<double>& local_values,
    const Params& params,
    const Slab& slab) {
    hsize_t file_dims[3] = {
        static_cast<hsize_t>(params.nx),
        1,
        static_cast<hsize_t>(params.nz),
    };
    hsize_t mem_dims[3] = {
        static_cast<hsize_t>(params.nx),
        1,
        static_cast<hsize_t>(slab.k_count),
    };
    hsize_t start[3] = {0, 0, static_cast<hsize_t>(slab.k_begin)};
    hsize_t count[3] = {
        static_cast<hsize_t>(params.nx),
        1,
        static_cast<hsize_t>(slab.k_count),
    };

    hid_t file_space = check_hdf5_id(H5Screate_simple(3, file_dims, nullptr), "create field slice file dataspace");
    hid_t dataset = check_hdf5_id(H5Dcreate2(group, name, H5T_NATIVE_DOUBLE, file_space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
        "create field slice dataset");
    hid_t mem_space = check_hdf5_id(H5Screate_simple(3, mem_dims, nullptr), "create field slice memory dataspace");
    check_hdf5(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, nullptr, count, nullptr), "select field slice hyperslab");
    check_hdf5(H5Dwrite(dataset, H5T_NATIVE_DOUBLE, mem_space, file_space, dxpl, local_values.data()), "write field slice hyperslab");
    check_hdf5(H5Sclose(mem_space), "close field slice memory dataspace");
    check_hdf5(H5Dclose(dataset), "close field slice dataset");
    check_hdf5(H5Sclose(file_space), "close field slice file dataspace");
}

void write_hdf5_coord_dataset(hid_t group, const char* name, const std::vector<double>& values, int rank) {
    hsize_t dims[1] = {static_cast<hsize_t>(values.size())};
    hid_t space = check_hdf5_id(H5Screate_simple(1, dims, nullptr), "create coordinate dataspace");
    hid_t dataset = check_hdf5_id(H5Dcreate2(group, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
        "create coordinate dataset");
    if (rank == 0) {
        check_hdf5(H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data()), "write coordinate dataset");
    }
    check_hdf5(H5Dclose(dataset), "close coordinate dataset");
    check_hdf5(H5Sclose(space), "close coordinate dataspace");
}

void write_hdf5_coords(hid_t file, const Params& params, int rank, bool slice_only, int y_index) {
    std::vector<double> x(static_cast<std::size_t>(params.nx));
    std::vector<double> y(slice_only ? 1 : static_cast<std::size_t>(params.ny));
    std::vector<double> z(static_cast<std::size_t>(params.nz));
    for (int i = 0; i < params.nx; ++i) {
        x[static_cast<std::size_t>(i)] = (static_cast<double>(i) + 0.5) * params.dx();
    }
    if (slice_only) {
        y[0] = (static_cast<double>(y_index) + 0.5) * params.dy();
    } else {
        for (int j = 0; j < params.ny; ++j) {
            y[static_cast<std::size_t>(j)] = (static_cast<double>(j) + 0.5) * params.dy();
        }
    }
    for (int k = 0; k < params.nz; ++k) {
        z[static_cast<std::size_t>(k)] = (static_cast<double>(k) + 0.5) * params.dz();
    }
    hid_t coords = check_hdf5_id(H5Gcreate2(file, "coords", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), "create coords group");
    write_hdf5_coord_dataset(coords, "x", x, rank);
    write_hdf5_coord_dataset(coords, "y", y, rank);
    write_hdf5_coord_dataset(coords, "z", z, rank);
    check_hdf5(H5Gclose(coords), "close coords group");
}
#endif

void write_frame_dump_cuda_mpi(const CudaMpiState& state, const Params& params, const Slab& slab, MPI_Comm comm, int step_number) {
#ifndef WIRELES_HAVE_HDF5
    (void)state;
    (void)params;
    (void)slab;
    (void)comm;
    (void)step_number;
    throw std::runtime_error("CUDA-MPI transient field output requires configuring with -DWIRELES_ENABLE_HDF5=ON and a parallel HDF5 build");
#else
    if (slab.rank == 0) {
        ensure_directory_cuda_mpi(params.frame_dump_output_dir);
    }
    check_mpi(MPI_Barrier(comm), "barrier before HDF5 field output");

    const std::string path = join_path_cuda_mpi(params.frame_dump_output_dir, frame_filename_cuda_mpi(params, step_number));
    hid_t fapl = check_hdf5_id(H5Pcreate(H5P_FILE_ACCESS), "create file access property list");
    check_hdf5(H5Pset_fapl_mpio(fapl, comm, MPI_INFO_NULL), "set MPI-IO file access");
    hid_t file = check_hdf5_id(H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl), "create transient field file");
    check_hdf5(H5Pclose(fapl), "close file access property list");

    const double time = static_cast<double>(step_number) * params.dt;
    write_hdf5_scalar_attr(file, "step", H5T_NATIVE_INT, &step_number);
    write_hdf5_scalar_attr(file, "time", H5T_NATIVE_DOUBLE, &time);
    write_hdf5_scalar_attr(file, "dt", H5T_NATIVE_DOUBLE, &params.dt);
    write_hdf5_scalar_attr(file, "nx", H5T_NATIVE_INT, &params.nx);
    write_hdf5_scalar_attr(file, "nz", H5T_NATIVE_INT, &params.nz);
    write_hdf5_scalar_attr(file, "lx", H5T_NATIVE_DOUBLE, &params.lx);
    write_hdf5_scalar_attr(file, "ly", H5T_NATIVE_DOUBLE, &params.ly);
    write_hdf5_scalar_attr(file, "lz", H5T_NATIVE_DOUBLE, &params.lz);
    const int y_index = params.frame_dump_y_index < 0 ? params.ny / 2 : params.frame_dump_y_index;
    const int ny_attr = params.frame_dump_slice_only ? 1 : params.ny;
    const int slice_only_attr = params.frame_dump_slice_only ? 1 : 0;
    write_hdf5_scalar_attr(file, "ny", H5T_NATIVE_INT, &ny_attr);
    write_hdf5_scalar_attr(file, "source_ny", H5T_NATIVE_INT, &params.ny);
    write_hdf5_scalar_attr(file, "frame_y_index", H5T_NATIVE_INT, &y_index);
    write_hdf5_scalar_attr(file, "frame_slice_only", H5T_NATIVE_INT, &slice_only_attr);
    write_hdf5_coords(file, params, slab.rank, params.frame_dump_slice_only, y_index);

    hid_t fields = check_hdf5_id(H5Gcreate2(file, "fields", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), "create fields group");
    hid_t dxpl = check_hdf5_id(H5Pcreate(H5P_DATASET_XFER), "create dataset transfer property list");
    check_hdf5(H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE), "set collective dataset transfer");

    const std::size_t local_count = static_cast<std::size_t>(params.nx)
        * static_cast<std::size_t>(params.frame_dump_slice_only ? 1 : params.ny)
        * static_cast<std::size_t>(slab.k_count);
    DeviceBuffer<double> device_buffer(local_count);
    std::vector<double> host_buffer;
    if (params.frame_dump_slice_only) {
        fill_hdf5_center_slice_buffer(state.u, device_buffer, host_buffer, params, slab, y_index);
        write_hdf5_field_slice_dataset(fields, dxpl, "u", host_buffer, params, slab);
        fill_hdf5_center_slice_buffer(state.v, device_buffer, host_buffer, params, slab, y_index);
        write_hdf5_field_slice_dataset(fields, dxpl, "v", host_buffer, params, slab);
        fill_hdf5_w_slice_buffer(state, device_buffer, host_buffer, params, slab, y_index);
        write_hdf5_field_slice_dataset(fields, dxpl, "w", host_buffer, params, slab);
        fill_hdf5_center_slice_buffer(state.p, device_buffer, host_buffer, params, slab, y_index);
        write_hdf5_field_slice_dataset(fields, dxpl, "p", host_buffer, params, slab);
        fill_hdf5_center_slice_buffer(state.theta, device_buffer, host_buffer, params, slab, y_index);
        write_hdf5_field_slice_dataset(fields, dxpl, "theta", host_buffer, params, slab);
        if (params.moisture_enabled) {
            fill_hdf5_center_slice_buffer(state.theta_l, device_buffer, host_buffer, params, slab, y_index);
            write_hdf5_field_slice_dataset(fields, dxpl, "theta_l", host_buffer, params, slab);
            fill_hdf5_center_slice_buffer(state.qt, device_buffer, host_buffer, params, slab, y_index);
            write_hdf5_field_slice_dataset(fields, dxpl, "qt", host_buffer, params, slab);
            fill_hdf5_center_slice_buffer(state.qv, device_buffer, host_buffer, params, slab, y_index);
            write_hdf5_field_slice_dataset(fields, dxpl, "qv", host_buffer, params, slab);
            fill_hdf5_center_slice_buffer(state.ql, device_buffer, host_buffer, params, slab, y_index);
            write_hdf5_field_slice_dataset(fields, dxpl, "ql", host_buffer, params, slab);
        }
    } else {
        fill_hdf5_center_buffer(state.u, device_buffer, host_buffer, params, slab);
        write_hdf5_field_dataset(fields, dxpl, "u", host_buffer, params, slab);
        fill_hdf5_center_buffer(state.v, device_buffer, host_buffer, params, slab);
        write_hdf5_field_dataset(fields, dxpl, "v", host_buffer, params, slab);
        fill_hdf5_w_buffer(state, device_buffer, host_buffer, params, slab);
        write_hdf5_field_dataset(fields, dxpl, "w", host_buffer, params, slab);
        fill_hdf5_center_buffer(state.p, device_buffer, host_buffer, params, slab);
        write_hdf5_field_dataset(fields, dxpl, "p", host_buffer, params, slab);
        fill_hdf5_center_buffer(state.theta, device_buffer, host_buffer, params, slab);
        write_hdf5_field_dataset(fields, dxpl, "theta", host_buffer, params, slab);
        if (params.moisture_enabled) {
            fill_hdf5_center_buffer(state.theta_l, device_buffer, host_buffer, params, slab);
            write_hdf5_field_dataset(fields, dxpl, "theta_l", host_buffer, params, slab);
            fill_hdf5_center_buffer(state.qt, device_buffer, host_buffer, params, slab);
            write_hdf5_field_dataset(fields, dxpl, "qt", host_buffer, params, slab);
            fill_hdf5_center_buffer(state.qv, device_buffer, host_buffer, params, slab);
            write_hdf5_field_dataset(fields, dxpl, "qv", host_buffer, params, slab);
            fill_hdf5_center_buffer(state.ql, device_buffer, host_buffer, params, slab);
            write_hdf5_field_dataset(fields, dxpl, "ql", host_buffer, params, slab);
        }
    }

    check_hdf5(H5Pclose(dxpl), "close dataset transfer property list");
    check_hdf5(H5Gclose(fields), "close fields group");
    check_hdf5(H5Fclose(file), "close transient field file");
#endif
}

void require_cuda_mpi_supported(const Params& params) {
    if (params.sgs_model != "none" && params.sgs_model != "smagorinsky"
        && params.sgs_model != "amd") {
        throw std::runtime_error("CUDA-MPI slab mode supports SGS none, smagorinsky, or amd");
    }
    if (params.scalar_sgs_model != "fixed_prandtl"
        && params.scalar_sgs_model != "fixed_smagorinsky"
        && params.scalar_sgs_model != "amd") {
        throw std::runtime_error("CUDA-MPI slab mode supports fixed or AMD scalar SGS models");
    }
    if (params.momentum_wall_model != "none" && params.momentum_wall_model != "abl") {
        throw std::runtime_error("CUDA-MPI slab mode supports --wall none or --wall abl");
    }
    if (params.wall_stress_model != "dynamic_neutral" && params.wall_stress_model != "prescribed_ustar") {
        throw std::runtime_error("CUDA-MPI slab mode supports wall stress dynamic_neutral or prescribed_ustar");
    }
    if (params.time_scheme != "euler" && params.time_scheme != "ab2"
        && params.time_scheme != "ab3") {
        throw std::runtime_error("CUDA-MPI slab mode supports --time-scheme euler, ab2, or ab3");
    }
    if (params.momentum_advection_form != "advective"
        && params.momentum_advection_form != "rotational") {
        throw std::runtime_error("CUDA-MPI slab mode supports advective or rotational momentum advection");
    }
    if (params.dealiasing != "sharp") {
        throw std::runtime_error("CUDA-MPI slab mode currently supports NCAR sharp 2/3 dealiasing only");
    }
}

}  // namespace

int run_cuda_mpi_slab(const Params& params, int argc, char** argv) {
    MpiRuntime mpi(argc, argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    try {
        require_cuda_mpi_supported(params);
        int device_count = 0;
        check_cuda(cudaGetDeviceCount(&device_count), "device query");
        if (device_count <= 0) {
            throw std::runtime_error("CUDA-MPI slab mode found no CUDA devices");
        }
        check_cuda(cudaSetDevice(rank % device_count), "device selection");
        const Slab slab = make_slab(params, comm);
        if ((params.moisture_enabled || params.sgs_model == "amd") && size != 1) {
            throw std::runtime_error(
                "the moist/AMD CUDA implementation is intentionally single-GPU; launch with one MPI rank");
        }
        CudaMpiState state(params, slab);
        HaloScratch halo(plane_size(params));
        PressureWorkspace pressure(params, slab);
        RhsWorkspace rhs(params, slab);
        BomexAccumulator bomex;
        BomexAccumulator bomex_last_hour;
        FlowState host_output_state(params);
        upload_initialized_state(state, params, slab);
        exchange_state_halos(state, params, slab, comm, halo);
        cuda_mpi_project(state, params, slab, comm, halo, pressure);
        DiagnosticsLocal diag = diagnostics_cuda_mpi(state, pressure, params, slab, comm);
        if (should_write_frame_cuda_mpi(params, 0)) {
            write_frame_dump_cuda_mpi(state, params, slab, comm, 0);
        }

        if (rank == 0) {
            std::cout << (size == 1
                    ? "# wireles single-GPU solver (MPI runtime interface)\n"
                    : "# wireles CUDA-aware MPI z-slab solver\n");
            std::cout << "# ranks " << size << ", devices-per-node visible " << device_count
                      << (size == 1
                            ? ", local z-slab/y-pencil pressure fast path\n"
                            : ", z-slab physical layout, y-pencil device all-to-all\n");
            std::cout << "# grid " << params.nx << "x" << params.ny << "x" << params.nz
                      << ", dt=" << params.dt << ", scheme=" << params.time_scheme
                      << ", nu=" << params.nu << '\n';
            std::cout << "# wall=" << params.momentum_wall_model
                      << ", sgs=" << params.sgs_model
                      << ", thermo=" << (params.thermo_enabled ? "on" : "off")
                      << ", moisture=" << (params.moisture_enabled ? "on" : "off")
                      << ", dealias=" << (params.horizontal_dealias ? "on" : "off")
                      << ", cuda=on, mpi=on\n";
            std::cout << "#  step        ke_max         div_max       cfl";
            if (params.moisture_enabled) {
                std::cout << "        qv_min         qv_max         ql_max    column_water";
            }
            std::cout << '\n';
            std::cout << " " << 0 << " " << diag.ke_max << " " << diag.div_max << " " << diag.cfl;
            if (params.moisture_enabled) {
                std::cout << " " << diag.qv_min << " " << diag.qv_max
                          << " " << diag.ql_max << " " << diag.column_water;
            }
            std::cout << '\n';
        }

        const auto run_start = std::chrono::steady_clock::now();
        for (int step = 1; step <= params.steps; ++step) {
            cuda_mpi_step(state, rhs, pressure, params, slab, comm, halo);
            if (should_write_frame_cuda_mpi(params, step)) {
                write_frame_dump_cuda_mpi(state, params, slab, comm, step);
            }
            const bool needs_bomex = params.bomex_diagnostics_enabled
                && params.initial_condition == "bomex"
                && (step % params.bomex_sample_every == 0 || step == params.steps)
                && static_cast<double>(step) * params.dt >= params.bomex_average_start_seconds;
            if (needs_bomex) {
                download_single_gpu_state(state, host_output_state, params, slab);
                add_bomex_sample(bomex, host_output_state, params);
                add_single_gpu_sgs_bomex_sample(bomex, host_output_state, rhs, params);
                const double last_hour_start = std::max(
                    params.bomex_average_start_seconds,
                    static_cast<double>(params.steps) * params.dt - 3600.0);
                if (static_cast<double>(step) * params.dt >= last_hour_start) {
                    add_bomex_sample(bomex_last_hour, host_output_state, params);
                    add_single_gpu_sgs_bomex_sample(
                        bomex_last_hour, host_output_state, rhs, params);
                }
            }
            if (step % params.log_every == 0 || step == params.steps) {
                diag = diagnostics_cuda_mpi(state, pressure, params, slab, comm);
                if (rank == 0) {
                    std::cout << " " << step << " " << diag.ke_max << " " << diag.div_max << " " << diag.cfl;
                    if (params.moisture_enabled) {
                        std::cout << " " << diag.qv_min << " " << diag.qv_max
                                  << " " << diag.ql_max << " " << diag.column_water;
                    }
                    std::cout << '\n';
                    if (!std::isfinite(diag.ke_max) || !std::isfinite(diag.div_max)
                        || !std::isfinite(diag.cfl)
                        || (params.moisture_enabled
                            && (!std::isfinite(diag.qv_min) || !std::isfinite(diag.qv_max)
                                || !std::isfinite(diag.ql_max) || !std::isfinite(diag.column_water)))) {
                        throw std::runtime_error("non-finite CUDA diagnostics encountered; stopping run");
                    }
                }
            }
        }
        check_cuda(cudaDeviceSynchronize(), "final timestep synchronization");
        const double elapsed_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - run_start).count();
        if (rank == 0 && params.steps > 0) {
            std::cout << "[cuda-performance] elapsed_s=" << elapsed_seconds
                      << " steps_per_s=" << static_cast<double>(params.steps) / elapsed_seconds
                      << " simulated_s_per_wall_s="
                      << static_cast<double>(params.steps) * params.dt / elapsed_seconds << '\n';
        }
        if (rank == 0 && params.bomex_diagnostics_enabled
            && params.initial_condition == "bomex") {
            if (host_output_state.step_count != state.step_count) {
                download_single_gpu_state(state, host_output_state, params, slab);
            }
            print_bomex_summary(bomex, params);
            write_bomex_outputs(bomex, host_output_state, params);
            if (bomex_last_hour.samples > 0) {
                Params last_hour_params = params;
                last_hour_params.bomex_output_dir = join_path_cuda_mpi(
                    params.bomex_output_dir, "fig3_last_hour");
                write_bomex_outputs(bomex_last_hour, host_output_state, last_hour_params);
            }
        }
    } catch (const std::exception& exc) {
        std::cerr << "[rank " << rank << "] ERROR: " << exc.what() << '\n';
        MPI_Abort(comm, 1);
        return 1;
    }
    return 0;
}

}  // namespace wireles

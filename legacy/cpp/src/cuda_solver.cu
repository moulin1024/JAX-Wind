#include "wireles/cuda_solver.hpp"

#include <cuda_runtime.h>
#include <cufft.h>

#include <stdexcept>
#include <string>

#include "wireles/timestep.hpp"

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

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t count) { allocate(count); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    void allocate(std::size_t count) {
        count_ = count;
        if (count_ > 0) {
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)), "allocation");
        }
    }

    T* data() { return data_; }
    const T* data() const { return data_; }
    std::size_t size() const { return count_; }

    void copy_from(const Field& field, const char* name) {
        if (field.size() != count_) {
            throw std::runtime_error(std::string("CUDA input size mismatch for ") + name);
        }
        if (count_ > 0) {
            check_cuda(cudaMemcpy(data_, field.data(), count_ * sizeof(T), cudaMemcpyHostToDevice), "host-to-device copy");
        }
    }

    void copy_to(Field& field, const char* name) const {
        if (field.size() != count_) {
            throw std::runtime_error(std::string("CUDA output size mismatch for ") + name);
        }
        if (count_ > 0) {
            check_cuda(cudaMemcpy(field.data(), data_, count_ * sizeof(T), cudaMemcpyDeviceToHost), "device-to-host copy");
        }
    }

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

__device__ double kx_value_device(int, double lx, int ih) {
    return 2.0 * 3.141592653589793238462643383279502884 * static_cast<double>(ih) / lx;
}

__device__ double ky_derivative_device(int ny, double ly, int j) {
    if ((ny % 2) == 0 && j == ny / 2) {
        return 0.0;
    }
    const int signed_j = (j <= ny / 2) ? j : j - ny;
    return 2.0 * 3.141592653589793238462643383279502884 * static_cast<double>(signed_j) / ly;
}

__device__ double ky_value_device(int ny, double ly, int j) {
    const int signed_j = (j <= ny / 2) ? j : j - ny;
    return 2.0 * 3.141592653589793238462643383279502884 * static_cast<double>(signed_j) / ly;
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

__global__ void w_to_center_kernel(const double* w, double* w_center, int nx, int ny, int nz) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    const std::size_t center_count = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
    if (n >= center_count) {
        return;
    }
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    w_center[n] = 0.5 * (w[n] + w[n + plane]);
}

__global__ void center_to_w_kernel(const double* q, double* out, int nx, int ny, int nz) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    const std::size_t face_count = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz + 1);
    if (n >= face_count) {
        return;
    }
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const int k = static_cast<int>(n / plane);
    if (k == 0) {
        out[n] = q[n];
    } else if (k == nz) {
        out[n] = q[n - plane];
    } else {
        out[n] = 0.5 * (q[n - plane] + q[n]);
    }
}

__global__ void ddz_center_kernel(const double* q, double* out, int nx, int ny, int nz, double inv_dz) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    const std::size_t center_count = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
    if (n >= center_count) {
        return;
    }
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const int k = static_cast<int>(n / plane);
    if (k == 0) {
        out[n] = (q[n + plane] - q[n]) * inv_dz;
    } else if (k == nz - 1) {
        out[n] = (q[n] - q[n - plane]) * inv_dz;
    } else {
        out[n] = (q[n + plane] - q[n - plane]) * (0.5 * inv_dz);
    }
}

__global__ void add_vertical_laplacian_center_kernel(double* lap, const double* q, int nx, int ny, int nz, double inv_dz2) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    const std::size_t center_count = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
    if (n >= center_count) {
        return;
    }
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const int k = static_cast<int>(n / plane);
    double value = 0.0;
    if (k == 0) {
        value = (q[n + plane] - q[n]) * inv_dz2;
    } else if (k == nz - 1) {
        value = (q[n - plane] - q[n]) * inv_dz2;
    } else {
        value = (q[n - plane] - 2.0 * q[n] + q[n + plane]) * inv_dz2;
    }
    lap[n] += value;
}

__global__ void ddz_center_to_w_kernel(const double* q, double* out, int nx, int ny, int nz, double inv_dz) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    const std::size_t face_count = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz + 1);
    if (n >= face_count) {
        return;
    }
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const int k = static_cast<int>(n / plane);
    if (k <= 0 || k >= nz) {
        out[n] = 0.0;
    } else {
        out[n] = (q[n] - q[n - plane]) * inv_dz;
    }
}

__global__ void ddz_w_face_kernel(const double* w, double* out, int nx, int ny, int nz, double inv_dz) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    const std::size_t face_count = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz + 1);
    if (n >= face_count) {
        return;
    }
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const int k = static_cast<int>(n / plane);
    if (k <= 0 || k >= nz) {
        out[n] = 0.0;
    } else {
        out[n] = (w[n + plane] - w[n - plane]) * (0.5 * inv_dz);
    }
}

__global__ void add_vertical_laplacian_w_kernel(double* lap, const double* w, int nx, int ny, int nz, double inv_dz2) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    const std::size_t face_count = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz + 1);
    if (n >= face_count) {
        return;
    }
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const int k = static_cast<int>(n / plane);
    if (k <= 0 || k >= nz) {
        lap[n] = 0.0;
    } else {
        lap[n] += (w[n - plane] - 2.0 * w[n] + w[n + plane]) * inv_dz2;
    }
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
    double coeff) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    const double s11 = dudx[n];
    const double s22 = dvdy[n];
    const double s33 = dwdz[n];
    const double s12 = 0.5 * (dudy[n] + dvdx[n]);
    const double s13 = 0.5 * (dudz[n] + dwdx[n]);
    const double s23 = 0.5 * (dvdz[n] + dwdy[n]);
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
    int nz) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    const int plane = nx * ny;
    const int k = static_cast<int>(n / static_cast<std::size_t>(plane));
    if (k <= 0 || k >= nz) {
        txz[n] = 0.0;
        tyz[n] = 0.0;
        return;
    }
    txz[n] = nu_t_face[n] * (dudz_face[n] + dwdx_face[n]);
    tyz[n] = nu_t_face[n] * (dvdz_face[n] + dwdy_face[n]);
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
    int nz,
    double fft_scale) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    const int plane = nx * ny;
    const int k = static_cast<int>(n / static_cast<std::size_t>(plane));
    if (k <= 0 || k >= nz) {
        return;
    }
    rhs_w[n] += fft_scale * (dtxz_dx[n] + dtyz_dy[n]) + dtzz_dz[n];
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
    double geostrophic_v) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    double ru = -(u[n] * dudx[n] + v[n] * dudy[n] + w_center[n] * dudz[n]) + nu * lap_u[n];
    double rv = -(u[n] * dvdx[n] + v[n] * dvdy[n] + w_center[n] * dvdz[n]) + nu * lap_v[n];
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
    int nz,
    double nu) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    const int plane = nx * ny;
    const int k = static_cast<int>(n / static_cast<std::size_t>(plane));
    if (k <= 0 || k >= nz) {
        rhs_w[n] = 0.0;
        return;
    }
    rhs_w[n] = -(u_on_w[n] * dwdx_face[n] + v_on_w[n] * dwdy_face[n] + w[n] * dwdz_face[n])
        + nu * lap_w[n];
}

__global__ void advance_kernel(
    double* q,
    const double* rhs,
    double* rhs_prev,
    std::size_t count,
    int use_ab2,
    double dt) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    const double tendency = use_ab2 != 0 ? (1.5 * rhs[n] - 0.5 * rhs_prev[n]) : rhs[n];
    q[n] += dt * tendency;
    rhs_prev[n] = rhs[n];
}

__global__ void enforce_walls_kernel(double* w, int nx, int ny, int nz) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    if (n >= plane) {
        return;
    }
    w[n] = 0.0;
    w[static_cast<std::size_t>(nz) * plane + n] = 0.0;
}

__global__ void dwdz_center_kernel(const double* w, double* dwdz, int nx, int ny, int nz, double inv_dz) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    const std::size_t center_count = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
    if (n >= center_count) {
        return;
    }
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    dwdz[n] = (w[n + plane] - w[n]) * inv_dz;
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
    div_hat[n] = make_cuDoubleComplex(
        -kx * u_hat[n].y - ky * v_hat[n].y + dwdz_hat[n].x,
        kx * u_hat[n].x + ky * v_hat[n].x + dwdz_hat[n].y);
}

__global__ void pressure_solve_kernel(
    const cufftDoubleComplex* div_hat,
    cufftDoubleComplex* p_hat,
    cufftDoubleComplex* cp,
    cufftDoubleComplex* dp,
    int nkx,
    int ny,
    int nz,
    int nx,
    double lx,
    double ly,
    double inv_dz2,
    double inv_dt) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int columns = nkx * ny;
    if (col >= columns) {
        return;
    }
    const int j = col / nkx;
    const int ih = col - j * nkx;
    const double kx = kx_derivative_device(nx, lx, ih);
    const double ky = ky_derivative_device(ny, ly, j);
    const double kh2 = kx * kx + ky * ky;

    cufftDoubleComplex denom;
    cufftDoubleComplex a;
    cufftDoubleComplex b;
    cufftDoubleComplex c;
    cufftDoubleComplex d;

    if (kh2 == 0.0) {
        b = make_cuDoubleComplex(1.0, 0.0);
        c = make_cuDoubleComplex(0.0, 0.0);
        d = make_cuDoubleComplex(0.0, 0.0);
    } else {
        b = make_cuDoubleComplex(-inv_dz2 - kh2, 0.0);
        c = make_cuDoubleComplex(inv_dz2, 0.0);
        d = cscale(div_hat[col], inv_dt);
    }
    denom = b;
    cp[col] = cdiv(c, denom);
    dp[col] = cdiv(d, denom);

    for (int k = 1; k < nz; ++k) {
        const std::size_t row = static_cast<std::size_t>(k) * static_cast<std::size_t>(columns) + static_cast<std::size_t>(col);
        a = make_cuDoubleComplex(inv_dz2, 0.0);
        c = (k == nz - 1) ? make_cuDoubleComplex(0.0, 0.0) : make_cuDoubleComplex(inv_dz2, 0.0);
        if (kh2 == 0.0) {
            b = (k == nz - 1) ? make_cuDoubleComplex(-inv_dz2, 0.0) : make_cuDoubleComplex(-2.0 * inv_dz2, 0.0);
        } else {
            b = (k == nz - 1) ? make_cuDoubleComplex(-inv_dz2 - kh2, 0.0) : make_cuDoubleComplex(-2.0 * inv_dz2 - kh2, 0.0);
        }
        d = cscale(div_hat[row], inv_dt);
        denom = csub(b, cmul(a, cp[row - static_cast<std::size_t>(columns)]));
        cp[row] = (k == nz - 1) ? make_cuDoubleComplex(0.0, 0.0) : cdiv(c, denom);
        dp[row] = cdiv(csub(d, cmul(a, dp[row - static_cast<std::size_t>(columns)])), denom);
    }

    cufftDoubleComplex x = dp[static_cast<std::size_t>(nz - 1) * static_cast<std::size_t>(columns) + static_cast<std::size_t>(col)];
    p_hat[static_cast<std::size_t>(nz - 1) * static_cast<std::size_t>(columns) + static_cast<std::size_t>(col)] = x;
    for (int k = nz - 2; k >= 0; --k) {
        const std::size_t row = static_cast<std::size_t>(k) * static_cast<std::size_t>(columns) + static_cast<std::size_t>(col);
        x = csub(dp[row], cmul(cp[row], x));
        p_hat[row] = x;
    }
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

__global__ void scale_real_kernel(double* q, std::size_t count, double scale) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n < count) {
        q[n] *= scale;
    }
}

__global__ void subtract_horizontal_pressure_gradient_kernel(
    double* u,
    double* v,
    const double* dpdx,
    const double* dpdy,
    std::size_t count,
    double dt,
    double fft_scale) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    if (n >= count) {
        return;
    }
    u[n] -= dt * dpdx[n] * fft_scale;
    v[n] -= dt * dpdy[n] * fft_scale;
}

__global__ void subtract_vertical_pressure_gradient_kernel(
    double* w,
    const double* p,
    int nx,
    int ny,
    int nz,
    double dt,
    double inv_dz) {
    const std::size_t n = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x)
        + static_cast<std::size_t>(threadIdx.x);
    const std::size_t face_count = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz + 1);
    if (n >= face_count) {
        return;
    }
    const std::size_t plane = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny);
    const int k = static_cast<int>(n / plane);
    if (k <= 0 || k >= nz) {
        w[n] = 0.0;
        return;
    }
    w[n] -= dt * (p[n] - p[n - plane]) * inv_dz;
}

int blocks_for(std::size_t count, int threads) {
    return static_cast<int>((count + static_cast<std::size_t>(threads) - 1) / static_cast<std::size_t>(threads));
}

}  // namespace

struct CudaFlowState::Impl {
    explicit Impl(const Params& params)
        : center_count(params.real_size()),
          face_count(params.z_face_size()),
          spectral_count(params.spectral_size()),
          spectral_face_count(static_cast<std::size_t>(params.nz + 1) * static_cast<std::size_t>(params.ny) * static_cast<std::size_t>(params.nkx())),
          plane_count(static_cast<std::size_t>(params.nx) * static_cast<std::size_t>(params.ny)),
          nx(params.nx),
          ny(params.ny),
          nz(params.nz),
          nkx(params.nkx()),
          lx(params.lx),
          ly(params.ly),
          dz(params.dz()),
          dt(params.dt),
          u(center_count),
          v(center_count),
          w(face_count),
          p(center_count),
          theta(center_count),
          rhs_u_prev(center_count),
          rhs_v_prev(center_count),
          rhs_w_prev(face_count),
          rhs_theta_prev(center_count),
          rhs_u(center_count),
          rhs_v(center_count),
          rhs_w(face_count),
          rhs_theta(center_count),
          w_center(center_count),
          dudx(center_count),
          dudy(center_count),
          dudz(center_count),
          dvdx(center_count),
          dvdy(center_count),
          dvdz(center_count),
          lap_u(center_count),
          lap_v(center_count),
          u_on_w(face_count),
          v_on_w(face_count),
          dwdx_face(face_count),
          dwdy_face(face_count),
          dwdz_face(face_count),
          lap_w(face_count),
          dwdz_center(center_count),
          dwdx_center(center_count),
          dwdy_center(center_count),
          strain(center_count),
          nu_t(center_count),
          nu_t_face(face_count),
          dudz_face(face_count),
          dvdz_face(face_count),
          txx(center_count),
          txy(center_count),
          tyy(center_count),
          tzz(center_count),
          txz(face_count),
          tyz(face_count),
          dtxx_dx(center_count),
          dtxy_dy(center_count),
          dtxy_dx(center_count),
          dtyy_dy(center_count),
          dtxz_dz(center_count),
          dtyz_dz(center_count),
          dtxz_dx(face_count),
          dtyz_dy(face_count),
          dtzz_dz(face_count),
          dpdx(center_count),
          dpdy(center_count),
          u_hat(spectral_count),
          v_hat(spectral_count),
          dwdz_hat(spectral_count),
          div_hat(spectral_count),
          p_hat(spectral_count),
          pressure_cp(spectral_count),
          pressure_dp(spectral_count),
          dpdx_hat(spectral_count),
          dpdy_hat(spectral_count),
          w_face_hat(spectral_face_count),
          face_hat_scratch(spectral_face_count) {
        int n[2] = {ny, nx};
        check_cufft(cufftPlanMany(&r2c.handle, 2, n, nullptr, 1, nx * ny, nullptr, 1, ny * nkx, CUFFT_D2Z, nz), "D2Z plan creation");
        check_cufft(cufftPlanMany(&c2r.handle, 2, n, nullptr, 1, ny * nkx, nullptr, 1, nx * ny, CUFFT_Z2D, nz), "Z2D plan creation");
        check_cufft(cufftPlanMany(&r2c_face.handle, 2, n, nullptr, 1, nx * ny, nullptr, 1, ny * nkx, CUFFT_D2Z, nz + 1), "face D2Z plan creation");
        check_cufft(cufftPlanMany(&c2r_face.handle, 2, n, nullptr, 1, ny * nkx, nullptr, 1, nx * ny, CUFFT_Z2D, nz + 1), "face Z2D plan creation");
    }

    std::size_t center_count;
    std::size_t face_count;
    std::size_t spectral_count;
    std::size_t spectral_face_count;
    std::size_t plane_count;
    int nx;
    int ny;
    int nz;
    int nkx;
    double lx;
    double ly;
    double dz;
    double dt;
    bool flow_current = false;

    DeviceBuffer<double> u;
    DeviceBuffer<double> v;
    DeviceBuffer<double> w;
    DeviceBuffer<double> p;
    DeviceBuffer<double> theta;
    DeviceBuffer<double> rhs_u_prev;
    DeviceBuffer<double> rhs_v_prev;
    DeviceBuffer<double> rhs_w_prev;
    DeviceBuffer<double> rhs_theta_prev;
    DeviceBuffer<double> rhs_u;
    DeviceBuffer<double> rhs_v;
    DeviceBuffer<double> rhs_w;
    DeviceBuffer<double> rhs_theta;
    DeviceBuffer<double> w_center;
    DeviceBuffer<double> dudx;
    DeviceBuffer<double> dudy;
    DeviceBuffer<double> dudz;
    DeviceBuffer<double> dvdx;
    DeviceBuffer<double> dvdy;
    DeviceBuffer<double> dvdz;
    DeviceBuffer<double> lap_u;
    DeviceBuffer<double> lap_v;
    DeviceBuffer<double> u_on_w;
    DeviceBuffer<double> v_on_w;
    DeviceBuffer<double> dwdx_face;
    DeviceBuffer<double> dwdy_face;
    DeviceBuffer<double> dwdz_face;
    DeviceBuffer<double> lap_w;
    DeviceBuffer<double> dwdz_center;
    DeviceBuffer<double> dwdx_center;
    DeviceBuffer<double> dwdy_center;
    DeviceBuffer<double> strain;
    DeviceBuffer<double> nu_t;
    DeviceBuffer<double> nu_t_face;
    DeviceBuffer<double> dudz_face;
    DeviceBuffer<double> dvdz_face;
    DeviceBuffer<double> txx;
    DeviceBuffer<double> txy;
    DeviceBuffer<double> tyy;
    DeviceBuffer<double> tzz;
    DeviceBuffer<double> txz;
    DeviceBuffer<double> tyz;
    DeviceBuffer<double> dtxx_dx;
    DeviceBuffer<double> dtxy_dy;
    DeviceBuffer<double> dtxy_dx;
    DeviceBuffer<double> dtyy_dy;
    DeviceBuffer<double> dtxz_dz;
    DeviceBuffer<double> dtyz_dz;
    DeviceBuffer<double> dtxz_dx;
    DeviceBuffer<double> dtyz_dy;
    DeviceBuffer<double> dtzz_dz;
    DeviceBuffer<double> dpdx;
    DeviceBuffer<double> dpdy;
    DeviceBuffer<cufftDoubleComplex> u_hat;
    DeviceBuffer<cufftDoubleComplex> v_hat;
    DeviceBuffer<cufftDoubleComplex> dwdz_hat;
    DeviceBuffer<cufftDoubleComplex> div_hat;
    DeviceBuffer<cufftDoubleComplex> p_hat;
    DeviceBuffer<cufftDoubleComplex> pressure_cp;
    DeviceBuffer<cufftDoubleComplex> pressure_dp;
    DeviceBuffer<cufftDoubleComplex> dpdx_hat;
    DeviceBuffer<cufftDoubleComplex> dpdy_hat;
    DeviceBuffer<cufftDoubleComplex> w_face_hat;
    DeviceBuffer<cufftDoubleComplex> face_hat_scratch;
    CufftPlan r2c;
    CufftPlan c2r;
    CufftPlan r2c_face;
    CufftPlan c2r_face;
};

template <typename Impl>
void scale_real_buffer(Impl& d, DeviceBuffer<double>& q, std::size_t count) {
    constexpr int threads = 256;
    const double fft_scale = 1.0 / static_cast<double>(d.nx * d.ny);
    scale_real_kernel<<<blocks_for(count, threads), threads>>>(q.data(), count, fft_scale);
    check_cuda(cudaGetLastError(), "inverse FFT scaling kernel launch");
}

template <typename Impl>
void inverse_center_scratch(Impl& d, DeviceBuffer<cufftDoubleComplex>& spec, DeviceBuffer<double>& out) {
    check_cufft(cufftExecZ2D(d.c2r.handle, spec.data(), out.data()), "center inverse transform");
    scale_real_buffer(d, out, d.center_count);
}

template <typename Impl>
void inverse_face_scratch(Impl& d, DeviceBuffer<cufftDoubleComplex>& spec, DeviceBuffer<double>& out) {
    check_cufft(cufftExecZ2D(d.c2r_face.handle, spec.data(), out.data()), "face inverse transform");
    scale_real_buffer(d, out, d.face_count);
}

template <typename Impl>
void compute_center_horizontal_derivatives_and_laplacian(
    Impl& d,
    DeviceBuffer<double>& q,
    DeviceBuffer<cufftDoubleComplex>& q_hat,
    DeviceBuffer<double>& dx,
    DeviceBuffer<double>& dy,
    DeviceBuffer<double>& lap) {
    constexpr int threads = 256;
    check_cufft(cufftExecD2Z(d.r2c.handle, q.data(), q_hat.data()), "center forward transform");

    spectral_derivative_x_kernel<<<blocks_for(d.spectral_count, threads), threads>>>(
        q_hat.data(), d.div_hat.data(), d.spectral_count, d.nkx, d.ny, d.nx, d.lx);
    check_cuda(cudaGetLastError(), "center spectral dx kernel launch");
    inverse_center_scratch(d, d.div_hat, dx);

    spectral_derivative_y_kernel<<<blocks_for(d.spectral_count, threads), threads>>>(
        q_hat.data(), d.div_hat.data(), d.spectral_count, d.nkx, d.ny, d.ly);
    check_cuda(cudaGetLastError(), "center spectral dy kernel launch");
    inverse_center_scratch(d, d.div_hat, dy);

    spectral_laplacian_kernel<<<blocks_for(d.spectral_count, threads), threads>>>(
        q_hat.data(), d.div_hat.data(), d.spectral_count, d.nkx, d.ny, d.nx, d.lx, d.ly);
    check_cuda(cudaGetLastError(), "center spectral laplacian kernel launch");
    inverse_center_scratch(d, d.div_hat, lap);

    add_vertical_laplacian_center_kernel<<<blocks_for(d.center_count, threads), threads>>>(
        lap.data(), q.data(), d.nx, d.ny, d.nz, 1.0 / (d.dz * d.dz));
    check_cuda(cudaGetLastError(), "center vertical laplacian kernel launch");
}

template <typename Impl>
void compute_center_horizontal_derivatives(
    Impl& d,
    DeviceBuffer<double>& q,
    DeviceBuffer<cufftDoubleComplex>& q_hat,
    DeviceBuffer<double>& dx,
    DeviceBuffer<double>& dy) {
    constexpr int threads = 256;
    check_cufft(cufftExecD2Z(d.r2c.handle, q.data(), q_hat.data()), "center forward transform");

    spectral_derivative_x_kernel<<<blocks_for(d.spectral_count, threads), threads>>>(
        q_hat.data(), d.div_hat.data(), d.spectral_count, d.nkx, d.ny, d.nx, d.lx);
    check_cuda(cudaGetLastError(), "center spectral dx kernel launch");
    inverse_center_scratch(d, d.div_hat, dx);

    spectral_derivative_y_kernel<<<blocks_for(d.spectral_count, threads), threads>>>(
        q_hat.data(), d.div_hat.data(), d.spectral_count, d.nkx, d.ny, d.ly);
    check_cuda(cudaGetLastError(), "center spectral dy kernel launch");
    inverse_center_scratch(d, d.div_hat, dy);
}

template <typename Impl>
void compute_center_horizontal_derivative_x(Impl& d, DeviceBuffer<double>& q, DeviceBuffer<double>& dx) {
    constexpr int threads = 256;
    check_cufft(cufftExecD2Z(d.r2c.handle, q.data(), d.div_hat.data()), "center forward transform");
    spectral_derivative_x_kernel<<<blocks_for(d.spectral_count, threads), threads>>>(
        d.div_hat.data(), d.p_hat.data(), d.spectral_count, d.nkx, d.ny, d.nx, d.lx);
    check_cuda(cudaGetLastError(), "center spectral dx kernel launch");
    inverse_center_scratch(d, d.p_hat, dx);
}

template <typename Impl>
void compute_center_horizontal_derivative_y(Impl& d, DeviceBuffer<double>& q, DeviceBuffer<double>& dy) {
    constexpr int threads = 256;
    check_cufft(cufftExecD2Z(d.r2c.handle, q.data(), d.div_hat.data()), "center forward transform");
    spectral_derivative_y_kernel<<<blocks_for(d.spectral_count, threads), threads>>>(
        d.div_hat.data(), d.p_hat.data(), d.spectral_count, d.nkx, d.ny, d.ly);
    check_cuda(cudaGetLastError(), "center spectral dy kernel launch");
    inverse_center_scratch(d, d.p_hat, dy);
}

template <typename Impl>
void compute_face_horizontal_derivatives_and_laplacian(Impl& d) {
    constexpr int threads = 256;
    check_cufft(cufftExecD2Z(d.r2c_face.handle, d.w.data(), d.w_face_hat.data()), "face forward transform");

    spectral_derivative_x_kernel<<<blocks_for(d.spectral_face_count, threads), threads>>>(
        d.w_face_hat.data(), d.face_hat_scratch.data(), d.spectral_face_count, d.nkx, d.ny, d.nx, d.lx);
    check_cuda(cudaGetLastError(), "face spectral dx kernel launch");
    inverse_face_scratch(d, d.face_hat_scratch, d.dwdx_face);

    spectral_derivative_y_kernel<<<blocks_for(d.spectral_face_count, threads), threads>>>(
        d.w_face_hat.data(), d.face_hat_scratch.data(), d.spectral_face_count, d.nkx, d.ny, d.ly);
    check_cuda(cudaGetLastError(), "face spectral dy kernel launch");
    inverse_face_scratch(d, d.face_hat_scratch, d.dwdy_face);

    spectral_laplacian_kernel<<<blocks_for(d.spectral_face_count, threads), threads>>>(
        d.w_face_hat.data(), d.face_hat_scratch.data(), d.spectral_face_count, d.nkx, d.ny, d.nx, d.lx, d.ly);
    check_cuda(cudaGetLastError(), "face spectral laplacian kernel launch");
    inverse_face_scratch(d, d.face_hat_scratch, d.lap_w);

    add_vertical_laplacian_w_kernel<<<blocks_for(d.face_count, threads), threads>>>(
        d.lap_w.data(), d.w.data(), d.nx, d.ny, d.nz, 1.0 / (d.dz * d.dz));
    check_cuda(cudaGetLastError(), "face vertical laplacian kernel launch");
}

template <typename Impl>
void compute_face_horizontal_derivative_x(Impl& d, DeviceBuffer<double>& q, DeviceBuffer<double>& dx) {
    constexpr int threads = 256;
    check_cufft(cufftExecD2Z(d.r2c_face.handle, q.data(), d.w_face_hat.data()), "face forward transform");
    spectral_derivative_x_kernel<<<blocks_for(d.spectral_face_count, threads), threads>>>(
        d.w_face_hat.data(), d.face_hat_scratch.data(), d.spectral_face_count, d.nkx, d.ny, d.nx, d.lx);
    check_cuda(cudaGetLastError(), "face spectral dx kernel launch");
    inverse_face_scratch(d, d.face_hat_scratch, dx);
}

template <typename Impl>
void compute_face_horizontal_derivative_y(Impl& d, DeviceBuffer<double>& q, DeviceBuffer<double>& dy) {
    constexpr int threads = 256;
    check_cufft(cufftExecD2Z(d.r2c_face.handle, q.data(), d.w_face_hat.data()), "face forward transform");
    spectral_derivative_y_kernel<<<blocks_for(d.spectral_face_count, threads), threads>>>(
        d.w_face_hat.data(), d.face_hat_scratch.data(), d.spectral_face_count, d.nkx, d.ny, d.ly);
    check_cuda(cudaGetLastError(), "face spectral dy kernel launch");
    inverse_face_scratch(d, d.face_hat_scratch, dy);
}

template <typename Impl>
void apply_smagorinsky_sgs(Impl& d, const Params& params) {
    constexpr int threads = 256;
    const double inv_dz = 1.0 / params.dz();
    const double length = params.smagorinsky_cs * params.sgs_delta();
    const double coeff = length * length;

    compute_center_horizontal_derivatives(d, d.w_center, d.dwdz_hat, d.dwdx_center, d.dwdy_center);
    dwdz_center_kernel<<<blocks_for(d.center_count, threads), threads>>>(d.w.data(), d.dwdz_center.data(), params.nx, params.ny, params.nz, inv_dz);
    check_cuda(cudaGetLastError(), "center dwdz kernel launch");

    build_smagorinsky_center_kernel<<<blocks_for(d.center_count, threads), threads>>>(
        d.dudx.data(),
        d.dudy.data(),
        d.dudz.data(),
        d.dvdx.data(),
        d.dvdy.data(),
        d.dvdz.data(),
        d.dwdx_center.data(),
        d.dwdy_center.data(),
        d.dwdz_center.data(),
        d.strain.data(),
        d.nu_t.data(),
        d.txx.data(),
        d.txy.data(),
        d.tyy.data(),
        d.tzz.data(),
        d.center_count,
        coeff);
    check_cuda(cudaGetLastError(), "smagorinsky center stress kernel launch");

    center_to_w_kernel<<<blocks_for(d.face_count, threads), threads>>>(d.nu_t.data(), d.nu_t_face.data(), params.nx, params.ny, params.nz);
    check_cuda(cudaGetLastError(), "nu_t face interpolation kernel launch");
    ddz_center_to_w_kernel<<<blocks_for(d.face_count, threads), threads>>>(d.u.data(), d.dudz_face.data(), params.nx, params.ny, params.nz, inv_dz);
    check_cuda(cudaGetLastError(), "dudz face kernel launch");
    ddz_center_to_w_kernel<<<blocks_for(d.face_count, threads), threads>>>(d.v.data(), d.dvdz_face.data(), params.nx, params.ny, params.nz, inv_dz);
    check_cuda(cudaGetLastError(), "dvdz face kernel launch");

    build_smagorinsky_face_kernel<<<blocks_for(d.face_count, threads), threads>>>(
        d.nu_t_face.data(),
        d.dudz_face.data(),
        d.dvdz_face.data(),
        d.dwdx_face.data(),
        d.dwdy_face.data(),
        d.txz.data(),
        d.tyz.data(),
        d.face_count,
        params.nx,
        params.ny,
        params.nz);
    check_cuda(cudaGetLastError(), "smagorinsky face stress kernel launch");

    compute_center_horizontal_derivative_x(d, d.txx, d.dtxx_dx);
    compute_center_horizontal_derivative_y(d, d.txy, d.dtxy_dy);
    compute_center_horizontal_derivative_x(d, d.txy, d.dtxy_dx);
    compute_center_horizontal_derivative_y(d, d.tyy, d.dtyy_dy);
    dwdz_center_kernel<<<blocks_for(d.center_count, threads), threads>>>(d.txz.data(), d.dtxz_dz.data(), params.nx, params.ny, params.nz, inv_dz);
    check_cuda(cudaGetLastError(), "dtxz dz kernel launch");
    dwdz_center_kernel<<<blocks_for(d.center_count, threads), threads>>>(d.tyz.data(), d.dtyz_dz.data(), params.nx, params.ny, params.nz, inv_dz);
    check_cuda(cudaGetLastError(), "dtyz dz kernel launch");

    add_center_sgs_divergence_kernel<<<blocks_for(d.center_count, threads), threads>>>(
        d.rhs_u.data(),
        d.rhs_v.data(),
        d.dtxx_dx.data(),
        d.dtxy_dy.data(),
        d.dtxy_dx.data(),
        d.dtyy_dy.data(),
        d.dtxz_dz.data(),
        d.dtyz_dz.data(),
        d.center_count,
        1.0);
    check_cuda(cudaGetLastError(), "center SGS divergence kernel launch");

    compute_face_horizontal_derivative_x(d, d.txz, d.dtxz_dx);
    compute_face_horizontal_derivative_y(d, d.tyz, d.dtyz_dy);
    ddz_center_to_w_kernel<<<blocks_for(d.face_count, threads), threads>>>(d.tzz.data(), d.dtzz_dz.data(), params.nx, params.ny, params.nz, inv_dz);
    check_cuda(cudaGetLastError(), "dtzz dz kernel launch");
    add_face_sgs_divergence_kernel<<<blocks_for(d.face_count, threads), threads>>>(
        d.rhs_w.data(),
        d.dtxz_dx.data(),
        d.dtyz_dy.data(),
        d.dtzz_dz.data(),
        d.face_count,
        params.nx,
        params.ny,
        params.nz,
        1.0);
    check_cuda(cudaGetLastError(), "face SGS divergence kernel launch");
}

template <typename Impl>
void compute_device_momentum_rhs(Impl& d, const Params& params) {
    constexpr int threads = 256;
    const double inv_dz = 1.0 / params.dz();

    w_to_center_kernel<<<blocks_for(d.center_count, threads), threads>>>(d.w.data(), d.w_center.data(), params.nx, params.ny, params.nz);
    check_cuda(cudaGetLastError(), "w center interpolation kernel launch");
    center_to_w_kernel<<<blocks_for(d.face_count, threads), threads>>>(d.u.data(), d.u_on_w.data(), params.nx, params.ny, params.nz);
    check_cuda(cudaGetLastError(), "u face interpolation kernel launch");
    center_to_w_kernel<<<blocks_for(d.face_count, threads), threads>>>(d.v.data(), d.v_on_w.data(), params.nx, params.ny, params.nz);
    check_cuda(cudaGetLastError(), "v face interpolation kernel launch");

    ddz_center_kernel<<<blocks_for(d.center_count, threads), threads>>>(d.u.data(), d.dudz.data(), params.nx, params.ny, params.nz, inv_dz);
    check_cuda(cudaGetLastError(), "dudz kernel launch");
    ddz_center_kernel<<<blocks_for(d.center_count, threads), threads>>>(d.v.data(), d.dvdz.data(), params.nx, params.ny, params.nz, inv_dz);
    check_cuda(cudaGetLastError(), "dvdz kernel launch");
    ddz_w_face_kernel<<<blocks_for(d.face_count, threads), threads>>>(d.w.data(), d.dwdz_face.data(), params.nx, params.ny, params.nz, inv_dz);
    check_cuda(cudaGetLastError(), "face dwdz kernel launch");

    compute_center_horizontal_derivatives_and_laplacian(d, d.u, d.u_hat, d.dudx, d.dudy, d.lap_u);
    compute_center_horizontal_derivatives_and_laplacian(d, d.v, d.v_hat, d.dvdx, d.dvdy, d.lap_v);
    compute_face_horizontal_derivatives_and_laplacian(d);

    build_center_rhs_kernel<<<blocks_for(d.center_count, threads), threads>>>(
        d.u.data(),
        d.v.data(),
        d.w_center.data(),
        d.dudx.data(),
        d.dudy.data(),
        d.dudz.data(),
        d.dvdx.data(),
        d.dvdy.data(),
        d.dvdz.data(),
        d.lap_u.data(),
        d.lap_v.data(),
        d.rhs_u.data(),
        d.rhs_v.data(),
        d.center_count,
        params.nu,
        params.coriolis_f,
        params.geostrophic_u,
        params.geostrophic_v);
    check_cuda(cudaGetLastError(), "center momentum RHS kernel launch");

    build_w_rhs_kernel<<<blocks_for(d.face_count, threads), threads>>>(
        d.w.data(),
        d.u_on_w.data(),
        d.v_on_w.data(),
        d.dwdx_face.data(),
        d.dwdy_face.data(),
        d.dwdz_face.data(),
        d.lap_w.data(),
        d.rhs_w.data(),
        d.face_count,
        params.nx,
        params.ny,
        params.nz,
        params.nu);
    check_cuda(cudaGetLastError(), "w momentum RHS kernel launch");

    if (params.sgs_model == "smagorinsky") {
        apply_smagorinsky_sgs(d, params);
    }
}

CudaFlowState::CudaFlowState(const Params& params)
    : impl_(std::make_unique<Impl>(params)) {}

CudaFlowState::~CudaFlowState() = default;

bool cuda_available() {
    int device_count = 0;
    const cudaError_t result = cudaGetDeviceCount(&device_count);
    return result == cudaSuccess && device_count > 0;
}

void cuda_upload_flow_state(CudaFlowState& device, const FlowState& state, const Params&) {
    auto& d = *device.impl_;
    d.u.copy_from(state.u, "u");
    d.v.copy_from(state.v, "v");
    d.w.copy_from(state.w, "w");
    d.p.copy_from(state.p, "p");
    d.theta.copy_from(state.theta, "theta");
    d.rhs_u_prev.copy_from(state.rhs_u_prev, "rhs_u_prev");
    d.rhs_v_prev.copy_from(state.rhs_v_prev, "rhs_v_prev");
    d.rhs_w_prev.copy_from(state.rhs_w_prev, "rhs_w_prev");
    d.rhs_theta_prev.copy_from(state.rhs_theta_prev, "rhs_theta_prev");
    d.flow_current = true;
}

void cuda_download_flow_state(const CudaFlowState& device, FlowState& state, const Params&) {
    const auto& d = *device.impl_;
    d.u.copy_to(state.u, "u");
    d.v.copy_to(state.v, "v");
    d.w.copy_to(state.w, "w");
    d.p.copy_to(state.p, "p");
    d.theta.copy_to(state.theta, "theta");
    d.rhs_u_prev.copy_to(state.rhs_u_prev, "rhs_u_prev");
    d.rhs_v_prev.copy_to(state.rhs_v_prev, "rhs_v_prev");
    d.rhs_w_prev.copy_to(state.rhs_w_prev, "rhs_w_prev");
    d.rhs_theta_prev.copy_to(state.rhs_theta_prev, "rhs_theta_prev");
}

void cuda_enforce_walls(TimestepWorkspace& workspace, const Params& params) {
    auto& d = *workspace.cuda->impl_;
    constexpr int threads = 256;
    enforce_walls_kernel<<<blocks_for(d.plane_count, threads), threads>>>(d.w.data(), params.nx, params.ny, params.nz);
    check_cuda(cudaGetLastError(), "wall enforcement kernel launch");
    check_cuda(cudaDeviceSynchronize(), "wall enforcement kernel execution");
}

void cuda_project(FlowState&, TimestepWorkspace& workspace, const Params& params) {
    auto& d = *workspace.cuda->impl_;
    constexpr int threads = 256;
    const double inv_dz = 1.0 / params.dz();
    const double fft_scale = 1.0 / static_cast<double>(params.nx * params.ny);

    dwdz_center_kernel<<<blocks_for(d.center_count, threads), threads>>>(d.w.data(), d.dwdz_center.data(), params.nx, params.ny, params.nz, inv_dz);
    check_cuda(cudaGetLastError(), "dwdz kernel launch");

    check_cufft(cufftExecD2Z(d.r2c.handle, d.u.data(), d.u_hat.data()), "u forward transform");
    check_cufft(cufftExecD2Z(d.r2c.handle, d.v.data(), d.v_hat.data()), "v forward transform");
    check_cufft(cufftExecD2Z(d.r2c.handle, d.dwdz_center.data(), d.dwdz_hat.data()), "dwdz forward transform");

    spectral_divergence_kernel<<<blocks_for(d.spectral_count, threads), threads>>>(
        d.u_hat.data(),
        d.v_hat.data(),
        d.dwdz_hat.data(),
        d.div_hat.data(),
        d.spectral_count,
        params.nkx(),
        params.ny,
        params.nx,
        params.lx,
        params.ly);
    check_cuda(cudaGetLastError(), "spectral divergence kernel launch");

    const int columns = params.nkx() * params.ny;
    const int pressure_threads = 128;
    pressure_solve_kernel<<<blocks_for(static_cast<std::size_t>(columns), pressure_threads), pressure_threads>>>(
        d.div_hat.data(),
        d.p_hat.data(),
        d.pressure_cp.data(),
        d.pressure_dp.data(),
        params.nkx(),
        params.ny,
        params.nz,
        params.nx,
        params.lx,
        params.ly,
        1.0 / (params.dz() * params.dz()),
        1.0 / params.dt);
    check_cuda(cudaGetLastError(), "pressure solve kernel launch");

    check_cufft(cufftExecZ2D(d.c2r.handle, d.p_hat.data(), d.p.data()), "pressure inverse transform");
    scale_real_kernel<<<blocks_for(d.center_count, threads), threads>>>(d.p.data(), d.center_count, fft_scale);
    check_cuda(cudaGetLastError(), "pressure scaling kernel launch");

    spectral_pressure_gradient_kernel<<<blocks_for(d.spectral_count, threads), threads>>>(
        d.p_hat.data(),
        d.dpdx_hat.data(),
        d.dpdy_hat.data(),
        d.spectral_count,
        params.nkx(),
        params.ny,
        params.nx,
        params.lx,
        params.ly);
    check_cuda(cudaGetLastError(), "pressure gradient spectral kernel launch");
    check_cufft(cufftExecZ2D(d.c2r.handle, d.dpdx_hat.data(), d.dpdx.data()), "dpdx inverse transform");
    check_cufft(cufftExecZ2D(d.c2r.handle, d.dpdy_hat.data(), d.dpdy.data()), "dpdy inverse transform");

    subtract_horizontal_pressure_gradient_kernel<<<blocks_for(d.center_count, threads), threads>>>(
        d.u.data(),
        d.v.data(),
        d.dpdx.data(),
        d.dpdy.data(),
        d.center_count,
        params.dt,
        fft_scale);
    check_cuda(cudaGetLastError(), "horizontal projection kernel launch");
    subtract_vertical_pressure_gradient_kernel<<<blocks_for(d.face_count, threads), threads>>>(
        d.w.data(),
        d.p.data(),
        params.nx,
        params.ny,
        params.nz,
        params.dt,
        inv_dz);
    check_cuda(cudaGetLastError(), "vertical projection kernel launch");
    check_cuda(cudaDeviceSynchronize(), "projection kernel execution");
    d.flow_current = true;
}

void cuda_step_device_resident(FlowState& state, TimestepWorkspace& workspace, bool use_ab2, const Params& params) {
    if (!workspace.cuda) {
        workspace.cuda = std::make_unique<CudaFlowState>(params);
    }
    auto& d = *workspace.cuda->impl_;
    if (!d.flow_current) {
        cuda_upload_flow_state(*workspace.cuda, state, params);
    }

    compute_device_momentum_rhs(d, params);

    constexpr int threads = 256;
    advance_kernel<<<blocks_for(d.center_count, threads), threads>>>(
        d.u.data(), d.rhs_u.data(), d.rhs_u_prev.data(), d.center_count, use_ab2 ? 1 : 0, params.dt);
    check_cuda(cudaGetLastError(), "u advance kernel launch");
    advance_kernel<<<blocks_for(d.center_count, threads), threads>>>(
        d.v.data(), d.rhs_v.data(), d.rhs_v_prev.data(), d.center_count, use_ab2 ? 1 : 0, params.dt);
    check_cuda(cudaGetLastError(), "v advance kernel launch");
    advance_kernel<<<blocks_for(d.face_count, threads), threads>>>(
        d.w.data(), d.rhs_w.data(), d.rhs_w_prev.data(), d.face_count, use_ab2 ? 1 : 0, params.dt);
    check_cuda(cudaGetLastError(), "w advance kernel launch");

    enforce_walls_kernel<<<blocks_for(d.plane_count, threads), threads>>>(d.w.data(), params.nx, params.ny, params.nz);
    check_cuda(cudaGetLastError(), "wall enforcement kernel launch");
    check_cuda(cudaDeviceSynchronize(), "device-resident advance kernel execution");

    cuda_project(state, workspace, params);
    d.flow_current = true;
}

}  // namespace wireles

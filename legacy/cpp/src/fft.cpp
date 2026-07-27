#include "wireles/fft.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace wireles {

FftwXY::PaddedWorkspace::~PaddedWorkspace() {
    if (forward != nullptr) {
        fftw_destroy_plan(forward);
    }
    if (inverse != nullptr) {
        fftw_destroy_plan(inverse);
    }
    fftw_free(real_a);
    fftw_free(real_b);
    fftw_free(real_c);
    fftw_free(real_d);
    fftw_free(spec);
}

FftwXY::FftwXY(const Params& params)
    : FftwXY(params, params.nz + 1) {}

FftwXY::FftwXY(const Params& params, int max_planes)
    : nx_(params.nx),
      ny_(params.ny),
      nkx_(params.nkx()),
      max_planes_(max_planes),
      real_size_(static_cast<std::size_t>(params.nx) * static_cast<std::size_t>(params.ny)),
      spectral_size_(static_cast<std::size_t>(params.nkx()) * static_cast<std::size_t>(params.ny)) {
    if (max_planes_ <= 0) {
        throw std::runtime_error("FFT max_planes must be positive");
    }
    real_ = static_cast<double*>(fftw_malloc(sizeof(double) * real_size_ * static_cast<std::size_t>(max_planes_)));
    spec_ = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * spectral_size_ * static_cast<std::size_t>(max_planes_)));
    if (real_ == nullptr || spec_ == nullptr) {
        throw std::runtime_error("failed to allocate FFTW work buffers");
    }
}

FftwXY::~FftwXY() {
    for (auto& [_, plan] : forward_plans_) {
        if (plan != nullptr) {
            fftw_destroy_plan(plan);
        }
    }
    for (auto& [_, plan] : inverse_plans_) {
        if (plan != nullptr) {
            fftw_destroy_plan(plan);
        }
    }
    fftw_free(real_);
    fftw_free(spec_);
}

void FftwXY::forward_planes(const Field& in, int planes, SpectralField& out, const Params& params) {
    validate_planes(planes);
    fftw_plan plan = forward_plan(planes);
    out.resize(static_cast<std::size_t>(planes) * static_cast<std::size_t>(params.ny) * static_cast<std::size_t>(params.nkx()));
    for (int k = 0; k < planes; ++k) {
        copy_plane_to_real(in, k, k);
    }
    fftw_execute(plan);
    for (int k = 0; k < planes; ++k) {
        copy_spec_to_output(out, params, k, k);
    }
}

void FftwXY::inverse_planes(const SpectralField& in, int planes, Field& out, const Params& params) {
    validate_planes(planes);
    fftw_plan plan = inverse_plan(planes);
    out.resize(static_cast<std::size_t>(planes) * static_cast<std::size_t>(params.ny) * static_cast<std::size_t>(params.nx));
    const double scale = 1.0 / static_cast<double>(params.nx * params.ny);
    for (int k = 0; k < planes; ++k) {
        copy_input_to_spec(in, params, k, k);
    }
    fftw_execute(plan);
    for (int k = 0; k < planes; ++k) {
        copy_real_to_plane(out, k, k, scale);
    }
}

void FftwXY::forward_all(const Field& in, SpectralField& out, const Params& params) {
    forward_planes(in, params.nz, out, params);
}

void FftwXY::inverse_all(const SpectralField& in, Field& out, const Params& params) {
    inverse_planes(in, params.nz, out, params);
}

void FftwXY::derivative_x_planes(const Field& in, int planes, Field& out, const Params& params) {
    forward_planes(in, planes, spectral_scratch_, params);
    spectral_derivative_x_planes(spectral_scratch_, planes, out, params);
}

void FftwXY::derivative_y_planes(const Field& in, int planes, Field& out, const Params& params) {
    forward_planes(in, planes, spectral_scratch_, params);
    spectral_derivative_y_planes(spectral_scratch_, planes, out, params);
}

void FftwXY::derivative_x(const Field& in, Field& out, const Params& params) {
    derivative_x_planes(in, params.nz, out, params);
}

void FftwXY::derivative_y(const Field& in, Field& out, const Params& params) {
    derivative_y_planes(in, params.nz, out, params);
}

void FftwXY::spectral_derivative_x_planes(const SpectralField& spec, int planes, Field& out, const Params& params) {
    spectral_derivative_scratch_.resize(spec.size());
    for (int k = 0; k < planes; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int ih = 0; ih < params.nkx(); ++ih) {
                const Complex i_kx{0.0, kx_derivative_value(params, ih)};
                spectral_derivative_scratch_[sidx(params, ih, j, k)] = i_kx * spec[sidx(params, ih, j, k)];
            }
        }
    }
    inverse_planes(spectral_derivative_scratch_, planes, out, params);
}

void FftwXY::spectral_derivative_y_planes(const SpectralField& spec, int planes, Field& out, const Params& params) {
    spectral_derivative_scratch_.resize(spec.size());
    for (int k = 0; k < planes; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            const double ky = ky_derivative_value(params, j);
            for (int ih = 0; ih < params.nkx(); ++ih) {
                const Complex i_ky{0.0, ky};
                spectral_derivative_scratch_[sidx(params, ih, j, k)] = i_ky * spec[sidx(params, ih, j, k)];
            }
        }
    }
    inverse_planes(spectral_derivative_scratch_, planes, out, params);
}

void FftwXY::spectral_derivative_x(const SpectralField& spec, Field& out, const Params& params) {
    spectral_derivative_x_planes(spec, params.nz, out, params);
}

void FftwXY::spectral_derivative_y(const SpectralField& spec, Field& out, const Params& params) {
    spectral_derivative_y_planes(spec, params.nz, out, params);
}

void FftwXY::horizontal_laplacian_planes(const Field& in, int planes, Field& out, const Params& params) {
    forward_planes(in, planes, spectral_scratch_, params);
    for (int k = 0; k < planes; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            const double ky = ky_value(params, j);
            for (int ih = 0; ih < params.nkx(); ++ih) {
                const double kx = kx_value(params, ih);
                spectral_scratch_[sidx(params, ih, j, k)] *= -(kx * kx + ky * ky);
            }
        }
    }
    inverse_planes(spectral_scratch_, planes, out, params);
}

void FftwXY::horizontal_laplacian(const Field& in, Field& out, const Params& params) {
    horizontal_laplacian_planes(in, params.nz, out, params);
}

void FftwXY::horizontal_derivatives_plane_range(
    const Field& in,
    int plane_begin,
    int planes,
    Field& dx,
    Field& dy,
    const Params& params) {
    validate_planes(planes);
    if (plane_begin < 0) {
        throw std::runtime_error("horizontal derivatives plane_begin must be non-negative");
    }
    const std::size_t total_planes = in.size() / real_size_;
    if (in.size() % real_size_ != 0
        || static_cast<std::size_t>(plane_begin + planes) > total_planes) {
        throw std::runtime_error("horizontal derivatives plane range exceeds input field");
    }

    fftw_plan forward = forward_plan(planes);
    fftw_plan inverse = inverse_plan(planes);
    const std::size_t spectral_count = static_cast<std::size_t>(planes) * spectral_size_;
    spectral_scratch_.resize(spectral_count);

    for (int k = 0; k < planes; ++k) {
        copy_plane_to_real(in, plane_begin + k, k);
    }
    fftw_execute(forward);
    for (std::size_t n = 0; n < spectral_count; ++n) {
        spectral_scratch_[n] = Complex{spec_[n][0], spec_[n][1]};
    }

    for (int k = 0; k < planes; ++k) {
        const std::size_t batch_offset = static_cast<std::size_t>(k) * spectral_size_;
        for (int j = 0; j < params.ny; ++j) {
            for (int ih = 0; ih < params.nkx(); ++ih) {
                const Complex i_kx{0.0, kx_derivative_value(params, ih)};
                const std::size_t n = batch_offset
                    + static_cast<std::size_t>(j) * static_cast<std::size_t>(params.nkx())
                    + static_cast<std::size_t>(ih);
                const Complex value = i_kx * spectral_scratch_[n];
                spec_[n][0] = value.real();
                spec_[n][1] = value.imag();
            }
        }
    }
    fftw_execute(inverse);
    if (dx.size() != in.size()) {
        dx.resize(in.size());
    }
    const double scale = 1.0 / static_cast<double>(params.nx * params.ny);
    for (int k = 0; k < planes; ++k) {
        copy_real_to_plane(dx, plane_begin + k, k, scale);
    }

    for (int k = 0; k < planes; ++k) {
        const std::size_t batch_offset = static_cast<std::size_t>(k) * spectral_size_;
        for (int j = 0; j < params.ny; ++j) {
            const Complex i_ky{0.0, ky_derivative_value(params, j)};
            for (int ih = 0; ih < params.nkx(); ++ih) {
                const std::size_t n = batch_offset
                    + static_cast<std::size_t>(j) * static_cast<std::size_t>(params.nkx())
                    + static_cast<std::size_t>(ih);
                const Complex value = i_ky * spectral_scratch_[n];
                spec_[n][0] = value.real();
                spec_[n][1] = value.imag();
            }
        }
    }
    fftw_execute(inverse);
    if (dy.size() != in.size()) {
        dy.resize(in.size());
    }
    for (int k = 0; k < planes; ++k) {
        copy_real_to_plane(dy, plane_begin + k, k, scale);
    }
}

void FftwXY::horizontal_derivatives_laplacian_plane_range(
    const Field& in,
    int plane_begin,
    int planes,
    Field& dx,
    Field& dy,
    Field& lap,
    const Params& params) {
    validate_planes(planes);
    if (plane_begin < 0) {
        throw std::runtime_error("horizontal derivative/laplacian plane_begin must be non-negative");
    }
    const std::size_t total_planes = in.size() / real_size_;
    if (in.size() % real_size_ != 0
        || static_cast<std::size_t>(plane_begin + planes) > total_planes) {
        throw std::runtime_error("horizontal derivative/laplacian plane range exceeds input field");
    }

    fftw_plan forward = forward_plan(planes);
    fftw_plan inverse = inverse_plan(planes);
    const std::size_t spectral_count = static_cast<std::size_t>(planes) * spectral_size_;
    spectral_scratch_.resize(spectral_count);

    for (int k = 0; k < planes; ++k) {
        copy_plane_to_real(in, plane_begin + k, k);
    }
    fftw_execute(forward);
    for (std::size_t n = 0; n < spectral_count; ++n) {
        spectral_scratch_[n] = Complex{spec_[n][0], spec_[n][1]};
    }

    for (int k = 0; k < planes; ++k) {
        const std::size_t batch_offset = static_cast<std::size_t>(k) * spectral_size_;
        for (int j = 0; j < params.ny; ++j) {
            for (int ih = 0; ih < params.nkx(); ++ih) {
                const Complex i_kx{0.0, kx_derivative_value(params, ih)};
                const std::size_t n = batch_offset
                    + static_cast<std::size_t>(j) * static_cast<std::size_t>(params.nkx())
                    + static_cast<std::size_t>(ih);
                const Complex value = i_kx * spectral_scratch_[n];
                spec_[n][0] = value.real();
                spec_[n][1] = value.imag();
            }
        }
    }
    fftw_execute(inverse);
    if (dx.size() != in.size()) {
        dx.resize(in.size());
    }
    const double scale = 1.0 / static_cast<double>(params.nx * params.ny);
    for (int k = 0; k < planes; ++k) {
        copy_real_to_plane(dx, plane_begin + k, k, scale);
    }

    for (int k = 0; k < planes; ++k) {
        const std::size_t batch_offset = static_cast<std::size_t>(k) * spectral_size_;
        for (int j = 0; j < params.ny; ++j) {
            const Complex i_ky{0.0, ky_derivative_value(params, j)};
            for (int ih = 0; ih < params.nkx(); ++ih) {
                const std::size_t n = batch_offset
                    + static_cast<std::size_t>(j) * static_cast<std::size_t>(params.nkx())
                    + static_cast<std::size_t>(ih);
                const Complex value = i_ky * spectral_scratch_[n];
                spec_[n][0] = value.real();
                spec_[n][1] = value.imag();
            }
        }
    }
    fftw_execute(inverse);
    if (dy.size() != in.size()) {
        dy.resize(in.size());
    }
    for (int k = 0; k < planes; ++k) {
        copy_real_to_plane(dy, plane_begin + k, k, scale);
    }

    for (int k = 0; k < planes; ++k) {
        const std::size_t batch_offset = static_cast<std::size_t>(k) * spectral_size_;
        for (int j = 0; j < params.ny; ++j) {
            const double ky = ky_value(params, j);
            for (int ih = 0; ih < params.nkx(); ++ih) {
                const double kx = kx_value(params, ih);
                const std::size_t n = batch_offset
                    + static_cast<std::size_t>(j) * static_cast<std::size_t>(params.nkx())
                    + static_cast<std::size_t>(ih);
                const Complex value = -(kx * kx + ky * ky) * spectral_scratch_[n];
                spec_[n][0] = value.real();
                spec_[n][1] = value.imag();
            }
        }
    }
    fftw_execute(inverse);
    if (lap.size() != in.size()) {
        lap.resize(in.size());
    }
    for (int k = 0; k < planes; ++k) {
        copy_real_to_plane(lap, plane_begin + k, k, scale);
    }
}

void FftwXY::horizontal_divergence_plane_range(
    const Field& flux_x,
    const Field& flux_y,
    int plane_begin,
    int planes,
    Field& out,
    const Params& params) {
    if (flux_x.size() != flux_y.size()) {
        throw std::runtime_error("horizontal divergence requires matching flux field sizes");
    }
    validate_planes(planes);
    if (plane_begin < 0) {
        throw std::runtime_error("horizontal divergence plane_begin must be non-negative");
    }
    const std::size_t total_planes = flux_x.size() / real_size_;
    if (flux_x.size() % real_size_ != 0
        || static_cast<std::size_t>(plane_begin + planes) > total_planes) {
        throw std::runtime_error("horizontal divergence plane range exceeds input field");
    }

    fftw_plan forward = forward_plan(planes);
    fftw_plan inverse = inverse_plan(planes);
    const std::size_t spectral_count = static_cast<std::size_t>(planes) * spectral_size_;
    spectral_scratch_.resize(spectral_count);

    for (int k = 0; k < planes; ++k) {
        copy_plane_to_real(flux_x, plane_begin + k, k);
    }
    fftw_execute(forward);
    for (std::size_t n = 0; n < spectral_count; ++n) {
        spectral_scratch_[n] = Complex{spec_[n][0], spec_[n][1]};
    }

    for (int k = 0; k < planes; ++k) {
        copy_plane_to_real(flux_y, plane_begin + k, k);
    }
    fftw_execute(forward);
    for (int k = 0; k < planes; ++k) {
        const std::size_t batch_offset = static_cast<std::size_t>(k) * spectral_size_;
        for (int j = 0; j < params.ny; ++j) {
            const Complex i_ky{0.0, ky_derivative_value(params, j)};
            for (int ih = 0; ih < params.nkx(); ++ih) {
                const Complex i_kx{0.0, kx_derivative_value(params, ih)};
                const std::size_t n = batch_offset
                    + static_cast<std::size_t>(j) * static_cast<std::size_t>(params.nkx())
                    + static_cast<std::size_t>(ih);
                const Complex y_hat{spec_[n][0], spec_[n][1]};
                const Complex div_hat = i_kx * spectral_scratch_[n] + i_ky * y_hat;
                spec_[n][0] = div_hat.real();
                spec_[n][1] = div_hat.imag();
            }
        }
    }

    fftw_execute(inverse);
    if (out.size() != flux_x.size()) {
        out.resize(flux_x.size());
    }
    const double scale = 1.0 / static_cast<double>(params.nx * params.ny);
    for (int k = 0; k < planes; ++k) {
        copy_real_to_plane(out, plane_begin + k, k, scale);
    }
}

void FftwXY::horizontal_advective_derivative_3_2(
    const Field& advect_x,
    const Field& advect_y,
    const Field& q,
    int plane_begin,
    int planes,
    Field& out,
    const Params& params) {
    if (advect_x.size() != advect_y.size() || advect_x.size() != q.size()) {
        throw std::runtime_error("3/2 advective derivative requires matching field sizes");
    }
    validate_planes(planes);
    if (plane_begin < 0) {
        throw std::runtime_error("3/2 advective derivative plane_begin must be non-negative");
    }
    const std::size_t total_planes = q.size() / real_size_;
    if (q.size() % real_size_ != 0 || static_cast<std::size_t>(plane_begin + planes) > total_planes) {
        throw std::runtime_error("3/2 advective derivative plane range exceeds input field");
    }

    PaddedWorkspace& ws = padded_workspace(planes, params);
    const int padded_nx = ws.nx;
    const int padded_ny = ws.ny;
    const int padded_nkx = ws.nkx;
    const std::size_t padded_spectral_size = ws.spectral_size;
    const std::size_t padded_real_count = ws.real_count;
    const std::size_t padded_spectral_count = ws.spectral_count;
    double* padded_advect_x = ws.real_a;
    double* padded_advect_y = ws.real_b;
    double* padded_dqdx = ws.real_c;
    double* padded_dqdy = ws.real_d;
    fftw_complex* padded_spec = ws.spec;

    auto padded_index = [=](int ih, int j, int k) {
        return static_cast<std::size_t>(k) * padded_spectral_size
            + static_cast<std::size_t>(j) * static_cast<std::size_t>(padded_nkx)
            + static_cast<std::size_t>(ih);
    };
    auto padded_y_index = [=](int j) {
        int signed_j = j;
        if (signed_j >= params.ny / 2) {
            signed_j -= params.ny;
        }
        return signed_j >= 0 ? signed_j : padded_ny + signed_j;
    };
    auto is_nyquist_mode = [&](int ih, int j) {
        return (params.nx % 2 == 0 && ih == params.nx / 2)
            || (params.ny % 2 == 0 && j == params.ny / 2);
    };
    auto to_padded_real = [&](const Field& input, double* padded_real, int derivative_direction) {
        for (std::size_t n = 0; n < padded_spectral_count; ++n) {
            padded_spec[n][0] = 0.0;
            padded_spec[n][1] = 0.0;
        }
        fftw_plan forward = forward_plan(planes);
        for (int k = 0; k < planes; ++k) {
            copy_plane_to_real(input, plane_begin + k, k);
        }
        fftw_execute(forward);
        for (int k = 0; k < planes; ++k) {
            const std::size_t source_offset = static_cast<std::size_t>(k) * spectral_size_;
            for (int j = 0; j < params.ny; ++j) {
                const int padded_j = padded_y_index(j);
                for (int ih = 0; ih < params.nkx(); ++ih) {
                    Complex value{
                        spec_[source_offset + static_cast<std::size_t>(j) * static_cast<std::size_t>(params.nkx())
                            + static_cast<std::size_t>(ih)][0],
                        spec_[source_offset + static_cast<std::size_t>(j) * static_cast<std::size_t>(params.nkx())
                            + static_cast<std::size_t>(ih)][1]};
                    if (is_nyquist_mode(ih, j)) {
                        value = Complex{0.0, 0.0};
                    }
                    if (derivative_direction == 1) {
                        value *= Complex{0.0, kx_derivative_value(params, ih)};
                    } else if (derivative_direction == 2) {
                        value *= Complex{0.0, ky_derivative_value(params, j)};
                    }
                    const std::size_t target = padded_index(ih, padded_j, k);
                    padded_spec[target][0] = value.real();
                    padded_spec[target][1] = value.imag();
                }
            }
        }
        fftw_execute_dft_c2r(ws.inverse, padded_spec, padded_real);
        const double scale = 1.0 / static_cast<double>(params.nx * params.ny);
        for (std::size_t n = 0; n < padded_real_count; ++n) {
            padded_real[n] *= scale;
        }
    };

    to_padded_real(advect_x, padded_advect_x, 0);
    to_padded_real(advect_y, padded_advect_y, 0);
    to_padded_real(q, padded_dqdx, 1);
    to_padded_real(q, padded_dqdy, 2);
    for (std::size_t n = 0; n < padded_real_count; ++n) {
        padded_dqdx[n] = padded_advect_x[n] * padded_dqdx[n] + padded_advect_y[n] * padded_dqdy[n];
    }
    fftw_execute_dft_r2c(ws.forward, padded_dqdx, padded_spec);

    const double truncate_scale = static_cast<double>(params.nx * params.ny)
        / static_cast<double>(padded_nx * padded_ny);
    for (int k = 0; k < planes; ++k) {
        const std::size_t target_offset = static_cast<std::size_t>(k) * spectral_size_;
        for (int j = 0; j < params.ny; ++j) {
            const int padded_j = padded_y_index(j);
            for (int ih = 0; ih < params.nkx(); ++ih) {
                const std::size_t source = padded_index(ih, padded_j, k);
                const std::size_t target = target_offset
                    + static_cast<std::size_t>(j) * static_cast<std::size_t>(params.nkx())
                    + static_cast<std::size_t>(ih);
                if (is_nyquist_mode(ih, j)) {
                    spec_[target][0] = 0.0;
                    spec_[target][1] = 0.0;
                } else {
                    spec_[target][0] = truncate_scale * padded_spec[source][0];
                    spec_[target][1] = truncate_scale * padded_spec[source][1];
                }
            }
        }
    }

    fftw_execute(inverse_plan(planes));
    if (out.size() != q.size()) {
        out.resize(q.size());
    }
    const double output_scale = 1.0 / static_cast<double>(params.nx * params.ny);
    for (int k = 0; k < planes; ++k) {
        copy_real_to_plane(out, plane_begin + k, k, output_scale);
    }

}

void FftwXY::horizontal_flux_divergence_3_2(
    const Field& advect_x,
    const Field& advect_y,
    const Field& q,
    int plane_begin,
    int planes,
    Field& out,
    const Params& params) {
    if (advect_x.size() != advect_y.size() || advect_x.size() != q.size()) {
        throw std::runtime_error("3/2 flux divergence requires matching field sizes");
    }
    validate_planes(planes);
    if (plane_begin < 0) {
        throw std::runtime_error("3/2 flux divergence plane_begin must be non-negative");
    }
    const std::size_t total_planes = q.size() / real_size_;
    if (q.size() % real_size_ != 0 || static_cast<std::size_t>(plane_begin + planes) > total_planes) {
        throw std::runtime_error("3/2 flux divergence plane range exceeds input field");
    }

    PaddedWorkspace& ws = padded_workspace(planes, params);
    const int padded_nx = ws.nx;
    const int padded_ny = ws.ny;
    const int padded_nkx = ws.nkx;
    const std::size_t padded_spectral_size = ws.spectral_size;
    const std::size_t padded_real_count = ws.real_count;
    const std::size_t padded_spectral_count = ws.spectral_count;
    double* padded_advect_x = ws.real_a;
    double* padded_advect_y = ws.real_b;
    double* padded_q = ws.real_c;
    double* padded_flux = ws.real_d;
    fftw_complex* padded_spec = ws.spec;

    auto padded_index = [=](int ih, int j, int k) {
        return static_cast<std::size_t>(k) * padded_spectral_size
            + static_cast<std::size_t>(j) * static_cast<std::size_t>(padded_nkx)
            + static_cast<std::size_t>(ih);
    };
    auto padded_y_index = [=](int j) {
        int signed_j = j;
        if (signed_j >= params.ny / 2) {
            signed_j -= params.ny;
        }
        return signed_j >= 0 ? signed_j : padded_ny + signed_j;
    };
    auto is_nyquist_mode = [&](int ih, int j) {
        return (params.nx % 2 == 0 && ih == params.nx / 2)
            || (params.ny % 2 == 0 && j == params.ny / 2);
    };
    auto to_padded_real = [&](const Field& input, double* padded_real) {
        for (std::size_t n = 0; n < padded_spectral_count; ++n) {
            padded_spec[n][0] = 0.0;
            padded_spec[n][1] = 0.0;
        }
        fftw_plan forward = forward_plan(planes);
        for (int k = 0; k < planes; ++k) {
            copy_plane_to_real(input, plane_begin + k, k);
        }
        fftw_execute(forward);
        for (int k = 0; k < planes; ++k) {
            const std::size_t source_offset = static_cast<std::size_t>(k) * spectral_size_;
            for (int j = 0; j < params.ny; ++j) {
                const int padded_j = padded_y_index(j);
                for (int ih = 0; ih < params.nkx(); ++ih) {
                    const std::size_t source = source_offset
                        + static_cast<std::size_t>(j) * static_cast<std::size_t>(params.nkx())
                        + static_cast<std::size_t>(ih);
                    const std::size_t target = padded_index(ih, padded_j, k);
                    if (is_nyquist_mode(ih, j)) {
                        padded_spec[target][0] = 0.0;
                        padded_spec[target][1] = 0.0;
                    } else {
                        padded_spec[target][0] = spec_[source][0];
                        padded_spec[target][1] = spec_[source][1];
                    }
                }
            }
        }
        fftw_execute_dft_c2r(ws.inverse, padded_spec, padded_real);
        const double scale = 1.0 / static_cast<double>(params.nx * params.ny);
        for (std::size_t n = 0; n < padded_real_count; ++n) {
            padded_real[n] *= scale;
        }
    };
    auto add_truncated_divergence = [&](int derivative_direction, bool reset_output) {
        fftw_execute_dft_r2c(ws.forward, padded_flux, padded_spec);
        const double truncate_scale = static_cast<double>(params.nx * params.ny)
            / static_cast<double>(padded_nx * padded_ny);
        for (int k = 0; k < planes; ++k) {
            const std::size_t target_offset = static_cast<std::size_t>(k) * spectral_size_;
            for (int j = 0; j < params.ny; ++j) {
                const int padded_j = padded_y_index(j);
                const double wave = derivative_direction == 1
                    ? kx_derivative_value(params, 0)
                    : ky_derivative_value(params, j);
                for (int ih = 0; ih < params.nkx(); ++ih) {
                    const double component_wave = derivative_direction == 1
                        ? kx_derivative_value(params, ih)
                        : wave;
                    const std::size_t source = padded_index(ih, padded_j, k);
                    const Complex value{
                        truncate_scale * padded_spec[source][0],
                        truncate_scale * padded_spec[source][1]};
                    const Complex deriv = Complex{0.0, component_wave} * value;
                    const std::size_t target = target_offset
                        + static_cast<std::size_t>(j) * static_cast<std::size_t>(params.nkx())
                        + static_cast<std::size_t>(ih);
                    if (is_nyquist_mode(ih, j)) {
                        spec_[target][0] = 0.0;
                        spec_[target][1] = 0.0;
                        continue;
                    }
                    if (reset_output) {
                        spec_[target][0] = deriv.real();
                        spec_[target][1] = deriv.imag();
                    } else {
                        spec_[target][0] += deriv.real();
                        spec_[target][1] += deriv.imag();
                    }
                }
            }
        }
    };

    to_padded_real(advect_x, padded_advect_x);
    to_padded_real(advect_y, padded_advect_y);
    to_padded_real(q, padded_q);
    for (std::size_t n = 0; n < padded_real_count; ++n) {
        padded_flux[n] = padded_advect_x[n] * padded_q[n];
    }
    add_truncated_divergence(1, true);
    for (std::size_t n = 0; n < padded_real_count; ++n) {
        padded_flux[n] = padded_advect_y[n] * padded_q[n];
    }
    add_truncated_divergence(2, false);

    fftw_execute(inverse_plan(planes));
    if (out.size() != q.size()) {
        out.resize(q.size());
    }
    const double output_scale = 1.0 / static_cast<double>(params.nx * params.ny);
    for (int k = 0; k < planes; ++k) {
        copy_real_to_plane(out, plane_begin + k, k, output_scale);
    }

}

void FftwXY::filter_plane(const Field& in, int k, Field& out_plane, const Params& params, double filter_width) {
    if (filter_width <= 0.0) {
        throw std::runtime_error("filter_width must be positive");
    }
    validate_planes(1);
    fftw_plan forward = forward_plan(1);
    fftw_plan inverse = inverse_plan(1);
    out_plane.resize(static_cast<std::size_t>(params.nx) * static_cast<std::size_t>(params.ny));
    copy_plane_to_real(in, k, 0);
    fftw_execute(forward);
    apply_configured_filter(1, params, filter_width);
    fftw_execute(inverse);
    const double scale = 1.0 / static_cast<double>(params.nx * params.ny);
    copy_real_to_flat_plane(out_plane, 0, scale);
}

void FftwXY::filter_planes(const Field& in, int planes, Field& out, const Params& params, double filter_width) {
    if (filter_width <= 0.0) {
        throw std::runtime_error("filter_width must be positive");
    }
    validate_planes(planes);
    fftw_plan forward = forward_plan(planes);
    fftw_plan inverse = inverse_plan(planes);
    out.resize(static_cast<std::size_t>(planes) * real_size_);
    for (int k = 0; k < planes; ++k) {
        copy_plane_to_real(in, k, k);
    }
    fftw_execute(forward);
    apply_configured_filter(planes, params, filter_width);
    fftw_execute(inverse);
    const double scale = 1.0 / static_cast<double>(params.nx * params.ny);
    for (int k = 0; k < planes; ++k) {
        copy_real_to_plane(out, k, k, scale);
    }
}

void FftwXY::filter_plane_range(
    const Field& in,
    int plane_begin,
    int planes,
    Field& out,
    const Params& params,
    double filter_width) {
    if (filter_width <= 0.0) {
        throw std::runtime_error("filter_width must be positive");
    }
    validate_planes(planes);
    fftw_plan forward = forward_plan(planes);
    fftw_plan inverse = inverse_plan(planes);
    if (out.size() != in.size()) {
        out.resize(in.size());
    }
    for (int k = 0; k < planes; ++k) {
        copy_plane_to_real(in, plane_begin + k, k);
    }
    fftw_execute(forward);
    apply_configured_filter(planes, params, filter_width);
    fftw_execute(inverse);
    const double scale = 1.0 / static_cast<double>(params.nx * params.ny);
    for (int k = 0; k < planes; ++k) {
        copy_real_to_plane(out, plane_begin + k, k, scale);
    }
}

void FftwXY::clear_nyquist_planes(const Field& in, int planes, Field& out, const Params& params) {
    validate_planes(planes);
    fftw_plan forward = forward_plan(planes);
    fftw_plan inverse = inverse_plan(planes);
    out.resize(static_cast<std::size_t>(planes) * real_size_);
    for (int k = 0; k < planes; ++k) {
        copy_plane_to_real(in, k, k);
    }
    fftw_execute(forward);
    apply_nyquist_zero(planes, params);
    fftw_execute(inverse);
    const double scale = 1.0 / static_cast<double>(params.nx * params.ny);
    for (int k = 0; k < planes; ++k) {
        copy_real_to_plane(out, k, k, scale);
    }
}

void FftwXY::clear_nyquist_plane_range(
    const Field& in,
    int plane_begin,
    int planes,
    Field& out,
    const Params& params) {
    validate_planes(planes);
    fftw_plan forward = forward_plan(planes);
    fftw_plan inverse = inverse_plan(planes);
    if (out.size() != in.size()) {
        out.resize(in.size());
    }
    for (int k = 0; k < planes; ++k) {
        copy_plane_to_real(in, plane_begin + k, k);
    }
    fftw_execute(forward);
    apply_nyquist_zero(planes, params);
    fftw_execute(inverse);
    const double scale = 1.0 / static_cast<double>(params.nx * params.ny);
    for (int k = 0; k < planes; ++k) {
        copy_real_to_plane(out, plane_begin + k, k, scale);
    }
}

void FftwXY::filter_plane_fortran_sharp(const Field& in, int k, Field& out_plane, const Params& params, double filter_width) {
    if (filter_width <= 0.0) {
        throw std::runtime_error("filter_width must be positive");
    }
    if (params.ly <= 0.0) {
        throw std::runtime_error("ly must be positive for spectral filtering");
    }
    validate_planes(1);
    fftw_plan forward = forward_plan(1);
    fftw_plan inverse = inverse_plan(1);
    out_plane.resize(static_cast<std::size_t>(params.nx) * static_cast<std::size_t>(params.ny));
    copy_plane_to_real(in, k, 0);
    fftw_execute(forward);
    apply_fortran_sharp_cutoff(1, params, filter_width);
    fftw_execute(inverse);
    const double scale = 1.0 / static_cast<double>(params.nx * params.ny);
    copy_real_to_flat_plane(out_plane, 0, scale);
}

void FftwXY::filter_planes_fortran_sharp(const Field& in, int planes, Field& out, const Params& params, double filter_width) {
    if (filter_width <= 0.0) {
        throw std::runtime_error("filter_width must be positive");
    }
    validate_planes(planes);
    fftw_plan forward = forward_plan(planes);
    fftw_plan inverse = inverse_plan(planes);
    out.resize(static_cast<std::size_t>(planes) * real_size_);
    for (int k = 0; k < planes; ++k) {
        copy_plane_to_real(in, k, k);
    }
    fftw_execute(forward);
    apply_fortran_sharp_cutoff(planes, params, filter_width);
    fftw_execute(inverse);
    const double scale = 1.0 / static_cast<double>(params.nx * params.ny);
    for (int k = 0; k < planes; ++k) {
        copy_real_to_plane(out, k, k, scale);
    }
}

void FftwXY::filter_plane_range_fortran_sharp(
    const Field& in,
    int plane_begin,
    int planes,
    Field& out,
    const Params& params,
    double filter_width) {
    if (filter_width <= 0.0) {
        throw std::runtime_error("filter_width must be positive");
    }
    validate_planes(planes);
    fftw_plan forward = forward_plan(planes);
    fftw_plan inverse = inverse_plan(planes);
    if (out.size() != in.size()) {
        out.resize(in.size());
    }
    for (int k = 0; k < planes; ++k) {
        copy_plane_to_real(in, plane_begin + k, k);
    }
    fftw_execute(forward);
    apply_fortran_sharp_cutoff(planes, params, filter_width);
    fftw_execute(inverse);
    const double scale = 1.0 / static_cast<double>(params.nx * params.ny);
    for (int k = 0; k < planes; ++k) {
        copy_real_to_plane(out, plane_begin + k, k, scale);
    }
}

void FftwXY::filter_many_plane_range_fortran_sharp(
    const std::vector<const Field*>& inputs,
    int plane_begin,
    int planes,
    const std::vector<Field*>& outputs,
    const Params& params,
    double filter_width) {
    if (filter_width <= 0.0) {
        throw std::runtime_error("filter_width must be positive");
    }
    if (inputs.size() != outputs.size()) {
        throw std::runtime_error("filter_many_plane_range_fortran_sharp requires matching input/output counts");
    }
    if (inputs.empty()) {
        return;
    }
    validate_planes(planes);
    if (plane_begin < 0) {
        throw std::runtime_error("filter_many plane_begin must be non-negative");
    }
    for (std::size_t field = 0; field < inputs.size(); ++field) {
        if (inputs[field] == nullptr || outputs[field] == nullptr) {
            throw std::runtime_error("filter_many requires non-null field pointers");
        }
        if (inputs[field]->size() % real_size_ != 0) {
            throw std::runtime_error("filter_many input field size is not a whole number of planes");
        }
        const std::size_t total_planes = inputs[field]->size() / real_size_;
        if (static_cast<std::size_t>(plane_begin + planes) > total_planes) {
            throw std::runtime_error("filter_many plane range exceeds input field");
        }
        if (outputs[field]->size() != inputs[field]->size()) {
            outputs[field]->resize(inputs[field]->size());
        }
    }

    const int fields_per_batch = std::max(1, max_planes_ / planes);
    const double scale = 1.0 / static_cast<double>(params.nx * params.ny);
    for (std::size_t first = 0; first < inputs.size(); first += static_cast<std::size_t>(fields_per_batch)) {
        const int field_count = static_cast<int>(
            std::min<std::size_t>(static_cast<std::size_t>(fields_per_batch), inputs.size() - first));
        const int batch_planes = field_count * planes;
        validate_planes(batch_planes);
        fftw_plan forward = forward_plan(batch_planes);
        fftw_plan inverse = inverse_plan(batch_planes);

        for (int field = 0; field < field_count; ++field) {
            for (int k = 0; k < planes; ++k) {
                copy_plane_to_real(*inputs[first + static_cast<std::size_t>(field)], plane_begin + k, field * planes + k);
            }
        }
        fftw_execute(forward);
        apply_fortran_sharp_cutoff(batch_planes, params, filter_width);
        fftw_execute(inverse);
        for (int field = 0; field < field_count; ++field) {
            for (int k = 0; k < planes; ++k) {
                copy_real_to_plane(
                    *outputs[first + static_cast<std::size_t>(field)],
                    plane_begin + k,
                    field * planes + k,
                    scale);
            }
        }
    }
}

void FftwXY::validate_planes(int planes) const {
    if (planes <= 0 || planes > max_planes_) {
        throw std::runtime_error("invalid FFT plane count: " + std::to_string(planes));
    }
}

fftw_plan FftwXY::forward_plan(int planes) {
    validate_planes(planes);
    const auto found = forward_plans_.find(planes);
    if (found != forward_plans_.end()) {
        return found->second;
    }
    int n[2] = {ny_, nx_};
    int inembed[2] = {ny_, nx_};
    int onembed[2] = {ny_, nkx_};
    fftw_plan plan = fftw_plan_many_dft_r2c(
        2,
        n,
        planes,
        real_,
        inembed,
        1,
        static_cast<int>(real_size_),
        spec_,
        onembed,
        1,
        static_cast<int>(spectral_size_),
        FFTW_MEASURE);
    if (plan == nullptr) {
        throw std::runtime_error("failed to create batched FFTW forward plan for " + std::to_string(planes) + " planes");
    }
    forward_plans_.emplace(planes, plan);
    return plan;
}

fftw_plan FftwXY::inverse_plan(int planes) {
    validate_planes(planes);
    const auto found = inverse_plans_.find(planes);
    if (found != inverse_plans_.end()) {
        return found->second;
    }
    int n[2] = {ny_, nx_};
    int inembed[2] = {ny_, nkx_};
    int onembed[2] = {ny_, nx_};
    fftw_plan plan = fftw_plan_many_dft_c2r(
        2,
        n,
        planes,
        spec_,
        inembed,
        1,
        static_cast<int>(spectral_size_),
        real_,
        onembed,
        1,
        static_cast<int>(real_size_),
        FFTW_MEASURE);
    if (plan == nullptr) {
        throw std::runtime_error("failed to create batched FFTW inverse plan for " + std::to_string(planes) + " planes");
    }
    inverse_plans_.emplace(planes, plan);
    return plan;
}

FftwXY::PaddedWorkspace& FftwXY::padded_workspace(int planes, const Params& params) {
    validate_planes(planes);
    const auto found = padded_workspaces_.find(planes);
    if (found != padded_workspaces_.end()) {
        return *found->second;
    }

    auto workspace = std::make_unique<PaddedWorkspace>();
    workspace->planes = planes;
    workspace->nx = 3 * params.nx / 2;
    workspace->ny = 3 * params.ny / 2;
    workspace->nkx = workspace->nx / 2 + 1;
    if (workspace->nx <= params.nx || workspace->ny <= params.ny) {
        throw std::runtime_error("3/2 padding requires positive horizontal dimensions");
    }
    workspace->real_size = static_cast<std::size_t>(workspace->nx) * static_cast<std::size_t>(workspace->ny);
    workspace->spectral_size = static_cast<std::size_t>(workspace->nkx) * static_cast<std::size_t>(workspace->ny);
    workspace->real_count = static_cast<std::size_t>(planes) * workspace->real_size;
    workspace->spectral_count = static_cast<std::size_t>(planes) * workspace->spectral_size;

    workspace->real_a = static_cast<double*>(fftw_malloc(sizeof(double) * workspace->real_count));
    workspace->real_b = static_cast<double*>(fftw_malloc(sizeof(double) * workspace->real_count));
    workspace->real_c = static_cast<double*>(fftw_malloc(sizeof(double) * workspace->real_count));
    workspace->real_d = static_cast<double*>(fftw_malloc(sizeof(double) * workspace->real_count));
    workspace->spec = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * workspace->spectral_count));
    if (workspace->real_a == nullptr || workspace->real_b == nullptr || workspace->real_c == nullptr
        || workspace->real_d == nullptr || workspace->spec == nullptr) {
        throw std::runtime_error("failed to allocate cached 3/2 padded FFT buffers");
    }

    int n[2] = {workspace->ny, workspace->nx};
    int real_embed[2] = {workspace->ny, workspace->nx};
    int spec_embed[2] = {workspace->ny, workspace->nkx};
    workspace->inverse = fftw_plan_many_dft_c2r(
        2,
        n,
        planes,
        workspace->spec,
        spec_embed,
        1,
        static_cast<int>(workspace->spectral_size),
        workspace->real_a,
        real_embed,
        1,
        static_cast<int>(workspace->real_size),
        FFTW_MEASURE);
    workspace->forward = fftw_plan_many_dft_r2c(
        2,
        n,
        planes,
        workspace->real_d,
        real_embed,
        1,
        static_cast<int>(workspace->real_size),
        workspace->spec,
        spec_embed,
        1,
        static_cast<int>(workspace->spectral_size),
        FFTW_MEASURE);
    if (workspace->inverse == nullptr || workspace->forward == nullptr) {
        throw std::runtime_error("failed to create cached 3/2 padded FFT plans");
    }

    auto [it, _] = padded_workspaces_.emplace(planes, std::move(workspace));
    return *it->second;
}

void FftwXY::copy_plane_to_real(const Field& in, int source_k, int batch_k) {
    const std::size_t source_offset = static_cast<std::size_t>(source_k) * real_size_;
    const std::size_t batch_offset = static_cast<std::size_t>(batch_k) * real_size_;
    std::copy_n(in.data() + source_offset, real_size_, real_ + batch_offset);
}

void FftwXY::copy_real_to_plane(Field& out, int out_k, int batch_k, double scale) {
    const std::size_t out_offset = static_cast<std::size_t>(out_k) * real_size_;
    const std::size_t batch_offset = static_cast<std::size_t>(batch_k) * real_size_;
    for (std::size_t n = 0; n < real_size_; ++n) {
        out[out_offset + n] = scale * real_[batch_offset + n];
    }
}

void FftwXY::copy_real_to_flat_plane(Field& out_plane, int batch_k, double scale) {
    const std::size_t batch_offset = static_cast<std::size_t>(batch_k) * real_size_;
    for (std::size_t n = 0; n < real_size_; ++n) {
        out_plane[n] = scale * real_[batch_offset + n];
    }
}

void FftwXY::copy_spec_to_output(SpectralField& out, const Params& params, int out_k, int batch_k) {
    const std::size_t batch_offset = static_cast<std::size_t>(batch_k) * spectral_size_;
    for (int j = 0; j < params.ny; ++j) {
        for (int ih = 0; ih < params.nkx(); ++ih) {
            const std::size_t n = batch_offset + static_cast<std::size_t>(j) * static_cast<std::size_t>(params.nkx())
                + static_cast<std::size_t>(ih);
            out[sidx(params, ih, j, out_k)] = Complex{spec_[n][0], spec_[n][1]};
        }
    }
}

void FftwXY::copy_input_to_spec(const SpectralField& in, const Params& params, int in_k, int batch_k) {
    const std::size_t batch_offset = static_cast<std::size_t>(batch_k) * spectral_size_;
    for (int j = 0; j < params.ny; ++j) {
        for (int ih = 0; ih < params.nkx(); ++ih) {
            const std::size_t n = batch_offset + static_cast<std::size_t>(j) * static_cast<std::size_t>(params.nkx())
                + static_cast<std::size_t>(ih);
            const Complex value = in[sidx(params, ih, j, in_k)];
            spec_[n][0] = value.real();
            spec_[n][1] = value.imag();
        }
    }
}

void FftwXY::apply_configured_filter(int planes, const Params& params, double filter_width) {
    if (params.spectral_filter == "sharp") {
        apply_fortran_sharp_cutoff(planes, params, filter_width);
    } else if (params.spectral_filter == "floor_sharp") {
        apply_floor_cutoff(planes, params, filter_width);
    } else if (params.spectral_filter == "exponential") {
        apply_exponential_filter(planes, params, filter_width);
    } else {
        throw std::runtime_error("unsupported spectral_filter: " + params.spectral_filter);
    }
}

void FftwXY::apply_floor_cutoff(int planes, const Params& params, double filter_width) {
    const double cutoff_x = std::floor(static_cast<double>(params.nx) / (2.0 * filter_width));
    const double cutoff_y = std::floor(static_cast<double>(params.ny) / (2.0 * filter_width));
    for (int k = 0; k < planes; ++k) {
        const std::size_t batch_offset = static_cast<std::size_t>(k) * spectral_size_;
        for (int j = 0; j < params.ny; ++j) {
            const double ky_mode = std::abs(static_cast<double>((j <= params.ny / 2) ? j : j - params.ny));
            for (int ih = 0; ih < params.nkx(); ++ih) {
                const bool keep = (std::abs(static_cast<double>(ih)) < cutoff_x) && (ky_mode < cutoff_y);
                if (!keep) {
                    const std::size_t n = batch_offset + static_cast<std::size_t>(j) * static_cast<std::size_t>(params.nkx())
                        + static_cast<std::size_t>(ih);
                    spec_[n][0] = 0.0;
                    spec_[n][1] = 0.0;
                }
            }
        }
    }
}

void FftwXY::apply_fortran_sharp_cutoff(int planes, const Params& params, double filter_width) {
    if (params.ly <= 0.0) {
        throw std::runtime_error("ly must be positive for spectral filtering");
    }
    // Match legacy Fortran filter_kernel: ii >= nint(nx/(2R)) or
    // abs((j-mode)*l_r) >= nint(l_r*ny/(2R)) is removed.
    const double length_ratio = params.lx / params.ly;
    const int ny_shift_cutoff = static_cast<int>(std::round(static_cast<double>(params.ny) / 2.0));
    const int cutoff_x = static_cast<int>(std::round(static_cast<double>(params.nx) / (2.0 * filter_width)));
    const double cutoff_y =
        std::round(std::abs(length_ratio) * static_cast<double>(params.ny) / (2.0 * filter_width));
    for (int k = 0; k < planes; ++k) {
        const std::size_t batch_offset = static_cast<std::size_t>(k) * spectral_size_;
        for (int j = 0; j < params.ny; ++j) {
            int signed_j = j;
            if (signed_j >= ny_shift_cutoff) {
                signed_j -= params.ny;
            }
            const double scaled_j = static_cast<double>(signed_j) * length_ratio;
            for (int ih = 0; ih < params.nkx(); ++ih) {
                const bool keep = (ih < cutoff_x) && (std::abs(scaled_j) < cutoff_y);
                if (!keep) {
                    const std::size_t n = batch_offset + static_cast<std::size_t>(j) * static_cast<std::size_t>(params.nkx())
                        + static_cast<std::size_t>(ih);
                    spec_[n][0] = 0.0;
                    spec_[n][1] = 0.0;
                }
            }
        }
    }
}

void FftwXY::apply_nyquist_zero(int planes, const Params& params) {
    for (int k = 0; k < planes; ++k) {
        const std::size_t batch_offset = static_cast<std::size_t>(k) * spectral_size_;
        for (int j = 0; j < params.ny; ++j) {
            const bool y_nyquist = params.ny % 2 == 0 && j == params.ny / 2;
            for (int ih = 0; ih < params.nkx(); ++ih) {
                const bool x_nyquist = params.nx % 2 == 0 && ih == params.nx / 2;
                if (x_nyquist || y_nyquist) {
                    const std::size_t n = batch_offset + static_cast<std::size_t>(j) * static_cast<std::size_t>(params.nkx())
                        + static_cast<std::size_t>(ih);
                    spec_[n][0] = 0.0;
                    spec_[n][1] = 0.0;
                }
            }
        }
    }
}

void FftwXY::apply_exponential_filter(int planes, const Params& params, double filter_width) {
    if (params.ly <= 0.0) {
        throw std::runtime_error("ly must be positive for spectral filtering");
    }
    if (params.spectral_filter_alpha < 0.0 || params.spectral_filter_order <= 0) {
        throw std::runtime_error("exponential spectral filter requires alpha >= 0 and order > 0");
    }
    const double length_ratio = params.lx / params.ly;
    const int ny_shift_cutoff = static_cast<int>(std::round(static_cast<double>(params.ny) / 2.0));
    const double cutoff_x = std::max(1.0, static_cast<double>(params.nx) / (2.0 * filter_width));
    const double cutoff_y = std::max(
        1.0,
        std::abs(length_ratio) * static_cast<double>(params.ny) / (2.0 * filter_width));
    for (int k = 0; k < planes; ++k) {
        const std::size_t batch_offset = static_cast<std::size_t>(k) * spectral_size_;
        for (int j = 0; j < params.ny; ++j) {
            int signed_j = j;
            if (signed_j >= ny_shift_cutoff) {
                signed_j -= params.ny;
            }
            const double eta_y = std::abs(static_cast<double>(signed_j) * length_ratio) / cutoff_y;
            for (int ih = 0; ih < params.nkx(); ++ih) {
                const double eta_x = std::abs(static_cast<double>(ih)) / cutoff_x;
                const double eta = std::max(eta_x, eta_y);
                const double sigma = std::exp(
                    -params.spectral_filter_alpha * std::pow(eta, params.spectral_filter_order));
                const std::size_t n = batch_offset + static_cast<std::size_t>(j) * static_cast<std::size_t>(params.nkx())
                    + static_cast<std::size_t>(ih);
                spec_[n][0] *= sigma;
                spec_[n][1] *= sigma;
            }
        }
    }
}

}  // namespace wireles

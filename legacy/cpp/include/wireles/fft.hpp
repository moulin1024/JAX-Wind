#pragma once

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <vector>

#include <fftw3.h>

#include "wireles/field.hpp"

namespace wireles {

class FftwXY {
public:
    explicit FftwXY(const Params& params);
    FftwXY(const Params& params, int max_planes);
    FftwXY(const FftwXY&) = delete;
    FftwXY& operator=(const FftwXY&) = delete;
    ~FftwXY();

    void forward_planes(const Field& in, int planes, SpectralField& out, const Params& params);
    void inverse_planes(const SpectralField& in, int planes, Field& out, const Params& params);
    void forward_all(const Field& in, SpectralField& out, const Params& params);
    void inverse_all(const SpectralField& in, Field& out, const Params& params);
    void derivative_x_planes(const Field& in, int planes, Field& out, const Params& params);
    void derivative_y_planes(const Field& in, int planes, Field& out, const Params& params);
    void derivative_x(const Field& in, Field& out, const Params& params);
    void derivative_y(const Field& in, Field& out, const Params& params);
    void spectral_derivative_x_planes(const SpectralField& spec, int planes, Field& out, const Params& params);
    void spectral_derivative_y_planes(const SpectralField& spec, int planes, Field& out, const Params& params);
    void spectral_derivative_x(const SpectralField& spec, Field& out, const Params& params);
    void spectral_derivative_y(const SpectralField& spec, Field& out, const Params& params);
    void horizontal_laplacian_planes(const Field& in, int planes, Field& out, const Params& params);
    void horizontal_laplacian(const Field& in, Field& out, const Params& params);
    void horizontal_derivatives_plane_range(
        const Field& in,
        int plane_begin,
        int planes,
        Field& dx,
        Field& dy,
        const Params& params);
    void horizontal_derivatives_laplacian_plane_range(
        const Field& in,
        int plane_begin,
        int planes,
        Field& dx,
        Field& dy,
        Field& lap,
        const Params& params);
    void horizontal_divergence_plane_range(
        const Field& flux_x,
        const Field& flux_y,
        int plane_begin,
        int planes,
        Field& out,
        const Params& params);
    void horizontal_advective_derivative_3_2(
        const Field& advect_x,
        const Field& advect_y,
        const Field& q,
        int plane_begin,
        int planes,
        Field& out,
        const Params& params);
    void horizontal_flux_divergence_3_2(
        const Field& advect_x,
        const Field& advect_y,
        const Field& q,
        int plane_begin,
        int planes,
        Field& out,
        const Params& params);
    void filter_plane(const Field& in, int k, Field& out_plane, const Params& params, double filter_width);
    void filter_planes(const Field& in, int planes, Field& out, const Params& params, double filter_width);
    void filter_plane_range(
        const Field& in,
        int plane_begin,
        int planes,
        Field& out,
        const Params& params,
        double filter_width);
    void filter_plane_fortran_sharp(const Field& in, int k, Field& out_plane, const Params& params, double filter_width);
    void filter_planes_fortran_sharp(const Field& in, int planes, Field& out, const Params& params, double filter_width);
    void filter_plane_range_fortran_sharp(
        const Field& in,
        int plane_begin,
        int planes,
        Field& out,
        const Params& params,
        double filter_width);
    void filter_many_plane_range_fortran_sharp(
        const std::vector<const Field*>& inputs,
        int plane_begin,
        int planes,
        const std::vector<Field*>& outputs,
        const Params& params,
        double filter_width);
    void clear_nyquist_planes(const Field& in, int planes, Field& out, const Params& params);
    void clear_nyquist_plane_range(const Field& in, int plane_begin, int planes, Field& out, const Params& params);

private:
    struct PaddedWorkspace {
        int planes = 0;
        int nx = 0;
        int ny = 0;
        int nkx = 0;
        std::size_t real_size = 0;
        std::size_t spectral_size = 0;
        std::size_t real_count = 0;
        std::size_t spectral_count = 0;
        double* real_a = nullptr;
        double* real_b = nullptr;
        double* real_c = nullptr;
        double* real_d = nullptr;
        fftw_complex* spec = nullptr;
        fftw_plan forward = nullptr;
        fftw_plan inverse = nullptr;

        PaddedWorkspace() = default;
        PaddedWorkspace(const PaddedWorkspace&) = delete;
        PaddedWorkspace& operator=(const PaddedWorkspace&) = delete;
        ~PaddedWorkspace();
    };

    void validate_planes(int planes) const;
    fftw_plan forward_plan(int planes);
    fftw_plan inverse_plan(int planes);
    PaddedWorkspace& padded_workspace(int planes, const Params& params);
    void copy_plane_to_real(const Field& in, int source_k, int batch_k);
    void copy_real_to_plane(Field& out, int out_k, int batch_k, double scale);
    void copy_real_to_flat_plane(Field& out_plane, int batch_k, double scale);
    void copy_spec_to_output(SpectralField& out, const Params& params, int out_k, int batch_k);
    void copy_input_to_spec(const SpectralField& in, const Params& params, int in_k, int batch_k);
    void apply_configured_filter(int planes, const Params& params, double filter_width);
    void apply_floor_cutoff(int planes, const Params& params, double filter_width);
    void apply_fortran_sharp_cutoff(int planes, const Params& params, double filter_width);
    void apply_nyquist_zero(int planes, const Params& params);
    void apply_exponential_filter(int planes, const Params& params, double filter_width);

    int nx_ = 0;
    int ny_ = 0;
    int nkx_ = 0;
    int max_planes_ = 0;
    std::size_t real_size_ = 0;
    std::size_t spectral_size_ = 0;
    double* real_ = nullptr;
    fftw_complex* spec_ = nullptr;
    std::unordered_map<int, fftw_plan> forward_plans_;
    std::unordered_map<int, fftw_plan> inverse_plans_;
    std::unordered_map<int, std::unique_ptr<PaddedWorkspace>> padded_workspaces_;
    SpectralField spectral_scratch_;
    SpectralField spectral_derivative_scratch_;
};

}  // namespace wireles

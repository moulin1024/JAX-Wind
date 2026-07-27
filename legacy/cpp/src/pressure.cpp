#include "wireles/pressure.hpp"

#include <cmath>
#include <stdexcept>

#include "wireles/operators.hpp"

namespace wireles {
namespace {

std::vector<Complex> solve_tridiagonal(
    const std::vector<Complex>& a,
    const std::vector<Complex>& b,
    const std::vector<Complex>& c,
    const std::vector<Complex>& d) {
    const int n = static_cast<int>(d.size());
    std::vector<Complex> cp(n, Complex{0.0, 0.0});
    std::vector<Complex> dp(n, Complex{0.0, 0.0});
    std::vector<Complex> x(n, Complex{0.0, 0.0});

    Complex denom = b[0];
    if (std::abs(denom) == 0.0) {
        throw std::runtime_error("singular tridiagonal pivot at row 0");
    }
    cp[0] = c[0] / denom;
    dp[0] = d[0] / denom;

    for (int i = 1; i < n; ++i) {
        denom = b[i] - a[i] * cp[i - 1];
        if (std::abs(denom) == 0.0) {
            throw std::runtime_error("singular tridiagonal pivot");
        }
        cp[i] = (i == n - 1) ? Complex{0.0, 0.0} : c[i] / denom;
        dp[i] = (d[i] - a[i] * dp[i - 1]) / denom;
    }

    x[n - 1] = dp[n - 1];
    for (int i = n - 2; i >= 0; --i) {
        x[i] = dp[i] - cp[i] * x[i + 1];
    }
    return x;
}

}  // namespace

SpectralField solve_pressure_hat(const SpectralField& div_hat, const Params& params) {
    SpectralField p_hat(params.spectral_size(), Complex{0.0, 0.0});
    const double inv_dz2 = 1.0 / (params.dz() * params.dz());

    std::vector<Complex> a(static_cast<std::size_t>(params.nz));
    std::vector<Complex> b(static_cast<std::size_t>(params.nz));
    std::vector<Complex> c(static_cast<std::size_t>(params.nz));
    std::vector<Complex> d(static_cast<std::size_t>(params.nz));

    for (int j = 0; j < params.ny; ++j) {
        const double ky = ky_derivative_value(params, j);
        for (int ih = 0; ih < params.nkx(); ++ih) {
            const double kx = kx_derivative_value(params, ih);
            const double kh2 = kx * kx + ky * ky;

            for (int k = 0; k < params.nz; ++k) {
                a[static_cast<std::size_t>(k)] = Complex{0.0, 0.0};
                b[static_cast<std::size_t>(k)] = Complex{0.0, 0.0};
                c[static_cast<std::size_t>(k)] = Complex{0.0, 0.0};
                d[static_cast<std::size_t>(k)] = div_hat[sidx(params, ih, j, k)] / params.dt;
            }

            if (kh2 == 0.0) {
                b[0] = Complex{1.0, 0.0};
                d[0] = Complex{0.0, 0.0};
                for (int k = 1; k < params.nz; ++k) {
                    a[static_cast<std::size_t>(k)] = Complex{inv_dz2, 0.0};
                    b[static_cast<std::size_t>(k)] = Complex{-2.0 * inv_dz2, 0.0};
                    c[static_cast<std::size_t>(k)] = Complex{inv_dz2, 0.0};
                }
                a[static_cast<std::size_t>(params.nz - 1)] = Complex{inv_dz2, 0.0};
                b[static_cast<std::size_t>(params.nz - 1)] = Complex{-inv_dz2, 0.0};
                c[static_cast<std::size_t>(params.nz - 1)] = Complex{0.0, 0.0};
            } else {
                b[0] = Complex{-inv_dz2 - kh2, 0.0};
                c[0] = Complex{inv_dz2, 0.0};
                for (int k = 1; k < params.nz - 1; ++k) {
                    a[static_cast<std::size_t>(k)] = Complex{inv_dz2, 0.0};
                    b[static_cast<std::size_t>(k)] = Complex{-2.0 * inv_dz2 - kh2, 0.0};
                    c[static_cast<std::size_t>(k)] = Complex{inv_dz2, 0.0};
                }
                a[static_cast<std::size_t>(params.nz - 1)] = Complex{inv_dz2, 0.0};
                b[static_cast<std::size_t>(params.nz - 1)] = Complex{-inv_dz2 - kh2, 0.0};
            }

            const std::vector<Complex> x = solve_tridiagonal(a, b, c, d);
            for (int k = 0; k < params.nz; ++k) {
                p_hat[sidx(params, ih, j, k)] = x[static_cast<std::size_t>(k)];
            }
        }
    }
    return p_hat;
}

void project(FlowState& state, const Params& params, FftwXY& fft) {
    const Field div = divergence(state.u, state.v, state.w, params, fft);
    SpectralField div_hat;
    fft.forward_all(div, div_hat, params);
    const SpectralField p_hat = solve_pressure_hat(div_hat, params);
    fft.inverse_all(p_hat, state.p, params);

    Field dpdx;
    Field dpdy;
    fft.spectral_derivative_x(p_hat, dpdx, params);
    fft.spectral_derivative_y(p_hat, dpdy, params);
    const Field dpdz = ddz_center_to_w(state.p, params);

    for (std::size_t n = 0; n < state.u.size(); ++n) {
        state.u[n] -= params.dt * dpdx[n];
        state.v[n] -= params.dt * dpdy[n];
    }
    for (std::size_t n = 0; n < state.w.size(); ++n) {
        state.w[n] -= params.dt * dpdz[n];
    }
    enforce_walls(state.w, params);
}

}  // namespace wireles

#include "wireles/operators.hpp"

#include <algorithm>

namespace wireles {

void ddz_center(const Field& q, Field& out, const Params& params) {
    out.resize(params.real_size());
    const double dz = params.dz();
    for (int k = 0; k < params.nz; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                double value = 0.0;
                if (k == 0) {
                    value = (q[idx(params, i, j, 1)] - q[idx(params, i, j, 0)]) / dz;
                } else if (k == params.nz - 1) {
                    value = (q[idx(params, i, j, params.nz - 1)] - q[idx(params, i, j, params.nz - 2)]) / dz;
                } else {
                    value = (q[idx(params, i, j, k + 1)] - q[idx(params, i, j, k - 1)]) / (2.0 * dz);
                }
                out[idx(params, i, j, k)] = value;
            }
        }
    }
}

Field ddz_center(const Field& q, const Params& params) {
    Field out;
    ddz_center(q, out, params);
    return out;
}

void d2dz2_center(const Field& q, Field& out, const Params& params) {
    out.resize(params.real_size());
    const double inv_dz2 = 1.0 / (params.dz() * params.dz());
    for (int k = 0; k < params.nz; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                double value = 0.0;
                if (k == 0) {
                    value = (q[idx(params, i, j, 1)] - q[idx(params, i, j, 0)]) * inv_dz2;
                } else if (k == params.nz - 1) {
                    value = (q[idx(params, i, j, params.nz - 2)] - q[idx(params, i, j, params.nz - 1)]) * inv_dz2;
                } else {
                    value = (q[idx(params, i, j, k - 1)] - 2.0 * q[idx(params, i, j, k)] + q[idx(params, i, j, k + 1)])
                        * inv_dz2;
                }
                out[idx(params, i, j, k)] = value;
            }
        }
    }
}

Field d2dz2_center(const Field& q, const Params& params) {
    Field out;
    d2dz2_center(q, out, params);
    return out;
}

void w_to_center(const Field& w, Field& out, const Params& params) {
    out.resize(params.real_size());
    for (int k = 0; k < params.nz; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                out[idx(params, i, j, k)] = 0.5 * (
                    w[z_face_idx(params, i, j, k)] + w[z_face_idx(params, i, j, k + 1)]);
            }
        }
    }
}

Field w_to_center(const Field& w, const Params& params) {
    Field out;
    w_to_center(w, out, params);
    return out;
}

void center_to_w(const Field& q, Field& out, const Params& params) {
    out.resize(params.z_face_size());
    for (int j = 0; j < params.ny; ++j) {
        for (int i = 0; i < params.nx; ++i) {
            out[z_face_idx(params, i, j, 0)] = q[idx(params, i, j, 0)];
            out[z_face_idx(params, i, j, params.nz)] = q[idx(params, i, j, params.nz - 1)];
        }
    }
    for (int k = 1; k < params.nz; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                out[z_face_idx(params, i, j, k)] = 0.5 * (
                    q[idx(params, i, j, k - 1)] + q[idx(params, i, j, k)]);
            }
        }
    }
}

Field center_to_w(const Field& q, const Params& params) {
    Field out;
    center_to_w(q, out, params);
    return out;
}

void ddz_w_to_center(const Field& w, Field& out, const Params& params) {
    out.resize(params.real_size());
    const double inv_dz = 1.0 / params.dz();
    for (int k = 0; k < params.nz; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                out[idx(params, i, j, k)] =
                    (w[z_face_idx(params, i, j, k + 1)] - w[z_face_idx(params, i, j, k)]) * inv_dz;
            }
        }
    }
}

Field ddz_w_to_center(const Field& w, const Params& params) {
    Field out;
    ddz_w_to_center(w, out, params);
    return out;
}

void ddz_center_to_w(const Field& q, Field& out, const Params& params) {
    out.assign(params.z_face_size(), 0.0);
    const double inv_dz = 1.0 / params.dz();
    for (int k = 1; k < params.nz; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                out[z_face_idx(params, i, j, k)] =
                    (q[idx(params, i, j, k)] - q[idx(params, i, j, k - 1)]) * inv_dz;
            }
        }
    }
}

Field ddz_center_to_w(const Field& q, const Params& params) {
    Field out;
    ddz_center_to_w(q, out, params);
    return out;
}

void ddz_w(const Field& w, Field& out, const Params& params) {
    out.assign(params.z_face_size(), 0.0);
    const double inv_dz = 1.0 / params.dz();
    for (int k = 1; k < params.nz; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                out[z_face_idx(params, i, j, k)] =
                    (w[z_face_idx(params, i, j, k + 1)] - w[z_face_idx(params, i, j, k - 1)]) * (0.5 * inv_dz);
            }
        }
    }
}

Field ddz_w(const Field& w, const Params& params) {
    Field out;
    ddz_w(w, out, params);
    return out;
}

void d2dz2_w(const Field& w, Field& out, const Params& params) {
    out.assign(params.z_face_size(), 0.0);
    const double inv_dz2 = 1.0 / (params.dz() * params.dz());
    for (int k = 1; k < params.nz; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                out[z_face_idx(params, i, j, k)] =
                    (w[z_face_idx(params, i, j, k - 1)] - 2.0 * w[z_face_idx(params, i, j, k)]
                    + w[z_face_idx(params, i, j, k + 1)])
                    * inv_dz2;
            }
        }
    }
}

Field d2dz2_w(const Field& w, const Params& params) {
    Field out;
    d2dz2_w(w, out, params);
    return out;
}

void laplacian_center(const Field& q, Field& out, const Params& params, FftwXY& fft) {
    fft.horizontal_laplacian(q, out, params);
    const double inv_dz2 = 1.0 / (params.dz() * params.dz());
    for (int k = 0; k < params.nz; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                double value = 0.0;
                if (k == 0) {
                    value = (q[idx(params, i, j, 1)] - q[idx(params, i, j, 0)]) * inv_dz2;
                } else if (k == params.nz - 1) {
                    value = (q[idx(params, i, j, params.nz - 2)] - q[idx(params, i, j, params.nz - 1)]) * inv_dz2;
                } else {
                    value = (q[idx(params, i, j, k - 1)] - 2.0 * q[idx(params, i, j, k)] + q[idx(params, i, j, k + 1)])
                        * inv_dz2;
                }
                out[idx(params, i, j, k)] += value;
            }
        }
    }
}

Field laplacian_center(const Field& q, const Params& params, FftwXY& fft) {
    Field out;
    laplacian_center(q, out, params, fft);
    return out;
}

void laplacian_w(const Field& w, Field& out, const Params& params, FftwXY& fft) {
    fft.horizontal_laplacian_planes(w, params.nz + 1, out, params);
    const double inv_dz2 = 1.0 / (params.dz() * params.dz());
    for (int k = 1; k < params.nz; ++k) {
        for (int j = 0; j < params.ny; ++j) {
            for (int i = 0; i < params.nx; ++i) {
                out[z_face_idx(params, i, j, k)] +=
                    (w[z_face_idx(params, i, j, k - 1)] - 2.0 * w[z_face_idx(params, i, j, k)]
                     + w[z_face_idx(params, i, j, k + 1)])
                    * inv_dz2;
            }
        }
    }
}

Field laplacian_w(const Field& w, const Params& params, FftwXY& fft) {
    Field out;
    laplacian_w(w, out, params, fft);
    return out;
}

void divergence(const Field& u, const Field& v, const Field& w, Field& div, const Params& params, FftwXY& fft) {
    Field dudx;
    Field dvdy;
    fft.derivative_x(u, dudx, params);
    fft.derivative_y(v, dvdy, params);
    Field dwdz = ddz_w_to_center(w, params);
    div.resize(params.real_size());
    for (std::size_t n = 0; n < div.size(); ++n) {
        div[n] = dudx[n] + dvdy[n] + dwdz[n];
    }
}

Field divergence(const Field& u, const Field& v, const Field& w, const Params& params, FftwXY& fft) {
    Field div;
    divergence(u, v, w, div, params, fft);
    return div;
}

}  // namespace wireles

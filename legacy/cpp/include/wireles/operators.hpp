#pragma once

#include "wireles/fft.hpp"
#include "wireles/field.hpp"

namespace wireles {

void ddz_center(const Field& q, Field& out, const Params& params);
Field ddz_center(const Field& q, const Params& params);
void d2dz2_center(const Field& q, Field& out, const Params& params);
Field d2dz2_center(const Field& q, const Params& params);
void w_to_center(const Field& w, Field& out, const Params& params);
Field w_to_center(const Field& w, const Params& params);
void center_to_w(const Field& q, Field& out, const Params& params);
Field center_to_w(const Field& q, const Params& params);
void ddz_w_to_center(const Field& w, Field& out, const Params& params);
Field ddz_w_to_center(const Field& w, const Params& params);
void ddz_center_to_w(const Field& q, Field& out, const Params& params);
Field ddz_center_to_w(const Field& q, const Params& params);
void ddz_w(const Field& w, Field& out, const Params& params);
Field ddz_w(const Field& w, const Params& params);
void d2dz2_w(const Field& w, Field& out, const Params& params);
Field d2dz2_w(const Field& w, const Params& params);
void laplacian_center(const Field& q, Field& out, const Params& params, FftwXY& fft);
Field laplacian_center(const Field& q, const Params& params, FftwXY& fft);
void laplacian_w(const Field& w, Field& out, const Params& params, FftwXY& fft);
Field laplacian_w(const Field& w, const Params& params, FftwXY& fft);
void divergence(const Field& u, const Field& v, const Field& w, Field& div, const Params& params, FftwXY& fft);
Field divergence(const Field& u, const Field& v, const Field& w, const Params& params, FftwXY& fft);

}  // namespace wireles

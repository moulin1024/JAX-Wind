#pragma once

#include <string>

#include "wireles/params.hpp"

namespace wireles {

void apply_config_file(Params& params, const std::string& path);

}  // namespace wireles

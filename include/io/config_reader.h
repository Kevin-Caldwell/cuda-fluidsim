#pragma once

#include <iostream>

#include "errors.h"
#include "fsim/sim_params.h"

namespace config_reader
{

ret_t parse_fsim_config(const char *filename, sim_params_t *params);

void print_params(sim_params_t *params);
}  // namespace config_reader

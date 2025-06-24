#pragma once

#include <iostream>

#include "errors.h"
#include "sim_params.h"

namespace config_reader {

ret_t parse_fsim_config(const char* filename, sim_params_t *params);

}

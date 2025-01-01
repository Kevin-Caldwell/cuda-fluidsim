#ifndef CONFIG_READER_H
#define CONFIG_READER_H

#include <iostream>

#include "errors.hpp"
#include "sim_params.hpp"

ret_t parse_fsim_config(const char* filename, sim_params_t *params);

#endif /* CONFIG_READER_H */

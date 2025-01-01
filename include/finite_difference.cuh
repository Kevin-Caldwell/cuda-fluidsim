#ifndef FINITE_DIFFERENCE_H
#define FINITE_DIFFERENCE_H

/**
 * finite_difference.h
 * @author Kevin Caldwell
 * @date 27/09/2024
 * 
 * File contains the CUDA implementation of multiple kernel-level functions for 
 * computing various quantities for CFD simulations using the Finite Difference 
 * Method.
 */

/** Imports */
#include <cuda_runtime.h>
#include <helper_cuda.h>

/** References */
#include "sim_params.hpp"
#include "defs.h"

/**
 * Calculates new pressure value for given grid point, stores in data.pressure
 */
__global__ void thread_pressure_smoothing(SimData* data);

/** Calculates horizontal velocity component after dt, stores in data.u */
__global__ void thread_update_u(SimData* data);

/** Calculates vertical velocity component after dt, stores in data.v*/
__global__ void thread_update_v(SimData* data);

/** Calculate Vorticity and store in data.temp_0 */
__global__ void thread_calculate_vorticity(SimData* data);

#endif /* FINITE_DIFFERENCE_H */

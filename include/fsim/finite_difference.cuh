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
#include "defs.h"
#include "sim_params.h"

__global__ void thread_pressure_smoothing(const float *u,
                                          const float *v,
                                          const float *pr,
                                          float *temp,
                                          const int dim_x,
                                          const int dim_y,
                                          const float dx,
                                          const float dy,
                                          const float dt,
                                          const float density,
                                          const float viscosity);

__global__ void thread_update_u(const float *u,
                                const float *v,
                                const float *pr,
                                float *temp,
                                const int dim_x,
                                const int dim_y,
                                const float dx,
                                const float dy,
                                const float dt,
                                const float offset_vel_x,
                                const float density,
                                const float viscosity);

__global__ void thread_update_v(const float *u,
                                const float *v,
                                const float *pr,
                                float *temp,
                                const int dim_x,
                                const int dim_y,
                                const float dx,
                                const float dy,
                                const float dt,
                                const float offset_vel_y,
                                const float density,
                                const float viscosity);

__global__ void thread_calculate_vorticity(const float *u,
                                           const float *v,
                                           float *temp,
                                           const int dim_x,
                                           const int dim_y,
                                           const float dx,
                                           const float dy);

#endif /* FINITE_DIFFERENCE_H */

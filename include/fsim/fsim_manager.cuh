#pragma once

/** Imports */
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "finite_difference.cuh"

namespace fluidsim
{

void __host__ fsim_smooth_pressure(SimData* d_s,
                                   const SimParams& params,
                                   dim3 blocks);

void __host__ fsim_update_u(SimData* d_s, const SimParams& params, dim3 blocks);

void __host__ fsim_update_v(SimData* d_s, const SimParams& params, dim3 blocks);

void __host__ fsim_vorticity_map(SimData* d_s,
                                 const SimParams& params,
                                 dim3 blocks);

void fsim_save_scalar_field(float* h_field,
                            int* size_x,
                            int* size_y,
                            const char* filename);

void fsim_csv_append(float* h_field, int* size_x, int* size_y, FILE* fp);

void fsim_display_scalar_field(float* d_field,
                               float* h_buffer,
                               int size,
                               int s_x,
                               int s_y);

}  // namespace fluidsim

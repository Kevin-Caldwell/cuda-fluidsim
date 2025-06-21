/** Imports */
#include <stdio.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
// #include <thread>
#include <random>
#include <unistd.h>
// #include <limits.h>

#include "backups.h"
#include "config_reader.h"
#include "fsim_manager.cuh"
// #include "include/ppm_handler.h"
#include "array_utils.h"
#include "device_utils.h"
#include "img_thread.h"
#include "log.h"
#include "ppm_handler.h"

char wbuf[40];

#define DEBUG

void velocity_field_init(int elem_count, int buffer_size, int dim_x, int dim_y,
                         float *h_buffer, SimParams *params, float *d_u,
                         float *d_v, float *d_pressure) {
  for (int i = 0; i < elem_count; i++) {
    h_buffer[i] = params->offset_vel_x;
  }

  h_buffer[elem(dim_x / 2, dim_y / 2, dim_x)] = 0.5;

  // for (int i = dim_y / 3; i < 2 * dim_y / 3; i++)
  // {
  //     h_buffer[elem(dim_x / 2, i, dim_x)] = 0.1;
  // }

  // fill_random(h_buffer, elem_count);

  cudaMemcpy(d_u, h_buffer, buffer_size, cudaMemcpyHostToDevice);

  for (int i = 0; i < elem_count; i++) {
    h_buffer[i] = params->offset_vel_y;
  }

  cudaMemcpy(d_v, h_buffer, buffer_size, cudaMemcpyHostToDevice);

  for (int i = 0; i < elem_count; i++) {
    h_buffer[i] = 0.0;
  }

  cudaMemcpy(d_pressure, h_buffer, buffer_size, cudaMemcpyHostToDevice);
}

// char *h_chbuffer = NULL;

// void render_scalar_field(ppm_handler img_creator, int index,
//                          const char *annotation, int elem_count, float
//                          *d_data, float *h_buffer) {

//   char filename_buf[100];

//   cudaMemcpy(h_buffer, d_data, elem_count * sizeof(float),
//              cudaMemcpyDeviceToHost);
//   float max = utils::array::max_arr(h_buffer, elem_count);
//   float min = utils::array::min_arr(h_buffer, elem_count);
//   // printf("(%03f, %03f)", max, min);
//   utils::array::fl_to_char_arr(h_buffer, h_chbuffer, elem_count,
//                                255 / (max - min), min);
//   snprintf(filename_buf, 100, "temp/%s_%03d.ppm", annotation, index);
//   img_creator.write_ppm(filename_buf, h_chbuffer);
// }

void print_field(float *arr, int size_x, int size_y) {
  for (int i = 0; i < size_y; i++) {
    printf("[%f ", arr[size_x * i]);
    for (int j = 1; j < size_x; j++) {
      printf(", %f", arr[size_x * i + j]);
    }
    printf("]\n");
  }
  printf("\n");
}

int main(void) {
  if (gpu::check_cuda_dev()) {
    return -1;
  }

  long int time;
  utils::log::tick();

  SimParams params;
  SimParams *d_params = NULL;
  SimData h_data;
  SimData *d_data = NULL;

  float *d_pressure = NULL, *d_u = NULL, *d_v = NULL;
  float *d_temp0 = NULL, *d_temp1 = NULL;
  float *h_buffer = NULL;

  utils::config_reader::parse_fsim_config("fsim.config", &params);

  const int dim_x = params.dim_x, dim_y = params.dim_y;
  const float tf = params.tf;

  params.dx = params.size_x / (float)params.dim_x;
  params.dy = params.size_y / (float)params.dim_y;

  cudaMalloc(&d_params, sizeof(SimParams));
  cudaMemcpy(d_params, &params, sizeof(SimParams), cudaMemcpyHostToDevice);

  int buffer_size = params.dim_x * params.dim_y * sizeof(float);
  int elem_count = params.dim_x * params.dim_y;

  // Allocate Memory for Velocity and Pressure Fields
  cudaMalloc((void **)&d_pressure, buffer_size);
  cudaMalloc((void **)&d_u, buffer_size);
  cudaMalloc((void **)&d_v, buffer_size);
  // Allocate Fields for holding Intermediate Values
  cudaMalloc((void **)&d_temp0, buffer_size);
  cudaMalloc((void **)&d_temp1, buffer_size);

  // Allocate Temporary RAM Buffer to hold data to set/use Device Memory
  h_buffer = (float *)malloc(buffer_size);
  // h_chbuffer = (char *)malloc(elem_count * sizeof(char));

  dim3 blocks = {(unsigned)params.dim_x, (unsigned)params.dim_y};

  // Initialize SimData Struct for the host
  h_data = {.params = d_params,
            .u = d_u,
            .v = d_v,
            .pressure = d_pressure,
            .temp_0 = d_temp0,
            .temp_1 = d_temp1};

  cudaMalloc((void **)&d_data, sizeof(SimData));
  cudaMemcpy(d_data, &h_data, sizeof(SimData), cudaMemcpyHostToDevice);

  printf("Successfully Allocated Memory...\n");

  utils::backup::setup_backup();

  velocity_field_init(elem_count, buffer_size, dim_x, dim_y, h_buffer, &params,
                      d_u, d_v, d_pressure);

  system("mkdir temp");

  bool save_arena = true;
  unsigned int iterations = 0;

  ppm_handler img_creater = ppm_handler(dim_x, dim_y, 0);

  time = utils::log::tock();
  printf("Setup Time: %ld ms ", time / 1000000);

  for (; params.t < tf; params.t += params.dt) {
    write(STDOUT_FILENO, "\r", 1);
    printf("%08f/%08f\t", params.t, tf, utils::log::tock() / 1000000);
    // save_arena = iterations % 1 == 0;

    // Copy U, V, P from GPU to Memory and save to csv for Debugging
    if (save_arena) {
      // printf("Render Time: %d\t", utils::log::tock() / 1000000);
      // render_scalar_field(img_creater, iterations, "u", elem_count, d_u,
      //                     h_buffer);
      // printf("%d\t", utils::log::tock() / 1000000);
      // render_scalar_field(img_creater, iterations, "v", elem_count, d_v,
      //                     h_buffer);
      // printf("%d\t", utils::log::tock() / 1000000);
      // render_scalar_field(img_creater, iterations, "pressure", elem_count,
      //                     d_pressure, h_buffer);
      // printf("%d\t", utils::log::tock() / 1000000);
    }

    // Iteratively Smooth Pressure
    for (int i = 0; i < params.smoothing; i++) {
      fluidsim::fsim_smooth_pressure(d_data, blocks);
      cudaMemcpy(d_pressure, d_temp0, buffer_size, cudaMemcpyDeviceToDevice);
    }
    printf("%d\t", utils::log::tock() / 1000000);

    // Update Velocity Vector Field
    fluidsim::fsim_update_u(d_data, blocks);
    fluidsim::fsim_update_v(d_data, blocks);
    printf("%d\t", utils::log::tock() / 1000000);

    cudaMemcpy(d_u, d_temp0, buffer_size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_v, d_temp1, buffer_size, cudaMemcpyDeviceToDevice);
    printf("%d\t", utils::log::tock() / 1000000);

    fflush(0);
    iterations++;
  }

  printf("Freeing all Allocated Memory.\n");

  // Free all allocated Buffers
  free(h_buffer);
  cudaFree(d_pressure);
  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_temp0);
  cudaFree(d_temp1);

  system("./../../../vidgen.sh");
  system("rm -r temp");

  utils::backup::exit_backup(&params);
  return 0;
}
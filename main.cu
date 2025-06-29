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
#include "devptr.h"
#include "img_thread.h"
#include "log.h"
#include "ppm_handler.h"

// #define LOG_STEP_TIME

char wbuf[40];

#define DEBUG

void velocity_field_init(int elem_count, int buffer_size, int dim_x, int dim_y,
                         float *h_buffer, SimParams *params, float *d_u,
                         float *d_v, float *d_pressure) {
  for (int i = 0; i < elem_count; i++) {
    h_buffer[i] = params->offset_vel_x;
  }

  h_buffer[elem(dim_x / 2, dim_y / 2, dim_x)] = 0.5;

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

int main(void) {
  if (gpu::check_cuda_dev()) {
    perror("GPU not found");
    exit(CUDA_NOT_FOUND);
  }

  long int time;
  utils::log::tick();

  SimParams params;
  // SimParams *d_params = NULL;
  // SimData h_data;
  SimData *d_data = NULL;

  float *d_pressure = NULL, *d_u = NULL, *d_v = NULL;
  float *d_temp0 = NULL, *d_temp1 = NULL;
  float *h_buffer = NULL;

  config_reader::parse_fsim_config("fsim.config", &params);

  // const int dim_x = params.dim_x, dim_y = params.dim_y;
  const float tf = params.tf;

  params.dx = params.size_x / (float)params.dim_x;
  params.dy = params.size_y / (float)params.dim_y;

  int buffer_size = params.dim_x * params.dim_y * sizeof(float);
  int elem_count = params.dim_x * params.dim_y;

  // Allocate Memory for Velocity and Pressure Fields
  ptr<float> dev_pressure(elem_count, DEV_PTR);
  ptr<float> dev_u(elem_count, DEV_PTR);
  ptr<float> dev_v(elem_count, DEV_PTR);

  d_pressure = dev_pressure.get();
  d_u = dev_u.get();
  d_v = dev_u.get();

  // Allocate Fields for holding Intermediate Values
  ptr<float> dev_temp0(elem_count, DEV_PTR);
  d_temp0 = dev_temp0.get();

  ptr<float> dev_temp1(elem_count, DEV_PTR);
  d_temp1 = dev_temp1.get();

  // Allocate Temporary RAM Buffer to hold data to set/use Device Memory
  ptr<float> host_buf(elem_count, HOST_PTR);
  h_buffer = host_buf.get();

  dim3 blocks = {(unsigned)params.dim_x, (unsigned)params.dim_y};

  printf("Successfully Allocated Memory...\n");

  backup::setup_backup();

  velocity_field_init(elem_count, buffer_size, params.dim_x, params.dim_y,
                      h_buffer, &params, d_u, d_v, d_pressure);

  system("mkdir temp");

  bool save_arena = true;
  unsigned int iterations = 0;

  ppm_handler img_creater = ppm_handler(params.dim_x, params.dim_y, 0);

  time = utils::log::tock();
  printf("Setup Time: %ld ms\n", time / 1000000);

  for (params.t = 0; params.t < tf; params.t += params.dt) {

    write(STDOUT_FILENO, "\r", 1);
    printf("%08f/%08f\t%f", params.t, tf, (float)utils::log::tock() / 1000000);

    // Copy U, V, P from GPU to Memory and save to csv for Debugging
    if (save_arena) {
#ifdef LOG_STEP_TIME
      printf("\t Render Time: %d\t", utils::log::tock() / 1000000);
#endif

      render_scalar_field(img_creater, iterations, "u", elem_count, d_u,
                          h_buffer);

#ifdef LOG_STEP_TIME
      printf("%d\t", utils::log::tock() / 1000000);
#endif

      render_scalar_field(img_creater, iterations, "v", elem_count, d_v,
                          h_buffer);

#ifdef LOG_STEP_TIME
      printf("%d\t", utils::log::tock() / 1000000);
#endif

      render_scalar_field(img_creater, iterations, "pressure", elem_count,
                          d_pressure, h_buffer);

#ifdef LOG_STEP_TIME
      printf("%d\t", utils::log::tock() / 1000000);
#endif
    }

    // Iteratively Smooth Pressure
    for (int i = 0; i < params.smoothing; i++) {
      fluidsim::fsim_smooth_pressure(d_data, blocks);
      cudaMemcpy(d_pressure, d_temp0, buffer_size, cudaMemcpyDeviceToDevice);
    }

#ifdef LOG_STEP_TIME
    printf("%d\t", utils::log::tock() / 1000000);
#endif

    // Update Velocity Vector Field
    fluidsim::fsim_update_u(d_data, blocks);
    fluidsim::fsim_update_v(d_data, blocks);

#ifdef LOG_STEP_TIME
    printf("%d\t", utils::log::tock() / 1000000);
#endif
    printf("IFEfe fe JIFEJ1\n");
    dev_u.copy_data(&dev_temp0);
    printf("IFEfe fe JIFEJ2\n");
    dev_v.copy_data(&dev_temp1);
    printf("IFEfe fe JIFEJ3\n");

#ifdef LOG_STEP_TIME
    printf("%d\t", utils::log::tock() / 1000000);
#endif

    fflush(0);
    iterations++;
  }

  printf("\nFreeing all Allocated Memory.\n");

  // Free all allocated Buffers
  free(h_buffer);
  cudaFree(d_pressure);
  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_temp0);
  cudaFree(d_temp1);

  system("./../../../vidgen.sh");
  system("rm -r temp");

  backup::exit_backup(&params);
  return 0;
}
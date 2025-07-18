#include "img_thread.h"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#include "array_utils.h"

char *h_chbuffer = NULL;

void render_scalar_field(ppm_handler img_creator,
                         const int index,
                         const char *annotation,
                         const int elem_count,
                         const float *d_data,
                         float *h_buffer)
{
  h_chbuffer = (char *)malloc(elem_count * sizeof(char));

  char filename_buf[100];

  cudaMemcpy(h_buffer, d_data, elem_count * sizeof(float),
             cudaMemcpyDeviceToHost);
  float max = utils::array::max_arr(h_buffer, elem_count);
  float min = utils::array::min_arr(h_buffer, elem_count);

  // printf("(%03f, %03f)", max, min);
  utils::array::fl_to_char_arr(h_buffer, h_chbuffer, elem_count,
                               255 / (max - min), min);
  snprintf(filename_buf, 100, "temp/%s_%03d.ppm", annotation, index);

  img_creator.write_ppm(filename_buf, h_chbuffer);

  free(h_chbuffer);
}

ImageWriteThread::ImageWriteThread(uint32_t width,
                                   uint32_t height,
                                   float multiplier,
                                   int index,
                                   const char *annotation,
                                   int element_count,
                                   float *dev_data,
                                   float *host_buffer,
                                   pthread_cond_t *cond_image_write,
                                   pthread_cond_t *enable,
                                   bool *thread_running)
    : image_handler_(width, height, multiplier),
      annotation_(annotation),
      index_(index),
      write_image_(cond_image_write),
      dev_data_(dev_data),
      element_count_(element_count),
      host_buffer_(host_buffer),
      enable_(enable),
      thread_running_(thread_running)
{
}

void ImageWriteThread::ImageWriteThreadRun() {
  while (this->thread_running_) {
    pthread_cond_wait(this->write_image_, NULL);
  }
}

void run_ppm_thread() {}

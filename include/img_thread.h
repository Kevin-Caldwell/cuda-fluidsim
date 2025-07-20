#pragma once

#include <inttypes.h>
#include <io/ppm_handler.h>
#include <pthread.h>

void render_scalar_field(ppm_handler img_creator,
                         const int index,
                         const char *annotation,
                         const int elem_count,
                         const float *d_data,
                         float *h_buffer);

class ImageWriteThread {
private:
  ppm_handler image_handler_;
  int index_;
  const char *annotation_;

  int element_count_;
  float *dev_data_;
  float *host_buffer_;

  bool *thread_running_;

  pthread_cond_t *enable_;
  pthread_cond_t *write_image_;

public:
 ImageWriteThread(uint32_t width,
                  uint32_t height,
                  int index,
                  const char *annotation,
                  int element_count,
                  float *dev_data,
                  float *host_buffer,
                  pthread_cond_t *cond_image_write,
                  pthread_cond_t *enable,
                  bool *thread_running);

 void ImageWriteThreadRun();
};

void run_ppm_thread();

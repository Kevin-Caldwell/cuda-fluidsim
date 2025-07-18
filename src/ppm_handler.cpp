#include "ppm_handler.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char whitespace = ' ';

ppm_handler::ppm_handler(unsigned int width, unsigned int height, float mul)
    : height_(height), width_(width), multiplier_(mul)
{
}

ppm_handler::~ppm_handler() {}

ret_t ppm_handler::write_ppm(const char *filename, const char *data) {
  FILE *img_file = fopen(filename, "wb");
  if (!img_file) {
    perror("fopen");
    return ERR_FOPEN;
  }

  char header_buf[200];

  snprintf(header_buf,
           200,
           "%s %u %u %u\n",
           magic_,
           width_,
           height_,
           this->maxval);

  fwrite(header_buf, sizeof(char), strnlen(header_buf, 200), img_file);

  int counter = 0;
  header_buf[0] = '\n';
  header_buf[1] = ' ';

  for (int i = 0; i < height_; i++) {
    for (int j = 0; j < width_; j++) {
      int index = (i * width_ + j);
      for (int l = 0; l < 3; l++) {
        fwrite(data + index, sizeof(char), 1, img_file);
      }
    }
  }

  fclose(img_file);
  return RES_OK;
}

#ifndef PPM_HANDLER_H
#define PPM_HANDLER_H

#include "errors.h"

#define MAX_PIXEL_VAL 255

class ppm_handler
{
 private:
  /* data */
  const char *magic_ = "P6";
  unsigned int width_ = 0;
  unsigned int height_ = 0;
  const unsigned maxval = MAX_PIXEL_VAL;
  // float multiplier_;

 public:
  ppm_handler(unsigned int width, unsigned int height);
  ~ppm_handler();

  unsigned int &width() { return width_; }

  unsigned int &height() { return height_; }

  ret_t write_ppm(const char *filename, const char *data);
};

#endif /* PPM_HANDLER_H */

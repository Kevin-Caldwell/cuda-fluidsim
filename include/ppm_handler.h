#ifndef PPM_HANDLER_H
#define PPM_HANDLER_H

#include "errors.h"

#define MAX_PIXEL_VAL 255

class ppm_handler
{
private:
    /* data */
    const char* magic = "P6";
    unsigned int width = 0;
    unsigned int height = 0;
    const unsigned maxval = MAX_PIXEL_VAL;
    float multiplier;

public:
    ppm_handler(unsigned int width, unsigned int height, float multiplier);
    ~ppm_handler();

    ret_t write_ppm(const char* filename, const char* data);
};

#endif /* PPM_HANDLER_H */

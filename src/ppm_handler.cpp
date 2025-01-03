#include "ppm_handler.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char whitespace = ' ';

ppm_handler::ppm_handler(unsigned int width, unsigned int height, float mul)
{
    this->height = height;
    this->width = width;
    this->multiplier = mul;
}

ppm_handler::~ppm_handler()
{
}

ret_t ppm_handler::write_ppm(const char *filename, const char *data)
{
    FILE *img_file = fopen(filename, "wb");
    if (!img_file)
    {
        perror("fopen");
        return ERR_FOPEN;
    }

    char header_buf[200];

    snprintf(header_buf, 200, "%s %u %u %u\n", this->magic, this->width,
             this->height, this->maxval);

    fwrite(header_buf, sizeof(char), strnlen(header_buf, 200), img_file);

    int counter = 0;
    header_buf[0] = '\n';
    header_buf[1] = ' ';

    for (int i = 0; i < this->height; i++)
    {
        for (int k = 0; k < 3; k++)
        {
            fwrite(data + i * this->width, sizeof(char), this->width, img_file);
            // printf("%d\n", i * this->width);
            // for(int j = 0; j < this->width; j++){
            //     int index = (i * this->width + j);
            //     for(int l = 0; l < 3; l++){
            //         printf("%d: %d   ", index, *(data + index));
            //         fwrite(data + index, sizeof(char), 1, img_file);
            //         fwrite(header_buf+1, sizeof(char), 1, img_file);
            //     }
        }
    }
    // }

    fclose(img_file);
    return RES_OK;
}

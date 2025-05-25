#ifndef ARRAY_UTILS_H
#define ARRAY_UTILS_H

float max_arr(float* arr, int len);

float min_arr(float* arr, int len);

void fl_to_char_arr(
    float *fl_arr, char *ch_arr, int len,
    float scaling, float offset
);

#endif /* ARRAY_UTILS_H */

#include "array_utils.h"

float utils::array::max_arr(float *arr, int len) {
  float max = arr[0];

  for (int i = 1; i < len; i++) {
    if (arr[i] > max) {
      max = arr[i];
    }
  }

  return max;
}

float utils::array::min_arr(float *arr, int len) {
  float min = arr[0];

  for (int i = 1; i < len; i++) {
    if (arr[i] < min) {
      min = arr[i];
    }
  }

  return min;
}

void utils::array::fl_to_char_arr(float *fl_arr, char *ch_arr, int len,
                                  float scaling, float offset) {
  for (int i = 0; i < len; i++) {
    ch_arr[i] = (char)((float)(fl_arr[i] * scaling + offset));
  }
}

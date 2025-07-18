#include "utils/array_utils.h"

#include "stdio.h"

namespace utils {

float array::max_arr(float *arr, int len) {
  float max = arr[0];

  for (int i = 1; i < len; i++) {
    if (arr[i] > max) {
      max = arr[i];
    }
  }

  return max;
}

float array::min_arr(float *arr, int len) {
  float min = arr[0];

  for (int i = 1; i < len; i++) {
    if (arr[i] < min) {
      min = arr[i];
    }
  }

  return min;
}

void array::fl_to_char_arr(float *fl_arr, char *ch_arr, int len, float scaling,
                           float offset) {
  for (int i = 0; i < len; i++) {
    ch_arr[i] = (char)((float)(fl_arr[i] * scaling + offset));
  }
}

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

} // namespace utils

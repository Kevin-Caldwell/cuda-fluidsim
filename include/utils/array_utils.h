#pragma once

namespace utils
{
namespace array
{

float max_arr(float* arr, int len);

float min_arr(float* arr, int len);

void fl_to_char_arr(float* fl_arr,
                    char* ch_arr,
                    int len,
                    float scaling,
                    float offset);
}  // namespace array
}  // namespace utils

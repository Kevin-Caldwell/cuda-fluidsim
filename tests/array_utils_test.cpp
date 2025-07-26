#include <gtest/gtest.h>
#include "utils/array_utils.h"

TEST(ArrayMaxTest, Single) {
  float arr[] = {10};
  float res = utils::array::max_arr(arr, 1);
  EXPECT_EQ(res, 10);
}

TEST(ArrayMaxTest, Multiple) {
  float arr[] = {1, 2, -5, 1.3, 0.34};
  float res = utils::array::max_arr(arr, 5);
  EXPECT_EQ(res, 2);
}

TEST(ArrayMinTest, Single) {
  float arr[] = {10};
  float res = utils::array::min_arr(arr, 1);
  EXPECT_EQ(res, 10);
}

TEST(ArrayMinTest, Multiple) {
  float arr[] = {1, 2, -5, 1.3, 0.34};
  float res = utils::array::min_arr(arr, 5);
  EXPECT_EQ(res, -5);
}

// TEST(Array)

#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "errors.h"

#define PTR_DEBUG 1

enum ptr_location { NO_INIT = 0, HOST_PTR, DEV_PTR };

typedef ptr_location loc_t;

template <typename T> struct ptr {
private:
  T *ptr_;    // Casted Pointer
  int count_; // Size in bytes
  int sz_;

  /* Location of Pointer in system */
  loc_t loc_;

  inline void update_sz() { sz_ = count_ * sizeof(T); }
  inline ret_t alloc(void **ptr, int sz, loc_t loc);

public:
  ptr(int count, loc_t loc);
  ~ptr();

  ret_t resize(int new_count, bool preserve);
  ret_t dealloc();
};

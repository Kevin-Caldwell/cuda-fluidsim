#pragma once

#include <cstring>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "errors.h"

#define PTR_DEBUG 1

enum ptr_location { NO_INIT = 0, HOST_PTR, DEV_PTR };

typedef ptr_location loc_t;

template <typename T> struct ptr {
private:
  T *ptr_; // Casted Pointer
  int count_;
  int sz_; // Size in bytes

  /* Location of Pointer in system */
  loc_t loc_;

  inline void update_sz() { sz_ = count_ * sizeof(T); }
  inline ret_t alloc_(void **ptr, int sz, loc_t loc) {
    switch (loc) {
    case NO_INIT:
      ERROR_EQ(true, true, "Pointer Location Invalid", ERR_PTR_NOINIT);
      break;

    case HOST_PTR: {
      *ptr = (T *)malloc(sz);
      ERROR_EQ(ptr, nullptr, "Host Alloc Failed.", HOST_ALLOC_FAIL);
      break;
    }

    case DEV_PTR: {
      printf("%p, %d\n", ptr, sz);
      ERROR_NEQ(cudaMalloc(ptr, sz), cudaSuccess, "Device Alloc Failed.",
                CUDA_ALLOC_FAIL);
      break;
    }

    default:
      ERROR_EQ(true, true, "Invalid location.", ERR_PTR_NOINIT);
      break;
    }

    return RES_OK;
  }
  inline ret_t free_(void *ptr, loc_t loc) {
    switch (loc) {
    case HOST_PTR:
      free(ptr);
      break;

    case DEV_PTR:
      ERROR_NEQ(cudaFree(ptr), cudaSuccess, "CUDA Dealloc failed.",
                CUDA_DEALLOC_FAIL);
      break;

    default:
      ERROR_EQ(true, true, "", ERR_PTR_NOINIT);
      break;
    }

    // Reset vars for reuse
    loc_ = NO_INIT;
    count_ = 0;
    sz_ = 0;
    ptr_ = NULL;

    return RES_OK;
  }

public:
  ptr(int count, loc_t loc) : count_(count), loc_(loc), sz_(count * sizeof(T)) {
    ERROR_NEQ(alloc_((void **)&ptr_, sz_, loc_), RES_OK, "Alloc Failed", );
  }

  ~ptr() { ERROR_NEQ(free_(ptr_, loc_), RES_OK, "Unable to Free", ); }

  T *get() const { return ptr_; }

  ret_t resize(int new_count, bool preserve) {
    ERROR_EQ(loc_, NO_INIT, "Ptr location not initialized", ERR_PTR_NOINIT);

    void *temp;

    if (preserve && sz_ != 0) {

      ERROR_NEQ(alloc_(&temp, sz_, loc_), RES_OK, "Unable to resize",
                ERR_GENERIC);
    }

    if (sz_ != 0) {
      ret_t res = free_(ptr_, loc_);
      ERROR_NEQ(res, RES_OK, "Unable to deallocate for Resize",
                CUDA_DEALLOC_FAIL);
    }

    count_ = new_count;
    update_sz();
  }

  ret_t copy_data(ptr<T> *src) {

    ERROR_EQ(src->loc_, NO_INIT, "", ERR_PTR_NOINIT);
    printf("FEIJ {%p, %d, %d, %d} to {%p, %d, %d, %d}  \n", ptr_, count_, sz_,
           loc_, src->ptr_, src->count_, src->sz_, src->loc_);

    if (loc_ == NO_INIT) {
      loc_ = HOST_PTR;
      alloc_((void **)&ptr_, sz_, loc_);
    }

    if (loc_ == HOST_PTR && src->loc_ == HOST_PTR) {
      ERROR_NEQ(memcpy(ptr_, src->ptr_, src->sz_), ptr_,
                "copy_data: memcpy failed", MEMCPY_FAIL);
    } else if (loc_ == HOST_PTR && src->loc_ == DEV_PTR) {
      ERROR_NEQ(cudaMemcpy(ptr_, src->ptr_, src->sz_, cudaMemcpyDeviceToHost),
                cudaSuccess, "CUDA memcpy to host failed.", MEMCPY_FAIL);
    } else if (loc_ == DEV_PTR && src->loc_ == HOST_PTR) {
      ERROR_NEQ(cudaMemcpy(ptr_, src->ptr_, src->sz_, cudaMemcpyHostToDevice),
                cudaSuccess, "CUDA memcpy to device failed.", MEMCPY_FAIL);
    } else if (loc_ == DEV_PTR && src->loc_ == DEV_PTR) {
      // cudaError_t res = cudaMemcpy(ptr_, src->ptr_, src->sz_,
      // cudaMemcpyDeviceToDevice); printf("ERROR: %d\n", res);
      ERROR_NEQ(cudaMemcpy(ptr_, src->ptr_, src->sz_, cudaMemcpyDeviceToDevice),
                cudaSuccess, "CUDA memcpy to device failed.", MEMCPY_FAIL);
    }

    return RES_OK;
  }
};

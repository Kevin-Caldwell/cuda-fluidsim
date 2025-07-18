#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include <cstring>
#include <iostream>
#include <string>

#include "errors.h"

#define PTR_DEBUG 1

enum ptr_location { NO_INIT = 0, HOST_PTR, DEV_PTR };

std::string location_to_str(ptr_location loc)
{
  switch (loc) {
    case NO_INIT:
      return "NOINIT";
      break;
    case HOST_PTR:
      return "HOST";

    case DEV_PTR:
      return "DEVICE";

    default:
      return "INVALID";
  }
}

typedef ptr_location loc_t;

template <typename T>
struct ptr {
 private:
  T *ptr_;  // Casted Pointer
  int count_;
  int sz_;  // Size in bytes

  /* Location of Pointer in system */
  loc_t loc_;

  inline void update_sz() { sz_ = count_ * sizeof(T); }
  inline ret_t alloc_(void **p, int sz, loc_t loc)
  {
    switch (loc) {
      case NO_INIT:
        ERROR_EQ(true, true, "Pointer Location Invalid", ERR_PTR_NOINIT);
        break;

      case HOST_PTR: {
        *p = (T *)malloc(sz);
        ERROR_EQ(p, nullptr, "Host Alloc Failed.", HOST_ALLOC_FAIL);
        break;
      }

      case DEV_PTR: {
        ERROR_NEQ(cudaMalloc(p, sz),
                  cudaSuccess,
                  "Device Alloc Failed.",
                  CUDA_ALLOC_FAIL);
        break;
      }

      default:
        ERROR_EQ(true, true, "Invalid location.", ERR_PTR_NOINIT);
        break;
    }

    return RES_OK;
  }
  inline ret_t free_(void *ptr, loc_t loc)
  {
#ifdef VERBOSE
    printf("FREEING %p from %d", ptr, loc);
#endif

    switch (loc) {
      case HOST_PTR:
        free(ptr);
        break;

      case DEV_PTR:
        ERROR_NEQ(cudaFree(ptr),
                  cudaSuccess,
                  "CUDA Dealloc failed.",
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
  ptr(int count, loc_t loc) : count_(count), loc_(loc), sz_(count * sizeof(T))
  {
    ERROR_NEQ(alloc_((void **)&ptr_, sz_, loc_), RES_OK, "Alloc Failed", );

#ifdef VERBOSE
    printf("Allocated: {%p, %d, %d, %s}\n",
           ptr_,
           count_,
           sz_,
           location_to_str(loc_).c_str());
#endif
  }

  ~ptr() { ERROR_NEQ(free_(ptr_, loc_), RES_OK, "Unable to Free", ); }

  T *get() const { return ptr_; }

  int size() const { return sz_; }

  int count() const { return count_; }

  inline std::string loc() const
  {
    switch (loc_) {
      case HOST_PTR:
        return "HOST";

      case DEV_PTR:
        return "DEVICE";

      default:
        return "NOINIT";
    }
  }

  ret_t resize(int new_count, bool preserve)
  {
    ERROR_EQ(loc_, NO_INIT, "Ptr location not initialized", ERR_PTR_NOINIT);

    void *temp;

    if (preserve && sz_ != 0) {
      ERROR_NEQ(alloc_(&temp, sz_, loc_),
                RES_OK,
                "Unable to resize",
                ERR_GENERIC);
    }

    if (sz_ != 0) {
      ret_t res = free_(ptr_, loc_);
      ERROR_NEQ(res,
                RES_OK,
                "Unable to deallocate for Resize",
                CUDA_DEALLOC_FAIL);
    }

    count_ = new_count;
    update_sz();
  }

  ret_t copy_data(ptr<T> *src)
  {
#ifdef VERBOSE
    std::cout << "Copying from " << *src << " to " << *this << std::endl;
#endif
    ERROR_EQ(src->loc_, NO_INIT, "", ERR_PTR_NOINIT);

    if (loc_ == NO_INIT) {
      loc_ = HOST_PTR;
      alloc_((void **)&ptr_, sz_, loc_);
    }

    if (loc_ == HOST_PTR && src->loc_ == HOST_PTR) {
      ERROR_NEQ(memcpy(ptr_, src->ptr_, src->sz_),
                ptr_,
                "copy_data: memcpy failed",
                MEMCPY_FAIL);
    } else if (loc_ == HOST_PTR && src->loc_ == DEV_PTR) {
      ERROR_NEQ(cudaMemcpy(ptr_, src->ptr_, src->sz_, cudaMemcpyDeviceToHost),
                cudaSuccess,
                "CUDA memcpy to host failed.",
                MEMCPY_FAIL);
    } else if (loc_ == DEV_PTR && src->loc_ == HOST_PTR) {
      ERROR_NEQ(cudaMemcpy(ptr_, src->ptr_, src->sz_, cudaMemcpyHostToDevice),
                cudaSuccess,
                "CUDA memcpy to device failed.",
                MEMCPY_FAIL);
    } else if (loc_ == DEV_PTR && src->loc_ == DEV_PTR) {
      // cudaError_t res =
      // cudaMemcpy(ptr_, src->ptr_, src->sz_, cudaMemcpyDeviceToDevice);
      // printf("ERROR: %d\n", res);
      ERROR_NEQ(cudaMemcpy((void *)ptr_,
                           (void *)src->ptr_,
                           src->sz_,
                           cudaMemcpyDeviceToDevice),
                cudaSuccess,
                "CUDA memcpy to device failed.",
                MEMCPY_FAIL);
    }

    return RES_OK;
  }
};

template <class T>
std::ostream &operator<<(std::ostream &os, const ptr<T> &p)
{
  os << "{" << p.get() << ", " << p.count() << ", " << p.size() << ", "
     << p.loc().c_str() << "}";
  return os;
}

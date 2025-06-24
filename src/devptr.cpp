#include "devptr.h"

#include <cstring>

#include "errors.h"

#define ERROR_NEQ(expr, expected, msg, ret_val)                                \
  if ((expr) != expected) {                                                    \
    perror(msg);                                                               \
    return ret_val;                                                            \
  }

#define ERROR_EQ(expr, expected, msg, ret_val)                                 \
  if ((expr) == expected) {                                                    \
    perror(msg);                                                               \
    return ret_val;                                                            \
  }

template <typename T>
ptr<T>::ptr(int count, loc_t loc)
    : count_(count), loc_(loc), sz_(count * sizeof(T)) {

  ERROR_NEQ(alloc((void **)ptr_, sz_, loc_), RES_OK, "Alloc Failed", );
}

template <typename T> ptr<T>::~ptr() {
  ERROR_NEQ(dealloc(), RES_OK, "Unable to Free", );
}

template <class T> ret_t ptr<T>::resize(int new_count, bool preserve) {

  ERROR_EQ(loc_, NO_INIT, "Ptr location not initialized", ERR_PTR_NOINIT);

  void *temp;

  if (preserve && sz_ != 0) {

    ERROR_NEQ(alloc(&temp, sz_, loc_), RES_OK, "Unable to resize", ERR_GENERIC);
  }

  if (sz_ != 0) {
    ret_t res = dealloc();
    ERROR_NEQ(res, RES_OK, "Unable to deallocate for Resize",
              CUDA_DEALLOC_FAIL);
  }

  count_ = new_count;
  update_sz();
}

template <class T> inline ret_t ptr<T>::alloc(void **ptr, int sz, loc_t loc) {
  switch (loc_) {
  case NO_INIT:
    ERROR_EQ(true, true, "Pointer Location Invalid", ERR_PTR_NOINIT);
    break;

  case HOST_PTR: {
    ptr_ = (T *)malloc(sz_);
    ERROR_EQ(ptr_, NULL, "Host Alloc Failed.", HOST_ALLOC_FAIL);
    break;
  }

  case DEV_PTR: {
    ERROR_NEQ(cudaMalloc(&ptr_, sz_), cudaSuccess, "Device Alloc Failed.",
              CUDA_ALLOC_FAIL);
    break;
  }

  default:
    ERROR_EQ(true, true, "Invalid location.", ERR_PTR_NOINIT);
    break;
  }

  return RES_OK;
}

template <class T> ret_t ptr<T>::dealloc() {

  switch (loc_) {
  case HOST_PTR:
    free(ptr_);
    break;

  case DEV_PTR:
    ERROR_NEQ(cudaFree(ptr_), cudaSuccess, "CUDA Dealloc failed.",
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

/**
 * If destination has not been initialized,
 * default to Host memory
 */
template <class T> ret_t copy_data(ptr<T> *dest, ptr<T> *src) {

  ERROR_EQ(src->loc_, NO_INIT, "", ERR_PTR_NOINIT);

  if (src->loc_ == NO_INIT) {
    return ERR_PTR_NOINIT;
  }

  if (dest->loc_ == HOST_PTR && src->loc_ == HOST_PTR) {
    ERROR_NEQ(memcpy(dest->ptr_, src->ptr_, src->sz_), dest->ptr_,
              "copy_data: memcpy failed", MEMCPY_FAIL);
  }

  if (dest->loc_ == NO_INIT) {
    dest->loc_ = HOST_PTR;
    dest->resize(src->count_, false);
  }

  return RES_OK;
}

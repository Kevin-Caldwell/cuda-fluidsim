#pragma once

typedef enum ret_t {
  RES_OK = 0,
  ERR_GENERIC,
  ERR_FOPEN,
  ERR_PTR_NOINIT,
  CUDA_ALLOC_FAIL,
  CUDA_DEALLOC_FAIL,
  HOST_ALLOC_FAIL,
  MEMCPY_FAIL,
} ret_t;

typedef enum exit_code_t {
  EXIT_OK = 0,
  CUDA_NOT_FOUND,
  BACKUP_INIT_FAILED,

} exit_code_t;
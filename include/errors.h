#pragma once

#include <iostream>

#include <execinfo.h> /* backtrace, backtrace_symbols_fd */
#include <unistd.h>   /* STDOUT_FILENO */

void print_stacktrace(void);

#define ERROR_NEQ(expr, expected, msg, ret_val)                                \
  if ((expr) != expected) {                                                    \
    perror(msg);                                                               \
    printf("File %s, Line %d\n", __FILE__, __LINE__);                          \
    print_stacktrace();                                                        \
    exit(-1);                                                                  \
    return ret_val;                                                            \
  }

#define ERROR_EQ(expr, expected, msg, ret_val)                                 \
  if ((expr) == expected) {                                                    \
    perror(msg);                                                               \
    printf("File %s, Line %d\n", __FILE__, __LINE__);                          \
    print_stacktrace();                                                        \
    exit(-1);                                                                  \
    return ret_val;                                                            \
  }

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
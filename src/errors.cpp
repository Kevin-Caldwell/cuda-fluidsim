#include "errors.h"

void print_stacktrace(void) {
  size_t size;
  enum Constexpr { MAX_SIZE = 1024 };
  void *array[MAX_SIZE];
  size = backtrace(array, MAX_SIZE);
  backtrace_symbols_fd(array, size, STDOUT_FILENO);
}
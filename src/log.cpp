#include "log.h"

#include <cstdio>
#include <ctime>

#define LOG_CLOCK CLOCK_MONOTONIC

#define NANO (1000000000)

long int last_recorded_time = 0;

long int get_clocktime() {
  struct timespec ts;

  if (clock_gettime(LOG_CLOCK, &ts) == -1) {
    perror("Clock Gettime Failed");
  }

  return (long int)ts.tv_sec * NANO + ts.tv_nsec;
}

long int utils::log::tick() {
  last_recorded_time = get_clocktime();
  return last_recorded_time;
}

long int utils::log::tock() {
  long int cur_time = get_clocktime();
  long int elapsed = cur_time - last_recorded_time;
  last_recorded_time = cur_time;

  return elapsed;
}

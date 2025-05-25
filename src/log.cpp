#include "log.hpp"

#include <ctime>
#include <cstdio>

long int last_recorded_time;

#define LOG_CLOCK CLOCK_REALTIME

long int get_clocktime()
{
    struct timespec ts;

    if (clock_gettime(LOG_CLOCK, &ts) == -1) {
        perror("Clock Gettime Failed");
    }

    return (long int) ts.tv_nsec;
}

long int utils::log::tick() 
{
    last_recorded_time = get_clocktime();
    return last_recorded_time;

}

long int utils::log::tock() 
{
    long int cur_time = get_clocktime();
    long int elapsed = cur_time - last_recorded_time;
    last_recorded_time = cur_time;

    return elapsed;
}

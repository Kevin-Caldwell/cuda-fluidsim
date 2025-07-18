#include <stdio.h>

#include "utils/device_utils.h"

int gpu::check_cuda_dev()
{
    int nDevices;

    cudaGetDeviceCount(&nDevices);
    printf("Devices Found: %d\n", nDevices);

    if (!nDevices)
    {
        return 1;
    }

    for (int i = 0; i < nDevices; i++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    }

    return 0;
}
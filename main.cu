/** Imports */
#include <stdio.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <thread>
#include <unistd.h>
#include <random>

#include "config_reader.hpp"
#include "fsim_manager.cuh"
#include "data_server.hpp"
#include "backups.hpp"

char wbuf[40];

#define DEBUG

int main(void)
{

    int nDevices;

    cudaGetDeviceCount(&nDevices);
    printf("Devices Found: %d\n", nDevices);
    
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

    SimParams params;
    SimParams *d_params = NULL;
    SimData h_data;
    SimData *d_data = NULL;

    float *d_pressure = NULL, *d_u = NULL, *d_v = NULL;
    float *d_temp0 = NULL, *d_temp1 = NULL;
    float *h_buffer = NULL;

    parse_fsim_config("fsim.config", &params);

    const int dim_x = params.dim_x, dim_y = params.dim_y;
    const float tf = params.tf;

    params.dx = params.size_x / (float)params.dim_x;
    params.dy = params.size_y / (float)params.dim_y;

    cudaMalloc(&d_params, sizeof(SimParams));
    cudaMemcpy(d_params, &params, sizeof(SimParams), cudaMemcpyHostToDevice);

    int buffer_size = params.dim_x * params.dim_y * sizeof(float);
    int elem_count = params.dim_x * params.dim_y;

    // Allocate Memory for Velocity and Pressure Fields
    cudaMalloc((void **)&d_pressure, buffer_size);
    cudaMalloc((void **)&d_u, buffer_size);
    cudaMalloc((void **)&d_v, buffer_size);
    // Allocate Fields for holding Intermediate Values
    cudaMalloc((void **)&d_temp0, buffer_size);
    cudaMalloc((void **)&d_temp1, buffer_size);

    // Allocate Temporary RAM Buffer to hold data to set/use Device Memory
    h_buffer = (float *)malloc(buffer_size);

    dim3 blocks = {(unsigned)params.dim_x, (unsigned)params.dim_y};

    // Initialize SimData Struct for the host
    h_data = {
        .params = d_params,
        .u = d_u,
        .v = d_v,
        .pressure = d_pressure,
        .temp_0 = d_temp0,
        .temp_1 = d_temp1};

    cudaMalloc((void **)&d_data, sizeof(SimData));
    cudaMemcpy(d_data, &h_data, sizeof(SimData), cudaMemcpyHostToDevice);

    printf("Successfully Allocated Memory...\n");

    setup_backup();

    // Initialize U Field
    for (int i = 0; i < elem_count; i++)
    {
        h_buffer[i] = params.offset_vel_x;
    }

    for (int i = dim_y / 3; i < 2 * dim_y / 3; i++)
    {
        h_buffer[elem(dim_x / 2, i, dim_x)] = 1;
    }

    // fill_random(h_buffer, elem_count);
    // h_buffer[elem(dim_x / 2, dim_y / 2, dim_x)] = 0.01;
    cudaMemcpy(d_u, h_buffer, buffer_size, cudaMemcpyHostToDevice);

    for (int i = 0; i < elem_count; i++)
    {
        h_buffer[i] = params.offset_vel_y;
    }

    // fill_random(h_buffer, elem_count);
    // h_buffer[elem(dim_x / 2, 2 * dim_y / 3, dim_x)] = -0.01;
    cudaMemcpy(d_v, h_buffer, buffer_size, cudaMemcpyHostToDevice);

    for (int i = 0; i < elem_count; i++)
    {
        h_buffer[i] = 0.0;
    }

    // h_buffer[elem(5, 5, 10)] = 10.0;

    cudaMemcpy(d_pressure, h_buffer, buffer_size, cudaMemcpyHostToDevice);

    system("rm -r temp");
    system("mkdir temp");
    system("./../../../capture-fields.sh &");
    data_server *ds_u = new data_server("u", dim_x, dim_y);
    data_server *ds_v = new data_server("v", dim_x, dim_y);
    data_server *ds_w = new data_server("w", dim_x, dim_y);
    data_server *ds_p = new data_server("pressure", dim_x, dim_y);

    bool save_arena = true;
    unsigned int iterations = 0;

    while (params.t < tf)
    {
        save_arena = iterations % 4 == 0;

        // Copy U, V, P from GPU to Memory and save to csv for Debugging
        if (save_arena)
        {
            cudaMemcpy(h_buffer, d_u, buffer_size, cudaMemcpyDeviceToHost);
            ds_u->send_frame(h_buffer);
            // fsim_csv_append(h_buffer, &params.dim_x, &params.dim_y, u_csv_fp);
            cudaMemcpy(h_buffer, d_v, buffer_size, cudaMemcpyDeviceToHost);
            ds_v->send_frame(h_buffer);
            // fsim_csv_append(h_buffer, &params.dim_x, &params.dim_y, v_csv_fp);
            cudaMemcpy(h_buffer, d_pressure, buffer_size, cudaMemcpyDeviceToHost);
            ds_p->send_frame(h_buffer);

            fsim_vorticity_map(d_data, blocks);
            cudaMemcpy(h_buffer, d_temp0, buffer_size, cudaMemcpyDeviceToHost);
            ds_w->send_frame(h_buffer);
        }
        // fsim_csv_append(h_buffer, &params.dim_x, &params.dim_y, pressure_fp);

        // Iteratively Smooth Pressure
        for (int i = 0; i < params.smoothing; i++)
        {
            fsim_smooth_pressure(d_data, blocks);
            cudaMemcpy(d_pressure, d_temp0, buffer_size, cudaMemcpyDeviceToDevice);
        }

        // Update Velocity Vector Field
        fsim_update_u(d_data, blocks);
        fsim_update_v(d_data, blocks);

        cudaMemcpy(d_u, d_temp0, buffer_size, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_v, d_temp1, buffer_size, cudaMemcpyDeviceToDevice);

        params.t += params.dt;
        printf("\r%f/%f", params.t, tf);
        fflush(0);
        iterations++;
    }

    printf("Freeing all Allocated Memory.\n");

    // Free all allocated Buffers
    free(h_buffer);
    cudaFree(d_pressure);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_temp0);
    cudaFree(d_temp1);

    delete ds_u;
    delete ds_v;
    delete ds_w;
    delete ds_p;

    system("./../../../vidgen.sh");
    system("rm -r temp");

    exit_backup(&params);
    return 0;
}
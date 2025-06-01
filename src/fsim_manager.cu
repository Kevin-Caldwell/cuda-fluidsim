#include "fsim_manager.cuh"

#include <stdio.h>


void __host__ fluidsim::fsim_smooth_pressure(SimData *d_s, dim3 blocks)
{
    thread_pressure_smoothing<<<blocks, 1>>>(d_s);
}

void __host__ fluidsim::fsim_update_u(SimData *d_s, dim3 blocks)
{
    thread_update_u<<<blocks, 1>>>(d_s);
}

void __host__ fluidsim::fsim_update_v(SimData *d_s, dim3 blocks)
{
    thread_update_v<<<blocks, 1>>>(d_s);
}

void __host__ fluidsim::fsim_vorticity_map(SimData *d_s, dim3 blocks)
{
    thread_calculate_vorticity<<<blocks, 1>>>(d_s);
}

void fluidsim::fsim_save_scalar_field(float* h_field, int* size_x, int* size_y, const char* filename){
    FILE* fp = fopen(filename, "ab");
    if(fp){
        fwrite(size_x, sizeof(int), 1, fp);
        fwrite(size_y, sizeof(int), 1, fp);
        fwrite(h_field, sizeof(float),(*size_x) * (*size_y), fp);
        fclose(fp);
    } else {
        perror("Cannot Open File.");
    }
}

void fluidsim::fsim_csv_append(float* h_field, int* size_x, int* size_y, FILE* fp){
    if(fp){
        char str_buf[100];
        snprintf(str_buf, 100, "%d, %d\n", *size_x, *size_y);
        fwrite(str_buf, sizeof(char), strlen(str_buf), fp);

        for(int i = 0; i < *size_y; i++){

            snprintf(str_buf, 100, "%f", h_field[(*size_x) * i]);
            fwrite(str_buf, sizeof(char), strlen(str_buf), fp);

            for(int j = 1; j < *size_x; j++){

                snprintf(str_buf, 100, ",%f", h_field[i * (*size_x) + j]);
                fwrite(str_buf, sizeof(char), strlen(str_buf), fp);
            }

            snprintf(str_buf, 100, "\n");
            fwrite(str_buf, sizeof(char), strlen(str_buf), fp);
        }
    }
}

void fluidsim::fsim_display_scalar_field(float* d_field, float* h_buffer, int size, int s_x, int s_y){
    cudaMemcpy(h_buffer, d_field, size, cudaMemcpyDeviceToHost);
    for(int i = 0; i < s_x; i++){
        for(int j = 0; j < s_y; j++){
            printf("%.3f ", h_buffer[j * s_x + i]);
        }
        printf("\n");
    }
    printf("\n");
}
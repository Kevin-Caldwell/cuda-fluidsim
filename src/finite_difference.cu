#include "finite_difference.cuh"
#include <stdio.h>

#define CLAMP(x, l, h) ((x) <= (l) ? (l) : ((x) > (h) ? (h)))

#define MULTIPLIER 1

__global__ void thread_pressure_smoothing(SimData *data)
{
    int i = blockIdx.x;
    int j = blockIdx.y;

    SimParams *p = data->params;
    float *u = data->u;
    float *v = data->v;
    float *pr = data->pressure;

    int eij = elem(i, j, p->dim_x);
    int eidj = elem(i > 1 ? (i - 1) : p->dim_x - 1, j, p->dim_x);
    int eiuj = elem(i < p->dim_x ? (i + 1) : 0, j, p->dim_x);
    int eijd = elem(i, j > 1 ? (j - 1) : p->dim_y - 1, p->dim_x);
    int eiju = elem(i, j < p->dim_y ? (j + 1) : 0, p->dim_x);

    double coef = (double) (p->dx * p->dx * p->dy * p->dy * p->density) 
        / (2 * (p->dx + p->dy));
    // printf("dx = %f, dy = %f, density = %f\n", p->dx, p->dy, p->density);

    double t0 = ((u[eiuj] - u[eidj]) / (2 * p->dx) * MULTIPLIER);
    t0 *= t0;

    double t1 = 2 * ((u[eiju] - u[eijd]) / (2 * p->dy)) * 
        ((v[eiuj] - v[eijd]) / (2 * p->dx) * MULTIPLIER);

    double t2 = (v[eiju] - v[eidj]) / (2 * p->dy) * MULTIPLIER;
    t2 *= t0;

    double t3 = -0.5 * ((pr[eiuj] + pr[eidj]) / (p->dx * p->dx) + 
        (pr[eiju] + pr[eijd]) / (p->dy * p->dy)) * MULTIPLIER;

    data->temp_0[eij] = (float) (coef * (t0 + t1 + t2 + t3)) / MULTIPLIER;
    // printf("%f\n", data->temp_0[eij]);
}

__global__ void thread_update_u(SimData *data)
{
    int i = blockIdx.x;
    int j = blockIdx.y;


    SimParams *p = data->params;
    float *u = data->u;
    float *v = data->v;
    float *pr = data->pressure;

    int eij = elem(i, j, p->dim_x);
    int eidj = elem(i > 1 ? (i - 1) : p->dim_x - 1, j, p->dim_x);
    int eiuj = elem(i < p->dim_x ? (i + 1) : 0, j, p->dim_x);
    int eijd = elem(i, j > 1 ? (j - 1) : p->dim_y - 1, p->dim_x);
    int eiju = elem(i, j < p->dim_y ? (j + 1) : 0, p->dim_x);

    // if (i == 5 && j == 5)
    // {
        // data->temp_0[elem(p->dim_x/2, p->dim_y/2, p->dim_x)] = 10.0;
        // return;
    // }
    // int eij = elem(i, j, blockDim.x);
    // int eidj = elem(i - 1, j, blockDim.x);
    // int eiuj = elem(i + 1, j, blockDim.x);
    // int eijd = elem(i, j - 1, blockDim.x);
    // int eiju = elem(i, j + 1, blockDim.x);

    double kp = -(pr[eiuj] - pr[eidj]) / (2 * p->dx * p->density) * MULTIPLIER;
    double kx2 = (u[eiuj] - 2 * u[eij] + u[eidj]) / (p->dx * p->dx) * MULTIPLIER;
    double ky2 = (u[eiju] - 2 * u[eij] + u[eijd]) / (p->dy * p->dy) * MULTIPLIER;
    double kx = -u[eij] * (u[eij] - u[eidj]) / p->dx * MULTIPLIER;
    double ky = -v[eij] * (u[eij] - u[eijd]) / p->dy * MULTIPLIER;

    double da = p->dt * (kp + p->viscosity * (kx2 + ky2) + kx + ky) / MULTIPLIER + p->offset_vel_x;
#ifdef FLOATING_POINT_ERROR_SUPPRESSOR
    da = min(1e-5, da);
    da = max(-1e-5, da);
#endif

    if (da != da)
    {
        data->temp_0[eij] = u[eij] * 1.1;
    }
    else
    {
        data->temp_0[eij] = u[eij] + da;
    }
}

__global__ void thread_update_v(SimData *data)
{
    int i = blockIdx.x;
    int j = blockIdx.y;

    SimParams *p = data->params;
    float *u = data->u;
    float *v = data->v;
    float *pr = data->pressure;

    int eij = elem(i, j, p->dim_x);
    int eidj = elem(i > 1 ? (i - 1) : p->dim_x - 1, j, p->dim_x);
    int eiuj = elem(i < p->dim_x ? (i + 1) : 0, j, p->dim_x);
    int eijd = elem(i, j > 1 ? (j - 1) : p->dim_y - 1, p->dim_x);
    int eiju = elem(i, j < p->dim_y ? (j + 1) : 0, p->dim_x);

    // int eij = elem(i, j, blockDim.x);
    // int eidj = elem(clamp(i - 1), j, blockDim.x);
    // int eiuj = elem(i + 1, j, blockDim.x);
    // int eijd = elem(i, j - 1, blockDim.x);
    // int eiju = elem(i, j + 1, blockDim.x);

    double kp = -(pr[eiuj] - pr[eidj]) / (2 * p->dx * p->density) * MULTIPLIER;
    double kx2 = (v[eiuj] - 2 * v[eij] + v[eidj]) / (p->dx * p->dx) * MULTIPLIER;
    double ky2 = (v[eiju] - 2 * v[eij] + v[eijd]) / (p->dy * p->dy) * MULTIPLIER;
    double kx = -u[eij] * (v[eij] - v[eidj]) / p->dx * MULTIPLIER;
    double ky = -v[eij] * (v[eij] - v[eijd]) / p->dy * MULTIPLIER;
    double ka = p->dt * (kp + p->viscosity * (kx2 + ky2) + kx + ky) / MULTIPLIER + p->offset_vel_y;

#ifdef FLOATING_POINT_ERROR_SUPPRESSOR
    ka = min(1e-5, ka);
    ka = max(-1e-5, ka);
#endif

    if (ka != ka)
    {
        data->temp_1[eij] = v[eij] * 10;
    }
    else
    {
        data->temp_1[eij] = v[eij] + ka;
    }
}

__global__ void thread_calculate_vorticity(SimData *data)
{
    int i = blockIdx.x;
    int j = blockIdx.y;

    SimParams *p = data->params;
    float *u = data->u;
    float *v = data->v;

    // int eij = elem(i, j, blockDim.x);
    // int eidj = elem(i - 1, j, blockDim.x);
    // int eijd = elem(i, j - 1, blockDim.x);

    int eij = elem(i, j, p->dim_x);
    int eidj = elem(i > 1 ? (i - 1) : p->dim_x - 1, j, p->dim_x);
    int eijd = elem(i, j > 1 ? (j - 1) : p->dim_y - 1, p->dim_x);

    data->temp_0[eij] = (v[eij] - v[eidj]) / p->dx - (u[eij] - u[eijd]) / p->dy;
}
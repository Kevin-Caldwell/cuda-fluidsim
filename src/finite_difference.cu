#include <stdio.h>

#include "fsim/finite_difference.cuh"

#define CLAMP(x, l, h) ((x) <= (l) ? (l) : ((x) > (h) ? (h)))

#define MULTIPLIER 1.f

__global__ void thread_pressure_smoothing(const float *u,
                                          const float *v,
                                          const float *pr,
                                          float *temp,
                                          const int dim_x,
                                          const int dim_y,
                                          const float dx,
                                          const float dy,
                                          const float dt,
                                          const float density,
                                          const float viscosity)
{
  const int i = blockIdx.x;
  const int j = blockIdx.y;

  const int eij = elem(i, j, dim_x);
  const int eidj = elem(i > 1 ? (i - 1) : dim_x - 1, j, dim_x);
  const int eiuj = elem(i < dim_x ? (i + 1) : 0, j, dim_x);
  const int eijd = elem(i, j > 1 ? (j - 1) : dim_y - 1, dim_x);
  const int eiju = elem(i, j < dim_y ? (j + 1) : 0, dim_x);

  const double coef = (double)(dx * dx * dy * dy * density) / (2 * (dx + dy));

  double t0 = ((u[eiuj] - u[eidj]) / (2 * dx) * MULTIPLIER);
  t0 *= t0;

  const double t1 = 2 * ((u[eiju] - u[eijd]) / (2 * dy)) *
                    ((v[eiuj] - v[eijd]) / (2 * dx) * MULTIPLIER);

  const double t2 = (v[eiju] - v[eidj]) / (2 * dy) * MULTIPLIER * t0;

  const double t3 =
      -0.5 *
      ((pr[eiuj] + pr[eidj]) / (dx * dx) + (pr[eiju] + pr[eijd]) / (dy * dy)) *
      MULTIPLIER;

  temp[eij] = (float)(coef * (t0 + t1 + t2 + t3)) / MULTIPLIER;
}

__global__ void thread_update_u(const float *u,
                                const float *v,
                                const float *pr,
                                float *temp,
                                const int dim_x,
                                const int dim_y,
                                const float dx,
                                const float dy,
                                const float dt,
                                const float offset_vel_x,
                                const float density,
                                const float viscosity)
{
  const int i = blockIdx.x;
  const int j = blockIdx.y;

  const int eij = elem(i, j, dim_x);
  const int eidj = elem(i > 1 ? (i - 1) : dim_x - 1, j, dim_x);
  const int eiuj = elem(i < dim_x ? (i + 1) : 0, j, dim_x);
  const int eijd = elem(i, j > 1 ? (j - 1) : dim_y - 1, dim_x);
  const int eiju = elem(i, j < dim_y ? (j + 1) : 0, dim_x);

  //   printf("running %d, %d\n", i, j);
  const double kp = -(pr[eiuj] - pr[eidj]) / (2 * dx * density) * MULTIPLIER;
  const double kx2 = (u[eiuj] - 2 * u[eij] + u[eidj]) / (dx * dx) * MULTIPLIER;
  const double ky2 = (u[eiju] - 2 * u[eij] + u[eijd]) / (dy * dy) * MULTIPLIER;
  const double kx = -u[eij] * (u[eij] - u[eidj]) / dx * MULTIPLIER;
  const double ky = -v[eij] * (u[eij] - u[eijd]) / dy * MULTIPLIER;

  double da =
      dt * (kp + viscosity * (kx2 + ky2) + kx + ky) / MULTIPLIER + offset_vel_x;
#ifdef FLOATING_POINT_ERROR_SUPPRESSOR
    da = min(1e-5, da);
    da = max(-1e-5, da);
#endif

    // printf("temp: %lf, da: %lf\n", u[eij], da);
    temp[eij] = u[eij] + da;
}

__global__ void thread_update_v(const float *u,
                                const float *v,
                                const float *pr,
                                float *temp,
                                const int dim_x,
                                const int dim_y,
                                const float dx,
                                const float dy,
                                const float dt,
                                const float offset_vel_y,
                                const float density,
                                const float viscosity)
{
    const int i = blockIdx.x;
    const int j = blockIdx.y;

    const int eij = elem(i, j, dim_x);
    const int eidj = elem(i > 1 ? (i - 1) : dim_x - 1, j, dim_x);
    const int eiuj = elem(i < dim_x ? (i + 1) : 0, j, dim_x);
    const int eijd = elem(i, j > 1 ? (j - 1) : dim_y - 1, dim_x);
    const int eiju = elem(i, j < dim_y ? (j + 1) : 0, dim_x);

    const double kp = -(pr[eiuj] - pr[eidj]) / (2 * dx * density) * MULTIPLIER;
    const double kx2 =
        (v[eiuj] - 2 * v[eij] + v[eidj]) / (dx * dx) * MULTIPLIER;
    const double ky2 =
        (v[eiju] - 2 * v[eij] + v[eijd]) / (dy * dy) * MULTIPLIER;
    const double kx = -u[eij] * (v[eij] - v[eidj]) / dx * MULTIPLIER;
    const double ky = -v[eij] * (v[eij] - v[eijd]) / dy * MULTIPLIER;
    double ka = dt * (kp + viscosity * (kx2 + ky2) + kx + ky) / MULTIPLIER +
                offset_vel_y;

#ifdef FLOATING_POINT_ERROR_SUPPRESSOR
    ka = min(1e-5, ka);
    ka = max(-1e-5, ka);
#endif

    temp[eij] = v[eij] + ka;
}

__global__ void thread_calculate_vorticity(const float *u,
                                           const float *v,
                                           float *temp,
                                           const int dim_x,
                                           const int dim_y,
                                           const float dx,
                                           const float dy)
{
    const int i = blockIdx.x;
    const int j = blockIdx.y;

    const int eij = elem(i, j, dim_x);
    const int eidj = elem(i > 1 ? (i - 1) : dim_x - 1, j, dim_x);
    const int eijd = elem(i, j > 1 ? (j - 1) : dim_y - 1, dim_x);

    temp[eij] = (v[eij] - v[eidj]) / dx - (u[eij] - u[eijd]) / dy;
}

#ifndef SIM_PARAMS_H
#define SIM_PARAMS_H

struct SimParams {
  /** Simulation Parameters */
  int dim_x = 0;            /** Grid X-dimension */
  int dim_y = 0;            /** Grid Y-dimension */

  float bound_left = 0.0;   /** X-coordinate of Grid(0, 0) */
  float bound_high = 0.0;   /** Y-coordinate of Grid(0, 0) */

  float size_x = 0.0;       /** Cartesian Length of Rectangle */
  float size_y = 0.0;       /** Cartesian Width of Rectangle */

  float dx = 0.0;           /** Distance of  Horizontally adjacent points */
  float dy = 0.0;           /** Distance of Vertically adjacent points  */
  float dt = 0.0;           /** Time Quantization */

  int smoothing = 1;        /** Smoothing Iterator input */

  float offset_vel_x = 0.0; /** Forced horizontal velocity component */
  float offset_vel_y = 0.0; /** Forced vertical velocity component */

  /** Fluid Parameters */
  float density;
  float viscosity;

  float t = 0.0; /** Time variable */
  float tf = 0.0;
};

typedef struct SimParams SimParams;
typedef struct SimParams sim_params_t;

struct SimData {
  SimParams params; /** Store Relevant parameters */

  /** Variables */
  float* u;        /** Stores horizontally aligned velocity components */
  float* v;        /** Stores vertically aligned velocity components */
  float* pressure; /** Stores scalar Pressure Values */

  /** Temporary Arrays */
  float* temp_0; /** Storage Bank used only by DEVICE */
  float* temp_1; /** Storage Bank used only by DEVICE */
};

typedef struct SimData SimData;

void InitializeAir(SimParams* params,
                   int dim_x,
                   int dim_y,
                   float width,
                   float height,
                   float dt,
                   float offset_x,
                   float offset_y);

void write_parameters_to_file(const char* filename, SimParams* params);

#endif /* SIM_PARAMS_H */

#include "sim_params.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

void InitializeAir(
    SimParams *params, 
    int dim_x, 
    int dim_y, 
    float width, 
    float height, 
    float dt,
    float offset_x,
    float offset_y
){
    params->dim_x = dim_x;
    params->dim_y = dim_y;
    params->bound_high = 0.0;
    params->bound_left = 0.0;
    params->size_x = width;
    params->size_y = height;
    params->dt = dt;
    params->offset_vel_x = offset_x;
    params->offset_vel_y = offset_y;
    params->t = 0.0;
    params->density = 1.293;
    params->viscosity = 0;
    // params->viscosity = 1.48e-5;
}

void write_parameters_to_file(const char* filename, SimParams* params){

    std::ofstream outfile(filename);

    if(!outfile.is_open()){
        std::cout << "Error Opening File\n";
        return;
    }

    outfile << "DIM_X:" << params->dim_x << std::endl;
    outfile << "DIM_Y:" << params->dim_y << std::endl;
    outfile << "WIDTH:" << params->size_x << std::endl;
    outfile << "HEIGHT:" << params->size_y << std::endl;
    outfile << "DT:" << params->dt << std::endl;
    outfile << "TF:" << params->tf << std::endl;
    outfile << "U:" << params->offset_vel_x << std::endl;
    outfile << "V:" << params->offset_vel_y << std::endl;
    outfile << "SMOOTH:" << params->smoothing << std::endl;
    outfile << "DENSITY:" << params->density << std::endl;
    outfile << "VISC:" << params->viscosity << std::endl;

    outfile.close();
    std::cout << "Parameters written to " << filename << std::endl;
}
#include "io/config_reader.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

ret_t config_reader::parse_fsim_config(const char *filename,
                                       sim_params_t *params) {

  std::filesystem::path current_path = std::filesystem::current_path();
  std::cout << "Current working directory: " << current_path << std::endl;

  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Error opening file.\n";
    return ERR_FOPEN;
  }

  std::string line;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    std::string key, value;
    std::getline(iss, key, ':');
    std::getline(iss, value);

    if (key == "DIM_X") {
      params->dim_x = std::stoi(value);
    } else if (key == "DIM_Y") {
      params->dim_y = std::stoi(value);
    } else if (key == "WIDTH") {
      params->size_x = std::stoi(value);
    } else if (key == "HEIGHT") {
      params->size_y = std::stoi(value);
    } else if (key == "DT") {
      params->dt = std::stod(value);
    } else if (key == "TF") {
      params->tf = std::stod(value);
    } else if (key == "U") {
      params->offset_vel_x = std::stod(value);
    } else if (key == "V") {
      params->offset_vel_y = std::stod(value);
    } else if (key == "SMOOTH") {
      params->smoothing = std::stoi(value);
    } else if (key == "DENSITY") {
      params->density = std::stod(value);
    } else if (key == "VISC") {
      params->viscosity = std::stod(value);
    }
  }

  file.close();

  return RES_OK;
}

void config_reader::print_params(sim_params_t *params) {
  printf("_____________________\n");
  printf("SIMULATION PARAMETERS\n");
  printf("_____________________\n");
  printf("%-10s %10d\n", "DIMX :", params->dim_x);
  printf("%-10s %10d\n", "DIMY :", params->dim_y);
  printf("%-10s %10f\n", "WIDTH :", params->size_x);
  printf("%-10s %10f\n", "HEIGHT :", params->size_y);
  printf("%-10s %10f\n", "DT :", params->dt);
  printf("%-10s %10f\n", "TF :", params->tf);
  printf("%-10s %10f\n", "U :", params->offset_vel_x);
  printf("%-10s %10f\n", "V :", params->offset_vel_y);
  printf("%-10s %10d\n", "SMOOTH :", params->smoothing);
  printf("%-10s %10f\n", "DENSITY :", params->density);
  printf("%-10s %10f\n", "VISC :", params->viscosity);
  printf("_____________________\n");
}

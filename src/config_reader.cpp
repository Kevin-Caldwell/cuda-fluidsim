#include "config_reader.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

ret_t config_reader::parse_fsim_config(const char *filename,
                                       sim_params_t *params) {

  std::filesystem::path current_path = std::filesystem::current_path();
  std::cout << "Current working directory: " << current_path << std::endl;

  std::ifstream file(
      filename); // Replace "config.txt" with your actual file name
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
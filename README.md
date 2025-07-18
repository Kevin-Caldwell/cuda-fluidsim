# CUDA Fluid Sim

**Author**: Kevin Caldwell

This project implements a 2D fluid simulation hardware accelerated for
CUDA devices. This is a part of a larger project analyzing the effects
of wind on Truck-Drone delivery within cities.

## Prerequisites

The following must be present before building the project.

- GCC
- CMake
- NVIDIA Graphics Drivers (Requires an NVIDIA GPU)
- CUDA Toolkit

## Installation

The project uses the CMake build system. 
After cloning the repo, run the command in the `cuda-fluidsim` repo:

```bash
mkdir build
cd build
cmake ..
make -j
cd ../
```

This should generate two executables, `cuda_fluidsim` and `ppm_handler`.
Ensure the Current working directory is `cuda-fluidsim` before running
the executable.

Run the fluid simulation using the command:

```bash
./build/cuda_fluidsim
```

#!/bin/python3

import argparse
import subprocess

BUILD_FOLDER = "./build/"

FSIM_NAME = "cuda_fluidsim"
PPM_HANDLER_NAME = "ppm_handler"
TEST_EXE_NAME = "fsim_test"

FSIM_PATH = BUILD_FOLDER + FSIM_NAME
PPM_HANDLER_PATH = BUILD_FOLDER + PPM_HANDLER_NAME
TEST_EXE_PATH = BUILD_FOLDER + TEST_EXE_NAME


if __name__ == "__main__":
    subprocess.run([FSIM_PATH])
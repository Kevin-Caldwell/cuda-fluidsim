#!/bin/python3

import argparse
import sys
import os
import subprocess

BUILD_FOLDER = "build"

FSIM = "cuda_fluidsim"
PPM_HANDLER= "ppm_handler"
TEST_EXE = "fsim_test"

def main():
    os.chdir(BUILD_FOLDER)
    subprocess.run(["cmake", '..'])
    subprocess.run(['make', '-j'])
    os.chdir("../")
    subprocess.run([f"./{BUILD_FOLDER}/{TEST_EXE}"])
    subprocess.run([f"./{BUILD_FOLDER}/{FSIM}"])

if __name__ == "__main__":
    main()

import struct

import sys

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

file_pointer = 0
frame_count = 0

def csv_scalar_field() -> list[list]:
    global frame_count
    line = file_pointer.readline()

    dims = line.split(',')
    for i in range(len(dims)):
        dims[i] = int(dims[i])

    scalar_values = []

    for i in range(dims[0]):
        t = file_pointer.readline()
        scalar_values.append([float(x.strip()) for x in t.strip().split(',')])
    
    # print(scalar_values)
    frame_count -= 1
    return scalar_values



import os
import subprocess
import progressbar
if __name__ == '__main__':

    # read_scalar_field('data/test.txt')
    file_pointer = open(f'data/{sys.argv[1]}.csv', 'r')
    frame_count = int(file_pointer.readline().strip())

    print(frame_count)

    for i in progressbar.progressbar(range(frame_count)):
        a = csv_scalar_field()
        # plt.imshow(a, vmin=0.0, vmax=100)
        plt.imshow(a)
        plt.savefig("temp"+f"/{sys.argv[1]}file%02d.png" % i)
    
    subprocess.call(['ffmpeg', '-framerate', '30', '-i', f'temp/{sys.argv[1]}file%02d.png', f'data/{sys.argv[2]}.mp4', '-y'])

    file_pointer.close()
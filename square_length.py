import numpy as np
import math
import sys

def get_length(positions):
    N = int(round(math.sqrt(len(positions))))
    if N*N != len(positions):
        raise RuntimeError("Cannot calculate square length with a non-square # of particles")

    lengths_sum = 0
    for i in range(N):
        col = positions[i * N: (i + 1) * N]
        lengths_sum += (col[-1][1] - col[0][1]) / (N - 1)

    return lengths_sum / N

if __name__ == '__main__':
    particles = np.genfromtxt("POSITIONS", dtype='i,f,f,i,f,i', names=True, skip_header=3)
    positions = sorted(zip(particles['X'], particles['Y']))
    print(get_length(positions))

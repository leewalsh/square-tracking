#!/usr/bin/env python
import sys
import numpy as np
from scipy.spatial import KDTree, Voronoi

import math

def get_angle(p, q):
    # angle of (q - p) from the horizontal
    r = (q[1] - p[1], q[0] - p[0])
    return math.atan2(r[1], r[0])

def exp(C):
    # C complex
    return math.exp(C.real) * (math.cos(C.imag) + 1j * math.sin(C.imag))

def poly_area(corners):
    # calculate area of polygon
    area = 0.0
    n = len(corners)
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    return abs(area) / 2.0

M = 4 # number of neighbors

if len(sys.argv) < 2:
    print("Please specify a filename.")
    sys.exit(0)

fname = sys.argv[1]
data = np.loadtxt(fname + "_POSITIONS.txt", dtype="i,f,f,i,f,i")
frames = [[(row[1], row[2]) for row in data if row[0] == i] for i in
          range(data[-1][0] + 1)]
psi_data = []
frame_densities = []

for frame in frames:
    vor = Voronoi(frame)
    areas = []
    for region in vor.regions:
        if -1 in region: # infinite Voronoi cell
            continue
        areas.append(poly_area([vor.vertices[i] for i in region]))
    areas = np.asarray(areas)
    densities = 1. / areas[areas > 0.]
    frame_densities.append(densities)

    tree = KDTree(frame)
    psi_frame = []
    for i, p in enumerate(frame):
        query_ret = tree.query([p], k=M+1)

        # remove p from the neighbors list by slicing
        neighbors = [tree.data[x] for x in query_ret[1][0]][1:]
        # if p is an edge or corner, remove extra neighbors
        min_dist = min((n[0]-p[0])**2 + (n[1]-p[1])**2 for n in neighbors)
        thresh = min_dist * math.sqrt(2) * .95 # slightly less than a diagonal
        neighbors = [n for n in neighbors if (n[0]-p[0])**2 + (n[1]-p[1])**2 < thresh]
        N = len(neighbors)
        psi = sum(exp(M * get_angle(p, n) * 1j) for n in neighbors) / N
        psi_frame.append(abs(psi))

    psi_data.append(psi_frame)

max_density = max([max(densities) for densities in frame_densities])
frame_densities = [densities / max_density for densities in frame_densities]
np.savez(fname + "_DATA.npz", psi=psi_data, densities=frame_densities)

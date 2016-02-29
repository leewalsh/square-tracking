#!/usr/bin/env python
from orientation import find_corner
import sys
import numpy as np

'''
Given an input prefix, looks at the associated NumPy files
for positions and corners and correlates the two.
'''

if len(sys.argv) < 2:
    print("Please specify a filename")
    sys.exit(0)

fname = sys.argv[1]
data = np.load(fname + "_TRACKS.npz")["data"]
corner_data = np.load(fname + "_CORNER_TRACKS.npz")["data"]

first_frame = data[data['f']==0]
first_corner_frame = corner_data[corner_data['f']==0]
corner_positions = [np.asarray((row[1], row[2])) for row in first_corner_frame]

corner_map = {}
for row in first_frame:
    pos = np.asarray((row[1], row[2]))
    corners, theta, dist = find_corner(pos, corner_positions, n=2, do_average=False)
    for corner in corners:
        for i, corner_pos in enumerate(corner_positions):
            if all(corner_pos == corner):
                corner_index = i
                break
        corner_map.setdefault(row[3], []).append(first_corner_frame[i][3])

out = fname + "_CORNERS.npz"
np.savez(out, corner_map=corner_map)
    

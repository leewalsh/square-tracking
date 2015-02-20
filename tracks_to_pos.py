#!/usr/bin/env python
import sys
import numpy as np

if len(sys.argv) < 2:
    print("Please specify a filename.")
    sys.exit(0)

fname = sys.argv[1]
data = np.load(fname + "_TRACKS.npz")["data"]
data = [tuple(row)[:-1] for row in data]
np.savetxt(fname + "_POSITIONS.txt", data, delimiter='     ',
           fmt=['%6d', '%7.3f', '%7.3f', '%4d', '%1.3f', '%5d'])

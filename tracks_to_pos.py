#!/usr/bin/env python
import sys
import numpy as np

if len(sys.argv) < 2:
    print("Please specify a filename.")
    sys.exit(0)

fname = sys.argv[1]
data = np.load(fname + "_TRACKS.npz")["data"]

# cut out still frames at beginning
nframes = data[-1][0] + 1
original = {}
for row in data[data['f']==0]:
    original[row[3]] = (row[1], row[2])

for frame in range(nframes):
    disp = 0
    for row in data[data['f']==frame]:
        ID = row[3]
        disp += (row[1]-original[ID][0])**2 + (row[2]-original[ID][1])**2
    if disp > 0.04 * len(original): # avg 0.04 squared displacement
        break

data = [(row[0]-frame+1,)+tuple(row)[1:-1] for row in data if \
        row[0] >= frame - 1]

with open(fname + "_POSITIONS.txt", "w") as f:
    f.write('# Frame    X           Y             Label  Eccen        Area\n')
    np.savetxt(f, data, delimiter='     ',
               fmt=['%6d', '%7.3f', '%7.3f', '%4d', '%1.3f', '%5d'])

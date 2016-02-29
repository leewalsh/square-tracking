#!/usr/bin/env python
import sys
import numpy as np

'''
This script inputs a tracking file output by tracks.py and turns it
into a text file of particle positions at each frame. given "filename",
it assumes "filename_TRACKS.npz" exists

Example usage:
tracks_to_pos.py 12x12_random
'''

if len(sys.argv) < 2:
    print("Please specify a filename.")
    sys.exit(0)

fname = sys.argv[1]
data = np.load(fname + "_TRACKS.npz")["data"]

# cut out still frames at beginning
CUT_STILL = False
frame = 1

if CUT_STILL:
    nframes = data[-1][0] + 1
    original = {}   # dict of initial position of each particle (x, y)
    for row in data[data['f']==0]:
        # row[3] is the 'lab' field, showing track ID as per new tracks.py
        # row[1] and row[2] are 'x' and 'y'
        original[row[3]] = (row[1], row[2])

    for frame in range(nframes):
        disp = 0
        for row in data[data['f']==frame]:
            ID = row[3] # the label field, showing track ID not particle ID?
            disp += (row[1]-original[ID][0])**2 + (row[2]-original[ID][1])**2
        if disp > 0.04 * len(original): # avg 0.04 squared displacement
            break # here frame=first movement

# row[0] means row['f']
first = frame - 1
data = [(row[0]-first,) + tuple(row)[1:-1]
        for row in data if row[0] >= first]

with open(fname + "_POSITIONS.txt", "w") as f:
    f.write('# Frame    X           Y             Label  Eccen        Area\n')
    np.savetxt(f, data, delimiter='     ',
               fmt=['%6d', '%7.3f', '%7.3f', '%4d', '%1.3f', '%5d'])

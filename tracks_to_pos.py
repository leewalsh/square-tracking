#!/usr/bin/env python
import sys
import numpy as np

import helpy

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
data = helpy.load_data(fname, 't')

# cut out still frames at beginning
CUT_STILL = False

if CUT_STILL:
    nf = 100
    tsets = helpy.load_tracksets(data, min_length=nf, run_fill_gaps='interp',
                                 run_remove_dupes=True, run_track_orient=True)
    disp_thresh = 0.04*len(tsets)

    disp = np.zeros(nf)
    for t, tset in tsets.iteritems():
        for i in 'xy':
            disp += (tset[i][:nf] - tset[i][0])**2
    first = np.argmax(disp >= disp_thresh)
    data['f'] -= first - 1

np.savez_compressed(fname + '_TRACKS.npz', data=data[data['f'] >= 0])

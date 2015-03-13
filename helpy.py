from itertools import izip
import numpy as np

def splitter(data, frame, method='diff'):
    """ Splits a dataset into subarrays with unique frame value
        `data` : the dataset (will be split along first axis)
        `frame`: the values to group by
    """
    if method.lower().startswith('d'):
        return np.split(data, np.diff(frame).nonzero()[0] + 1)
    elif method.lower().startswith('u'):
        u, i = np.unique(frame, return_index=True)
        return izip(u, np.split(data, i[1:]))


# Pixel-Physical Unit Conversions
# Physical measurements
R_inch = 4.0           # as machined
R_mm   = R_inch * 25.4
S_measured = np.array([4,3,6,7,9,1,9,0,0,4,7,5,3,6,2,6,0,8,8,4,3,4,0,-1,0,1,7,7,5,7])*1e-4 + .309
S_inch = S_measured.mean()
S_mm = S_inch * 25.4
R_S = R_inch / S_inch

# Still (D5000)
R_slr = 2459 / 2
S_slr_m = np.array([3.72, 2.28, 4.34, 3.94, 2.84, 4.23, 4.87, 4.73, 3.77]) + 90 # don't use this, just use R_slr/R_S

# Video (Phantom)
R_vid = 585.5 / 2
S_vid_m = 22 #ish


# What we'll use:
R = R_S         # radius in particle units
S_vid = R_vid/R # particle in video pixels
S_slr = R_slr/R # particle in still pixels


#!/usr/bin/env python
import sys
import numpy as np
from glob import glob
import os

if len(sys.argv) < 2:
    fnames = glob('*_POSITIONS.txt')
    if not os.path.exists('start_adjusted'):
        os.makedirs('start_adjusted')
    out = 'start_adjusted/{0}_POSITIONS.txt'
else:
    fnames = [sys.argv[1]]
    out = '{0}_POSITIONS_NEW.txt'
for fname in fnames:
    print("**** {0} ****".format(fname))
    if fname.endswith("_POSITIONS.txt"):
        fname = fname[:-14]
    data = np.genfromtxt(fname+"_POSITIONS.txt", dtype="i,f,f,i,f,i",
                         names="f,x,y,lab,ecc,area")
    frame0 = data[data['f']==0]
    start_pos = {}
    for line in frame0:
        start_pos[line['lab']] = (line['x'], line['y'])

    THRESH = 5.

    for i in range(1, max(data['f'])+1):
        frame = data[data['f']==i]
        total_squared_disp = 0.
        for p in frame:
            x, y = p['x'], p['y']
            total_squared_disp += (x-start_pos[p['lab']][0])**2 + \
                                  (y-start_pos[p['lab']][1])**2
        print('{0}: {1}'.format(i, total_squared_disp))
        if total_squared_disp > THRESH:
            break

    data = data[data['f']>=i-1]
    for j, l in enumerate(data):
        data[j]['f']=l['f']+1-i
    np.savetxt(out.format(fname), data, delimiter='     ',
               fmt=['%6d', '%7.3f', '%7.3f', '%4d', '%1.3f', '%5d'])

#!/usr/bin/env python
# encoding: utf-8
''' This script plots a histogram of the orientation velocity noise
for a one or several data sets.
The histogram figure is saved to file prefix.plothist.pdf

Run from the folder containing the positions file.

Sarah Schlossberg
August 2015
'''

from __future__ import division
import os
import helpy

prefix = raw_input('Prefix without trial number: ')
sets = int(raw_input('How many sets? '))
particle = raw_input('Particle type: ')
save = helpy.bool_input('Save figure? ')

import numpy as np
import matplotlib.pyplot as plt

def get_stats(hist):
    #Computes mean, D_T or D_R, and standard error for a list.
    hist = np.asarray(hist)
    hist *= 2.4
    mean = np.mean(hist)
    variance = np.var(hist)
    D = 0.5*variance
    SE = np.sqrt(variance) / np.sqrt(len(hist))
    return mean, D, SE

trackcount = 0
histbins = 100
histv = []

if sets > 1:
    for setnum in range(1, sets+1):
        spfprefix = prefix + str(setnum)
        spfprefix = os.path.join(spfprefix+'_d', os.path.basename(spfprefix))
        data, trackids, odata, omask = helpy.load_data(spfprefix)
        data = data[omask]
        odata = odata[omask]
        print odata

        for track in range(data['lab'].max()):
            mask = data['lab']==track
            if np.count_nonzero(mask) > 0:
                todata = odata[mask]['orient']
                vo = helpy.der(todata, iwidth=3)
                histv = np.concatenate((histv, vo), axis=1)
                trackcount += 1

elif sets == 1:
    spfprefix = prefix
    data, trackids, odata, omask = helpy.load_data(spfprefix)
    data = data[omask]
    odata = odata[omask]

    for track in range(data['lab'].max()):
        mask = data['lab']==track
        if np.count_nonzero(mask) > 0:
            todata = odata[mask]['orient']
            vo = helpy.der(todata, iwidth=3)
            histv = np.concatenate((histv, vo), axis=1)
            trackcount += 1

mean, D_R, SE = get_stats(histv)

fig = plt.figure()
plt.hist(histv, int(histbins), log=True, color=['blue'], label=['Mean = {:.5f} \n $D_R$ = {:.5f} \n Standard error = {:.5f}'.format(mean, D_R, SE)])
plt.legend(loc='center left')
plt.xlabel('Velocity step size in rad/frame')
plt.ylabel('Frequency')
plt.title("{} orientation tracks of {} ({})".format(trackcount, prefix, particle))

if save:
    print 'Saving plot to {}.plothistorient.pdf'.format(os.path.abspath(prefix))
    plt.savefig(prefix+'.plothistorient.pdf')
plt.show()

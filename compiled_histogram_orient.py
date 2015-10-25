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

def compile_for_hist(spfprefix):
    '''Adds data from one trial to two lists for transverse and orientation
    histograms.'''
    data, trackids, odata, omask = helpy.load_data(spfprefix)
    data = data[omask]
    odata = odata[omask]

    trackcount = 0
    for track in range(data['lab'].max()):
        mask = data['lab']==track
        if np.count_nonzero(mask) > 0:
            trackcount += 1
            todata = odata[mask]['orient']
            vo = helpy.der(todata, iwidth=3)
            histv.extend(vo)
    return trackcount

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
histv = []

if sets > 1:
    for setnum in range(1, sets+1):
        spfprefix = prefix + str(setnum)
        spfprefix = os.path.join(spfprefix+'_d', os.path.basename(spfprefix))
        settrackcount = compile_for_hist(spfprefix)
        trackcount += settrackcount

elif sets == 1:
    trackcount = compile_for_hist(prefix)
    spfprefix = prefix

def plot_hist(hist, ax=1, bins=100, log=True, title_suf=''):
    stats = get_stats(hist)
    ax = plt.gca()
    ax.hist(hist, bins, log=log, color='b',
            label=('$\\langle v \\rangle = {:.5f}$\n'
                   '$D_R = {:.5f}$\n'
                   '$\\sigma/\\sqrt{{N}} = {:.5f}$').format(*stats))
    ax.legend(loc='best')
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Velocity step size (rad/frame)')
    ax.set_title("{} orientation tracks of {} ({}){}".format(
                 trackcount, prefix.strip('/._'), particle, title_suf))
    return ax

plot_hist(histv)

if save:
    print 'Saving plot to {}.plothistorient.pdf'.format(os.path.abspath(prefix))
    plt.savefig(prefix+'.plothistorient.pdf')
plt.show()

#!/usr/bin/env python
# encoding: utf-8
''' This script plots a histogram of the translational velocity noise
for a one or several data sets. Includes option to subtract v_0.
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
subtract = helpy.bool_input('Subtract v0? ')
save = helpy.bool_input('Save figure? ')

import numpy as np
import matplotlib.pyplot as plt

S = 17

def compile_for_hist(spfprefix):
    '''Adds data from one trial to two lists for transverse and orientation
    histograms.'''
    data, trackids, odata, omask = helpy.load_data(spfprefix)
    data = data[omask]
    odata = odata[omask]

    for track in range(data['lab'].max()):
        mask = data['lab']==track
        if np.count_nonzero(mask) > 0:
            tdata = data[mask]
            todata = odata[mask]['orient']
            vx = helpy.der(tdata['x']/S, iwidth=3)
            vy = helpy.der(tdata['y']/S, iwidth=3)

            histv.extend(vx)
            histv.extend(vy)

            vnot = vx * np.cos(todata) + vy * np.sin(todata)
            etax = vx - vnot * np.cos(todata)
            etay = vy - vnot * np.sin(todata)
            eta.extend(etax)
            eta.extend(etay)
    return histv, eta

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
histv = eta = []

if sets > 1:
    for setnum in range(1, sets+1):
        spfprefix = prefix + str(setnum)
        spfprefix = os.path.join(spfprefix+'_d', os.path.basename(spfprefix))
        histv, eta = compile_for_hist(spfprefix)

        trackcount += 1

elif sets == 1:
    histv, eta = compile_for_hist(prefix)

def plot_hist(hist, ax=1, bins=100, log=True, title_suf=''):
    stats = get_stats(hist)
    if isinstance(ax, int):
        ax = plt.subplot(2, 1, ax)
    ax.hist(hist, bins, log=log, color='b',
            label=('$\\langle v \\rangle = {:.5f}$\n'
                   '$D_T = {:.5f}$\n'
                   '$\\sigma/\\sqrt{{N}} = {:.5f}$').format(*stats))
    ax.legend(loc='best')
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Velocity step size (particle/frame)')
    ax.set_title("{} tracks of {} ({}){}".format(
                 trackcount, prefix.strip('/._'), particle, title_suf))
    return ax

plot_hist(histv)
if subtract:
    plot_hist(eta, ax=2, title_suf=' with $v_0$ subtracted')

if save:
    print 'Saving plot to {}.plothist.pdf'.format(os.path.abspath(prefix))
    plt.savefig(prefix+'.plothist.pdf')

plt.show()

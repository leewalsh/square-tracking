#!/usr/bin/env python
''' This script plots a histogram of the transverse data for a one or several
data sets. Includes option to subtract v_0. The histogram is saved in the
format prefix.plothist.pdf

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
        tdata = data[data['lab']==track]
        todata = odata[data['lab']==track]
        if len(tdata) > 0:
            vx = helpy.der(tdata['x']/S, iwidth=3)
            vy = helpy.der(tdata['y']/S, iwidth=3)

            histv.extend(vx)
            histv.extend(vy)

            vnot = vx * np.cos(todata['orient']) + vy * np.sin(todata['orient'])
            etax = vx - vnot * np.cos(todata['orient'])
            etay = vy - vnot * np.sin(todata['orient'])
            eta.extend(etax)
            eta.extend(etay)
    return histv, eta

def get_stats(hist):
    #Computes mean, D_T or D_R, and standard error for a list.
    ret = []
    hist = np.asarray(hist)
    hist *= 2.4
    mean = np.mean(hist)
    ret.append(mean)
    variance = np.var(hist)
    D_T = 0.5*variance
    ret.append(D_T)
    SE = np.sqrt(variance) / np.sqrt(len(hist))
    ret.append(SE)
    return ret

trackcount = 0
histbins = 100
histv = eta = []

if sets > 1:
    for setnum in range(1, sets+1):
        spfprefix = prefix + str(setnum)
        spfprefix = os.path.join(spfprefix+'_d', os.path.basename(spfprefix))
        histv, eta = compile_for_hist(spfprefix)

        trackcount = trackcount + 1

elif sets == 1:
    histv, eta = compile_for_hist(prefix)

histvstat = get_stats(histv)
etastat = get_stats(eta)

fig = plt.figure()
if subtract:
    top = plt.subplot(2, 1, 1)
    top.hist(histv, int(histbins), log=True, color=['blue'], label=['Mean = {:.5f} \n $D_T$ = {:.5f} \n Standard error = {:.5f}'.format(histvstat[0], histvstat[1], histvstat[2])])
    top.legend(loc='center left')
    top.set_ylabel('Frequency')
    top.set_title("{} tracks of {} ({})".format(trackcount, prefix, particle))

    low = plt.subplot(2, 1, 2)
    low.hist(eta, int(histbins), log=True, color=['blue'], label=['Mean = {:.5f} \n $D_T$ = {:.5f} \n Standard error = {:.5f}'.format(etastat[0], etastat[1], etastat[2])])
    low.legend(loc='center left')
    low.set_xlabel('Velocity step size in rad/frame')
    low.set_ylabel('Frequency')
    low.set_title("{} tracks of {} ({}) with $v_0$ subtracted".format(trackcount, prefix, particle))

else:
    plt.subplot(2, 1, 1)
    plt.hist(histv, int(histbins), log=True, color=['blue'], label=['Mean = {:.5f} \n $D_T$ = {:.5f} \n Standard error = {:.5f}'.format(histvstat[0], histvstat[1], histvstat[2])])
    plt.legend(loc='center left')
    plt.xlabel('Velocity step size in rad/frame')
    plt.ylabel('Frequency')
    plt.title("{} tracks of {} ({})".format(trackcount, prefix, particle))

if save:
    print 'Saving plot to {}.plothist.pdf'.format(os.path.abspath(prefix))
    fig.savefig(prefix+'.plothist.pdf')

plt.show()

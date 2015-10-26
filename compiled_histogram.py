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

from argparse import ArgumentParser
p = ArgumentParser()
p.add_argument('prefix', help='Prefix without trial number')
p.add_argument('--sets', type=int, default=1, metavar='N', help='Number of sets')
p.add_argument('--particle', type=str, default='', metavar='NAME', help='Particle type name')
p.add_argument('--savefig', action='store_true', dest='savefig',
               help='Save figure?')
p.add_argument('--subtract', action='store_true', help='Subtract v0?')
p.add_argument('-s', '--side', type=float, default=17,
               help='Particle size in pixels, for unit normalization')
p.add_argument('-f', '--fps', type=float, default=2.4,
               help="Number of frames per second (or per shake) "
                    "for unit normalization")
args = p.parse_args()
prefix = args.prefix

import os
import numpy as np
import matplotlib.pyplot as plt
import helpy

def compile_for_hist(spfprefix):
    '''Adds data from one trial to two lists for transverse and orientation
    histograms.'''
    #TODO: orientation.track_orient() on todata before derivative?
    data, trackids, odata, omask = helpy.load_data(spfprefix)
    tracksets, otracksets = helpy.load_tracksets(data, trackids, odata, omask)

    for track in tracksets:
        tdata = tracksets[track]
        todata = otracksets[track]
        vx = helpy.der(tdata['x']/args.side, iwidth=3)
        vy = helpy.der(tdata['y']/args.side, iwidth=3)

        histv.extend(vx)
        histv.extend(vy)

        vnot = vx * np.cos(todata) + vy * np.sin(todata)
        etax = vx - vnot * np.cos(todata)
        etay = vy - vnot * np.sin(todata)

        eta.extend(etax)
        eta.extend(etay)
    return len(tracksets)

def get_stats(hist):
    #Computes mean, D_T or D_R, and standard error for a list.
    hist = np.asarray(hist)
    hist *= args.fps
    mean = np.mean(hist)
    variance = np.var(hist)
    D = 0.5*variance
    SE = np.sqrt(variance) / np.sqrt(len(hist))
    return mean, D, SE

trackcount = 0
histv = []
eta = []

if args.sets > 1:
    for setnum in range(1, args.sets+1):
        spfprefix = prefix + str(setnum)
        spfprefix = os.path.join(spfprefix+'_d', os.path.basename(spfprefix))
        settrackcount = compile_for_hist(spfprefix)
        trackcount += settrackcount

elif args.sets == 1:
    trackcount = compile_for_hist(prefix)

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
                 trackcount, prefix, args.particle, title_suf))
    return ax

prefix = prefix.strip('/._')

plot_hist(histv)
if args.subtract:
    plot_hist(eta, ax=2, title_suf=' with $v_0$ subtracted')

if args.savefig:
    print 'Saving plot to {}.plothist.pdf'.format(os.path.abspath(prefix))
    plt.savefig(prefix+'.plothist.pdf')
else:
    plt.show()

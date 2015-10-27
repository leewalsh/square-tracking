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

from argparse import ArgumentParser
p = ArgumentParser()
arg = p.add_argument
arg('prefix', help='Prefix without trial number')
arg('--sets', type=int, default=1, metavar='N', help='Number of sets')
arg('--particle', type=str, default='', metavar='NAME', help='Particle type name')
arg('--savefig', action='store_true', dest='savefig', help='Save figure?')
arg('--lin', action='store_false', dest='log', help='Plot on a linear scale?')
arg('--log', action='store_true', help='Plot on a log scale?')
arg('--torient', action='store_true', help='Track orientation?')
arg('--minlen', type=int, default=10, help='Minimum track length. Default: %(default)s')
arg('--subtract', action='store_true', help='Subtract v0?')
arg('-s', '--side', type=float, default=17, help='Particle size in pixels, '
    'for unit normalization. Default: %(default)s')
arg('-f', '--fps', type=float, default=2.4, help="Number of frames per second "
    "(or per shake) for unit normalization. Default: %(default)s")
args = p.parse_args()
prefix = args.prefix

import os
import numpy as np
import matplotlib.pyplot as plt
import helpy

def compile_for_hist(prefix):
    '''Adds data from one trial to two lists for transverse and orientation
    histograms.'''
    #TODO: orientation.track_orient() on todata before derivative?
    data, trackids, odata, omask = helpy.load_data(prefix)
    tracksets, otracksets = helpy.load_tracksets(data, trackids, odata, omask,
            min_length=args.minlen, run_track_orient=args.torient)

    for track in tracksets:
        todata = otracksets[track]
        vo = helpy.der(todata, iwidth=3)
        histv.extend(vo)
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
    ax = plt.gca()
    ax.hist(hist, bins, log=log, color='b',
            label=('$\\langle v \\rangle = {:.5f}$\n'
                   '$D_R = {:.5f}$\n'
                   '$\\sigma/\\sqrt{{N}} = {:.5f}$').format(*stats))
    ax.legend(loc='best')
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Velocity step size (rad/frame)')
    ax.set_title("{} orientation tracks of {} ({}){}".format(
                 trackcount, prefix, args.particle, title_suf))
    return ax

prefix = prefix.strip('/._')

plot_hist(histv, log=args.log)

if args.savefig:
    print 'Saving plot to {}.plothistorient.pdf'.format(os.path.abspath(prefix))
    plt.savefig(prefix+'.plothistorient.pdf')
else:
    plt.show()

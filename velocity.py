#!/usr/bin/env python
# encoding: utf-8

from __future__ import division

description = """This script plots a histogram of the velocity noise for one or
several data sets. Includes option to subtract v_0 from translational noise.
The histogram figure is optionally saved to file prefix.plothist[orient].pdf
Run from the folder containing the positions file.
Copyright (c) 2015 Sarah Schlossberg, Lee Walsh; all rights reserved.
"""

from argparse import ArgumentParser
p = ArgumentParser(description=description)
arg = p.add_argument
arg('prefix', help='Prefix without trial number')
arg('-o', '--orientation', action='store_false',
    dest='do_translation', help='Only orientational noise?')
arg('-t', '--translation', action='store_false',
    dest='do_orientation', help='Only translational noise?')
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
from collections import defaultdict
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

        if args.do_orientation:
            vo = helpy.der(todata, iwidth=3)
            vs['o'].extend(vo)

        if args.do_translation:
            tdata = tracksets[track]
            vx = helpy.der(tdata['x']/args.side, iwidth=3)
            vy = helpy.der(tdata['y']/args.side, iwidth=3)
            vs['x'].extend(vx)
            vs['y'].extend(vy)

            if args.subtract:
                vnot = vx * np.cos(todata) + vy * np.sin(todata)
                etax = vx - vnot * np.cos(todata)
                etay = vy - vnot * np.sin(todata)

                vs['etax'].extend(etax)
                vs['etay'].extend(etay)
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
vs = defaultdict(list)

if args.sets > 1:
    for setnum in range(1, args.sets+1):
        spfprefix = prefix + str(setnum)
        spfprefix = os.path.join(spfprefix+'_d', os.path.basename(spfprefix))
        settrackcount = compile_for_hist(spfprefix)
        trackcount += settrackcount

elif args.sets == 1:
    trackcount = compile_for_hist(prefix)

def plot_hist(hist, nax=1, axi=1, bins=100, log=True, orient=False, title_suf=''):
    stats = get_stats(hist)
    ax = plt.subplot(nax, 1, axi)
    ax.hist(hist, bins, log=log, color='b',
            label=('$\\langle v \\rangle = {:.5f}$\n'
                   '$D = {:.5f}$\n'
                   '$\\sigma/\\sqrt{{N}} = {:.5f}$').format(*stats))
    ax.legend(loc='best')
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Velocity ({}/vibation)'.format('rad' if orient else 'particle'))
    ax.set_title("{} tracks of {} ({}){}".format(
                 trackcount, prefix, args.particle, title_suf))
    return ax

prefix = prefix.strip('/._')

nax = sum([args.do_orientation, args.do_translation, args.do_translation and args.subtract])
axi = 1
if args.do_orientation:
    plot_hist(vs['o'], nax, axi, log=args.log, orient=True)
    axi += 1
if args.do_translation:
    plot_hist(vs['x'] + vs['y'], nax, axi, log=args.log)
    axi += 1
    if args.subtract:
        plot_hist(vs['etax'] + vs['etay'], nax, axi, log=args.log, title_suf=' with $v_0$ subtracted')
        axi += 1

if args.savefig:
    print 'Saving plot to {}.plothist.pdf'.format(os.path.abspath(prefix))
    plt.savefig(prefix+'.plothist.pdf')
else:
    plt.show()

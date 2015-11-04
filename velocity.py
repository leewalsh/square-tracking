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
arg('--untrackorient', action='store_false', dest='torient', help='Untracked raw orientation?')
arg('--minlen', type=int, default=10, help='Minimum track length. Default: %(default)s')
arg('--nosubtract', action='store_false', dest='subtract', help="Don't subtract v0?")
arg('-s', '--side', type=float, default=17, help='Particle size in pixels, '
    'for unit normalization. Default: %(default)s')
arg('-f', '--fps', type=float, default=2.4, help="Number of frames per second "
    "(or per shake) for unit normalization. Default: %(default)s")
args = p.parse_args()
prefix = args.prefix

import os
from collections import defaultdict
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import helpy

def compile_for_hist(prefix, do_orientation=True, do_translation=True,
                     subtract=True, minlen=10, torient=True, side=1, fps=1):
    '''Adds data from one trial to two lists for transverse and orientation
    histograms.'''
    data, trackids, odata, omask = helpy.load_data(prefix)
    tracksets, otracksets = helpy.load_tracksets(data, trackids, odata, omask,
            min_length=minlen, run_track_orient=torient)

    for track in tracksets:
        tdata = tracksets[track]
        todata = otracksets[track]
        x = tdata['f']/fps

        if do_orientation:
            vo = helpy.der(todata, x=x, iwidth=3)
            vs['o'].extend(vo)

        if do_translation:
            cos, sin = np.cos(todata), np.sin(todata)
            vs['cos'].extend(cos)
            vs['sin'].extend(sin)

            vx = helpy.der(tdata['x']/side, x=x, iwidth=3)
            vy = helpy.der(tdata['y']/side, x=x, iwidth=3)
            vs['x'].extend(vx)
            vs['y'].extend(vy)

            vI = vx*cos + vy*sin
            vT = vx*sin - vy*cos
            vs['I'].extend(vI)
            vs['T'].extend(vT)

            if subtract:
                v0 = vI.mean()
                etax = vx - v0*cos
                etay = vy - v0*sin
                vs['etax'].extend(etax)
                vs['etay'].extend(etay)

                etaI = vI - v0
                etaT = vT
                vs['etaI'].extend(etaI)
                vs['etaT'].extend(etaT)
    return len(tracksets)

def get_stats(a):
    #Computes mean, D_T or D_R, and standard error for a list.
    a = np.asarray(a)
    n = len(a)
    M = a.mean()
    c = a - M
    variance = np.dot(c, c)/n
    D = 0.5*variance
    SE = sqrt(variance)/sqrt(n)
    return M, D, SE

trackcount = 0
vs = defaultdict(list)

if args.sets > 1:
    for setnum in range(1, args.sets+1):
        spfprefix = prefix + str(setnum)
        spfprefix = os.path.join(spfprefix+'_d', os.path.basename(spfprefix))
        settrackcount = compile_for_hist(spfprefix,
                do_orientation=args.do_orientation, do_translation=args.do_translation,
                subtract=args.subtract, minlen=args.minlen,
                torient=args.torient, side=args.side, fps=args.fps)
        trackcount += settrackcount

elif args.sets == 1:
    trackcount = compile_for_hist(prefix,
            do_orientation=args.do_orientation, do_translation=args.do_translation,
            subtract=args.subtract, minlen=args.minlen,
            torient=args.torient, side=args.side, fps=args.fps)

def plot_hist(a, nax=1, axi=1, bins=100, log=True, orient=False, subtitle=''):
    stats = get_stats(a)
    ax = plt.subplot(nax, 1, axi)
    ax.hist(a, bins, log=log, color='b',
            label=('$\\langle v \\rangle = {:.5f}$\n'
                   '$D = {:.5f}$\n'
                   '$\\sigma/\\sqrt{{N}} = {:.5f}$').format(*stats))
    ax.legend(loc='best')
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Velocity ({}/vibation)'.format('rad' if orient else 'particle'))
    ax.set_title("{} tracks of {} ({})".format(trackcount, prefix, subtitle))
    return ax

prefix = prefix.strip('/._')

nax = sum([args.do_orientation, args.do_translation, args.do_translation and args.subtract])
axi = 1
if args.do_orientation:
    plot_hist(vs['o'], nax, axi, log=args.log, orient=True, subtitle=args.particle)
    axi += 1
if args.do_translation:
    plot_hist(vs['x'] + vs['y'], nax, axi, log=args.log, subtitle=args.particle)
    axi += 1
    if args.subtract:
        plot_hist(vs['etax'] + vs['etay'], nax, axi, log=args.log,
                  subtitle=', '.join([args.particle, '$v_0$ subtracted']))
        axi += 1

if args.savefig:
    print 'Saving plot to {}.plothist.pdf'.format(os.path.abspath(prefix))
    plt.savefig(prefix+'.plothist.pdf')
else:
    plt.show()

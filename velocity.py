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
arg('--width', type=float, default=3, metavar='W', help='Smoothing width for derivative')
arg('--particle', type=str, default='', metavar='NAME', help='Particle type name')
arg('--save', type=str, nargs='?', const='plothist', default='', help='Save figure?')
arg('--lin', action='store_false', dest='log', help='Plot on a linear scale?')
arg('--log', action='store_true', help='Plot on a log scale?')
arg('--dupes', action='store_true', help='Remove duplicates from tracks')
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
if args.dupes:
    from tracks import remove_duplicates

def compile_for_hist(prefix, do_orientation=True, do_translation=True, width=3,
                     subtract=True, minlen=10, torient=True, side=1, fps=1, dupes=False):
    '''Adds data from one trial to two lists for transverse and orientation
    histograms.'''
    data, trackids, odata, omask = helpy.load_data(prefix)
    if dupes:
        trackids = remove_duplicates(trackids, data)
    tracksets, otracksets = helpy.load_tracksets(data, trackids, odata, omask,
            min_length=minlen, run_track_orient=torient)

    for track in tracksets:
        tdata = tracksets[track]
        todata = otracksets[track]
        x = tdata['f']/fps

        if do_orientation:
            vo = helpy.der(todata, x=x, iwidth=width)
            vs['o'].extend(vo)

        if do_translation:
            cos, sin = np.cos(todata), np.sin(todata)
            vs['cos'].extend(cos)
            vs['sin'].extend(sin)

            vx = helpy.der(tdata['x']/side, x=x, iwidth=width)
            vy = helpy.der(tdata['y']/side, x=x, iwidth=width)
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

compile_args = dict(do_orientation=args.do_orientation, do_translation=args.do_translation,
                subtract=args.subtract, minlen=args.minlen, dupes=args.dupes,
                torient=args.torient, side=args.side, fps=args.fps, width=args.width)

if args.allwidths:
    widths = np.arange(0, 4, args.allwidths) + args.allwidths
    allvs = defaultdict(lambda: defaultdict(list))
    stats = {v: {s: np.empty_like(widths)
                 for s in 'mean var stderr'.split()}
             for v in 'o I T etaI etaT'.split()}
    for i, width in enumerate(widths):
        compile_args['width'] = width
        vs = allvs[width]
        for setnum in range(1, args.sets+1):
            spfprefix = prefix + str(setnum)
            spfprefix = os.path.join(spfprefix+'_d', os.path.basename(spfprefix))
            compile_for_hist(spfprefix, **compile_args)
        for v in stats:
            for s, stat in zip(stats[v], get_stats(vs[v])):
                stats[v][s][i] = stat
    for v in stats:
        for s in stats[v]:
            plt.plot(widths, stats[v][s], label=' '.join([v,s]))
    import sys; sys.exit()

if args.sets > 1:
    for setnum in range(1, args.sets+1):
        spfprefix = prefix + str(setnum)
        spfprefix = os.path.join(spfprefix+'_d', os.path.basename(spfprefix))
        settrackcount = compile_for_hist(spfprefix, **compile_args)
        trackcount += settrackcount

elif args.sets == 1:
    trackcount = compile_for_hist(prefix, **compile_args)

def plot_hist(a, nax=1, axi=1, bins=100, log=True, orient=False, label='v', title='', subtitle=''):
    stats = get_stats(a)
    ax = axi[0] if isinstance(axi, tuple) else plt.subplot(nax, 2, axi*2-1)
    bins = ax.hist(a, bins, log=False, alpha=0.7,
            label=('$\\langle {} \\rangle = {:.5f}$\n'
                   '$D = {:.5f}$\n'
                   '$\\sigma/\\sqrt{{N}} = {:.5f}$').format(label, *stats))[1]
    ax.legend(loc='upper left', fontsize='xx-small', frameon=False)
    ax.set_ylabel('Frequency')
    if orient:
        l, r = ax.set_xlim(bins[0], bins[-1])
        xticks = np.linspace(l, r, 5)
        ax.set_xticks(xticks)
        ax.set_xticklabels(['${:.2f}\pi$'.format(x) for x in xticks/np.pi], fontsize='small')
    ax.set_xlabel('Velocity ({}/vibation)'.format('rad' if orient else 'particle'))
    ax.set_title("{} ({})".format(title, subtitle), fontsize='medium')
    ax2 = axi[1] if isinstance(axi, tuple) else plt.subplot(nax, 2, axi*2)
    bins = ax2.hist(a, bins*2, log=True, alpha=0.7)[1]
    if orient:
        l, r = ax2.set_xlim(bins[0], bins[-1])
        xticks = np.linspace(l, r, 9)
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(['${:.2f}\pi$'.format(x) for x in xticks/np.pi], fontsize='small')
    return ax, ax2


nax = sum([args.do_orientation, args.do_translation, args.do_translation and args.subtract])
axi = 1
if args.do_orientation:
    plot_hist(vs['o'], nax, axi, bins=np.linspace(-np.pi/2,np.pi/2,51), log=args.log, orient=True,
            label=r'\xi', title='Orientation',
            subtitle = args.particle or prefix.strip('/._'))
    axi += 1
if args.do_translation:
    ax, ax2 = plot_hist(vs['I'], nax, axi, log=args.log,
            bins=np.linspace(-1,1), label='v_\parallel')
    plot_hist(vs['T'], nax, (ax, ax2), log=args.log, bins=np.linspace(-1,1), label='v_\perp',
            title='Parallel & Transverse', subtitle = args.particle or prefix.strip('/._'))
    axi += 1
    if args.subtract:
        plot_hist(vs['etax'] + vs['etay'], nax, axi, log=args.log,
            label=r'\eta_\alpha', bins=np.linspace(-1,1,51),
            title='$v_0$ subtracted', subtitle = args.particle or prefix.strip('/._'))
        axi += 1

if args.save:
    savename = '.'.join([os.path.abspath(args.prefix.rstrip('/._')), args.save, 'pdf'])
    print 'Saving plot to {}'.format(savename)
    plt.savefig(savename)
else:
    plt.show()

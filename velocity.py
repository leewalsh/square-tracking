#!/usr/bin/env python
# encoding: utf-8

from __future__ import division

description = """This script plots a histogram of the velocity noise for one or
several data sets. Includes option to subtract v_0 from translational noise.
The histogram figure is optionally saved to file prefix.plothist[orient].pdf
Run from the folder containing the positions file.
Copyright (c) 2015 Sarah Schlossberg, Lee Walsh; all rights reserved.
"""

if __name__=='__main__':
    from argparse import ArgumentParser
    p = ArgumentParser(description=description)
    arg = p.add_argument
    arg('prefix', help='Prefix without trial number')
    arg('-o', '--orientation', action='store_false',
        dest='do_translation', help='Only orientational noise?')
    arg('-t', '--translation', action='store_false',
        dest='do_orientation', help='Only translational noise?')
    arg('--sets', type=int, default=1, metavar='N', help='Number of sets')
    arg('--width', type=float, default=3, metavar='W', nargs='?', const=-.5,
            help='Smoothing width for derivative')
    arg('--particle', type=str, default='', metavar='NAME', help='Particle type name')
    arg('--save', type=str, nargs='?', const='plothist', default='', help='Save figure?')
    arg('--lin', action='store_false', dest='log', help='Plot on a linear scale?')
    arg('--log', action='store_true', help='Plot on a log scale?')
    arg('--dupes', action='store_true', help='Remove duplicates from tracks')
    arg('--normalize', action='store_true', help='Normalize by max?')
    arg('--untrackorient', action='store_false', dest='torient', help='Untracked raw orientation?')
    arg('--minlen', type=int, default=10, help='Minimum track length. Default: %(default)s')
    arg('--nosubtract', action='store_false', dest='subtract', help="Don't subtract v0?")
    arg('-s', '--side', type=float, default=17, help='Particle size in pixels, '
        'for unit normalization. Default: %(default)s')
    arg('-f', '--fps', type=float, default=2.4, help="Number of frames per second "
        "(or per shake) for unit normalization. Default: %(default)s")
    args = p.parse_args()

import os
from collections import defaultdict
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import helpy, tracks

def noise_derivatives(tdata, todata, width=1, side=1, fps=1, xy=False,
                      do_orientation=True, do_translation=True, subtract=True):
    x = tdata['f']/fps
    ret = {}
    ws = [width] if np.isscalar(width) else width
    if do_orientation:
        ret['o'] = np.array([helpy.der(todata, x=x, iwidth=w)
                             for w in ws]).squeeze()
    if do_translation:
        cos, sin = np.cos(todata), np.sin(todata)
        vx, vy = [np.array([helpy.der(tdata[i]/side, x=x, iwidth=w)
                            for w in ws]).squeeze() for i in 'xy']
        if xy:
            ret['x'], ret['y'] = vx, vy
        else:
            vI = vx*cos + vy*sin
            vT = vx*sin - vy*cos
            ret['par'], ret['perp'] = vI, vT
        if subtract:
            v0 = vI.mean(-1, keepdims=vI.ndim>1)
            if xy:
                ret['etax'] = vx - v0*cos
                ret['etay'] = vy - v0*sin
            else:
                ret['etapar'] = vI - v0
    return ret

def compile_noise(prefixes, vs=defaultdict(list), width=3, side=1, fps=1,
                  do_orientation=True, do_translation=True, subtract=True,
                  minlen=10, torient=True, dupes=False, **ignored):
    if np.isscalar(prefixes):
        prefixes = [prefixes]
    for prefix in prefixes:
        print "Loading data for", prefix
        data, trackids, odata, omask = helpy.load_data(prefix)
        if dupes:
            trackids = tracks.remove_duplicates(trackids, data)
        tracksets, otracksets = helpy.load_tracksets(data, trackids, odata, omask,
                min_length=minlen, run_track_orient=torient)
        for track in tracksets:
            tdata = tracksets[track]
            todata = otracksets[track]
            velocities = noise_derivatives(tdata, todata, width=width,
                    side=side, fps=fps, do_orientation=do_orientation,
                    do_translation=do_translation, subtract=subtract)
            for v in velocities:
                vs[v].append(velocities[v])
    for v in vs:
        vs[v] = np.concatenate(vs[v], -1)
    return len(tracksets)

def get_stats(a):
    #Computes mean, D_T or D_R, and standard error for a list.
    a = np.asarray(a)
    n = a.shape[-1]
    M = a.mean(-1, keepdims=a.ndim>1)
    c = a - M
    variance = np.einsum('...j,...j->...', c, c)/n
    D = 0.5*variance
    SE = np.sqrt(variance)/sqrt(n)
    return M, D, SE

def compile_widths(widths, prefixes, **compile_args):
    stats = {v: {s: np.empty_like(widths)
                 for s in 'mean var stderr'.split()}
             for v in 'o par perp etapar'.split()}
    compile_args['width'] = widths
    vs = defaultdict(list)
    compile_noise(prefixes, vs, **compile_args)
    for v, s in stats.items():
        s['mean'], s['var'], s['stderr'] = get_stats(vs[v])
    return stats

def plot_widths(widths, stats, normalize=False):
    ls = {'o': '-', 'par': '-.', 'perp': ':', 'etapar': '--'}
    cs = {'mean': 'r', 'var': 'g', 'stderr': 'b'}
    label = {'o': r'$\xi$', 'par': r'$v_\parallel$', 'perp': r'$v_\perp$', 'etapar': r'$\eta_\perp$'}
    fig = plt.figure(figsize=(8,12))
    for i, s in enumerate(stats['o']):
        ax = fig.add_subplot(len(stats['o']), 1, i+1)
        for v in stats:
            val = stats[v][s]
            if normalize:
                sign = np.sign(val.sum())
                val = sign*val
                val = val/val.max()
                ax.axhline(1, lw=0.5, c='k', ls=':', alpha=0.5)
            ax.plot(widths, val, '.'+ls[v]+cs[s], label=label[v])
        ax.set_title(s)
        ax.margins(y=0.1)
        if normalize:
            ax.set_ylim(-0.1, 1.1)
        ax.legend(loc='best')
    return fig

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

if __name__=='__main__':
    compile_args = dict(args.__dict__)
    prefix = compile_args.pop('prefix')
    prefixes = ["{}{i}_d/{}{i}".format(
                        args.prefix, os.path.basename(args.prefix), i=i+1)
                for i in xrange(args.sets)]
    if args.width < 0:
        widths = np.arange(0, 4, -args.width) - args.width
        stats = compile_widths(widths, prefixes, **compile_args)
        plot_widths(widths, stats, normalize=args.normalize)
    else:
        vs = defaultdict(list)
        trackcount = compile_noise(prefixes, vs, **compile_args)

        nax = sum([args.do_orientation, args.do_translation, args.do_translation and args.subtract])
        axi = 1
        subtitle = args.particle or os.path.basename(args.prefix).strip('/._')
        if args.do_orientation:
            plot_hist(vs['o'], nax, axi, bins=np.linspace(-np.pi/2,np.pi/2,51), log=args.log,
                    orient=True, label=r'\xi', title='Orientation', subtitle=subtitle)
            axi += 1
        if args.do_translation:
            ax, ax2 = plot_hist(vs['par'], nax, axi, log=args.log,
                    bins=np.linspace(-1,1), label='v_\parallel')
            plot_hist(vs['perp'], nax, (ax, ax2), log=args.log, bins=np.linspace(-1,1),
                    label='v_\perp', title='Parallel & Transverse', subtitle=subtitle)
            axi += 1
            if args.subtract:
                plot_hist(np.concatenate([vs['etapar'], vs['perp']]), nax, axi, log=args.log,
                    label=r'\eta_\alpha', bins=np.linspace(-1,1,51),
                    title='$v_0$ subtracted', subtitle=subtitle)
                axi += 1

    if args.save:
        savename = '.'.join([os.path.abspath(args.prefix.rstrip('/._')), args.save, 'pdf'])
        print 'Saving plot to {}'.format(savename)
        plt.savefig(savename)
    else:
        plt.show()

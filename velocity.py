#!/usr/bin/env python
# encoding: utf-8

from __future__ import division

import os
from collections import defaultdict
from glob import iglob
from math import sqrt

import numpy as np
import matplotlib.pyplot as plt

import helpy
import tracks
import correlation as corr

description = """This script plots a histogram of the velocity noise for one or
several data sets. Includes option to subtract v_0 from translational noise.
The histogram figure is optionally saved to file prefix.plothist[orient].pdf
Run from the folder containing the positions file.
Copyright (c) 2015 Sarah Schlossberg, Lee Walsh; all rights reserved.
"""

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description=description)
    arg = parser.add_argument
    arg('prefix', help='Prefix without trial number')
    arg('-o', '--orientation', action='store_false',
        dest='do_translation', help='Only orientational noise?')
    arg('-t', '--translation', action='store_false',
        dest='do_orientation', help='Only translational noise?')
    arg('--sets', type=int, default=0, metavar='N', nargs='?', const=1,
        help='Number of sets')
    arg('--width', type=float, default=(0.65,), metavar='W', nargs='*',
        help='Smoothing width for derivative, may give several')
    arg('--particle', type=str, default='', help='Particle name')
    arg('--save', type=str, nargs='?', const='velocity', default='',
        help='Save figure (optionally provide suffix)?')
    arg('--lin', action='store_true', help='Plot on linear scale?')
    arg('--log', action='store_true', help='Plot on a log scale?')
    arg('--dupes', action='store_true', help='Remove duplicates from tracks')
    arg('--normalize', action='store_true', help='Normalize by max?')
    arg('--autocorr', action='store_true', help='Plot <vv> autocorrelation?')
    arg('--untrackorient', action='store_false', dest='torient',
        help='Untracked raw orientation (mod 2pi)?')
    arg('-g', '--gaps', choices=['interp', 'nans', 'leave'], default='interp',
        nargs='?', const='nans', help="Gap handling: choose from %(choices)s. "
        "default is %(default)s, `-g` or `--gaps` alone gives %(const)s")
    arg('--stub', type=int, help='Min track length. Default: 10')
    arg('--nosubtract', action='store_false', dest='subtract',
        help="Don't subtract v0?")
    arg('-s', '--side', type=float,
        help='Particle size in pixels, for unit normalization')
    arg('-f', '--fps', type=float, help="Number frames per shake "
        "(or second) for unit normalization.")
    arg('-v', '--verbose', action='count', help="Be verbose")
    args = parser.parse_args()

pi = np.pi
vcol = (1, 0.4, 0)
pcol = (0.25, 0.5, 0)
ncol = (0.4, 0.4, 1)


def noise_derivatives(tdata, width=(1,), side=1, fps=1, xy=False,
                      do_orientation=True, do_translation=True, subtract=True):
    x = tdata['f']/fps
    ret = {}
    if do_orientation:
        ret['o'] = np.array([helpy.der(tdata['o'], x=x, iwidth=w)
                             for w in width]).squeeze()
    if do_translation:
        cos, sin = np.cos(tdata['o']), np.sin(tdata['o'])
        vx, vy = [np.array([helpy.der(tdata[i]/side, x=x, iwidth=w)
                            for w in width]).squeeze() for i in 'xy']
        if xy:
            ret['x'], ret['y'] = vx, vy
        else:
            vI = vx*cos + vy*sin
            vT = vx*sin - vy*cos
            ret['par'], ret['perp'] = vI, vT
        if subtract:
            v0 = vI.mean(-1, keepdims=vI.ndim > 1)
            if xy:
                ret['etax'] = vx - v0*cos
                ret['etay'] = vy - v0*sin
            else:
                ret['etapar'] = vI - v0
    return ret


def compile_noise(prefixes, vs, width=(1,), side=1, fps=1, cat=True,
                  do_orientation=True, do_translation=True, subtract=True,
                  stub=10, torient=True, gaps='interp', dupes=False, **ignored):
    if np.isscalar(prefixes):
        prefixes = [prefixes]
    for prefix in prefixes:
        if args.verbose:
            print "Loading data for", prefix
        data = helpy.load_data(prefix, 'tracks')
        if dupes:
            data['t'] = tracks.remove_duplicates(data['t'], data)
        tracksets = helpy.load_tracksets(data, min_length=stub,
                                         run_track_orient=torient,
                                         run_fill_gaps=gaps)
        for track in tracksets:
            tdata = tracksets[track]
            velocities = noise_derivatives(
                tdata, width=width, side=side, fps=fps, subtract=subtract,
                do_orientation=do_orientation, do_translation=do_translation)
            for v in velocities:
                vs[v].append(velocities[v])
    if cat:
        for v in vs:
            vs[v] = np.concatenate(vs[v], -1)
    return len(tracksets)


def get_stats(a):
    """Computes mean, D_T or D_R, and standard error for a list.
    """
    a = np.asarray(a)
    n = a.shape[-1]
    M = np.nanmean(a, -1, keepdims=a.ndim > 1)
    # c = a - M
    # variance = np.einsum('...j,...j->...', c, c)/n
    variance = np.nanvar(a, -1, keepdims=a.ndim > 1, ddof=1)
    D = 0.5*variance
    SE = np.sqrt(variance)/sqrt(n - 1)
    return M, D, SE


def compile_widths(prefixes, **compile_args):
    stats = {v: {s: np.empty_like(compile_args['width'])
                 for s in 'mean var stderr'.split()}
             for v in 'o par perp etapar'.split()}
    vs = defaultdict(list)
    compile_noise(prefixes, vs, **compile_args)
    for v, s in stats.items():
        s['mean'], s['var'], s['stderr'] = get_stats(vs[v])
    return stats


def plot_widths(widths, stats, normalize=False):
    ls = {'o': '-', 'par': '-.', 'perp': ':', 'etapar': '--'}
    cs = {'mean': 'r', 'var': 'g', 'stderr': 'b'}
    label = {'o': r'$\xi$', 'par': r'$v_\parallel$', 'perp': r'$v_\perp$',
             'etapar': r'$\eta_\parallel$'}
    fig = plt.figure(figsize=(8, 12))
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
        ax.minorticks_on()
        ax.grid(axis='x', which='both')
        if normalize:
            ax.set_ylim(-0.1, 1.1)
        ax.legend(loc='best')
    return fig


def plot_hist(a, nax=(1, 1), axi=1, bins=100, log=True, lin=True, orient=False,
              label='v', title='', subtitle=''):
    stats = get_stats(a)
    nrows, ncols = nax
    if isinstance(axi, tuple):
        ax = axi[0]
    else:
        ax = plt.subplot(nrows, ncols, (axi - 1)*ncols + 1)
    label = '\n'.join(['$\\langle {val} \\rangle = {0:.5f}$',
                       '$\ D_{sub}\  = {1:.5f}$',
                       '$\\sigma/\\sqrt{{N}} = {2:.5f}$'][:2]).format(
                           *stats[:2], **label)
    counts, bins, _ = ax.hist(a, bins, log=False, alpha=0.7, label=label)
    plot_gaussian(stats[0], stats[1], bins, counts.sum(), ax)
    ax.legend(loc='upper left', fontsize='small', frameon=False)
    ax.set_ylabel('Frequency')
    l, r = ax.set_xlim(bins[0], bins[-1])
    if orient:
        xticks = np.linspace(l, r, 5)
        ax.set_xticks(xticks)
        xticklabels = map('${:.2f}\pi$'.format, xticks/pi)
        ax.set_xticklabels(xticklabels, fontsize='small')
    xlabel = 'Velocity ({}/vibration)'.format('rad' if orient else 'particle')
    ax.set_xlabel(xlabel)
    ax.set_title(" ".join([title, subtitle]), fontsize='medium')
    if ncols < 2:
        return stats, (ax,)
    if isinstance(axi, tuple):
        ax2 = axi[1]
    else:
        ax2 = plt.subplot(nrows, ncols, axi*ncols)
    counts, bins, _ = ax2.hist(a, bins*2, log=True, alpha=0.7)
    plot_gaussian(stats[0], stats[1], bins, counts.sum(), ax2)
    if orient:
        l, r = ax2.set_xlim(bins[0], bins[-1])
        xticks = np.linspace(l, r, 9)
        ax2.set_xticks(xticks)
        xticklabels = map('${:.2f}\pi$'.format, xticks/pi)
        ax2.set_xticklabels(xticklabels, fontsize='small')
    ax2.set_ylim(1, 10**int(1 + np.log10(counts.max())))
    return stats, (ax, ax2)


def plot_gaussian(M, D, bins, count=1, ax=None):
    ax = ax or plt.gca()
    dx = bins[1] - bins[0]
    var = 2*D
    g = np.exp(-0.5 * (bins-M)**2 / var)
    g /= sqrt(2*pi*var) / (dx*count)
    ax.plot(bins, g, c=pcol, lw=2)


def vv_autocorr(prefixes, corrlen=0.5, **compile_args):
    vs = defaultdict(list)
    compile_noise(prefixes, vs, cat=False, **compile_args)
    vvs = {}
    for v, tvs in vs.iteritems():
        vcorrlen = int(corrlen*max(map(len, tvs))) if corrlen < 1 else corrlen
        vv = np.full((len(tvs), vcorrlen), np.nan, float)
        for i, tv in enumerate(tvs):
            ac = corr.autocorr(tv, norm=1, cumulant=False)
            vv[i, :len(ac)] = ac[:corrlen]
        vvcount = np.isfinite(vv).sum(0)
        vv = vv[:, vvcount > 0]
        vv = np.nanmean(vv, 0)
        dvv = np.nanstd(vv, 0)/np.sqrt(vvcount)
        vvs[v] = vv, dvv
    return vvs


if __name__ == '__main__':
    suf = '_TRACKS.npz'
    if '*' in args.prefix or '?' in args.prefix:
        fs = iglob(args.prefix+suf)
    else:
        dirname, prefix = os.path.split(args.prefix)
        dirm = (dirname or '*') + (prefix + '*/')
        basm = prefix.strip('/._')
        fs = iglob(dirm + basm + '*' + suf)
    prefixes = [s[:-len(suf)] for s in fs] or args.prefix

    helpy.save_log_entry(args.prefix, 'argv')
    meta = helpy.load_meta(args.prefix)
    if args.verbose:
        print 'prefixes:',
        print '\n          '.join(np.atleast_1d(prefixes))
    helpy.sync_args_meta(args, meta,
                         ['side', 'fps'], ['sidelength', 'fps'], [1, 1])
    compile_args = dict(args.__dict__)

    label = {'o': r'$\xi$', 'par': r'$v_\parallel$', 'perp': r'$v_\perp$',
             'etapar': r'$\eta_\parallel$'}
    ls = {'o': '-', 'par': '-.', 'perp': ':', 'etapar': '--'}
    cs = {'mean': 'r', 'var': 'g', 'stderr': 'b'}
    if len(args.width) > 1:
        stats = compile_widths(prefixes, **compile_args)
        plot_widths(args.width, stats, normalize=args.normalize)
    elif args.autocorr:
        vvs = vv_autocorr(prefixes, corrlen=10*args.fps, **compile_args)
        fig, ax = plt.figure()
        for v in vvs:
            vv, dvv = vvs[v]
            t = np.arange(len(vv))/args.fps
            ax.errorbar(t, vv, yerr=dvv, label=label[v], ls=ls[v])
        ax.set_title(r"Velocity Autocorrelation $\langle v(t) v(0) \rangle$")
        ax.legend(loc='best')
    else:
        args.width = args.width[0]
        helpy.sync_args_meta(args, meta,
                             ['stub', 'gaps', 'width'],
                             ['vel_stub', 'vel_gaps', 'vel_dx_width'],
                             [10, 'interp', 0.65])
        args.width = [args.width]
        vs = defaultdict(list)
        trackcount = compile_noise(prefixes, vs, **compile_args)

        nax = (args.do_orientation + args.do_translation*(args.subtract + 1),
               args.log + args.lin)
        axi = 1
        subtitle = args.particle
        bins = np.linspace(-1, 1, 51)
        brange = 0.5
        if args.do_orientation:
            title = 'Orientation'
            label = {'val': r'\xi', 'sub': 'R'}
            stats, axes = plot_hist(vs['o'], nax, axi, bins=bins*pi/2,
                                    lin=args.lin, log=args.log, label=label,
                                    orient=True, title=title, subtitle=subtitle)
            # could also save fit_vo_w0=stats[0])
            meta.update(fit_vo_DR=stats[1])
            axi += 1
        if args.do_translation:
            title = 'Parallel & Transverse'
            label = {'val': 'v_\perp', 'sub': '\perp'}
            stats, axes = plot_hist(vs['perp'], nax, axi, bins=bins*brange,
                                    lin=args.lin, log=args.log, label=label,
                                    title=title, subtitle=subtitle)
            meta.update(fit_vt_DT=stats[1])
            label = {'val': 'v_\parallel', 'sub': '\parallel'}
            stats, axes = plot_hist(vs['par'], nax, axes, bins=bins*brange,
                                    lin=args.lin, log=args.log, label=label)
            meta.update(fit_vn_v0=stats[0], fit_vn_DT=stats[1])
            axi += 1
            if args.subtract:
                plot_hist(np.concatenate([vs['etapar'], vs['perp']]), nax, axi,
                          lin=args.lin, log=args.log, label=r'\eta_\alpha',
                          bins=bins, title='$v_0$ subtracted',
                          subtitle=subtitle)
                axi += 1

    if args.save:
        savename = os.path.abspath(args.prefix.rstrip('/._?*'))
        helpy.save_meta(savename, meta)
        savename += '_' + helpy.with_prefix(args.save, 'velocity_') + '.pdf'
        print 'Saving plot to {}'.format(savename)
        plt.savefig(savename)
    else:
        plt.show()

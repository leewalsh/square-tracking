#!/usr/bin/env python
# encoding: utf-8

from __future__ import division

import os
from glob import iglob
from math import sqrt

import numpy as np
from scipy.stats import skew, kurtosis, skewtest, kurtosistest
import matplotlib.pyplot as plt

import helpy
import correlation as corr
import curve

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
    arg('--noshow', action='store_false', dest='show',
        help="Don't show figures (just save them)")
    arg('--nosave', action='store_false', dest='save',
        help="Don't save outputs or figures")
    arg('--suffix', type=str, default='', help='Suffix to append to savenames')
    arg('--lin', action='store_true', help='Plot on linear scale?')
    arg('--log', action='store_true', help='Plot on a log scale?')
    arg('--dupes', action='store_true', help='Remove duplicates from tracks')
    arg('--normalize', action='store_true', help='Normalize by max?')
    arg('--autocorr', action='store_true', help='Plot <vv> autocorrelation?')
    arg('--frame', choices=['lab', 'self'], default='self',
        help='Correlations in "lab" or "self" frame?')
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

    if args.save or args.show:
        plt.rc('text', usetex=True)

pi = np.pi
vcol = (1, 0.4, 0)
pcol = (0.25, 0.5, 0)
ncol = (0.4, 0.4, 1)


def noise_derivatives(tdata, width=(0.65,), side=1, fps=1):
    """calculate angular and positional derivatives in lab & particle frames.

    Returns the derivatives in structured array with dtype having fields
        'o', 'x', 'y', 'par', 'perp'
    for the velocity in
        orientation, x, y, parallel, and perpendicular directions
    and noise fields
        'etax', 'etay', 'etapar'
    for velocity with average forward motion subtracted

    the arrays dtype is defined by helpy.vel_dtype
    """
    shape = ((len(width),) if len(width) > 1 else ()) + tdata.shape
    v = np.empty(shape, helpy.vel_dtype)
    x = tdata['f']/fps
    cos, sin = np.cos(tdata['o']), np.sin(tdata['o'])
    unit = {'x': side, 'y': side, 'o': 1}
    for oxy in 'oxy':
        v[oxy] = np.array([curve.der(tdata[oxy]/unit[oxy], x=x, iwidth=w)
                           for w in width]).squeeze()
    v['v'] = np.hypot(v['x'], v['y'])
    v['par'] = v['x']*cos + v['y']*sin
    v['perp'] = v['x']*sin - v['y']*cos
    v0 = v['par'].mean(-1, keepdims=len(shape) > 1)
    v['etax'] = v['x'] - v0*cos
    v['etay'] = v['y'] - v0*sin
    v['eta'] = np.hypot(v['etax'], v['etay'])
    v['etapar'] = v['par'] - v0
    return v


def compile_noise(prefixes, width=(0.65,), side=1, fps=1, cat=True, stub=10,
                  torient=True, gaps='interp', dupes=False, verbose=False, **_):
    if np.isscalar(prefixes):
        prefixes = [prefixes]
    vs = {}
    for prefix in prefixes:
        if verbose:
            print "Loading data for", prefix
        data = helpy.load_data(prefix, 'tracks')
        tsets = helpy.load_tracksets(data, min_length=stub, run_repair=gaps,
                                     verbose=verbose, run_remove_dupes=dupes,
                                     run_track_orient=torient)
        vsets = {t: noise_derivatives(ts, width=width, side=side, fps=fps)
                 for t, ts in tsets.iteritems()}
        if cat:
            fsets, fvsets = helpy.load_framesets((tsets, vsets))
            fs = sorted(fsets)
            # tdata = np.concatenate([fsets[f] for f in fs])
            vdata = np.concatenate([fvsets[f] for f in fs])
            vs[prefix] = vdata
        else:
            vs[prefix] = vsets
    return vs


def get_stats(a):
    """Computes mean, D_T or D_R, and standard error for a list.
    """
    a = np.asarray(a)
    n = a.shape[-1]
    keepdims = a.ndim > 1
    M = np.nanmean(a, -1, keepdims=keepdims)
    # c = a - M
    # variance = np.einsum('...j,...j->...', c, c)/n
    variance = np.nanvar(a, -1, keepdims=keepdims, ddof=1)
    D = 0.5*variance
    SE = np.sqrt(variance)/sqrt(n - 1)
    SK = skew(a, -1, nan_policy='omit')
    KU = kurtosis(a, -1, nan_policy='omit')
    SK_t = skewtest(a, -1, nan_policy='omit')
    KU_t = kurtosistest(a, -1, nan_policy='omit')
    if args.verbose:
        print 'skewtest:', SK, SK_t
        print 'kurtosistest:', KU, KU_t
    if keepdims:
        SK = SK[..., None]
        KU = KU[..., None]
    else:
        SK = float(SK)
        KU = float(KU)
    return {'mean': M, 'var': D, 'std': SE, 'skew': SK, 'kurt': KU}


def compile_widths(prefixes, **compile_args):
    vs = compile_noise(prefixes, cat=True, **compile_args)
    stats = {v: get_stats(np.concatenate(pvs.values()))
             for v, pvs in helpy.transpose_dict(vs).items()}
    return stats


def plot_widths(widths, stats, normalize=False):
    ls = {'o': '-', 'par': '-.', 'perp': ':', 'etapar': '--'}
    cs = {'mean': 'r', 'var': 'g', 'std': 'b', 'skew': 'm', 'kurt': 'k'}
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
              label='v', title='', subtitle='', c=vcol):
    if args.verbose:
        print title + subtitle + str(label)
    stats = get_stats(a)
    nrows, ncols = nax
    if isinstance(axi, tuple):
        ax = axi[0]
    else:
        ax = plt.subplot(nrows, ncols, (axi - 1)*ncols + 1)
    label.update(stats)
    label = '\n'.join(['$\\langle {val} \\rangle = {mean:.5f}$',
                       '$\ D_{sub}\  = {var:.5f}$',
                       '$\ \gamma_1 = {skew:.5f}$',
                       '$\ \gamma_2 = {kurt:.5f}$',
                       '$\\sigma/\\sqrt{{N}} = {std:.5f}$'][:2]).format(**label)
    counts, bins, _ = ax.hist(a, bins, range=(bins[0], bins[-1]), log=False,
                              alpha=0.7, label=label, color=c)
    plot_gaussian(stats['mean'], stats['var'], bins, counts.sum(), ax)
    ax.legend(loc='upper left', fontsize='small', frameon=False)
    # ax.set_ylabel('Frequency')
    xlabel = r'$\Delta r \times f/\ell$'
    l, r = ax.set_xlim(bins[0], bins[-1])
    if orient:
        xticks = np.linspace(l, r, 5)
        ax.set_xticks(xticks)
        xticklabels = map('${:.2f}\pi$'.format, xticks/pi)
        ax.set_xticklabels(xticklabels, fontsize='small')
        xlabel = r'$\Delta\theta \times f$'
    ax.set_xlabel(xlabel)
    # ax.set_title(" ".join([title, subtitle]), fontsize='medium')
    if ncols < 2:
        return stats, (ax,)
    if isinstance(axi, tuple):
        ax2 = axi[1]
    else:
        ax2 = plt.subplot(nrows, ncols, axi*ncols)
    counts, bins, _ = ax2.hist(a, bins*2, range=(2*bins[0], 2*bins[-1]),
                               log=True, alpha=0.7, color=c)
    plot_gaussian(stats['mean'], stats['var'], bins, counts.sum(), ax2)
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


def vv_autocorr(vs, normalize=False):
    normalize = normalize and 1
    flat_dt = 'f4'
    vvs = [corr.autocorr(tv.view((flat_dt, 8)), norm=normalize, cumulant=False)
           for pvs in vs.itervalues() for tv in pvs.itervalues()]
    vvs, vv, dvv = helpy.avg_uneven(vvs, weight=True)
    return [np.array(a, order='C').astype('f4').view(helpy.vel_dtype).squeeze()
            for a in vvs, vv, dvv]


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
             'etapar': r'$\eta_\parallel$', 'x': '$v_x$', 'y': '$v_y$'}
    ls = {'o': '-', 'x': '-.', 'y': ':',
          'par': '-.', 'perp': ':', 'etapar': '--'}
    cs = {'mean': 'r', 'var': 'g', 'std': 'b'}

if __name__ == '__main__':
    if len(args.width) > 1:
        stats = compile_widths(prefixes, **compile_args)
        plot_widths(args.width, stats, normalize=args.normalize)
    elif args.autocorr:
        vs = compile_noise(prefixes, cat=False, **compile_args)
        vvs, vv, dvv = vv_autocorr(vs, normalize=args.normalize)
        fig, ax = plt.subplots()
        t = np.arange(len(vv))/args.fps
        for v in ['o', 'par', 'perp', 'etapar']:
            ax.errorbar(t, vv[v], yerr=dvv[v], label=label[v], ls=ls[v])
        ax.set_xlim(0, 10*args.fps)
        ax.set_title(r"Velocity Autocorrelation $\langle v(t) v(0) \rangle$")
        ax.legend(loc='best')
    else:
        args.width = args.width[0]
        helpy.sync_args_meta(args, meta,
                             ['stub', 'gaps', 'width'],
                             ['vel_stub', 'vel_gaps', 'vel_dx_width'],
                             [10, 'interp', 0.65])
        args.width = [args.width]
        compile_args.update(args.__dict__)
        vs = compile_noise(prefixes, cat=True, **compile_args)
        if not (args.log or args.lin):
            args.log = args.lin = True

        nax = (args.do_orientation + args.do_translation*(args.subtract + 1),
               args.log + args.lin)
        plt.figure(figsize=(5*nax[1], 2.5*nax[0]))
        axi = 1
        subtitle = args.particle
        bins = np.linspace(-1, 1, 51)
        brange = 0.5
        if args.do_orientation:
            title = 'Orientation'
            label = {'val': r'\xi', 'sub': 'R'}
            stats, axes = plot_hist(vs['o'], nax, axi, bins=bins*pi/2, c=ncol,
                                    lin=args.lin, log=args.log, label=label,
                                    orient=True, title=title, subtitle=subtitle)
            # could also save fit_vo_w0=stats['mean'])
            meta.update(fit_vo_DR=stats['var'])
            axi += 1
        if args.do_translation:
            title = 'Parallel & Transverse'
            label = {'val': 'v_\perp', 'sub': '\perp'}
            stats, axes = plot_hist(vs['perp'], nax, axi, bins=bins*brange,
                                    lin=args.lin, log=args.log, label=label,
                                    title=title, subtitle=subtitle, c=ncol)
            meta.update(fit_vt_DT=stats['var'])
            label = {'val': 'v_\parallel', 'sub': '\parallel'}
            stats, axes = plot_hist(vs['par'], nax, axes, bins=bins*brange,
                                    lin=args.lin, log=args.log, label=label)
            meta.update(fit_vn_v0=stats['mean'], fit_vn_DT=stats['var'])
            axi += 1
            if args.subtract:
                label = {'val': r'\eta_\alpha', 'sub': r'\alpha'}
                plot_hist(np.concatenate([vs['etapar'], vs['perp']]), nax, axi,
                          lin=args.lin, log=args.log, label=label,
                          bins=bins, title='$v_0$ subtracted',
                          subtitle=subtitle)
                axi += 1

if __name__ == '__main__' and args.save:
    savename = os.path.abspath(args.prefix.rstrip('/._?*'))
    helpy.save_meta(savename, meta)
    savename += '_velocity'
    if args.suffix:
        savename += '_' + args.suffix.strip('_')
    savename += '.pdf'
    print 'Saving plot to {}'.format(savename)
    plt.savefig(savename)

if __name__ == '__main__' and args.show:
    plt.show()

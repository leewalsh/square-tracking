#!/usr/bin/env python
# encoding: utf-8

from __future__ import division

import os
from glob import iglob
from functools import partial
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
    arg('command', nargs='+', choices=['widths', 'hist', 'autocorr'],
        help=('Which command to run: '
              '`widths`: plot several derivative widths, '
              '`hist`: plot velocity histograms, '
              '`autocorr`: plot <vv> autocorrelation'))
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

ls = {'o': '-', 'x': '-.', 'y': ':', 'par': '-.', 'perp': ':', 'etapar': '--'}
cs = {'mean': 'r', 'var': 'g', 'std': 'b', 'skew': 'm', 'kurt': 'k'}
texlabel = {'o': r'$\xi$', 'x': '$v_x$', 'y': '$v_y$', 'par': r'$v_\parallel$',
            'perp': r'$v_\perp$', 'etapar': r'$\eta_\parallel$'}
englabel = {'o': 'rotation', 'x': 'x (lab)', 'y': 'y (lab)',
            'par': 'longitudinal', 'perp': 'transverse', 'etapar': 'longitudinal'}

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
    v = helpy.add_self_view(v, ('x', 'y'), 'xy')
    v = helpy.add_self_view(v, ('par', 'perp'), 'nt')
    return v


def compile_noise(tracksets, width=(0.65,), cat=True, side=1, fps=1):
    vs = {}
    for prefix, tsets in tracksets.iteritems():
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
    return np.concatenate(vs.values()) if cat else vs


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
    print 'skewness:', SK, SK_t
    print 'kurtosis:', KU, KU_t
    if keepdims:
        SK = SK[..., None]
        KU = KU[..., None]
    else:
        SK = float(SK)
        KU = float(KU)
    return {'mean': M, 'var': D, 'std': SE,
            'skew': SK, 'skew_test': float(SK_t.statistic),
            'kurt': KU, 'kurt_test': float(KU_t.statistic)}


def compile_widths(tracksets, widths, side=1, fps=1):
    vs = compile_noise(tracksets, widths, cat=True, side=side, fps=fps)
    stats = {v: get_stats(np.concatenate(pvs.values()))
             for v, pvs in helpy.transpose_dict(vs).items()}
    return stats


def plot_widths(widths, stats, fig, normalize=False):
    for i, s in enumerate(stats['o']):
        ax = fig.add_subplot(len(stats['o']), 1, i+1)
        for v in stats:
            val = stats[v][s]
            if normalize:
                sign = np.sign(val.sum())
                val = sign*val
                val = val/val.max()
                ax.axhline(1, lw=0.5, c='k', ls=':', alpha=0.5)
            ax.plot(widths, val, '.'+ls[v]+cs[s], label=texlabel[v])
        ax.set_title(s)
        ax.margins(y=0.1)
        ax.minorticks_on()
        ax.grid(axis='x', which='both')
        if normalize:
            ax.set_ylim(-0.1, 1.1)
        ax.legend(loc='best')


def plot_hist(a, ax, bins=100, log=True, orient=False,
              label='v', title='', subtitle='', c=vcol):
    if args.verbose:
        print title + subtitle + str(label)
    stats = get_stats(a)
    if isinstance(label, dict):
        label.update(stats)
        label = '\n'.join([r'$\langle {val} \rangle = {mean:.5f}$',
                           r'$\ D_{sub}\  = {var:.5f}$',
                           r'$\ \gamma_1 = {skew:.5f}$',
                           r'$\ \gamma_2 = {kurt:.5f}$',
                           r'$\sigma/\sqrt{{N}} = {std:.5f}$']).format(**label)
    counts, bins, _ = ax.hist(a, bins, range=(bins[0], bins[-1]), label=label,
                              log=log, alpha=0.7, color=c)
    plot_gaussian(stats['mean'], stats['var'], bins, counts.sum(), ax)
    ax.legend(loc='upper left', fontsize='small', frameon=False)
    # ax.set_ylabel('Frequency')
    xlabel = r'$\Delta r \times f/\ell$'
    l, r = ax.set_xlim(bins[0], bins[-1])
    if orient:
        xticks = np.linspace(l, r, 5)
        ax.set_xticks(xticks)
        xticklabels = map(r'${:.2f}\pi$'.format, xticks/pi)
        ax.set_xticklabels(xticklabels, fontsize='small')
        xlabel = r'$\Delta\theta \times f$'
    if log:
        ax.set_ylim(1, 10**int(1 + np.log10(counts.max())))
    ax.set_xlabel(xlabel)
    title = " ".join(filter(None, [title, subtitle]))
    if title:
        ax.set_title(title)
    return stats


def plot_gaussian(M, D, bins, count=1, ax=None):
    ax = ax or plt.gca()
    dx = bins[1] - bins[0]
    var = 2*D
    g = np.exp(-0.5 * (bins-M)**2 / var)
    g /= sqrt(2*pi*var) / (dx*count)
    ax.plot(bins, g, c=pcol, lw=2, zorder=0.5)


def vv_autocorr(vs, normalize=False):
    normalize = normalize and 1
    fields = helpy.vel_dtype.names
    vvs = [corr.autocorr(helpy.consecutive_fields_view(tv, fields),
                         norm=normalize, cumulant=False)
           for pvs in vs.itervalues() for tv in pvs.itervalues()]
    vvs, vv, dvv = helpy.avg_uneven(vvs, weight=True)
    return [np.array(a, order='C').astype('f4').view(helpy.vel_dtype).squeeze()
            for a in vvs, vv, dvv]


def dot_or_multiply(a, b):
    out = a * b
    if out.ndim > 1:
        return np.sqrt(out.sum(-1))
    else:
        return out


def radial_vv_correlation(fpsets, fvsets, side=1, bins=10):
    components = 'o', 'v', 'eta', 'xy'  # 'nt'
    try:
        nbins = len(bins) - 1
    except TypeError:
        nbins = bins
    vv_radial = np.zeros((len(components), nbins), dtype=float)
    vv_counts = np.zeros(nbins, dtype=int)
    correlator = partial(corr.radial_correlation,
                         bins=bins, corrland=dot_or_multiply, do_avg=False)
    for f in fpsets:
        pos = fpsets[f]['xy']/side
        vels = tuple([fvsets[f][k] for k in components])
        if len(pos) < 2:
            continue
        total, counts, bins = correlator(pos, vels)
        vv_radial += total
        vv_counts += counts
    return vv_radial / vv_counts, bins


def command_widths(tsets, compile_args, args, fig=None):
    stats = compile_widths(tsets, **compile_args)
    if fig is None:
        fig = plt.figure(figsize=(8, 12))
    plot_widths(args.width, stats, fig, normalize=args.normalize)
    return fig


def command_autocorr(tsets, args, comps='o par perp etapar', ax=None):
    vs = compile_noise(tsets, args.width, cat=False,
                       side=args.side, fps=args.fps)
    vvs, vv, dvv = vv_autocorr(vs, normalize=args.normalize)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    n = 10
    t = np.arange(n)/args.fps
    for v in comps.split():
        ax.errorbar(t, vv[v][:n], yerr=dvv[v][:n], ls=ls[v],
                    label=' '.join([texlabel[v], englabel[v]]))
    ax.set_xlabel(r'$tf$')
    ax.set_ylabel(r'$\langle v(t) v(0) \rangle$')
    ax.legend(title=r"Velocity Autocorrelation", loc='best', frameon=False)
    return fig, ax


def command_hist(args, meta, compile_args, axes=None):
    args.width = args.width[0]
    helpy.sync_args_meta(args, meta,
                         ['stub', 'gaps', 'width'],
                         ['vel_stub', 'vel_gaps', 'vel_dx_width'],
                         [10, 'interp', 0.65])
    fits = {}
    args.width = [args.width]
    compile_args.update(args.__dict__)
    vs = compile_noise(tsets, args.width, cat=True,
                       side=args.side, fps=args.fps)

    nrows = args.do_orientation + args.do_translation*(args.subtract + 1)
    ncols = args.log + args.lin
    if axes is None:
        fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                                figsize=(5*ncols, 2.5*nrows))
    else:
        fig = axes[0, 0].figure
    irow = 0
    subtitle = args.particle
    bins = np.linspace(-1, 1, 51)
    brange = 0.5
    if args.do_orientation:
        for icol in range(ncols):
            title = ''#Orientation'
            v = 'o'
            label = texlabel[v] + ' ' + englabel[v]
            if args.verbose:
                label = {'val': label, 'sub': 'R'}
            stats = plot_hist(vs[v], axes[irow, icol], bins=bins*pi/2, c=ncol,
                              log=args.log and icol or not args.lin, label=label,
                              orient=True, title=title, subtitle=subtitle)
            fit = helpy.make_fit(func='vo', DR='var', w0='mean')
            fits[fit] = {'DR': float(stats['var']), 'w0': float(stats['mean']),
                         'KU': stats['kurt'], 'SK': stats['skew'],
                         'KT': stats['kurt_test'], 'ST': stats['skew_test']}
        irow += 1
    if args.do_translation:
        title = ''#Parallel & Transverse'
        for icol in range(ncols):
            v = 'perp'
            label = texlabel[v] + ' ' + englabel[v]
            if args.verbose:
                label = {'val': label, 'sub': r'\perp'}
            stats = plot_hist(vs[v], axes[irow, icol], bins=bins*brange,
                              log=args.log and icol or not args.lin, label=label,
                              title=title, subtitle=subtitle, c=ncol)
            fit = helpy.make_fit(func='vt', DT='var')
            fits[fit] = {'DT': float(stats['var']), 'vt': float(stats['mean']),
                         'KU': stats['kurt'], 'SK': stats['skew'],
                         'KT': stats['kurt_test'], 'ST': stats['skew_test']}
            v = 'par'
            label = texlabel[v] + ' ' + englabel[v]
            if args.verbose:
                label = {'val': label, 'sub': r'\parallel'}
            stats = plot_hist(vs[v], axes[irow, icol], bins=bins*brange,
                              log=args.log and icol or not args.lin,
                              label=label, title=title)
            fit = helpy.make_fit(func='vn', v0='mean', DT='var')
            fits[fit] = {'v0': float(stats['mean']), 'DT': float(stats['var']),
                         'KU': stats['kurt'], 'SK': stats['skew'],
                         'KT': stats['kurt_test'], 'ST': stats['skew_test']}
        irow += 1
        if args.subtract:
            for icol in range(ncols):
                v = 'etapar'
                label = texlabel[v] + ' ' + englabel[v]
                if args.verbose:
                    label = {'val': label, 'sub': r'\alpha'}
                plot_hist(np.concatenate([vs[v], vs['perp']]),
                          axes[irow, icol], bins=bins,
                          log=args.log and icol or not args.lin, label=label,
                          title='$v_0$ subtracted', subtitle=subtitle)
            irow += 1

    return fig, fits


if __name__ == '__main__':
    suf = '_TRACKS.npz'
    if '*' in args.prefix or '?' in args.prefix:
        fs = iglob(args.prefix+suf)
    else:
        dirname, prefix = os.path.split(args.prefix)
        dirm = (dirname or '*') + (prefix + '*/')
        basm = prefix.strip('/._')
        fs = iglob(dirm + basm + '*' + suf)
    prefixes = [s[:-len(suf)] for s in fs] or [args.prefix]

    if not (args.log or args.lin):
        args.log = args.lin = True
    helpy.save_log_entry(args.prefix, 'argv')
    meta = helpy.load_meta(args.prefix)
    if args.verbose:
        print 'prefixes:',
        print '\n          '.join(prefixes)
    helpy.sync_args_meta(args, meta,
                         ['side', 'fps'], ['sidelength', 'fps'], [1, 1])
    compile_args = dict(args.__dict__)

    data = {prefix: helpy.load_data(prefix, 'tracks') for prefix in prefixes}
    tsets = {prefix: helpy.load_tracksets(
                data[prefix], min_length=args.stub, verbose=args.verbose,
                run_remove_dupes=args.dupes, run_repair=args.gaps,
                run_track_orient=args.torient)
             for prefix in prefixes}

    if 'widths' in args.command:
        fig = command_widths(tsets, compile_args, args)
    else:
        nrows = args.do_orientation + args.do_translation*(args.subtract+1)
        ncols = args.log + args.lin + (len(args.command) > 1)
        fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                                figsize=(5*ncols, 2.5*nrows))
        if 'hist' in args.command:
            fig, fits = command_hist(args, meta, compile_args, axes)
        if 'autocorr' in args.command:
            i = 0
            if args.do_orientation:
                command_autocorr(tsets, args, 'o', axes[i, -1])
                i += 1
            if args.do_translation:
                command_autocorr(tsets, args, 'etapar perp', axes[i, -1])


if __name__ == '__main__' and args.save:
    savename = os.path.abspath(args.prefix.rstrip('/._?*'))
    helpy.save_meta(savename, meta)
    if 'hist' in args.command:
        helpy.save_fits(savename, fits)
    savename += '_v' + ('corr' if args.autocorr else 'hist')
    if args.suffix:
        savename += '_' + args.suffix.strip('_')
    savename += '.pdf'
    print 'Saving plot to {}'.format(savename)
    fig.savefig(savename)

if __name__ == '__main__' and args.show:
    plt.show()

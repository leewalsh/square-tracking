#!/usr/bin/env python
# encoding: utf-8
"""Calculate and analyze the velocity noise in granular particle motion. Plot
noise distribution and autocorrelation.

Copyright (c) 2015--2017 Lee Walsh, 2015 Sarah Schlossberg, Department of
Physics, University of Massachusetts; all rights reserved.
"""

from __future__ import division

import os
from glob import iglob
from functools import partial
from math import sqrt, exp

import numpy as np
from scipy.stats import skew, kurtosis, skewtest, kurtosistest
import matplotlib.pyplot as plt

import helpy
import correlation as corr
import curve


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description=__doc__)
    arg = parser.add_argument
    arg('prefix', help='Prefix without trial number')
    arg('command', nargs='+', choices=['widths', 'hist', 'autocorr', 'spatial'],
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
    arg('--width', metavar='WID',
        help='Width (in frames) for derivative kernel, may give slice')
    arg('--smooth', metavar='SIG',
        help='Smoothing width (in frames) of velocities, may give slice. '
        'Differs from --width in that this smoothing happens after derivative.')
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

pi = np.pi

ls = {'o': '-', 'x': '-.', 'y': ':', 'par': '--', 'perp': '-.', 'etapar': ':'}
marker = {'o': 'o', 'x': '-', 'y': '|', 'par': '^', 'perp': 'v', 'etapar': '^'}
cs = {'mean': 'r', 'var': 'g', 'std': 'b', 'skew': 'm', 'kurt': 'k',
      'fit': 'k', 'D': 'g',
      'o': plt.cm.PRGn(0.9), 'x': plt.cm.PRGn(0.1), 'y': plt.cm.PRGn(0.1),
      'par': plt.cm.RdBu(0.8), 'etapar': plt.cm.RdBu(0.8),
      'perp': plt.cm.RdBu(0.2)}

texlabel = {'o': r'$\xi$', 'x': '$v_x$', 'y': '$v_y$', 'par': r'$v_\parallel$',
            'perp': r'$\eta_\perp$', 'etapar': r'$\eta_\parallel$'}
englabel = {'o': 'rotation', 'x': 'x (lab)', 'y': 'y (lab)',
            'par': 'longitudinal', 'perp': 'transverse', 'etapar': 'longitudinal'}
labels = 0

def noise_derivatives(tdata, width=(0.65,), smooth=None, side=1, fps=1):
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
    v0 = np.nanmean(v['par'], -1, keepdims=len(shape) > 1)
    v['etax'] = v['x'] - v0*cos
    v['etay'] = v['y'] - v0*sin
    v['eta'] = np.hypot(v['etax'], v['etay'])
    v['etapar'] = v['par'] - v0
    if smooth is not None:
        from scipy.ndimage import correlate1d
        smooth = np.atleast_1d(smooth)
        vsmooth = np.empty((len(smooth),) + shape, v.dtype)
        for si, s in enumerate(smooth):
            kern = curve.gaussian_kernel(s)
            for f in v.dtype.names:
                vsmooth[si][f] = correlate1d(v[f], kern, mode='nearest')
        v = vsmooth

    v = helpy.add_self_view(v, ('x', 'y'), 'xy')
    v = helpy.add_self_view(v, ('par', 'perp'), 'nt')
    return v.T


def compile_noise(tracksets, width=(0.65,), smooth=None, cat=True, side=1, fps=1):
    vs = {}
    for prefix, tsets in tracksets.iteritems():
        vsets = {t: noise_derivatives(ts, width, smooth, side=side, fps=fps)
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


def get_stats(a, stat_axis=-1, keepdims=None, nan_policy='omit'):
    """Compute mean, D_T or D_R, and standard error for a list.
    """
    a = np.asarray(a)
    n = a.shape[stat_axis]
    if keepdims is None:
        keepdims = a.ndim > 1
    M = np.nanmean(a, stat_axis, keepdims=keepdims)
    variance = np.nanvar(a, stat_axis, keepdims=keepdims, ddof=1)
    SE = np.sqrt(variance)/sqrt(n - 1)
    SK = skew(a, stat_axis, nan_policy=nan_policy)
    KU = kurtosis(a, stat_axis, nan_policy=nan_policy)
    SK_t = skewtest(a, stat_axis, nan_policy=nan_policy)
    KU_t = kurtosistest(a, stat_axis, nan_policy=nan_policy)
    if keepdims:
        SK = np.array(SK)[..., None]
        KU = np.array(KU)[..., None]
        SK_t = np.array(SK_t.statistic)[..., None]
        KU_t = np.array(KU_t.statistic)[..., None]
    else:
        SK = np.array(SK)
        KU = np.array(KU)
        SK_t = np.array(SK_t.statistic)
        KU_t = np.array(KU_t.statistic)
    stat = {'mean': M, 'var': variance, 'std': SE,
            'skew': SK, 'skew_test': SK_t, 'kurt': KU, 'kurt_test': KU_t}
    if not keepdims:
        print '\n'.join(['{:>10}: {: .4f}'.format(k, float(v))
                         for k, v in stat.items()])
    return stat


def compile_widths(tracksets, widths, smooths, side=1, fps=1, **kwargs):
    """collect various statistics as a function of derivative kernel width"""
    vs = compile_noise(tracksets, widths, smooths, cat=True, side=side, fps=fps)
    stats = {v: get_stats(vs[v].T) for v in 'o par perp etapar'.split()}
    return stats


def plot_widths(widths, stats, normalize=False, ax=None):
    """plot various statistics as a function of derivative kernel width"""
    statnames = 'mean var D skew kurt'.split()
    naxs = len(statnames)
    if ax is None:
        fig, axs = plt.subplots(figsize=(6, 2*naxs), nrows=naxs, sharex=True)
    else:
        axs = ax
        fig = axs[0].figure
    for s, ax in zip(statnames, axs):
        for v in stats:
            val = stats[v][s]
            if normalize:
                sign = np.sign(val.sum())
                val = sign*val
                #val = val - val.mean()
                val = val/val.max()
                ax.axhline(1, lw=0.5, c='k', ls=':', alpha=0.5)
            ax.plot(widths, val, '.'+ls[v]+cs[s], label=texlabel[v])
        ax.margins(y=0.1)
        ax.minorticks_on()
        ax.grid(axis='x', which='both')
        ax.grid(axis='y', which='major')
        if normalize:
            ax.set_ylim(-0.1, 1.1)
        ax.legend(title=s, loc='best', ncol=2)
    ax.set_xlabel('kernel width (frames)')
    top = axs[0].twiny()
    top.set_xlabel('kernel width (vibrations)')
    fig.tight_layout(h_pad=0)
    top.set_xlim([l/args.fps for l in ax.get_xlim()])
    return fig


def plot_hist(a, ax, bins=100, log=True, orient=False,
              label='v', title='', subtitle='', c=cs['o'], histtype='step'):
    """plot a histogram of a given distribution of velocities"""
    if args.verbose:
        print title + subtitle + str(label)
    stats = get_stats(a)
    if isinstance(label, dict):
        label.update(stats)
        label = '\n'.join([r'$\langle {val} \rangle = {mean:.5f}$',
                           r'$\ D_{sub}\  = {D:.5f}$',
                           r'$\ \gamma_1 = {skew:.5f}$',
                           r'$\ \gamma_2 = {kurt:.5f}$',
                           r'$\sigma/\sqrt{{N}} = {std:.5f}$']).format(**label)
    if label.startswith('long'):
        xlim = bins[1]
        bins = bins + stats['mean']/2
        xlim = xlim, bins[-2]
    else:
        xlim = bins[1], bins[-2]
    counts, bins, _ = ax.hist(a, bins, range=(bins[0], bins[-1]), label=label,
                              log=log, alpha=1 if histtype == 'step' else 0.6,
                              color=c, histtype=histtype, lw=1.5)
    plot_gaussian(stats['mean'], stats['var'], bins, counts.sum(), ax)
    #ax.tick_params(top=False, which='both')
    ax.tick_params(direction='in', which='both')
    #ax.axvline(x=0, color='grey', linestyle='-', linewidth=0.5, zorder=0)

    leg_handles, leg_labels = ax.get_legend_handles_labels()
    ax.legend(leg_handles[::-1], leg_labels[::-1], bbox_to_anchor=(0, 1.02),
              loc='upper left', fontsize='small', frameon=False,
              handlelength=1, handletextpad=0.5, labelspacing=.1,
              borderaxespad=0.2)
    ax.set_ylabel(r'$N(\eta)$', labelpad=2)
    xlabel = r'$\Delta r \, f/s$'
    l, r = ax.set_xlim(xlim)
    if orient:
        #xticks = np.linspace(l, r, 3)
        xticks = np.array([-pi/4, 0, pi/4])
        ax.set_xticks(xticks)
        #xticklabels = map(r'${:.2f}\pi$'.format, xticks/pi)
        xticklabels = [r'$-\pi/4$', r'$0$', r'$\pi/4$']
        ax.set_xticklabels(xticklabels, fontsize='small')
        xlabel = r'$\Delta\theta \, f$'
        ax.set_ylabel(r'$N(\xi)$', labelpad=2)
    helpy.mark_value(
        ax, stats['mean'], r'$v_o$' if label.startswith('long') else '',
        line=dict(color=c, coords='data', linestyle='-',
                  start=0, stop=counts[np.searchsorted(bins, stats['mean'])]),
        annotate=dict(xy=(stats['mean'], 0), xytext=(0, 9), ha='center')
    )
    if log:
        ypowb, ypowt = 0.5, int(1.9 + np.log10(counts.max()))
        ylim = ax.set_ylim(10**ypowb, 10**ypowt - 1)
        yticks = 10**np.arange(int(ypowb), ypowt)
        yticks_minor = (yticks * np.arange(2, 10)[:, None]).flatten()
        ax.set_yticks(yticks[yticks >= ylim[0]])
        ax.set_yticks(yticks_minor[yticks_minor >= ylim[0]], minor=True)
    ax.set_xlabel(xlabel, labelpad=2)
    title = " ".join(filter(None, [title, subtitle]))
    if title:
        ax.set_title(title)
    return stats


def plot_gaussian(M, var, bins, count=1, ax=None, show_var=False):
    ax = ax or plt.gca()
    dx = bins[1] - bins[0]
    g = np.exp(-0.5 * (bins-M)**2 / var)
    a = sqrt(2*pi*var) / (dx*count)
    g /= a
    if show_var:
        varx, vary = sqrt(var), np.exp(-0.5)/a
        print 'variance arrow', varx, vary
        for x in (M-varx, M+varx):
            ax.annotate(r'$\sigma$', xytext=(M, vary), ha='center', va='center',
                        xy=(x, vary), arrowprops=dict(arrowstyle="->"))

    ax.plot(bins, g, c=cs['fit'], lw=1, zorder=0.5)


def vv_autocorr(vs, normalize=False):
    fields = helpy.vel_dtype.names
    vvs = [corr.autocorr(helpy.consecutive_fields_view(tv, fields),
                         norm=normalize, cumulant=True)
           for pvs in vs.itervalues() for tv in pvs.itervalues()]
    vvs, vv, dvvn, dvv, vvn, vvi = helpy.avg_uneven(vvs, weight=True, ret_all=1)
    return [np.array(a, order='C').astype('f4').view(helpy.vel_dtype).squeeze()
            for a in vvs, vv, dvv]


def dot_or_multiply(a, b):
    out = a * b
    if out.ndim > 1:
        return np.sqrt(out.sum(-1))
    else:
        return out


def vv_rad_corr(fpsets, fvsets, bins=10, comps='o x y etapar perp'):
    """Autocorrelate the spatial velocity "field"

    Parameters
        fpsets: framesets of positional (track) data
        fvsets: framesets of velocity data
        bins:   number of bins or bin edges
        comps:  list of strings or single space-separated string
                of fields from helpy.vel_dtype to calculate.

    Returns
        vv_rad: the autocorrelations as a function of radial separation
            shape is (N comps, N bins)
        bins:   bin edges
    """
    if isinstance(comps, basestring):
        comps = comps.split()

    # Masking is a pain because each component is valid over a different range.
    # - Positional-only velocities are valid everywhere,
    # - body-frame translational velocities are valid wherever orientation is
    # - orientational velocity is valid only where orientation is known, with a
    #   buffer the width of the derivative kernel (which spreads out the nans).
    # So we split the components into groups based on which mask must be used.
    vmasks = {vm: cs for vm, cs in
              [(None, [c for c in comps if c in 'xyv']),
               ('v', [c for c in comps if c in 'o']),
               ('p', [c for c in comps if c in 'par perp eta etax etay etapar'])
               ] if cs}
    vmask_arr = {'p': fpsets, 'v': fvsets}
    try:
        nbins = len(bins) - 1 or len(bins[0]) - 1
    except TypeError:
        nbins = bins
    corr_args = dict(bins=bins, correland=dot_or_multiply, do_avg=False)

    vv_tot = {vm: np.zeros((len(vmasks[vm]), nbins), dtype=float)
              for vm in vmasks}
    vv_count = {vm: np.zeros(nbins, dtype=int)
                for vm in vmasks}
    for f in fpsets:
        if len(fpsets[f]) < 2:
            continue
        for vm, cs in vmasks.iteritems():
            if vm is None:
                xy = fpsets[f]['xy']
                fvset = fvsets[f]
            else:
                vmi = np.where(np.isfinite(vmask_arr[vm][f]['o']))
                xy = fpsets[f][vmi]['xy']
                fvset = fvsets[f][vmi]
            if len(fvset) < 2:
                continue
            vel = tuple([fvset[c] for c in cs])
            total, counts, bins = corr.radial_correlation(xy, vel, **corr_args)
            vv_tot[vm] += total
            vv_count[vm] += counts
    # divide by counts to get mean, store as dict by component
    vv_rad_comps = {c: (v, vv_count[vm]) for vm, cs in vmasks.iteritems()
                    for c, v in zip(cs, vv_tot[vm]/vv_count[vm])}
    # restore order from comps
    vv_rad, counts = zip(*[vv_rad_comps[c] for c in comps])
    return vv_rad, counts, bins


def command_spatial(tsets, args, do_absolute=False, do_radial=False):
    """Run spatial velocity autocorrelation plotting script"""
    fig, ax = plt.subplots(figsize=(3.5, 2.675))
    vv_self_sq = []
    vv_self_mn = []
    vv_rads = []
    v_means = []
    vv_counts = []
    comps = 'o x y etapar perp'.split()
    binsize, rmax = sqrt(2)/3, 5*sqrt(2)
    sbins = np.arange(sqrt(2) - 2*binsize, rmax + binsize, binsize)
    for p in tsets:
        meta = helpy.load_meta(p)
        side = meta['sidelength']
        fps = meta['fps']

        vsets = compile_noise({p: tsets[p]}, cat=False, side=side, fps=fps)
        fsets, fvsets = helpy.load_framesets((tsets[p], vsets[p]))
        tdata = np.concatenate([fsets[f] for f in sorted(fsets)])
        vdata = np.concatenate([fvsets[f] for f in sorted(fvsets)])

        vdata_plain = vdata.view((helpy.vel_dtype[0], len(helpy.vel_dtype)))
        vv_self_sq.append(np.nanmean(vdata_plain**2, 0).view(helpy.vel_dtype))
        vv_self_mn.append(np.nanmean(vdata_plain, 0).view(helpy.vel_dtype))

        if do_absolute:
            xy = tdata['xy'] - meta['boundary'][:2]
            xycount, bins = np.histogramdd(xy, bins=50)
            # extent = bins[1][0], bins[1][-1], bins[0][0], bins[0][-1]
            v_mean = {k: np.histogramdd(xy, bins, weights=vdata[k])[0]/xycount
                      for k in helpy.vel_dtype.names}
            v_means.append(v_mean)

        if do_radial:
            vv_rad, cnt, bins = vv_rad_corr(fsets, fvsets, side*sbins[None])
            vv_rads.append(vv_rad)
            vv_counts.append(cnt)

    vv_self_sq = np.concatenate(vv_self_sq)
    vv_self_mn = np.concatenate(vv_self_mn)
    vv_counts = np.array(vv_counts)
    vv_rads = np.array(vv_rads)
    vv_rad_mean = np.nansum(vv_rads*vv_counts, 0)/np.nansum(vv_counts, 0)
    for vv_rad_comp, comp in zip(vv_rad_mean, comps):
        print comp,
        print 'self magnitude, squared:', np.nanmean(vv_self_sq[comp]),
        print 'mean:', np.nanmean(vv_self_mn[comp])
        curve.bin_plot(sbins, vv_rad_comp/np.nanmean(vv_self_sq[comp]), ax,
                       label=texlabel[comp], linestyle=ls[comp], c=cs[comp])
    helpy.mark_value(ax, sqrt(2), 'steric\ncontact',
                     annotate=dict(xy=(sqrt(2), 0.68)))
    ax.legend(loc='best', frameon=False)
    ax.set_xlabel(r'$|\vec r_i - \vec r_j| / s$')
    ax.set_ylabel(r'$\langle v(\vec r_i) \, v(\vec r_j) \rangle'
                  r'/ \langle v^2 \rangle$')
    ax.set_ylim(None, ax.get_ylim()[1]*1.4)
    return fig, ax, vv_rad_mean


def command_widths(tsets, compile_args, args):
    """Run width-comparison plotting script"""
    widths = helpy.parse_slice(args.width or (.25, 1, .05), index_array=True)
    smooths = helpy.parse_slice(args.smooth or (.25, 1, .05), index_array=True)
    nwidth, nsmooth = len(widths), len(smooths)
    stats = compile_widths(tsets, widths, smooths, **compile_args)
    # stats is a nested dict of stats arrays
    #     stats[vcomp][statname].shape == (nsmooth, nwidth, 1)
    # for vcomp in ('par', 'perp', 'o', ...)
    # and statname in ('mean', 'var', 'D', ...)
    vcomps = stats.keys()
    statnames = stats[vcomps[0]].keys()

    if nwidth > 1 and nsmooth > 1:
        fig, axs = plt.subplots(figsize=(4.8, 6.4), ncols=2, nrows=5,
                                sharex='col', sharey='row')
        axl, axr = axs.T
    else:
        axl, axr = None, None
    # slice the array at the middle smoothing value to plot vs width:
    stats_width = {v: {s: stats[v][s][nsmooth//2, :, 0]
                       for s in statnames} for v in vcomps}
    print "Plot vs derivative width, with smoothing", smooths[nsmooth//2]
    f_width = plot_widths(widths, stats_width,
                          normalize=args.normalize, ax=axl)
    f_width.suptitle('Various derivative widths, smoothing at: {}'.format(
        smooths[nsmooth//2]))
    f_width.tight_layout()

    # slice the array at the middle width value to plot vs smoothing:
    stats_smooth = {v: {s: stats[v][s][:, nwidth//2, 0]
                        for s in statnames} for v in vcomps}
    print "Plot vs smoothing width, with derivative width", widths[nwidth//2]
    f_smooth = plot_widths(smooths, stats_smooth,
                           normalize=args.normalize, ax=axr)
    f_smooth.suptitle('Various smoothing widths, derivative kernel: {}'.format(
        widths[nwidth//2]))
    f_smooth.tight_layout()
    return stats, f_width, f_smooth


def command_autocorr(tsets, args, comps='o par perp etapar', ax=None, markt=''):
    """Run the velocity autocorrelation plotting script"""
    width = helpy.parse_slice(args.width, index_array=True)
    vs = compile_noise(tsets, width, cat=False,
                       side=args.side, fps=args.fps)
    vvs, vv, dvv = vv_autocorr(vs, normalize=args.normalize)
    fits = {}
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    n = 18
    t = np.arange(n)/args.fps
    vmax = args.normalize or max(vv[v][0] for v in comps.split())
    ax.set_ylim(-0.05*vmax, 1.05*vmax)
    ax.set_xlim(-0.2, t[-1])
    for v in comps.split():
        ax.errorbar(t, vv[v][:n], yerr=dvv[v][:n], ls=ls[v], marker=marker[v],
                    linewidth=1, markersize=4, color=cs[v], label=texlabel[v])
        final = vv[v][n:2*n].mean()
        #init = vv[v][0]
        vvnormed = (vv[v][:n] - final)#/(1 - final/init)
        init = vvnormed[0]
        vvtime = curve.decay_scale(vvnormed, t, method='int',
                                   smooth='', rectify=False)
        print v, 'autocorr time:', vvtime, final
        if v == 'o':
            source = 'vac-final' if final else 'vac'
            if args.normalize:
                fit = helpy.make_fit(func='oo', TR=source)
                fits[fit] = {'TR': vvtime}
            else:
                fit = helpy.make_fit(func='oo', DR=source)
                fits[fit] = {'DR': vvtime*init}
                print 'D = mag*integral =', init, '*', vvtime, '=', vvtime*init
                ax.annotate(r'$\langle \xi^2 \rangle = {:.4f}$'.format(init),
                            xy=(0, init),
                            xytext=(10, 0), textcoords='offset points',
                            ha='left', va='center',
                            arrowprops=dict(arrowstyle='->', lw=0.5))
        if markt:
            markstyle = dict(lw=0.5, colors=cs[v], linestyles='-', zorder=0.1)
            vv_at_time = np.interp(vvtime, t, vv[v][:n])
            ax.vlines(vvtime, ax.get_ylim()[0], vv_at_time, **markstyle)
            ax.hlines(vv_at_time, ax.get_xlim()[0], vvtime, **markstyle)
            ax.annotate(r'$\tau = {:.2f}$'.format(vvtime),
                        xy=(vvtime, vv_at_time),
                        xytext=(10, 20), textcoords='offset points',
                        ha='left', va='baseline',
                        arrowprops=dict(arrowstyle='->', lw=0.5))

    ax.tick_params(direction='in', which='both')
    ax.set_xticks(np.arange(0, t[-1], t[-1]//10 + 1).astype(int))

    ax.set_xlabel(r'$t \, f$', labelpad=2)
    ylabel = r'$\langle {0}(t) \, {0}(0) \rangle$'.format(
        texlabel.get(comps, r'\eta').strip('$'))
    ylabelpad = 2
    if args.normalize and not markt:
        ax.set_yticks([0., 1.])
        ylabelpad = -4
    ax.set_ylabel(ylabel, labelpad=ylabelpad)

    leg_title = r"Velocity Autocorrelation"*labels
    leg_handles, leg_labels = ax.get_legend_handles_labels()
    # remove errorbars (has_yerr=False), lines (handlelength=0) from legend keys
    for leg_handle in leg_handles:
        leg_handle.has_yerr = False
    ax.legend(leg_handles, leg_labels, title=leg_title, loc='best',
              numpoints=1, markerfirst=False, handlelength=0,
              frameon=False, fontsize='small')
    return fig, fits


def command_hist(args, meta, compile_args, axes=None):
    """Run the velocity histogram plotting script"""
    helpy.sync_args_meta(args, meta,
                         ['stub', 'gaps', 'width'],
                         ['vel_stub', 'vel_gaps', 'vel_dx_width'],
                         [10, 'interp', 0.65])
    hist_fits = {}
    width = helpy.parse_slice(args.width, index_array=True)
    compile_args.update(args.__dict__)
    vs = compile_noise(tsets, width, cat=True,
                       side=args.side, fps=args.fps)

    dt = 2 * sqrt(3) * width / args.fps
    nrows = args.do_orientation + args.do_translation*(args.subtract + 1)
    ncols = args.log + args.lin
    if axes is None:
        fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                                 figsize=(5*ncols, 2.5*nrows))
    else:
        fig = axes[0, 0].figure
    irow = 0
    subtitle = args.particle
    bins = np.linspace(-1, 1, 26)
    brange = 0.7
    if args.do_orientation:
        for icol in range(ncols):
            title = ''#Orientation'
            v = 'o'
            label = englabel[v]
            if args.verbose:
                label = {'val': label, 'sub': 'R'}
            stats = plot_hist(
                vs[v], axes[irow, icol], bins=bins*pi/3, c=cs[v],
                log=args.log and icol or not args.lin, label=label,
                orient=True, title=title, subtitle=subtitle)
            D_R = 0.5*float(stats['var'])*dt
            fit = helpy.make_fit(func='vo', TR=None, DR='var*dt', w0='mean')
            hist_fits[fit] = {
                'DR': D_R, 'w0': float(stats['mean']),
                'KU': stats['kurt'], 'SK': stats['skew'],
                'KT': stats['kurt_test'], 'ST': stats['skew_test']}
        irow += 1
    if args.do_translation:
        title = ''#Parallel & Transverse'
        for icol in range(ncols):
            v = 'perp'
            label = englabel[v] + r' $\perp$'
            if args.verbose:
                label = {'val': label, 'sub': r'\perp'}
            stats = plot_hist(
                vs[v], axes[irow, icol], bins=bins*brange,
                log=args.log and icol or not args.lin,
                label=label, title=title, subtitle=subtitle, c=cs[v])
            fit = helpy.make_fit(func='vt', DT='var')
            hist_fits[fit] = {
                'DT': 0.5*float(stats['var'])*dt, 'vt': float(stats['mean']),
                'KU': stats['kurt'], 'SK': stats['skew'],
                'KT': stats['kurt_test'], 'ST': stats['skew_test']}
            v = 'par'
            label = englabel[v] + r' $\parallel$'
            if args.verbose:
                label = {'val': label, 'sub': r'\parallel'}
            stats = plot_hist(vs[v], axes[irow, icol], bins=bins*brange,
                              log=args.log and icol or not args.lin,
                              label=label, title=title, c=cs[v])
            fit = helpy.make_fit(func='vn', v0='mean', DT='var')
            hist_fits[fit] = {
                'v0': float(stats['mean']), 'DT': 0.5*float(stats['var'])*dt,
                'KU': stats['kurt'], 'SK': stats['skew'],
                'KT': stats['kurt_test'], 'ST': stats['skew_test']}
        irow += 1
        if args.subtract:
            for icol in range(ncols):
                v = 'etapar'
                label = englabel[v]
                if args.verbose:
                    label = {'val': label, 'sub': r'\alpha'}
                plot_hist(np.concatenate([vs[v], vs['perp']]),
                          axes[irow, icol], bins=bins,
                          log=args.log and icol or not args.lin, label=label,
                          title='$v_0$ subtracted', subtitle=subtitle)
            irow += 1

    return fig, hist_fits


def find_data(prefix, verbose=False):
    """load data for one or more prefixes into a dict"""
    if prefix == 'simulate':
        import simulate as sim
        spar = {'DR': 1/21, 'v0': 0.3678, 'DT': 0.01,
                'fps': args.fps, 'side': args.side, 'size': 1000}
        print spar
        sdata = [sim.SimTrack(num=t, **spar) for t in xrange(1, 1001)]
        sdata = np.concatenate([sdatum.track for sdatum in sdata])
        sdata['id'] = np.arange(len(sdata))
        return {'simulate': sdata}

    suf = '_TRACKS.npz'
    if '*' in prefix or '?' in prefix:
        fs = iglob(prefix+suf)
    else:
        dirname, basename = os.path.split(prefix)
        dirm = (dirname or '*') + (basename + '*/')
        basm = basename.strip('/._')
        fs = iglob(dirm + basm + '*' + suf)
    prefixes = [s[:-len(suf)] for s in fs] or [prefix]
    if verbose:
        print 'prefixes:',
        print '\n          '.join(prefixes)
    return {prefix: helpy.load_data(prefix, 'tracks') for prefix in prefixes}

if __name__ == '__main__':
    helpy.save_log_entry(args.prefix, 'argv')
    meta = helpy.load_meta(args.prefix)
    helpy.sync_args_meta(args, meta,
                         ['side', 'fps'], ['sidelength', 'fps'], [1, 1])
    if not (args.log or args.lin):
        args.log = args.lin = True
    compile_args = dict(args.__dict__)
    data = find_data(args.prefix, args.verbose)
    tsets = {
        prefix: helpy.load_tracksets(
            data[prefix], min_length=args.stub, verbose=args.verbose,
            run_remove_dupes=args.dupes, run_repair=args.gaps,
            run_track_orient=args.torient)
        for prefix in data}

    fits = {}
    rcParams_for_context = {'text.usetex': args.save or args.show}
    with plt.rc_context(rc=rcParams_for_context):
        print rcParams_for_context
        if 'widths' in args.command:
            stats, f_width, f_smooth = command_widths(tsets, compile_args, args)
        elif 'spatial' in args.command:
            fig, ax, vv_rad = command_spatial(
                tsets, args, do_absolute=False, do_radial=True)
        else:
            nrows = args.do_orientation + args.do_translation*(args.subtract+1)
            ncols = len(args.command)
            if 'hist' in args.command and args.log and args.lin:
                ncols += 1
            fig, axes = plt.subplots(
                nrows, ncols, squeeze=False,
                figsize=(3.5*ncols, 3.0*nrows) if labels
                else (3.5, 3.0*nrows/ncols),
                gridspec_kw={'wspace': 0.2, 'hspace': 0.4})
            if 'hist' in args.command:
                fig, new_fits = command_hist(args, meta, compile_args, axes)
                fits.update(new_fits)
            if 'autocorr' in args.command:
                i = 0
                if args.do_orientation:
                    fig, new_fits = command_autocorr(tsets, args,
                                                     'o', axes[i, -1])
                    fits.update(new_fits)
                    i += 1
                if args.do_translation:
                    fig, new_fits = command_autocorr(tsets, args,
                                                     'etapar perp', axes[i, -1])
                    fits.update(new_fits)

        if args.save:
            savename = os.path.abspath(args.prefix.rstrip('/._?*'))
            helpy.save_meta(savename, meta)
            if fits:
                print 'saving fits'
                print fits
                helpy.save_fits(savename, fits)
            savename += '_velocity_' + '_'.join(args.command)
            if args.suffix:
                savename += '_' + args.suffix.strip('_')
            savename += '.pdf'
            print 'Saving plot to {}'.format(savename)
            fig.savefig(savename, bbox_inches='tight', pad_inches=0.05)

        if args.show:
            plt.show()
        else:
            plt.close('all')

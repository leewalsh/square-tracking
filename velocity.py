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
    arg('--sets', type=int, default=1, metavar='N', nargs='?', const=0,
            help='Number of sets')
    arg('--width', type=float, default=0.75, metavar='W', nargs='?', const=-.5,
            help='Smoothing width for derivative')
    arg('--particle', type=str, default='', metavar='NAME', help='Particle type name')
    arg('--save', type=str, nargs='?', const='velocity', default='', help='Save figure?')
    arg('--lin', action='store_false', dest='log', help='Plot on a linear scale?')
    arg('--log', action='store_true', help='Plot on a log scale?')
    arg('--dupes', action='store_true', help='Remove duplicates from tracks')
    arg('--normalize', action='store_true', help='Normalize by max?')
    arg('--autocorr', action='store_true', help='Plot the <vv> autocorrelation?')
    arg('--decimate', type=int, default=0, metavar='MOD', nargs='?', const=1,
            help='Decimate by vibration phase? Default: numerator of simplified fps')
    arg('--untrackorient', action='store_false', dest='torient', help='Untracked raw orientation?')
    arg('--interp', action='store_true', dest='interp', help='interpolate gaps?')
    arg('--minlen', type=int, default=10, help='Minimum track length. Default: %(default)s')
    arg('--nosubtract', action='store_false', dest='subtract', help="Don't subtract v0?")
    arg('-s', '--side', type=float, default=17, help='Particle size in pixels, '
        'for unit normalization. Default: %(default)s')
    arg('-f', '--fps', type=float, default=2.4, help="Number of frames per second "
        "(or per shake) for unit normalization. Default: %(default)s")
    arg('-v', '--verbose', action='count', help="Be verbose")
    args = p.parse_args()

import os
from collections import defaultdict
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import helpy, tracks, correlation as corr

def noise_derivatives(tdata, width=1, side=1, fps=1, xy=False,
                      do_orientation=True, do_translation=True, subtract=True):
    x = tdata['f']/fps
    ret = {}
    ws = [width] if np.isscalar(width) else width
    if do_orientation:
        ret['o'] = np.array([helpy.der(tdata['o'], x=x, iwidth=w)
                             for w in ws]).squeeze()
    if do_translation:
        cos, sin = np.cos(tdata['o']), np.sin(tdata['o'])
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

def compile_noise(prefixes, vs, width=3, side=1, fps=1, cat=True, decimate=False,
                  do_orientation=True, do_translation=True, subtract=True,
                  minlen=10, torient=True, interp=True, dupes=False, **ignored):
    if np.isscalar(prefixes):
        prefixes = [prefixes]
    if decimate and len(prefixes)>1:
        raise ValueError, "Cannot decimate more than one dataset"
    if int(decimate)==1:
        from fractions import Fraction
        decimate = Fraction(str(fps)).numerator
    for prefix in prefixes:
        if args.verbose:
            print "Loading data for", prefix
        data = helpy.load_data(prefix, 'tracks')
        if dupes:
            data['t'] = tracks.remove_duplicates(data['t'], data)
        tracksets = helpy.load_tracksets(data, min_length=minlen,
                run_track_orient=torient, run_fill_gaps=interp)
        for track in tracksets:
            tdata = tracksets[track]
            velocities = noise_derivatives(tdata, width=width,
                    side=side, fps=fps, do_orientation=do_orientation,
                    do_translation=do_translation, subtract=subtract)
            for v in velocities:
                if decimate:
                    # reshape last axis from (tracklen) to (decimate, tracklen/decimate)
                    # but make sure it starts from phase 0 first, and ends on phase decimate-1
                    veloc = velocities[v]
                    frame = tdata['f']
                    assert len(frame)==len(veloc), '{} frames, {} velocities'.format(len(frame), len(veloc))
                    phase = frame % decimate
                    start = np.where(phase[:2*decimate])[-1][0]
                    exend = (len(frame) - start) % decimate
                    frame = frame[start:-exend]
                    phase = phase[start:-exend]
                    veloc = veloc[start:-exend]
                    vdiff = np.diff(frame)-1
                    fgaps = vdiff.nonzero()[0]
                    fgaps = np.repeat(fgaps, vdiff[fgaps]) + 1
                    veloc = np.insert(veloc, fgaps, np.nan)
                    shape = veloc.shape
                    veloc.resize(shape[:-1] + (shape[-1]//decimate, decimate))
                    veloc = np.swapaxes(veloc, -1, -2)
                    vs[v].append(veloc)
                else:
                    vs[v].append(velocities[v])
    if cat:
        for v in vs:
            vs[v] = np.concatenate(vs[v], -1)
    return len(tracksets)

def get_stats(a):
    #Computes mean, D_T or D_R, and standard error for a list.
    a = np.asarray(a)
    if a.ndim==1:
        a = a[np.isfinite(a)]
    else:
        print 'ndims =', a.ndim
    n = np.sum(np.isfinite(a), -1)# if np.all(np.isfinite(a)) else a.shape[-1]
    M = np.nanmean(a, -1, keepdims=a.ndim>1)
    c = np.nan_to_num(a - M)
    variance = np.einsum('...j,...j->...', c, c)/n
    D = 0.5*variance
    SE = np.sqrt(variance/n)
    return M, D, SE

def compile_widths(width, prefixes, **compile_args):
    stats = {v: {s: np.empty_like(width)
                 for s in 'mean var stderr'.split()}
             for v in 'o par perp etapar'.split()}
    compile_args['width'] = width
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
        ax.minorticks_on()
        ax.grid(axis='x', which='both')
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

if __name__=='__main__':
    compile_args = dict(args.__dict__)
    full_prefix = compile_args.pop('prefix')
    dirname, prefix = os.path.split(full_prefix)
    if '*' in full_prefix or '?' in full_prefix:
        from glob import iglob
        suf = '_TRACKS.npz'
        prefixes = [ s[:-len(suf)] for s in iglob(full_prefix+suf) ]
    elif args.sets:
        prefixes = ["{}{i}_d/{}{i}".format(full_prefix, prefix, i=i+1)
                    for i in xrange(args.sets)]
    else:
        from glob import iglob
        depth = 1
        dirm = (dirname or '*') + (prefix + '*/')*depth
        basm = prefix.strip('/._')
        endm = '*_TRACKS.npz'
        prefixes = [p[:1-len(endm)] for p in iglob(dirm*depth + basm + endm)] or full_prefix
    if args.verbose:
        print 'using'
        print '\n'.join([prefixes] if np.isscalar(prefixes) else prefixes)

    label = {'o': r'$\xi$', 'par': r'$v_\parallel$', 'perp': r'$v_\perp$',
             'etapar': r'$\eta_\parallel$'}
    ls = {'o': '-', 'par': '-.', 'perp': ':', 'etapar': '--'}
    cs = {'mean': 'r', 'var': 'g', 'stderr': 'b'}
    if args.width < 0:
        widths = np.arange(0, 1.5, -args.width) - args.width
        widths = np.append(widths, [1.5, 2, 2.5, 3])
        stats = compile_widths(widths, prefixes, **compile_args)
        plot_widths(widths, stats, normalize=args.normalize)
    elif args.autocorr:
        vvs = vv_autocorr(prefixes, corrlen=10*args.fps, **compile_args)
        plt.figure()
        for v in vvs:
            vv, dvv = vvs[v]
            t = np.arange(len(vv))/args.fps
            plt.errorbar(t, vv , yerr=dvv, label=label[v], ls=ls[v])
        plt.title(r"Velocity Autocorrelation $\langle v(t) v(0) \rangle$")
        plt.legend(loc='best')
    elif args.decimate:
        if int(args.decimate)==1:
            from fractions import Fraction
            args.decimate = Fraction(str(args.fps)).numerator
        print 'decimating by', args.decimate
        stats = compile_widths(prefixes=prefixes, cat=True, **compile_args)
        plot_widths(np.arange(args.decimate), stats, normalize=args.normalize)
    else:
        vs = defaultdict(list)
        trackcount = compile_noise(prefixes, vs, **compile_args)

        nax = sum([args.do_orientation, args.do_translation, args.do_translation and args.subtract])
        axi = 1
        subtitle = args.particle or prefix.strip('/._')
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
        savename = '.'.join([os.path.abspath(full_prefix.rstrip('/._?*')), args.save, 'pdf'])
        print 'Saving plot to {}'.format(savename)
        plt.savefig(savename)
    else:
        plt.show()

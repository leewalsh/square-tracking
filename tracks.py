#!/usr/bin/env python
# encoding: utf-8

from __future__ import division

import sys
import itertools as it
from functools import partial
from collections import defaultdict
from math import sqrt, log, exp

import numpy as np
from lmfit import Model
from cycler import cycler

import helpy
import correlation as corr
import curve

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    arg = parser.add_argument
    arg('prefix', metavar='PRE', help="Filename prefix with full or relative "
        "path (<prefix>_POSITIONS.npz, <prefix>_CORNER_POSITIONS.npz, etc)")
    arg('-t', '--track', action='store_true', help='Connect the dots!')
    arg('-n', '--number', type=int, default=0, help='Total number of tracks to '
        'keep. Default = 0 keeps all, -1 attempts to count particles')
    arg('-o', '--orient', action='store_true', help='Find the orientations')
    arg('--ncorners', type=int, default=2,
        help='Number of corner dots per particle. default = 2')
    arg('-r', '--rcorner', type=float,
        help='Distance between corner and center dot, in pixels.')
    arg('--drcorner', type=float,
        help='Allowed error in r/rcorner, in pixels. Default is sqrt(r)')
    arg('--angsep', type=float, help='Angular separation between corner dots, '
        'assumed degrees if angsep > pi')
    arg('--dangsep', type=float,
        help='Allowed error in r/rcorner, in pixels. Default is sqrt(r)')
    arg('-l', '--load', action='store_true', help='Create and save structured '
        'array from prefix[_CORNER]_POSITIONS.txt file')
    arg('-c', '--corner', action='store_true', help='Load corners not centers')
    arg('-k', '--check', action='store_true', help='Plot an animation of '
        'detected positions, orientations, and tracks to check their quality')
    arg('-i', '--slice', nargs='?', const=True, help='Slice to limit frames')
    arg('-p', '--plottracks', action='store_true', help='Plot the tracks')
    arg('--noshow', action='store_false', dest='show',
        help="Don't show figures (just save them)")
    arg('--nosave', action='store_false', dest='save',
        help="Don't save outputs or figures")
    arg('--noplot', action='store_false', dest='plot',
        help="Don't generate (fewer, not none at this point) figures")
    arg('--maxdist', type=float, help="maximum single-frame travel distance in "
        "pixels for track. default = side/fps if defined")
    arg('--giveup', type=int, default=10,
        help="maximum number of frames in track gap. default = 10")
    arg('-d', '--msd', action='store_true', help='Calculate the MSD')
    arg('--plotmsd', action='store_true', help='Plot MSD (requires msd first)')
    arg('-s', '--side', type=float,
        help='Particle size in pixels, for unit normalization')
    arg('-f', '--fps', type=float,
        help="Number of frames per shake (or second) for unit normalization")
    arg('--dt0', type=int, default=1, help='Stepsize for time-averaging of a '
        'single track at different time starting points. default = 1')
    arg('--dtau', type=int, default=1, help='Stepsize for values of tau at '
        'which to calculate MSD(tau). default = 1')
    arg('--msdvec', choices=['displacement', 'progression', 'diversion'],
        default='displacement', help='msd vector component')
    arg('--dot', action='store_true', help='combine components of correlations')
    arg('--killflat', type=int, default=0,
        help='Minimum growth factor for a single MSD track for inclusion')
    arg('--killjump', type=int, default=100000,
        help='Maximum initial jump for a single MSD track at first time step')
    arg('--stub', type=int, default=10, help='Minimum length (in frames) of a '
        'track for it to be included. default = 10')
    arg('--retire', type=int, default=None, help='Maximum length (in frames) of'
        ' a track to be kept. default = None keeps all')
    arg('--reverse', action='store_true', help='Optionally reverse track time.')
    arg('-g', '--gaps', choices=['interp', 'nans', 'leave'], default='interp',
        nargs='?', const='nans', help="Gap handling: choose from %(choices)s. "
        "default is %(default)s, `-g` or `--gaps` alone gives %(const)s")
    arg('--singletracks', type=int, nargs='*', help='Only plot these tracks')
    arg('--showtracks', action='store_true', help='Show individual tracks')
    arg('--cut', action='store_true', help='Cut individual tracks at boundary')
    arg('--boundary', type=float, nargs=3, default=False,
        metavar=('X0', 'Y0', 'R'), help='Boundary for track cutting')
    arg('--nn', action='store_true', help='Calculate and plot <nn> correlation')
    arg('--rn', const='full', nargs='?', choices=['full', 'pos', 'max', 'anti'],
        default=False, help='Calculate and plot <rn> correlation')
    arg('--rr', action='store_true', help='Calculate and plot <rr> correlation')
    arg('--fitdr', action='store_true', help='D_R as free parameter in rn fit')
    arg('--fitv0', const=True, nargs='?', default=False,
        help='v_0 as free parameter in MSD fit')
    arg('--fit0', action='store_true', help='include t = 0, 1 in MSD fit?')
    arg('--colored', default=False, const=True, nargs='?', type=float,
        help='fit with colored noise')
    arg('--fixdt', default=True, const=False, nargs='?',
        dest='fitdt', help='D_T as fixed value in MSD fit')
    arg('--fittr', action='store_true', help='tau_R as free parameter in fits')
    arg('--dx', type=float, default=0.25, help='Positional measurement '
        'uncertainty (units are pixels unless SIDE is given and DX < 0.1)')
    arg('--dtheta', type=float, default=0.02,
        help='Angular measurement uncertainty (radians)')
    arg('-z', '--zoom', metavar="ZOOM", type=float,
        help="Factor by which to zoom out (in if ZOOM < 1)")
    arg('--clean', action='store_true', help='Make clean plot for presentation')
    arg('--ax', type=int)
    arg('--fig', type=int)
    arg('--vcol', help='Color for velocity and displacement')
    arg('--pcol', help='Color for parameters and fits')
    arg('--ncol', help='Color for noise and histograms')
    arg('-v', '--verbose', action='count', help='Be verbose, may repeat: -vv')
    arg('-q', '--quiet', action='count', help='Be quiet, subtracts from -v')
    arg('--suffix', type=str, default='', help='Suffix to append to savenames')

    args = parser.parse_args()

    import os.path
    relprefix = args.prefix
    absprefix = os.path.abspath(relprefix)
    readprefix = absprefix
    saveprefix = absprefix
    if args.suffix:
        saveprefix += '_' + args.suffix.strip('_')
    locdir, prefix = os.path.split(absprefix)
    locdir += os.path.sep
    if args.prefix == 'simulate':
        relprefix = absprefix = readprefix = prefix = saveprefix = 'simulate'

    need_plt = any([args.plottracks, args.plotmsd, args.check,
                    args.nn, args.rn, args.rr])
    args.verbose = args.verbose or 0
    args.quiet = args.quiet or 0
    args.verbose = max(0, args.verbose - args.quiet)
    verbose = args.verbose
    if verbose:
        print 'using prefix', prefix
    warnlevel = {0: 'ignore', None: 'ignore', 1: 'warn', 2: 'warn', 3: 'raise'}
    np.seterr(divide=warnlevel[verbose], invalid=warnlevel[verbose])
else:
    verbose = False
    need_plt = True


if need_plt:
    if helpy.gethost() == 'foppl':
        import matplotlib
        matplotlib.use("agg")
    import matplotlib.pyplot as plt
    if __name__ == '__main__':
        args.vcol = args.vcol or plt.cm.PRGn(0.9)   #(1, 0.4, 0)
        args.pcol = args.pcol or 'black'            #(0.25, 0.5, 0)
        args.ncol = args.ncol or plt.cm.PRGn(0.1)   #(0.4, 0.4, 1)
        args.lcol = plt.cm.RdBu(.8)
        args.tcol = plt.cm.RdBu(.2)


sf = helpy.SciFormatter().format

pi = np.pi
twopi = 2*pi
rt2 = sqrt(2)


def find_closest(thisdot, trackids, n=1, maxdist=20., giveup=10, cut=False):
    """recursive function to find nearest dot in previous frame.

    looks further back until it finds the nearest particle
    returns the trackid for that nearest dot, else returns new trackid
    """
    info = 'New track {:5d}, frame {:4d}, n {:2d}, dot {:5d}, from'.format
    frame = thisdot[0]
    if cut is not False and cut[thisdot[-1]]:
        return -1
    if frame < n:  # at (or recursed back to) the first frame
        newtrackid = trackids.max() + 1
        if verbose:
            print info(newtrackid, frame, n, thisdot[-1]),
            print "first frame!"
        return newtrackid
    oldtree = pftrees[frame-n]
    thisdotxy = thisdot[1:3]
    maxdist_n = sqrt(n) * maxdist
    mindist, mini = oldtree.query(thisdotxy, distance_upper_bound=maxdist_n)
    if mindist < maxdist_n:
        # a close one! Is there another dot in the current frame that's closer?
        closest = pfsets[frame-n].item(mini)
        curtree = pftrees[frame]
        closestxy = closest[1:3]
        mindist2, mini2 = curtree.query(closestxy, distance_upper_bound=mindist)
        if mindist2 < mindist:
            # create new trackid to be deleted (or overwritten?)
            newtrackid = trackids.max() + 1
            if verbose:
                print info(newtrackid, frame, n, thisdot[-1]),
                print "dot {} closer to parent {} (track {})".format(
                    pfsets[frame].item(mini2)[-1], closest[-1],
                    trackids[closest[-1]])

            return newtrackid
        if cut is not False and cut[closest[-1]]:
            newtrackid = trackids.max() + 1
            if verbose:
                print info(newtrackid, frame, n, thisdot[-1]),
                print "cutting track", trackids[closest[-1]]
            return newtrackid
        else:
            oldtrackid = trackids[closest[-1]]
            if oldtrackid == -1:
                newtrackid = trackids.max() + 1
                if verbose:
                    print info(newtrackid, frame, n, thisdot[-1]),
                    print "previous track cut"
                return newtrackid
            else:
                return oldtrackid
    elif n < giveup:
        return find_closest(thisdot, trackids, n=n+1,
                            maxdist=maxdist, giveup=giveup, cut=cut)
    else:
        # give up after giveup frames
        newtrackid = trackids.max() + 1
        if verbose:
            print info(newtrackid, frame, n, thisdot[-1]),
            print "passed", giveup, "frames"
        return newtrackid


def find_tracks(pdata, maxdist=20, giveup=10, n=0, stub=0,
                cut=False, boundary=None, margin=0):
    """ Track dots from frame-to-frame, giving each particle a unique and
        persistent id, called the trackid.

        parameters
        ----------
        pdata : the main positions data array
        maxdist : maximal separation in pixels between a particles current
            and previous position. i.e., the maximum distance a particle is
            allowed to travel to be considered part of the same track
            A good choice for this value is the size of one particle.
        giveup : maximal number of frames to recurse over seeking the parent
        n : the number of particles to track, useful if you wish to have one
            track per physical particle. Not useful if you want tracks to be
            cut when the particle hits the boundary
        cut : whether or not to cut tracks (assign a new trackid to the same
            physical particle) when the particle nears or hits the boundary
            if True, it requires either args.boundary or for user to click on
            an image to mark the center and boundary. Particle at the boundary
            (between two tracks) will have track id of -1
        stub : minimal length of a track for it to be kept. trackids of any
            track with length less than `stub` will be set to -1

        returns
        -------
        trackids : an array of length `len(pdata)`, giving the track id number
            for each point in data. Any point not belonging to any track has a
            track id of -1
    """
    from sys import setrecursionlimit, getrecursionlimit
    setrecursionlimit(max(getrecursionlimit(), 2*giveup))

    trackids = -np.ones(pdata.shape, dtype=int)
    if n == -1:
        n = -helpy.mode(pdata['f'], count=True)
        print "Found {n} particles, will use {n} longest tracks".format(n=n)

    if cut:
        boundary = boundary or meta.get('boundary')
        if boundary is None or boundary == [0.]*3:
            bgpath, bgimg, _ = helpy.find_tiffs(
                prefix=relprefix, frames=1, single=True, load=True)
            boundary = helpy.circle_click(bgimg)
            meta['path_to_tiffs'] = bgpath
        x0, y0, R = boundary
        mm = R/101.6             # dish radius R = 4 in = 101.6 mm
        margin = margin or 6*mm  # use 6 mm if margin not specified
        meta.update(boundary=boundary, track_cut_margin=margin)
        rs = np.hypot(pdata['x'] - x0, pdata['y'] - y0)
        cut = rs > R - margin
        print "cutting at boundary", boundary,
        print 'with margin {:.1f} pix = {:.1f} mm'.format(margin, margin/mm)

    print "seeking tracks"
    for i in xrange(len(pdata)):
        # This must remain a simple loop because trackids gets modified
        # and passed into the function with each iteration
        trackids[i] = find_closest(pdata.item(i), trackids,
                                   maxdist=maxdist, giveup=giveup, cut=cut)

    if verbose:
        datalen = len(pdata)
        assert datalen == len(trackids), "too few/many trackids"
        assert np.all(pdata['id'] == np.arange(datalen)), "gap in particle id"

    if n < 0 or stub > 0:
        track_lens = np.bincount(trackids+1)[1:]
        if n < 0:
            stubs = np.argsort(track_lens)[:n]  # all but the longest n
        elif stub > 0:
            stubs = np.where(track_lens < stub)[0]
            if verbose:
                print "removing {} stubs".format(len(stubs))
        isstub = np.in1d(trackids, stubs)
    elif n > 0:
        stubs = np.array([], int)
        isstub = False
    if n or stub > 0:
        if n > 0:
            isstub |= trackids >= n + np.count_nonzero(stubs < n)
        trackids[isstub] = -1
        return trackids


def remove_duplicates(trackids=None, data=None, tracksets=None,
                      target='', inplace=False, verbose=False):
    if tracksets is None:
        target = target or 'trackids'
        tracksets = helpy.load_tracksets(data, trackids, min_length=0)
    elif trackids is None:
        target = target or 'tracksets'
    else:
        target = target or 'trackids'
    rejects = defaultdict(dict)
    for t, tset in tracksets.iteritems():
        fs = tset['f']
        count = np.bincount(fs)
        dup_fs = np.where(count > 1)[0]
        if not len(dup_fs):
            continue
        ftsets = helpy.load_framesets(tset, fs)
        for f in dup_fs:
            prv = fs[np.searchsorted(fs, f, 'left') - 1] if f > fs[0] else None
            nxt = fs[np.searchsorted(fs, f, 'right')] if f < fs[-1] else None
            if nxt is not None and nxt in dup_fs and nxt < fs[-1]:
                nxt = fs[np.searchsorted(fs, nxt, 'right')]
                if nxt is not None and nxt in dup_fs:
                    nxt = None
                    if prv is None:
                        raise RuntimeError(
                            "Duplicate track particles in too many frames in a "
                            "row at frame {} for track {}".format(f, t))
            seps = np.zeros(count[f])
            for g in (prv, nxt):
                if g is None:
                    continue
                if count[g] > 1 and g in rejects[t]:
                    notreject = np.in1d(ftsets[g]['id'], rejects[t][g],
                                        assume_unique=True, invert=True)
                    ftsets[g] = ftsets[g][notreject]
                sepx = ftsets[f]['x'] - ftsets[g]['x']
                sepy = ftsets[f]['y'] - ftsets[g]['y']
                seps += sepx*sepx + sepy*sepy
            rejects[t][f] = ftsets[f][seps > seps.min()]['id']
    if not rejects:
        if verbose:
            print "no duplicate tracks"
        if inplace:
            return
        else:
            return trackids if target == 'trackids' else tracksets
    elif verbose:
        print "repairing {} duplicate tracks".format(len(rejects))
        if verbose > 1:
            for t, r in rejects.iteritems():
                print '\ttrack {}: {} duplicate frames'.format(t, len(r))
                if verbose > 2:
                    print sorted(r)
    if target == 'tracksets':
        if not inplace:
            tracksets = tracksets.copy()
        for t, tr in rejects.iteritems():
            trs = np.concatenate(tr.values())
            tr = np.in1d(tracksets[t]['id'], trs, True, True)
            new = tracksets[t][tr]
            if inplace:
                tracksets[t] = new
        return None if inplace else tracksets
    elif target == 'trackids':
        if not inplace:
            trackids = trackids.copy()
        rejects = np.concatenate([tfr for tr in rejects.itervalues()
                                  for tfr in tr.itervalues()])
        if data is None:
            data_from_tracksets = np.concatenate(tracksets.values())
            if len(data_from_tracksets) != len(trackids):
                raise ValueError("Must provide data to return/modify trackids")
            ids = data_from_tracksets['id']
            ids.sort()
        else:
            ids = data['id']
        rejects = np.searchsorted(ids, rejects)
        trackids[rejects] = -1
        return None if inplace else trackids


def plt_text(*args, **kwargs):
    return [plt.text(*arg, **kwargs) for arg in zip(*args)]


def set_axes_size_inches(ax, axsize, clear=None, tight=None):
    """Resize figure to control size of axes in inches"""
    if clear is not None:
        properties = {'ticks': {'xticks': [], 'yticks': []},
                      'labels': {'xlabel': '', 'ylabel': ''},
                      'title': {'title': ''}}
        ax.set(**{k: v for c in clear for (k, v) in properties[c].iteritems()})
    if isinstance(tight, dict):
        ax.figure.tight_layout(**tight)
    elif tight is True:
        ax.figure.tight_layout()
    elif tight is None or tight is False:
        pass
    elif np.isscalar(tight):
        ax.figure.tight_layout(pad=tight)
    else:
        raise ValueError('unexpected value `{}` for tight'.format(tight))

    portion_of_fig = np.diff(ax.get_position(), axis=0)[0]
    ax.figure.set_size_inches(axsize / portion_of_fig, forward=True)

    return ax.figure.get_size_inches()


def animate_detection(imstack, fsets, fcsets, fosets=None, fisets=None,
                      meta={}, f_nums=None, verbose=False, clean=0):

    global f_idx, f_num, xlim, ylim

    def advance(event):
        global f_idx, f_num, xlim, ylim
        key = event.key
        if verbose:
            print '\tpressed {}'.format(key),
        if key in ('left', 'up'):
            # going back resets to original limits, so fix them here
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            if f_idx >= 1:
                f_idx -= 1
        elif key in ('right', 'down'):
            # save new limits in case we go back and need to fix them
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            if f_idx < f_max - 1:
                f_idx += 1
        elif key == 'b':
            f_idx = 0
        elif key == 'e':
            f_idx = f_max - 1
        elif key == 's':
            savename = raw_input('save pdf to '+os.getcwd()+'/')
            if savename:
                ax.set(title='', xticks=[], yticks=[])
                if not savename.endswith('.pdf'):
                    savename += '.pdf'
                fig.savefig(savename)#, bbox_inches='tight', pad_inches=0)
                print "saved to", savename
        elif key == 'i':
            print title.format(f_num, nos, nts, ncs)
        else:
            plt.close()
            f_idx = -1
        f_num = f_nums[f_idx] if 0 <= f_idx < f_max else -1
        if verbose:
            if f_idx >= 0:
                print 'next up {} ({})'.format(f_idx, f_num),
            else:
                print 'will exit'
            sys.stdout.flush()

    #imstack = (imstack.astype('f8')**2 / 4096.0).round().astype('u2')
    imstack = imstack - np.median(imstack, axis=0)

    side = meta.get('sidelength', 17)
    rc = meta.get('orient_rcorner')
    drc = meta.get('orient_drcorner') or sqrt(rc)
    txtoff = min(rc, side/2)/2

    title = "frame {:5d}\n{:3d} oriented, {:3d} tracked, {:3d} detected"
    ppi = meta['boundary'][-1]/4
    h, w = imstack[0].shape
    actualsize = np.array([w, h]) / ppi
    fig, ax = plt.subplots(figsize=actualsize*1.02)
    p = ax.imshow(imstack[0], cmap='gray')
    xlim, ylim = (0, w), (0, h)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if actualsize is not None:
        print 'plotting actual size {:.2f}x{:.2f} in'.format(*actualsize)
        print '{:d}x{:d} pix {:.2f} ppi'.format(w, h, ppi)
        set_axes_size_inches(ax, actualsize, clear=['title', 'ticks'], tight=0)
    else:
        ax.set_title(title.format(-1, 0, 0, 0))
        fig.tight_layout()
    need_legend = not clean

    if meta.get('track_cut', False):
        bndx, bndy, bndr = meta['boundary']
        cutr = bndr - meta['track_cut_margin']
        bndc = [[bndy, bndx]]*2
        bndpatch, cutpatch = helpy.draw_circles(bndc, [bndr, cutr], ax=ax,
                                                color='r', fill=False, zorder=1)
        cutpatch.set_label('cut margin')

    lengths = map(len, [imstack, fsets, fcsets])
    f_max = min(lengths)
    assert f_max, 'Lengths imstack: {}, fsets: {}, fcsets: {}'.format(*lengths)
    if f_nums is None:
        f_nums = range(f_max)
    elif isinstance(f_nums, tuple):
        f_nums = range(*f_nums)
    elif len(f_nums) > f_max:
        f_nums = f_nums[:f_max]
    elif len(f_nums) < f_max:
        f_nums = range(f_max)
    else:
        raise ValueError('expected `f_nums` to be None, tuple, or list')
    f_idx = repeat = f_old = 0
    f_num = f_nums[f_idx]

    while 0 <= f_idx < f_max:
        if repeat > 5:
            if verbose:
                print 'stuck on frame {} ({})'.format(f_idx, f_num),
            break
        if f_idx == f_old:
            repeat += 1
        else:
            repeat = 0
            f_old = f_idx
        assert f_num == f_nums[f_idx], "f_num != f_nums[f_idx]"
        if verbose:
            print 'showing frame {} ({})'.format(f_idx, f_num),
        xyo = helpy.consecutive_fields_view(fsets[f_num], 'xyo')
        xyc = helpy.consecutive_fields_view(fcsets[f_num], 'xy')
        x, y, o = xyo.T

        p.set_data(imstack[f_idx])
        remove = []

        # plot the detected dots
        ts = helpy.quick_field_view(fsets[f_num], 't')
        tracked = ts >= 0
        if args.plottracks:
            cmap = plt.get_cmap('Set3')
            colors = cmap(ts[tracked] % cmap.N)**3  # cube to darken
            ax.scatter(y[tracked], x[tracked],
                       s=3, c=colors,
                       zorder=0.1*f_idx/f_max,  # later tracks on top
                      )
        else:
            colors = 'white'
        ps = ax.scatter(y[tracked], x[tracked], s=64, c=colors,
                        marker='o', edgecolors='black')
        remove.append(ps)
        if not clean:
            us = ax.scatter(y[~tracked], x[~tracked], c='c', zorder=.8)
            cs = ax.scatter(xyc[:, 1], xyc[:, 0], c='g', s=10, zorder=.6)
            remove.extend([us, cs])
            for dot, xy in zip(('center', 'corner'), (xyo, xyc)):
                ph = helpy.draw_circles(xy[:, 1::-1], meta[dot+'_kern'], ax=ax,
                                        lw=.5, color='k', fill=False, zorder=.6)
                remove.extend(ph)

        # plot the orientations
        omask = np.isfinite(o)
        xyoo = xyo[omask]
        to = ts[omask]
        xo, yo, oo = xyoo.T
        so, co = np.sin(oo), np.cos(oo)
        q = ax.quiver(yo, xo, so, co, angles='xy', units='xy',
                      width=side/8, scale_units='xy', scale=10/side, zorder=.4,
                      edgecolor='white', facecolor='black', linewidth=0.75)
        remove.append(q)

        # label n-hat
        t, d = 3, 10
        nhat_offsets = d * np.stack([(1.5*so - co), (1.5*co + so)], axis=1)
        nhat_is = np.where(to == t)[0] if clean else xrange(len(to))
        nhats = [ax.annotate(
            r'$\hat n$',
            xy=(yo[nhat_i], xo[nhat_i]),
            xytext=nhat_offsets[nhat_i],
            textcoords='offset points', ha='center', va='center',
            fontsize='large')
            for nhat_i in nhat_is]
        remove.extend(nhats)

        if not clean:
            for dr in [-drc, 0, drc]:
                patches = helpy.draw_circles(xyo[:, 1::-1], rc+dr, ax=ax, lw=.5,
                                             color='g', fill=False, zorder=.5)
                remove.extend(patches)
        if fisets is not None and f_num in fisets:
            xyi, oi, tsi = [fisets[f_num][fld] for fld in ['xy', 'o', 't']]
            pim = fisets[f_num]['id'] == 0
            if np.any(pim):
                ips = ax.scatter(xyi[pim, 1], xyi[pim, 0],
                                 c=['pink', 'r'][clean], zorder=.7)
                remove.append(ips)
                if not clean:
                    itxt = plt_text(xyi[pim, 1], xyi[pim, 0]+txtoff,
                                    tsi[pim].astype('S'), color='pink',
                                    zorder=.9, horizontalalignment='center')
                    remove.extend(itxt)
            iq = ax.quiver(xyi[:, 1], xyi[:, 0], np.sin(oi), np.cos(oi),
                           angles='xy', units='xy', width=side/8,
                           scale_units='xy', scale=1/side, zorder=.3,
                           facecolor='black' if clean else 'gray',
                           edgecolor='white', linewidth=0.75)
            remove.append(iq)

        if fosets is not None:
            oc = helpy.quick_field_view(fosets[f_num], 'corner')
            oca = oc.reshape(-1, 2)
            ocs = ax.plot(oca[:, 1], oca[:, 0], linestyle='', marker='o',
                          color='white' if clean else 'black',
                          markeredgecolor='black', markersize=4, zorder=1)
            remove.extend(ocs)

            # corner displacements has shape (n_particles, n_corners, n_dim)
            cdisp = oc[omask] - xyoo[:, None, :2]
            cang = corr.dtheta(np.arctan2(cdisp[..., 1], cdisp[..., 0]))
            deg_str = np.nan_to_num(np.degrees(cang)).astype(int).astype('S')
            cx, cy = oc[omask].mean(1).T
            if not clean:
                ctxt = plt_text(cy, cx, deg_str, color='orange', zorder=.9,
                                horizontalalignment='center',
                                verticalalignment='center')
                remove.extend(ctxt)

        if not clean:
            txt = plt_text(y[tracked], x[tracked]+txtoff,
                           ts[tracked].astype('S'),
                           color='r', zorder=.9, horizontalalignment='center')
            remove.extend(txt)

        nts = np.count_nonzero(tracked)
        nos = np.count_nonzero(omask)
        ncs = len(o)
        ax.set_title(title.format(f_num, nos, nts, ncs))

        if need_legend:
            need_legend = False
            if rc > 0:
                patches[0].set_label('r to corner')
            q.set_label('orientation')
            ps.set_label('centers')
            if fosets is None:
                cs.set_label('corners')
            else:
                cs.set_label('unused corner')
                ocs[0].set_label('used corners')
            if len(txt):
                txt[0].set_label('track id')
            ax.legend(fontsize='small')

        fig.canvas.draw()
        fig.canvas.mpl_connect('key_press_event', advance)
        plt.waitforbuttonpress()
        for rem in remove:
            rem.remove()
        if verbose:
            print '\tdone with frame {} ({})'.format(f_old, f_nums[f_old])
            sys.stdout.flush()
    if verbose:
        print 'loop broken'


def gapsize_distro(tracksetses, fields='fo', title=''):
    fig, ax = plt.subplots()
    for field in fields:
        isf = field == 'f'
        gaps = [np.diff(t['f'] if isf else np.where(~np.isnan(t[field]))[0]) - 1
                for tsets in tracksetses for t in tsets.itervalues()]
        gaps = np.concatenate(gaps)
        gmax = gaps.max()
        if not gmax or gmax > 1e3:
            continue
        bins = np.arange(gmax)+1
        dist = np.bincount(gaps)[1:]
        wght = dist*bins
        ax.bar(bins-.4, dist, .4,
               color='yr'[isf], alpha=.5, label=repr(field)+' gaps')
        ax.bar(bins, wght, .4,
               color='gb'[isf], alpha=.5, label=repr(field)+' frames')
        ax.set_ylabel('number of gaps or frames')
        ax.set_xlabel('gap size (frames)')
    nframes = np.sum([len(tset)
                      for tsets in tracksetses for tset in tsets.itervalues()])
    axr = ax.twinx()
    axr.set_ylim(None, ax.get_ylim()[-1]/nframes)
    axr.set_yticks(ax.get_yticks()/nframes)
    axr.set_ylabel('fraction of frames')
    ax.legend()
    ax.set_title(title)


def repair_tracks(tracksets, max_gap=10, interp=['xy', 'o'],
                  inplace=True, verbose=False):
    if not inplace:
        tracksets = {t: s.copy() for t, s in tracksets.iteritems()}
    if verbose:
        print 'filling gaps with nans'
        if interp:
            print 'and interpolating nans in', ', '.join(interp)
        print '\ttrack length max_gap num_gaps num_frames'
        fmt = '\t{:5} {:6} {:7} {:8} {:10}'.format
    for t, tset in tracksets.items():
        fs = tset['f']
        filled = curve.fill_gaps(tset, fs, max_gap=max_gap,
                                 ret_gaps=verbose, verbose=verbose)
        if verbose:
            gaps = filled[2]
            print fmt(t, len(tset), gaps.max(), len(gaps), gaps.sum())
        tset, fs = filled[:2]
        if tset is None:
            tracksets.pop(t)
        else:
            tset['f'] = fs
            tset['t'] = t
            tracksets[t] = tset
            for field in interp or []:
                curve.interp_nans(tset[field], fs, max_gap=max_gap,
                                  inplace=True, verbose=verbose and verbose - 1)
    if verbose:
        print
    return tracksets


def plot_tracks(data, trackids=None, bgimage=None, mask=None,
                save=False, show=True):
    """ Plots the tracks of particles in 2D

    parameters
    ----------
    data : the main data array of points
    trackids : the array of track ids
    bgimage : a background image to plot on top of (the first frame tif, e.g.)
    mask : a boolean mask to filter the data (to show certain frames or tracks)
    fignum : a pyplot figure number to add the plot to
    save : where to save the figure
    show : whether to show the figure
    """
    fig, ax = plt.subplots()
    if bgimage is not None:
        if isinstance(bgimage, basestring):
            bgimage = plt.imread(bgimage)
        ax.imshow(bgimage, cmap='gray')
    if trackids is None:
        trackids = data['t']
    if mask is None:
        mask = np.where(trackids >= 0)
    data = data[mask]
    trackids = trackids[mask]
    cmap = plt.get_cmap('Dark2')
    ax.scatter(data['y'], data['x'], c=trackids % cmap.N,
               marker='o', cmap=cmap, lw=0)
    ax.set_aspect('equal')
    ax.set_xlim(data['y'].min()-10, data['y'].max()+10)
    ax.set_ylim(data['x'].min()-10, data['x'].max()+10)
    ax.set_title(prefix)
    if save:
        save = save + '_tracks.png'
        print "saving tracks image to",
        print save if verbose else os.path.basename(save)
        fig.savefig(save, frameon=False, dpi=300)
    if show:
        plt.show()

# Mean Squared Displacement
# dx^2 (tau) = < ( x_i(t0 + tau) - x_i(t0) )^2 >
#              <  averaged over t0, then i   >


def t0avg(trackset, tracklen, tau):
    """ Averages the squared displacement of a track for a certain value of tau
        over all valid values of t0 (such that t0 + tau < tracklen)

        That is, for a given particle, do the average over all t0
            <[(r_i(t0 + tau) - r_i(t0)]^2>
        for a single particle i and fixed time shift tau

        parameters
        ----------
        trackset : a subset of data for all points in the given track
        tracklen : the length (duration) of the track
        tau : the time separation for the displacement: r(tau) - r(0)

        returns
        -------
        the described mean, a scalar
    """
    totsqdisp = 0.0
    nt0s = 0.0
    tfsets = helpy.load_framesets(trackset)
    # for t0 in (T - tau - 1), by dt0 stepsize
    for t0 in np.arange(1, (tracklen-tau-1), dt0):
        olddot = tfsets[t0]
        newdot = tfsets[t0+tau]
        if len(newdot) != 1 or len(olddot) != 1:
            continue
        sqdisp = (newdot['x'] - olddot['x'])**2 + (newdot['y'] - olddot['y'])**2
        if len(sqdisp) == 1:
            if verbose > 1:
                print 'unflattened'
            totsqdisp += sqdisp
        elif len(sqdisp[0]) == 1:
            if verbose:
                print 'flattened once'
            totsqdisp += sqdisp[0]
        else:
            if verbose:
                print "fail"
            continue
        nt0s += 1.0
    return totsqdisp/nt0s if nt0s else None


def trackmsd(trackset, dt0, dtau, ret_vector=False):
    """ finds the mean squared displacement as a function of tau,
        averaged over t0, for one track (particle)

        parameters
        ----------
        trackset : a subset of the data for a given track
        dt0 : spacing stepsize for values of t0, gives the number of starting
            points averaged over in `t0avg`
        dtau : spacing stepsize for values of tau, gives the spacing of the
            points in time at which the msd is evaluated

        For dt0, dtau:  Small values will take longer to calculate without
            adding to the statistics. Large values calculate faster but give
            worse statistics. For the special case dt0 = dtau = 1, a
            correlation is used for a signicant speedup

        returns
        -------
        a list of tuples (tau, msd(tau)) the value of tau and the mean squared
        displacement for a single track at that value of tau

    """
    if dt0 == dtau == 1:
        xy = helpy.consecutive_fields_view(trackset, 'xy')
        if ret_vector.startswith(('body', 'prog', 'div')):
            theta = trackset['o']
            tmsd = corr.msd_body(xy, theta, ret_taus=True)
        else:
            tmsd = corr.msd(xy, ret_taus=True, ret_vector=ret_vector)
        return tmsd
    elif ret_vector:
        raise ValueError("Cannot return vector with dt0 or dtau > 0")

    trackbegin, trackend = trackset['f'][[0, -1]]
    tracklen = trackend - trackbegin + 1
    if verbose:
        print "length {} from {} to {}".format(tracklen, trackbegin, trackend)
    if isinstance(dtau, float):
        taus = helpy.farange(dt0, tracklen, dtau)
    elif isinstance(dtau, int):
        taus = xrange(dtau, tracklen, dtau)

    tmsd = []
    for tau in taus:
        avg = t0avg(trackset, tracklen, tau)
        if avg > 0 and not np.isnan(avg):
            tmsd.append([tau, avg[0]])
    if verbose:
        print "\t...actually", len(tmsd)
    return tmsd


def find_msds(tracksets, dt0, dtau, ret_vector=False):
    """ Calculates the MSDs for all tracks

        parameters
        ----------
        dt0, dtau : see documentation for `trackmsd`
        tracksets : dict of subsets of the data for a given track
        ret_vector : whether to return vector components of msd

        returns
        -------
        msds : a list of all trackmsds (each in the format given by `trackmsd`)
        msdids : a list of the trackids corresponding to each msd
    """
    print "Calculating MSDs with",
    print "dt0 = {}, dtau = {}".format(dt0, dtau)
    if verbose:
        print "for track",
    msds = []
    msdids = []
    for t in sorted(tracksets):
        if verbose:
            print t,
            sys.stdout.flush()
        tmsd = trackmsd(tracksets[t], dt0, dtau, ret_vector)
        if len(tmsd) > 1:
            msds.append(tmsd)
            msdids.append(t)
        elif verbose:
            print "\ntrack {}'s msd is too short!".format(t)
    if verbose:
        print
    return msds, msdids


def make_taus(dtau, dt0, nframes, maxlength=None):
    if verbose:
        print "using dtau = {}, dt0 = {}".format(dtau, dt0)
    try:
        dtau = np.asscalar(dtau)
    except AttributeError:
        pass
    if isinstance(dtau, (float, np.float)):
        taus = helpy.farange(dt0, nframes+1, dtau)
    elif isinstance(dtau, (int, np.int)):
        taus = np.arange(dtau, nframes+1, dtau, dtype=float)
    return taus[:maxlength]

def mean_msd(msds, dtau, dt0, nframes, A=1, fps=1, kill_flats=1, kill_jumps=1e9,
             msdids=None, show_tracks=False, singletracks=None,
             tnormalize=False):
    """ return the mean of several track msds """

    # msd has shape (number of tracks, length of tracks, msd vector dimension)
    msdshape = [len(singletracks or msds),
                max(map(len, msds))]
    vdim = len(msds[0][0]) - 1
    if vdim > 1:
        msdshape += [vdim]
    msd = np.full(msdshape, np.nan, float)
    taus = make_taus(dtau, dt0, nframes, maxlength=msdshape[1])
    if verbose:
        print 'msdshape:', msdshape
        print 'taushape:', taus.shape,
        print 'assembling msd tracks'

    if msdids is not None:
        allmsds = it.izip(xrange(len(msds)), msds, msdids)
    elif msdids is None:
        allmsds = enumerate(msds)
    for thismsd in allmsds:
        if singletracks\
               and msdids is not None\
               and msdid not in singletracks:
            continue
        if msdids is not None:
            ti, tmsd, msdid = thismsd
        else:
            ti, tmsd = thismsd
        if len(tmsd) < 2:
            continue
        if verbose:
            print ti,
        tmsd = np.asarray(tmsd)
        tmsdt = tmsd[:, 0]
        tmsdd = tmsd[:, 1:].squeeze()
        if tmsdd[-50:].mean() < kill_flats or tmsdd[:2].mean() > kill_jumps:
            continue
        tau_match = np.searchsorted(taus, tmsdt)
        msd[ti, tau_match] = tmsdd
    if verbose:
        print 'shapes of msd, taus:',
        print msd.shape, taus.shape
    msd, msd_mean, msd_err, msd_std, added, enough = helpy.avg_uneven(
        msd, ret_all=True)
    if verbose:
        not_enough = np.where(added <= 2)[0]
        bad_taus = taus[not_enough]
    taus = taus[enough]
    if verbose:
        print 'shapes of msd, taus, msd_mean with `enough` tracks:',
        print msd.shape, taus.shape, msd_mean.shape
        assert np.all(np.isfinite(msd_mean)), 'msd_mean not finite'
        print 'msd_std min max:', msd_std[:-1].min()/A, msd_std[:-1].max()/A
        print 'msd_err min max:', msd_err[:-1].min()/A, msd_err[:-1].max()/A
        print 'shape of msd_err:', msd_err.shape
    if verbose > 1:
        global rrerrfig
        oldax = plt.gca()
        rrerrfig = plt.figure()
        erraxl = rrerrfig.gca()
        erraxl.set_xscale('log')
        erraxl.set_yscale('log')
        erraxl.plot(taus/fps, msd_std/A, '.c', label='stddev')
        erraxl.plot(taus/fps, msd_err/A, '.g', label='stderr')
        erraxr = erraxl.twinx()
        erraxr.set_yscale('log')
        erraxr.plot(taus/fps, added[enough], '.b', label='N added')
        erraxr.plot(bad_taus/fps, np.maximum(added[not_enough], .8),
                    'vr', mec='none', label='N <= 2')
        erraxr.set_ylim(0.8, None)
        erraxl.legend(loc='upper left', fontsize='x-small')
        erraxr.legend(loc='upper right', fontsize='x-small')
        plt.sca(oldax)
    if show_tracks and vdim == 1:
        plt.plot(taus/fps, (msd/(taus/fps)**tnormalize).T/A,
                 'b', alpha=.2, lw=0.5)
    return taus, msd_mean, msd_err


def plot_msd(taus, msd, msd_err, S=1, ang=False, errorbars=False,
             fig=(8, 6), fps=1, labels=True, legend=False,
             prefix='', save='', show=True, sys_size=0, title=None,
             tnormalize=False, **plt_kwargs):
    """ Plots the MS(A)Ds """
    A = 1 if ang else S**2
    if verbose:
        print "using S = {} pixels, thus A = {} px^2".format(S, A)
    if isinstance(fig, plt.Figure):
        ax = fig.gca()
    elif isinstance(fig, plt.Axes):
        ax = fig
        fig = ax.figure
    elif isinstance(fig, tuple):
        fig, ax = plt.subplots(figsize=fig)

    taus = taus / fps
    msd = msd / A
    if errorbars:
        msd_err = msd_err / A

    ax.set_xscale(plt_kwargs.pop('xscale', 'log'))
    ax.set_yscale(plt_kwargs.pop('yscale', 'log'))
    xlim = plt_kwargs.pop('xlim', None)
    ylim = plt_kwargs.pop('ylim', None)

    if labels is True:
        label = "Mean Sq {}Disp".format("Angular "*ang)
    elif isinstance(labels, basestring):
        label = labels
    else:
        label = ''

    if tnormalize:
        tnorm = taus**tnormalize
        msd = msd/tnorm
        if errorbars:
            msd_err = msd_err/tnorm
        label += "/Time{}".format("^{}".format(tnormalize)*(tnormalize != 1))
        ax.plot(taus, msd[0]*taus/tnorm, 'k-', label="ref slope = 1", lw=2)

    if errorbars:
        default = {'capthick': 0, 'elinewidth': 1, 'errorevery': 3,
                   'color': 'r', 'linewidth': 0, 'marker': 'o'}
        plt_kwargs = dict(default, **plt_kwargs)
        ax.errorbar(taus, msd, msd_err, label=label, **plt_kwargs)
    else:
        default = {'color': 'r', 'linewidth': 1, 'marker': ''}
        plt_kwargs = dict(default, **plt_kwargs)
        ax.plot(taus, msd, label=label, **plt_kwargs)
    if sys_size:
        ax.axhline(sys_size, ls='--', lw=.5, c='k', label='System Size')
    if title is None:
        title = "Mean Sq {}Disp".format("Angular " if ang else "")
    ax.set_title(title)
    xlabel = r'$t \enspace f$' if 1 < fps < 60 else 'Time ({}s)'.format('frame'*(fps == 1))
    ax.set_xlabel(xlabel)
    ylabel = (r"${}\langle\left[\vec r(t) -".format('' if args.clean else r'\left')
              + r"\vec r(0)\right]^2{}\rangle / \ell^2$".format(
                  '' if args.clean else r'\right'))
    ax.set_ylabel(ylabel, labelpad=-2, fontsize='large')
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if legend:
        ax.legend(loc='best')
    if save is True:
        save = prefix + "_MS{}D.pdf".format('A' if ang else '')
    if save:
        print "saving to", save if verbose else os.path.basename(save)
        fig.savefig(save)
    if show:
        plt.show()
    return fig, ax


def make_fitname(fit):
    """Attempt to create a nickname for a fit given a Fit instance"""
    # ('func', 'TR', 'DR', 'lp', 'DT', 'v0', 'w0')
    d = {'func':
         {'vo': 'vo', 'vn': 'vn', 'vt': 'vt',
          'nn': 'nn',
          'rn': 'rn', 'rp': 'rp', 'rs': 'rs', 'ra': 'ra', 'rm': 'rm',
          'rr': 'rr', 'r0': 'r0',
          'pr': 'pr', 'p0': 'p0', 'dr': 'dr', 'd0': 'd0',
         },
         'TR': {None    : '0',
                'free'  : 'f',
                1.8     : 'm',
               },
         'DR': {'free'  : 'f',

               },
        }

    fmt = '{func}_T{TR}_R{DR}_L{lp}_D{DT}'.format
    fmt(func=fit.func, TR=fit.TR or 0, DR=fit.DR, lp=fit.lp, DT=fit.DT)
    return


def make_fitnames():
    """Build up a mapping to nicknames for most conceivablel fits"""

    # from velocities
    mkf = helpy.make_fit
    cf = {'free': 'free', None: None}
    cf.update({
        'vn': mkf(func='vn', v0='mean', DT='var'),
        'vt': mkf(func='vt', DT='var'),
        'vo': mkf(func='vo', DR='var', w0='mean'),
    })

    # from orientation autocorrelation: nn
    mkf = partial(helpy.make_fit, func='nn', DR='free')
    cf.update({
        'nn_T0_Rf': mkf(TR=None),
        'nn_Tf_Rf': mkf(TR='free'),
        'nn_Tm_Rf': mkf(TR=1.8),
    })

    # from forward displacement: rn rp rs ra rm
    mkf = partial(helpy.make_fit, lp='free')
    for func, TR, DR in [('r'+r, 'T'+T, 'R'+R)
                         for r in 'npsam' for T in '0nm' for R in 'fn']:
        desc = '_'.join([func, TR, DR, 'Lf'])
        TRf = {'T0': None,
               'Tn': 'nn_Tf_Rf',
               'Tm': 'nn_Tm_Rf'}[TR]
        DRf = {'Rf': 'free',
               'Rn': TRf or 'nn_'+TR+'_Rf'}[DR]
        cf[desc] = mkf(func=func, TR=cf[TRf], DR=cf[DRf])

    # from mean squared displacement: rr pr dr r0 p0 d0
    mkf = partial(helpy.make_fit, DT='free')
    for func, TR, lp, DT in [(r+z, 'T'+T, 'L'+l, 'D'+D) for T in '0nm'
                             for D in 'fd' for l in 'fnabqrd'
                             for z in 'r0' for r in 'rpd']:
        if func[0] in (lp[1], DT[1]):
            continue  # cannot source lp or DT from self
        desc = '_'.join([func, TR, 'Rn', lp, DT])
        TRf = {'T0': None,
               'Tn': 'nn_Tf_Rf',
               'Tm': 'nn_Tm_Rf'}[TR]
        DRf = TRf or 'nn_T0_Rf'
        lpf = {'Lf': 'free',
               'Ln': 'rn_'+TR+'_Rn_Lf',
               'La': 'ra_'+TR+'_Rn_Lf',
               'Lb': 'ra_'+TR+'_Rf_Lf',
               'Lq': 'rn_'+TR+'_Rf_Lf',
               'Lr': '_'.join(['r'+func[1], TR, 'Rn', 'Lf', 'Df']),
               'Lp': '_'.join(['p'+func[1], TR, 'Rn', 'Lf', 'Df']),
               'Ld': '_'.join(['d'+func[1], TR, 'Rn', 'Lf', 'Df']),
              }[lp]
        DTf = {'Df': 'free',
               'Dr': '_'.join(['r'+func[1], TR, 'Rn', 'Lf', 'Df']),
               'Dp': '_'.join(['p'+func[1], TR, 'Rn', 'Lf', 'Df']),
               'Dd': '_'.join(['d'+func[1], TR, 'Rn', 'Lf', 'Df']),
              }[DT]
        cf[desc] = mkf(func=func, TR=cf[TRf], DR=cf[DRf],
                       lp=cf[lpf], DT=cf[DTf])

    del cf['free'], cf[None]
    return cf, {cf[k]: k for k in cf}


def get_param_value(name, fits=(), sources=()):
    """get the appropriate value and source for a parameter."""
    sources = sources or 'nn rn ra rp rr r0 pr p0 dr d0'.split()
    if hasattr(name, '__iter__'):
        # if multiple names are given, return two dicts of {name: value}
        return [dict(zip(name, v))
                for v in zip(*(get_param_value(n, fits, sources)
                               for n in name))]
    for fit in [f for s in sources for f in fits if f.func == s]:
        try:
            val = float(fits[fit].get(name, fit._asdict()[name]))
            return val, fit
        except (ValueError, TypeError):
            continue
    guesses = {'TR': 1.6, 'DR': 1/16, 'lp': 2.5, 'v0': 0.16, 'DT': 0.01}
    return guesses[name], helpy.make_fit({name: 'guess'})


def format_fit(result, model_name=None, sources=None):
    """format the results for printing, labeling plots, and saving."""
    tex_name = {'DT': 'D_T', 'DR': 'D_R', 'v0': 'v_0',
                'lp': '\\ell_p', 'TR': '\\tau_R'}

    def print_fmt(params, sep='\n'):
        fmt = '{:>8s}: {:.4g}'.format
        return sep.join(fmt(p, params[p]) for p in params)

    def tex_fmt(params, sep=', '):
        fmt = partial(helpy.SciFormatter().format, "${0:s}={1:.3T}$")
        return sep.join(fmt(tex_name[p], params[p]) for p in params)

    fixed, free = {}, {}
    for p in result.params.values():
        (free if p.vary else fixed)[p.name] = float(p)

    fit = helpy.make_fit(dict.fromkeys(free, 'free'),
                         {p: sources[p] for p in fixed},
                         func=model_name or result.model.name)
    fitname = fit_desc[fit]

    print fitname
    print "fixed params:"
    print print_fmt(fixed)
    print "free params:"
    print print_fmt(free)

    tex_eqn = result.model.func.func_doc.replace('\n', '')
    if args.clean:
        tex_eqn= tex_eqn.replace('(v_0/D_R)', r'\ell_p')
    for_tex = [tex_eqn,
               'fixed: ' + tex_fmt(fixed),
               'free: ' + tex_fmt(free)]

    return fit, fitname, free, for_tex


def save_corr_plot(fig, fit_desc):
    save = '{}_{}-corr.pdf'.format(saveprefix, fit_desc)
    if args.fig == 0:
        return
    print 'saving <{}> correlation plot to'.format(fit_desc),
    print os.path.relpath(save, absprefix.replace(relprefix, ''))
    fig.savefig(save)
    return save


def plot_fit(result, tex_fits, args, t=None, data=None,
             ax=None, plt_kwargs=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4) if args.clean else (8, 6))
    elif isinstance(ax, int):
        fig = plt.figure(ax)
        ax = fig.gca()
    else:
        fig = ax.figure
    if t is None:
        t = result.userkws[result.model.independent_vars[0]]  # x-value from fit
    if data is None:
        data = result.data
    if args.showtracks:
        raise NotImplementedError('cannot show tracks here')
        ax.plot(t, all_corrs.T, 'b', alpha=.2, lw=0.5)

    kws = dict(label='experiment'*labels, color=args.vcol, linestyle='',
               marker='o', markersize=4, markeredgecolor=args.vcol,
               markerfacecolor='none', capthick=0, elinewidth=1, errorevery=3)
    kws.update(plt_kwargs.get('errorbar', {}))
    ax.errorbar(t, data, 1/result.weights, None, zorder=2, **kws)
    fitlabel = '\n'.join(tex_fits[:None if labels else 1])
    ax.plot(t, result.best_fit, c=args.pcol, lw=1, label=fitlabel, zorder=2.5)

    axes_setter(ax, **plt_kwargs.get('axes', {}))
    ax.legend(**dict({'fontsize': 'small', 'framealpha': 1, 'frameon': labels,
                      'numpoints': 1, 'handlelength': 1},
                     **plt_kwargs.get('legend', {})))

    return fig, ax


def nn_corr(tracksets, args):
    """Calculate the <nn> correlation for all the tracks in a given dataset
    """
    corr_nn = partial(corr.autocorr, cumulant=False, norm=False)
    if args.verbose:
        print 'calculating <nn> correlations'
    if args.dot:
        all_corrs = [corr_nn(np.cos(ts['o'])) + corr_nn(np.sin(ts['o']))
                     for ts in tracksets.itervalues()]
    else:
        all_corrs = [c for ts in tracksets.itervalues() for c in
                     [corr_nn(np.cos(ts['o'])), corr_nn(np.sin(ts['o']))]]

    all_corrs, meancorr, errcorr = helpy.avg_uneven(all_corrs, pad=True)
    taus = np.arange(len(meancorr))/args.fps
    return taus, meancorr, errcorr


def rn_corr(tracksets, args):
    """Calculate the <rn> correlation for all the tracks in a given dataset
    """
    correlate_rn = partial(corr.crosscorr, side='both', ret_dx=True,
                           cumulant=(True, False), norm=0)

    # shape (track, x_or_y, time_or_correlation, time)
    rn_corrs = np.array([[correlate_rn(ts['x']/args.side, np.cos(ts['o'])),
                          correlate_rn(ts['y']/args.side, np.sin(ts['o']))]
                         for ts in tracksets.itervalues()])
    # Align and merge them
    taus = rn_corrs[:, :, 0]/args.fps
    if rn_corrs.ndim == 4:
        if verbose:
            print "Already aligned: all tracks have same length"
        taus = taus[0, 0]
        rn_corrs = rn_corrs[:, :, 1]
    else:
        if verbose:
            print "Aligning tracks around tau=0"
        tau0 = np.array(map(partial(np.searchsorted, v=0), taus.flat))
        taus = taus.flat[tau0.argmax()]
        rn_corrs = helpy.pad_uneven(rn_corrs[:, :, 1], np.nan, align=tau0)
    if args.dot:
        rn_corrs = rn_corrs.sum(1)  # sum over x and y components
    else:
        rn_corrs = rn_corrs.reshape(-1, len(taus))
    rn_corrs, meancorr, errcorr, stddev, added, enough = helpy.avg_uneven(
        rn_corrs, pad=False, ret_all=True, weight=False)
    taus = taus[enough]
    return taus, meancorr, errcorr


def rr_corr(msds, msdids, data, args):
    """Calculate the <rr> (MSD) correlation for all the tracks in the dataset
    """
    taus, msd, msd_err = mean_msd(
        msds, args.dtau, args.dt0, data['f'].max()+1,
        msdids=msdids, A=args.side**2, fps=args.fps, tnormalize=0,
        kill_flats=args.killflat, kill_jumps=args.killjump*args.side**2,
        singletracks=args.singletracks, show_tracks=args.showtracks)

    if msd.ndim == 2:
        # tuples of (displacement, progression, diversion)
        msd = np.array((msd.sum(1),) + tuple(msd.T))
        msd_err = np.array((np.hypot(*msd_err.T),) + tuple(msd_err.T))
    elif args.msdvec.startswith(('prog', 'div')):
        msg = 'You need to rerun `tracks.py {} --msd --msdvec`'
        raise ValueError(msg.format(args.prefix))

    return taus, msd, msd_err


def nn_form_dot_white(s, DR):
    r"""$e^{-D_R t}$
    """
    return np.exp(-DR*s)


def nn_form_dot_color(s, DR, TR):
    r"""$e^{-D_R\left[t - \tau \left(1 - e^{-t/\tau}\right)\right]}$
    """
    if TR > 1e-3/args.fps:
        # only calculate if TR will have significant effect
        s = s + TR*np.expm1(-s/TR)
    return np.exp(-DR*s)


def nn_form_components_white(s, DR):
    r"""$\frac{1}{2}e^{-D_R t}$
    """
    return 0.5*np.exp(-DR*s)


def nn_form_components_color(s, DR, TR):
    r"""$\frac{1}{2}e^{-D_R
        \left[t - \tau \left(1 - e^{-t/\tau}\right)\right]}$
    """
    if TR > 1e-3/args.fps:
        # only calculate if TR will have significant effect
        s = s + TR*np.expm1(-s/TR)
    return 0.5*np.exp(-DR*s)


def rn_form_dot_white(s, lp, DR):
    r"""$\frac{v_0}{D_R}(1 - e^{-D_R|t|})\operatorname{sign}(t)$
    """
    return -lp*np.sign(s)*np.expm1(-DR*np.abs(s))


def rn_form_dot_color(s, lp, DR, TR):
    r"""$\frac{v_0}{D_R}e^{D_R\tau}(1 - e^{-D_R|t|})\operatorname{sign}(t)$
    """
    return -lp*exp(DR*TR)*np.sign(s)*np.expm1(-DR*np.abs(s))


def rn_form_components_white(s, lp, DR):
    r"""$\frac{v_0}{2D_R}(1 - e^{-D_R|t|})\operatorname{sign}(t)$
    """
    return -0.5*lp*np.sign(s)*np.expm1(-DR*np.abs(s))


def rn_form_components_color(s, lp, DR, TR):
    r"""$\frac{v_0}{2D_R}e^{D_R\tau}(1 - e^{-D_R|t|})\operatorname{sign}(t)$
    """
    return -0.5*lp*exp(DR*TR)*np.sign(s)*np.expm1(-DR*np.abs(s))


def quartic(a):
    quadratic = a*a
    return quadratic * quadratic


def rr_form_total(s, DT, lp, DR, TR):
    r"""$2(v_0/D_R)^2 e^{D_R\tau} \left(D_Rt - 1 + e^{-D_Rt}\right) + 4 D_Tt$
    """
    color = exp(DR*TR)
    persistence = color*lp**2
    decay = np.exp(-DR*s)
    diffusion = 2*DT*s
    propulsion = decay - 1 + DR*s
    return 2*(persistence*propulsion + diffusion)


def rr_form_prog(s, DT, lp, DR, TR):
    r"""$\ell_p^2 e^{D_R \tau}
    \left(
        D_R t - 1 + e^{-D_Rt}
        + \frac{1}{12} e^{4D_R\tau} (e^{-4D_Rt}-4e^{-D_Rt}+3)
    \right) + 2 D_Tt$
    """
    color = exp(DR*TR)
    persistence = color*lp**2
    decay = np.exp(-DR*s)
    diffusion = 2*DT*s
    propulsion = decay - 1 + DR*s
    anisotropy = quartic(color)*(quartic(decay) - 4*decay + 3)/12
    return persistence*(propulsion + anisotropy) + diffusion


def rr_form_div(s, DT, lp, DR, TR):
    r"""$\ell_p^2 e^{D_R \tau}
    \left(
        D_R t - 1 + e^{-D_Rt}
        - \frac{1}{12} e^{4D_R\tau} (e^{-4D_Rt}-4e^{-D_Rt}+3)
    \right) + 2 D_Tt$
    """
    color = exp(DR*TR)
    persistence = color*lp**2
    decay = np.exp(-DR*s)
    diffusion = 2*DT*s
    propulsion = decay - 1 + DR*s
    anisotropy = quartic(color)*(quartic(decay) - 4*decay + 3)/12
    return persistence*(propulsion - anisotropy) + diffusion


def limiting_regimes(s, DT, lp, DR, TR):
    ll = lp*lp  # lp is squared everywhere
    DR_time = 1/DR
    DT_time = DT/ll*DR_time**2
    if DT_time > DR_time:
        return np.full_like(s, np.nan)
    timescales = (DT_time, DR_time)
    regimes = np.searchsorted(timescales, s)

    early = 2*DT*s          # s < DT_time
    middle = 2*ll*(DR*s)**2     # DT_time < s < DR_time
    late = 2*(ll*DR + DT)*s                   # DR_time < s
    lines = np.choose(regimes, [early, middle, late])

    transitions = np.clip(np.searchsorted(s, timescales), 0, len(s)-1)
    lines[transitions] = np.nan

    transitions += 1
    return lines, transitions


def get_axes_ratio(ax=None):
    """Calculate the aspect ratio of an axes boundary/frame"""
    if ax is None:
        ax = plt.gca()
    fig = ax.figure

    ll, ur = ax.get_position() * fig.get_size_inches()
    width, height = ur - ll

    return height / width


def get_data_ratio(ax=None):
    """Calculate the data limit ratio of an axes"""
    if ax is None:
        ax = plt.gca()

    xscale, yscale = ax.get_xscale(), ax.get_yscale()
    if xscale != yscale:
        raise ValueError('Cannot get data ratio for semilog plot')
    elif xscale == 'linear':
        return ax.get_data_ratio()
    elif xscale == 'log':
        return ax.get_data_ratio_log()
    else:
        raise ValueError(
            "Unknown scale `{}`; expected `'linear'` or `'log'`".format(xscale))


def get_aspect(ax=None):
    """Calculate the aspect ratio of an axes"""
    if ax is None:
        ax = plt.gca()

    return get_axes_ratio(ax) / get_data_ratio(ax)


def axes_set(ax, attr, *args, **kwargs):
    set_attr = getattr(ax, 'set_{}'.format(attr))
    return set_attr(*args, **kwargs)


def axes_setter(ax=None, **attr_dict):
    if ax is None:
        ax = plt.gca()
    props = {}
    for attr, args in attr_dict.iteritems():
        if isinstance(args, tuple):
            if len(args) > 1 and isinstance(args[-1], dict):
                args, kwargs = args[:-1], args[-1]
            else:
                args, kwargs = args, {}
        elif isinstance(args, dict):
            args, kwargs = (), args
        else:
            args, kwargs = (args,), {}
        props[attr] = axes_set(ax, attr, *args, **kwargs)
    return props


def move_axis_label(ax, which, x=None, y=None, ha=None, va=None):
    ax.figure.canvas.draw_idle() # update current label position before moving
    #ax.figure.tight_layout(pad=0)
    axis = {'y': ax.yaxis, 'x': ax.xaxis}[which]
    label = axis.get_label()
    if ha is not None:
        label.set_ha(ha)
    if va is not None:
        label.set_va(va)
    orig_x, orig_y = label.get_position()
    axis.set_label_coords(orig_x if x is None else x,
                          orig_y if y is None else y,
                          transform=label.get_transform())


def nn_plot(tracksets, fits, args, ax=None):
    model_name = 'nn'
    taus, meancorr, errcorr = nn_corr(tracksets, args)
    tmax = 50*args.zoom
    if verbose > 1:
        nnerrfig, nnerrax = plt.subplots()
        nnerrax.set_yscale('log')
    else:
        nnerrax = False
    sigma = curve.sigma_for_fit(meancorr, errcorr, x=taus, plot=nnerrax,
                                const=args.dtheta/rt2, ignore=[0, tmax],
                                verbose=verbose)
    if nnerrax:
        nnerrax.legend(loc='lower left', fontsize='x-small')
        nnerrfig.savefig(saveprefix+'_nn-corr_sigma.pdf')

    color_val, args.colored = args.colored, bool(args.colored)
    form = [[nn_form_components_white, nn_form_components_color],
            [nn_form_dot_white, nn_form_dot_color]][args.dot][args.colored]
    model = Model(form)

    params = model.make_params()
    vals, sources = get_param_value(params.keys())
    params['DR'].set(vals['DR'], min=0)
    if color_val:
        if color_val is True:
            params['TR'].set(vals['TR'], min=0)
        else:
            params['TR'].set(color_val, vary=False)
            sources['TR'] = color_val

    result = model.fit(meancorr, params, 1/sigma, s=taus)

    fit, fitname, free, tex_fits = format_fit(result, model_name, sources)
    fits[fit] = free

    if not args.nn:
        return result, fits, None

    xlim_pad = 2.5
    tmax = xlim_pad*args.zoom/result.params['DR'] + result.params.get('TR', 0)
    plt_kwargs = {
        'axes': {'xlim': (0, tmax),
                 'ylim': (exp(-xlim_pad*args.zoom), 1.05),
                 'yscale': 'log',
                 'ylabel': (r"$\langle \hat n(t) \cdot \hat n(0) \rangle$",
                            {'labelpad': -16, 'fontsize': 'large'}),
                 'xlabel': r"$t \enspace f$",
                },
        'legend': {'loc': 'upper right' if args.zoom <= 1 else 'lower left',
                   'markerfirst': False},
    }
    if labels:
        plt_kwargs['axes']['title'] = '\n'.join(filter(None, [
            "Orientation Autocorrelation", relprefix, fitname]))
    ax = ax or args.fig
    fig, ax = plot_fit(result, tex_fits, args,
                       ax=ax, plt_kwargs=plt_kwargs)

    if args.save:
        save_corr_plot(fig, fitname)

    return result, fits, ax


def rn_plot(tracksets, fits, args, ax=None):
    model_name = {'full': 'rn', 'pos': 'rp', 'anti': 'ra', 'max': 'rm'}[args.rn]
    taus, meancorr, errcorr = rn_corr(tracksets, args)

    form = [[rn_form_components_white, rn_form_components_color],
            [rn_form_dot_white, rn_form_dot_color]][args.dot][args.colored]
    model = Model(form)

    params = model.make_params()
    vals, sources = get_param_value(params.keys(), fits)
    params['DR'].set(vals['DR'], min=0, vary=args.fitdr or not args.nn)
    params['lp'].set(vals['lp'], min=0)
    if args.colored:
        params['TR'].set(vals['TR'], min=0, vary=args.fittr)

    xlim_pad = 3.0
    tmax = xlim_pad*args.zoom/params['DR']
    tmin = 0 if args.rn == 'pos' else -tmax
    if verbose > 1:
        rnerrfig, rnerrax = plt.subplots()
    else:
        rnerrax = False
    rnuncert = np.hypot(args.dtheta, args.dx)/rt2
    sigma = curve.sigma_for_fit(meancorr, errcorr, x=taus, plot=rnerrax,
                                const=rnuncert, ignore=[0, tmin, tmax],
                                verbose=verbose)

    #TODO: integrate weighted by sigma
    sym = curve.symmetry(meancorr, taus, parity=1, integrate=True)
    print "asymmetry: {:.3g}".format(sym.symmetry)

    if rnerrax:
        rnerrax.legend(loc='upper center', fontsize='x-small')
        rnerrfig.savefig(saveprefix+'_rn-corr_sigma.pdf')

    if args.rn in ('full', 'pos'):
        plot_data = None
        result = model.fit(meancorr, params, 1/sigma, s=taus)
    elif args.rn in ('anti', 'max'):
        plot_data = meancorr[sym.i]
        if args.rn == 'max':
            t_max = sym.antisymmetric.argmax()
            l_max = sym.antisymmetric[t_max] / exp(params['TR']*params['DR'])
            params['lp'].set(l_max, vary=False)
            sources['lp'] = 'max'
        result = model.fit(sym.antisymmetric, params, 1/sigma[sym.i], s=sym.x)
        if args.rn == 'max':
            result.params['lp'].set(vary=True)

    fit, fitname, free, tex_fits = format_fit(result, model_name, sources)
    fits[fit] = free
    fits[fit]['sym'] = float(sym.symmetry)

    print ' ==>  v0: {:.3f}'.format(result.params['lp']*result.params['DR'])

    ylim_pad = 1.2
    subtitle = {'full': '', 'pos': '\nfit only to positive half $t>0$',
                'sym': '\nfit only to symmetric part $f(t) + f(-t)$',
                'anti': '\nfit only to anti-symmetric part $f(t) - f(-t)$',
                'max': '\n$l_p=$ max of anti-symmetric part $f(t) + f(-t)$',
               }[args.rn]
    plt_kwargs = {
        'errorbar': {'errorevery': 6,},
        'axes': {'xlim': (-tmax, tmax),
                 'ylim': (ylim_pad*result.best_fit.min(),
                          ylim_pad*result.best_fit.max()),
                 'xlabel': r"$t \enspace f$",
                 'ylabel': (r"$\langle \vec r(t) \cdot \hat n(0) \rangle/\ell$",
                            {'labelpad': -2, 'fontsize': 'large'}),
                },
        'legend': {'loc': 'upper left',},
    }
    if labels:
        plt_kwargs['axes']['title'] = '\n'.join(filter(None, [
            "Position-Orientation Correlation", subtitle, relprefix, fitname]))
    ax = ax or args.fig and args.fig + 1
    fig, ax = plot_fit(result, tex_fits, args, data=plot_data,
                       ax=ax, plt_kwargs=plt_kwargs)
    #ax.plot(sym.x, sym.symmetric, '--r', label="symmetric part")
    ax.plot(sym.x, sym.antisymmetric, '-', c=args.ncol,
            linewidth=1, label="anti-symmetric", zorder=2.1)
    ax.axhline(y=0, color='grey', linestyle='--', linewidth=0.5, zorder=0.1)
    ax.axvline(x=0, color='grey', linestyle='--', linewidth=0.5, zorder=0.1)

    if args.rn == 'max':
        ax.scatter(sym.x[[t_max, -t_max]], sym.antisymmetric[[t_max, -t_max]],
                   marker='o', c=args.pcol)

    DR_time = 1/result.params['DR']
    if ax.get_xlim()[0] < DR_time < ax.get_xlim()[1]:
        ax.axvline(DR_time, 0, 0.4, ls='--', c='k')
        ax.annotate(r'$1/D_R$',
                    xy=(DR_time, 0.2), xycoords=('data', 'axes fraction'),
                    xytext=(6, 0), textcoords='offset points')

    if args.save:
        save_corr_plot(fig, fitname)

    return result, fits, ax


def rr_plot(msds, msdids, data, fits, args, ax=None):
    taus, msd, msd_err = rr_corr(msds, msdids, data, args)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4) if args.clean else (8, 6))
    else:
        fig = ax.figure

    # list arguments in order to be run
    comp_kwargs = list(cycler(**{
        'msdvec':  [0, -1, 1],
        'label':   ['total', 'transverse', 'longitudinal'],
        'fitv0':   [args.fitv0, False, 'disp'],
        'fitdt':   [True, False, 'div'],
        'do_plot': [True, True, True],
        'do_fit':  [True, False, False],
    }))

    # list arguments in order of msdvec: 0, 1, -1
    plt_kwargs = list(cycler(**{
        'color':        [args.vcol, args.lcol, args.tcol],
        'linewidth':    [0, 0, 0],
        'marker':       ['o', '^', 'v'],
        'markersize':   [4, 3, 3],
        'markeredgecolor': ['none', args.lcol, args.tcol],
        'markerfacecolor': [args.vcol, 'w', 'w'],
        'zorder':       [1, .9, .8],
    }))
    fitnames = [None]*3
    results = [None]*3
    for comp_kwarg in comp_kwargs:
        msdvec = comp_kwarg['msdvec']
        label = comp_kwarg['label']
        args.fitv0 = comp_kwarg['fitv0']
        args.fitdt = comp_kwarg['fitdt']
        if comp_kwarg['do_plot']:
            plot_msd(taus, msd[msdvec], msd_err[msdvec],
                     labels=label, S=args.side, save='', show=False,
                     fps=args.fps, tnormalize=0, prefix=saveprefix, fig=ax,
                     errorbars=True, capthick=0, elinewidth=1, errorevery=1,
                     title='' if args.clean else None, **plt_kwargs[msdvec])

        if comp_kwarg['do_fit']:
            fitname, result = rr_comp(taus, msd[msdvec], msd_err[msdvec],
                                      ax, fits, args, msdvec)
            fitnames[msdvec] = fitname
            results[msdvec] = result
    if labels:
        ax.set_title('\n'.join(["Mean Squared Displacement",
                                relprefix, fitnames[0]]))
    else:
        ax.tick_params(which='both', top=False, right=False)

    leg_handles, leg_labels = ax.get_legend_handles_labels()
    # remove errorbars (has_yerr=False) from legend keys
    for leg_handle in leg_handles:
        leg_handle.has_yerr = False
    # reorder from (eqn, tot, trans, long) to (eqn, long, trans, tot)
    ax.legend(leg_handles[:1] + leg_handles[:0:-1],
              leg_labels[:1] + leg_labels[:0:-1],
              loc='upper left', fontsize='small', frameon=False,
              numpoints=1, handlelength=1)

    if args.save:
        save_corr_plot(fig, fitnames[0])
    return results, fits, ax


def rr_comp(taus, msd, msd_err, ax, fits, args, msdvec=0):
    z = 'r0'[args.fit0]
    model_name = 'rpd'[msdvec] + z

    taus = taus/args.fps
    msd = msd/args.side**2
    msd_err = msd_err/args.side**2

    tmax = int(200*args.zoom)
    if verbose > 1:
        rrerrax = rrerrfig.axes[0]
        rrerrax.set_yscale('log')
        rrerrax.set_xscale('log')
    else:
        rrerrax = False
    rruncert = rt2*args.dx
    sigma = curve.sigma_for_fit(msd, msd_err, x=taus, plot=rrerrax, xnorm=1,
                                const=rruncert, ignore=[0]*(1-args.fit0)+[tmax],
                                verbose=verbose)
    if rrerrax:
        rrerrax.legend(loc='upper left', fontsize='x-small')
        if args.save:
            rrerrfig.savefig(saveprefix+'_rr-corr_sigma.pdf')

    form = [rr_form_total, rr_form_prog, rr_form_div][msdvec]
    model = Model(form)
    params = model.make_params()
    sources = {}
    frr = not (args.nn or args.rn)
    if args.colored:
        val, sources['TR'] = get_param_value('TR', fits)
        params['TR'].set(val, min=0, vary=frr or args.fittr)
    else:
        params['TR'].set(0, vary=False)
        sources['TR'] = None
    val, sources['DR'] = get_param_value('DR', fits)
    params['DR'].set(val, min=0, vary=frr)

    pname = 'lp'
    if args.fitv0 is True:
        val, sources[pname] = get_param_value(pname, fits)
        params[pname].set(val, min=0, vary=args.fitv0 or not args.rn)
    elif isinstance(args.fitv0, basestring):
        source = [{'disp': 'r', 'prog': 'p', 'div':'d'}[args.fitv0]+z]
        val, sources[pname] = get_param_value(pname, fits, source)
        params[pname].set(val, vary=False)
    else:
        val, sources[pname] = get_param_value(pname, fits)
        params[pname].set(val, vary=False)

    pname = 'DT'
    if args.fitdt is True:
        params[pname].set(0.1, min=0)
    elif isinstance(args.fitdt, float):
        params[pname].set(args.fitdt or (params['lp']*params['DR'])**2, vary=0)
    else:
        source = [{'disp': 'r', 'prog': 'p', 'div':'d'}[args.fitdt]+z]
        val, sources[pname] = get_param_value(pname, fits, source)
        params[pname].set(val, vary=False)

    result = model.fit(msd, params, 1/sigma, s=taus)

    fit, fitname, free, tex_fits = format_fit(result, model_name, sources)
    if free:
        fits[fit] = free
    else:
        fitname = ', '.join(fit_desc.get(f, str(f)) for f in fit if f)

    fitlabel = '\n'.join(tex_fits[:None if labels else 1])
    ax.plot(taus, result.best_fit, c=args.pcol, lw=1, label=fitlabel)

    ax.set_ylim(min(result.best_fit[0], msd[0]), result.best_fit[-1])
    xlim = ax.set_xlim(taus[0], tmax)
    if verbose > 1:
        rrerrax.set_xlim(taus[0], taus[-1])
        map(rrerrax.axvline, xlim)

    guide, transitions = limiting_regimes(taus, **result.best_values)
    ax.plot(taus, 2*guide, linestyle=':', color='black')
    ax.figure.tight_layout()
    ratio = get_aspect(ax)
    for (transition, slope) in zip(transitions, [2, 1]):
        xy = (taus[transition], 2*guide[transition])
        rotation = np.degrees(np.arctan(slope*ratio))
        ax.annotate(
            'slope {:d}'.format(slope), fontsize='small',
            xy=xy, xycoords='data', xytext=(0, 6), textcoords='offset points',
            rotation=rotation, rotation_mode='anchor', ha='left', va='bottom')

    DT_time = result.params['DT']/(result.params['lp']*result.params['DR'])**2
    DR_time = 1/result.params['DR']
    if xlim[0] < DT_time < xlim[1]:
        ax.axvline(DT_time, 0, 0.2, ls='--', c='k')
        ax.annotate(r'$D_T/v_0^2$', fontsize='small', ha='left', va='center',
                    xy=(DT_time, 0.1), xycoords=('data', 'axes fraction'),
                    xytext=(6, 0), textcoords='offset points')
    if xlim[0] < DR_time < xlim[1]:
        ax.axvline(DR_time, 0, 0.4, ls='--', c='k')
        ax.annotate(r'$1/D_R$', fontsize='small', ha='left', va='center',
                    xy=(DR_time, 0.2), xycoords=('data', 'axes fraction'),
                    xytext=(6, 0), textcoords='offset points')

    return fitname, result

if __name__ == '__main__':
    helpy.save_log_entry(readprefix, 'argv')
    meta = helpy.load_meta(readprefix)
    names = 'rcorner ncorners drcorner angsep dangsep'.split()
    helpy.sync_args_meta(args, meta, ['side', 'fps'] + names,
                         ['sidelength', 'fps'] + map('orient_{}'.format, names),
                         [1, 1, None, 2, None, None, None])
    if args.load:
        helpy.txt_to_npz(readprefix+'_CORNER'*args.corner+'_POSITIONS.txt',
                         verbose=True, compress=True)
        if args.orient or args.track:
            print 'NOTICE: not tracking, only converting file from txt to npz'
            print '        please run again without `-l` to track/orient'
        sys.exit()

    if args.track or args.orient:
        from scipy.spatial import cKDTree as KDTree
        if args.orient:
            if not args.track:
                print "Would you like to simultaneously track and find",
                print "orientations? (It's faster)",
                args.track = helpy.bool_input()
            pdata, cdata = helpy.load_data(readprefix, 'position corner')
        else:
            pdata = helpy.load_data(readprefix, 'position')
        pfsets = helpy.load_framesets(pdata)
        pftrees = {f: KDTree(helpy.consecutive_fields_view(pfset, 'xy'),
                             leafsize=50) for f, pfset in pfsets.iteritems()}
    if args.track:
        if args.maxdist is None:
            mdx = args.side if args.side > 1 else 20
            mdt = args.fps if args.fps <= 60 else 1
            args.maxdist = mdx/mdt
        meta.update(track_maxdist=args.maxdist, track_maxtime=args.giveup,
                    track_stub=args.stub, track_cut=args.cut)
        trackids = find_tracks(pdata, maxdist=args.maxdist, giveup=args.giveup,
                               n=args.number, stub=args.stub, cut=args.cut,
                               boundary=args.boundary, margin=args.side)
        trackids = remove_duplicates(trackids, data=pdata, verbose=args.verbose)
    else:
        trackids = None
    if args.orient:
        if args.rcorner is None:
            raise ValueError("argument -r/--rcorner is required")
        if args.drcorner is None:
            args.drcorner = meta.get('corner_kern')
        from orientation import get_angles
        cfsets = helpy.load_framesets(cdata)
        cftrees = {f: KDTree(helpy.consecutive_fields_view(cfset, 'xy'),
                             leafsize=50) for f, cfset in cfsets.iteritems()}
        odata, omask = get_angles(pdata, cdata, pfsets, cfsets, cftrees,
                                  nc=args.ncorners, rc=args.rcorner,
                                  drc=args.drcorner, ang=args.angsep,
                                  dang=args.dangsep)
        if args.save:
            save = saveprefix+'_ORIENTATION.npz'
            print "saving orientation data to",
            print save if verbose else os.path.basename(save)
            np.savez_compressed(save, odata=odata, omask=omask)
        orients = odata['orient']
    else:
        orients = None
    if args.track or args.orient:
        data = helpy.initialize_tdata(pdata, trackids, orients)
        if args.save:
            save = saveprefix+"_TRACKS.npz"
            print "saving track data to",
            print save if verbose else os.path.basename(save)
            np.savez_compressed(save, data=data)
    else:
        if readprefix == 'simulate':
            import simulate as sim
            spar = {'DR': 1/21, 'v0': 0.3678, 'DT': 0.01,
                    'fps': args.fps, 'side': args.side, 'size': 1000}
            print spar
            sdata = [sim.SimTrack(num=i, **spar)
                     for i in xrange(1, 1001)]
            data = np.concatenate([sdatum.track for sdatum in sdata])
            data['id'] = np.arange(len(data))
        else:
            data = helpy.load_data(readprefix, 'track')

    if args.check:
        path_to_tiffs, imstack, frames = helpy.find_tiffs(
            prefix=relprefix, frames=args.slice,
            load=True, verbose=args.verbose, maxsize=200e6)
        meta.update(path_to_tiffs=path_to_tiffs)
        tdata, cdata, odata = helpy.load_data(readprefix, 't c o')
        fisets = helpy.load_framesets(tdata, run_repair='interp')
        ftsets, fosets = helpy.load_framesets((tdata, odata))
        fcsets = helpy.load_framesets(cdata)
        animate_detection(imstack, ftsets, fcsets, fosets, fisets, meta=meta,
                          f_nums=frames, verbose=args.verbose, clean=args.clean)

    if args.side > 1 and args.dx > 0.1:
        # args.dx is in units of pixels
        args.dx /= args.side

    if args.nn or args.rn or args.rr:
        fits = {}
        helpy.sync_args_meta(args, meta, ['zoom'], ['corr_zoom'], [1])

    if args.msd or args.nn or args.rn:
        # for backwards compatability, clean up reverse, retire.
        if args.reverse or args.retire:
            if args.slice:
                raise ValueError('Cannot specify slice and reverse or retire')
            args.slice = slice(None, args.retire, -args.reverse or None)
        meta.pop('corr_reverse', None)
        meta.pop('corr_retire', None)
        meta.update(corr_stub=args.stub, corr_gaps=args.gaps,
                    corr_slice=args.slice)
        tracksets = helpy.load_tracksets(
            data, min_length=args.stub, track_slice=args.slice,
            run_track_orient=True, run_repair=args.gaps, verbose=args.verbose)

    if args.msd:
        msds, msdids = find_msds(
            tracksets, args.dt0, args.dtau, ret_vector=args.msdvec)
        meta.update(msd_dt0=args.dt0, msd_dtau=args.dtau, msd_stub=args.stub)
        if args.save:
            save = saveprefix+"_MSD.npz"
            print "saving msd data to",
            print save if verbose else os.path.basename(save)
            np.savez(save, msds=np.asarray(msds), msdids=np.asarray(msdids),
                     dt0=np.asarray(args.dt0), dtau=np.asarray(args.dtau))
    elif args.plotmsd or args.rr:
        if verbose:
            print "loading msd data from npz files"
        msdnpz = np.load(readprefix+"_MSD.npz")
        msds = msdnpz['msds']
        try:
            msdids = msdnpz['msdids']
        except KeyError:
            msdids = None
        helpy.sync_args_meta(args, meta, ['dt0', 'dtau'],
                             ['msd_dt0', 'msd_dtau'])

    if args.plotmsd:
        if verbose:
            print 'plotting msd now!'
        msd_taus, msd_mean, msd_err = mean_msd(
            msds, args.dtau, args.dt0, data['f'].max()+1, msdids=msdids,
            A=args.side**2, fps=args.fps, kill_flats=args.killflat,
            kill_jumps=args.killjump*args.side**2, show_tracks=args.showtracks,
            singletracks=args.singletracks)
        plot_msd(msd_taus, msd_mean, msd_err, prefix=saveprefix,
                 show=args.show, S=args.side, save=args.save, fps=args.fps,
                 fig=(5, 4) if args.clean else (8, 6), labels=not args.clean)

    if args.plottracks and not args.check:
        if verbose:
            print 'plotting tracks now!'
        if args.slice:
            allframes = data['f']
            nframes = allframes.max()+1
            frames = helpy.parse_slice(args.slice, index_array=True)
            mask = np.in1d(allframes, frames)
            bgind = frames[0]
        else:
            bgind = 1
            mask = None
        bgimage = helpy.find_tiffs(prefix=relprefix, frames=bgind,
                                   single=True, load=True)[1]
        if not args.singletracks:
            tracksets = helpy.load_tracksets(data, min_length=args.stub,
                                             run_repair=args.gaps,
                                             verbose=args.verbose)
            args.singletracks = tracksets.keys()
        if trackids is None:
            trackids = data['t']
        mask &= np.in1d(trackids, args.singletracks)
        plot_tracks(data, trackids, bgimage, mask=mask,
                    save=saveprefix*args.save, show=args.show)

    if args.save:
        helpy.save_meta(saveprefix, meta)

    if args.nn or args.rn or args.rr:
        fit_config, fit_desc = make_fitnames()

        labels = not args.clean

        rcParams_for_context = {
            'text.usetex': args.clean,
            'text.latex.preamble': list(set(
                plt.rcParams['text.latex.preamble'] + [r'\usepackage{amsmath}']
            )),
            'font.size': 8 if args.fig == 0 else plt.rcParams['font.size'],
        }

        with plt.rc_context(rc=rcParams_for_context):
            if args.fig == 0:
                fig, axs = plt.subplots(
                    ncols=3,
                    figsize=(7.0, 2.0),
                    gridspec_kw={'width_ratios': [2.5, 3, 4]})
            else:
                axs = [None]*3

            if args.nn or args.colored:
                print "====== <nn> ======"
                nn_result, fits, nn_ax = nn_plot(
                    tracksets, fits, args, ax=axs[0])

            if args.rn:
                print "====== <rn> ======"
                rn_result, fits, rn_ax = rn_plot(
                    tracksets, fits, args, ax=axs[1])

            if args.rr:
                print "====== <rr> ======"
                rr_result, fits, rr_ax = rr_plot(
                    msds, msdids, data, fits, args, axs[2])

            if args.fig == 0:
                fig.tight_layout(pad=0, w_pad=0, h_pad=0)
                for ax in axs:
                    move_axis_label(ax, 'y', x=None, y=2/3,
                                    ha='center', va='bottom')
            if args.save:
                helpy.save_fits(saveprefix, fits)

                save = '{}-correlations.pdf'.format(saveprefix)
                print 'saving all correlations figure to',
                print os.path.relpath(save, absprefix.replace(relprefix, ''))
                fig.savefig(save, bbox_inches='tight', pad_inches=0)

            if args.show:
                plt.show()
            else:
                plt.close('all')
    elif need_plt:
        if args.show:
            plt.show()
        else:
            plt.close('all')

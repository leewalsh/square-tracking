#!/usr/bin/env python
# encoding: utf-8

from __future__ import division

import sys
import itertools as it
from collections import defaultdict

import numpy as np
from scipy.optimize import curve_fit

import helpy
import correlation as corr

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
    arg('--maxdist', type=int, help="maximum single-frame travel distance in "
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
    arg('--killflat', type=int, default=0,
        help='Minimum growth factor for a single MSD track for inclusion')
    arg('--killjump', type=int, default=100000,
        help='Maximum initial jump for a single MSD track at first time step')
    arg('--stub', type=int, default=10, help='Minimum length (in frames) of a '
        'track for it to be included. default = 10')
    arg('-g', '--gaps', choices=['interp', 'nans', 'leave'], default='interp',
        nargs='?', const='nans', help="Gap handling: choose from %(choices)s. "
        "default is %(default)s, `-g` or `--gaps` alone gives %(const)s")
    arg('--singletracks', type=int, nargs='*', help='Only plot these tracks')
    arg('--showtracks', action='store_true', help='Show individual tracks')
    arg('--cut', action='store_true', help='Cut individual tracks at boundary')
    arg('--boundary', type=float, nargs=3, default=False,
        metavar=('X0', 'Y0', 'R'), help='Boundary for track cutting')
    arg('--nn', action='store_true', help='Calculate and plot <nn> correlation')
    arg('--rn', action='store_true', help='Calculate and plot <rn> correlation')
    arg('--rr', action='store_true', help='Calculate and plot <rr> correlation')
    arg('--fitdr', action='store_true', help='D_R as free parameter in rn fit')
    arg('--fitv0', action='store_true', help='v_0 as free parameter in MSD fit')
    arg('--dx', type=float, default=0.25, help='Positional measurement '
        'uncertainty (units are pixels unless SIDE is given and DX < 0.1)')
    arg('--dtheta', type=float, default=0.02,
        help='Angular measurement uncertainty (radians)')
    arg('-z', '--zoom', metavar="ZOOM", type=float, default=1,
        help="Factor by which to zoom out (in if ZOOM < 1)")
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

    need_plt = any([args.plottracks, args.plotmsd, args.check,
                    args.nn, args.rn, args.rr])
    args.verbose = args.verbose or 0
    args.quiet = args.quiet or 0
    args.verbose = max(0, args.verbose - args.quiet)
    verbose = args.verbose
    if verbose:
        print 'using prefix', prefix
    from warnings import filterwarnings
    warnlevel = {0: 'ignore', None: 'ignore', 1: 'once', 2: 'once', 3: 'error'}
    filterwarnings(warnlevel[verbose], category=RuntimeWarning,
                   module='numpy|scipy|matplot')
else:
    verbose = False


if __name__ != '__main__' or need_plt:
    if helpy.gethost() == 'foppl':
        import matplotlib
        matplotlib.use("agg")
    import matplotlib.pyplot as plt


sf = helpy.SciFormatter().format

pi = np.pi
twopi = 2*pi
rt2 = np.sqrt(2)


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
    mindist, mini = oldtree.query(thisdotxy, distance_upper_bound=maxdist)
    if mindist < maxdist:
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
        if boundary is None or boundary == [0.,]*3:
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
        ftsets = helpy.splitter(tset, fs, ret_dict=True)
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

    def plt_text(*args, **kwargs):
        return [plt.text(*arg, **kwargs) for arg in zip(*args)]

    side = meta.get('sidelength', 17)
    rc = meta.get('orient_rcorner')
    drc = meta.get('orient_drcorner') or np.sqrt(rc)
    txtoff = min(rc, side/2)/2

    title = "frame {:5d}\n{:3d} oriented, {:3d} tracked, {:3d} detected"
    fig, ax = plt.subplots(figsize=(12, 12))
    p = ax.imshow(imstack[0], cmap='gray')
    h, w = imstack[0].shape
    xlim, ylim = (0, w), (0, h)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
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
        ps = ax.scatter(y[tracked], x[tracked], c='r', zorder=.8)
        remove.append(ps)
        if not clean:
            us = ax.scatter(y[~tracked], x[~tracked], c='c', zorder=.8)
            cs = ax.scatter(xyc[:, 1], xyc[:, 0], c='g', s=10, zorder=.6)
            remove.extend([us, cs])

        # plot the orientations
        omask = np.isfinite(o)
        xyoo = xyo[omask]
        xo, yo, oo = xyoo.T
        q = ax.quiver(yo, xo, np.sin(oo), np.cos(oo), angles='xy', units='xy',
                      width=side/8, scale_units='xy', scale=1/side, zorder=.4)
        remove.append(q)
        if not clean:
            for dr in [-drc, 0, drc]:
                patches = helpy.draw_circles(xyo[:, 1::-1], rc+dr, ax=ax, lw=.5,
                                             color='g', fill=False, zorder=.5)
                remove.extend(patches)
        if fisets is not None:
            xyi, oi, tsi = [fisets[f_num][fld] for fld in ['xy', 'o', 't']]
            pim = fisets[f_num]['id'] == 0
            if np.any(pim):
                ips = ax.scatter(xyi[pim, 1], xyi[pim, 0],
                                 c=['pink', 'r'][clean], zorder=.7)
                itxt = plt_text(xyi[pim, 1], xyi[pim, 0]+txtoff,
                                tsi[pim].astype('S'), color='pink',
                                zorder=.9-clean, horizontalalignment='center')
                remove.extend(itxt)
                remove.append(ips)
            iq = ax.quiver(xyi[:, 1], xyi[:, 0], np.sin(oi), np.cos(oi),
                           angles='xy', units='xy', width=side/8,
                           scale_units='xy', scale=1/side, zorder=.3,
                           color=(.3*(1-clean),)*3)
            remove.append(iq)

        if fosets is not None:
            oc = helpy.quick_field_view(fosets[f_num], 'corner')
            oca = oc.reshape(-1, 2)
            ocs = ax.scatter(oca[:, 1], oca[:, 0], c='orange', s=10, zorder=1)
            remove.append(ocs)

            # corner displacements has shape (n_particles, n_corners, n_dim)
            cdisp = oc[omask] - xyoo[:, None, :2]
            cang = corr.dtheta(np.arctan2(cdisp[..., 1], cdisp[..., 0]))
            deg_str = np.nan_to_num(np.degrees(cang)).astype(int).astype('S')
            cx, cy = oc[omask].mean(1).T
            ctxt = plt_text(cy, cx, deg_str, color='orange', zorder=.9-clean,
                            horizontalalignment='center',
                            verticalalignment='center')
            remove.extend(ctxt)

        txt = plt_text(y[tracked], x[tracked]+txtoff, ts[tracked].astype('S'),
                       color='r', zorder=.9-clean, horizontalalignment='center')
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
                ocs.set_label('used corners')
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
        ind = lambda t: t['f'] if isf else np.where(~np.isnan(t[field]))[0]
        gaps = np.concatenate([np.diff(ind(tset)) - 1
                               for tsets in tracksetses
                               for tset in tsets.itervalues()])
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


def interp_nans(f, x=None, max_gap=10, inplace=False):
    """ Replace nans in function f(x) with their linear interpolation"""
    n = len(f)
    if n < 3:
        return f
    if f.ndim == 1:
        nans = np.isnan(f)
    elif f.ndim == 2:
        nans = np.isnan(f[:, 0])
    else:
        raise ValueError("Only 1d or 2d")
    if np.count_nonzero(nans) in (0, n):
        return f
    if not inplace:
        f = f.copy()
    ifin = (~nans).nonzero()[0]
    nf = len(ifin)
    if nf < 2:
        return f
    bef, aft = int(nans[0]), int(nans[-1])
    if bef or aft:
        bfin = np.empty(nf+bef+aft, int)
        if bef:
            bfin[0] = -1
        if aft:
            bfin[-1] = len(f)
        bfin[bef:-aft or None] = ifin
    else:
        bfin = ifin
    gaps = np.diff(bfin) - 1
    inan = ((gaps > 0) & (gaps <= max_gap)).nonzero()[0]
    if len(inan) < 1:
        return f
    gaps = gaps[inan]
    inan = np.repeat(inan, gaps)
    inan = np.concatenate(map(range, gaps)) + bfin[inan] + 1
    xnan, xfin = (inan, ifin) if x is None else (x[inan], x[ifin])
    if not inplace:
        f = f.copy()
    for c in f.T if f.ndim > 1 else [f]:
        c[inan] = np.interp(xnan, xfin, c[ifin])
    return f


def fill_gaps(tracksets, max_gap=10, interp=['xy','o'], inplace=True, verbose=False):
    if not inplace:
        tracksets = {t: s.copy() for t, s in tracksets.iteritems()}
    if verbose:
        print 'filling gaps with nans'
        if interp:
            print 'and interpolating nans in', ', '.join(interp)
    for t, tset in tracksets.items():
        if verbose:
            print '\t{} ({})'.format(t, len(tset)),
        fs = tset['f']
        gaps = np.diff(fs) - 1
        mx = gaps.max()
        if not mx:
            if 'o' in interp:
                interp_nans(tset['o'], tset['f'], inplace=True)
            continue
        elif mx > max_gap:
            if verbose:
                print "dropped gap {}".format(mx),
            tracksets.pop(t)
            continue
        gapi = gaps.nonzero()[0]
        gaps = gaps[gapi]
        gapi = np.repeat(gapi, gaps)
        missing = np.full(len(gapi), np.nan, tset.dtype)
        if verbose:
            print "filled {:2} gaps ({:3} frames, max {})".format(
                len(gapi), len(gaps), mx)
        missing['f'] = np.concatenate(map(range, gaps)) + fs[gapi] + 1
        missing['t'] = t
        tset = np.insert(tset, gapi+1, missing)
        if interp:
            for field in interp:
                view = helpy.consecutive_fields_view(tset, field)
                interp_nans(view, inplace=True)
        tracksets[t] = tset
    return tracksets


def plot_tracks(data, trackids=None, bgimage=None, mask=None,
                fignum=None, save=True, show=True):
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
    fig = plt.figure(fignum)
    ax = fig.gca()
    if bgimage is not None:
        if isinstance(bgimage, basestring):
            bgimage = plt.imread(bgimage)
        ax.imshow(bgimage, cmap=plt.cm.gray, origin='upper')
    if trackids is None:
        trackids = data['t']
    if mask is None:
        mask = np.where(trackids >= 0)
    data = data[mask]
    trackids = trackids[mask]
    ax.scatter(data['y'], data['x'], c=trackids%12, marker='o', alpha=.5, lw=0)
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
    tfsets = helpy.splitter(trackset, trackset['f'], ret_dict=True)
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
            if verbose: print "fail"
            continue
        nt0s += 1.0
    return totsqdisp/nt0s if nt0s else None


def trackmsd(trackset, dt0, dtau):
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
        return corr.msd(xy, ret_taus=True)

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


def find_msds(tracksets, dt0, dtau, min_length=0):
    """ Calculates the MSDs for all tracks

        parameters
        ----------
        dt0, dtau : see documentation for `trackmsd`
        tracksets : dict of subsets of the data for a given track
        min_length : a cutoff to exclude tracks shorter than min_length

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
        tmsd = trackmsd(tracksets[t], dt0, dtau)
        if len(tmsd) > 1:
            msds.append(tmsd)
            msdids.append(t)
        elif verbose:
            print "\ntrack {}'s msd is too short!".format(t)
    if verbose:
        print
    return msds, msdids


def mean_msd(msds, taus, msdids=None, tnormalize=False, fps=1, A=1,
             kill_flats=1, kill_jumps=1e9, errorbars=False,
             show_tracks=False, singletracks=None):
    """ return the mean of several track msds """

    # msd has shape (number of tracks, length of tracks)
    msdshape = (len(singletracks) if singletracks else len(msds),
                max(map(len, msds)))
    msd = np.full(msdshape, np.nan, float)
    if verbose:
        print 'msdshape:', msdshape
        print 'taushape:', taus.shape,
    taus = taus[:msdshape[1]]
    if verbose:
        print '-->', taus.shape

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
        tmsdt, tmsdd = np.asarray(tmsd).T
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
    if show_tracks:
        plt.plot(taus/fps, (msd/(taus/fps)**tnormalize).T/A, 'b', alpha=.2)
    return taus, msd_mean, msd_err


def plot_msd(msds, msdids, dtau, dt0, nframes, prefix='', tnormalize=False,
             xscale='log', meancol='', figsize=(8, 6), title=None, xlim=None,
             ylim=None, lw=1, legend=False, errorbars=False, show_tracks=True,
             singletracks=None, sys_size=0, kill_flats=0, kill_jumps=1e9, S=1,
             fps=1, ang=False, save='', show=True,):
    """ Plots the MS(A)Ds """
    A = 1 if ang else S**2
    if verbose:
        print "using dtau = {}, dt0 = {}".format(dtau, dt0)
        print "using S = {} pixels, thus A = {} px^2".format(S, A)
    try:
        dtau = np.asscalar(dtau)
    except AttributeError:
        pass
    if isinstance(dtau, (float, np.float)):
        taus = helpy.farange(dt0, nframes+1, dtau)
    elif isinstance(dtau, (int, np.int)):
        taus = np.arange(dtau, nframes+1, dtau, dtype=float)
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()

    # Get the mean of msds
    msd_taus, msd_mean, msd_err = mean_msd(msds, taus, msdids, fps=fps, A=A,
        kill_flats=kill_flats, kill_jumps=kill_jumps, show_tracks=show_tracks,
        singletracks=singletracks, tnormalize=tnormalize, errorbars=errorbars)
    if verbose:
        print '     taus:', taus.shape
        print ' from msd:', msd_taus.shape
        print '      msd:', msd_mean.shape
        taus_crop = taus[:len(msd_mean)]
        print 'shortened:', taus_crop.shape
        print '   match?:', np.allclose(taus_crop, msd_taus)
    taus = msd_taus / fps
    msd = msd_mean / A

    if tnormalize:
        ax.plot(taus, msd/taus**tnormalize, meancol,
                label="Mean Sq {}Disp/Time{}".format(
                      "Angular " if ang else "",
                      "^{}".format(tnormalize) if tnormalize != 1 else ''))
        ax.plot(taus, msd[0]*taus**(1-tnormalize)/dtau,
                'k-', label="ref slope = 1", lw=2)
        ax.plot(taus, (twopi**2 if ang else 1)/(taus)**tnormalize,
                'k--', lw=2, label=r"$(2\pi)^2$" if ang else
                ("One particle area" if S > 1 else "One Pixel"))
        ax.set_xscale(xscale)
        ax.set_ylim([0, 1.3*np.max(msd/taus**tnormalize)])
    else:
        ax.loglog(taus, msd, c=meancol, lw=lw,
                  label="Mean Squared {}Displacement".format('Angular '*ang))
        #ax.loglog(taus, msd[0]*taus/dtau/2, meancol+'--', lw=2,
        #          label="slope = 1")
    if errorbars:
        msd_err /= A
        ax.errorbar(taus, msd/taus**tnormalize, msd_err/taus**tnormalize, lw=lw,
                    c=meancol, capthick=0, elinewidth=1, errorevery=errorbars)
    if sys_size:
        ax.axhline(sys_size, ls='--', lw=.5, c='k', label='System Size')
    ax.set_title(title or "Mean Sq {}Disp".format("Angular " if ang else ""))
    xlabel = '$tf$' if 1 < fps < 60 else 'Time ({}s)'.format('frame'*(fps == 1))
    ax.set_xlabel(xlabel, fontsize='x-large')
    ylabel = 'Squared {}Displacement ({})'.format(
        'Angular '*ang,
        '$rad^2$' if ang else 'particle area' if S > 1 else 'square pixels')
    ax.set_ylabel(ylabel, fontsize='x-large')
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
    return fig, ax, taus, msd, msd_err


def propagate(func, uncert, size=1000, domain=1, plot=False, verbose=False):
    if size >= 10:
        size = np.log10(size)
    size = int(round(size))
    print '1e{}'.format(size),
    size = 10**size
    if np.isscalar(uncert):
        uncert = [uncert]*2
    domain = np.atleast_1d(domain)
    domains = []
    for dom in domain:
        if np.isscalar(dom):
            domains.append((0, dom))
        elif len(dom) == 1:
            domains.append((0, dom[0]))
        else:
            domains.append(dom)
    #x_true = np.row_stack([np.linspace(*dom, num=size) for dom in domains])
    x_true = np.row_stack([np.random.rand(size)*(dom[1]-dom[0]) + dom[0]
                           for dom in domains])
    x_err = np.row_stack([np.random.normal(scale=u, size=size) if u > 0 else
                          np.zeros(size) for u in uncert])
    x_meas = x_true + x_err
    if verbose:
        print
        print 'x_true:', x_true.shape, 'min', x_true.min(1), 'max', x_true.max(1)
        print 'x_meas:', x_meas.shape, 'min', x_meas.min(1), 'max', x_meas.max(1)
        print 'x_err: ', x_err.shape, 'min', x_err.min(1), 'max', x_err.max(1)
    xfmt = 'x: [{d[1][0]:5.2g}, {d[1][1]:5.2g}) +/- {dx:<5.4g} '
    thetafmt = 'theta: [{d[0][0]:.2g}, {d[0][1]:.3g}) +/- {dtheta:<5.4g} '
    if func == 'nn':
        dtheta, _ = uncert
        print thetafmt.format(dtheta=dtheta, d=domains)+'->',
        f = lambda x: np.cos(x[0])*np.cos(x[1])
        f_uncert = dtheta/rt2
    elif func == 'rn':
        dtheta, dx = uncert
        print (thetafmt+xfmt+'->').format(dtheta=dtheta, dx=dx, d=domains),
        f = lambda x: np.cos(x[0])*x[1]
        f_uncert = np.sqrt(dx**2 + (x_true[1]*dtheta)**2).mean()/rt2
    elif func == 'rr':
        dx, _ = uncert
        print xfmt.format(dx=dx, d=domains)+'->',
        f = lambda x: x[0]*x[1]  # (x[0]-x[0].mean())*(x[1]-x[1].mean())
        f_uncert = rt2*dx*np.sqrt((x_true[0]**2).mean())
    else:
        f_uncert = None
    f_true = f(x_true)
    f_meas = f(x_meas)
    f_err = f_meas - f_true
    if False and 'r' in func:
        #print 'none',
        #f_err /= np.maximum(np.abs(f_meas), np.abs(f_true))
        #print 'maxm',
        #f_err /= np.sqrt(np.abs(f_meas*f_true))
        #print 'geom',
        f_err /= np.sqrt(f_meas**2 + f_true**2)/2
        print 'quad',
        #f_err /= f_meas
        #print 'meas',
        #f_err /= f_true
        #print 'true',
    if plot:
        fig = plt.gcf()
        fig.clear()
        ax = plt.gca()
        if size <= 10000:
            ax.scatter(f_true, f_err, marker='.', c='k', label='f_err v f_true')
        else:
            ax.hexbin(f_true, f_err)
    nbins = 25 if plot else 7
    f_bins = np.linspace(f_true.min(), f_true.max()*(1+1e-8), num=1+nbins)
    f_bini = np.digitize(f_true, f_bins)
    ubini = np.unique(f_bini)
    f_stds = [f_err[f_bini == i].std() for i in ubini]
    if plot:
        ax.plot((f_bins[1:]+f_bins[:-1])/2, f_stds, 'or')
    if verbose:
        print
        print '[', ', '.join(map('{:.3g}'.format, f_bins)), ']'
        print np.row_stack([ubini, np.bincount(f_bini)[ubini]])
        print '[', ', '.join(map('{:.3g}'.format, f_stds)), ']'
    f_err_std = f_err.std()
    ratio = f_uncert/f_err_std
    missed = ratio - 1
    print '{:< 9.4f}/{:< 9.4f} = {:<.3f} ({: >+7.2%})'.format(
            f_uncert, f_err_std, ratio, missed),
    print '='*int(-np.log10(np.abs(missed)))
    if verbose:
        print
    #[tracks.propagate('rr', dx, 5, 2*[i]) for dt, dx, i in it.product(
    #     [.01, .1], [.1, 1, 10], [(1,100), (10,100),(90,100)])]
    #[tracks.propagate('rn', (dt, dx), 5, [(0, 2*np.pi), (10**(i-2), 10**i)])
    # for dt, dx, i in it.product([0, .01, .1], [.1, 1, 10], [2,3,4])]
    #[tracks.propagate('nn', dt, 5, [(0, 2*np.pi),(10**(i-1), 10**i)])
    # for dt, dx, i in it.product([.01, .1], [0, .1, 1, 10], [2,3,4])]
    return f_err_std


def sigma_for_fit(arr, std_err, std_dev=None, added=None, x=None, plot=False,
                  relative=None, const=None, xnorm=None, ignore=None):
    if x is None:
        x = np.arange(len(arr))
    if ignore is not None:
        x0 = np.searchsorted(x, 0)
        ignore = np.array([x0-1, x0, x0+1][x0 < 1:])
    if plot:
        ax = plot if isinstance(plot, plt.Axes) else plt.gca()
        plot = True
        plotted = []
        colors = it.cycle('rgbcmyk')
    try:
        mods = it.product(const, relative, xnorm)
    except TypeError:
        mods = [(const, relative, xnorm)]
    for const, relative, xnorm in mods:
        signame = 'std_err'
        sigma = std_err.copy()
        sigma[ignore] = np.inf
        if plot:
            c = colors.next()
            if signame not in plotted:
                ax.plot(x, std_err, '.'+c, label=signame)
                plotted.append(signame)
        if relative:
            sigma /= arr
            signame += '/arr'
            if plot and signame not in plotted:
                ax.plot(x, sigma, ':'+c, label=signame)
                plotted.append(signame)
        if const is not None:
            isconst = np.isscalar(const)
            offsetname = '({:.3g})'.format(const) if isconst else 'const'
            sigma = np.hypot(sigma, const)
            signame = 'sqrt({}^2 + {}^2)'.format(signame, offsetname)
            if verbose:
                print 'adding const',
                print 'sqrt(sigma^2 + {}^2)'.format(offsetname)
            if plot and signame not in plotted:
                ax.plot(x, sigma, '-'+c, label=signame)
                if isconst:
                    ax.axhline(const, ls='--', c=c, label='const')
                else:
                    ax.plot(x, const, '^'+c, label='const')
        if xnorm:
            if xnorm == 'log':
                label = 'log(1 + x)'
                xnorm = np.log1p(x)
            elif xnorm == 1:
                label = 'x'
                xnorm = x
            else:
                label = 'x^{}'.format(xnorm)
                xnorm = x**xnorm
            signame += '*' + label
            sigma *= xnorm
            if plot and label not in plotted:
                ax.plot(x, xnorm, '--'+c, label=label)
                plotted.append(label)
            if plot and signame not in plotted:
                ax.plot(x, sigma, '-.'+c, label=signame)
                plotted.append(signame)
        if verbose:
            print 'sigma =', signame
            print 'nan_info',
            helpy.nan_info(sigma, True)
            print 'sigprint', sigprint(sigma)
    if plot:
        ax.legend(loc='upper left', fontsize='x-small')
    return sigma


def plot_params(params, s, ys, xs=None, by='source', ax=None, **pltargs):
    """plot the fit parameters from correlations and histograms

    plot parameters `y` vs `x` selected by `s`

    parameters
    params : structured array with keys for parameter name, from meta_meta
    x, y, s : keys or lists of keys to plot on each axis (`s` will be color?)
    """

    if ax is None:
        fig, ax = plt.subplots()
    if xs is None:
        xs = (np.arange(len(s)) for y in ys)
    else:
        xs = (ps[x] for x in xs)

    ps = params[np.in1d(params[by], s)]
    for x, y in zip(xs, ys):
        ax.plot(x, ps[y], label=y, **pltargs)
    ax.legend()

    if by == 'dict':
        for k in s:
            ps = params[k]
            ax.scatter(ps[x], ps[y], **pltargs)
            pltargs['label'] = None  # only label once

    return ax


def plot_parametric(params, sources, xy=None, scale='log', lims=(1e-3, 1),
                    label_source=False, savename=''):
    ps = params[np.in1d(params['source'], sources)]
    lims = {'DR': (0, 0.1), 'v0': (0, 0.3), 'DT': (0, 0.01)}[xy]
    xys = {'DR': [('fit_nn_DR', 'fit_vo_DR', 'r', 'o', '$D_R$')],
          'v0': [('fit_rn_v0', 'fit_vn_v0', 'g', '^', '$v_0$')],
          'DT': [('fit_rr_DT', 'fit_vt_DT', 'b', 's', '$D_T$')]}[xy]
                # ('fit_rr_DT', 'fit_vt_DT', 'b', 'D', None)]

    fig, ax = plt.subplots(figsize=(4,4))
    for x, y, c, m, l in xys:
        ax.scatter(ps[y], ps[x], marker=m, c=c, label=l)
        if label_source:
            [plt.text(p[y], p[x], p['source']) for p in ps]
        # ax = plot_parametric(params, x, y, sources, 'source', ax=ax,

    ax.set_xlabel('Noise statistics from velocity', usetex=True)
    ax.set_ylabel('Fits from correlation functions', usetex=True)
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    ax.set_aspect('equal', adjustable='box')
    xlim = ax.set_xlim(lims)
    ylim = ax.set_ylim(lims)
    ax.plot(xlim, ylim, '--k', alpha=0.7)
    ax.legend(loc='upper left')
    if savename:
        fig.savefig('/Users/leewalsh/Physics/Squares/colson/Output/stats/'
                    'parametric_{}.pdf'.format(savename))
    return ax


def sigprint(sigma):
    sigfmt = ('{:7.4g}, '*5)[:-2].format
    mn, mx = sigma.min(), sigma.max()
    return sigfmt(mn, sigma.mean(), mx, sigma.std(ddof=1), mx/mn)


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
        pfsets = helpy.splitter(pdata, ret_dict=True)
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
        from orientation import get_angles
        cfsets = helpy.splitter(cdata, ret_dict=True)
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
        data = helpy.load_data(readprefix, 'track')

    if args.check:
        path_to_tiffs, imstack, frames = helpy.find_tiffs(
                prefix=relprefix, frames=args.slice,
                load=True, verbose=args.verbose)
        meta.update(path_to_tiffs=path_to_tiffs)
        tdata, cdata, odata = helpy.load_data(readprefix, 't c o')
        idata = helpy.load_tracksets(tdata, run_fill_gaps='interp')
        idata = np.concatenate(idata.values())
        fisets = helpy.splitter(idata, 'f', noncontiguous=True, ret_dict=True)
        ftsets, fosets = helpy.splitter((tdata, odata), 'f')
        fcsets = helpy.splitter(cdata)
        animate_detection(imstack, ftsets, fcsets, fosets, fisets, meta=meta,
                          f_nums=frames, verbose=args.verbose, clean=args.quiet)

    if args.side > 1 and args.dx > 0.1:
        # args.dx is in units of pixels
        args.dx /= args.side

    if args.rr or args.nn or args.rn:
        fit_source = {}

    if args.msd or args.nn or args.rn:
        meta.update(corr_stub=args.stub, corr_gaps=args.gaps)
        tracksets = helpy.load_tracksets(
            data, min_length=args.stub, run_track_orient=True,
            run_fill_gaps=args.gaps, verbose=args.verbose)

    if args.msd:
        msds, msdids = find_msds(
            tracksets, args.dt0, args.dtau, min_length=args.stub)
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
        plot_msd(msds, msdids, args.dtau, args.dt0, data['f'].max()+1,
                 tnormalize=False, prefix=saveprefix, show=args.show,
                 show_tracks=args.showtracks, singletracks=args.singletracks,
                 kill_flats=args.killflat, S=args.side, save=args.save,
                 kill_jumps=args.killjump*args.side**2, fps=args.fps)

    if args.plottracks:
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
                                             run_fill_gaps=args.gaps,
                                             verbose=args.verbose)
            args.singletracks = tracksets.keys()
        if trackids is None:
            trackids = data['t']
        mask &= np.in1d(trackids, args.singletracks)
        plot_tracks(data, trackids, bgimage, mask=mask,
                    save=saveprefix*args.save, show=args.show)

    if args.save:
        helpy.save_meta(saveprefix, meta)


if __name__ == '__main__' and args.nn:
    print "====== <nn> ======"
    # Calculate the <nn> correlation for all the tracks in a given dataset

    corr_args = {'cumulant': False, 'norm': False}
    if args.verbose:
        print 'calculating <nn> correlations for track'
        coscorrs = []
        sincorrs = []
        for t, trackset in tracksets.iteritems():
            print t,
            o = trackset['o']
            if args.verbose > 1:
                print o.shape,
            sys.stdout.flush()
            cos = np.cos(o)
            sin = np.sin(o)
            coscorr = corr.autocorr(cos, **corr_args)
            sincorr = corr.autocorr(sin, **corr_args)
            coscorrs.append(coscorr)
            sincorrs.append(sincorr)
    else:
        coscorrs = [corr.autocorr(np.cos(trackset['o']), **corr_args)
                    for trackset in tracksets.values()]
        sincorrs = [corr.autocorr(np.sin(trackset['o']), **corr_args)
                    for trackset in tracksets.values()]

    # Gather all the track correlations and average them
    allcorrs = coscorrs + sincorrs
    allcorrs, meancorr, errcorr = helpy.avg_uneven(allcorrs, pad=True)
    taus = np.arange(len(meancorr))/args.fps
    tmax = int(50*args.zoom)
    fmax = np.searchsorted(taus, tmax)
    if verbose > 1:
        nnerrfig, nnerrax = plt.subplots()
        nnerrax.set_yscale('log')
    else:
        nnerrax = False
    nnuncert = args.dtheta/rt2
    sigma = sigma_for_fit(meancorr, errcorr, x=taus,
                          plot=nnerrax, const=nnuncert, ignore=True)
    if nnerrax:
        nnerrax.legend(loc='lower left', fontsize='x-small')
        nnerrfig.savefig(saveprefix+'_nn-corr_sigma.pdf')

    # Fit to functional form:
    D_R = meta.get('fit_nn_DR', 0.1)
    p0 = [D_R]

    def fitform(s, DR):
        return 0.5*np.exp(-DR*s)

    fitstr = r"$\frac{1}{2}e^{-D_R t}$"
    try:
        popt, pcov = curve_fit(fitform, taus[:fmax], meancorr[:fmax],
                               p0=p0, sigma=sigma[:fmax])
    except RuntimeError as e:
        print "RuntimeError:", e.message
        print "Using inital guess", p0
        popt = p0
    D_R = float(popt[0])
    print "Fits to <nn>:"
    print '   D_R: {:.4g}'.format(D_R)
    fit_source['DR'] = 'nn'
    if args.save:
        helpy.save_meta(saveprefix, fit_nn_DR=D_R)

    fig, ax = plt.subplots()
    fit = fitform(taus, *popt)
    plot_individual = True
    if plot_individual:
        ax.plot(taus, allcorrs.T, 'b', alpha=.2)
    ax.errorbar(taus, meancorr, errcorr, None, c=(1, 0.4, 0), lw=3,
                label="Mean Orientation Autocorrelation",
                capthick=0, elinewidth=0.5, errorevery=3)
    fitinfo = sf("$D_R={0:.4T}$, $D_R^{{-1}}={1:.3T}$", D_R, 1/D_R)
    ax.plot(taus, fit, c=(0.25, 0.5, 0), lw=2, label=fitstr + '\n' + fitinfo)
    ax.set_xlim(0, tmax)
    ax.set_ylim(fit[fmax], 1)
    ax.set_yscale('log')

    ax.set_title("Orientation Autocorrelation\n"+prefix)
    ax.set_ylabel(r"$\langle \hat n(t) \hat n(0) \rangle$")
    ax.set_xlabel("$tf$")
    ax.legend(loc='upper right' if args.zoom <= 1 else 'lower left',
              framealpha=1)

    if args.save:
        save = saveprefix+'_nn-corr.pdf'
        print 'saving <nn> correlation plot to',
        print save if verbose else os.path.basename(save)
        fig.savefig(save)

if __name__ == '__main__' and args.rn:
    print "====== <rn> ======"
    # Calculate the <rn> correlation for all the tracks in a given dataset
    # TODO: fix this to combine multiple datasets (more than one prefix)

    if not args.nn:
        D_R = meta.get('fit_nn_DR', meta.get('fit_rn_DR', 1/16))

    corr_args = {'side': 'both', 'ret_dx': True,
                 'cumulant': (True, False), 'norm': 0}

    xcoscorrs = [corr.crosscorr(trackset['x']/args.side, np.cos(trackset['o']),
                 **corr_args) for trackset in tracksets.values()]
    ysincorrs = [corr.crosscorr(trackset['y']/args.side, np.sin(trackset['o']),
                 **corr_args) for trackset in tracksets.values()]

    # Align and merge them
    fmax = int(2*args.fps/D_R*args.zoom)
    fmin = -fmax
    allcorrs = xcoscorrs + ysincorrs
    # TODO: align these so that even if a track doesn't reach the fmin edge,
    # that is, if f.min() > fmin for a track, then it still aligns at zero
    allcorrs = [rn[np.searchsorted(f, fmin):np.searchsorted(f, fmax)]
                for f, rn in allcorrs if f.min() <= fmin]
    allcorrs, meancorr, errcorr = helpy.avg_uneven(allcorrs, pad=True)
    taus = np.arange(fmin, fmax)/args.fps
    if verbose > 1:
        rnerrfig, rnerrax = plt.subplots()
    else:
        rnerrax = False
    rnuncert = np.hypot(args.dtheta, args.dx)/rt2
    sigma = sigma_for_fit(meancorr, errcorr, x=taus, plot=rnerrax,
                          const=rnuncert, ignore=True)
    if rnerrax:
        rnerrax.legend(loc='upper center', fontsize='x-small')
        rnerrfig.savefig(saveprefix+'_rn-corr_sigma.pdf')

    def fitform(s, v_D, D=D_R):
        return 0.5*np.sign(s)*v_D*(1 - corr.exp_decay(np.abs(s), 1/D))

    fitstr = r'$\frac{v_0}{2D_R}(1 - e^{-D_R|s|})\operatorname{sign}(s)$'

    # p0 = [v_0/D_R, D_R]
    v0 = meta.get('fit_rn_v0', 0.1)
    p0 = [v0]
    if not args.nn or args.fitdr:
        p0 += [D_R]

    try:
        popt, pcov = curve_fit(fitform, taus, meancorr, p0=p0, sigma=sigma)
    except RuntimeError as e:
        try:
            p0[0] *= -1  # maybe dots are backwards, try starting from  v0 < 0
            popt, pcov = curve_fit(fitform, taus, meancorr, p0=p0, sigma=sigma)
        except RuntimeError as e:
            p0[0] *= -1  # restore original positive value
            print "RuntimeError:", e.message
            print "Using inital guess", p0
            popt = p0

    print "Fits to <rn>:"
    nfree = len(popt)
    if nfree > 1:
        D_R = popt[1]
        fit_source['DR'] = 'rn'
    v0 = D_R*popt[0]
    fit_source['v0'] = 'rn'
    print '\n'.join([' v0/D_R: {:.4g}',
                     '    D_R: {:.4g}'][:nfree]).format(*popt)
    print "Giving:"
    print '\n'.join(['     v0: {:.4f}',
                     'D_R(rn): {:.4f}'][:nfree]
                    ).format(*[v0, D_R][:nfree])
    if args.save:
        if nfree == 1:
            psources = '_nn'
            meta_fits = {'fit'+psources+'_rn_v0': v0}
        elif nfree == 2:
            psources = ''
            meta_fits = {'fit_rn_v0': v0, 'fit_rn_DR': D_R}
        helpy.save_meta(saveprefix, meta_fits)

    fig, ax = plt.subplots()
    fit = fitform(taus, *popt)
    plot_individual = True
    sgn = np.sign(v0)
    if plot_individual:
        ax.plot(taus, sgn*allcorrs.T, 'b', alpha=.2)
    ax.errorbar(taus, sgn*meancorr, errcorr, None, c=(1, 0.4, 0), lw=3,
                label="Mean Position-Orientation Correlation",
                capthick=0, elinewidth=0.5, errorevery=3)
    # fitinfo = ', '.join(['$v_0$: {:.3f}', '$t_0$: {:.3f}', '$D_R$: {:.3f}'
    fitinfo = sf(', '.join(['$v_0={0:.3T}$', '$D_R={1:.3T}$'][:nfree]),
                 *(abs(v0), D_R)[:nfree])
    ax.plot(taus, sgn*fit, c=(0.25, 0.5, 0), lw=2, label=fitstr + '\n' + fitinfo)

    ylim = ax.set_ylim(1.5*fit.min(), 1.5*fit.max())
    xlim = ax.set_xlim(taus.min(), taus.max())
    tau_R = 1/D_R
    if xlim[0] < tau_R < xlim[1]:
        ax.axvline(tau_R, 0, 2/3, ls='--', c='k')
        ax.text(tau_R, 1e-2, ' $1/D_R$')

    ax.set_title("Position - Orientation Correlation")
    ax.set_ylabel(r"$\langle \vec r(t) \hat n(0) \rangle / \ell$")
    ax.set_xlabel("$tf$")
    ax.legend(loc='upper left', framealpha=1)

    if args.save:
        save = saveprefix + psources + '_rn-corr.pdf'
        print 'saving <rn> correlation plot to',
        print save if verbose else os.path.basename(save)
        fig.savefig(save)

if __name__ == '__main__' and args.rr:
    print "====== <rr> ======"
    fig, ax, taus, msd, errcorr = plot_msd(
        msds, msdids, args.dtau, args.dt0, data['f'].max()+1, save=False,
        show=False, tnormalize=0, errorbars=3, prefix=saveprefix,
        show_tracks=True, meancol=(1, 0.4, 0), lw=3, singletracks=args.singletracks,
        fps=args.fps, S=args.side, kill_flats=args.killflat,
        kill_jumps=args.killjump*args.side**2)

    tmax = int(200*args.zoom)
    fmax = np.searchsorted(taus, tmax)
    if verbose > 1:
        rrerrax = rrerrfig.axes[0]
        rrerrax.set_yscale('log')
        rrerrax.set_xscale('log')
    else:
        rrerrax = False
    rruncert = rt2*args.dx
    sigma = sigma_for_fit(msd, errcorr, x=taus, plot=rrerrax,
                          const=rruncert, xnorm=1, ignore=True)
    if rrerrax:
        rrerrax.legend(loc='upper left', fontsize='x-small')
        if args.save:
            rrerrfig.savefig(saveprefix+'_rr-corr_sigma.pdf')

    # Fit to functional form:
    D_T = meta.get('fit_rr_DT', 0.01)
    p0 = [D_T]
    if args.fitv0 or not args.rn:
        v0 = meta.get('fit_rn_v0', 0.1)
        sgn = np.sign(v0)
        p0 += [v0]
    if not (args.nn or args.rn):
        D_R = meta.get('fit_nn_DR', meta.get('fit_rn_DR', 1/16))
        p0 += [D_R]

    def fitform(s, D, v=v0, DR=D_R):
        return 2*(v/DR)**2 * (DR*s + np.exp(-DR*s) - 1) + 2*D*s

    def limiting_regimes(s, D, v=v0, DR=D_R):
        v *= v  # v is squared everywhere
        tau_T = D/v
        tau_R = 1/DR
        taus = (tau_T, tau_R)

        early = 2*D*s       # s < tau_T
        middle = v*s**2         # tau_T < s < tau_R
        late = 2*(v/DR + D)*s               # tau_R < s
        lines = np.choose(np.searchsorted(taus, s), [early, middle, late])

        tau_f = np.searchsorted(s, taus)
        if tau_f[-1] >= len(s):
            tau_f = tau_f[0]
        lines[tau_f] = np.nan
        return lines

    fitstr = r"$2(v_0/D_R)^2 (D_Rt + e^{{-D_Rt}} - 1) + 2D_Tt$"

    try:
        popt, pcov = curve_fit(fitform, taus[:fmax], msd[:fmax],
                               p0=p0, sigma=sigma[:fmax])
    except RuntimeError as e:
        print "RuntimeError:", e.message
        if not args.fitv0:
            p0 = [0, v0]
        print "Using inital guess", p0
        popt = p0

    print "Fits to <rr>:"
    nfree = len(popt)
    D_T = popt[0]
    fit_source['DT'] = 'rr'
    if nfree > 1:
        v0 = popt[1]
        fit_source['v0'] = 'rr'
        if nfree > 2:
            D_R = popt[2]
            fit_source['DR'] = 'rr'
    print '\n'.join(['   D_T: {:.3g}',
                     'v0(rr): {:.3g}',
                     '   D_R: {:.3g}'][:nfree]).format(*popt)
    if nfree > 1:
        print "Giving:"
        print "v0/D_R: {:.3g}".format(v0/D_R)
    if args.save:
        if nfree == 1:
            psources = '_{DR}_{v0}'.format(**fit_source)
            meta_fits = {'fit'+psources+'_rr_DT': D_T}
        elif nfree == 2:
            psources = '_{DR}'.format(**fit_source)
            meta_fits = {'fit'+psources+'_rr_DT': D_T,
                         'fit'+psources+'_rr_v0': v0}
        elif nfree == 3:
            psources = ''
            meta_fits = {'fit_rr_DT': D_T,
                         'fit_rr_v0': v0,
                         'fit_rr_DR': D_R}
        helpy.save_meta(saveprefix, meta_fits)
    fit = fitform(taus, *popt)

    fitinfo = sf(', '.join(["$D_T={0:.3T}$",
                            "$v_0={1:.3T}$",
                            "$D_R={2:.3T}$"][:nfree]),
                 *(popt*[1, sgn, 1][:nfree]))
    ax.plot(taus, fit, c=(0.25, 0.5, 0), lw=2, label=fitstr + "\n" + fitinfo)

    guide = limiting_regimes(taus, *popt)
    ax.plot(taus, guide, '--k', lw=1)

    ylim = ax.set_ylim(min(fit[0], msd[0]), fit[np.searchsorted(taus, tmax)])
    xlim = ax.set_xlim(taus[0], tmax)
    if verbose > 1:
        rrerrax.set_xlim(taus[0], taus[-1])
        map(rrerrax.axvline, xlim)
    ax.legend(loc='upper left')

    tau_T = D_T/v0**2
    tau_R = 1/D_R
    if xlim[0] < tau_T < xlim[1]:
        ax.axvline(tau_T, 0, 1/3, ls='--', c='k')
        ax.text(tau_T, 2e-2, ' $D_T/v_0^2$')
    if xlim[0] < tau_R < xlim[1]:
        ax.axvline(tau_R, 0, 2/3, ls='--', c='k')
        ax.text(tau_R, 2e-1, ' $1/D_R$')

    if args.save:
        save = saveprefix + psources + '_rr-corr.pdf'
        print 'saving <rr> correlation plot to',
        print save if verbose else os.path.basename(save)
        fig.savefig(save)

if __name__ == '__main__' and need_plt:
    if args.show:
        plt.show()
    else:
        plt.close('all')

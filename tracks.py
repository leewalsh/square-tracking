#!/usr/bin/env python
# encoding: utf-8

from __future__ import division

if __name__=='__main__':
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('prefix', metavar='PRE',
                   help="Filename prefix with full or relative path (filenames"
                   " prefix_POSITIONS.npz, prefix_CORNER_POSITIONS.npz, etc)")
    p.add_argument('-t','--track', action='store_true',
                   help='Connect the dots and save in the array')
    p.add_argument('-n', '--number', type=int, default=0,
                   help='Total number of tracks to keep. Default = 0 keeps all,'
                        ' -1 attempts to count particles')
    p.add_argument('-o','--orient', action='store_true',
                   help='Find the orientations and save')
    p.add_argument('--ncorners', type=int, default=2,
                   help='Number of corner dots per particle. default = 2')
    p.add_argument('-r', '--rcorner', type=float, default=-1,
                   help='Distance to corner dot from central dot, in pixels.')
    p.add_argument('--drcorner', type=float, default=-1,
                   help='Allowed error in r (rcorner), in pixels. Default is sqrt(r)')
    p.add_argument('-l','--load', action='store_true',
                   help='Create and save structured array from '
                        'prefix[_CORNER]_POSITIONS.txt file')
    p.add_argument('-c', '--corner', action='store_true',
                   help='Load corners instead of centers')
    p.add_argument('-p', '--plottracks', action='store_true',
                   help='Plot the tracks')
    p.add_argument('--noshow', action='store_false', dest='show',
                   help="Don't show figures (just save them)")
    p.add_argument('--nosave', action='store_false', dest='save',
                   help="Don't save outputs or figures")
    p.add_argument('--maxdist', type=int, default=0,
                   help="maximum single-frame travel distance in "
                        "pixels for track. default = S if S>1 else 20")
    p.add_argument('--giveup', type=int, default=10,
                   help="maximum number of frames in track gap. default = 10")
    p.add_argument('-d', '--msd', action='store_true',
                   help='Calculate the MSD')
    p.add_argument('--plotmsd', action='store_true',
                   help='Plot the MSD (requires --msd first)')
    p.add_argument('-s', '--side', type=float, default=1,
                   help='Particle size in pixels, for unit normalization')
    p.add_argument('-f', '--fps', type=float, default=1,
                   help="Number of frames per second (or per shake) "
                        "for unit normalization")
    p.add_argument('--dt0', type=int, default=1,
                   help='Stepsize for time-averaging of a single track '
                        'at different time starting points. default = 1')
    p.add_argument('--dtau', type=int, default=1,
                   help='Stepsize for values of tau at which '
                        'to calculate MSD(tau). default = 1')
    p.add_argument('--killflat', type=int, default=0,
                   help='Minimum growth factor for a single MSD track '
                        'for it to be included')
    p.add_argument('--killjump', type=int, default=100000,
                   help='Maximum initial jump for a single MSD track '
                        'at smallest time step')
    p.add_argument('--stub', type=int, default=10,
                   help='Minimum length (in frames) of a track '
                        'for it to be included. default = 10')
    p.add_argument('--singletracks', type=int, nargs='*',
                   help='identify single track ids to plot')
    p.add_argument('--showtracks', action='store_true',
                   help='Show individual tracks')
    p.add_argument('--cut', action='store_true',
                   help='cut individual tracks at collision with boundary')
    p.add_argument('--boundary', type=float, nargs=3, default=False,
                   metavar=('X0','Y0','R'), help='Optionally provide boundary')
    p.add_argument('--nn', action='store_true',
                   help='Calculate and plot the <nn> correlation')
    p.add_argument('--rn', action='store_true',
                   help='Calculate and plot the <rn> correlation')
    p.add_argument('--rr', action='store_true',
                   help='Calculate and plot the <rr> correlation')
    p.add_argument('--fitv0', action='store_true',
                   help='Allow v_0 to be a free parameter in fit to MSD (<rr>)')
    p.add_argument('-v', '--verbose', action='count',
                        help='Print verbosity')

    args = p.parse_args()

    import os.path
    locdir, prefix = os.path.split(os.path.abspath(args.prefix))
    locdir += '/'
    if args.orient and args.rcorner <= 0:
        raise ValueError, "argument -r/--rcorner is required"

    S = args.side
    A = S**2
    if args.maxdist == 0:
        args.maxdist = S if S>1 else 20
    fps = args.fps
    dtau = args.dtau
    dt0 = args.dt0

    if args.number == -1:
        args.number = True

    verbose = args.verbose
    if verbose:
        print 'using prefix', prefix
    else:
        from warnings import filterwarnings
        #filterwarnings('ignore', category=RuntimeWarning, module='numpy')
        #filterwarnings('ignore', category=RuntimeWarning, module='scipy')
        #filterwarnings('ignore', category=RuntimeWarning, module='matpl')
        filterwarnings('ignore', category=RuntimeWarning)
else:
    verbose = False

from socket import gethostname
hostname = gethostname()
if 'foppl' in hostname:
    import matplotlib
    matplotlib.use("agg")

import sys
from itertools import izip
from collections import defaultdict

import numpy as np
from matplotlib import cm, pyplot as plt
from scipy.optimize import curve_fit

import helpy
import correlation as corr

sf = helpy.SciFormatter().format

pi = np.pi
twopi = 2*pi

def find_closest(thisdot, trackids, n=1, maxdist=20., giveup=10, cut=False):
    """ recursive function to find nearest dot in previous frame.
        looks further back until it finds the nearest particle
        returns the trackid for that nearest dot, else returns new trackid"""
    frame = thisdot[0]
    if cut is not False and cut[thisdot[-1]]:
        return -1
    if frame < n:  # at (or recursed back to) the first frame
        newtrackid = trackids.max() + 1
        if verbose:
            print "New track:", newtrackid
            print '\tframe:', frame,'n:', n,'dot:', thisdot[-1]
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
                print "found a closer child dot to the this dot's parent"
                print "New track:", newtrackid
                print '\tframe:', frame,'n:', n,
                print 'dot:', thisdot[-1],
                print 'closer:', pfsets[frame].item(mini2)[-1]
            return newtrackid
        if cut is not False and cut[closest[-1]]:
            newtrackid = trackids.max() + 1
            if verbose:
                print "cutting track:", trackids[closest[-1]]
                print "New track:", newtrackid
            return newtrackid
        else:
            oldtrackid = trackids[closest[-1]]
            if oldtrackid == -1:
                newtrackid = trackids.max() + 1
                if verbose:
                    print "new track since previous was cut", newtrackid
                return newtrackid
            else:
                return oldtrackid
    elif n < giveup:
        return find_closest(thisdot, trackids, n=n+1,
                            maxdist=maxdist, giveup=giveup, cut=cut)
    else: # give up after giveup frames
        newtrackid = trackids.max() + 1
        if verbose:
            print "Recursed {} times, giving up.".format(n)
            print "New track:", newtrackid
            print '\tframe:', frame, 'n:', n, 'dot:', thisdot[-1]
        return newtrackid

def find_tracks(pdata, maxdist=20, giveup=10, n=0, cut=False, stub=0):
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
    if n is True:
        # use the mode of number of particles per frame
        # np.argmax(np.bincount(x)) == mode(x)
        n = np.argmax(np.bincount(np.bincount(pdata['f'])))
        print "Found {n} particles, will use {n} longest tracks".format(n=n)

    if cut:
        if args.boundary:
            x0, y0, R = args.boundary
        else:
            not_found = ('Please give the path to a tiff image '
                         'from this dataset to identify boundary\n')
            bgimage = helpy.find_first_frame([locdir, prefix], err=not_found)
            x0, y0, R = helpy.circle_click(bgimage)
            print "Boundary (x0, y0, r):", x0, y0, R
        # assume 6mm particles if S not specified
        mm = R/101.6 # R = 4 in = 101.6 mm
        margin = S if S>1 else 6*mm
        print 'Cutting with margin {:.1f} pix = {:.1f} mm'.format(margin, margin/mm)
        rs = np.hypot(pdata['x'] - x0, pdata['y'] - y0)
        cut = rs > R - margin

    print "seeking tracks"
    for i in xrange(len(pdata)):
        # This must remain a simple loop because trackids gets modified and
        # passed into the function with each iteration
        trackids[i] = find_closest(pdata.item(i), trackids,
                                   maxdist=maxdist, giveup=giveup, cut=cut)

    if verbose:
        assert len(pdata) == len(trackids), "too few/many trackids"
        assert np.allclose(pdata['id'], np.arange(len(pdata))), "gap in particle id"

    if n or stub > 0:
        track_lens = np.bincount(trackids+1)[1:]
    if n:
        stubs = np.argsort(track_lens)[:-n] # all but the longest n
    elif stub > 0:
        stubs = np.where(track_lens < stub)[0]
        if verbose: print "removing {} stubs".format(len(stubs))
    if n or stub > 0:
        stubs = np.in1d(trackids, stubs)
        trackids[stubs] = -1
    return trackids

def remove_duplicates(trackids=None, data=None, tracksets=None,
                      target='', inplace=False):
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
        dup_fs = np.where(count>1)[0]
        if not len(dup_fs):
            continue
        ftsets = helpy.splitter(tset, fs, ret_dict=True)
        for f in dup_fs:
            prv = fs[np.searchsorted(fs, f, 'left') - 1] if f > fs[0] else None
            nxt = fs[np.searchsorted(fs, f, 'right')] if f < fs[-1] else None
            if nxt is not None and nxt in dup_fs:
                nxt = fs[np.searchsorted(fs, nxt, 'right')] if nxt < fs[-1] else None
                if nxt is not None and nxt in dup_fs:
                    nxt = None
                    assert prv is not None, ("Duplicate track particles in too many "
                            "frames in a row at frame {} for track {}".format(f, t))
            seps = np.zeros(count[f])
            for neigh in (prv, nxt):
                if neigh is None: continue
                if count[neigh] > 1 and neigh in rejects[t]:
                    isreject = np.in1d(ftsets[neigh]['id'], rejects[t][neigh], assume_unique=True)
                    ftsets[neigh] = ftsets[neigh][~isreject]
                sepx = ftsets[f]['x'] - ftsets[neigh]['x']
                sepy = ftsets[f]['y'] - ftsets[neigh]['y']
                seps += sepx*sepx + sepy*sepy
            rejects[t][f] = ftsets[f][seps > seps.min()]['id']
    if target=='tracksets':
        if not inplace:
            tracksets = tracksets.copy()
        for t, tr in rejects.iteritems():
            trs = np.concatenate(tr.values())
            tr = np.in1d(tracksets[t]['id'], trs, True, True)
            new = tracksets[t][tr]
            if inplace:
                tracksets[t] = new
        return None if inplace else tracksets
    elif target=='trackids':
        if not inplace:
            trackids = trackids.copy()
        rejects = np.concatenate([tfr for tr in rejects.itervalues()
                                for tfr in tr.itervalues()])
        if data is None:
            data_from_tracksets = np.concatenate(tracksets.values())
            if len(data_from_tracksets)!=len(trackids):
                raise ValueError, "You must provide data to return/modify trackids"
            ids = data_from_tracksets['id']
            ids.sort()
        else:
            ids = data['id']
        rejects = np.searchsorted(ids, rejects)
        trackids[rejects] = -1
        return None if inplace else trackids

def animate_detection(imstack, fsets, fcsets, fosets=None, rc=0, side=10, verbose=False):

    from matplotlib.patches import Circle

    def advance(event):
        key = event.key
        if verbose:
            print 'pressed', key, 'next frame:',
        global f_display
        if key=='left':
            if f_display>=1:
                f_display -= 1
        elif key=='right':
            f_display += 1
        else:
            plt.close()
            f_display = -1
        if verbose:
            print f_display
            sys.stdout.flush()

    plt_text = np.vectorize(plt.text)

    def draw_circles(ax, centers, r, *args, **kwargs):
        patches = [Circle(cent, r, *args, **kwargs) for cent in centers]
        map(ax.add_patch, patches)
        ax.figure.canvas.draw()
        return patches

    fig = plt.figure(figsize=(12, 12))
    p = plt.imshow(imstack[0], cmap='gray')
    ax = p.axes
    global f_display
    f_display = repeat = f_old = 0
    lengths = map(len, [imstack, fsets, fcsets])
    f_max = min(lengths)
    assert f_max, 'Lengths imstack: {}, fsets: {}, fcsets: {}'.format(*lengths)
    while 0 <= f_display < f_max:
        if repeat > 5:
            if verbose:
                print 'stuck on frame', f_display
            break
        f_display %= f_max
        if f_display==f_old:
            repeat += 1
        else:
            repeat = 0
        f_old = f_display
        if verbose:
            print 'starting loop with f_display =', f_display
        xyo = helpy.consecutive_fields_view(fsets[f_display], 'xyo', False)
        xyc = helpy.consecutive_fields_view(fcsets[f_display], 'xy', False)
        x, y, o = xyo.T
        omask = np.isfinite(o)
        xo, yo, oo = xyo[omask].T

        p.set_data(imstack[f_display])
        remove = []
        if rc:
            patches = draw_circles(ax, xyo[:, 1::-1], rc,
                                color='g', fill=False, zorder=.5)
            remove.extend(patches)
        q = plt.quiver(yo, xo, np.sin(oo), np.cos(oo), angles='xy',
                       units='xy', width=side/8, scale_units='xy', scale=1/side)
        ps = plt.scatter(y, x, c='r')#c=np.where(omask, 'r', 'b'))
        cs = plt.scatter(xyc[:,1], xyc[:,0], c='g', s=8)
        if fosets is not None:
            oc = helpy.quick_field_view(fosets[f_display], 'corner').reshape(-1, 2)
            ocs = plt.scatter(oc[:,1], oc[:,0], c='orange', s=8)
            remove.append(ocs)
        remove.extend([q, ps, cs])

        tstr = fsets[f_display]['t'].astype('S')
        txt = plt_text(y+rc, x+rc, tstr, color='r')
        remove.extend(txt)

        plt.xlim(0, imstack.shape[2])
        plt.ylim(0, imstack.shape[1])
        plt.title("frame {}\n{} orientations / {} particles detected".format(
                    f_display, np.count_nonzero(omask), len(o)))
        fig.canvas.draw()

        plt.waitforbuttonpress()
        fig.canvas.mpl_connect('key_press_event', advance)
        for rem in remove:
            rem.remove()
        if verbose:
            print 'ending frame', f_display
            sys.stdout.flush()
    if verbose:
        print 'loop broken'

def interp_nans(f, x=None, max_gap=5, inplace=False):
    """ Replace nans in function f(x) with their linear interpolation"""
    n = len(f)
    if n < 3:
        return f
    nans = np.isnan(f)
    if f.ndim > 1:
        nans = nans.all(1)
    if np.count_nonzero(nans) in (0, n):
        return f
    if not inplace:
        f = f.copy()
    ifin = (~nans).nonzero()[0]
    if len(ifin) < 2:
        return f
    inan = nans.nonzero()[0]
    gaps = np.diff(ifin) - 1
    mx = gaps.max()
    if mx > max_gap:
        spl = gaps.argmax()
        args = (max_gap, True)
        interp_nans(f[:spl], x if x is None else x[:spl], *args)
        interp_nans(f[spl+mx+1:], x if x is None else x[spl+mx+1:], *args)
        return f
    xnan, xfin = (inan, ifin) if x is None else (x[inan], x[ifin])
    for c in f.T if f.ndim>1 else [f]:
        c[inan] = np.interp(xnan, xfin, c[ifin])
    return f

def fill_gaps(tracksets, max_gap=5, interp=['xy','o'], inplace=True, verbose=False):
    if not inplace:
        tracksets = {t: s.copy() for t,s in tracksets.iteritems()}
    for t, tset in tracksets.items():
        fs = tset['f']
        gaps = np.diff(fs) - 1
        mx = gaps.max()
        if not mx:
            if 'o' in interp:
                interp_nans(tset['o'], tset['f'], inplace=True)
            continue
        elif mx > max_gap:
            print "Dropped track {}: has gap of {} > {}".format(t, mx, max_gap)
            tracksets.pop(t)
            continue
        gapi = gaps.nonzero()[0]
        gaps = gaps[gapi]
        gapi = np.repeat(gapi, gaps)
        missing = np.full(len(gapi), np.nan, tset.dtype)
        if verbose:
            print ("track {:4d}: missing {} frames in {} gaps"
                   " (biggest {})").format(t, len(gapi), len(gaps), mx)
        missing['f'] = np.concatenate(map(range, gaps)) + fs[gapi] + 1
        missing['t'] = t
        tset = np.insert(tset, gapi+1, missing)
        if interp:
            for field in interp:
                view = helpy.consecutive_fields_view(tset, field, careful=False)
                interp_nans(view, inplace=True)
        tracksets[t] = tset
    return tracksets

# Plotting tracks:
def plot_tracks(data, trackids, bgimage=None, mask=None,
                fignum=None, save=True, show=True):
    """ Plots the tracks of particles in 2D

    parameters
    ----------
    data : the main data array of points
    trackids : the array of track ids
    bgimage : a background image to plot on top of (the first frame tif, e.g.)
    mask : a boolean mask to filter the data (to show certain frames or tracks)
    fignum : a pyplot figure number to add the plot to
    save : whether to save the figure
    show : whether to show the figure
    """
    plt.figure(fignum)
    if bgimage:
        if isinstance(bgimage, basestring):
            bgimage = plt.imread(bgimage)
        plt.imshow(bgimage, cmap=cm.gray, origin='upper')
    if mask is None:
        mask = (trackids >= 0)
    else:
        mask = mask & (trackids >= 0)
    data = data[mask]
    trackids = trackids[mask]
    plt.scatter(data['y'], data['x'],
            c=np.array(trackids)%12, marker='o', alpha=.5, lw=0)
    plt.gca().set_aspect('equal')
    plt.xlim(data['y'].min()-10, data['y'].max()+10)
    plt.ylim(data['x'].min()-10, data['x'].max()+10)
    plt.title(prefix)
    if save:
        print "saving tracks image to", prefix+"_tracks.png"
        plt.savefig(locdir+prefix+"_tracks.png")
    if show: plt.show()

# Mean Squared Displacement
# dx^2 (tau) = < ( x_i(t0 + tau) - x_i(t0) )^2 >
#              <  averaged over t0, then i   >

def t0avg(trackdots, tracklen, tau):
    """ Averages the squared displacement of a track for a certain value of tau
        over all valid values of t0 (such that t0 + tau < tracklen)

        That is, for a given particle, do the average over all t0
            <[(r_i(t0 + tau) - r_i(t0)]^2>
        for a single particle i and fixed time shift tau

        parameters
        ----------
        trackdots : a subset of data for all points in the given track
        tracklen : the length (duration) of the track
        tau : the time separation for the displacement: r(tau) - r(0)

        returns
        -------
        the described mean, a scalar
    """
    totsqdisp = 0.0
    nt0s = 0.0
    for t0 in np.arange(1,(tracklen-tau-1),dt0): # for t0 in (T - tau - 1), by dt0 stepsize
        #print "t0=%d, tau=%d, t0+tau=%d, tracklen=%d"%(t0,tau,t0+tau,tracklen)
        olddot = trackdots[trackdots['f']==t0]
        newdot = trackdots[trackdots['f']==t0+tau]
        if len(newdot) != 1 or len(olddot) != 1:
            continue
        sqdisp  = (newdot['x'] - olddot['x'])**2 \
                + (newdot['y'] - olddot['y'])**2
        if len(sqdisp) == 1:
            if verbose > 1: print 'unflattened'
            totsqdisp += sqdisp
        elif len(sqdisp[0]) == 1:
            if verbose: print 'flattened once'
            totsqdisp += sqdisp[0]
        else:
            if verbose: print "fail"
            continue
        nt0s += 1.0
    return totsqdisp/nt0s if nt0s else None

def trackmsd(track, dt0, dtau):
    """ finds the mean squared displacement as a function of tau,
        averaged over t0, for one track (particle)

        parameters
        ----------
        track : a single integer giving the track id to be calculated
            or, a subset of the data for a given track (a 'trackset')
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
    if np.isscalar(track):
        trackdots = data[trackids==track]
    else:
        trackdots = track

    if dt0 == dtau == 1:
        if verbose: print "Using correlation"
        xy = np.column_stack([trackdots['x'], trackdots['y']])
        return corr.msd(xy, ret_taus=True)

    trackbegin, trackend = trackdots['f'][[0,-1]]
    tracklen = trackend - trackbegin + 1
    if verbose:
        print "tracklen =",tracklen
        print "\t from %d to %d"%(trackbegin, trackend)
    if isinstance(dtau, float):
        taus = helpy.farange(dt0, tracklen, dtau)
    elif isinstance(dtau, int):
        taus = xrange(dtau, tracklen, dtau)

    tmsd = []
    for tau in taus:  # for tau in T, by factor dtau
        #print "tau =", tau
        avg = t0avg(trackdots, tracklen, tau)
        #print "avg =", avg
        if avg > 0 and not np.isnan(avg):
            tmsd.append([tau,avg[0]])
    if verbose:
        print "\t...actually", len(tmsd)
    return tmsd

def find_msds(dt0, dtau, tracks=None, min_length=0):
    """ Calculates the MSDs for all tracks

        parameters
        ----------
        dt0, dtau : see documentation for `trackmsd`
        tracks : iterable of individual track numbers, or None for all tracks
        min_length : a cutoff to exclude tracks shorter than min_length

        returns
        -------
        msds : a list of all trackmsds (each in the format given by `trackmsd`)
        msdids : a list of the trackids corresponding to each msd
    """
    print "Calculating MSDs with",
    print "dt0 = {}, dtau = {}".format(dt0, dtau)
    msds = []
    msdids = []
    if tracks is None:
        tracks = helpy.load_tracksets(data, min_length=min_length)
    for trackid in sorted(tracks):
        if verbose: print "calculating msd for track", trackid
        tmsd = trackmsd(tracks[trackid], dt0, dtau)
        if len(tmsd) > 1:
            msds.append(tmsd)
            msdids.append(trackid)
    return msds, msdids

# Mean Squared Displacement:

def mean_msd(msds, taus, msdids=None, kill_flats=0, kill_jumps=1e9,
             show_tracks=False, singletracks=None, tnormalize=False,
             errorbars=False, fps=1, A=1):
    """ return the mean of several track msds """

    # msd has shape (number of tracks, length of tracks)
    msdshape = (len(singletracks) if singletracks else len(msds),
                max(map(len, msds)))
    msd = np.full(msdshape, np.nan, float)
    taus = taus[:msdshape[1]]

    if msdids is not None:
        allmsds = izip(xrange(len(msds)), msds, msdids)
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
        if len(tmsd) < 2: continue
        tmsdt, tmsdd = np.asarray(tmsd).T
        if tmsdd[-50:].mean() < kill_flats: continue
        if tmsdd[:2].mean() > kill_jumps: continue
        tau_match = np.searchsorted(taus, tmsdt)
        msd[ti, tau_match] = tmsdd
    if errorbars:
        added = np.sum(np.isfinite(msd), 0)
        msd_err = np.nanstd(msd, 0) + 1e-9
        msd_err /= np.nan_to_num(np.sqrt(added-1))
    if show_tracks:
        plt.plot(taus/fps, (msd/(taus/fps)**tnormalize).T/A, 'b', alpha=.2)
    msd = np.nanmean(msd, 0)
    return (msd, msd_err) if errorbars else msd

def plot_msd(msds, msdids, dtau, dt0, nframes, tnormalize=False, prefix='',
        show_tracks=True, figsize=(8,6), plfunc=plt.semilogx, meancol='',
        title=None, xlim=None, ylim=None, fignum=None, errorbars=False,
        lw=1, singletracks=None, fps=1, S=1, ang=False, sys_size=0,
        kill_flats=0, kill_jumps=1e9, show_legend=False, save='', show=True):
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
    fig = plt.figure(fignum, figsize)

    # Get the mean of msds
    msd = mean_msd(msds, taus, msdids,
            kill_flats=kill_flats, kill_jumps=kill_jumps, show_tracks=show_tracks,
            singletracks=singletracks, tnormalize=tnormalize, errorbars=errorbars,
            fps=fps, A=A)
    if errorbars: msd, msd_err = msd

    #print "Coefficient of diffusion ~", msd[np.searchsorted(taus, fps)]/A
    #print "Diffusion timescale ~", taus[np.searchsorted(msd, A)]/fps

    taus = taus[:len(msd)]
    taus /= fps
    msd /= A
    if errorbars: msd_err /= A

    if tnormalize:
        plfunc(taus, msd/taus**tnormalize, meancol,
               label="Mean Sq {}Disp/Time{}".format(
                     "Angular " if ang else "",
                     "^{}".format(tnormalize) if tnormalize != 1 else ''))
        plfunc(taus, msd[0]*taus**(1-tnormalize)/dtau,
               'k-', label="ref slope = 1", lw=2)
        plfunc(taus, (twopi**2 if ang else 1)/(taus)**tnormalize,
               'k--', lw=2, label=r"$(2\pi)^2$" if ang else
               ("One particle area" if S>1 else "One Pixel"))
        plt.ylim([0, 1.3*np.max(msd/taus**tnormalize)])
    else:
        plt.loglog(taus, msd, meancol, lw=lw,
                  label="Mean Squared {}Displacement".format('Angular '*ang))
        #plt.loglog(taus, msd[0]*taus/dtau/2, meancol+'--', lw=2,
        #          label="slope = 1")
    if errorbars:
        plt.errorbar(taus, msd/taus**tnormalize,
                    msd_err/taus**tnormalize,
                    fmt=meancol, capthick=0, elinewidth=1, errorevery=errorbars)
    if sys_size:
        plt.axhline(sys_size, ls='--', lw=.5, c='k', label='System Size')
    plt.title("Mean Sq {}Disp".format("Angular " if ang else "") if title is None else title)
    plt.xlabel('Time (' + ('s)' if fps > 1 else 'frames)'), fontsize='x-large')
    if ang:
        plt.ylabel('Squared Angular Displacement ($rad^2$)',
              fontsize='x-large')
    else:
        plt.ylabel('Squared Displacement ('+('particle area)' if S>1 else 'square pixels)'),
              fontsize='x-large')
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    if show_legend: plt.legend(loc='best')
    if save is True:
        save = locdir + prefix + "_MS{}D.pdf".format('A' if ang else '')
    if save:
        print "saving to", save
        plt.savefig(save)
    if show: plt.show()
    return [fig] + fig.get_axes() + [taus] + [msd, msd_err] if errorbars else [msd]

if __name__=='__main__':
    if args.load:
        datapath = locdir+prefix+'_CORNER'*args.corner+'_POSITIONS.txt'
        helpy.txt_to_npz(datapath, verbose=True, compress=True)
        if args.orient or args.track:
            print 'not tracking, only converting file from txt to npz'
        sys.exit()

    if args.track or args.orient:
        from scipy.spatial import cKDTree as KDTree
        if args.track != args.orient and helpy.bool_input("Would you like to "
                "simultaneously track and find orientations? (It's faster)\n"):
            args.track = args.orient = True
        if args.orient:
            pdata, cdata = helpy.load_data(locdir+prefix, 'position corner')
        else:
            pdata = helpy.load_data(locdir+prefix, 'position')
        pfsets = helpy.splitter(pdata, ret_dict=True)
        pftrees = { f: KDTree(np.column_stack([pfset['x'], pfset['y']]), leafsize=50)
                   for f, pfset in pfsets.iteritems() }
    if args.track:
        trackids = find_tracks(pdata, maxdist=args.maxdist, giveup=args.giveup,
                               n=args.number, cut=args.cut, stub=args.stub)
        trackids = remove_duplicates(trackids, data=pdata)
    else:
        trackids = None
    if args.orient:
        from orientation import get_angles_loop
        cfsets = helpy.splitter(cdata, ret_dict=True)
        cftrees = { f: KDTree(np.column_stack([cfset['x'], cfset['y']]), leafsize=50)
                   for f, cfset in cfsets.iteritems() }
        odata, omask = get_angles_loop(pdata, cdata, pfsets, cfsets, cftrees,
                           nc=args.ncorners, rc=args.rcorner, drc=args.drcorner)
        if args.save:
            save = locdir+prefix+'_ORIENTATION.npz'
            print "saving orientation data to", save
            np.savez_compressed(save, odata=odata, omask=omask)
        orients = odata['orient']
    else:
        orients = None
    if args.track or args.orient:
        data = helpy.initialize_tdata(pdata, trackids, orients)
        if args.save:
            save = locdir+prefix+"_TRACKS.npz"
            print "saving track data to", save
            np.savez_compressed(save, data=data)
    else:
        data = helpy.load_data(locdir+prefix, 'track')

    if args.msd:
        msds, msdids = find_msds(dt0, dtau, min_length=args.stub)
        if args.save:
            save = locdir+prefix+"_MSD.npz"
            print "saving msd data to", save
            np.savez(save,
                     msds = np.asarray(msds),
                     msdids = np.asarray(msdids),
                     dt0  = np.asarray(dt0),
                     dtau = np.asarray(dtau))
    elif args.plotmsd or args.rr:
        if verbose: print "loading msd data from npz files"
        datapath = locdir+prefix+"_MSD.npz"
        msdnpz = np.load(datapath)
        msds = msdnpz['msds']
        try: msdids = msdnpz['msdids']
        except KeyError: msdids = None
        try:
            dt0  = np.asscalar(msdnpz['dt0'])
            dtau = np.asscalar(msdnpz['dtau'])
        except KeyError:
            dt0  = 10 # here's assuming...
            dtau = 10 #  should be true for all from before dt* was saved
        if verbose: print "\t...loaded"

if __name__=='__main__':
    if args.plotmsd:
        if verbose: print 'plotting now!'
        plot_msd(msds, msdids, dtau, dt0, data['f'].max()+1, tnormalize=False,
                 prefix=prefix, show_tracks=args.showtracks, show=args.show,
                 singletracks=args.singletracks, fps=fps, S=S, save=args.save,
                 kill_flats=args.killflat, kill_jumps=args.killjump*S*S)
    if args.plottracks:
        bgimage = helpy.find_first_frame([locdir, prefix])
        if args.singletracks:
            mask = np.in1d(trackids, args.singletracks)
        else:
            mask = None
        plot_tracks(data, trackids, bgimage, mask=mask,
                    save=args.save, show=args.show)

if __name__=='__main__' and args.nn:
    # Calculate the <nn> correlation for all the tracks in a given dataset
    # TODO: fix this to combine multiple datasets (more than one prefix)

    data = helpy.load_data(locdir+prefix, 'tracks')
    omask = np.isfinite(data['o'])
    tracksets = helpy.load_tracksets(data, omask, min_length=args.stub)

    coscorrs = [ corr.autocorr(np.cos(trackset['o']), cumulant=False, norm=False)
                for trackset in tracksets.values() ]
    sincorrs = [ corr.autocorr(np.sin(trackset['o']), cumulant=False, norm=False)
                for trackset in tracksets.values() ]

    # Gather all the track correlations and average them
    allcorr = coscorrs + sincorrs
    allcorr = helpy.pad_uneven(allcorr, np.nan)
    tcorr = np.arange(allcorr.shape[1])/fps
    meancorr = np.nanmean(allcorr, 0)
    added = np.sum(np.isfinite(allcorr), 0)
    errcorr = np.nanstd(allcorr, 0)/np.sqrt(added - 1)
    sigma = errcorr + 1e-5*errcorr.std() # add something small to prevent 0
    if verbose:
        print "Merged nn corrs"

    # Fit to exponential decay
    tmax = 50
    fmax = np.searchsorted(tcorr, tmax)
    fitform = lambda s, DR: 0.5*np.exp(-DR*s)
    fitstr = r"$\frac{1}{2}e^{-D_R t}$"
    p0 = [1]
    try:
        popt, pcov = curve_fit(fitform, tcorr[:fmax], meancorr[:fmax],
                               p0=p0, sigma=sigma[:fmax])
    except RuntimeError as e:
        print "RuntimeError:", e.message
        print "Using inital guess", p0
        popt = p0
    D_R = popt[0]
    print "Fits to <nn>:"
    print '   D_R: {:.4g}'.format(D_R)

    plt.figure()
    plot_individual = True
    if plot_individual:
        plt.plot(tcorr, allcorr.T, 'b', alpha=.2)
    plt.errorbar(tcorr, meancorr, errcorr, None, 'ok',
                label="Mean Orientation Autocorrelation",
                capthick=0, elinewidth=1, errorevery=3)
    plt.plot(tcorr, fitform(tcorr, *popt), 'r',
            label=fitstr + '\n' + sf("$D_R={0:.4T}$, $D_R^{{-1}}={1:.3T}$",
                                      D_R, 1/D_R))
    plt.xlim(0, tmax)
    plt.ylim(fitform(tmax, *popt), 1)
    plt.yscale('log')

    plt.ylabel(r"$\langle \hat n(t) \hat n(0) \rangle$")
    plt.xlabel("$tf$")
    plt.title("Orientation Autocorrelation\n"+prefix)
    plt.legend(loc='upper right', framealpha=1)

    if args.save:
        save = locdir+prefix+'_nn-corr.pdf'
        print 'saving <nn> correlation plot to', save
        plt.savefig(save)
    if not (args.rn or args.rr) and args.show: plt.show()

if __name__=='__main__' and args.rn:
    # Calculate the <rn> correlation for all the tracks in a given dataset
    # TODO: fix this to combine multiple datasets (more than one prefix)

    if not args.nn:
        # if args.nn, then these have been loaded already
        data = helpy.load_data(locdir+prefix, 'tracks')
        omask = np.isfinite(data['o'])
        tracksets = helpy.load_tracksets(data, omask,
                                    min_length=max(100, args.stub))
        D_R = 1/12

    corr_args = {'side': 'both', 'ret_dx': True,
                 'cumulant': (True, False), 'norm': 0 }

    xcoscorrs = [ corr.crosscorr(trackset['x']/S, np.cos(trackset['o']),
                 **corr_args) for trackset in tracksets.values() ]
    ysincorrs = [ corr.crosscorr(trackset['y']/S, np.sin(trackset['o']),
                 **corr_args) for trackset in tracksets.values() ]

    # Align and merge them
    fmax = int(2*fps/D_R)
    fmin = -fmax
    rncorrs = xcoscorrs + ysincorrs
    # TODO: align these so that even if a track doesn't reach the fmin edge,
    # that is, if f.min() > fmin for a track, then it still aligns at zero
    rncorrs = helpy.pad_uneven([
                    rn[np.searchsorted(f, fmin):np.searchsorted(f, fmax)]
                              for f, rn in rncorrs if f.min() <= fmin ],
                    np.nan)
    tcorr = np.arange(fmin, fmax)/fps
    meancorr = np.nanmean(rncorrs, 0)
    added = np.sum(np.isfinite(rncorrs), 0)
    errcorr = np.nanstd(rncorrs, 0)/np.sqrt(added - 1)
    sigma = errcorr + errcorr.std() # add something small to prevent 0
    if verbose:
        print "Merged rn corrs"

    # Fit to capped exponential growth
    fitform = lambda s, v_D, D=D_R:\
                  0.5*np.sign(s)*v_D*(1 - corr.exp_decay(np.abs(s), 1/D))
    fitstr = r'$\frac{v_0}{2D_R}(1 - e^{-D_R|s|})\operatorname{sign}(s)$'
    # p0 = [v_0/D_R, D_R]
    p0 = [1]
    if not args.nn:
        p0 += [D_R]

    print "============="
    print "Fits to <rn>:"
    try:
        popt, pcov = curve_fit(fitform, tcorr, meancorr, p0=p0, sigma=sigma)
    except RuntimeError as e:
        try:
            p0[0] = -1 # If dots are backwards, has trouble fitting with v0>0
            popt, pcov = curve_fit(fitform, tcorr, meancorr, p0=p0, sigma=sigma)
        except RuntimeError as e:
            p[0] = 1 # restore original positive value
            print "RuntimeError:", e.message
            print "Using inital guess", p0
            popt = p0
    if len(popt) > 1:
        D_R = popt[1]
    v0 = D_R*popt[0]
    print '\n'.join(['v0/D_R: {:.4g}',
                     '   D_R: {:.4g}'][:len(popt)]).format(*popt)
    print "Giving:"
    print '\n'.join(['    v0: {:.4f}',
                     '   D_R: {:.4f}'][:3-len(popt)]
                    ).format(*[v0, D_R][:3-len(popt)])

    plt.figure()
    fit = fitform(tcorr, *popt)
    plot_individual = True
    sgn = np.sign(v0)
    if plot_individual:
        plt.plot(tcorr, sgn*rncorrs.T, 'b', alpha=.2)
    plt.errorbar(tcorr, sgn*meancorr, errcorr, None, 'ok',
                label="Mean Position-Orientation Correlation",
                capthick=0, elinewidth=1, errorevery=3)
    plt.plot(tcorr, sgn*fit, 'r', lw=2,
            #label=fitstr+'\n'+
            #       ', '.join(['$v_0$: {:.3f}', '$t_0$: {:.3f}', '$D_R$: {:.3f}'
            label=fitstr + '\n' + sf(', '.join(
                  ['$v_0={0:.3T}$', '$D_R={1:.3T}$'][:len(popt)]
                  ), *(abs(v0), D_R)[:len(popt)]))

    plt.axvline(1/D_R, 0, 2/3, ls='--', c='k')
    plt.text(1/D_R, 1e-2, ' $1/D_R$')

    plt.ylim(1.5*fit.min(), 1.5*fit.max())
    plt.xlim(tcorr.min(), tcorr.max())
    plt.title("Position - Orientation Correlation")
    plt.ylabel(r"$\langle \vec r(t) \hat n(0) \rangle / \ell$")
    plt.xlabel("$tf$")
    plt.legend(loc='upper left', framealpha=1)

    if args.save:
        save = locdir + prefix + '_rn-corr.pdf'
        print 'saving <rn> correlation plot to', save
        plt.savefig(save)
    if not args.rr and args.show: plt.show()

if __name__=='__main__' and args.rr:
    fig, ax, taus, msd, msderr = plot_msd(
            msds, msdids, dtau, dt0, data['f'].max()+1, tnormalize=False,
            errorbars=5, prefix=prefix, show_tracks=True, meancol='ok',
            singletracks=args.singletracks, fps=fps, S=S, show=False,
            kill_flats=args.killflat, kill_jumps=args.killjump*S*S)

    sigma = msderr + 1e-5*S*S
    tmax = 200
    fmax = np.searchsorted(taus, tmax)

    if not args.nn:
        D_R = 1
    if not args.rn:
        v0 = sgn = 1

    p0 = [0]
    if not (args.nn or args.rn):
        p0 += [v0, D_R]
    elif args.fitv0 or not args.rn:
        p0 += [v0]
    fitform = lambda s, D, v=v0, DR=D_R:\
              2*(v/DR)**2 * (DR*s + np.exp(-DR*s) - 1) + 2*D*s
    fitstr = r"$2(v_0/D_R)^2 (D_Rt + e^{{-D_Rt}} - 1) + 2D_Tt$"
    try:
        popt, pcov = curve_fit(fitform, taus[:fmax], msd[:fmax],
                               p0=p0, sigma=sigma[:fmax])
    except RuntimeError as e:
        print "RuntimeError:", e.message
        if not args.fitv0: p0 = [0, v0]
        print "Using inital guess", p0
        popt = p0

    print "============="
    print "Fits to <rr>:"

    D_T = popt[0]
    if len(popt) > 1:
        v0 = popt[1]
        if len(popt) > 2:
            D_R = popt[2]

    print '\n'.join(['   D_T: {:.3g}',
                     'v0(rr): {:.3g}',
                     '   D_R: {:.3g}'][:len(popt)]).format(*popt)
    if len(popt) > 1:
        print "Giving:"
        print "v0/D_R: {:.3g}".format(v0/D_R)
    fit = fitform(taus, *popt)
    ax.plot(taus, fit, 'r', lw=2,
            label=fitstr + "\n" + sf(', '.join(
                ["$D_T={0:.3T}$", "$v_0={1:.3T}$", "$D_R={2:.3T}$"][:len(popt)]
                ), *(popt*[1, sgn, 1][:len(popt)])))

    plt.axvline(D_T/v0**2, 0, 1/3, ls='--', c='k')
    plt.text(D_T/v0**2, 2e-2, ' $D_T/v_0^2$')
    plt.axvline(1/D_R, 0, 2/3, ls='--', c='k')
    plt.text(1/D_R, 2e-1, ' $1/D_R$')

    plt.ylim(min(fit[0], msd[0]), fit[np.searchsorted(taus, tmax)])
    plt.xlim(taus[0], tmax)
    plt.legend(loc='upper left')

    if args.save:
        save = locdir + prefix + '_rr-corr.pdf'
        print 'saving <rr> correlation plot to', save
        fig.savefig(save)
    if args.show: plt.show()

if __name__=='__main__' and not args.show:
    plt.close('all')

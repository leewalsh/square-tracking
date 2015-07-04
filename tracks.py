#!/usr/bin/env python
# encoding: utf-8

from __future__ import division

from socket import gethostname
hostname = gethostname()
if 'foppl' in hostname:
    import matplotlib
    matplotlib.use("agg")

from itertools import izip
from math import sqrt

import numpy as np
from matplotlib import cm, pyplot as pl
from scipy.optimize import curve_fit

import helpy
import correlation as corr

pi = np.pi
twopi = 2*pi
locdir = extdir = ''

if __name__=='__main__':
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('prefix', metavar='PRE',
                   help="Filename prefix with full or relative path (filenames"
                   " prefix_POSITIONS.txt, prefix_CORNER_POSITIONS.txt, etc)")
    p.add_argument('-c', '--corner', action='store_true',
                   help='Track corners instead of centers')
    p.add_argument('-n', '--number', type=int, default=0,
                   help='Total number of tracks to keep. Default = 0 keeps all,'
                        ' -1 attempts to count particles')
    p.add_argument('-l','--load', action='store_true',
                   help='Create and save structured array from '
                        'prefix[_CORNER]_POSITIONS.txt file')
    p.add_argument('-t','--track', action='store_true',
                   help='Connect the dots and save in the array')
    p.add_argument('-p', '--plottracks', action='store_true',
                   help='Plot the tracks')
    p.add_argument('--noshow', action='store_false', dest='show',
                   help="Don't show figures (just save them)")
    p.add_argument('--nosave', action='store_false', dest='save',
                   help="Don't save figures (just show them)")
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
    p.add_argument('--center', type=float, nargs=3, default=False,
                   metavar=('X0', 'Y0', 'R'),
                   help='Optionally provide center and radius')
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

    prefix = args.prefix
    dotfix = '_CORNER' if args.corner else ''

    S = args.side
    A = S**2
    if args.maxdist == 0:
        args.maxdist = S if S>1 else 20
    fps = args.fps
    dtau = args.dtau
    dt0 = args.dt0

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

def gen_data(datapath):
    """ Reads raw positions data into a numpy array and saves it as an npz file

        `datapath` is the path to the output file from finding particles
        it must end with "results.txt" or "POSITIONS.txt", depending on its
        source, and its structure is assumed to match a certain pattern
    """
    print "loading positions data from", datapath
    if  datapath.endswith('results.txt'):
        shapeinfo = False
        # imagej output (called *_results.txt)
        dtargs = {  'usecols' : [0,2,3,5],
                    'names'   : "id,x,y,f",
                    'dtype'   : [int,float,float,int]} \
            if not shapeinfo else \
                 {  'usecols' : [0,1,2,3,4,5,6],
                    'names'   : "id,area,mean,x,y,circ,f",
                    'dtype'   : [int,float,float,float,float,float,int]}
        data = np.genfromtxt(datapath, skip_header = 1,**dtargs)
        data['id'] -= 1 # data from imagej is 1-indexed
    elif datapath.endswith('POSITIONS.txt'):
        # positions.py output (called *_POSITIONS.txt)
        from numpy.lib.recfunctions import append_fields
        data = np.genfromtxt(datapath,
                             skip_header = 1,
                             names = "f,x,y,lab,ecc,area",
                             dtype = [int,float,float,int,float,int])
        ids = np.arange(len(data))
        data = append_fields(data, 'id', ids, usemask=False)
    else:
        print "is {} from imagej or positions.py?".format(datapath.split('/')[-1])
        print "Please rename it to end with _results.txt or _POSITIONS.txt"
    return data

def find_closest(thisdot, trackids, n=1, maxdist=20., giveup=10,
                 cut=False):
    """ recursive function to find nearest dot in previous frame.
        looks further back until it finds the nearest particle
        returns the trackid for that nearest dot, else returns new trackid"""
    frame = thisdot['f']
    if cut is False: cut = np.full(len(trackids), False)
    if cut[thisdot['id']]:
        return -1
    if frame < n:  # at (or recursed back to) the first frame
        newtrackid = trackids.max() + 1
        if verbose:
            print "New track:", newtrackid
            print '\tframe:', frame,'n:', n,'dot:', thisdot['id']
        return newtrackid
    olddots = fsets[frame-n]
    dists = ((thisdot['x'] - olddots['x'])**2 +
             (thisdot['y'] - olddots['y'])**2)
    mini = np.argmin(dists)
    mindist = dists[mini]
#    oldtree = ftrees[frame-n]
#    mindist, mini = oldtree.query([thisdot['x'], thisdot['y']])
    closest = olddots[mini]
    if mindist < maxdist:
        # a close one! Is there another dot in the current frame that's closer?
        curdots = fsets[frame]
        curdists = ((curdots['x'] - closest['x'])**2 +
                    (curdots['y'] - closest['y'])**2)
        mini2 = np.argmin(curdists)
        mindist2 = curdists[mini2]
#        curtree = ftrees[frame]
#        mindist2, closest2 = curtree.query([closest['x'], closest['y']])
        if mindist2 < mindist:
            # create new trackid to be deleted (or overwritten?)
            newtrackid = trackids.max() + 1
            if verbose:
                print "found a closer child dot to the this dot's parent"
                print "New track:", newtrackid
                print '\tframe:', frame,'n:', n,
                print 'dot:', thisdot['id'],
                print 'closer:', curdots[mini2]['id']
            return newtrackid
        if cut[closest['id']]:
            newtrackid = trackids.max() + 1
            if verbose:
                print "cutting track:", trackids[closest['id']]
                print "New track:", newtrackid
            return newtrackid
        else:
            oldtrackid = trackids[closest['id']]
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
            print '\tframe:', frame, 'n:', n, 'dot:', thisdot['id']
        return newtrackid

def find_tracks(maxdist=20, giveup=10, n=0, cut=False, stub=0):
    """ Track dots from frame-to-frame, giving each particle a unique and
        persistent id, called the trackid.

        parameters
        ----------
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
            if True, it requires either args.center or for user to click on an
            image to mark the center and boundary. Particle at the boundary
            (between two tracks) will have track id of -1
        stub : minimal length of a track for it to be kept. trackids of any
            track with length less than `stub` will be set to -1

        accesses
        --------
        data : the main data array

        modifies
        --------
        data : replaces the `data['lab']` field with the values from `trackids`

        returns
        -------
        trackids : an array of length `len(data)`, giving the track id number
            for each point in data. Any point not belonging to any track has a
            track id of -1
    """
    from sys import setrecursionlimit, getrecursionlimit
    setrecursionlimit(max(getrecursionlimit(), 2*giveup))

    trackids = -np.ones(data.shape, dtype=int)
    if n is True:
        n = np.count_nonzero(data['f']==0)
        if verbose: print "number of particles:", n

    if cut:
        if args.center:
            C = args.center[:2]
            R = args.center[2]
        else:
            from glob import glob
            bgimage = glob(locdir + prefix + "*.tif")
            if not bgimage:
                bgimage = glob(locdir + prefix + "/*.tif")
            if not bgimage:
                bgimage = glob(locdir + '../' + prefix + "/*.tif")
            if not bgimage:
                bgimage = raw_input('Please give the path to a tiff image '
                                    'from this dataset to identify boundary\n')
            else:
                bgimage = bgimage[0]
                print 'Opening', bgimage
            C, R = helpy.circle_click(bgimage)
            print "Boundary:", C, R
        margin = S if S>1 else R/16.9 # assume 6mm particles if S not specified
        rs = np.hypot(data['x'] - C[0], data['y'] - C[1])
        cut = rs > R - margin

    print "seeking tracks"
    for i in range(len(data)):
        trackids[i] = find_closest(data[i], trackids,
                                   maxdist=maxdist, giveup=giveup, cut=cut)

    # Michael used the data['lab'] field (as line[3] for line in data) to store
    # trackids. I'll keep doing that:
    assert len(data) == len(trackids), "too few/many trackids"
    assert np.allclose(data['id'], np.arange(len(data))), "gap in particle id"
    data['lab'] = trackids
    if n:
        data = data[trackids < n]

    stubs = np.where(np.bincount(trackids+1)[1:] < stub)[0]
    if verbose: print "removing {} stubs".format(len(stubs))
    stubs = np.in1d(trackids, stubs)
    trackids[stubs] = -1
    return trackids

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
    pl.figure(fignum)
    if mask is None:
        mask = (trackids >= 0)
    else:
        mask = mask & (trackids >= 0)
    data = data[mask]
    trackids = trackids[mask]
    pl.scatter(data['y'], data['x'],
            c=np.array(trackids)%12, marker='o', alpha=.5, lw=0)
    if bgimage:
        pl.imshow(bgimage, cmap=cm.gray, origin='upper')
    pl.gca().set_aspect('equal')
    pl.xlim(data['y'].min()-10, data['y'].max()+10)
    pl.ylim(data['x'].min()-10, data['x'].max()+10)
    pl.title(prefix)
    if save:
        print "saving tracks image to", prefix+"_tracks.png"
        pl.savefig(locdir+prefix+"_tracks.png")
    if show: pl.show()

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
    trackdots = data[trackids==track]

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
    print "dtau = {}, dtau = {}".format(dt0, dtau)
    msds = []
    msdids = []
    if tracks is None:
        if min_length:
            tracks = np.where(np.bincount(trackids+1)[1:] >= min_length)[0]
        else:
            tracks = np.unique(trackids)
            if tracks[0] == -1:
                tracks = tracks[1:]
    for trackid in tracks:
        if verbose: print "calculating msd for track", trackid
        tmsd = trackmsd(trackid, dt0, dtau)
        if len(tmsd) > 1:
            tmsdarr = np.asarray(tmsd)
            msds.append(tmsd)
            msdids.append(trackid)
    return msds, msdids

# Mean Squared Displacement:

def mean_msd(msds, taus, msdids=None, kill_flats=0, kill_jumps=1e9,
             show_tracks=False, singletracks=None, tnormalize=False,
             errorbars=False, fps=1, A=1):
    """ return the mean of several track msds """

    msdshape = (len(singletracks) if singletracks else len(msds),
                max(map(len, msds)))
    msd = np.full(msdshape, np.nan, float)

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
        pl.plot(taus/fps, (msd/(taus/fps)**tnormalize).T/A, 'b', alpha=.2)
    msd = np.nanmean(msd, 0)
    return (msd, msd_err) if errorbars else msd

def plot_msd(msds, msdids, dtau, dt0, nframes, tnormalize=False, prefix='',
        show_tracks=True, figsize=(8,6), plfunc=pl.semilogx, meancol='',
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
        taus = helpy.farange(dt0, nframes-1, dtau)
    elif isinstance(dtau, (int, np.int)):
        taus = np.arange(dtau, nframes-1, dtau, dtype=float)
    fig = pl.figure(fignum, figsize)

    # Get the mean of msds
    msd = mean_msd(msds, taus, msdids,
            kill_flats=kill_flats, kill_jumps=kill_jumps, show_tracks=show_tracks,
            singletracks=singletracks, tnormalize=tnormalize, errorbars=errorbars,
            fps=fps, A=A)
    if errorbars: msd, msd_err = msd
    #print "Coefficient of diffusion ~", msd[np.searchsorted(taus, fps)]/A
    #print "Diffusion timescale ~", taus[np.searchsorted(msd, A)]/fps

    if tnormalize:
        plfunc(taus/fps, msd/A/(taus/fps)**tnormalize, meancol,
               label="Mean Sq {}Disp/Time{}".format(
                     "Angular " if ang else "",
                     "^{}".format(tnormalize) if tnormalize != 1 else ''))
        plfunc(taus/fps, msd[0]/A*(taus/fps)**(1-tnormalize)/dtau,
               'k-', label="ref slope = 1", lw=2)
        plfunc(taus/fps, (twopi**2 if ang else 1)/(taus/fps)**tnormalize,
               'k--', lw=2, label=r"$(2\pi)^2$" if ang else
               ("One particle area" if S>1 else "One Pixel"))
        pl.ylim([0, 1.3*np.max(msd/A/(taus/fps)**tnormalize)])
    else:
        pl.loglog(taus/fps, msd/A, meancol, lw=lw,
                  label="Mean Squared {}Displacement".format('Angular '*ang))
        #pl.loglog(taus/fps, msd[0]/A*taus/dtau/2, meancol+'--', lw=2,
        #          label="slope = 1")
    if errorbars:
        pl.errorbar(taus/fps, msd/A/(taus/fps)**tnormalize,
                    msd_err/A/(taus/fps)**tnormalize,
                    fmt=meancol, capthick=0, elinewidth=1, errorevery=errorbars)
    if sys_size:
        pl.axhline(sys_size, ls='--', lw=.5, c='k', label='System Size')
    pl.title("Mean Sq {}Disp".format("Angular " if ang else "") if title is None else title)
    pl.xlabel('Time (' + ('s)' if fps > 1 else 'frames)'), fontsize='x-large')
    if ang:
        pl.ylabel('Squared Angular Displacement ($rad^2$)',
              fontsize='x-large')
    else:
        pl.ylabel('Squared Displacement ('+('particle area)' if S>1 else 'square pixels)'),
              fontsize='x-large')
    if xlim is not None:
        pl.xlim(*xlim)
    if ylim is not None:
        pl.ylim(*ylim)
    if show_legend: pl.legend(loc='best')
    if save is True:
        save = locdir + prefix + "_MS{}D.pdf".format('A' if ang else '')
    if save:
        print "saving to", save
        pl.savefig(save)
    if show: pl.show()
    return [fig] + fig.get_axes() + [taus] + [msd, msd_err] if errorbars else [msd]

if __name__=='__main__':
    if args.load:
        datapath = locdir+prefix+dotfix+'_POSITIONS.txt'
        data = gen_data(datapath)
        if verbose: print "\t...loaded"
    if args.track:
        if not args.load:
            data = np.load(locdir+prefix+'_POSITIONS.npz')['data']
        fsets = helpy.splitter(data, ret_dict=True)
#        from scipy.spatial.kdtree import KDTree
#        ftrees = { f: KDTree(np.column_stack([fset['x'], fset['y']]), leafsize=50)
#                   for f, fset in fsets.iteritems() }
        trackids = find_tracks(maxdist=args.maxdist, giveup=args.giveup,
                               n=args.number, cut=args.cut, stub=args.stub)
        # save the data record array and the trackids array
        print "saving track data to",
        print locdir+prefix+dotfix+"_TRACKS"
        np.savez(locdir+prefix+dotfix+"_TRACKS",
                data=data, trackids=trackids)

    elif args.load:
        print "saving " + dotfix.strip('_').lower() + " data (no tracks) to",
        print prefix + dotfix + "_POSITIONS.npz"
        np.savez(locdir+prefix+dotfix+"_POSITIONS",
                data = data)
        if verbose: print '\t...saved'
    else:
        # assume existing tracks.npz
        try:
            tracksnpz = np.load(locdir+prefix+"_TRACKS.npz")
            trackids = tracksnpz['trackids']
            if verbose:
                print "loading data and tracks from",
                print prefix + "_TRACKS.npz"
        except IOError:
            tracksnpz = np.load(locdir+prefix+"_POSITIONS.npz")
            if verbose:
                print "loading positions data from",
                print prefix + "_POSITIONS.npz"
        data = tracksnpz['data']
        if verbose: print "\t...loaded"

    if args.msd:
        msds, msdids = find_msds(dt0, dtau, min_length=args.stub)
        np.savez(locdir+prefix+"_MSD",
                 msds = np.asarray(msds),
                 msdids = np.asarray(msdids),
                 dt0  = np.asarray(dt0),
                 dtau = np.asarray(dtau))
        print "saved msd data to", prefix+"_MSD.npz"
    elif args.plotmsd or args.rr:
        if verbose: print "loading msd data from npz files"
        msdnpz = np.load(locdir+prefix+"_MSD.npz")
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
        try:
            bgimage = pl.imread(extdir+prefix+'_0001.tif')
        except IOError:
            try:
                bgimage = pl.imread(locdir+prefix+'_001.tif')
            except IOError:
                bgimage = None
        if args.singletracks:
            mask = np.in1d(trackids, args.singletracks)
        else:
            mask = None
        plot_tracks(data, trackids, bgimage, mask=mask,
                    save=args.save, show=args.show)

if __name__=='__main__' and args.nn:
    # Calculate the <nn> correlation for all the tracks in a given dataset
    # TODO: fix this to combine multiple datasets (more than one prefix)

    data, trackids, odata, omask = helpy.load_data(prefix, True, False)
    tracksets, otracksets = helpy.load_tracksets(data, trackids, odata, omask,
                                                 min_length=args.stub)

    coscorrs = [ corr.autocorr(np.cos(otrackset), cumulant=False, norm=False)
                for otrackset in otracksets.values() ]
    sincorrs = [ corr.autocorr(np.sin(otrackset), cumulant=False, norm=False)
                for otrackset in otracksets.values() ]

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
    print '   D_R: {:.4f}'.format(D_R)

    pl.figure()
    plot_individual = True
    if plot_individual:
        pl.plot(tcorr, allcorr.T, 'b', alpha=.2)
    pl.errorbar(tcorr, meancorr, errcorr, None, 'ok',
                label="Mean Orientation Autocorrelation",
                capthick=0, elinewidth=1, errorevery=3)
    pl.plot(tcorr, fitform(tcorr, *popt), 'r',
             label=r"$\frac{1}{2}e^{-D_R t}$" + '\n' +\
                    "$D_R = {:.4f}$, $D_R^{{-1}} = {:.3f}$".format(D_R, 1/D_R))

    pl.xlim(0, tmax)
    pl.ylim(fitform(tmax, *popt), 1)
    pl.yscale('log')

    pl.ylabel(r"$\langle \hat n(t) \hat n(0) \rangle$")
    pl.xlabel("$tf$")
    pl.title("Orientation Autocorrelation\n"+prefix)
    pl.legend(loc='upper right', framealpha=1)

    if args.save:
        save = locdir+prefix+'_nn-corr.pdf'
        print 'saving to', save
        pl.savefig(save)
    if not (args.rn or args.rr) and args.show: pl.show()

if __name__=='__main__' and args.rn:
    # Calculate the <rn> correlation for all the tracks in a given dataset
    # TODO: fix this to combine multiple datasets (more than one prefix)

    if not args.nn:
        # if args.nn, then these have been loaded already
        data, trackids, odata, omask = helpy.load_data(prefix, True, False)
        tracksets, otracksets = helpy.load_tracksets(data, trackids, odata, omask,
                                                   min_length=max(100, args.stub))

    corr_args = {'side': 'both', 'ret_dx': True,
                 'cumulant': (True, False), 'norm': 0 }

    xcoscorrs = [ corr.crosscorr(tracksets[t]['x']/S, np.cos(otracksets[t]),
                 **corr_args) for t in tracksets ]
    ysincorrs = [ corr.crosscorr(tracksets[t]['y']/S, np.sin(otracksets[t]),
                 **corr_args) for t in tracksets ]

    # Align and merge them
    fmax = int(2*fps/(D_R if args.nn else 12))
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
    if not args.nn: D_R = 1

    fitform = lambda s, v_D, D=D_R:\
                  np.sign(s)*v_D*(1 - corr.exp_decay(np.abs(s), 1/D))
    fitstr = r'$\frac{v_0}{D_R}(1 - e^{-D_R|s|})\operatorname{sign}(s)$'
    p0 = [1] if args.nn else [1, D_R] # [v_0/D_R, D_R]

    try:
        popt, pcov = curve_fit(fitform, tcorr, meancorr, p0=p0, sigma=sigma)
    except RuntimeError as e:
        print "RuntimeError:", e.message
        print "Using inital guess", p0
        popt = p0
    fit = fitform(tcorr, *popt)
    if not args.nn: D_R = popt[-1]
    v0 = D_R*popt[0]
    shift = popt[1] if len(popt) > 1 else 0
    print "Fits to <rn>:"
    print '\n'.join(['v0/D_R: {:.4f}',
                     ' shift: {:.4f}',
                     '   D_R: {:.4f}'][:len(popt)]).format(*popt)
    print "Giving:"
    print '\n'.join(['    v0: {:.4f}',
                     '   D_R: {:.4f}'][:4-len(popt)]
                    ).format(*[v0, D_R][:4-len(popt)])

    pl.figure()
    plot_individual = True
    sgn = np.sign(v0)
    if plot_individual:
        pl.plot(tcorr, sgn*rncorrs.T, 'b', alpha=.2)
    pl.errorbar(tcorr, sgn*meancorr, errcorr, None, 'ok',
                label="Mean Position-Orientation Correlation",
                capthick=0, elinewidth=1, errorevery=3)
    pl.plot(tcorr, sgn*fit, 'r', lw=2,
            #label=fitstr+'\n'+
            #       ', '.join(['$v_0$: {:.3f}', '$t_0$: {:.3f}', '$D_R$: {:.3f}'
            label=fitstr+'\n'+
                   ', '.join(['$v_0 = {:.3f}$', '$c_0 = {:.3f}$', '$D_R = {:.3f}$'
                              ][:len(popt)]).format(*(abs(v0), shift, D_R)[:len(popt)])
           )

    pl.axvline(1/D_R, 0, 2/3, ls='--', c='k')
    pl.text(1/D_R, 1e-2, ' $1/D_R$')

    pl.ylim(1.5*fit.min(), 1.5*fit.max())
    pl.xlim(tcorr.min(), tcorr.max())
    pl.title("Position - Orientation Correlation")
    pl.ylabel(r"$\langle \vec r(t) \hat n(0) \rangle / \ell$")
    pl.xlabel("$tf$")
    pl.legend(loc='upper left', framealpha=1)

    if args.save:
        save = locdir + prefix + '_rn-corr.pdf'
        print 'saving to', save
        pl.savefig(save)
    if not args.rr and args.show: pl.show()

if __name__=='__main__' and args.rr:
    fig, ax, taus, msd, msderr = plot_msd(
            msds, msdids, dtau, dt0, data['f'].max()+1, tnormalize=False,
            errorbars=5, prefix=prefix, show_tracks=True, meancol='ok',
            singletracks=args.singletracks, fps=fps, S=S, show=False,
            kill_flats=args.killflat, kill_jumps=args.killjump*S*S)

    sigma = msderr + 1e-5*S*S
    taus /= fps
    msd /= A
    tmax = 200
    fmax = np.searchsorted(taus, tmax)
    if not (args.nn or args.rn):
        D_R = v0 = 1
        p0 = [0, v0, D_R]
    elif not args.rn:
        v0 = 1
        p0 = [0, v0]
        sgn = 1
    else:
        p0 = [0, v0] if args.fitv0 else [0]# [D_T, v_0, D_R]
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
    print "Fits to <rr>:"
    print '\n'.join(['   D_T: {:.3f}',
                     'v0(rr): {:.3f}',
                     '   D_R: {:.3f}'][:len(popt)]).format(*popt)
    if len(popt) > 1:
        print "Giving:"
        print "v0/D_R: {:.3f}".format(popt[1]/(popt[2] if len(popt)>2 else D_R))
    fit = fitform(taus, *popt)
    ax.plot(taus, fit, 'r', lw=2,
            label=fitstr + "\n" + ', '.join(
                  ["$D_T= {:.3f}$", "$v_0 = {:.3f}$"][:len(popt)]
                  ).format(*(popt*np.array([1, sgn]))))

    pl.axvline(popt[0]/popt[1]**2, 0, 1/3, ls='--', c='k')
    pl.text(popt[0]/popt[1]**2, 2e-2, ' $D_T/v_0^2$')
    pl.axvline(1/D_R, 0, 2/3, ls='--', c='k')
    pl.text(1/D_R, 2e-1, ' $1/D_R$')

    pl.ylim(min(fit[0], msd[0]), fit[np.searchsorted(taus, tmax)])
    pl.xlim(taus[0], tmax)
    pl.legend(loc='upper left')

    if args.save:
        save = locdir + prefix + '_rr-corr.pdf'
        print 'saving to', save
        fig.savefig(save)
    if args.show: pl.show()

if __name__=='__main__' and not args.show:
    pl.close('all')

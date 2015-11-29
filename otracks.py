#!/usr/bin/env python
# encoding: utf-8

from __future__ import division

if __name__=='__main__':
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('prefix', metavar='PRE',
                   help="Filename prefix with full or relative path "
                        "(filenames prefix_POSITIONS.txt, "
                        "prefix_CORNER_POSITIONS.txt, etc)")
    p.add_argument('-o','--orient', action='store_true',
                   help='Find the orientations and save')
    p.add_argument('-n', '--ncorners', type=int, default=2,
                   help='Number of corner dots per particle. default = 2')
    p.add_argument('-r', '--rcorner', type=float, required=True,
                   help='Distance to corner dot from central dot, in pixels.')
    p.add_argument('--drcorner', type=float, default=-1,
                   help='Allowed error in r (rcorner), in pixels. Default is sqrt(r)')
    p.add_argument('-p', '--plottracks', action='store_true',
                   help='Plot the tracks and orientations as vectors')
    p.add_argument('-d', '--msd', action='store_true',
                   help='Calculate the MSAD')
    p.add_argument('--plotmsd', action='store_true',
                   help='Plot the MSAD (requires --msd first)')
    p.add_argument('--plotorient', action='store_true',
                   help='Plot the orientational trajectories')
    p.add_argument('-s', '--side', type=float, default=1,
                   help='Particle size in pixels, for unit normalization')
    p.add_argument('-f', '--fps', type=float, default=1,
                   help="Number of frames per second (or per shake) "
                        "for unit normalization")
    p.add_argument('--dt0', type=int, default=1,
                   help='Stepsize for time-averaging of a single '
                        'track at different time starting points')
    p.add_argument('--dtau', type=int, default=1,
                   help='Stepsize for values of tau '
                        'at which to calculate MSD(tau)')
    p.add_argument('--killflat', type=int, default=0,
                   help='Minimum growth factor for a single MSD track '
                        'for it to be included')
    p.add_argument('--killjump', type=int, default=100000,
                   help='Maximum initial jump for a single MSD track '
                        'at smallest time step')
    p.add_argument('--singletracks', type=int, nargs='*', default=xrange(1000),
                   help='identify single track ids to plot')
    p.add_argument('--showtracks', action='store_true',
                   help='Show individual tracks')
    p.add_argument('-v', '--verbose', action='count',
                        help='Print verbosity')

    args = p.parse_args()

    prefix = args.prefix

    S = args.side
    if args.drcorner == -1:
        args.drcorner= sqrt(args.rcorner)
    ang = True
    fps = args.fps
    dtau = args.dtau
    dt0 = args.dt0

    verbose = args.verbose
    if verbose:
        print 'using prefix', prefix
    else:
        from warnings import filterwarnings
        filterwarnings('ignore', category=RuntimeWarning, module='numpy')
        filterwarnings('ignore', category=RuntimeWarning, module='scipy')
        filterwarnings('ignore', category=RuntimeWarning, module='matpl')
else:
    verbose = False

from socket import gethostname
hostname = gethostname()
if 'foppl' in hostname:
    import matplotlib
    matplotlib.use("agg")

from math import sqrt

import numpy as np
from matplotlib import cm, pyplot as pl

import helpy

pi = np.pi
twopi = 2*pi
locdir = extdir = ''

def trackmsd(track, dt0, dtau, data, trackids, odata, omask, mod_2pi=False):
    """ trackmsd(track, dt0, dtau, data, trackids, odata, omask)
        finds the track msd, as function of tau, averaged over t0, for one track (worldline)
    """
    tmask = (trackids==track) & omask
    trackdots = data[tmask]
    trackodata = odata[tmask]['orient']
    if not mod_2pi:
        from orientation import track_orient
        trackodata = track_orient(trackodata)

        if dt0 == dtau == 1:
            if verbose: print "Using correlation"
            from correlation import msd as corrmsd
            return corrmsd(trackodata, ret_taus=True)

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
        avg = t0avg(trackdots, tracklen, tau, trackodata, dt0, mod_2pi=mod_2pi)
        #print "avg =", avg
        if avg > 0 and not np.isnan(avg):
            tmsd.append([tau,avg[0]])
    if verbose:
        print "\t...actually", len(tmsd)
    return tmsd

def t0avg(trackdots, tracklen, tau, trackodata, dt0, mod_2pi=False):
    """ t0avg() averages over all t0, for given track, given tau """
    totsqdisp = 0.0
    nt0s = 0.0
    for t0 in np.arange(1,(tracklen-tau-1),dt0): # for t0 in (T - tau - 1), by dt0 stepsize
        #print "t0=%d, tau=%d, t0+tau=%d, tracklen=%d"%(t0,tau,t0+tau,tracklen)
        olddot = trackodata[trackdots['f']==t0]
        newdot = trackodata[trackdots['f']==t0+tau]
        if len(newdot) != 1 or len(olddot) != 1:
            continue
        if mod_2pi:
            disp = (newdot - olddot) % twopi
            if disp > pi:
                disp -= twopi
        else:
            disp = newdot - olddot
        sqdisp = disp**2
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

def find_msds(dt0, dtau, data, trackids, odata, omask, tracks=None, mod_2pi=False):
    """ Calculates the MSDs"""
    print "Calculating MSADs with",
    print "dtau = {}, dtau = {}".format(dt0, dtau)
    msds = []
    msdids = []
    if tracks is None:
        tracks = set(trackids[omask])
    for trackid in tracks:
        if verbose: print "calculating msd for track", trackid
        tmsd = trackmsd(trackid, dt0, dtau, data, trackids, odata, omask, mod_2pi=mod_2pi)
        if len(tmsd) > 1:
            tmsdarr = np.asarray(tmsd)
            msds.append(tmsd)
            msdids.append(trackid)
    np.savez(locdir+prefix+"_MSAD",
             msds = np.asarray(msds),
             msdids = np.asarray(msdids),
             dt0  = np.asarray(dt0),
             dtau = np.asarray(dtau))
    print "saved msad data to", prefix+"_MSAD.npz"
    return msds, msdids

if __name__=='__main__':
    # assume existing tracks.npz
    try:
        tracksnpz = np.load(locdir+prefix+"_TRACKS.npz")
        trackids = tracksnpz['trackids']
        print "loading data and tracks from "+prefix+"_TRACKS.npz"
    except IOError:
        tracksnpz = np.load(locdir+prefix+"_POSITIONS.npz")
        print "loading positions data from "+prefix+"_POSITIONS.npz"
    data = tracksnpz['data']
    if verbose: print "\t...loaded"
    try:
        cdatanpz = np.load(locdir+prefix+'_CORNER_POSITIONS.npz')
        cdata = cdatanpz['data']
        if verbose: print "\t...loaded"
    except IOError:
        print prefix+"_CORNER_POSITIONS.npz file not found, have you run `tracks -lc`?"
    if args.orient:
        print "calculating orientation data"
        from orientation import get_angles_loop
        odata, omask = get_angles_loop(data, cdata,
                           nc=args.ncorners, rc=args.rcorner, drc=args.drcorner)
        np.savez(locdir+prefix+'_ORIENTATION.npz',
                odata=odata, omask=omask)
        if verbose: print '\t...saved'
    else:
        try:
            odatanpz = np.load(locdir+prefix+'_ORIENTATION.npz')
            odata = odatanpz['odata']
            omask = odatanpz['omask']
        except IOError:
            print "Cannot find ORIENTATION.npz file, have you run `otracks -o`?"

    if args.msd:
        msds, msdids = find_msds(dt0, dtau, data, trackids, odata, omask)
    elif args.plotmsd:
        print "loading msd data from npz files"
        msdnpz = np.load(locdir+prefix+"_MSAD.npz")
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

    if args.plotmsd:
        if verbose: print 'plotting now!'
        from tracks import plot_msd
        plot_msd(msds, msdids, dtau, dt0, data['f'].max()+1, tnormalize=False,
                 prefix=prefix, show_tracks=args.showtracks,
                 singletracks=args.singletracks, fps=fps, S=S,
                 ang=True, kill_flats=args.killflat, kill_jumps=args.killjump*S*S)
    if args.plotorient:
        from orientation import plot_orient_time
        plot_orient_time(data, odata, trackids, omask, singletracks=args.singletracks, save=locdir+prefix+'_ORIENTATION.pdf')
    if args.plottracks:
        from orientation import plot_orient_quiver
        try:
            bgimage = pl.imread(extdir+prefix+'_0001.tif')
        except IOError:
            try:
                bgimage = pl.imread(locdir+prefix+'_001.tif')
            except IOError:
                bgimage = None
        if args.singletracks:
            mask = np.in1d(trackids, args.singletracks)
        plot_orient_quiver(data, odata, mask=mask, imfile=bgimage)

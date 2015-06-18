#!/usr/bin/env python

from __future__ import division
import numpy as np
from math import sqrt
from PIL import Image as Im
from itertools import izip
import sys

import helpy

from socket import gethostname
hostname = gethostname()
if 'rock' in hostname:
    computer = 'rock'
    locdir = ''#/Users/leewalsh/Physics/Squares/orientation/'
    extdir = locdir#'/Volumes/bhavari/Squares/lighting/still/'
    plot_capable = True
elif 'foppl' in hostname:
    computer = 'foppl'
    locdir = '/home/lawalsh/Granular/Squares/diffusion/'
    extdir = '/media/bhavari/Squares/diffusion/still/'
    import matplotlib
    matplotlib.use("agg")
    plot_capable = False
elif 'peregrine' in hostname:
    computer = 'peregrine'
    locdir = extdir = ''
    plot_capable = True
else:
    print "computer not defined"
    locdir = extdir = ''
    plot_capable = helpy.bool_input("Are you able to plot?")
    if plot_capable:
        import matplotlib

from matplotlib import pyplot as pl
from matplotlib import cm as cm

pi = np.pi
twopi = 2*pi

if __name__=='__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('prefix', metavar='PRE',
                        help="Filename prefix with full or relative path "
                             "(filenames prefix_POSITIONS.txt, "
                             "prefix_CORNER_POSITIONS.txt, etc)")
    parser.add_argument('-o','--orient', action='store_true',
                        help='Find the orientations and save')
    parser.add_argument('-n', '--ncorners', type=int, default=3,
                        help='Number of corner dots per particle. Default is 3')
    parser.add_argument('-r', '--rcorner', type=float, default=10,
                        help='Distance to corner dot from central dot, in pixels.')
    parser.add_argument('--drcorner', type=float, default=-1,
                        help='Allowed error in r (rcorner), in pixels. Default is sqrt(r)')
    parser.add_argument('-p', '--plottracks', action='store_true',
                        help='Plot the tracks and orientations as vectors')
    parser.add_argument('-d', '--msd', action='store_true',
                        help='Calculate the MSAD')
    parser.add_argument('--plotmsd', action='store_true',
                        help='Plot the MSAD (requires --msd first)')
    parser.add_argument('--plotorient', action='store_true',
                        help='Plot the orientational trajectories')
    parser.add_argument('-s', '--side', type=float, default=1,
                        help='Particle size in pixels, for unit normalization')
    parser.add_argument('-f', '--fps', type=int, default=1,
                        help="Number of frames per second (or per shake) "
                             "for unit normalization")
    parser.add_argument('--dt0', type=int, default=10,
                        help='Stepsize for time-averaging of a single '
                             'track at different time starting points')
    parser.add_argument('--dtau', type=int, default=1,
                        help='Stepsize for values of tau '
                             'at which to calculate MSD(tau)')
    parser.add_argument('--killflat', type=int, default=0,
                        help='Minimum growth factor for a single MSD track '
                             'for it to be included')
    parser.add_argument('--killjump', type=int, default=100000,
                        help='Maximum initial jump for a single MSD track '
                             'at smallest time step')
    parser.add_argument('--singletracks', type=int, nargs='*', default=xrange(1000),
                        help='identify single track ids to plot')
    parser.add_argument('--showtracks', action='store_true',
                        help='Show individual tracks')
    parser.add_argument('-v', '--verbose', action='count',
                        help='Print verbosity')

    args = parser.parse_args()

    prefix = args.prefix
    print 'using prefix', prefix
    #dotfix = '_CORNER' if args.corner else ''

    plottracks = args.plottracks
    findmsd = args.msd
    plotmsd = args.plotmsd
    plotorient = args.plotorient
    findorient = args.orient

    S = args.side
    ang = True
    fps = args.fps
    nc = args.ncorners
    rc = args.rcorner
    drc = args.drcorner
    if drc == -1: drc = sqrt(rc)
    dtau = args.dtau
    dt0 = args.dt0

    kill_flats = args.killflat
    kill_jumps = args.killjump*S*S
    singletracks = args.singletracks
    show_tracks = args.showtracks
    verbose = args.verbose

else:
    verbose = False

def load_from_npz(prefix, locdir=None):
    """ given a prefix, returns:
        data, cdata, odata, omask
        [used in equilibrium.ipynb]
        """
    if locdir is None:
        from os import getcwd
        locdir = getcwd() +'/'
    odatanpz = np.load(locdir+prefix+'_ORIENTATION.npz')
    return (np.load(locdir+prefix+'_POSITIONS.npz')['data'],
            np.load(locdir+prefix+'_CORNER_POSITIONS.npz')['data'],
            odatanpz['odata'], odatanpz['omask'])


def trackmsd(track, dt0, dtau, data, trackids, odata, omask, mod_2pi=False):
    """ trackmsd(track, dt0, dtau, data, trackids, odata, omask)
        finds the track msd, as function of tau, averaged over t0, for one track (worldline)
    """
    tmask = (trackids==track) & omask
    trackdots = data[tmask]
    if mod_2pi:
        trackodata = odata[tmask]['orient']
    else:
        from orientation import track_orient
        trackodata = track_orient(odata, track, trackids, omask)

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
    print "Begin calculating MSADs"
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
    if True:
        # assume existing tracks.npz
        try:
            tracksnpz = np.load(locdir+prefix+"_TRACKS.npz")
            trackids = tracksnpz['trackids']
            print "loading data and tracks from "+prefix+"_TRACKS.npz"
        except IOError:
            tracksnpz = np.load(locdir+prefix+"_POSITIONS.npz")
            print "loading positions data from "+prefix+"_POSITIONS.npz"
        data = tracksnpz['data']
        print "\t...loaded"
    try:
        cdatanpz = np.load(locdir+prefix+'_CORNER_POSITIONS.npz')
        cdata = cdatanpz['data']
        print "\t...loaded"
    except IOError:
        print prefix+"_CORNER_POSITIONS.npz file not found, have you run `tracks -lc`?"
    if findorient:
        print "calculating orientation data"
        from orientation import get_angles_loop
        odata, omask = get_angles_loop(data, cdata, nc=nc, rc=rc, drc=drc)
        np.savez(locdir+prefix+'_ORIENTATION.npz',
                odata=odata, omask=omask)
        print '\t...saved'
    else:
        try:
            odatanpz = np.load(locdir+prefix+'_ORIENTATION.npz')
            odata = odatanpz['odata']
            omask = odatanpz['omask']
        except IOError:
            print "Cannot find ORIENTATION.npz file, have you run `otracks -o`?"

    if findmsd:
        msds, msdids = find_msds(dt0, dtau, data, trackids, odata, omask)
    elif plotmsd:
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
        print "\t...loaded"

if __name__=='__main__' and plot_capable:
    if plotmsd:
        print 'plotting now!'
        from tracks import plot_msd
        plot_msd(msds, msdids, dtau, dt0, data['f'].max()+1, tnormalize=False,
                 prefix=prefix, show_tracks=show_tracks,
                 singletracks=singletracks, fps=fps, S=S,
                 ang=True, kill_flats=kill_flats, kill_jumps=kill_jumps)
    if plotorient:
        from orientation import plot_orient_time
        plot_orient_time(data, odata, trackids, omask, singletracks=singletracks, save=locdir+prefix+'_ORIENTATION.pdf')
    if plottracks:
        from orientation import plot_orient_quiver
        try:
            bgimage = Im.open(extdir+prefix+'_0001.tif')
        except IOError:
            try:
                bgimage = Im.open(locdir+prefix+'_001.tif')
            except IOError:
                bgimage = None
        if singletracks:
            mask = np.in1d(trackids, singletracks)
        plot_orient_quiver(data, odata, mask=mask, imfile=bgimage)

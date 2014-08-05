#!/usr/bin/env python

import numpy as np
from PIL import Image as Im
import sys

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
    plot_capable = bool_input("Are you able to plot?")
    if plot_capable:
        import matplotlib

from matplotlib import pyplot as pl
from matplotlib import cm as cm


def bool_input(question):
    "Returns True or False from yes/no user-input question"
    answer = raw_input(question)
    return answer.lower().startswith('y') or answer.lower().startswith('t')

if __name__=='__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('prefix', metavar='PRE',
                        help="Filename prefix with full or relative path (filenames "
                             "prefix_POSITIONS.txt, prefix_CORNER_POSITIONS.txt, etc)")
    parser.add_argument('-c', '--corner', action='store_true',
                        help='Track corners instead of centers')
    parser.add_argument('-l','--load', action='store_true',
                        help='Create and save structured array from prefix[_CORNER]_POSITIONS.txt file')
    parser.add_argument('-t','--track', action='store_true',
                        help='Connect the dots and save in the array')
    parser.add_argument('-p', '--plottracks', action='store_true',
                        help='Plot the tracks')
    parser.add_argument('-d', '--msd', action='store_true',
                        help='Calculate the MSD')
    parser.add_argument('--plotmsd', action='store_true',
                        help='Plot the MSD (requires --msd first)')
    parser.add_argument('-s', '--side', type=int, default=1,
                        help='Particle size in pixels, for unit normalization')
    parser.add_argument('-f', '--fps', type=int, default=1,
                        help='Number of frames per second (or per shake) for unit normalization')
    parser.add_argument('--dt0', type=int, default=10,
                        help='Stepsize for time-averaging of a single track at different time starting points')
    parser.add_argument('--dtau', type=int, default=1,
                        help='Stepsize for values of tau at which to calculate MSD(tau)')
    parser.add_argument('--killflat', type=int, default=0,
                        help='Minimum growth factor for a single MSD track for it to be included')

    args = parser.parse_args()

    prefix = args.prefix
    print 'using prefix', prefix
    dotfix = '_CORNER' if args.corner else ''

    loaddata = args.load
    findtracks = args.track
    plottracks = args.plottracks
    findmsd = args.msd
    plotmsd = args.plotmsd
    loadmsd = plotmsd and not findmsd

    S = args.side
    if S > 1:
        S = float(S)
    fps = args.fps
    if fps > 1:
        fps = float(fps)
    dtau = args.dtau
    dt0 = args.dt0

    kill_flats = args.killflat

else:
    loaddata   = False    # Create and save structured array from data txt file?

    findtracks = False   # Connect the dots and save in 'trackids' field of data
    plottracks = False   # plot their tracks

    findmsd = False      # Calculate the MSD
    loadmsd = False      # load previoius MSD from npz file
    plotmsd = False      # plot the MSD
    prefix = 'n32_100mv_50hz'
    print 'using prefix', prefix
    dotfix = ''#_CORNER'
    dtau = 10
    dt0 = 10
    S = 22 # side length of particle

verbose = False

if loaddata:
    datapath = locdir+prefix+dotfix+'_POSITIONS.txt'

def find_closest(thisdot,trackids,n=1,maxdist=100.,giveup=1000):
    """ recursive function to find nearest dot in previous frame.
        looks further back until it finds the nearest particle
        returns the trackid for that nearest dot, else returns new trackid"""
    frame = thisdot['f']
    if frame < n:  # at (or recursed back to) the first frame
        newtrackid = max(trackids) + 1
        if verbose:
            print "New track:", newtrackid
            print '\tframe:', frame,'n:', n,'dot:', thisdot['id']
        return newtrackid
    else:
        oldframe = data[data['f']==frame-n]
        dists = (thisdot['x']-oldframe['x'])**2 + (thisdot['y']-oldframe['y'])**2
        closest = oldframe[np.argmin(dists)]
        if min(dists) < maxdist:
            return trackids[closest['id']]
        elif n < giveup:
            return find_closest(thisdot,trackids,n=n+1,maxdist=maxdist,giveup=giveup)
        else: # give up after giveup frames
            if verbose:
                print "Recursed {} times, giving up. frame = {} ".format(n,frame)
            newtrackid = max(trackids) + 1
            if verbose:
                print "New track:",newtrackid
                print '\tframe:', frame,'n:', n,'dot:', thisdot['id']
            return newtrackid

# Tracking
def load_data(datapath):
    print "loading data from",datapath
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
        from numpy.lib.recfunctions import append_fields
        # positions.py output (called *_POSITIONS.txt)
        data = np.genfromtxt(datapath,
                skip_header = 1,
                names = "f,x,y,lab,ecc,area",
                dtype = [int,float,float,int,float,int])
        data = append_fields(data,'id',np.arange(data.shape[0]), usemask=False)
    else:
        print "is {} from imagej or positions.py?".format(datapath.split('/')[-1])
        print "Please rename it to end with _results.txt or _POSITIONS.txt"
    return data

def find_tracks(data, giveup=1000):
    sys.setrecursionlimit(2*giveup)

    trackids = -np.ones(data.shape, dtype=int)

    print "seeking tracks"
    for i in range(len(data)):
        trackids[i] = find_closest(data[i], trackids)

    # save the data record array and the trackids array
    print "saving track data"
    np.savez(locdir+prefix+dotfix+"_TRACKS",
            data = data,
            trackids = trackids)

    return trackids

if __name__=='__main__':
    if loaddata:
        data = load_data(datapath)
        print "\t...loaded"
    if findtracks:
        if not loaddata:
            data = np.load(locdir+prefix+'_POSITIONS.npz')['data']
        trackids = find_tracks(data)
    elif loaddata:
        print "saving data only (no tracks)"
        np.savez(locdir+prefix+dotfix+"_POSITIONS",
                data = data)
        print '\t...saved'
    else:
        # assume existing tracks.npz
        print "loading tracks from npz files"
        try:
            tracksnpz = np.load(locdir+prefix+"_TRACKS.npz")
            trackids = tracksnpz['trackids']
        except IOError:
            tracksnpz = np.load(locdir+prefix+"_POSITIONS.npz")
        data = tracksnpz['data']
        print "\t...loaded"

# Plotting tracks:
def plot_tracks(data, trackids, bgimage=None):
    pl.figure()
    pl.scatter( data['y'], data['x'],
            c=np.array(trackids)%12, marker='o')
    if bgimage:
        pl.imshow(bgimage,cmap=cm.gray,origin='upper')
    pl.title(prefix)
    print "saving tracks image"
    pl.savefig(locdir+prefix+"_tracks.png")
    pl.show()

if plottracks and plot_capable:
    try:
        bgimage = Im.open(extdir+prefix+'_0001.tif') # for bkground in plot
    except IOError:
        try:
            bgimage = Im.open(locdir+prefix+'_001.tif') # for bkground in plot
        except IOError:
            bgimage = None
    plot_tracks(data, trackids, bgimage)

# Mean Squared Displacement
# dx^2 (tau) = < ( x_i(t0 + tau) - x_i(t0) )^2 >
#              <  averaged over t0, then i   >

def farange(start,stop,factor):
    start_power = np.log(start)/np.log(factor)
    stop_power = np.log(stop)/np.log(factor)
    return factor**np.arange(start_power,stop_power, dtype=type(factor))

def trackmsd(track, dt0, dtau):
    """ trackmsd(track,dt0,dtau)
        finds the track msd, as function of tau, averaged over t0, for one track (worldline)
    """
    tmsd = []
    trackdots = data[trackids==track]
    trackend =   trackdots['f'][-1]
    trackbegin = trackdots['f'][0]
    tracklen = trackend - trackbegin + 1
    if verbose:
        print "tracklen =",tracklen
        print "\t from %d to %d"%(trackbegin, trackend)
    if isinstance(dtau, float):
        taus = farange(dt0, tracklen, dtau)
    elif isinstance(dtau, int):
        taus = xrange(dtau, tracklen, dtau)
    for tau in taus:  # for tau in T, by factor dtau
        #print "tau =",tau
        avg = t0avg(trackdots, tracklen, tau)
        #print "avg =",avg
        if avg > 0 and not np.isnan(avg):
            tmsd.append([tau,avg[0]]) 
    if verbose:
        print "\t...actually",len(tmsd)
    return tmsd

def t0avg(trackdots, tracklen, tau):
    """ t0avg() averages over all t0, for given track, given tau """
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

def find_msds(dt0, dtau, tracks=None):
    """ Calculates the MSDs"""
    print "Begin calculating MSDs"
    msds = []
    if tracks is None:
        tracks = set(trackids)
    for trackid in tracks:
        if verbose:
            print "calculating msd for track", trackid
        tmsd = trackmsd(trackid, dt0, dtau)
        if len(tmsd) > 1:
            if kill_flats:
                tmsdarr = np.asarray(tmsd)
                if np.mean(tmsd[-10:]) > kill_flats*np.mean(tmsd[:10]):
                    msds.append(tmsd)
            else:
                msds.append(tmsd)

    msds = np.asarray(msds)
    print "saving msd data"
    np.savez(locdir+prefix+"_MSD",
            msds = msds,
            dt0  = np.asarray(dt0),
            dtau = np.asarray(dtau))
    return msds

if __name__=='__main__':
    if findmsd:
        msds = find_msds(dt0, dtau)
    elif loadmsd:
        print "loading msd data from npz files"
        msdnpz = np.load(locdir+prefix+"_MSD.npz")
        msds = msdnpz['msds']
        if msdnpz['dt0']:
            dt0  = msdnpz['dt0'][()] # [()] gets element from 0D array
            dtau = msdnpz['dtau'][()]
        else:
            dt0  = 10 # here's assuming...
            dtau = 10 #  should be true for all from before dt* was saved
        print "\t...loaded"

# Mean Squared Displacement:

def plot_msd(data, msds, dtau, dt0, tnormalize=0, show_tracks=False, prefix='', title=None, ylim=None):
    """ Plots the MSDs"""
    print "using dtau = {}, dt0 = {}".format(dtau, dt0)
    pl.figure(figsize=(5,4))
    nframes = data['f'].max()
    if isinstance(dtau, float):
        taus = farange(dt0, nframes, dtau)
    elif isinstance(dtau, int):
        taus = np.arange(dtau, nframes, dtau)
    msd = np.zeros(len(taus))
    added = np.zeros(len(msd), float)
    for tmsd in msds:
        if len(tmsd) > 0:
            tmsdt, tmsdd = np.asarray(tmsd).T
            tmsdd /= S**2 # convert to unit "particle area"
            if show_tracks:
                if tnormalize:
                    pl.semilogx(tmsdt, tmsdd/tmsdt**tnormalize)
                else:
                    pl.loglog(tmsdt, tmsdd)
            tau_match = np.searchsorted(taus, tmsdt)
            msd[tau_match] += tmsdd
            added[tau_match] += 1
    tau_mask = added > 0
    if not np.all(tau_mask):
        print "no tmsd for tau = {}; not using that tau".format(np.where(~tau_mask))
    msd = msd[tau_mask]
    taus = taus[tau_mask]
    added = added[tau_mask]
    msd /= added
    if tnormalize:
        pl.semilogx(taus, msd/taus**tnormalize,'k', lw=2, label="Mean Sq Disp/Time")
        pl.semilogx(taus, msd[0]*taus**(1-tnormalize)/dtau,
                    'k-',label="ref slope = 1")#,lw=4)
        pl.semilogx(taus, 1.*taus**(0)/taus**(tnormalize),
                    'k--', label="One particle area",lw=2)
        pl.ylim([0,1.5*np.max(msd/taus**tnormalize)])
    else:
        pl.loglog(taus, msd, 'k.', label="Mean Sq Disp")
        pl.loglog(taus, msd[0]*taus/dtau, 'k-', label="slope = 1")
    #pl.legend(loc='lower right')# 2 if tnormalize else 4)
    pl.title(prefix if title is None else title)
    pl.xlabel('Time (Image frames)', fontsize='x-large')
    pl.ylabel('Squared Displacement (particle area)', fontsize='x-large')# '+r'$s^2$'+')')
    if ylim is not None:
        pl.ylim(*ylim)
    pl.savefig(locdir+prefix+"_dt0=%d_dtau=%d.pdf"%(dt0, dtau))
    #print 'saved to '+locdir+prefix+"_dt0=%d_dtau=%d.png"%(dt0,dtau)
    pl.show()

if __name__=='__main__' and plotmsd and plot_capable:
    print 'plotting now!'
    plot_msd(data, msds, dtau, dt0, prefix=prefix, show_tracks=True)


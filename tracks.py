#!/usr/bin/env python
import numpy as np
from PIL import Image as Im
import sys

from socket import gethostname
hostname = gethostname()
if 'rock' in hostname:
    computer = 'rock'
    locdir = '/Users/leewalsh/Physics/Squares/lowdensity/'
    extdir = locdir#'/Volumes/bhavari/Squares/lighting/still/'
elif 'foppl' in hostname:
    computer = 'foppl'
    locdir = '/home/lawalsh/Granular/Squares/diffusion/'
    extdir = '/media/bhavari/Squares/diffusion/still/'
    import matplotlib
    matplotlib.use("agg")
else:
    print "computer not defined"
    print "where are you working?"

from matplotlib import pyplot as pl
from matplotlib import cm as cm


from argparse import ArgumentParser

parser = ArgumentParser()
parser.addargument('prefix')
args = parser.parse_args()

prefix = args.prefix#'n32_100mv_50hz'
dotfix = ''#_CORNER'

loaddata   = True   # Create and save structured array from data txt file?

findtracks = True   # Connect the dots and save in 'trackids' field of data
plottracks = False   # plot their tracks

findmsd = False      # Calculate the MSD
loadmsd = False      # load previoius MSD from npz file
plotmsd = False      # plot the MSD

verbose = False

if plottracks:
    bgimage = Im.open(extdir+prefix+'_0001.tif') # for bkground in plot
datapath = locdir+prefix+dotfix+'_POSITIONS.txt'

def find_closest(thisdot,trackids,n=1,maxdist=100.,giveup=1000):
    """ recursive function to find nearest dot in previous frame.
        looks further back until it finds the nearest particle
        returns the trackid for that nearest dot, else returns new trackid"""
    frame = thisdot['f']
    if frame < n:  # at (or recursed back to) the first frame
        newtrackid = max(trackids) + 1
        if verbose:
            print "New track:",newtrackid
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

if loaddata:
    data = load_data(datapath)
    print "\t...loaded"


def find_tracks(data, giveup = 1000):
    sys.setrecursionlimit(2*giveup)

    trackids = -np.ones(data.shape, dtype=int)

    print "seeking tracks"
    for i in range(len(data)):
        trackids[i] = find_closest(data[i],trackids)

    # save the data record array and the trackids array
    print "saving track data"
    np.savez(locdir+prefix+dotfix+"_TRACKS",
            data = data,
            trackids = trackids)

    return trackids

if findtracks:
    trackids = find_tracks(data)
elif loaddata:
    print "saving data only (no tracks)"
    np.savez(locdir+prefix+dotfix+"_POSITIONS",
            data = data)
else: 
    # assume existing tracks.npz
    print "loading tracks from npz files"
    tracksnpz = np.load(locdir+prefix+"_TRACKS.npz")
    data = tracksnpz['data']
    trackids = tracksnpz['trackids']
    print "\t...loaded"

# Plotting tracks:
def plot_tracks(data, trackids, bgimage=None):
    pl.figure()
    pl.scatter( data['y'], data['x'],
            c=np.array(trackids)%12, marker='o')
    pl.imshow(bgimage,cmap=cm.gray,origin='upper')
    pl.title(prefix)
    print "saving tracks image"
    pl.savefig(locdir+prefix+"_tracks.png")
    pl.show()

if plottracks and computer is 'rock':
    try:
        bgimage = Im.open(extdir+prefix+'_0001.tif') # for bkground in plot
    except IOError:
        bgimage = Im.open(locdir+prefix+'_0001.tif') # for bkground in plot
    plot_tracks(data, trackids, bgimage)

# Mean Squared Displacement
# dx^2 (tau) = < ( x_i(t0 + tau) - x_i(t0) )^2 >
#              <  averaged over t0, then i   >

def farange(start,stop,factor):
    start_power = np.log(start)/np.log(factor)
    stop_power = np.log(stop)/np.log(factor)
    return factor**np.arange(start_power,stop_power)

def trackmsd(track,dt0,dtau):
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
        print "\t from %d to %d"%(trackbegin,trackend)
    if type(dtau) is float:
        taus = farange(dt0,tracklen,dtau)
    elif type(dtau) is int:
        taus = xrange(dtau,tracklen,dtau)
    for tau in taus:  # for tau in T, by factor dtau
        #print "tau =",tau
        avg = t0avg(trackdots,tracklen,tau)
        #print "avg =",avg
        if avg > 0 and not np.isnan(avg):
            tmsd.append([tau,avg[0]]) 
    if verbose:
        print "\t...actually",len(tmsd)
    return tmsd

def t0avg(trackdots,tracklen,tau):
    """ t0avg() averages over all t0, for given track, given tau """
    totsqdisp = 0.0
    nt0s = 0.0
    for t0 in np.arange(1,(tracklen-tau-1),dt0): # for t0 in (T - tau - 1), by dt0 stepsize
        #print "t0=%d, tau=%d, t0+tau=%d, tracklen=%d"%(t0,tau,t0+tau,tracklen)
        olddot = trackdots[trackdots['f']==t0]
        newdot = trackdots[trackdots['f']==t0+tau]
        if (len(newdot) != 1):
            #print "newdot:",newdot
            continue
        elif (len(olddot) != 1):
            #print "olddot:",olddot
            continue
        sqdisp  = (newdot['x'] - olddot['x'])**2 \
                + (newdot['y'] - olddot['y'])**2
        if len(sqdisp) == 1:
            #print 'unflattened'
            totsqdisp += sqdisp
        elif len(sqdisp[0]) == 1:
            print 'flattened once'
            totsqdisp += sqdisp[0]
        else:
            print "fail"
            continue
        nt0s += 1.0
    return totsqdisp/nt0s if nt0s else None

def find_msds(dt0, dtau, tracks=None):
    """ Calculates the MSDs"""
    print "Begin calculating MSDs"
    msds = []
    if tracks is None:
        tracks = xrange(max(trackids) + 1)
        #tracks = (78,  95, 191, 203, 322)
    for trackid in tracks:
        if verbose:
            print "calculating msd for track",trackid
        tmsd = trackmsd(trackid,dt0,dtau)
        msds.append(tmsd)

    msds = np.asarray(msds)
    print "saving msd data"
    np.savez(locdir+prefix+"_MSD",
            msds = msds,
            dt0  = np.asarray(dt0),
            dtau = np.asarray(dtau))
    return msds

if findmsd:
    dt0  = 5 # small for better statistics, larger for faster calc
    dtau = 2 # int for stepwise, float for factorwise
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

tnormalize = True
def plot_msd(data,msds):
    """ Plots the MSDs"""
    nframes = max(data['f'])
    if isinstance(dtau,float):
        taus = farange(dt0,nframes,dtau)
        msd = np.transpose([taus,np.zeros_like(taus)])
    elif isinstance(dtau,int):
        taus = np.arange(dtau,nframes,dtau)
        msd = np.transpose([taus, np.zeros(-(-nframes/dtau) - 1)])
    pl.figure()
    added = np.zeros(len(msd), float)
    for tmsd in msds:
        if tmsd:
            tmsd = np.asarray(tmsd)
            tmsd[:,1] /= 22**2 # convert to unit "particle area"
            if tnormalize:
                tmsdt, tmsdd = zip(*tmsd)
                tmsdt = np.asarray(tmsdt)
                tmsdd = np.asarray(tmsdd)
                pl.semilogx(tmsdt, tmsdd/tmsdt)
            else:
                pl.loglog(*zip(*tmsd))
            lim = min(len(msd), len(tmsd))
            msd[:lim,1] += np.array(tmsd)[:lim,1]
            added[:lim] += 1.
    #assert not np.any(added==0), "no tmsd for some value of tau!"
    #TODO FIX THIS!  don't just divide these by one:
    added[added==0]=1
    msd[:,1] /= added
    if tnormalize:
        pl.semilogx(msd[:,0],msd[:,1]/msd[:,0],'ko',label="Mean Sq Disp/Time")
        pl.semilogx(taus, msd[0,1]*np.ones_like(taus)/dtau,
                'k-',label="ref slope = 1",lw=4)
    else:
        pl.loglog(msd[:,0],msd[:,1],'ko',label="Mean Sq Disp")
        pl.loglog(taus, msd[0,1]*taus/dtau,
                'k-',label="ref slope = 1",lw=4)
    pl.legend(loc=2 if tnormalize else 4)
    pl.title(prefix)
    pl.xlabel('Time tau (Image frames)')
    pl.ylabel('Squared Displacement (particle area '+r'$s^2$'+')')
    pl.savefig(locdir+prefix+"_dt0=%d_dtau=%d.png"%(dt0,dtau))
    pl.show()

if plotmsd and computer is 'rock':
    print 'plotting now!'
    plot_msd(data,msds)


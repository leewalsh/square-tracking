#!/usr/bin/env python

import numpy as np
from PIL import Image as Im
import sys

from socket import gethostname
hostname = gethostname()
if 'rock' in hostname:
    computer = 'rock'
    locdir = '/Users/leewalsh/Physics/Squares/orientation/'
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


if __name__=='__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('prefix')
    args = parser.parse_args()

    prefix = args.prefix#'n32_100mv_50hz'
    print 'using prefix', prefix
    dotfix = ''#_CORNER'
    if dotfix:
        print 'using dotfix', dotfix

loaddata   = False    # Create and save structured array from data txt file?

findtracks = False   # Connect the dots and save in 'trackids' field of data
plottracks = False   # plot their tracks

findmsd = False      # Calculate the MSD
loadmsd = False      # load previoius MSD from npz file
plotmsd = False      # plot the MSD

verbose = False

S = 22 # side length of particle

if plottracks:
    bgimage = Im.open(extdir+prefix+'_0001.tif') # for bkground in plot
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

if loaddata:
    data = load_data(datapath)
    print "\t...loaded"


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
    if findtracks:
        trackids = find_tracks(data)
    elif loaddata:
        print "saving data only (no tracks)"
        np.savez(locdir+prefix+dotfix+"_POSITIONS",
                data = data)
        print '\t...saved'
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
        print "\t from %d to %d"%(trackbegin, trackend)
    if isinstance(dtau, float):
        taus = farange(dt0, tracklen, dtau)
    elif isinstance(dtau, int):
        taus = xrange(dtau, tracklen, dtau)
    for tau in taus:  # for tau in T, by factor dtau
        #print "tau =",tau
        avg = t0avg(trackdots,tracklen,tau)
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
        tracks = set(trackids)
    for trackid in tracks:
        if verbose:
            print "calculating msd for track", trackid
        tmsd = trackmsd(trackid, dt0, dtau)
        msds.append(tmsd)

    msds = np.asarray(msds)
    print "saving msd data"
    np.savez(locdir+prefix+"_MSD",
            msds = msds,
            dt0  = np.asarray(dt0),
            dtau = np.asarray(dtau))
    return msds

if findmsd:
    dt0  = 100 # small for better statistics, larger for faster calc
    dtau = 10 # int for stepwise, float for factorwise
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
    pl.figure(figsize=(5,4))
    nframes = data['f'].max()
    if isinstance(dtau, float):
        taus = farange(dt0, nframes, dtau)
    elif isinstance(dtau, int):
        taus = np.arange(dtau, nframes, dtau)
    msd = np.zeros(len(taus))
    added = np.zeros(len(msd), float)
    for tmsd in msds:
        if tmsd:
            tmsdt, tmsdd =  np.asarray(tmsd).T
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
    assert np.all(tau_mask), "no tmsd for tau =\n" + str(np.where(~tau_mask))
    #msd = msd[tau_mask]
    #taus = taus[tau_mask]
    #added = added[tau_mask]
    msd /= added
    if tnormalize:
        pl.semilogx(taus, msd/taus**tnormalize,'k', lw=2, label="Mean Sq Disp/Time")
        pl.semilogx(taus, msd[0]*taus**(1-tnormalize)/dtau,
                'k-',label="ref slope = 1")#,lw=4)
        pl.semilogx(taus, 1.*taus**(0)/taus**(tnormalize),
                'k--',label="One particle area",lw=2)
        pl.ylim([0,1.5*np.max(msd/taus**tnormalize)])
    else:
        pl.loglog(taus, msd,'k.', label="Mean Sq Disp")
        pl.loglog(taus, msd[0]*taus/dtau, 'k-', label="slope = 1")
    pl.legend(loc='lower right')# 2 if tnormalize else 4)
    pl.title(prefix if title is None else title)
    pl.xlabel('Time (Image frames)')
    pl.ylabel('Displacement (particle area)')# '+r'$s^2$'+')')
    if ylim is not None:
        pl.ylim(*ylim)
    pl.savefig(locdir+prefix+"_dt0=%d_dtau=%d.pdf"%(dt0,dtau))#, dpi=180)
    #print 'saved to '+locdir+prefix+"_dt0=%d_dtau=%d.png"%(dt0,dtau)
    pl.show()

if plotmsd and computer is 'rock':
    print 'plotting now!'
    plot_msd(data, msds)


import numpy as np
from PIL import Image as Im
import sys

from socket import gethostname
hostname = gethostname()
if 'rock' in hostname:
    computer = 'rock'
    locdir = '/Users/leewalsh/Physics/Squares/lighting/'
    extdir = '/Volumes/bhavari/Squares/lighting/still/'
elif 'foppl' in hostname:
    computer = 'foppl'
    locdir = '/home/lawalsh/Granular/Squares/lighting/'
    extdir = '/media/bhavari/Squares/lighting/still/'
    import matplotlib
    matplotlib.use("agg")
else:
    print "computer not defined"
    print "where are you working?"

import matplotlib.pyplot as pl
import matplotlib.cm as cm


prefix = 'still'
dotfix = '_smalldot'

loaddata   = True   # Create and save structured array from data txt file?

findtracks = True   # Connect the dots and save in 'trackids' field of data
plottracks = True   # plot their tracks

findmsd = True      # Calculate the MSD
loadmsd = False     # load previoius MSD from npz file
plotmsd = True      # plot the MSD

bgimage = Im.open(extdir+prefix+'_0001.tif') # for bkground in plot
datapath = locdir+prefix+dotfix+'_POSITIONS.txt'


def find_closest(thisdot,trackids,n=1,maxdist=25.,giveup=1000):
    """ recursive function to find nearest dot in previous frame.
        looks further back until it finds the nearest particle
        returns the trackid for that nearest dot, else returns new trackid"""
    frame = thisdot['f']
    if frame < n:  # at (or recursed back to) the first frame
        newtrackid = max(trackids) + 1
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
            print "Recursed {} times, giving up. frame = {} ".format(n,frame)
            newtrackid = max(trackids) + 1
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

#(if findtracks is false and loaddata is false, assume existing tracks.npz)
else: 
    print "loading tracks from npz files"
    tracksnpz = np.load(locdir+prefix+dotfix+"_TRACKS.npz")
    data = tracksnpz['data']
    trackids = tracksnpz['trackids']
    print "\t...loaded"

# Plotting tracks:
def plot_tracks(data,trackids,  bgimage=bgimage):
    pl.figure()
    #bgheight = bgimage.size[1] # for flippin over y
    pl.scatter(
            data['x'],
            #bgheight-\
                data['y'],
            c=np.array(trackids)%12, marker='o')
    pl.imshow(bgimage,cmap=cm.gray,origin='lower')
    pl.title(prefix)
    print "saving tracks image"
    pl.savefig(locdir+prefix+"_tracks.png")
    pl.show()

if plottracks and computer is 'rock':
    plot_tracks(data, trackids)

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
            print "newdot:",newdot
            continue
        elif (len(olddot) != 1):
            print "olddot:",olddot
            continue
        sqdisp  = (newdot['x'] - olddot['x'])**2 \
                + (newdot['y'] - olddot['y'])**2
        if len(sqdisp) == 1:
            totsqdisp += sqdisp
        elif len(sqdisp[0]) == 1:
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
        #tracks = xrange(max(trackids) + 1)
        tracks = (78,  95, 191, 203, 322)
    for trackid in tracks:
        print "calculating msd for track",trackid
        tmsd = trackmsd(trackid,dt0,dtau)
        print "\tappending msd for track",trackid
        msds.append(tmsd)

    msds = np.asarray(msds)
    print "saving msd data"
    np.savez(locdir+prefix+"_MSD",
            msds = msds,
            dt0  = np.asarray(dt0),
            dtau = np.asarray(dtau))
    print "\t...saved"
    return msds

if findmsd:
    dt0  = 50 # small for better statistics, larger for faster calc
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

def plot_msd(data,msds):
    """ Plots the MSDs"""
    nframes = max(data['f'])
    if type(dtau) is float:
        taus = farange(dt0,nframes,dtau)
        msd = np.transpose([taus,np.zeros_like(taus)])
    elif type(dtau) is int:
        taus = np.arange(dtau,nframes,dtau)
        msd = np.transpose([np.arange(dtau,nframes,dtau),np.zeros(-(-nframes/dtau) - 1)])
    pl.figure()
    added = 0
    for tmsd in msds:
        if tmsd:
            added += 1.0
            pl.loglog(zip(*tmsd)[0],zip(*tmsd)[1])
            if len(tmsd)>=len(msd):
                msd[:,1] += np.array(tmsd)[:len(msd),1]
                print "yay"
            else:
                print "tmsd too short",len(tmsd),len(msd)

    try:
        msd[:,1] /= added
    except:
        print "none added!"
    pl.loglog(msd[:,0],msd[:,1],'ko',label="Mean Sq Disp")

    pl.loglog(taus, msd[0,1]*taus/dtau,
            'k-',label="ref slope = 1")
    pl.legend(loc=4)
    pl.title(prefix)
    pl.xlabel('Time tau (Image frames)')
    pl.ylabel('Squared Displacement ('+r'$pixels^2$'+')')
    pl.savefig(locdir+prefix+"_dt0=%d_dtau=%d.png"%(dt0,dtau))
    pl.show()

if plotmsd and computer is 'rock':
    plot_msd(data,msds)


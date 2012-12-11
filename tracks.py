#import matplotlib        #foppl
#matplotlib.use("agg")    #foppl
import matplotlib.pyplot as pl
import matplotlib.cm as cm
import numpy as np
from scipy.stats import nanmean
from PIL import Image as Im
import sys

#extdir = '/Volumes/Walsh_Lab/2D-Active/spatial_diffusion/'
locdir = '/Users/leewalsh/Physics/Squares/spatial_diffusion/'  #rock
#locdir = '/home/lawalsh/Granular/Squares/spatial_diffusion/'   #foppl

prefix = 'n448'

findtracks = True
plottracks = False
findmsd   = True
plotmsd   = True

bgimage = Im.open(locdir+prefix+'_0001.tif') # for bkground in plot
datapath = locdir+prefix+'_results.txt'


def find_closest(thisdot,n=1,maxdist=25.,giveup=1000):
    """ recursive function to find nearest dot in previous frame.
        looks further back until it finds the nearest particle"""
    frame = thisdot['s']
    if frame <= n:  # at (or recursed back to) the first frame
        newsid = max(trackids) + 1
        print "New track:",newsid
        print '\tframe:', frame,'n:', n,'dot:', thisdot['id']
        return newsid
    else:
        oldframes = data[data['s']==frame-n]
        dists = (thisdot['x']-oldframes['x'])**2 + (thisdot['y']-oldframes['y'])**2
        closest = oldframes[np.argmin(dists)]
        sid = trackids[closest['id']]
        if min(dists) < maxdist:
            return sid
        elif n < giveup:
            return find_closest(thisdot,n=n+1,maxdist=maxdist,giveup=giveup)
        else: # give up after giveup frames
            print "recursed", n, "times, giving up. frame =", frame
            print "New track:", max(trackids) + 1
            return max(trackids) + 1

# Tracking
if findtracks:
    print "loading data from",datapath
    data = np.genfromtxt(datapath,
            skip_header = 1,
            usecols = [0,2,3,5],
            names   = "id,x,y,s",
            dtype   = [int,float,float,int])
    data['id'] -= 1 # data from imagej is 1-indexed

    trackids = np.empty_like(data,dtype=int)
    trackids[:] = -1
    print "\t...loaded"
    
    giveup = 1000
    sys.setrecursionlimit(2*giveup)
    
    print "seeking tracks"
    for i in range(len(data)):
        trackids[i] = find_closest(data[i])

    # save the data record array and the trackids array
    print "saving track data"
    np.savez(locdir+prefix+"_TRACKS",
            data = data,
            trackids = trackids)

else:
    print "loading tracks from npz files"
    tracksnpz = np.load(locdir+prefix+"_TRACKS.npz")
    data = tracksnpz['data']
    trackids = tracksnpz['trackids']
    print "\t...loaded"

# Plotting tracks:
ntracks = max(trackids) + 1
if plottracks:
    pl.figure()
    bgheight = bgimage.size[1] # for flippin over y
    pl.scatter(
            data['x'], bgheight-data['y'],
            c=np.array(trackids)%12, marker='o')
    pl.imshow(bgimage,cmap=cm.gray,origin='lower')
    pl.title(prefix)
    print "saving tracks image"
    pl.savefig(locdir+prefix+"_tracks.png")

# Mean Squared Displacement
# dx^2 (tau) = < ( x_i(t0 + tau) - x_i(t0) )^2 >
#              <  averaged over t0, then i   >

# trackmsd finds the track msd, as function of tau, averaged over t0, for one track (worldline)
def farange(start,stop,factor):
    start_power = np.log(start)/np.log(factor)
    stop_power = np.log(stop)/np.log(factor)
    return factor**np.arange(start_power,stop_power)

def trackmsd(track):
    tmsd = []
    trackdots = data[trackids==track]
    trackend =   trackdots['s'][-1]
    trackbegin = trackdots['s'][0]
    tracklen = trackend - trackbegin + 1
    print "tracklen =",tracklen
    print "\t from %d to %d"%(trackbegin,trackend)
    for tau in farange(dt0,tracklen,dtau):  # for tau in T, by factor dtau
        print "tau =",tau
        avg = t0avg(trackdots,tracklen,tau)
        print "avg =",avg
        if avg > 0 and not np.isnan(avg):
            tmsd.append([tau,avg[0]]) 
    print "\t...actually",len(tmsd)
    return tmsd

def t0avg(trackdots,tracklen,tau):
    """ t0avg() averages over all t0, for given track, given tau """
    totsqdisp = 0.0
    nt0s = 0.0
    for t0 in np.arange(1,(tracklen-tau-1),dt0): # for t0 in (T - tau - 1), by dt0 stepsize
        print "t0=%d, tau=%d, t0+tau=%d, tracklen=%d"%(t0,tau,t0+tau,tracklen)
        olddot = trackdots[trackdots['s']==t0]
        newdot = trackdots[trackdots['s']==t0+tau]
        if (len(olddot) != 1) or (len(newdot) != 1):
            print "olddot:",olddot
            print "newdot:",newdot
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

dtau = 1.30 # small for better statistics, larger for faster calc
dt0  = 100 # small for better statistics, larger for faster calc
if findmsd:
    print "begin calculating msds"
    msds = []
    for trackid in range(ntracks):
        print "calculating msd for track",trackid
        tmsd = trackmsd(trackid)
        if tmsd:
            print '\tappending msd for track',trackid
            msds.append(tmsd)
        else:
            print '\tno msd for track',trackid

    msds=np.array(msds)
    print "saving msd data"
    np.savez(locdir+prefix+"_MSD",
            msds = msds,
            dt0  = np.array(dt0),
            dtau = np.array(dtau))
    print "\t...saved"
            
else:
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
if plotmsd:
    nframes = max(data['s'])
    taus = farange(dt0,nframes,dtau)
    msd = np.transpose([taus,np.zeros_like(taus)])
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
                #for tmsdrow in tmsd:
                #    print 'tmsdrow',tmsdrow
                #    print 'msd[(tmsdrow[0]==msd[:,0])[0],1]',\
                #           msd[(tmsdrow[0]==msd[:,0])[0],1]
                #    print 'tmsdrow[1]',tmsdrow[1]
                #    #msd[(tmsdrow[0]==msd[:,0])[0],1] += tmsdrow[1]

    msd[:,1] /= added
    pl.loglog(msd[:,0],msd[:,1],'ko',label="Mean Sq Disp")

    pl.loglog(
            taus,
            msd[0,1]*taus/dtau,
            'k-',label="ref slope = 1")
    pl.legend(loc=4)
    pl.title(prefix)
    pl.xlabel('Time tau (Image frames)')
    pl.ylabel('Squared Displacement ('+r'$pixels^2$'+')')
    pl.show()
    pl.savefig(locdir+prefix+"_dt0=%d_dtau=%d.png"%(dt0,dtau))


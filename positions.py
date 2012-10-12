import matplotlib        #foppl
matplotlib.use("agg")    #foppl
import matplotlib.pyplot as pl
import matplotlib.cm as cm
import numpy as np
from scipy.stats import nanmean
from PIL import Image as Im
import sys

#extdir = '/Volumes/Walsh_Lab/2D-Active/spatial_diffusion/'
#locdir = '/Users/leewalsh/Physics/Squares/spatial_diffusion/'  #rock
locdir = '/home/lawalsh/Granular/Squares/spatial_diffusion/'   #foppl

prefix = 'n256'

findtracks = False
plottracks = False
findmsd   = True
plotmsd   = False

bgimage = Im.open(locdir+prefix+'_0001.tif') # for bkground in plot
datapath = locdir+prefix+'_results.txt'


    # recursive function to find nearest dot in previous frame.
    # looks further back until it finds the nearest particle
def find_closest(thisdot,n=1,maxdist=25.,giveup=1000):
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
            data=data,trackids=trackids)

else:
    print "loading track data from npz files"
    tracknpz = np.load(locdir+prefix+"_TRACKS.npz")
    data = tracknpz['data']
    trackids = tracknpz['trackids']
    print "\t...loaded"

# Plotting tracks:
ntracks = max(trackids) + 1
if plottracks:
    pl.figure(1)
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
def trackmsd(track):
    tmsd = []
    trackdots = data[trackids==track]
    tracklen = trackdots['s'][-1] - trackdots['s'][0] + 1
    for tau in xrange(dtau,tracklen,dtau):  # for tau in T, by dtau stepsize
        avg = t0avg(trackdots,tracklen,tau)
        if avg > 0 and not np.isnan(avg):
            tmsd.append([tau,avg[0]]) 
    return tmsd

# t0avg() averages over all t0, for given track, given tau
def t0avg(trackdots,tracklen,tau):
    totsqdisp = 0.0
    nt0s = 0.0
    for t0 in np.arange(1,(tracklen-tau-1),dt0): # for t0 in (T - tau - 1), by dt0 stepsize
        olddot = trackdots[trackdots['s']==t0]
        newdot = trackdots[trackdots['s']==t0+tau]
        if (len(olddot) != 1) or (len(newdot) != 1):
            # sometimes olddot or newdot is a list
            continue
        sqdisp  = (newdot['x'] - olddot['x'])**2 \
                + (newdot['y'] - olddot['y'])**2
        if len(sqdisp) == 1:
            totsqdisp += sqdisp
        elif len(sqdisp[0]) == 1:
            totsqdisp += sqdisp[0]
        else: continue
        nt0s += 1.0
    return totsqdisp/nt0s if nt0s else None

dtau = 10 # small for better statistics, larger for faster calc
dt0  = 50 # small for better statistics, larger for faster calc
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
    np.savez(locdir+prefix+"_MSD_dt0"+str(dt0)+"_dtau"+str(dtau),
            msds=msds)
            
else:
    print "loading msd data from npz files"
    msdnpz = np.load(locdir+prefix+"_MSD_dt0"+str(dt0)+"_dtau"+str(dtau)+'.npz')
    msds = msdnpz[msds]
    print "\t...loaded"

# Mean Squared Displacement:
if plotmsd:
    nframes = max(data['s'])
    msd = [np.arange(dtau,nframes,dtau),np.zeros(-(-nframes/dtau) - 1)]
    msd = np.transpose(msd)
    pl.figure(2)
    added = 0
    for tmsd in msds:
        if tmsd:
            added += 1.0
            pl.loglog(zip(*tmsd)[0],zip(*tmsd)[1])
            if len(tmsd)==len(msd):
                msd[:,1] += np.array(tmsd)[:,1]
            else:
                for tmsdrow in tmsd:
                    print 'tmsdrow',tmsdrow
                    print 'msd[(tmsdrow[0]==msd[:,0])[0],1]',msd[(tmsdrow[0]==msd[:,0])[0],1]
                    print 'tmsdrow[1]',tmsdrow[1]
                    #msd[(tmsdrow[0]==msd[:,0])[0],1] += tmsdrow[1]

    msd[:,1] /= added
    pl.loglog(msd[:,0],msd[:,1],'ko',label="Mean Sq Disp")

    pl.loglog(
            np.arange(dtau,nframes,dtau),
            msd[0,1]*np.arange(dtau,nframes,dtau)/dtau,
            'k-',label="ref slope = 1")
    pl.legend(loc=4)
    pl.title(prefix)
    pl.xlabel('Time tau (Image frames)')
    pl.ylabel('Squared Displacement ('+r'$pixels^2$'+')')
    pl.savefig(locdir+prefix+"_dt0="+str(dt0)+"_dtau="+str(dtau)+".png")


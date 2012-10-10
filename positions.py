import matplotlib.pyplot as pl
import matplotlib.cm as cm
import numpy as np
from scipy.stats import nanmean
from PIL import Image as Im
import sys


#extdir = '/Volumes/Walsh_Lab/2D-Active/spatial_diffusion/'
#locdir = '/Users/leewalsh/Physics/Squares/spatial_diffusion/'  #rock
locdir = '/home/lawalsh/Granular/Squares/spatial_diffusion/'   #foppl
prefix = 'n8'

plottracks = False
plotmsd   = True

bgimage = Im.open(locdir+prefix+'_0001.tif') # for bkground in plot
datapath = locdir+prefix+'_4500.txt'

data = np.genfromtxt(datapath,
        skip_header=1,
        usecols = [0,2,3,5],
        names="id,x,y,s",
        dtype=[int,float,float,int])
data['id'] -= 1

trackids = np.empty_like(data,dtype=int)
trackids[:] = -1

# recursive function to find nearest dot in previous frame.
# looks further back until it finds the nearest particle
giveup = 1000
sys.setrecursionlimit(2*giveup)
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
            #print thisdot, closest, 'newsid:',sid, min(dists)
            return sid
        elif n < giveup:
            #print "still looking...",dist,'>',maxdist
            return find_closest(thisdot,n=n+1,maxdist=maxdist,giveup=giveup)
        else: # give up after giveup frames
            print "recursed", n, "times, giving up. frame =", frame
            print "created new static id:", max(trackids) + 1
            return max(trackids) + 1

# Tracking
for i in range(len(data)):
    trackids[i] = find_closest(data[i])

#print trackids

# Plotting tracks:
ntracks = max(trackids) + 1
if plottracks:
    pl.figure(1)
    bgheight = bgimage.size[1] # for flippin over y
    pl.scatter(data['x'], bgheight-data['y'], c=np.array(trackids)%12, marker='o')
    pl.imshow(bgimage,cmap=cm.gray,origin='lower')
    pl.title(prefix)
    pl.show()

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
    for t0 in np.arange(1,(tracklen-tau-1),dt0): # for t0 in T - tau - 1, by dt0 stepsize
        olddot = trackdots[trackdots['s']==t0]
        newdot = trackdots[trackdots['s']==t0+tau]
        if len(olddot)*len(newdot) == 0: continue
        sqdisp  = (newdot['x'] - olddot['x'])**2 \
                + (newdot['y'] - olddot['y'])**2
        if len(sqdisp)==1:#np.shape(totsqdisp)==np.shape(sqdisp):
            totsqdisp += sqdisp
        else:
            print "shape(totsqdisp)", np.shape(totsqdisp)
            print "shape(sqdisp)",    np.shape(sqdisp)
        nt0s += 1.0
    return totsqdisp/nt0s if nt0s else None

dtau = 10 # 1 for best statistics, more for faster calc
dt0  = 1 # 1 for best statistics, more for faster calc
msds = []#np.zeros(ntracks)
for trackid in range(ntracks):
    tmsd = trackmsd(trackid)
    if tmsd:
        print 'appending msd for track',trackid
        msds.append(tmsd)
    else:
        print 'no msd for track',trackid



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
                    print "we're all fucked"
                    print tmsdrow
                    print msd[(tmsdrow[0]==msd[:,0])[0],1]
                    print tmsdrow[1]
                    #msd[(tmsdrow[0]==msd[:,0])[0],1] += tmsdrow[1]

    msd[:,1] /= added
    pl.loglog(msd[:,0],msd[:,1],'ko',label="Mean of all tracks")

    pl.loglog(
            np.arange(dtau,nframes,dtau),
            msd[0,1]*np.arange(dtau,nframes,dtau)/dtau,
            'k--',label="ref slope = 1")
    pl.legend(loc=4)
    pl.title(prefix)
    pl.xlabel('Time tau (Image frames)')
    pl.ylabel('Squared Displacement ('+r'$pixels^2$'+')')

    #pl.show()

import matplotlib.pyplot as pl
import matplotlib.cm as cm
import numpy as np
from scipy.stats import nanmean
from PIL import Image as Im
import sys


#extdir = '/Volumes/Walsh_Lab/2D-Active/spatial_diffusion/'
locdir = '/Users/leewalsh/Physics/Squares/spatial_diffusion/'
prefix = 'n08'

plottracks = True
plotmsd   = True

bgimage = Im.open(locdir+'/n8_0001.tif') # for bkground in plot
datapath = locdir+prefix+'_4500.txt'

### Add 'ID' as label for first column in first line !!! ###
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
giveup = 2000
sys.setrecursionlimit(giveup+1)
def find_closest(thisdot,n=1,maxdist=20.,giveup=2000):
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
            print "created new static id:",newsid
            return max(trackids) + 1

# Tracking
nframes = max(data['s'])
print 'nframes:',nframes
#for frame in range(nframes):
    #for newdot in data[data['s']==frame+1]:
        #find_closest(newdot,frame,giveup=giveup)
for i in range(len(data)):
    trackids[i] = find_closest(data[i])

print trackids

# Plotting tracks:
ntracks = max(trackids)
if plottracks:
    pl.figure(1)

    bgheight = bgimage.size[1] # for flippin over y
    pl.scatter(data['x'], bgheight-data['y'], c=trackids, marker='.')
    pl.imshow(bgimage,cmap=cm.gray,origin='lower')
    pl.title(prefix)
    pl.legend()
    pl.show()
    exit()

# Mean Squared Displacement
# dx^2 (tau) = < ( x_i(t0 + tau) - x_i(t0) )^2 >
#              <  averaged over t0, then i   >

# trackmsd finds the msd, as function of tau, averaged over t0, for one track (worldline)
def trackmsd(track):
    tmsd = []#np.zeros(nframes/dtau) # to be function of tau
    trackdots = np.nonzero(trackids==track+1)[0] # just want the column indices (not row)
    #print trackdots[:3],'...',trackdots[-3:]
    tracklen = data[trackdots[-1]]['s'] - data[trackdots]['s'][0]
    for tau in np.arange(dtau,tracklen,dtau):  # for tau in T, by dtau stepsize
        avg = t0avg(trackdots,tracklen,tau)
        if avg > 0 and not np.isnan(avg):
            tmsd.append([tau,avg[0]]) 
    #print 'tmsd:',tmsd
    return tmsd

# t0avg() averages over all t0, for given track, given tau
def t0avg(trackdots,tracklen,tau):
    #TODO: if tau not in data[trackdots,iframe]: continue
    totsqdisp = 0.0
    nt0s = 0.0
    #print 'tau:',tau,dt0,'range',tracklen-tau
    for t0 in np.arange(1,(tracklen-tau-1),dt0): # for t0 in T - tau - 1, by dt0 stepsize
        olddot = np.nonzero(data[trackdots]['s']==t0)[0]
        newdot = np.nonzero(data[trackdots]['s']==t0+tau)[0]
        if not len(olddot)*len(newdot):
            #print('\t\t\tnot here')
            continue
        sqdisp  = (data[newdot]['x'] - data[olddot]['x'])**2 \
                + (data[newdot]['y'] - data[olddot]['y'])**2
        totsqdisp += sqdisp
        nt0s += 1.0
    return totsqdisp/nt0s if nt0s else None

dtau = 100 # 1 for best statistics, more for faster calc
dt0  = 1000 # 1 for best statistics, more for faster calc
msds = []#np.zeros(ntracks)
for trackid in range(ntracks):
    tmsd = trackmsd(track)
    if tmsd:
        print 'appending track',track
        msds.append(tmsd)
    else:
        print 'no track',track



# Mean Squared Displacement:
if plotmsd:
    pl.figure(2)
    for msd in msds:
        print 'plotting msd'#,msd
        if msd:
            pl.loglog(zip(*msd)[0],zip(*msd)[1],label='track ')

    #pl.loglog( np.arange(nframes)+1, np.arange(nframes)+1, 'k--') # slope = 1 for ref.
    pl.legend()
    pl.xlabel('Time (Image frames)')
    pl.ylabel('Squared Displacement'+r'$pixels^2$')

pl.show()

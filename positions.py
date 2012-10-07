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
data = [ line[:-1] for line in open(datapath)] # last char in each line is newline
datatypes = data[0].split()#('/t') # split with no arg figures it out
data = [ dataline.split() for dataline in data[1:] ] # remove first line, split each column
for dot in data:
    dot.append(0.0)
    dot.append(0.0)
data = np.array(data).astype(float)

# indices in file (number gives column)
iid    = datatypes.index('ID')    # unique particle id
iarea  = datatypes.index('Area')  # particle 
ix     = datatypes.index('X')     # x position
iy     = datatypes.index('Y')     # y position
iframe = datatypes.index('Slice') # slice (image frame) number
isid   = len(datatypes)           # static id (tracks particles)
idisp  = isid + 1                 # particle displacement from initial position
if (max(idisp,isid) + 1) > len(data[0]):
    print "too many column indices"

# recursive function to find nearest dot in previous frame.
# looks further back until it finds the nearest particle
giveup = 2000
sys.setrecursionlimit(giveup+1)
def find_closest(thisdot,frame,n=1,maxdist=20.,giveup=500):
    if frame < n:  # at (or recursed back to) the first frame
        newsid = max(data[:,isid]) + 1
        print "New track:",newsid
        print '\tframe:', frame,'n:', n,'dot:', thisdot[iid]
    else:
        oldframe = data[np.nonzero(data[:,iframe]==frame-n+1)]
        dist = maxdist
        for olddot in oldframe:
            newdist = (newdot[ix]-olddot[ix])**2 + (newdot[iy]-olddot[iy])**2
            if newdist < dist:
                dist = newdist
                newsid = olddot[isid]
                #print "found it! (",newsid,"frame:",frame,'; n:',n,')'
        if (n < giveup) and (dist >= maxdist):
            #print "still looking...",dist,'>',maxdist
            newsid = find_closest(thisdot,frame,n=n+1,maxdist=maxdist,giveup=giveup)
        if n >= giveup: # give up after giveup frames
            print "recursed", n, "times, giving up. frame =", frame
            newsid = max(data[:,isid]) + 1
            print "created new static id:",newsid
    data[thisdot[iid]-1,isid] = newsid
    return newsid

# Tracking
nframes = int(max(data[:,iframe]))
print 'nframes:',nframes
for frame in range(nframes):
    for newdot in data[np.nonzero(data[:,iframe]==frame+1)]:
        find_closest(newdot,frame,giveup=giveup)

# Plotting tracks:
ntracks = int(max(data[:,isid]))
if plottracks:
    pl.figure(1)
    for track in range(ntracks):
        # filter out all dots that belong to this track:
        trackdots = np.nonzero(data[:,isid]==track+1)[0] # just want the column indices (not row)
        c = cm.spectral(1-float(track)/ntracks) #spectral colormap, use prism for more colors

        # Locations plotted over image:
        bgheight = bgimage.size[1] # for flippin over y
        pl.plot(
                data[trackdots,ix],
                bgheight-data[trackdots,iy],
                '.',color=c,label="track "+str(track+1))
    pl.imshow(bgimage,cmap=cm.gray,origin='lower')
    pl.title(prefix)
    pl.legend()
    pl.show()

# Mean Squared Displacement
# dx^2 (tau) = < ( x_i(t0 + tau) - x_i(t0) )^2 >
#              <  averaged over t0, then i   >

# trackmsd finds the msd, as function of tau, averaged over t0, for one track (worldline)
def trackmsd(track):
    tmsd = []#np.zeros(nframes/dtau) # to be function of tau
    trackdots = np.nonzero(data[:,isid]==track+1)[0] # just want the column indices (not row)
    #print trackdots[:3],'...',trackdots[-3:]
    tracklen = data[trackdots,iframe][-1] - data[trackdots,iframe][0]
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
        olddot = np.nonzero(data[trackdots,iframe]==t0)[0]
        newdot = np.nonzero(data[trackdots,iframe]==t0+tau)[0]
        if not len(olddot)*len(newdot):
            #print('\t\t\tnot here')
            continue
        sqdisp  = (data[newdot,ix] - data[olddot,ix])**2 + (data[newdot,iy] - data[olddot,iy])**2
        totsqdisp += sqdisp
        nt0s += 1.0
    return totsqdisp/nt0s if nt0s else None

dtau = 100 # 1 for best statistics, more for faster calc
dt0  = 1000 # 1 for best statistics, more for faster calc
msds = []#np.zeros(ntracks)
for track in range(ntracks):
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

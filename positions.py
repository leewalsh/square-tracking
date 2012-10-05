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
data = [ dataline.split() for dataline in data[1:4000] ] # remove first line, split each column
for dot in data:
    dot.append(0.0)
    dot.append(0.0)
data = np.array(data).astype(float)

# indices in file (number gives column)
iid =    datatypes.index('ID')    # unique particle id
iarea =  datatypes.index('Area')  # particle 
ix =     datatypes.index('X')     # x position
iy =     datatypes.index('Y')     # y position
iframe = datatypes.index('Slice') # slice (image frame) number
isid = len(datatypes)             # static id (tracks particles)
idisp = isid + 1                  # particle displacement from initial position
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
        if (n < giveup) & (dist >= maxdist):
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

# Mean Squared Displacement
dtau = 5 # 1 for best statistics, more for faster calc
dt0  = 5 # 1 for best statistics, more for faster calc
msqdisp = np.zeros(nframes/dtau) # to be function of tau
ntracks = int(max(data[:,isid]))
msqdisps = []#np.zeros(ntracks)
for track in range(ntracks):
    trackdots = np.nonzero(data[:,isid]==track+1)[0] # just want the column indices (not row)
    print '\ttrack:',track
    print 'taus:',dtau*(1+np.arange(len(trackdots)/dtau))
    for tau in dtau*(1+np.arange(len(trackdots)/dtau)):  # for tau in T, by dtau stepsize
        print "\t\ttau:",tau
        print 't0s:',dt0*np.arange((len(trackdots)-tau)/dt0)
        totsqdisp = 0.0
        nt0s = 0.0
        for t0 in dt0*np.arange((len(trackdots)-tau-1)/dt0): # for t0 in T - tau - 1, by dt0 stepsize
            #print "\t\t\tt0:",t0
            olddot = np.nonzero(data[trackdots,iframe]==t0+1)[0]
            newdot = np.nonzero(data[trackdots,iframe]==t0+tau+1)[0]
            if not len(olddot)*len(newdot): print('\t\t\tnot here');continue
            sqdisp  = (data[newdot,ix] - data[olddot,ix])**2 \
                    + (data[newdot,iy] - data[olddot,iy])**2
            #print '\t\t\tsqdisp:',sqdisp
            totsqdisp += sqdisp
            nt0s += 1.0
        msqdisp[tau/dtau-1] = totsqdisp/nt0s if nt0s else None
    msqdisps.append(msqdisp)


# Plotting:
for track in range(ntracks):
    # filter out all dots that belong to this track:
    trackdots = np.nonzero(data[:,isid]==track+1)[0] # just want the column indices (not row)
    c = cm.spectral(1-float(track)/ntracks) #spectral colormap, use prism for more colors

    # Locations plotted over image:
    if plottracks:
        pl.figure(1)
        bgheight = bgimage.size[1] # for flippin over y
        pl.plot(
                data[trackdots,ix],
                bgheight-data[trackdots,iy],
                '.',color=c,label="track "+str(track+1))

    # Mean Squared Displacement:
    if plotmsd:
        pl.figure(2)
        #pl.loglog(data[np.nonzero(data[:,isid]==track+1)][:,idisp],color=c,label="track "+str(track+1))
        #pl.loglog( data[trackdots,idisp], color=c,label="track "+str(track+1))
        for msd in msqdisps:
            pl.loglog(msd)

if plottracks:
    pl.figure(1)
    pl.imshow(bgimage,cmap=cm.gray,origin='lower')
    pl.title(prefix)
    pl.legend()

if plotmsd:
    pl.figure(2)
    pl.loglog(msqdisp,'ko') # mean
    pl.loglog(
            np.arange(nframes)+1,
            np.arange(nframes)+1,
            'k--') # slope = 1 for ref.
    pl.legend()
    pl.xlabel('Time (Image frames)')
    pl.ylabel('Squared Displacement'+r'$pixels^2$')

pl.show()

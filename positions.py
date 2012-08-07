import matplotlib.pyplot as pl
import matplotlib.cm as cm
import numpy as np
from scipy.stats import nanmean
from PIL import Image as Im

bgimage = Im.open('n20_bw_dots/n20_b_w_dots_0010.tif') # for bkground in plot
datapath = 'n20_bw_dots/Results-all.txt'

data = [ line[:-1] for line in open(datapath)] # last char in each line is newline
datatypes = data[0].split()#('/t') # split with no arg figures it out
data = [ dataline.split() for dataline in data[1:] ]    # remove first line, split each column
for dot in data:
    dot.append(0.0)
    dot.append(0.0)
data = np.array(data).astype(float)


# indices in file (number gives column)
iid = 0     # unique particle id
iarea = 1   # particle 
ix = 2      # x position
iy = 3      # y position
islice = 4  # slice (image) number
isid = 5    # static id (tracks particles)
idisp = 6   # particle displacement from initial position

## Assuming all particles will be found in each frame
#ndots = 10 
#nimgs = len(data) / ndots
#
#dots = [ [ dotline for dotline in data[i*ndots:(i+1)*ndots] ] for i in range(nimgs) ]

def find_closest(thisdot,slice,n=1,maxdist=12.):
    #print slice, thisdot[iid], n
    if (slice>1) &(slice - n < 1):
        print "back to beginning, something's wrong"
        print '\t', slice, n, slice-n
        newsid = 0
        return newsid
    oldframe = data[np.nonzero(data[:,islice]==slice-n+1)]
    dist = maxdist
    for olddot in oldframe:
        newdist = (newdot[ix]-olddot[ix])**2 + (newdot[iy]-olddot[iy])**2
        #print '\t', newdist, '\t', dist
        if newdist < dist:
            dist = newdist
            newsid = olddot[isid]
    if ((slice - n > 0) & (n <= 500) ) & (dist >= maxdist):
        #print "recursing!\n\tslice =", slice, ", n =", n
        #print (slice-n > 0),'and',(dist >= maxdist)
        #print dist, '>=', maxdist, '? ', (dist >= maxdist)
        newsid = find_closest(thisdot,slice,n=n+1,maxdist=maxdist)
    if n > 500:
        print "recursed 500 times, giving up. slice =", slice
        newsid = 0.0
    data[thisdot[iid]-1,isid] = newsid
    #print np.shape(data)
    return newsid


nslice = int(data[len(data)-1][islice])
msqdisp = np.zeros(nslice)
dists = []
for slice in range(nslice):
    prevprev = np.nonzero(data[:,islice]==slice-1)
    prev = np.nonzero(data[:,islice]==slice)
    curr = np.nonzero(data[:,islice]==slice+1)

    # for first image, use original particle ID as statid ID
    if slice == 0:
        firs = curr
        data[firs,isid] = data[firs,iid]
        #data[curr,idisp] = 0.0
        initx = data[firs,ix]
        inity = data[firs,iy]
    else:
        for newdot in data[curr]:
            newsid = find_closest(newdot,slice)
#            dist=12.
#            for olddot in data[prev]:
#                newdist = (newdot[ix]-olddot[ix])**2 + (newdot[iy]-olddot[iy])**2
#                if newdist < dist:
#                    dist = newdist 
#                    #print "Closer"
#                    #print "new",newdot," old",olddot
#                    #newdot[isid] = olddot[isid] ### why doesn't this update data????
#                    newsid = olddot[isid]
#                    data[newdot[iid]-1,isid] = newsid #must do this; previous line doesn't work
#                    #print "Now:", newdot
#                    #print "data: ", data[curr]
#            dists.append(dist)
#            if dist>11.9:
#                #print 'slice ',slice,' missing ',newdot[iid]
#                for olderdot in data[prevprev]:
#                    newdist = (newdot[ix]-olderdot[ix])**2 + (newdot[iy]-olderdot[iy])**2
#                    if newdist < dist:
#                        dist = newdist 
#                        newsid = olderdot[isid]
#                        data[newdot[iid]-1,isid] = newsid #must do this; previous line doesn't work
#            if dist>11.9:
#                newsid=0
#            data[newdot[iid]-1,isid] = newsid
            #sqdisp = (newdot[ix] - olddot[ix])**2 + (newdot[iy]-olddot[iy])**2
            #print newsid
            sqdisp = (newdot[ix] - data[np.nonzero(data[:,iid]==newsid),ix])**2 \
                    + (newdot[iy] - data[np.nonzero(data[:,iid]==newsid),iy])**2
            #sqdisp = (newdot[ix] - data[newsid-1,ix])**2 \
            #       + (newdot[iy] - data[newsid-1,iy])**2
            data[newdot[iid]-1,idisp] = float(sqdisp) if bool(newsid) & (np.shape(sqdisp)==(1,1)) else None
            #print type(sqdisp),np.shape(sqdisp),sqdisp
        msqdisp[slice] = nanmean(data[np.nonzero(data[:,islice]==slice+1),idisp][0])
        #print slice, msqdisp[slice],data[np.nonzero(data[:,islice]==slice+1),idisp],data[np.nonzero(data[:,islice]==slice+1),isid]
        #if slice==6: break


# Plot them:

#xs = [ [ dot[ix] for dot in img] for img in dots]
#ys = [ [ dot[iy] for dot in img] for img in dots]
#ys = [ [ 600-float(yn) for yn in yi ] for yi in ys]

#for slice in 

#pl.plot(xs,ys,',')
#pl.imshow(bgimage,origin='lower')
#pl.axes().set_aspect(1)
#pl.show()

nparticles = int(max(data[:,isid]))
for part in range(nparticles):
    thispart = np.nonzero(data[:,isid]==part+1)
    c = cm.spectral(1-float(part)/nparticles)
    pl.figure(1)
    pl.plot(data[thispart,ix],600-data[thispart,iy],'o',color=c,label=str(part))

    # Find Squared Displacement
    pl.figure(2)
    thispart = np.nonzero(data[:,isid]==part+1)
    #initx = data[thispart[0][0],ix]
    #inity = data[thispart[0][0],iy]
    thispartxs = data[thispart,ix]
    thispartys = data[thispart,iy]
    pl.loglog(data[np.nonzero(data[:,isid]==part+1)][:,idisp],color=c,label=str(part))

# Mean Squared Displacement:
pl.loglog(msqdisp,'ko')
pl.loglog(np.arange(nslice)+1,np.arange(nslice)+1,'k--')
pl.legend()
pl.xlabel('Time\n(Image frames)')
pl.ylabel('Squared Displacement\n'+r'$pixels^2$')
pl.figure(1)
pl.imshow(bgimage,origin='lower')
pl.show()




import matplotlib.pyplot as pl
import numpy as np
from PIL import Image

bgimage = Image.open('n20_bw_dots/n20_b_w_dots_0010.tif') # for bkground in plot
datapath = 'n20_bw_dots/Results-all.txt'

data = [ line[:-1] for line in open(datapath)] # last char in each line is newline
datatypes = data[0].split()#('/t') # split with no arg figures it out
data = [ dataline.split() for dataline in data[1:50] ]    # remove first line, split each column
for dot in data:
    dot.append(0.0)
data = np.array(data).astype(float)


# indices in file (number gives column)
iid = 0     # unique particle id
iarea = 1   # particle 
ix = 2      # x position
iy = 3      # y position
islice = 4  # slice (image) number
isid = 5    # static id (tracks particles)

## Assuming all particles will be found in each frame
#ndots = 10 
#nimgs = len(data) / ndots
#
#dots = [ [ dotline for dotline in data[i*ndots:(i+1)*ndots] ] for i in range(nimgs) ]

#xs = [ [ dot[ix] for dot in img] for img in dots]
#ys = [ [ dot[iy] for dot in img] for img in dots]
#ys = [ [ 600-float(yn) for yn in yi ] for yi in ys]

#for slice in 

#pl.plot(xs,ys,',')
#pl.imshow(bgimage,origin='lower')
#pl.axes().set_aspect(1)
#pl.show()

nslice = int(data[len(data)-1][islice])

for slice in range(nslice):
    prev = np.nonzero(data[:,islice]==slice)
    curr = np.nonzero(data[:,islice]==slice+1)

    # for first image, use original particle ID as statid ID
    if slice == 0:
        data[curr,isid] = data[curr,iid]
    else:
        for newdot in data[curr]:
            dist=1000.
            for olddot in data[prev]:
                newdist = (newdot[ix]-olddot[ix])**2 + (newdot[iy]-olddot[iy])**2
                if newdist < dist:
                    dist = newdist 
                    #print "Closer"
                    #print "new",newdot
                    #print "old",olddot
                    #newdot[isid] = olddot[isid] ### why doesn't this update data????
                    data[newdot[iid]-1,isid] = olddot[isid] #must do this; previous line doesn't work
                    #print "Now:", newdot
                    #print "data:"
                    #print data[curr]







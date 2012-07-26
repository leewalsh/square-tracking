import matplotlib.pyplot as pl
import numpy as np
from PIL import Image

bimage = Image.open('n20_bw_dots/n20_b_w_dots_0010.tif')
datapath = 'n20_bw_dots/Results-all.txt'

data = [ line[:-1] for line in open(datapath)] # last char in each line is newline
datatypes = data[0].split()#('/t') # split with no arg figures it out
data = [ dataline.split() for dataline in data[1:] ]    # remove first line, split each column

ndots = 10
nimgs = len(data) / ndots

dots = [ [ dotline for dotline in data[i*ndots:(i+1)*ndots] ] for i in range(nimgs) ]

# indices in file (number gives column)
ii = 0
isize = 1
ix = 2
iy = 3
islice = 4

xs = [ [ dot[ix] for dot in img] for img in dots]
ys = [ [ dot[iy] for dot in img] for img in dots]
ys = [ [ 600-float(yn) for yn in yi ] for yi in ys]

for slice in 

pl.plot(xs,ys,',')
pl.imshow(bimage,origin='lower')
#pl.axes().set_aspect(1)
pl.show()



import matplotlib.pyplot as pl
import matplotlib.cm as cm
import numpy as np
#from scipy.stats import nanmean
from PIL import Image as Im

if True:#def get_fft(ifile)
    if True:
        ifile= "n20_bw_dots/n20_b_w_dots_0010.tif"
        x = 424.66; y = 155.56; area = 125
        #x = 302.95; y = 221.87; area = 145
        #x = 386.27; y = 263.62; area = 141
        #x = 35.39; y = 305.92; area = 154
        location = x,y
    wdth = int(24 * np.sqrt(2))
    hght = wdth
    cropbox = map(int,(x - wdth/2., y - hght/2.,\
            x + wdth/2., y + hght/2.))
    i = Im.open(ifile)
    i = i.crop(cropbox)
    i.show()
    a = np.asarray(i)
    aa = np.gradient(a)
    b = np.fft.fft2(a)
    j = Im.fromarray(abs(b))
    ii = Im.fromarray(aa)
    ii.show()

if True:#def get_orientation(b)
    p = []
    for (mi,m) in enumerate(b):
        for (ni, n) in enumerate(m):
            ang = np.arctan2(mi - hght/2, ni - wdth/2)
            p.append([ang,abs(n)])
    p = np.asarray(p)
    p[:,0] = p[:,0] + np.pi
    slices = 45
    slicewidth = 2*np.pi/slices
    s = []
    for sl in range(slices):
        si = np.nonzero(abs(p[:,0] - sl*slicewidth) < slicewidth)
        sm = np.average(p[si,1])
        s.append([sl*slicewidth,sm])
    s = np.asarray(s)
    #return s, p
    pl.figure()
    #pl.plot(p[:,0],p[:,1],'.',label='p')
    #pl.plot(s[:,0],s[:,1],'o',label='s')
    pl.plot(p[:,0]%(np.pi/2),p[:,1],'.',label='p')
    pl.plot(s[:,0]%(np.pi/2),s[:,1],'o',label='s')
    pl.legend()
    pl.show()



if False:
    # FFT information from:
    # http://stackoverflow.com/questions/2652415/fft-and-array-to-image-image-to-array-conversion
    import Image, numpy
    im = Image.open('img.png')
    if im.mode != 'L':
        print "convert to grayscale"
        im = im.convert('L')
    a = numpy.asarray(im) # a is readonly

    b = abs(numpy.fft.rfft2(a)) # abs because fft returns complex array

    jm = Image.fromarray(b)
    jm.save('jmg.png')

    # To also perform an inverse FFT and get back the original image, the
    # following works for me:

    b = numpy.fft.rfft2(a)
    aa = numpy.fft.irfft2(b)

    imim = Image.fromarray(aa.astype(numpy.uint8))
    imim.save('imgimg.png')


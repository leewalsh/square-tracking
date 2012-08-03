import matplotlib.pyplot as pl
import matplotlib.cm as cm
import numpy as np
#from scipy.stats import nanmean
from PIL import Image as Im

if True:#def get_fft(ifile)
    if True:
        ifile= "n20_bw_dots/n20_b_w_dots_0010.tif"
        x = 424.66
        y = 155.56
        area = 125
        location = x,y
    wdth = int(24 * np.sqrt(2))
    hght = wdth
    cropbox = map(int,(x - wdth/2., y - hght/2.,\
            x + wdth/2., y + hght/2.))
    i = Im.open(ifile)
    i = i.crop(cropbox)
    i.show()
    a = np.asarray(i)
    b = np.fft.fft2(a)
    j = Im.fromarray(abs(b))
    j.show()

#def get_orientation(j)



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


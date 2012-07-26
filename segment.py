import matplotlib.pyplot as pl
from PIL import Image
import pymeanshift as pms
import numpy as np
import multiprocessing as mp
pool = mp.Pool()


nims = 2 # number of images to segment

img = [Image.open('black_white_dots/black_white_dots_%02d.tif'%(i+1)) for i in range(nims)]
img = pool.map(np.array,img)

segmenter = pms.Segmenter()
segmenter.spatial_radius = 3
segmenter.range_radius = 50
segmenter.min_density = 50

if nims > 1:
    segout = pool.map(segmenter, img)
    (segmented, labels, regions) = np.swapaxes(segout,0,1)
else:
    (segmented, labels, regions) = segmenter(img)

def segment_centroid(i):
    return np.mean(np.nonzero(labels == i), axis=1)

if nims == 1:
    centroids = np.array(pool.map(segment_centroid, range(regions)))

    pl.imshow(img)
    pl.imshow(segmented, alpha=0.5)
    pl.plot(img.shape[1] - centroids[:,1], img.shape[0] - centroids[:,0], 'o', alpha=0.2)
else:
    for i in range(nims):
        pl.figure()
        pl.imshow(img[i])
        pl.imshow(segmented[i], alpha=0.5)
        #pl.plot(img[i].shape[1] - centroids[i][:,1], img[i].shape[0] - centroids[i][:,0], 'o', alpha=0.2)


pl.show()


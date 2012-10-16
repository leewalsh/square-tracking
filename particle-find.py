#!/usr/bin/env python

import numpy as np
from scipy import ndimage
from skimage import filter, measure
from skimage import segmentation as seg
from skimage.morphology import label, square, binary_closing, skeletonize
from collections import namedtuple
import PIL.Image as image

def label_particles_edge(im, sigma=3, closing_size=3):
    """ label_particles_edge(image, sigma=3, closing_size=3)

        Returns the labels for an image.

        keyword arguments:
        image        -- The image in which to find particles
        sigma        -- The size of the Canny filter
        closing_size -- The size of the closing filter
    """
    edges = filter.canny(im, sigma=sigma)
    edges = binary_closing(edges, square(closing_size))
    edges = skeletonize(edges)
    labels = label(edges)
    labels = np.ma.array(labels, mask=edges==0)
    return labels

def label_particles_walker(im, min_thresh=0.3, max_thresh=0.5):
    labels = np.zeros_like(im)
    labels[im>max_thresh] = 1
    labels[im<min_thresh] = 2
    return label(seg.random_walker(im, labels))

Particle = namedtuple('Particle', 'x y label ecc area'.split())

def filter_particles(labels, max_ecc=0.5, min_area=15, max_area=200):
    """ filter_particles(labels, max_ecc=0.5, min_area=15, max_area=200) -> [Particle]

        Returns a list of Particles as well as masks out labels for
        particles not meeting acceptance criteria.
    """
    pts = []
    rprops = measure.regionprops(labels, ['Area', 'Eccentricity', 'Centroid'])
    for props in rprops:
        label = props['Label']
        if min_area > props['Area'] or props['Area'] > max_area \
                or props['Eccentricity'] > max_ecc:
            labels[labels==label] = np.ma.masked
        else:
            (x,y) = props['Centroid']
            pts.append(Particle(x,y, label, props['Eccentricity'], props['Area']))
    return pts

def drop_labels(labels, take_labels):
    a = np.zeros_like(labels, dtype=bool)
    for l in take_labels:
        a = np.logical_or(a, labels == l)
    labels[np.logical_not(a)] = np.ma.masked
    return labels

def find_particles(im, gaussian_size=3, method='walker', **kwargs):
    """ find_particles(im, gaussian_size=3, **kwargs) -> [Particle],labels

        Find the particles in image im. The arguments in kwargs is
        passed to label_particles and filter_particles.

        Returns the list of found particles and the label image.
    """
    im = ndimage.gaussian_filter(im, gaussian_size)
    labels = None
    if method == 'walker':
        labels = label_particles_walker(im, **kwargs)
    elif method == 'edge':
        labels = label_particles(im, **kwargs)
    else:
        raise RuntimeError('Undefined method "%s"' % method)
    particles = filter_particles(labels, **kwargs)
    return (particles,labels)

def find_particles_in_image(f, **kwargs):
    """ find_particles_in_image(im, **kwargs)
    """
    im = image.open(f)
    im = np.array(im)
    im = im / 255.
    return find_particles(im, **kwargs)

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as pl
    from multiprocessing import Pool
    from argparse import ArgumentParser, FileType

    parser = ArgumentParser()
    parser.add_argument('files', metavar='FILE', nargs='+',
                        help='Images to process')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='Produce a plot for each image')
    parser.add_argument('-o', '--output', default='points',
                        help='Output file')
    parser.add_argument('-N', '--threads', default=1, type=int,
                        help='Number of worker threads')
    args = parser.parse_args()
    cm = pl.cm.prism_r

    def f((n,file)):
        pts, labels = find_particles_in_image(file)
        print '%20s: Found %d points' % (file, len(pts))
        if args.plot:
            pl.imshow(labels, cmap=cm)
            pts = np.array(pts)
            pl.scatter(pts[:,1], pts[:,0], c=pts[:,2], cmap=cm)
            pl.savefig(file+'.png')
        return np.hstack([n*np.ones((len(pts),1)), pts])

    p = Pool(args.threads)
    filenames = sorted(args.files)
    points = p.map(f, enumerate(filenames))
    points = np.vstack(points)
    with open(args.output, 'w') as output:
        output.write('# Frame    X           Y             Label  Eccen        Area\n')
        np.savetxt(output, points, delimiter='     ', fmt=['%6d', '%7.3f', '%7.3f', '%4d', '%1.3f', '%5d'])
 

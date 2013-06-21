#!/usr/bin/env python

import numpy as np
from scipy.ndimage import gaussian_filter, binary_erosion, convolve
from skimage import filter, measure
from skimage import segmentation as seg
from skimage.morphology import label, square, binary_closing, skeletonize
from skimage.morphology import disk as _disk
from collections import namedtuple
import PIL.Image as image

def label_particles_edge(im, sigma=2, closing_size=0, **extra_args):
    """ label_particles_edge(image, sigma=3, closing_size=3)

        Returns the labels for an image.
        Segments using Canny edge-finding filter.

        keyword arguments:
        image        -- The image in which to find particles
        sigma        -- The size of the Canny filter
        closing_size -- The size of the closing filter
    """
    edges = filter.canny(im, sigma=sigma)
    if closing_size > 0:
        edges = binary_closing(edges, square(closing_size))
    edges = skeletonize(edges)
    labels = label(edges)
    print "found {} segments".format(max(labels.flatten()))
    labels = np.ma.array(labels, mask=edges==0) # in ma.array mask, False is True, and vice versa
    return labels

def label_particles_walker(im, min_thresh=0.3, max_thresh=0.5, sigma=3):
    """ label_particles_walker(image, min_thresh=0.3, max_thresh=0.5)

        Returns the labels for an image.
        Segments using random_walker method.

        keyword arguments:
        image        -- The image in which to find particles
        min_thresh   -- The lower limit for binary threshold
        max_thresh   -- The upper limit for binary threshold
    """
    im = gaussian_filter(im, sigma)
    labels = np.zeros_like(im)
    labels[im>max_thresh*im.max()] = 1
    labels[im<min_thresh*im.max()] = 2
    return label(seg.random_walker(im, labels))

Particle = namedtuple('Particle', 'x y label ecc area'.split())

def filter_particles(labels, max_ecc=0.5, min_area=15, max_area=200, **extra_args):
    """ filter_particles(labels, max_ecc=0.5, min_area=15, max_area=200) -> [Particle]

        Returns a list of Particles and masks out labels for
        particles not meeting acceptance criteria.
    """
    pts = []
    rprops = measure.regionprops(labels, ['Area', 'Eccentricity', 'Centroid'])
    for props in rprops:
        label = props['Label']
        if min_area > props['Area'] or props['Area'] > max_area \
                or props['Eccentricity'] > max_ecc:
            pass
            #labels[labels==label] = np.ma.masked
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

def find_particles(im, method='edge', **kwargs):
    """ find_particles(im, gaussian_size=3, **kwargs) -> [Particle],labels

        Find the particles in image im. The arguments in kwargs is
        passed to label_particles and filter_particles.

        Returns the list of found particles and the label image.
    """
    labels = None
    print "Seeking particles using", method
    if method == 'walker':
        labels = label_particles_walker(im, **kwargs)
    elif method == 'edge':
        labels = label_particles_edge(im, **kwargs)
    else:
        raise RuntimeError('Undefined method "%s"' % method)
    particles = filter_particles(labels, **kwargs)
    return (particles,labels)

def find_particles_in_image(f, **kwargs):
    """ find_particles_in_image(im, **kwargs)
    """
    print "opening", f
    im = image.open(f)
    im = np.array(im, dtype=float)
    im = im / im.max()
    return find_particles(im, **kwargs)

def remove_segments(orig, particles, labels):
    """ remove_segments(orig, particles, labels)
        attempts to remove the found big dot segment as found in original
    """
    return

def disk(n):
    return _disk(n).astype(int)

def gdisk(n):
    w = 2*n
    g = np.arange(-w, w+1)
    g = np.exp(-.5 * g**2 / n**2)
    g = np.outer(g, g)  # or g*g[...,None]
    g -= g.mean()
    g /= g.std()
    assert g.sum() == 0, 'sum is nonzero: {}'.format(g.sum())
    return g

def remove_disks(orig, particles, dsk=disk(6)):
    """ remove_disks(method=['disk' or 'segment'])
        removes a disk of given size centered at dot location
        inputs:
            orig   -    input image as ndarray or PIL Image
            particles - list of particles (namedtuple Particle)
            dsk   -    radius of disk, default is skimage.morphology.disk(r)
                            (size of square array is 2*r+1)
        output:
            the original image with big dots removed
    """
    disks = np.ones(orig.shape, int)
    if isinstance(particles[0], Particle):
        xys = zip(*(map(int,(p.x,p.y)) for p in pts))
    elif 'X' in particles.dtype.names:
        xys = np.round(particles['X']).astype(int),np.round(particles['Y']).astype(int)
    disks[xys] = 0
    disks = binary_erosion(disks,dsk)
    return orig*disks

def convolve_disks(orig, dsk):
    if isinstance(dsk, int):
        dsk = gdisk(dsk)
    return convolve(orig, dsk)

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as pl
    from multiprocessing import Pool
    from argparse import ArgumentParser
    from os import path

    parser = ArgumentParser()
    parser.add_argument('files', metavar='FILE', nargs='+',
                        help='Images to process')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='Produce a plot for each image')
    parser.add_argument('-o', '--output', default='POSITIONS',
                        help='Output file')
    parser.add_argument('-N', '--threads', default=1, type=int,
                        help='Number of worker threads')
    parser.add_argument('-c', '--corner', action='store_true',
                        help='Also find small corner dots')
    parser.add_argument('--slr', action='store_true',
                        help='Full resolution SLR was used')
    args = parser.parse_args()
    cm = pl.cm.prism_r

    if args.plot:
        pdir = path.split(path.abspath(args.output))[0]
    threshargs = {'max_ecc' :  .4 if args.slr else  .7,
                  'min_area': 160 if args.slr else  15,
                  'max_area': 250 if args.slr else 200}

    def f((n,filename)):
        print filename
        pts, labels = find_particles_in_image(filename, **threshargs)
        print '%20s: Found %d points' % ('', len(pts))
        if args.corner:
            diff = remove_disks()
            #TODO pass removed image to find corners
            if args.plot:
                #TODO plot stuff with corner
                pass 
        elif args.plot:
            pl.clf()
            pl.imshow(labels, cmap=cm)
            pts = np.asarray(pts)
            pl.scatter(pts[:,1], pts[:,0], c=pts[:,2], cmap=cm)
            savename = path.join(pdir,path.split(filename)[-1].split('.')[-2])+'_POSITIONS.png'
            print 'saving to',savename
            pl.savefig(savename, dpi=300)
        return np.hstack([n*np.ones((len(pts),1)), pts])

    filenames = sorted(args.files)
    if args.threads > 1:
        print "Multiprocessing with {} threads".format(args.threads)
        p = Pool(args.threads)
        points = p.map(f, enumerate(filenames))
    else:
        points = map(f, enumerate(filenames))
    points = np.vstack(points)
    if args.corner and args.output == 'POSITIONS':
        # change default for CORNER dot output
        args.output = 'CORNER_POSITIONS'
    with open(args.output, 'w') as output:
        output.write('# Frame    X           Y             Label  Eccen        Area\n')
        np.savetxt(output, points, delimiter='     ',
                fmt=['%6d', '%7.3f', '%7.3f', '%4d', '%1.3f', '%5d'])
 

#!/usr/bin/env python

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, binary_erosion, convolve, center_of_mass, imread
from skimage import filter, measure, segmentation
from skimage.morphology import label, square, binary_closing, skeletonize
from skimage.morphology import disk as _disk
from collections import namedtuple

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
    print "found {} segments".format(labels.max())
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
    if sigma>0:
        im = gaussian_filter(im, sigma)
    labels = np.zeros_like(im)
    labels[im<min_thresh*im.max()] = 1
    labels[im>max_thresh*im.max()] = 2
    return segmentation.random_walker(im, labels)

def label_particles_convolve(im, thresh=2, rmv=None, csize=0, **extra_args):
    """ label_particles_convolve(im, thresh=2)
        Returns the labels for an image
        Segments using a threshold after convolution with proper gaussian kernel
        Uses center of mass from original image to find centroid of segments

        Input:
            image   the original image
            pos     if given, the positions at which to remove large dots
            thresh  the threshold above which pixels are included
    """
    if rmv is not None:
        im = remove_disks(im, rmv, disk(8.5))
    if csize == 0:
        print 'csize not set'
    convolved = convolve(im, gdisk(csize))
    convolved -= convolved.min()
    convolved /= convolved.max()

    if isinstance(thresh, int):
        thresh = convolved.mean() + thresh*convolved.std()

    labels = label(convolved > thresh)
    #print "found {} segments above thresh".format(labels.max())
    return labels

Segment = namedtuple('Segment', 'x y label ecc area'.split())

def filter_segments(labels, max_ecc=0.5, min_area=15, max_area=200, intensity=None, **extra_args):
    """ filter_segments(labels, max_ecc=0.5, min_area=15, max_area=200) -> [Segment]
        Returns a list of Particles and masks out labels for
        particles not meeting acceptance criteria.
    """
    pts = []
    centroid = 'Centroid' if intensity is None else 'WeightedCentroid'
    rprops = measure.regionprops(labels, ['Area', 'Eccentricity', centroid], intensity)
    for props in rprops:
        label = props['Label']
        if min_area > props['Area']:
            #print 'too small:',props['Area']
            pass
        elif props['Area'] > max_area:
            #print 'too big:',props['Area']
            pass
        elif props['Eccentricity'] > max_ecc:
            #print 'too eccentric:',props['Eccentricity']
            #labels[labels==label] = np.ma.masked
            pass
        else:
            x, y = props[centroid]
            pts.append(Segment(x, y, label, props['Eccentricity'], props['Area']))
    return pts

def find_particles(im, method='edge', **kwargs):
    """ find_particles(im, gaussian_size=3, **kwargs) -> [Segment],labels
        Find the particles in image im. The arguments in kwargs is
        passed to label_particles and filter_segments.

        Returns the list of found particles and the label image.
    """
    labels = None
    intensity = None
    #print "Seeking particles using", method
    if method == 'walker':
        labels = label_particles_walker(im, **kwargs)
    elif method == 'edge':
        labels = label_particles_edge(im, **kwargs)
    elif method == 'convolve':
        labels = label_particles_convolve(im, **kwargs)
        intensity = im
    else:
        raise RuntimeError('Undefined method "%s"' % method)

    segments = filter_segments(labels, intensity=intensity, **kwargs)
    return (segments, labels)

def find_particles_in_image(f, **kwargs):
    """ find_particles_in_image(im, **kwargs)
    """
    print "opening", f
    im = imread(f).astype(float)
    if f.lower().endswith('tif'):
        # clean pixel noise from phantom images
        im = median_filter(im, size=2)
    elif f.lower().endswith('jpg') and im.ndim == 3:
        # use just the green channel from color slr images
        im = im[...,1]
    im /= im.max()
    return find_particles(im, **kwargs)

def disk(n):
    return _disk(n).astype(int)

def gdisk(n, w=None):
    """ gdisk(n):
        return a gaussian kernel
    """
    if w is None:
        w = 2*n
    g = np.arange(-w, w+1)
    g = np.exp(-.5 * g**2 / n**2)
    g = np.outer(g, g)  # or g*g[...,None]
    g -= g.mean()
    g /= g.std()
    assert np.allclose(g.sum(),0), 'sum is nonzero: {}'.format(g.sum())
    return g

def remove_segments(orig, particles, labels):
    """ remove_segments(orig, particles, labels)
        attempts to remove the found big dot segment as found in original
    """
    return

def remove_disks(orig, particles, dsk=disk(6)):
    """ remove_disks(method=['disk' or 'segment'])
        removes a disk of given size centered at dot location
        inputs:
            orig   -    input image as ndarray or PIL Image
            particles - list of particles (namedtuple Segment)
            dsk   -    radius of disk, default is skimage.morphology.disk(r)
                            (size of square array is 2*r+1)
        output:
            the original image with big dots removed
    """
    disks = np.ones(orig.shape, int)
    if isinstance(particles[0], Segment):
        xys = zip(*(map(int,(p.x,p.y)) for p in particles))
    elif 'X' in particles.dtype.names:
        xys = np.round(particles['X']).astype(int),np.round(particles['Y']).astype(int)
    disks[xys] = 0
    disks = binary_erosion(disks,dsk)
    return orig*disks

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
    threshargs = {'max_ecc' :   .7 if args.slr else  .7, # .6
                  'min_area':  800 if args.slr else  15, # 870
                  'max_area': 1600 if args.slr else 200, # 1425
                  'csize'   :   22 if args.slr else  10}
    cthreshargs = {'max_ecc' :  .8 if args.slr else .8,
                   'min_area':  80 if args.slr else  3, # 92
                   'max_area': 200 if args.slr else 36, # 180
                   'csize'   :   5 if args.slr else  2}

    def f((n,filename)):
        pts, labels = find_particles_in_image(filename,
                            method='convolve', **threshargs)
        centers = np.hstack([n*np.ones((len(pts),1)), pts])
        print '%20s: Found %d particles' % ('', len(pts))
        if args.plot:
            pl.clf()
            pl.imshow(labels, cmap=cm)
            ptsarr = np.asarray(pts)
            pl.scatter(ptsarr[:,1], ptsarr[:,0], c='r')#ptsarr[:,2], cmap=cm)
            savename = path.join(pdir,path.split(filename)[-1].split('.')[-2])+'_POSITIONS.png'
            #print 'saving to',savename
            pl.savefig(savename, dpi=300)

        if args.corner:
            cpts, clabels = find_particles_in_image(filename,
                                    method='convolve', rmv=None, **cthreshargs)
            print '%20s: Found %d corners' % ('', len(cpts))
            if args.plot:
                #pl.imshow(clabels, cmap=cm)
                cpts = np.asarray(cpts)
                pl.scatter(cpts[:,1], cpts[:,0], c='w')#cpts[:,2], cmap=cm)
                #print 'saving corners to',savename
                pl.savefig(savename, dpi=300)

            corners = np.hstack([n*np.ones((len(cpts),1)), cpts])
            return centers, corners

        return centers

    filenames = sorted(args.files)
    if args.threads > 1:
        print "Multiprocessing with {} threads".format(args.threads)
        p = Pool(args.threads)
        points = p.map(f, enumerate(filenames))
    else:
        points = map(f, enumerate(filenames))

    if args.corner:
        points, corners = map(np.vstack, zip(*points))
        if 'POSITIONS' in args.output:
            coutput = args.output.replace('POSITIONS','CORNER_POSITIONS')
        else:
            coutput = ''.join(args.output.split('.')[:-1]+['_CORNER.']+args.output.split('.')[-1:])
        with open(coutput, 'w') as coutput:
            print "Saving corner positions to ", coutput
            coutput.write('# Frame    X           Y             Label  Eccen        Area\n')
            np.savetxt(coutput, corners, delimiter='     ',
                    fmt=['%6d', '%7.3f', '%7.3f', '%4d', '%1.3f', '%5d'])
    else:
        points = np.vstack(points)

    if 'CORNER_' in args.output:
        args.output = args.output.replace('CORNER_','')
    elif 'CORNER' in args.output:
        args.output = args.output.replace('CORNER','')
    with open(args.output, 'w') as output:
        print "Saving positions to ", args.output
        output.write('# Frame    X           Y             Label  Eccen        Area\n')
        np.savetxt(output, points, delimiter='     ',
                fmt=['%6d', '%7.3f', '%7.3f', '%4d', '%1.3f', '%5d'])
 

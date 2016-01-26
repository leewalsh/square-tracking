#!/usr/bin/env python
# encoding: utf-8

from __future__ import division

if __name__ == '__main__':
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('files', metavar='FILE', nargs='+',
                   help='Images to process')
    p.add_argument('-p', '--plot', action='count',
                   help="Produce a plot for each image. Use more p's for more images")
    p.add_argument('-v', '--verbose', action='count',
                   help="Control verbosity")
    p.add_argument('-o', '--output', default='POSITIONS.txt',
                   help='Output file')
    p.add_argument('-z', '--nozip', action='store_false', dest='gz',
                   help="Don't compress output files?")
    p.add_argument('-N', '--threads', default=-1, type=int,
                   help='Number of worker threads for parallel processing. '
                        'Uses all available cores if 0')
    p.add_argument('-s', '--select', action='store_true',
                   help='Open the first image and specify the circle of interest')
    p.add_argument('-b', '--both', action='store_true',
                   help='find both center and corner dots')
    p.add_argument('--slr', action='store_true',
                   help='Full resolution SLR was used')
    p.add_argument('-k', '--kern', default=0, type=float, required=True,
                   help='Kernel size for convolution')
    p.add_argument('--min', default=-1, type=int,
                   help='Minimum area')
    p.add_argument('--max', default=-1, type=int,
                   help='Maximum area')
    p.add_argument('--ecc', default=.8, type=float,
                   help='Maximum eccentricity')
    p.add_argument('-c', '--ckern', default=0, type=float,
                   help='Kernel size for convolution for corner dots')
    p.add_argument('--cmin', default=-1, type=int,
                   help='Minimum area for corner dots')
    p.add_argument('--cmax', default=-1, type=int,
                   help='Maximum area for corner dots')
    p.add_argument('--cecc', default=.8, type=float,
                        help='Maximum eccentricity for corner dots')
    args = p.parse_args()

from distutils.version import StrictVersion as version
import skimage
skversion = version(skimage.__version__)

import numpy as np
from scipy.ndimage import gaussian_filter, binary_erosion, convolve, center_of_mass, imread
if skversion < version('0.10'):
    from skimage.morphology import label as sklabel
    from skimage.measure import regionprops
else:
    from skimage.measure import regionprops, label as sklabel
from skimage.morphology import disk as _disk
from collections import namedtuple
from itertools import izip

import helpy

def label_particles_edge(im, sigma=2, closing_size=0, **extra_args):
    """ label_particles_edge(image, sigma=3, closing_size=3)
        Returns the labels for an image.
        Segments using Canny edge-finding filter.

        keyword arguments:
        image        -- The image in which to find particles
        sigma        -- The size of the Canny filter
        closing_size -- The size of the closing filter
    """
    from skimage.morphology import square, binary_closing, skeletonize
    if skversion < version('0.11'):
        from skimage.filter import canny
    else:
        from skimage.filters import canny
    edges = canny(im, sigma=sigma)
    if closing_size > 0:
        edges = binary_closing(edges, square(closing_size))
    edges = skeletonize(edges)
    labels = sklabel(edges)
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
    from skimage.segmentation import random_walker
    if sigma>0:
        im = gaussian_filter(im, sigma)
    labels = np.zeros_like(im)
    labels[im<min_thresh*im.max()] = 1
    labels[im>max_thresh*im.max()] = 2
    return random_walker(im, labels)

def label_particles_convolve(im, thresh=3, rmv=None, kern=0, **extra_args):
    """ label_particles_convolve(im, thresh=2)
        Returns the labels for an image
        Segments using a threshold after convolution with proper gaussian kernel
        Uses center of mass from original image to find centroid of segments

        Input:
            image   the original image
            thresh  the threshold above which pixels are included
                        if integer, in units of intensity std dev
                        if float, in absolute units of intensity
            rmv     if given, the positions at which to remove large dots
            kern   kernel size
    """
    # Michael removed disks post-convolution
    if rmv is not None:
        im = remove_disks(im, *rmv)
    if kern == 0:
        raise ValueError('kernel size `kern` not set')
    kernel = np.sign(kern)*gdisk(abs(kern))
    convolved = convolve(im, kernel)

    convolved -= convolved.min()
    convolved /= convolved.max()

    if isinstance(thresh, int):
        if rmv is not None:
            thresh -= 1 # smaller threshold for corners
        thresh = convolved.mean() + thresh*convolved.std()

    labels = sklabel(convolved > thresh)
    #print "found {} segments above thresh".format(labels.max())
    return labels, convolved

Segment = namedtuple('Segment', 'x y label ecc area'.split())

def filter_segments(labels, max_ecc, min_area, max_area, max_detect=None,
                    circ=None, intensity=None, **extra_args):
    """ filter_segments(labels, max_ecc=0.5, min_area=15, max_area=200) -> [Segment]
        Returns a list of Particles and masks out labels for
        particles not meeting acceptance criteria.
    """
    pts = []
    strengths = []
    centroid = 'Centroid' if intensity is None else 'WeightedCentroid'
    if skversion < version('0.10'):
        rprops = regionprops(labels, ['Area', 'Eccentricity', centroid], intensity)
    else:
        rprops = regionprops(labels, intensity)
    for rprop in rprops:
        area = rprop['area']
        if area < min_area or area > max_area:
            continue
        ecc = rprop['eccentricity']
        if ecc > max_ecc:
            continue
        x, y = rprop[centroid]
        if circ:
            co, ro = circ
            if (x - co[0])**2 + (y - co[1])**2 > ro**2:
                continue
        pts.append(Segment(x, y, rprop.label, ecc, area))
        if max_detect is not None:
            strengths.append(rprop['mean_intensity'])
    if max_detect is not None:
        pts = pts[np.argsort(-strengths)]
    return pts[:max_detect]

def prep_image(imfile):
    if args.verbose: print "opening", imfile
    im = imread(imfile).astype(float)
    if im.ndim == 3 and imfile.lower().endswith('jpg'):
        # use just the green channel from color slr images
        im = im[..., 1]

    # clip to two standard deviations about the mean
    # and normalize to [0, 1]
    s = 2*im.std()
    m = im.mean()
    im -= m - s
    im /= 2*s
    np.clip(im, 0, 1, out=im)
    return im

def find_particles(im, method, return_image=False, **kwargs):
    """ find_particles(im, method, **kwargs) -> [Segment],labels
        Find the particles in image im. The arguments in kwargs is
        passed to label_particles and filter_segments.

        Returns the list of found particles and the label image.
    """
    intensity = None
    if method == 'walker':
        labels = label_particles_walker(im, **kwargs)
    elif method == 'edge':
        labels = label_particles_edge(im, **kwargs)
    elif method == 'convolve':
        labels, convolved = label_particles_convolve(im, **kwargs)
        intensity = im if kwargs['kern'] > 0 else 1 - im
    else:
        raise RuntimeError('Undefined method "%s"' % method)

    pts = filter_segments(labels, intensity=intensity, **kwargs)
    return (pts, labels) + ((convolved,) if return_image else ())

def disk(n):
    return _disk(n).astype(int)

def gdisk(n, w=None):
    """ gdisk(n):
        return a gaussian kernel with zero integral and unity std dev.
    """
    if w is None:
        w = 2*n
    circ = ((np.indices([2*w+1, 2*w+1]) - w)**2).sum(0) <= (w+1)**2
    g = np.arange(-w, w+1, dtype=float)
    g = np.exp(-.5 * g**2 / n**2)
    g = np.outer(g, g)  # or g*g[...,None]
    g -= g[circ].mean()
    g /= g[circ].std()
    g[~circ] = 0
    assert np.allclose(g.sum(),0), 'sum is nonzero: {}'.format(g.sum())
    return g

def remove_segments(orig, particles, labels):
    """ remove_segments(orig, particles, labels)
        attempts to remove the found big dot segment as found in original
    """
    return

def remove_disks(orig, particles, dsk):
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
    if np.isscalar(dsk): dsk = disk(dsk)
    disks = np.ones(orig.shape, 'uint8')
    if isinstance(particles[0], Segment):
        xys = zip(*(map(int, (p.x, p.y)) for p in particles))
    elif 'X' in particles.dtype.names:
        xys = np.round(particles['X']).astype(int), np.round(particles['Y']).astype(int)
    disks[xys] = 0
    disks = binary_erosion(disks, dsk)
    return orig*disks

if __name__ == '__main__':
    if helpy.gethost()=='foppl':
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as pl
    from multiprocessing import Pool, cpu_count
    from os import path, makedirs
    import sys
    import shutil

    first = args.files[0]
    if '*' in first or '?' in first:
        from glob import glob
        filenames = sorted(glob(first))
        filepattern = first
        argv = 'argv'
    elif len(args.files) > 1:
        filenames = sorted(args.files)
        filepattern = reduce(helpy.str_union, args.files)
        i = sys.argv.index(first)
        argv = filter(lambda s: s not in filenames, sys.argv)
        argv.insert(i, filepattern)
        argv[0] = path.basename(argv[0])
        argv = ' '.join(argv)
    elif len(args.files)==1:
        filenames = sorted(args.files)

    if args.plot and len(filenames) > 10:
        args.plot = helpy.bool_input(
                    "Are you sure you want to make plots for all {} frames?"
                    " ".format(len(filenames)))

    suffix = '_POSITIONS'
    gz = args.gz
    output = args.output
    outdir = path.abspath(path.dirname(output))
    if not path.exists(outdir):
        print "Creating new directory", outdir
        makedirs(outdir)
    if output.endswith('.gz'):
        gz = 1
        output = output[:-3]
    if output.endswith('.txt'):
        output = output[:-4]
    if output.endswith(suffix):
        prefix = output[:-len(suffix)]
    else:
        prefix = output
        output += suffix
    outs = output, prefix + '_CORNER' + suffix

    helpy.save_log_entry(prefix, argv)
    helpy.save_meta(prefix, path_to_tiffs=path.abspath(filepattern))

    kern_area = np.pi*args.kern**2
    if args.min == -1:
        args.min = kern_area/2
        if args.verbose: print "using min =", args.min
    if args.max == -1:
        args.max = 2*kern_area
        if args.verbose: print "using max =", args.max

    thresh = {'center': {'max_ecc' : args.ecc,
                         'min_area': args.min,
                         'max_area': args.max,
                         'kern'   : args.kern}}

    if args.ckern:
        args.both = True
    if args.both:
        ckern_area = np.pi*args.ckern**2
        if args.cmin == -1: args.cmin = ckern_area/2
        if args.cmax == -1: args.cmax = 2*ckern_area
        thresh.update({'corner':
                        {'max_ecc' : args.cecc,
                         'min_area': args.cmin,
                         'max_area': args.cmax,
                         'kern'   : args.ckern}})
    dots = sorted(thresh)

    if args.select:
        co, ro = helpy.circle_click(filenames[0])

    def plot_positions(savebase, level, pts, labels, convolved=None,):
        cm = pl.cm.prism_r
        pl.clf()
        labels_mask = labels.astype(float)
        labels_mask[labels_mask==0] = np.nan
        pl.imshow(labels_mask, cmap=cm, interpolation='nearest')
        ax = pl.gca()
        xl, yl = ax.get_xlim(), ax.get_ylim()
        if level > 1:
            ptsarr = np.asarray(pts)
            pl.scatter(ptsarr[:,1], ptsarr[:,0], s=10, c='r')#ptsarr[:,2], cmap=cm)
            pl.xlim(xl); pl.ylim(yl)
        savename = savebase + '_POSITIONS.png'
        if args.verbose: print 'saving positions image to', savename
        pl.savefig(savename, dpi=300)
        if level > 2:
            pl.clf()
            pl.imshow(convolved, cmap='gray')
            if args.plot > 3:
                ptsarr = np.asarray(pts)
                pl.scatter(ptsarr[:,1], ptsarr[:,0], s=10, c='r')#ptsarr[:,2], cmap=cm)
                pl.xlim(xl); pl.ylim(yl)
            savename = savebase + '_CONVOLVED.png'
            if args.verbose: print 'saving positions with background to', savename
            pl.savefig(savename, dpi=300)

    def get_positions((n,filename)):
        circ = (co, ro) if args.select else None
        image = prep_image(filename)
        ret = []
        for dot in dots:
            rmv = None if dot=='center' else (pts, abs(args.kern))
            out = find_particles(image, method='convolve', circ=circ,
                    rmv=rmv, return_image=args.plot>2, **thresh[dot])
            pts = out[0]
            nfound = len(pts)
            if nfound:
                centers = np.hstack([np.full((nfound,1), n, 'f8'), pts])
            else:
                centers = np.empty((0, 6)) # 6 = id + len(Segment)
            if args.plot:
                savebase = path.join(outdir, path.splitext(path.basename(filename))[0])
                plot_positions(savebase+'_'+dot.upper(), args.plot, *out)
            ret += [centers]
        if not n % print_freq:
            print path.basename(filename).rjust(20), 'Found',\
              ', '.join(['{:3d} {}s'.format(len(r),d) for r,d in zip(ret,dots)])
        return ret if args.both else ret[0]

    print_freq = 1 if args.verbose else len(filenames)//100 + 1
    threads = args.threads
    if threads < 1:
        cpus = cpu_count()
        if threads==-1:
            threads = int(raw_input(
                "How many cpu threads to use? [{}] ".format(cpus)) or 0)
    threads = threads or cpus
    if threads > 1:
        print "Multiprocessing with {} threads".format(threads)
        p = Pool(threads)
        mapper = p.map
    else:
        mapper = map
    points = mapper(get_positions, enumerate(filenames))
    points = map(np.vstack, izip(*points)) if args.both else [np.vstack(points)]

    firstframe = prefix+'_'+path.basename(filenames[0])
    shutil.copy(filenames[0], firstframe)

    for dot, point, out in zip(dots, points, outs):
        txt = '.txt'+'.gz'*gz
        print "Saving {} positions to {}{{{},.npz}}".format(dot, out, txt)
        header = ('Kern {kern:.2f}, Min area {min_area:d}, '
          'Max area {max_area:d}, Max eccen {max_ecc:.2f}\n'
          'Frame    X           Y             Label  Eccen        Area').format
        np.savetxt(out+txt, point, delimiter='     ', header=header(**thresh[dot]),
                fmt=['%6d', '%7.3f', '%7.3f', '%4d', '%1.3f', '%5d'])
        helpy.txt_to_npz(out+txt, verbose=args.verbose, compress=gz)
        key_prefix = 'corner_' if dot=='corner' else ''
        helpy.save_meta(prefix, {key_prefix+k: v
                                 for k, v in thresh[dot].iteritems()})

#!/usr/bin/env python
# encoding: utf-8

from __future__ import division

if __name__ == '__main__':
    from argparse import ArgumentParser
    p = ArgumentParser()
    p.add_argument('files', metavar='FILE', nargs='+',
                   help='Images to process')
    p.add_argument('-p', '--plot', action='count',
                   help="Produce plots for each image. Two p's gives lots more")
    p.add_argument('-v', '--verbose', action='count',
                   help="Control verbosity")
    p.add_argument('-o', '--output', default='POSITIONS.txt',
                   help='Output file')
    p.add_argument('-z', '--nozip', action='store_false', dest='gz',
                   help="Don't compress output files?")
    p.add_argument('-N', '--threads', type=int,
                   help='Number of worker threads for parallel processing. '
                        'Uses all available cores if 0')
    p.add_argument('-s', '--select', action='store_true',
                   help='Open the first image and specify the circle of interest')
    p.add_argument('-b', '--both', action='store_true',
                   help='find both center and corner dots')
    p.add_argument('--remove', action='store_true',
                   help='Remove large-dot masks before small-dot convolution')
    p.add_argument('--thresh', type=float, default=3, help='Binary threshold '
                   'for defining segments, in units of standard deviation')
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
from scipy.ndimage import gaussian_filter, binary_dilation, convolve, imread
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
    if rmv is not None:
        im = remove_disks(im, *rmv)
    if kern == 0:
        raise ValueError('kernel size `kern` not set')
    kernel = np.sign(kern)*gdisk(abs(kern))
    convolved = convolve(im, kernel)
    if args.plot > 1:
        snapshot('kern', kernel)
        snapshot('convolved', convolved)

    convolved -= convolved.min()
    convolved /= convolved.max()

    if rmv is not None:
        thresh -= 1  # smaller threshold for corners
    threshed = convolved > convolved.mean() + thresh*convolved.std()
    labels = sklabel(threshed)
    if args.plot > 1:
        snapshot('threshed', threshed)
        snapshot('labeled', np.where(labels, labels, np.nan), cmap='prism_r')
    return labels, convolved

Segment = namedtuple('Segment', 'x y label ecc area'.split())

def filter_segments(labels, max_ecc, min_area, max_area, max_detect=None,
                    circ=None, intensity=None, **extra_args):
    """filter_segments(labels, max_ecc, min_area, max_area) -> [Segment]

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
    if args.plot > 1:
        snapshot('orig', im)
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
    if args.plot > 1:
        snapshot('clip', im)
    return im

def find_particles(im, method, **kwargs):
    """find_particles(im, method, **kwargs) -> [Segment], labels

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
    return pts, labels, convolved

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

def remove_disks(orig, particles, dsk, replace='sign', out=None):
    """removes a disk of given size centered at dot location
       inputs:
           orig:      input image as ndarray or PIL Image
           particles: list of particles (namedtuple Segment)
           dsk:       radius of disk, default is skimage.morphology.disk(r)
                          (size of square array is 2*r+1)
           replace:   value to replace disks with. Generally should be 0, 1,
                          0.5, the image mean, or 'mean' to calculate the mean,
                          or 'sign' to use 0 or 1 depending on the sign of `dsk`
           out:       array to save new value in (can be `orig` to do inplace)
       output:
           the original image with big dots replaced with `replace`
    """
    if np.isscalar(dsk):
        sign = np.sign(dsk)
        dsk = disk(abs(dsk))
    else:
        sign = 1
    if replace == 'mean':
        replace = orig.mean()
    elif replace == 'sign':
        replace = (1 - sign)/2

    if isinstance(particles[0], Segment):
        xys = zip(*(map(int, np.round((p.x, p.y))) for p in particles))
    elif 'X' in particles.dtype.names:
        xys = np.round(particles['X']).astype(int), np.round(particles['Y']).astype(int)
    disks = np.zeros(orig.shape, bool)
    disks[xys] = True
    disks = binary_dilation(disks, dsk)
    if out is None:
        out = orig.copy()
    out[disks] = replace
    if args.plot > 1:
        snapshot('disks', disks)
        snapshot('removed', out)
    return out

if __name__ == '__main__':
    if helpy.gethost()=='foppl':
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as pl
    from multiprocessing import Pool, cpu_count
    from os import path, makedirs
    import sys
    import shutil

    def snapshot(desc, im, **kwargs):
        global snapshot_num
        fname = '{}_snapshot{:03d}_{}.png'.format(prefix, snapshot_num, desc)
        pl.imsave(fname, im, **kwargs)
        snapshot_num += 1

    first = args.files[0]
    if len(args.files) > 1:
        filenames = sorted(args.files)
        filepattern = reduce(helpy.str_union, args.files)
        i = sys.argv.index(first)
        argv = filter(lambda s: s not in filenames, sys.argv)
        argv.insert(i, filepattern)
        argv[0] = path.basename(argv[0])
        argv = ' '.join(argv)
    else:
        from glob import glob
        filenames = sorted(glob(first))
        filepattern = first
        argv = 'argv'

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

    sizes = {'center': {'max_ecc': args.ecc,
                        'min_area': args.min,
                        'max_area': args.max,
                        'kern': args.kern}}

    if args.ckern:
        args.both = True
    if args.both:
        ckern_area = np.pi*args.ckern**2
        if args.cmin == -1: args.cmin = ckern_area/2
        if args.cmax == -1: args.cmax = 2*ckern_area
        sizes.update({'corner': {'max_ecc': args.cecc,
                                 'min_area': args.cmin,
                                 'max_area': args.cmax,
                                 'kern': args.ckern}})
    dots = sorted(sizes)

    if args.select:
        co, ro = helpy.circle_click(filenames[0])

    def plot_scatter(pts, ax, s=10, c='r', cm=None):
        xl, yl = ax.get_xlim(), ax.get_ylim()
        ptsarr = np.asarray(pts)
        if c == 'id':
            c = ptsarr[:, 2]
        pl.scatter(ptsarr[:,1], ptsarr[:,0], s=abs(s), c=c, cmap=cm)
        pl.xlim(xl)
        pl.ylim(yl)

    def plot_positions(save, pts, labels, convolved=None, **pltargs):
        cm = 'prism_r'
        # dpi = 300 gives 2.675 pixels for each image pixel, or 112.14 real
        # pixels per inch. This may be unreliable, but assume that many image
        # pixels per inch, and use integer multiples of that for dpi
        # PPI = 112.14 if figsize (8, 6)
        PPI = 84.638  # if figsize (8, 8)
        dpi = 4*PPI

        labels_mask = labels.astype(float)
        labels_mask[labels_mask==0] = np.nan
        pl.clf()
        pl.imshow(labels_mask, cmap=cm, interpolation='nearest')
        ax = pl.gca()
        plot_scatter(pts, ax, **pltargs)
        savename = save + '_SEGMENTS.png'
        if args.verbose: print 'saving positions image to', savename
        pl.savefig(savename, dpi=dpi)
        pl.clf()
        pl.imshow(convolved, cmap='gray')
        ax = pl.gca()
        plot_scatter(pts, ax, **pltargs)
        savename = save + '_CONVOLVED.png'
        if args.verbose:
            print 'saving positions with background to', savename
        pl.savefig(savename, dpi=dpi)

    def get_positions((n, filename)):
        global snapshot_num
        snapshot_num = 100*n
        circ = (co, ro) if args.select else None
        image = prep_image(filename)
        ret = []
        for dot in dots:
            if args.remove and dot == 'corner':
                rmv = segments, args.kern
            else:
                rmv = None
            out = find_particles(image, method='convolve', circ=circ, rmv=rmv,
                                 thresh=args.thresh, **sizes[dot])
            segments = out[0]
            nfound = len(segments)
            if nfound:
                centers = np.hstack([np.full((nfound, 1), n, 'f8'), segments])
            else:  # empty line of length 6 = id + len(Segment)
                centers = np.empty((0, 6))
            if args.plot:
                save = '_'.join([prefix,
                                 path.splitext(path.basename(filename))[0],
                                 dot.upper()])
                plot_positions(save, *out, s=sizes[dot]['kern'])
            ret.append(centers)
        if not n % print_freq:
            print path.basename(filename).rjust(20), 'Found',\
              ', '.join(['{:3d} {}s'.format(len(r),d) for r,d in zip(ret,dots)])
        return ret if args.both else ret[0]

    print_freq = 1 if args.verbose else len(filenames)//100 + 1
    threads = args.threads
    if threads < 1:
        cpus = cpu_count()
        if threads is None:
            print "How many cpu threads to use? [{}] ".format(cpus),
            threads = int(raw_input() or cpus)
    threads = bool(args.plot) or threads or cpus
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

    fig, axes = pl.subplots(nrows=len(dots), ncols=2, sharey='row')
    for dot, point, out, axis in zip(dots, points, outs, np.atleast_2d(axes)):
        eax, aax = axis
        eax.hist(point[:, 4], bins=20, range=(0,1), alpha=0.5, color='r', label=dot+' eccen')
        eax.set_xlim(0, 1)
        eax.axvline(sizes[dot]['max_ecc'], 0, 0.5, c='r', lw=2)
        eax.legend(loc='best')
        aax.hist(point[:, 5], bins=20, alpha=0.5, color='g', label=dot+' area')
        aax.axvline(sizes[dot]['min_area'], c='g', lw=2)
        aax.set_xlim(0, sizes[dot]['max_area'])
        aax.legend(loc='best')
        fig.savefig(prefix+'_SEGMENTS.pdf')
        txt = '.txt'+'.gz'*gz
        print "Saving {} positions to {}{{{},.npz}}".format(dot, out, txt)
        hfmt = ('Kern {kern:.2f}, Min area {min_area:d}, '
          'Max area {max_area:d}, Max eccen {max_ecc:.2f}\n'
          'Frame    X           Y             Label  Eccen        Area').format
        np.savetxt(out+txt, point, delimiter='     ', header=hfmt(**sizes[dot]),
                fmt=['%6d', '%7.3f', '%7.3f', '%4d', '%1.3f', '%5d'])
        helpy.txt_to_npz(out+txt, verbose=args.verbose, compress=gz)
        key_prefix = 'corner_' if dot=='corner' else ''
        helpy.save_meta(prefix, {key_prefix+k: v
                                 for k, v in sizes[dot].iteritems()})

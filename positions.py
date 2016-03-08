#!/usr/bin/env python
# encoding: utf-8

from __future__ import division

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    arg = parser.add_argument
    arg('files', metavar='FILE', nargs='+', help='Images to process')
    arg('-p', '--plot', action='count',
        help="Produce plots for each image. Two p's gives lots more")
    arg('-v', '--verbose', action='count', help="Control verbosity")
    arg('-o', '--output', help='Output filename prefix.')
    arg('-z', '--nozip', action='store_false', dest='gz', help="Don't compress")
    arg('--nosave', action='store_false', dest='save', help="Don't save output")
    arg('-N', '--threads', type=int, help='Number of worker threads for '
        'parallel processing. N=0 uses all available cores')
    arg('--boundary', type=float, nargs='*', metavar='X0 Y0 R',
        help='Specify system boundary, or open image to select it')
    arg('-b', '--both', action='store_true', help='find center and corner dots')
    arg('--remove', action='store_true',
        help='Remove large-dot masks before small-dot convolution')
    arg('--thresh', type=float, default=3, help='Binary threshold '
        'for defining segments, in units of standard deviation')
    arg('-k', '--kern', type=float, required=True,
        help='Kernel size for convolution')
    arg('--min', type=int, help='Minimum area')
    arg('--max', type=int, help='Maximum area')
    arg('--ecc', default=.8, type=float, help='Maximum eccentricity')
    arg('-c', '--ckern', type=float, help='Kernel for corner dots')
    arg('--cmin', type=int, help='Min area for corners')
    arg('--cmax', type=int, help='Max area for corners')
    arg('--cecc', default=.8, type=float, help='Max ecc for corners')
    args = parser.parse_args()

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
    kernel = np.sign(kern)*gdisk(abs(kern)/2, abs(kern))
    convolved = convolve(im, kernel)
    if args.plot > 1:
        snapshot('kern', kernel)
        snapshot('convolved', convolved)

    convolved -= convolved.min()
    convolved /= convolved.max()

    if rmv is not None:
        thresh -= 1  # smaller threshold for corners
    threshed = convolved > convolved.mean() + thresh*convolved.std()
    labels = sklabel(threshed, connectivity=1)
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
            xo, yo, ro = circ
            if (x - xo)**2 + (y - yo)**2 > ro**2:
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

def gdisk(width, inner=0, outer=None):
    """return a gaussian kernel with zero integral and unity std dev.

    parameters:
        width:  width (standard dev) of gaussian (approx half-width at half-max)
        inner:  inner radius of constant disk, before gaussian falloff
                default is 0
        outer:  outer radius of nonzero part (outside of this, gdisk = 0)
                default is inner + 2*width

    returns:
        gdisk:  a square array with values given by
                        / max for r <= inner
                g(r) = {  min + (max-min)*exp(.5*(r-inner)**2 / width**2)
                        \ 0 for r > outer
                where min and max are set so that the sum and std of the array
                are 0 and 1 respectively
    """
    outer = outer or inner + 2*width
    circ = disk(outer)
    incirc = circ.nonzero()

    x = np.arange(-outer, outer+1, dtype=float)
    x, y = np.meshgrid(x, x)
    r = x**2 + y**2 - inner**2
    np.clip(r, 0, None, r)

    g = np.exp(-0.5*r/width**2)
    g -= g[incirc].mean()
    g /= g[incirc].std()
    g *= circ
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
        first = filenames[0]
        argv = 'argv'
    if args.plot:
        if len(filenames) > 10:
            print "Are you sure you want to make plots for all",
            print len(filenames), "frames?"
            args.plot *= helpy.bool_input()
        if args.plot > 1 and (not args.save or len(filenames) > 2):
            print "Do you want to display all the snapshots without saving?"
            args.plot -= 1 - helpy.bool_input()

    suffix = '_POSITIONS'
    output = args.output
    outdir = path.abspath(path.dirname(output))
    if not path.exists(outdir):
        print "Creating new directory", outdir
        makedirs(outdir)
    if output.endswith('.gz'):
        args.gz = 1
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
    meta = helpy.load_meta(prefix)

    imdir = prefix + '_detection'
    if not path.isdir(imdir):
        makedirs(imdir)

    kern_area = np.pi*args.kern**2
    sizes = {'center': {'max_ecc': args.ecc,
                        'min_area': args.min or kern_area/2,
                        'max_area': args.max or kern_area*2,
                        'kern': args.kern}}
    if args.ckern:
        args.both = True
    if args.both:
        ckern_area = np.pi*args.ckern**2
        sizes.update({'corner': {'max_ecc': args.cecc,
                                 'min_area': args.cmin or ckern_area/2,
                                 'max_area': args.cmax or ckern_area*2,
                                 'kern': args.ckern}})
    dots = sorted(sizes)
    meta.update({dot + '_' + k: v
                 for dot in dots for k, v in sizes[dot].iteritems()})

    if args.boundary is not None:
        args.boundary = args.boundary or helpy.circle_click(first)
        meta.update(boundary=args.boundary)

    def snapshot(desc, im, **kwargs):
        global snapshot_num, imprefix
        if args.save:
            fname = '{}_{:02d}_{}.png'.format(imprefix, snapshot_num, desc)
            pl.imsave(fname, im, **kwargs)
        else:
            fig, ax = pl.subplots()
            ax.imshow(im, title=path.basename(imprefix)+'_'+desc, **kwargs)
        snapshot_num += 1

    def plot_points(pts, img, name='', s=10, c='r', cm=None,
                    vmin=None, vmax=None, interp=None, cbar=False):
        global snapshot_num, imprefix
        fig, ax = pl.subplots(figsize=(8+2*cbar, 8))
        # dpi = 300 gives 2.675 pixels for each image pixel, or 112.14 real
        # pixels per inch. This may be unreliable, but assume that many image
        # pixels per inch, and use integer multiples of that for dpi
        # PPI = 112.14 if figsize (8, 6)
        PPI = 84.638  # if figsize (8, 8)
        dpi = 4*PPI
        axim = ax.imshow(img, cmap=cm, vmin=vmin, vmax=vmax,
                         interpolation=interp)
        if cbar:
            fig.colorbar(axim)
        xl, yl = ax.get_xlim(), ax.get_ylim()
        s = abs(s)
        helpy.draw_circles(helpy.consecutive_fields_view(pts, 'xy')[:, ::-1], s,
                           ax, lw=max(s/10, .5), color=c, fill=False, zorder=2)
        if s > 3:
            ax.scatter(pts['y'], pts['x'], s, c, '+')
        ax.set_xlim(xl)
        ax.set_ylim(yl)

        if args.save:
            savename = '{}_{:02d}_{}.png'.format(imprefix, snapshot_num, name)
            fig.savefig(savename, dpi=dpi)
            snapshot_num += 1
            pl.close(fig)

    def plot_positions(segments, labels, convolved=None, **pltargs):
        Segment_dtype = np.dtype({'names': Segment._fields,
                                  'formats': [float, float, int, float, float]})
        pts = np.asarray(segments, dtype=Segment_dtype)
        pts_by_label = np.zeros(labels.max()+1, dtype=Segment_dtype)
        pts_by_label[0] = (np.nan, np.nan, 0, np.nan, np.nan)
        pts_by_label[pts['label']] = pts

        plot_points(pts, convolved, c='r', cm='gray',
                    name='CONVOLVED', **pltargs)

        labels_mask = np.where(labels, labels, np.nan)
        plot_points(pts, labels_mask, c='k', cm='prism_r', interp='nearest',
                    name='SEGMENTS', **pltargs)

        ecc_map = labels_mask*0
        ecc_map.flat = pts_by_label[labels.flat]['ecc']
        plot_points(pts, ecc_map, c='k', vmin=0, vmax=1, interp='nearest',
                    cbar=True, name='ECCEN', **pltargs)

        area_map = labels_mask*0
        area_map.flat = pts_by_label[labels.flat]['area']
        plot_points(pts, area_map, c='k', cbar=True, interp='nearest',
                    name='AREA', **pltargs)

    def get_positions((n, filename)):
        global snapshot_num, imprefix
        snapshot_num = 0
        filebase = path.splitext(path.basename(filename))[0]
        imbase = path.join(imdir, filebase)
        imprefix = imbase
        image = prep_image(filename)
        ret = []
        for dot in dots:
            imprefix = '_'.join([imbase, dot.upper()])
            snapshot_num = 0
            if args.remove and dot == 'corner':
                rmv = segments, args.kern
            else:
                rmv = None
            out = find_particles(image, method='convolve', circ=args.boundary,
                                 rmv=rmv, thresh=args.thresh, **sizes[dot])
            segments = out[0]
            nfound = len(segments)
            if nfound:
                centers = np.hstack([np.full((nfound, 1), n, 'f8'), segments])
            else:  # empty line of length 6 = id + len(Segment)
                centers = np.empty((0, 6))
            if args.plot:
                plot_positions(*out, s=sizes[dot]['kern'])
            ret.append(centers)
        if not n % print_freq:
            fmt = '{:3d} {}s'.format
            print path.basename(filename).rjust(20), 'Found',
            print ', '.join([fmt(len(r), d) for r, d in zip(ret, dots)])
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

    fig, axes = pl.subplots(nrows=len(dots), ncols=2, sharey='row')
    if args.save:
        savenotice = "Saving {} positions to {}{{{},.npz}}".format
        hfmt = ('Kern {kern:.2f}, Min area {min_area:d}, '
          'Max area {max_area:d}, Max eccen {max_ecc:.2f}\n'
          'Frame    X           Y             Label  Eccen        Area').format
        txtfmt = ['%6d', '%7.3f', '%7.3f', '%4d', '%1.3f', '%5d']
        ext = '.txt'+'.gz'*args.gz

    for dot, point, out, axis in zip(dots, points, outs, np.atleast_2d(axes)):
        size = sizes[dot]
        eax, aax = axis
        label = "{} eccen (max {})".format(dot, size['max_ecc'])
        eax.hist(point[:, 4], bins=40, range=(0, 1),
                 alpha=0.5, color='r', label=label)
        eax.axvline(size['max_ecc'], 0, 0.5, c='r', lw=2)
        eax.set_xlim(0, 1)
        eax.set_xticks(np.arange(0, 1.1, .1))
        eax.legend(loc='best', fontsize='small')

        areas = point[:, 5]
        amin, amax = size['min_area'], size['max_area']
        s = 1 + (amax-amin) // 40
        bins = np.arange(amin, amax+s, s)
        label = "{} area ({} - {})".format(dot, amin, amax)
        aax.hist(areas, bins, alpha=0.5, color='g', label=label)
        aax.axvline(size['min_area'], c='g', lw=2)
        aax.set_xlim(0, bins[-1])
        aax.set_xticks(bins[::1+len(bins)//10])
        aax.legend(loc='best', fontsize='small')
        if args.save:
            print savenotice(dot, out, ext)
            np.savetxt(out+ext, point, header=hfmt(**size),
                       delimiter='     ', fmt=txtfmt)
            helpy.txt_to_npz(out+ext, verbose=args.verbose, compress=args.gz)
    if args.save:
        from shutil import copy
        copy(first, prefix+'_'+path.basename(first))
        fig.savefig(prefix+'_SEGMENTSTATS.pdf')
        helpy.save_meta(prefix, meta, path_to_tiffs=path.abspath(filepattern),
                        detect_thresh=args.thresh, detect_removed=args.remove)
    else:
        pl.show()

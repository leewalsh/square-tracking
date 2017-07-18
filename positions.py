#!/usr/bin/env python
# encoding: utf-8
"""Detect the positions in images of marked granular particles and save them to
be tracked. Primary input is a sequence (or single) image file, and output is a
formatted list of the positions of the centers and orientation marks of the
particles.

Copyright (c) 2012--2017 Lee Walsh, Department of Physics, University of
Massachusetts; all rights reserved.
"""

from __future__ import division

from collections import namedtuple
from itertools import izip

import numpy as np
from numpy.lib.function_base import _hist_bin_auto as hist_bin_auto
from scipy import ndimage

# skimage (scikit-image) changed the location, names, and api of several
# functions at versions 0.10 and 0.11 (at leaste), but they still have
# effectively the same functionality. to run with old versions of skimage (from
# enthought or conda), we must check for the version and import them from the
# proper module and use them with the appropriate api syntax:
from distutils.version import StrictVersion
import skimage
skimage_version = StrictVersion(skimage.__version__)
from skimage.morphology import disk as skdisk
if skimage_version < StrictVersion('0.10'):
    from skimage.morphology import label as sklabel
    from skimage.measure import regionprops
else:
    from skimage.measure import regionprops, label as sklabel

import helpy


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description=__doc__)
    arg = parser.add_argument
    arg('files', metavar='FILE', nargs='+', help='Images to process')
    arg('-i', '--slice', nargs='?', const=True, help='Slice to limit frames')
    arg('-o', '--output', help='Output filename prefix.')
    arg('-p', '--plot', action='count', default=1,
        help="Produce plots for each image. Two p's gives lots more")
    arg('-v', '--verbose', action='count', help="Control verbosity")
    arg('-z', '--nozip', action='store_false', dest='gz', help="Don't compress")
    arg('--nosave', action='store_false', dest='save', help="Don't save output")
    arg('--noplot', action='store_true', help="Don't depend on matplotlib")
    arg('-N', '--threads', type=int, help='Number of worker threads for '
        'parallel processing. N=0 uses all available cores')
    arg('--boundary', type=float, nargs='*', metavar='X0 Y0 R',
        help='Specify system boundary, or open image to select it')
    arg('-b', '--both', action='store_true', help='find center and corner dots')
    arg('--remove', action='store_true',
        help='Remove large-dot masks before small-dot convolution')
    arg('--thresh', type=float, default=3, help='Binary threshold '
        'for defining segments, in units of standard deviation')
    arg('--cthresh', type=float, default=2, help='Binary threshold '
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


def label_particles_edge(im, sigma=2, closing_size=0, **extra_args):
    """ Segment image using Canny edge-finding filter.

        parameters
        ----------
        im : image in which to find particles
        sigma : size of the Canny filter
        closing_size : size of the closing filter

        returns
        -------
        labels : an image array of uniquely labeled segments
    """
    from skimage.morphology import square, binary_closing, skeletonize
    if skimage_version < StrictVersion('0.11'):
        from skimage.filter import canny
    else:
        from skimage.filters import canny
    edges = canny(im, sigma=sigma)
    if closing_size > 0:
        edges = binary_closing(edges, square(closing_size))
    edges = skeletonize(edges)
    labels = sklabel(edges)
    print "found {} segments".format(labels.max())
    # in ma.array mask, False is True, and vice versa
    labels = np.ma.array(labels, mask=edges == 0)
    return labels


def label_particles_walker(im, min_thresh=0.3, max_thresh=0.5, sigma=3):
    """ Segment image using random_walker method.

        parameters
        ----------
        image : image in which to find particles
        min_thresh : lower limit for binary threshold
        max_thresh : upper limit for binary threshold

        returns
        -------
        labels : an image array of uniquely labeled segments
    """
    from skimage.segmentation import random_walker
    if sigma > 0:
        im = ndimage.gaussian_filter(im, sigma)
    labels = np.zeros_like(im)
    labels[im < min_thresh*im.max()] = 1
    labels[im > max_thresh*im.max()] = 2
    return random_walker(im, labels)


def label_particles_convolve(im, kern, thresh=3, rmv=None, **extra_args):
    """ Segment image using convolution with gaussian kernel and threshold

        parameters
        ----------
        im : the original image to be labeled
        kern : kernel size
        thresh : the threshold above which pixels are included
            if thresh >= 1, in units of intensity std dev
            if thresh < 1, in absolute units of intensity
        rmv : if given, the positions at which to remove large dots

        returns
        -------
        labels : an image array of uniquely labeled segments
        convolved : the convolved image before thresholding and segementation
    """
    if rmv is not None:
        im = remove_disks(im, *rmv)
    kernel = np.sign(kern)*gdisk(abs(kern)/4, abs(kern))
    convolved = ndimage.convolve(im, kernel)
    convolved -= convolved.min()
    convolved /= convolved.max()

    if args.plot > 2:
        snapshot('kern', kernel)
        snapshot('convolved', convolved, cmap='gray')

    if thresh >= 1:
        if rmv is not None:
            thresh -= 1  # smaller threshold for corners
        thresh = thresh*convolved.std() + convolved.mean()
    threshed = convolved > thresh

    labels = sklabel(threshed, connectivity=1)

    if args.plot > 2:
        snapshot('threshed', threshed)
        snapshot('labeled', np.where(labels, labels, np.nan), cmap='prism_r')

    return labels, convolved

Segment = namedtuple('Segment', 'x y label ecc area'.split())


def filter_segments(labels, max_ecc, min_area, max_area, keep=False,
                    circ=None, intensity=None, **extra_args):
    """ filter out non-particle segments of an image based on shape criteria

        parameters
        ----------
        labels : an image array of uniquely labeled segments
        max_ecc : upper limit for segment eccentricity
        min_area : lower limit for segment size in pixels
        max_area : upper limit for segment size in pixels
        keep : whether to keep bad segments as well and return a mask
        circ : tuple of (x0, y0, r). accept only segments centered within a
            distrance r from center x0, y0.
        intensity : an image array of the same shape as `labels` used as the
            weighting to determine the centroid of each segment.

        returns
        -------
        pts : list of `Segment`s that meet criteria (or all of them if `keep`)
        pts_mask : if `keep`, also return a boolean array matching `pts` that is
            True where the segment meets acceptance criteria and False where not
    """
    pts = []
    pts_mask = []
    centroid = 'Centroid' if intensity is None else 'WeightedCentroid'
    if skimage_version < StrictVersion('0.10'):
        rpropargs = labels, ['Area', 'Eccentricity', centroid], intensity
    else:
        rpropargs = labels, intensity
    for rprop in regionprops(*rpropargs):
        area = rprop['area']
        good = min_area <= area <= max_area
        if not (good or keep):
            continue
        ecc = rprop['eccentricity']
        good &= ecc <= max_ecc
        if not (good or keep):
            continue
        x, y = rprop[centroid]
        if circ:
            xo, yo, ro = circ
            if (x - xo)**2 + (y - yo)**2 > ro**2:
                continue
        pts.append(Segment(x, y, rprop.label, ecc, area))
        if keep:
            pts_mask.append(good)
    if keep:
        return pts, np.array(pts_mask)
    return pts


def prep_image(imfile, width=2):
    """ Open an image from file, clip, normalize it, and return it as an array.

        parameters
        ----------
        imfile : a file or filename
        width : factor times std deviation to clip about mean

        returns
        -------
        im : 2-d image array as float, normalized to [0, 1]
    """
    if args.verbose:
        print "opening", imfile
    im = ndimage.imread(imfile).astype(float)
    if args.plot > 2:
        snapshot('orig', im, cmap='gray')
    if im.ndim == 3 and imfile.lower().endswith('jpg'):
        # use just the green channel from color slr images
        im = im[..., 1]

    # clip to `width` times the standard deviation about the mean
    # and normalize to [0, 1]
    s = width*im.std()
    m = im.mean()
    im -= m - s
    im /= 2*s
    np.clip(im, 0, 1, out=im)
    if args.plot > 2:
        snapshot('clip', im, cmap='gray')
    return im


def find_particles(im, method, **kwargs):
    """ Find the particles in an image via a certain method.

        parameters
        ----------
        im : image array, assumed normalized to [0, 1]
        method : one of 'walker', 'edge', or 'convolve'. The method to use to
            identify candidate particles before filtering.
        kwargs : arguments passed to label_particles and filter_segments.

        returns
        -------
        pts : list of `Segments` determined to be particles
        labels : an image with the same shape as im, with segments uniquely
            labled by sequential integers
        convolved : an image with same shape as im,
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
        raise ValueError('Undefined method "%s"' % method)

    keep = args.plot > 1
    pts = filter_segments(labels, intensity=intensity, keep=keep, **kwargs)
    return pts, labels, convolved


def disk(n):
    """create a binary array with a disk of size `n`"""
    return skdisk(n).astype(int)


def gdisk(width, inner=0, outer=None):
    """ create a gaussian kernel with constant central disk, zero sum, std dev 1

        shape is a disk of constant value and radius `inner`, which falls off as
        a gaussian with `width`, and is truncated at radius `outer`.

        parameters
        ----------
        width : width (standard dev) of gaussian (approx half-width at half-max)
        inner : radius of constant disk, before gaussian falloff (default 0)
        outer : full radius of nonzero part, beyond which array is truncated
            (default outer = inner + 2*width)

        returns
        -------
        gdisk:  a square array with values given by
                    / max for r <= inner
            g(r) = {  min + (max-min)*exp(.5*(r-inner)**2 / width**2)
                    \ 0 for r > outer
            min and max are set so that the sum of the array is 0 and std is 1
    """
    outer = outer or inner + 4*width
    circ = disk(outer)
    incirc = circ.nonzero()

    x = np.arange(-outer, outer+1, dtype=float)
    x, y = np.meshgrid(x, x)
    r = np.hypot(x, y) - inner
    np.clip(r, 0, None, r)

    g = np.exp(-0.5*(r/width)**2)
    g -= g[incirc].mean()
    g /= g[incirc].std()
    g *= circ
    return g


def remove_segments(orig, particles, labels):
    """remove the found big dot segment as found in original"""
    return


def remove_disks(orig, particles, removal_mask, replace='sign', out=None):
    """ remove a patch of given shape centered at each dot location

        parameters
        ----------
        orig : input image as ndarray or PIL Image
        particles : list of particles (namedtuple Segment)
        removal_mask : shape or size of patch to remove at site of each particle
            given as either a mask array that defines the patch
            or a scalar (int or float) that gives radius to create a disk using
            disk(r), which is a square array of size 2*r+1
        replace : value to replace disks with. Generally should be one of:
            - a float between 0 and 1 such as 0, 0.5, 1, or the image mean
            - 'mean', to calculate the mean
            - 'sign', to use 0 or 1 depending on the sign of `removal_mask`
        out : array to save new value in (can be `orig` to do in-place)

        returns
        -------
        out : the original image with big dots replaced with `replace`
    """
    if np.isscalar(removal_mask):
        sign = np.sign(removal_mask)
        removal_mask = disk(abs(removal_mask))
    else:
        sign = 1
    if replace == 'mean':
        replace = orig.mean()
    elif replace == 'sign':
        replace = (1 - sign)/2

    if isinstance(particles[0], Segment):
        xys = zip(*(map(int, np.round((p.x, p.y))) for p in particles))
    elif 'X' in particles.dtype.names:
        xys = tuple([np.round(particles[x]).astype(int) for x in 'XY'])
    disks = np.zeros(orig.shape, bool)
    disks[xys] = True
    disks = ndimage.binary_dilation(disks, removal_mask)
    if out is None:
        out = orig.copy()
    out[disks] = replace
    if args.plot > 2:
        snapshot('disks', disks)
        snapshot('removed', out)
    return out

if __name__ == '__main__':
    import os
    import sys

    if args.noplot:
        args.plot = 0
    if args.plot:
        if helpy.gethost() == 'foppl':
            import matplotlib
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    from multiprocessing import Pool, cpu_count

    first = args.files[0]
    if len(args.files) > 1:
        filenames = sorted(args.files)
        filepattern = helpy.str_union(args.files)
        i = sys.argv.index(first)
        argv = filter(lambda s: s not in filenames, sys.argv)
        argv.insert(i, filepattern)
        argv[0] = os.path.basename(argv[0])
        argv = ' '.join(argv)
    else:
        from glob import glob
        if first.endswith(']'):
            first, _, args.slice = first[:-1].rpartition('[')
        filenames = sorted(glob(first))
        if args.slice:
            args.slice = helpy.parse_slice(args.slice, len(filenames))
            filenames = filenames[args.slice]
        filepattern = first
        first = filenames[0]
        argv = 'argv'
    if args.plot > 1:
        if len(filenames) > 10:
            print "Are you sure you want to make plots for all",
            print len(filenames), "frames?",
            args.plot -= not helpy.bool_input()
        if args.plot > 2 and (not args.save or len(filenames) > 2):
            print "Do you want to display all the snapshots without saving?",
            args.plot -= not helpy.bool_input()

    suffix = '_POSITIONS'
    outdir = os.path.abspath(os.path.dirname(args.output))
    if not os.path.exists(outdir):
        print "Creating new directory", outdir
        os.makedirs(outdir)
    if args.output.endswith('.gz'):
        args.gz = 1
        args.output = args.output[:-3]
    if args.output.endswith('.txt'):
        args.output = args.output[:-4]
    if args.output.endswith(suffix):
        prefix = args.output[:-len(suffix)]
    else:
        prefix = args.output
        args.output += suffix
    outputs = args.output, prefix + '_CORNER' + suffix

    helpy.save_log_entry(prefix, argv)
    meta = helpy.load_meta(prefix)

    imdir = prefix + '_detection'
    if not os.path.isdir(imdir):
        os.makedirs(imdir)

    kern_area = np.pi*args.kern**2
    sizes = {'center': {'max_ecc': args.ecc,
                        'min_area': args.min or int(kern_area//2),
                        'max_area': args.max or int(kern_area*2 + 1),
                        'kern': args.kern,
                        'thresh': args.thresh}}
    if args.ckern:
        args.both = True
    if args.both:
        ckern_area = np.pi*args.ckern**2
        sizes.update({'corner': {'max_ecc': args.cecc,
                                 'min_area': args.cmin or int(ckern_area//2),
                                 'max_area': args.cmax or int(ckern_area*2 + 1),
                                 'kern': args.ckern,
                                 'thresh': args.cthresh}})
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
            plt.imsave(fname, im, **kwargs)
        else:
            fig, ax = plt.subplots()
            ax.imshow(im, title=os.path.basename(imprefix)+'_'+desc, **kwargs)
        snapshot_num += 1

    def plot_points(pts, img, name='', s=10, c='r', cmap=None,
                    vmin=None, vmax=None, cbar=False):
        global snapshot_num, imprefix
        fig, ax = plt.subplots(figsize=(8, 8))
        # dpi = 300 gives 2.675 pixels for each image pixel, or 112.14 real
        # pixels per inch. This may be unreliable, but assume that many image
        # pixels per inch, and use integer multiples of that for dpi
        # PPI = 112.14 if figsize (8, 6)
        PPI = 84.638  # if figsize (8, 8)
        dpi = 4*PPI
        axim = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax,
                         interpolation='nearest')
        if cbar:
            fig.tight_layout()
            cb_height = 4
            cax = fig.add_axes(np.array([10, 99-cb_height, 80, cb_height])/100)
            fig.colorbar(axim, cax=cax, orientation='horizontal')
        xl, yl = ax.get_xlim(), ax.get_ylim()
        s = abs(s)
        helpy.draw_circles(helpy.consecutive_fields_view(pts, 'xy')[:, ::-1], s,
                           ax, lw=max(s/10, .5), color=c, fill=False, zorder=2)
        if s > 3:
            ax.scatter(pts['y'], pts['x'], s, c, '+')
        ax.set_xlim(xl)
        ax.set_ylim(yl)
        ax.set_xticks([])
        ax.set_yticks([])

        if args.save:
            savename = '{}_{:02d}_{}.png'.format(imprefix, snapshot_num, name)
            fig.savefig(savename, dpi=dpi, bbox_inches='tight', pad_inches=0)
            snapshot_num += 1
            plt.close(fig)

    def plot_positions(segments, labels, convolved=None, **kwargs):
        Segment_dtype = np.dtype({'names': Segment._fields,
                                  'formats': [float, float, int, float, float]})
        pts = np.asarray(segments[0], dtype=Segment_dtype)
        pts_by_label = np.zeros(labels.max()+1, dtype=Segment_dtype)
        pts_by_label[0] = (np.nan, np.nan, 0, np.nan, np.nan)
        pts_by_label[pts['label']] = pts
        pts = pts[segments[1]]

        plot_points(pts, convolved, name='CONVOLVED',
                    s=kwargs['kern'], c='r', cmap='viridis')

        labels_mask = np.where(labels, labels, np.nan)
        plot_points(pts, labels_mask, name='SEGMENTS',
                    s=kwargs['kern'], c='k', cmap='prism_r')

        ecc_map = labels_mask*0
        ecc_map.flat = pts_by_label[labels.flat]['ecc']
        plot_points(pts, ecc_map, name='ECCEN',
                    s=kwargs['kern'], c='k', cmap='Paired',
                    vmin=0, vmax=1, cbar=True)

        area_map = labels_mask*0
        area_map.flat = pts_by_label[labels.flat]['area']
        plot_points(pts, area_map, name='AREA',
                    s=kwargs['kern'], c='k', cmap='Paired',
                    vmin=0, vmax=1.2*kwargs['max_area'], cbar=True)

    def get_positions((n, filename)):
        global snapshot_num, imprefix
        snapshot_num = 0
        filebase = os.path.splitext(os.path.basename(filename))[0]
        imbase = os.path.join(imdir, filebase)
        imprefix = imbase
        image = prep_image(filename)
        ret = []
        for dot in dots:
            imprefix = '_'.join([imbase, dot.upper()])
            snapshot_num = 0
            if args.remove and dot == 'corner':
                # segments will only be defined in second iteration of loop
                try:
                    rmv = segments, args.kern
                except NameError:
                    rmv = None
            else:
                rmv = None
            out = find_particles(image, method='convolve', circ=args.boundary,
                                 rmv=rmv, **sizes[dot])
            segments = out[0]
            if args.plot > 1:
                plot_positions(*out, **sizes[dot])
                segments = np.array(segments[0], dtype=object)[segments[1]]

            nfound = len(segments)
            if nfound:
                centers = np.hstack([np.full((nfound, 1), n, 'f8'), segments])
            else:  # empty line of length 6 = id + len(Segment)
                centers = np.empty((0, 6))
            ret.append(centers)
        if not n % print_freq:
            fmt = '{:3d} {}s'.format
            print os.path.basename(filename).rjust(20), 'Found',
            print ', '.join([fmt(len(r), d) for r, d in zip(ret, dots)])
        return ret if args.both else ret[0]

    print_freq = 1 if args.verbose else len(filenames)//100 + 1
    threads = args.threads
    if threads < 1:
        cpus = cpu_count()
        if threads is None and args.plot <= 1:
            print "How many cpu threads to use? [{}] ".format(cpus),
            threads = int(raw_input() or cpus)
    threads = (args.plot > 1) or threads or cpus
    if threads > 1:
        print "Multiprocessing with {} threads".format(threads)
        p = Pool(threads)
        mapper = p.map
    else:
        mapper = map
    points = mapper(get_positions, enumerate(filenames))
    points = map(np.vstack, izip(*points)) if args.both else [np.vstack(points)]

    if args.plot:
        fig, axes = plt.subplots(nrows=len(dots), ncols=2, sharey='row')
        axes = np.atleast_2d(axes)
    else:
        axes = [None, None]
    if args.save:
        savenotice = "Saving {} positions to {}{{{},.npz}}".format
        hfmt = ('Kern {kern:.2f}, Min area {min_area:d}, '
                'Max area {max_area:d}, Max eccen {max_ecc:.2f}\n'
                'Frame    X           Y             Label  Eccen        Area')
        txtfmt = ['%6d', '%7.3f', '%7.3f', '%4d', '%1.3f', '%5d']
        ext = '.txt'+'.gz'*args.gz

    for dot, point, out, axis in zip(dots, points, outputs, axes):
        size = sizes[dot]
        if args.plot:
            eax, aax = axis
            label = "{} eccen (max {})".format(dot, size['max_ecc'])
            eax.hist(point[:, 4], bins=40, range=(0, 1),
                     alpha=0.5, color='r', label=label)
            eax.axvline(size['max_ecc'], 0, 0.5, c='r', lw=2)
            eax.set_xlim(0, 1)
            eax.set_xticks(np.arange(0, 1.1, .1))
            eax.set_xticklabels(map('.{:d}'.format, np.arange(10)) + ['1'])
            eax.legend(loc='best', fontsize='small')

            areas = point[:, 5]
            amin, amax = size['min_area'], size['max_area']
            s = np.ceil(hist_bin_auto(areas))
            bins = np.arange(amin, amax+s, s)
            label = "{} area ({} - {})".format(dot, amin, amax)
            aax.hist(areas, bins, alpha=0.5, color='g', label=label)
            aax.axvline(size['min_area'], c='g', lw=2)
            aax.set_xlim(0, bins[-1])
            aax.legend(loc='best', fontsize='small')
        if args.save:
            print savenotice(dot, out, ext)
            np.savetxt(out+ext, point, header=hfmt.format(**size),
                       delimiter='     ', fmt=txtfmt)
            helpy.txt_to_npz(out+ext, verbose=args.verbose, compress=args.gz)
    if args.save:
        from shutil import copy
        copy(first, prefix+'_'+os.path.basename(first))
        helpy.save_meta(prefix, meta,
                        path_to_tiffs=os.path.abspath(filepattern),
                        first_frame=os.path.abspath(first),
                        detect_thresh=args.thresh, detect_removed=args.remove)
        if args.plot:
            fig.savefig(prefix+'_SEGMENTSTATS.pdf')
    elif args.plot:
        plt.show()

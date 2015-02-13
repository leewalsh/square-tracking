#!/usr/bin/env python

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter, binary_erosion, convolve, center_of_mass, imread
from skimage import measure, segmentation
from skimage.filter import canny
from skimage.measure import label
from skimage.morphology import square, binary_closing, skeletonize
from skimage.morphology import disk as _disk
from skimage.viewer import ImageViewer
from collections import namedtuple
from matplotlib import pyplot as plt
import matplotlib.cm

DIST_THRESH = 100.

def label_particles_edge(im, sigma=2, closing_size=0, **extra_args):
    """ label_particles_edge(image, sigma=3, closing_size=3)
        Returns the labels for an image.
        Segments using Canny edge-finding filter.

        keyword arguments:
        image        -- The image in which to find particles
        sigma        -- The size of the Canny filter
        closing_size -- The size of the closing filter
    """
    edges = skimage.filter.canny(im, sigma=sigma)
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

def label_particles_convolve(im, thresh=4, rmv=None, csize=0, **extra_args):
    """ label_particles_convolve(im, thresh=2)
        Returns the labels for an image
        Segments using a threshold after convolution with proper gaussian kernel
        Uses center of mass from original image to find centroid of segments

        Input:
            image   the original image
            pos     if given, the positions at which to remove large dots
            thresh  the threshold above which pixels are included
    """
    from matplotlib import pyplot as plt
    #plt.hist(im.flatten(), bins=100)
    #plt.show()
    if csize == 0:
        raise ValueError('csize not set')
    elif csize < 0:
        #print "negative kernel"
        convolved = convolve(im, -gdisk(-csize))
    else:
        #print "positive kernel"
        convolved = convolve(im, gdisk(csize))

    if rmv is not None:
        convolved = remove_disks(convolved, rmv[0], disk(rmv[1]))

    #convolved[convolved < 0.] = 0.
    convolved -= convolved.min()
    convolved /= convolved.max()

    if isinstance(thresh, int):
        if rmv is not None:
            thresh -= 1
        thresh = convolved.mean() + thresh*convolved.std()

    labels = label(convolved > thresh)
    
    #print "found {} segments above thresh".format(labels.max())
    return labels, convolved

Segment = namedtuple('Segment', 'x y label ecc area'.split())

def filter_segments(labels, max_ecc=0.5, min_area=15, max_area=200, intensity=None, **extra_args):
    """ filter_segments(labels, max_ecc=0.5, min_area=15, max_area=200) -> [Segment]
        Returns a list of Particles and masks out labels for
        particles not meeting acceptance criteria.
    """
    pts = []
    centroid = 'Centroid' if intensity is None else 'WeightedCentroid'
    #rprops = measure.regionprops(labels, ['Area', 'Eccentricity', centroid], intensity)
    rprops = measure.regionprops(labels, intensity)
    for props in rprops:
        label = props['Label']
        area = props['Area']
        ecc = props['Eccentricity']
        if min_area > area:
            #print 'too small:', area
            pass
        elif area > max_area:
            #print 'too big:', area
            pass
        elif ecc > max_ecc:
            #print 'too eccentric:', ecc
            #labels[labels==label] = np.ma.masked
            pass
        else:
            x, y = props[centroid]
            pts.append(Segment(x, y, label, ecc, area))
    return pts

def find_particles(imfile, method='edge', return_image=False, **kwargs):
    """ find_particles(imfile, gaussian_size=3, **kwargs) -> [Segment],labels
        Find the particles in image im. The arguments in kwargs is
        passed to label_particles and filter_segments.

        Returns the list of found particles and the label image.
    """
    if args.verbose: print "opening", imfile
    im = imread(imfile).astype(float)
    if imfile.lower().endswith('tif'):
        # clean pixel noise from phantom images
        pass #im = median_filter(im, size=2)
    elif imfile.lower().endswith('jpg') and im.ndim == 3:
        # use just the green channel from color slr images
        im = im[..., 1]
    im[im < im.mean() - 2*im.std()] = 0.
    #im = gaussian_filter(im, 1)
    x = im.mean()# + im.std()
    im[im > x] = x
    im /= im.max()

    pl.clf()
    pl.imshow(im, cmap=matplotlib.cm.Greys_r)
    pl.savefig("a.png", dpi=300)

    intensity = None

    #print "Seeking particles using", method
    if method == 'walker':
        labels = label_particles_walker(im, **kwargs)
    elif method == 'edge':
        labels = label_particles_edge(im, **kwargs)
    elif method == 'convolve':
        labels, convolved = label_particles_convolve(im, **kwargs)
        intensity = 1 - im
    else:
        raise RuntimeError('Undefined method "%s"' % method)

    pts = filter_segments(labels, intensity=intensity, **kwargs)
    return (pts, labels) + ((convolved,) if return_image else ())

def disk(n):
    return _disk(n).astype(int)

def gdisk(n, w=None):
    """ gdisk(n):
        return a gaussian kernel
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
    parser.add_argument('-p', '--plot', action='count',
                        help="Produce a plot for each image. Use more p's for more images")
    parser.add_argument('-v', '--verbose', action='count',
                        help="Control verbosity")
    parser.add_argument('-o', '--output', default='POSITIONS',
                        help='Output file')
    parser.add_argument('-N', '--threads', default=1, type=int,
                        help='Number of worker threads')
    parser.add_argument('-c', '--corner', action='store_true',
                        help='Also find small corner dots')
    parser.add_argument('--slr', action='store_true',
                        help='Full resolution SLR was used')
    parser.add_argument('-k', '--kern', default=0, type=float,
                        help='Kernel size for convolution')
    parser.add_argument('--min', default=-1, type=int,
                        help='Minimum area')
    parser.add_argument('--max', default=np.inf, type=int,
                        help='Maximum area')
    parser.add_argument('--ecc', default=.8, type=float,
                        help='Maximum eccentricity')
    parser.add_argument('--ckern', default=0, type=float,
                        help='Kernel size for convolution for corner dots')
    parser.add_argument('--cmin', default=-1, type=int,
                        help='Minimum area for corner dots')
    parser.add_argument('--cmax', default=np.inf, type=int,
                        help='Maximum area for corner dots')
    parser.add_argument('--cecc', default=.8, type=float,
                        help='Maximum eccentricity for corner dots')
    parser.add_argument('--circ', action='store_true',
                        help='Open the first image and specify the circle of interest')
    args = parser.parse_args()

    if '*' in args.files[0] or '?' in args.files[0]:
        from glob import glob
        filenames = sorted(glob(args.files[0]))
    else:
        filenames = sorted(args.files)

    kern_area = np.pi*args.kern**2
    if args.min == -1:
        args.min = kern_area/2
        if args.verbose: print "using min =", args.min
    if args.max == np.inf:
        args.max = 2*kern_area
        if args.verbose: print "using max =", args.max

    ckern_area = np.pi*args.ckern**2
    if args.cmin == -1: args.cmin = ckern_area/2
    if args.cmax == np.inf: args.cmax = 2*ckern_area

    if args.circ:
        first_img = imread(filenames[0])
        x = []
        y = []
        origin = []
        viewer = ImageViewer(first_img)
        r2 = []
        ax = viewer.canvas.figure.add_subplot(111)

        def on_click(event):
            eventx = (event.x - 15) * 600. / 548
            x.append(eventx)
            eventy = (600 - event.y - 52) * 598. / 546
            y.append(eventy)
            if len(x) == 3:
                C = [x1**2 + y1**2 for x1, y1 in zip(x, y)]
                top = ((C[2] - C[0])*(x[1] - x[0]) + (C[1] - C[0])*(x[0] - x[2]))
                Y = top / 2 / ((y[2] - y[0])*(x[1] - x[0]) + (y[1] - y[0]) * (x[0] - x[2]))
                X = (C[1] - C[0] + 2*Y*(y[0] - y[1])) / 2 / (x[1] - x[0])
                origin.append((X, Y))
                r2.append((eventx - X)**2 + (eventy - Y)**2)
                circ = matplotlib.patches.Circle((X, Y), radius=r2[0]**.5, color='g', fill=False)
                ax.add_patch(circ)
                viewer.canvas.draw()

        viewer.canvas.mpl_connect('button_press_event', on_click)
        viewer.show()
        origin = origin[0]
        r2 = r2[0]

    if args.plot:
        cm = pl.cm.prism_r
        pdir = path.split(path.abspath(args.output))[0]
    threshargs =  {'max_ecc' : args.ecc,
                   'min_area': args.min,
                   'max_area': args.max,
                   'csize'   : args.kern}
    cthreshargs = {'max_ecc' : args.cecc,
                   'min_area': args.cmin,
                   'max_area': args.cmax,
                   'csize'   : args.ckern}
    #threshargs =  {'max_ecc' :   .7 if args.slr else  .7, # .6
    #               'min_area':  800 if args.slr else  15, # 870
    #               'max_area': 1600 if args.slr else 200, # 1425
    #               'csize'   :   22 if args.slr else  10}
    #cthreshargs = {'max_ecc' :  .8 if args.slr else .8,
    #               'min_area':  80 if args.slr else  3, # 92
    #               'max_area': 200 if args.slr else 36, # 180
    #               'csize'   :   5 if args.slr else  2}

    def plot_positions(savebase, level, pts, labels, convolved=None,):
        pl.clf()
        labels_mask = labels.astype(float)
        labels_mask[labels_mask==0] = np.nan
        pl.imshow(labels_mask, cmap=cm)
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
        out = find_particles(filename, method='convolve',
                            return_image=args.plot>2, **threshargs)
        if args.plot > 2:
            pts, labels, convolved = out
        else:
            pts, labels = out

        if args.circ:
            pts = [p for p in pts if (p.x - origin[0])**2 + (p.y - origin[1])**2 < r2]
            out = (pts,) + out[1:]

        nfound = len(pts)
        if nfound < 1:
            print 'Found no particles in ', path.split(filename)[-1]
            return
        centers = np.hstack([n*np.ones((nfound,1)), pts])
        print '%20s: Found %d particles' % (path.split(filename)[-1], nfound)
        if args.plot:
            savebase = path.join(pdir, path.split(filename)[-1].split('.')[-2])
            plot_positions(savebase, args.plot, *out)

        if args.corner:
            out = find_particles(filename, method='convolve',
                                return_image=args.plot>2, rmv=(pts, abs(args.kern)),
                                 **cthreshargs)
            if args.plot > 2:
                cpts, clabels, cconvolved = out
            else:
                cpts, clabels = out

            if args.circ:
                cpts = [p for p in cpts if (p.x - origin[0])**2 + \
                        (p.y - origin[1])**2 < r2]
            old_cpts = cpts[:]
            cpts = []
            for pt in old_cpts:
                for bigpt in pts:
                    if (pt[0]-bigpt[0])**2 + (pt[1]-bigpt[1])**2 < DIST_THRESH:
                        cpts.append(pt)
                        break
            out = (cpts,) + out[1:]

            nfound = len(cpts)
            if nfound < 1:
                print 'Found no corners, returning only centers'
                return centers
            print '%20s: Found %d corners' % (path.split(filename)[-1], nfound)
            if args.plot:
                plot_positions(savebase+'_CORNER', args.plot, *out)
            corners = np.hstack([n*np.ones((nfound,1)), cpts])
            return centers, corners

        return centers

    if args.threads > 1:
        print "Multiprocessing with {} threads".format(args.threads)
        p = Pool(args.threads)
        mapper = p.map
    else:
        mapper = map
    points = filter(lambda x: len(x) > 0, mapper(get_positions, enumerate(filenames)))

    if args.corner:
        points, corners = map(np.vstack, zip(*points))
        if 'CORNER' in args.output:
            coutput = args.output
        else:
            if 'POSITIONS' in args.output:
                coutput = args.output.replace('POS','CORNER_POS')
            else:
                outnames = args.output.split('.')
                outnames.insert(-1, '_CORNER.')
                coutput = ''.join(outnames)
        with open(coutput, 'w') as coutput:
            print "Saving corner positions to ", coutput.name
            coutput.write('# Kern     Min area    Max area      Max eccen\n')
            coutput.write('#%5.2f%7d%13d%15.2f\n' % (args.ckern, args.cmin, args.cmax, args.cecc))
            coutput.write('#\n')
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
        output.write('# Kern     Min area    Max area      Max eccen\n')
        output.write('#%5.2f%7d%13d%15.2f\n' % (args.kern, args.min, args.max, args.ecc))
        output.write('#\n')
        output.write('# Frame    X           Y             Label  Eccen        Area\n')
        np.savetxt(output, points, delimiter='     ',
                fmt=['%6d', '%7.3f', '%7.3f', '%4d', '%1.3f', '%5d'])
 

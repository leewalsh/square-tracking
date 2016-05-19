#!/usr/bin/env python
# encoding: utf-8

from __future__ import division

from math import sqrt
from cmath import phase
from itertools import combinations

import numpy as np
from numpy.polynomial import polynomial as poly
from scipy.spatial.distance import pdist, cdist
from scipy.spatial import Voronoi, Delaunay, cKDTree as KDTree
from scipy.ndimage import gaussian_filter
from scipy.signal import hilbert, correlate, convolve
from scipy.stats import rv_continuous, vonmises
from scipy.optimize import curve_fit
from skimage.morphology import disk, binary_dilation

import helpy

ss = helpy.S_slr  # side length of square in pixels
rr = helpy.R_slr  # radius of disk in pixels

pi = np.pi
tau = 2*pi


def bulk(positions, margin=0, full_N=None, center=None, radius=None, ss=ss,
         verbose=False):
    """ Filter marginal particles from bulk particles to reduce boundary effects
            positions:  (N, 2) array of particle positions
            margin:     width of margin, in units of pixels or particle sides
            full_N:     actual number of particles, to renormalize assuming
                        uniform distribution of undetected particles.
            center:     if known, (2,) array of center position
            radius:     if known, radius of system in pixels

        returns
            bulk_N:     the number particles in the bulk
            bulk_mask:  a mask the shape of `positions`
    """
    if center is None:
        center = 0.5*(positions.max(0) + positions.min(0))
    if margin < ss:
        margin *= ss
    d = helpy.dist(positions, center)   # distances to center
    if radius is None:
        n_pts = len(positions)
        max_sep = 0 if n_pts > 1e4 else cdist(positions, positions).max()
        radius = max(max_sep/2, d.max()) + ss/2
    elif radius < ss:
        radius *= ss
    dmax = radius - margin
    depth = d - dmax
    bulk_mask = area_overlap(depth/ss, ss=1, method='aligned_square')
    bulk_N = bulk_mask.sum()
    # bulk_mask = bulk_mask >= 0.5   # mask of centers within bulk
    if full_N:
        bulk_N *= full_N/len(positions)
    if verbose:
        print 'margin:', margin/ss,
        print 'center:', center,
        print 'radius:', radius
        print 'max r: ', dmax/ss,
        print 'bulk_N:', bulk_N,
        print
    return bulk_N, bulk_mask, center, radius


def area_overlap(sep, ss=1, method='center'):
    """Overlap between particle and included area

    Aligned square against flat boundary
    """
    if method == 'center':
        overlap = ss*ss*(sep <= 0)
    elif method == 'aligned_square':
        overlap = ss * (ss/2 - sep)
        np.clip(overlap, 0, ss, out=overlap)
    else:
        raise ValueError('Unknown method "{}"'.format(method))
    return overlap


def pair_indices(n, asarray=False):
    """ pairs of indices to a 1d array of objects.
        equivalent to but faster than `np.triu_indices(n, 1)`
        stackoverflow.com/questions/22390418

        To index the upper triangle of a matrix, just use the returned tuple.
        Otherwise, use `i` and `j` separately to index the first then second of
        the pair
    """
    if n < 7:
        ind = combinations(xrange(n), 2)
        return np.array(tuple(ind)) if asarray else zip(*ind)
    rng = np.arange(1, n)
    i = np.repeat(rng - 1, rng[::-1])
    j = np.arange(n*(n-1)//2) + np.repeat(n - np.cumsum(rng[::-1]), rng[::-1])
    ind = i, j
    return np.array(ind).T if asarray else ind


def radial_distribution(positions, dr=ss/5, nbins=None, dmax=None, rmax=None,
                        margin=0, do_err=False, ss=ss):
    """ radial_distribution(positions):
        the pair correlation function g(r)
        calculated using a histogram of distances between particle pairs
        excludes pairs in margin of given width
    """
    center = 0.5*(positions.max(0) + positions.min(0))
    d = helpy.dist(positions, center)   # distances to center
    # faster than squareform(pdist(positions)) wtf
    r = cdist(positions, positions)
    radius = np.maximum(r.max()/2, d.max()) + ss/2
    if rmax is None:
        rmax = 2*radius     # this will have terrible statistics at large r
    if nbins is None:
        nbins = rmax//dr
    if dmax is None:
        if margin < ss:
            margin *= ss
        dmax = radius - margin
    ind = pair_indices(len(positions))
    # for weighting, use areas of the annulus, which is:
    #   number * arclength * dr = N alpha r dr
    #   where alpha = 2 arccos( (r2 + d2 - R2) / 2 r d )
    cosalpha = 0.5 * (r*r + d*d - radius*radius) / (r * d)
    alpha = 2 * np.arccos(np.clip(cosalpha, -1, None))
    dmask = d <= dmax
    w = np.where(dmask, np.reciprocal(alpha*r*dr), 0)
    w = 0.5*(w + w.T)
    assert np.all(np.isfinite(w[ind]))
    # Different ways to count `n`:
    # number of 'bulk' (inner) particles
    n = np.count_nonzero(dmask)
    # effective N from number of pairs:
    # n = 0.5*(1 + sqrt(1 + 8*np.count_nonzero(w[ind])))
    # total number of particles:
    # n = len(w)
    w *= 2/n
    assert np.allclose(positions.shape[0], map(len, [r, d, w, positions]))
    ret = np.histogram(r[ind], bins=nbins, range=(0, rmax), weights=w[ind])
    if do_err:
        return ret, np.histogram(r[ind], bins=nbins, range=(0, rmax)), n
    else:
        return ret + (n,)


def rectify(positions, margin=0, dangonly=False):
    angles, nmask, dmask = pair_angles(positions, margin=margin)
    try:
        # find four modal angles and gaps
        # rotate by angle of greater of first two gaps
        # so that first gap is larger of two, last gap is smaller
        pang = primary_angles(angles, m=4, bins=720, ret_hist=False)[0]
        # dang[i] = pang[i] - pang[i-1]
        dang = dtheta(pang, np.roll(pang, -1), m=1)
        rectang = np.nan if dangonly else -pang[np.argmax(dang[:2])]
    except RuntimeError:
        print "Can't find four peaks, using one"
        rectang = np.nan if dangonly else -pair_angle_op(angles, nmask, m=4)[1]
        dang = np.array([np.nan, np.nan, np.nan, np.nan])
    return rectang, dang


def distribution(positions, rmax=10, bins=10, margin=0, rectang=0):
    if margin < ss:
        margin *= ss
    center = 0.5*(positions.max(0) + positions.min(0))
    d = helpy.dist(positions, center)   # distances to center
    dmask = d < d.max() - margin
    r = cdist(positions, positions[dmask])  # .ravel()
    radius = np.maximum(r.max()/2, d.max()) + ss/2
    cosalpha = 0.5 * (r**2 + d[dmask]**2 - radius**2) / (r * d[dmask])
    alpha = 2 * np.arccos(np.clip(cosalpha, -1, None))
    dr = radius / bins
    w = dr**-2 * tau/alpha
    w[~np.isfinite(w)] = 0
    if rmax < ss:
        rmax *= ss
    rmask = r < rmax
    # origin must be within margin
    displacements = positions[:, None] - positions[None, dmask]
    if rectang:
        if rectang is True:
            rectang = rectify(positions, margin=margin)[0]
        rotate2d(displacements, rectify(positions, margin=margin))
    return np.histogramdd(displacements[rmask], bins=bins, weights=w[rmask])[0]


def rotate2d(vectors, angles=None, basis=None):
    """ rotate vectors by angles

        Parameters
        vectors:    vectors to rotate, must have shape (..., 2)
        angles:     angles by which to rotate, must broadcast to shape (...)
        basis:      or, change to this basis, shape (2, ...) or (2, 2, ...)
        inplace:    whether to modify `vectors` in place

        Returns
        None if inplace else vectors, rotated or in new basis
    """
    if basis is None or basis.shape[:2] != (2, 2):
        c, s = basis if angles is None else (np.cos(angles), -np.sin(angles))
        basis = np.array([[c, s], [-s, c]])
    # note we multiply basis.T * vectors, since basis is not a rotation matrix
    return np.einsum('ji...,...i->...j', basis, vectors, casting='same_kind')


def get_positions(data, frame, pid=None):
    """ get_positions(data, frame)

        Takes:
            data: structured array of data
            frame: int or list of ints of frame number

        Returns:
            list of tuples (x, y) of positions of all particles in those frames
    """
    if np.iterable(frame):
        fmask = np.in1d(data['f'], frame)
    else:
        fmask = data['f'] == frame
    if pid is not None:
        fiddata = data[fmask & (data['id'] == pid)]
        return np.array(fiddata['x'], fiddata['y'])
    return np.column_stack((data['x'][fmask], data['y'][fmask]))


def avg_hists(gs, rgs):
    """ avg_hists(gs, rgs)
        takes:
            gs: an array of g(r) for several frames
            rgs: their associated r values
        returns:
            g_avg: the average of gs over frames
            dg_avg: their std dev / sqrt(length)
            rg: r for the avgs (just uses rgs[0] for now)
    """
    assert np.allclose(rgs, rgs[:1])
    rg = rgs[0]
    g_avg = gs.mean(0)
    # dg_avg = gs.std(0)/sqrt(len(gs))
    dg_avg = gs.var(0)
    return g_avg, dg_avg, rg


def build_gs(data, framestep=1, dr=0.1, dmax=None, rmax=None, margin=0,
             do_err=False, ss=ss):
    """Calculate and build g(r) for each frame

    Parameters
        data: the structured array of data
        framestep: how many frames to skip
        dr: bin width in particle size units
        dmax: passed to radial distribution
        rmax: passed to radial distribution
        margin: passed to radial distribution

    Returns
        gs: an array of g(r) for several frames
        rgs: right-hand edges (exclusive upper bounds) of bins, so that:
            g[i] is the count of pairs with r[i-1] <= r < r[i]
    """
    frames = np.arange(data['f'].min(), data['f'].max()+1, framestep)
    dr *= ss
    nbins = 1 + rmax//dr if rmax and dr else None
    gs = rgs = egs = ergs = None
    for nf, frame in enumerate(frames):
        positions = get_positions(data, frame)
        g, rg, n = radial_distribution(positions, dr=dr, nbins=nbins, dmax=dmax,
                                       rmax=rmax, margin=margin, do_err=do_err,
                                       ss=ss)
        if do_err:
            (g, rg), (eg, erg), n = g, rg, n
            erg = erg[1:]
        rg = rg[1:]     # rg gives right-hand edges of bins
        if gs is None:
            nbins = g.size
            gs = np.zeros((frames.size, nbins))
            rgs = gs.copy()
            if do_err:
                egs = np.zeros((frames.size, nbins))
                ergs = gs.copy()
        gs[nf, :len(g)] = g
        rgs[nf, :len(g)] = rg
        if do_err:
            egs[nf, :len(eg)] = eg
            ergs[nf, :len(eg)] = erg
    return ((gs, rgs), (egs, ergs), n) if do_err else (gs, rgs, n)


def structure_factor(positions, m=4, margin=0):
    """return the 2d structure factor"""
    raise StandardError("um this isn't finished")
    from scipy.fftpack import fft2
    # center = 0.5*(positions.max(0) + positions.min(0))
    inds = np.round(positions - positions.min()).astype(int)
    f = np.zeros(inds.max(0)+1)
    f[inds[:, 0], inds[:, 1]] = 1
    f = binary_dilation(f, disk(ss/2))
    return fft2(f, overwrite_x=True)


def orient_op(orientations, m=4, positions=None, margin=0,
              ret_complex=True, do_err=False, globl=False, locl=False):
    """ orient_op(orientations, m=4)
        Returns the global m-fold particle orientational order parameter

                1   N    i m theta
        Phi  = --- SUM e          j
           m    N  j=1
    """
    if not (globl or locl):
        globl = True
        locl = orientations.ndim == 2
    np.mod(orientations, tau/m, orientations) # what's this for? (was tau/4 not tau/m)
    if margin:
        if margin < ss:
            margin *= ss
        center = 0.5*(positions.max(0) + positions.min(0))
        d = helpy.dist(positions, center)   # distances to center
        orientations = orientations[d < d.max() - margin]
    phis = np.exp(m*orientations*1j)
    if locl:
        phis = np.nanmean(phis, 1)
    if do_err:
        err = np.nanstd(phis, ddof=1)/sqrt(np.count_nonzero(~np.isnan(phis)))
    if not globl:
        return (np.abs(phis), err) if do_err else np.abs(phis)
    phi = np.nanmean(phis) if ret_complex else np.abs(np.nanmean(phis))
    if locl:
        return (np.abs(phis), phi, err) if do_err else (np.abs(phis), phi)
    return (phi, err) if do_err else phi


def dtheta(i, j=None, m=1, sign=False):
    """ given two angles or one array (N, 2) of pairs
        returns the _smallest angle between them, modulo m
        if sign is True, retuns a negative angle for i<j, else abs
    """
    ma = tau/m
    if j is not None:
        diff = i - j
    elif i.shape[1] == 2:
        diff = np.subtract(*i.T)
    diff = (diff + ma/2) % ma - ma/2
    return diff if sign else np.abs(diff)


def bin_average(r, f, bins=10):
    """ Binned average of function f(r)
        r : independent variable to be binned over
        f : function to be averaged
        bins (default 10): can be number of bins or bin edges len(nbins)+1
    """
    if np.iterable(bins):
        pass
    elif bins == 1:
        if r.dtype.kind not in 'iu':
            assert np.allclose(r, np.around(r)), 'need integer array for bins=1'
            print 'converting to int array'
            r = r.astype(int)
        n = np.bincount(r)
        return np.bincount(r, weights=f)/n
    elif bins < 1:
        bins = np.arange(r.min(), r.max()+1/bins, 1/bins)
    n, bins = np.histogram(r, bins)
    return np.histogram(r, bins, weights=f)[0]/n, bins


def autocorr(f, side='right', cumulant=True, norm=1, mode='same',
             verbose=False, reverse=False, ret_dx=False):
    """ autocorr(f, side='right', cumulant=True, norm=True, mode='same',
                 verbose=False, reverse=False, ret_dx=False):

        The auto-correlation of function f
        returns the auto-correlation function
            <f(x) f(x - dx)> averaged over x

        See also `crosscorr(f, g, ...)`

        f:      1d array, as function of x
        side:   'right' returns only dx > 0, (x' < x)
                'left'  returns only dx < 0, (x < x')
                'both'  returns entire correlation
        cumulant: if True, subtracts mean of the function before correlation
        mode:   passed to scipy.signal.correlate, has little effect here, but
                returns shorter correlation array
    """
    return crosscorr(f, f, side=side, cumulant=cumulant, norm=norm, mode=mode,
                     verbose=verbose, reverse=reverse, ret_dx=ret_dx)


def crosscorr(f, g, side='both', cumulant=True, norm=False, mode='same',
              verbose=False, reverse=False, ret_dx=False):
    """ crosscorr(f, g, side='both', cumulant=True, norm=False, mode='same',
                  verbose=False, reverse=False, ret_dx=False):

        The cross-correlation of f and g
        returns the cross-correlation function
            <f(x) g(x - dx)> averaged over x

        f, g:       1d arrays, as function of x, with same lengths
        side:       'right' returns only dx > 0, (x' < x)
                    'left'  returns only dx < 0, (x < x')
                    'both'  returns entire correlation
        cumulant:   if True, subtracts mean of the function before correlation
        mode:       passed to scipy.signal.correlate, has little effect here.
        norm:       normalize by the correlation at no shift,
                        that is, by <f(x) g(x) >
                    if 1, divide
                    if 0, subtract
        ret_dx:     if True, return the dx shift between f and g
                        that is, if we are looking at <f(x) g(x')>
                        then dx = x - x'
        reverse:    if True, flip g relative to f, that is, use convolve
                        instead of correlate, which calculates <f(x) g(dx - x)>
        verbose:    if True or 1, be careful but not very verbose
                    if 2 or greater, be careful and verbose
    """
    l = len(f)
    # midpoint (dx = 0), length of correlation
    m, L = (l//2, l) if mode == 'same' else (l-1, 2*l-1)
    if verbose > 1:
        print "l: {}, m: {}, l-m: {}, L: {}".format(l, m, l-m, L)
    msg = "len(f): {}, len(g): {}\nlengths must match for proper normalization"
    assert l == len(g), msg.format(l, len(g))

    if cumulant:
        if cumulant is True:
            f = f - f.mean()
            g = g - g.mean()
        elif cumulant[0]:
            f = f - f.mean()
        elif cumulant[1]:
            g = g - g.mean()

    c = convolve(f, g, mode=mode) if reverse else correlate(f, g, mode=mode)
    if verbose and (f is g):
        maxi = c.argmax()
        assert maxi == m, ("autocorrelation not peaked at 0: "
                           "max ({}) not at m ({})").format(maxi, m)

    # divide by overlap
    nl = np.arange(l - m, l)
    nr = np.arange(l, m - (L - l), -1)
    n = np.concatenate([nl, nr])
    if verbose:
        overlap = correlate(np.ones(l), np.ones(l), mode=mode).astype(int)
        if verbose > 1:
            print nl, nr
            print '      n: {}\noverlap: {}'.format(n, overlap)
        msg = "overlap miscalculated:\n\t{}\n\t{}"
        assert np.allclose(n, overlap), msg.format(n, overlap)
        assert n[m] == l, "overlap normalizer not l at m"
    c /= n
    if verbose:
        msg = ("normalization calculations don't all match: "
               "c[m]: {}, np.dot(f, g): {}, c.max(): {}")
        fgs = c[m], np.dot(f, g)/len(f), c[m]  #, c.max()
        if verbose > 1 and norm in (0, 1):
            print ("subtracting", "normalizing by")[norm], "scaler:", fgs[0]
        assert np.allclose(fgs[0], fgs), norm_assert_msg.format(*fgs)

    if norm is 1:
        c /= c[m]
    elif norm is 0:
        c -= c[m]
    elif verbose > 1:
        print 'central value:', c[m]

    if ret_dx:
        if side == 'both':
            return np.arange(-m, L-m), c
        elif side == 'left':
            # return np.arange(0, -m-1, -1), c[m::-1]
            return np.arange(-m, 1), c[:m+1]
        elif side == 'right':
            return np.arange(0, L-m), c[m:]

    if side == 'both':
        return c
    elif side == 'left':
        return c[m::-1]
    elif side == 'right':
        return c[m:]


def poly_exp(x, gamma, a, *coeffs):  # return_poly=False):
    """ exponential decay with a polynomial decay scale

                 - x
           ------------------
           a + b x + c x² ...
        e
    """
    return_poly = False
    d = poly.polyval(x, coeffs or (1,))
    f = a*np.exp(-x**gamma/d)
    return (f, d) if return_poly else f


def vary_gauss(a, sig=1, verbose=False):
    n = len(a)
    b = np.empty_like(a)

    if np.isscalar(sig):
        sig *= np.arange(n)
    elif isinstance(sig, tuple):
        sig = poly.polyval(np.arange(n), sig)
    elif callable(sig):
        sig = sig(np.arange(n))
    elif hasattr(sig, '__getitem__'):
        assert len(a) == len(sig)
    else:
        raise TypeError('`sig` is neither callable nor arraylike')
    for i, s in enumerate(sig):
        # build the kernel:
        w = round(2*s)  # kernel half-width, must be integer
        if s == 0:
            s = 1
        k = gauss(np.arange(-w, w+1, dtype=float), sig=s)

        # slice the array (min/max prevent going past ends)
        al = max(i - w,     0)
        ar = min(i + w + 1, n)
        ao = a[al:ar]

        # and the kernel
        kl = max(w - i,     0)
        kr = min(w - i + n, 2*w+1)
        ko = k[kl:kr]
        b[i] = np.dot(ao, ko)/ko.sum()

    return b


def msd(xs, ret_taus=False):
    """ So far:
          - only accepts the positions in 1 or 2d array (no data structure)
          - can only do dt0 = dtau = 1

        msd = < [x(t0 + tau) - x(t0)]**2 >
            = < x(t0 + tau)**2 > + < x(t0)**2 > - 2 * < x(t0+tau) x(t0) >
            = cumsum

        The first two terms are averaged over all values of t0 that are valid
        for the current value of tau. Thus, we have sums of x(t0) and x(t0+tau)
        for all values of t0 in [0, T - tau). For small values of tau, nearly
        all values of t0 are valid, and vice versa. The averages for increasing
        values of tau is the reverse of cumsum(x) / (T-tau)

        Time must be axis 0, but any number of dimensions is allowed (along axis 1)
    """

    xs = np.asarray(xs)
    d = xs.ndim
    if d == 1:
        T = len(xs)
        xs = xs[:, None]
    elif d == 2:
        T, d = xs.shape
    else:
        msg = "can't handle xs.ndims > 2. xs.shape is {}"
        raise ValueError(msg.format(xs.shape))

    # The last term is an autocorrelation for x(t):
    xx0 = np.apply_along_axis(autocorr, 0, xs, side='right', cumulant=False,
                              norm=False, mode='full', verbose=False,
                              reverse=False, ret_dx=False)

    ntau = np.arange(T, 0, -1)  # = T - tau
    x2 = xs * xs
    # x0avg = np.cumsum(x2)[::-1] / ntau
    # xavg = np.cumsum(x2[::-1])[::-1] / ntau
    # we'll only ever combine these, which can be done with one call:
    # x0avg + xavg == np.cumsum(x2 + x2[::-1])[::-1] / ntau
    # assert x0avg + xavg == np.cumsum(x2 + x2[::-1])[::-1] / ntau
    x2s = np.cumsum(x2 + x2[::-1], axis=0)[::-1] / ntau[:, None]

    msd = x2s - 2*xx0
    msd = msd.sum(1)    # straight sum over dimensions (x2 + y2 + ...)

    return np.column_stack([np.arange(T), msd]) if ret_taus else msd


def decay_scale(f, x=None, method='mean', smooth='gauss', rectify=True):
    """ Find the decay scale of a function f(x)
        f: a decaying 1d array
        x: independent variable, default is range(len(f))
        method: how to calculate
            'integrate': integral of f(t) assuming exp'l form
            'mean': mean lifetime < t > = integral of t*f(t)
        smooth: smooth data first using poly_exp
    """
    l = len(f)
    if x is None:
        x = np.arange(l)

    if smooth == 'fit':
        p, _ = curve_fit(poly_exp, x, f, [1, 1, 1])
        f = poly_exp(x, *p)
    elif smooth.startswith('gauss'):
        g = [gaussian_filter(f, sig, mode='constant', cval=f[sig])
             for sig in (1, 10, 100, 1000)]
        f = np.choose(np.repeat([0, 1, 2, 3], [10, 90, 900, len(f)-1000]), g)

    if rectify:
        np.maximum(f, 0, f)

    method = method.lower()
    if method.startswith('mean'):
        return np.dot(x, f) / f.sum()
    elif method.startswith('int'):
        return f.sum()
    elif method.startswith('inv'):
        return f.sum() / np.dot(1/(x+1), f)


def orient_corr(positions, orientations, m=4, margin=0, bins=10):
    """ orient_corr():
        the orientational correlation function g_m(r)
        given by mean(phi(0)*phi(r))
    """
    center = 0.5*(positions.max(0) + positions.min(0))
    d = helpy.dist(positions, center)   # distances to center
    if margin < ss:
        margin *= ss
    loc_mask = d < d.max() - margin
    r = pdist(positions[loc_mask])
    ind = np.column_stack(pair_indices(np.count_nonzero(loc_mask)))
    pairs = orientations[loc_mask][ind]
    diffs = np.cos(m*dtheta(pairs, m=m))
    return bin_average(r, diffs, bins)


def get_neighbors(tess, p, pm=None, ret_pairs=False):
    """ give neighbors in voronoi tessellation v of point id p
        if already calculated, pm is point mask
    """
    if isinstance(tess, Delaunay):
        indices, indptr = tess.vertex_neighbor_vertices
        if p == 'all':
            p = xrange(len(indices) - 1)
        if np.iterable(p):
            return [indptr[indices[q]:indices[q+1]] for q in p]
        return indptr[indices[p]:indices[p+1]]
    elif isinstance(tess, Voronoi):
        if np.iterable(p):
            raise ValueError("Can only find neighbors of 1 point with Voronoi")
        pm = tess.ridge_points == p if pm is None else pm[p]
        pm = np.any(pm, 1)
        pairs = tess.ridge_points[pm]
        return pairs if ret_pairs else pairs[pairs != p]


def neighborhoods(positions, voronoi=False, size=None, reach=None,
                  tess=None, tree=None):
    """Build a list of lists or padded array of neighborhoods around each point

    select neighbors by any combination of three basic choices:
        Voronoi/Delaunay, distance/ball, count/nearest/number

    parameters
    positions : array with shape (N, 2) or fields 'x' and 'y'
    voronoi : whether to require pairs to be voronoi or delaunay neighbors
    size : maximum size for each neighborhood excluding center/self
    reach : maximum distance to search (exclusive).  scalar for distance/ball
        for other criteria, it may be an array of distances or a str such as
        '[min|max|mean]*{factor}' where the function is of neighbor distances
    tess, tree : optionally provide spatial.Delaunay or spatial.KDTree instance

    returns
    neighbors : list of lists (or padded array) with shape (npoints, size)
        neighbors[i] gives indices in positions to neighbors of positions[i]
        i.e., the coordinates for all neighbors of positions[i] are given by
        positions[neighbors[i]], with shape (size, 2)
    mask : True if not a real neighbor
    distances : distance to the neighbor, only calculated if needed.
    """
    try:
        fewest, most = size
    except TypeError:
        fewest, most = None, size
    need_dist = True
    filter_reach = reach is not None
    try:
        dub = float(reach)
        filter_reach = False
    except (TypeError, ValueError):
        dub = np.inf
    if voronoi:
        tess = tess or Delaunay(positions)
        neighbors = get_neighbors(tess, 'all')
    elif most is not None:
        tree = tree or KDTree(positions)
        distances, neighbors = tree.query(
            positions, np.max(most)+1, distance_upper_bound=dub)
        distances, neighbors = distances[:, 1:], neighbors[:, 1:]  # remove self
        mask = np.isinf(distances)
        neighbors[mask] = np.where(mask)[0]
        need_dist = False
    elif reach is None:
        raise ValueError("No limits on neighborhood selection applied")
    else:
        tree = tree or KDTree(positions)
        neighbors = tree.query_ball_tree(tree, dub)
        for i in xrange(len(neighbors)):
            neighbors[i].remove(i)  # remove self
    if need_dist:
        ix = np.arange(len(positions))[:, None]
        neighbors, mask = helpy.pad_uneven(neighbors, ix, True, int)
        distances = cdist(positions, positions)[ix, neighbors]
        distances[mask] = np.inf
        sort = distances.argsort(1)
        distances, neighbors = distances[ix, sort], neighbors[ix, sort]
    if isinstance(reach, basestring):
        fun, fact = reach.split('*') if '*' in reach else (reach, 1)
        ix = np.arange(len(positions))
        fun = {'mean': np.nanmean, 'min': np.nanmin, 'max': np.nanmax,
               'median': np.nanmedian}[fun]
        fact = float(fact)
        reach = fun(np.where(mask, np.nan, distances), 1, keepdims=True)*fact
    if filter_reach:
        mask[distances >= reach] = True
        distances[mask] = np.inf
    if fewest is not None:
        mask[(~mask).sum(1) < fewest] = True
    if np.iterable(most):
        extra = np.clip(mask.shape[1] - most, 0, None)
        i = np.where(extra)
        extra = extra[i]
        i = np.repeat(i[0], extra)
        j = mask.shape[1] - np.concatenate(map(range, extra)) - 1
        mask[i, j] = True
        most = most.max()
    return neighbors[:, :most], mask[:, :most], distances[:, :most]


def poly_area(corners):
    # calculate area of polygon
    area = 0.0
    n = len(corners)
    for i in xrange(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    return abs(area) / 2.0


def density(positions, method, vor=None, **neigh):
    if method == 'vor':
        return voronoi_density(vor or positions)
    if 'dist' in method or 'inv' in method:
        neigh, nmask, dists = (neigh.get('neighbors') or
                               neighborhoods(positions, **neigh))
        dists = np.where(nmask, np.nan, dists)
        if 'area' in method:
            areas = dists**2
            if 'inv' in method:
                return np.nanmean(1/areas, 1)
            else:
                return 1/np.nanmean(areas, 1)
        else:
            if 'inv' in method:
                return np.nanmean(1/dists, 1)**2
            else:
                return 1/np.nanmean(dists, 1)**2


def gaussian_density(positions, scale=None, unit_length=1, extent=(600, 608)):
    dens = np.zeros(extent, float)
    indices = np.around(positions).astype('u4').T
    dens[indices] = unit_length*unit_length
    if scale is None:
        scale = 2*unit_length
    gaussian_filter(dens, scale, mode='constant')


def voronoi_density(pos_or_vor):
    vor = pos_or_vor if isinstance(pos_or_vor, Voronoi) else Voronoi(pos_or_vor)
    regions = (vor.regions[regi] for regi in vor.point_region)
    return np.array([0 if -1 in reg else 1/poly_area(vor.vertices[reg])
                     for reg in regions])

def binder(positions, orientations, bl, m=4, method='ball', margin=0):
    """Calculate the binder cumulant, given positions and orientations.

    bl: the binder length scale, such that
        B(bl) = 1 - .333 * S4 / S2^2
    where SN are <phibl^N> averaged over each block/cluster of size bl in frame.
    """
    if margin:
        if margin < ss:
            margin *= ss
        center = 0.5*(positions.max(0) + positions.min(0))
        dmask = d < d.max() - margin
        positions = positions[dmask]
        orientations = orientations[dmask]
    if 'neigh' in method or 'ball' in method:
        tree = KDTree(positions)
        balls = tree.query_ball_tree(tree, bl)
        balls, ball_mask = helpy.pad_uneven(balls, 0, True, int)
        ball_orient = orientations[balls]
        ball_orient[ball_mask] = np.nan
        phis = np.nanmean(np.exp(m*ball_orient*1j), 1)
        phi2 = np.dot(phis, phis) / len(phis)
        phiphi = phis*phis
        phi4 = np.dot(phiphi, phiphi) / len(phiphi)
        return 1 - phi4 / (3*phi2*phi2)
    else:  # elif method=='block':
        raise ValueError("method {} not implemented".format(method))
        # left, bottom = positions.min(0)
        # right, top = positions.max(0)
        # xbins = np.arange(left, right + bl, bl)
        # ybins = np.arange(bottom, top + bl, bl)
        # blocks = np.rollaxis(np.indices((xbins.size, ybins.size)), 0, 3)
        # block_ind = np.column_stack([
        #              np.digitize(positions[:, 0], xbins),
        #              np.digitize(positions[:, 1], ybins)])


def pair_angles(xy_or_orient, neighbors, nmask, ang_type='absolute', margin=0):
    """do something with the angles a given particle makes with its neighbors

    Parameters
    xy_or_orient:  either (N, 2) array of positions or (N,) array of angles
    neighbors:  (N, k) array of k neighbors
    nmask:      mask for neighbors
    ang_type:   string, choice of 'absolute' (default), 'relative', 'delta'
    margin:     is the width of excluded boundary margin

    Returns
    angles:     array of angles between neighboring pairs
    dmask:      margin mask, only returned if margin > 0
    """

    if xy_or_orient.ndim == 2:
        dx, dy = (xy_or_orient[neighbors] - xy_or_orient[:, None, :]).T
        angles = np.arctan2(dy, dx).T % tau
    else:
        angles = xy_or_orient[neighbors] - xy_or_orient[:, None]
    if ang_type == 'relative':
        # subtract off angle to nearest neighbor
        angles -= angles[:, :1]
    elif ang_type == 'delta':
        # sort by angle then take diff
        angles[nmask] = np.inf
        angles.sort(-1)
        angles -= np.roll(angles, 1, -1)
        # only keep if we have all k neighbors
        nmask = np.all(nmask, 1)
    elif ang_type != 'absolute':
        raise ValueError("unknown ang_type {}".format(ang_type))
    angles[nmask] = np.nan
    if margin:
        if margin < ss:
            margin *= ss
        center = 0.5*(xy_or_orient.max(0) + xy_or_orient.min(0))
        d = helpy.dist(xy_or_orient, center)
        dmask = d < d.max() - margin
        angles = angles[dmask]
        nmask = nmask[dmask]
    return (angles % tau, nmask) + ((dmask,) if margin else ())


def pair_angle_op(angles, nmask=None, m=4, globl=False, locl=False):
    """calculate the pair-angle (bond angle) order parameter

    the parameter for particle i is defined as:
        psi_m_i = < exp(i m theta_ij) >
    averaged over neighbors j of particle i
    the global parameter is the mean over all particles i:
        Psi_m = < psi_m_i >

    Parameters
    angles: angles between neighboring pairs (from pair_angles)
    nmask:  neighbor mask if invalid angles are not np.nan (None)
    m:      symmetryangles will be considered modulo tau/m

    Returns
    mag:    the absolute value |psi|
    ang:    the phase of psi mod tau/m
    psims:  the local values of psi for each particle
    """
    if not (globl or locl):
        globl = locl = True
    if nmask is not None:
        angles[nmask] = np.nan
    psims = np.nanmean(np.exp(m*angles*1j), 1)
    if not globl:
        return np.abs(psims)
    psim = np.nanmean(psims)
    mag = abs(psim)
    ang = phase(psim)/m
    if locl:
        return mag, ang, psims
    return mag, ang


def pair_angle_corr(positions, psims, rbins=10):
    assert len(positions) == len(psims), "positions does not match psi_m(r)"
    i, j = pair_indices(len(positions))
    psi2 = psims[i].conj() * psims[j]
    return bin_average(pdist(positions), psi2, rbins)


class vonmises_m(rv_continuous):
    def __init__(self, m):
        self.shapes = ''
        for i in range(m):
            self.shapes += 'k%d,l%d' % (i, i)
        self.shapes += ',scale'
        rv_continuous.__init__(self, a=-np.inf, b=np.inf, shapes=self.shapes)
        self.numargs = 2*m

    def _pdf(self, x, *lks):
        print 'lks', lks
        locs, kappas = lks[:len(lks)/2], lks[len(lks)/2:]
        print 'x', x
        print 'locs', locs
        print 'kapps', kappas
        # return np.sum([vonmises.pdf(x, l, k)
        #                for l, k in zip(locs, kappas)], 0)
        ret = np.zeros_like(x)
        for l, k in zip(locs, kappas):
            ret += vonmises.pdf(x, l, k)
        return ret / len(locs)


class vonmises_4(rv_continuous):
    def __init__(self):
        rv_continuous.__init__(self, a=-np.inf, b=np.inf)

    def _pdf(self, x,
             l1, l2, l3, l4,
             k1, k2, k3, k4,
             a1, a2, a3, a4, c):
        return a1*vonmises.pdf(x, k1, l1) + \
               a2*vonmises.pdf(x, k2, l2) + \
               a3*vonmises.pdf(x, k3, l3) + \
               a4*vonmises.pdf(x, k4, l4) + c


def vm4_pdf(x,
            l1, l2, l3, l4,
            k1, k2, k3, k4,
            a1, a2, a3, a4, c):
    return a1*vonmises.pdf(x, k1, l1) + \
           a2*vonmises.pdf(x, k2, l2) + \
           a3*vonmises.pdf(x, k3, l3) + \
           a4*vonmises.pdf(x, k4, l4) + c


def primary_angles(angles, m=4, bins=720, ret_hist=False):
    angles = angles[angles != 0].ravel()
    h, t = np.histogram(angles, bins, (0, tau), True)
    t = 0.5*(t[1:] + t[:-1])

    l0 = tuple((np.arange(0, tau, tau/m)+t[h.argmax()]) % tau)
    k0 = (100.,) * m
    a0 = (.02,) * m
    c0 = 1e-3,
    guess = l0 + k0 + a0 + c0
    vm_fit = curve_fit(vm4_pdf, t, h, guess)[0]
    l = vm_fit[:m]
    k = vm_fit[m:2*m]
    a = vm_fit[2*m:3*m]
    c = vm_fit[-1]
    if ret_hist:
        return l, k, a, c, h, t
    return l, k, a, c


def get_gdata(locdir, ns):
    return {'n'+str(n): np.load(locdir+'n'+str(n)+'_GR.npz') for n in ns}


def find_gpeaks(ns, locdir, binmax=258):
    """ find_gpeaks(ns, locdir, binmax)
        finds peaks and valleys in g(r) curve
        takes:
            ns, list of densities to analyse
            locdir, local directory for data
            binmax, the max bin number, hopefully temporary problem
        returns:
            peaks,  list of [list of peaks and list of valleys]
                    in format given by peakdetect.py
    """
    import peakdetect as pk
    ns = np.array([8, 16, 32, 64, 128, 192, 256, 320, 336,
                   352, 368, 384, 400, 416, 432, 448])
    binmax = 258
    gdata = get_gdata(locdir, ns)
    peaks, maxima, minima = {}, {}, {}
    for k in gdata:
        extrema = pk.peakdetect(gdata[k]['g'][:binmax]/22.0,
                                gdata[k]['rg'][:binmax]/22.,
                                lookahead=2., delta=1e-4)
        peaks[k] = extrema
        maxima[k] = np.asarray(extrema[0])
        minima[k] = np.asarray(extrema[1])
    return peaks


def plot_gpeaks(peaks, gdata, pksonly=False, hhbinmax=258):
    """plots locations and/or heights of peaks and/or valleys in g(r)

    takes:
        peaks,  list of peaks from output of find_gpeaks()
        gdata,  g(r) arrays, loaded from get_gdata()
        binmax, the max bin number, hopefully temporary problem
    side affects:
        creates a figure and plots things
    returns:
        nothing
    """
    pl.figure()
    for k in peaks:
        try:
            pl.plot(gdata[k]['rg'][:binmax]/22., gdata[k]['g'][:binmax]/22.,
                    ',-', label=k)
            pl.scatter(*np.asarray(peaks[k][0]).T, marker='o', label=k,
                       c=pl.cm.jet((int(k[1:])-200)*255/300))
            pl.scatter(*np.asarray(peaks[k][1]).T,
                       marker='x', label=k)  # minima

            if pksonly:
                pks = np.asarray(peaks[k]).T    # if peaks already just maxima
            else:
                pks = np.asarray(peaks[k][0]).T     # gets just maxima
            try:
                pkpos = pks[0]
            except:
                print "pks has wrong shape for k =", k
                print pks.shape
                continue
            pl.scatter(int(k[1:])*np.ones_like(pkpos), pkpos,
                       marker='*', label=k)  # maxima
        except:
            print "failed for", k
            continue
    pl.legend()


def apply_hilbert(a, sig=None, full=False):
    """ Attempts to apply hilbert transform to a signal about a mean.
        First, smooth the signal, then subtract the smoothed signal.
        Apply hilbert to the residual, and add the smoothed signal back in.
    """
    assert a.ndim == 1, "Only works for 1d arrays"
    if sig is None:
        sig = a.size/10.
    if sig:
        a_smoothed = gaussian_filter(a, sig, mode='reflect')
    else:
        a_smoothed = a.mean()
    h = hilbert(a - a_smoothed)
    if full:
        return h, a_smoothed
    else:
        return np.abs(h) + a_smoothed


def gpeak_decay(peaks, f, pksonly=False):
    """ gpeak_decay(peaks, f)
    fits curve to the peaks in g(r)
    takes:
        peaks,  list of peak/valley positions and heights
        f,      the function for the curve, right now either:
                    exp_decay or powerlaw

    returns:
        popt, a tuple of parameters for f
        pcov, their covariances
    """
    if pksonly is True:
        maxima = peaks
    else:
        maxima = {k: np.asarray(peaks[k][0]) for k in peaks}
        minima = {k: np.asarray(peaks[k][1]) for k in peaks}
    popt, pcov = {}, {}
    pl.figure()
    for k in peaks:
        maximak = maxima[k].T
        print "k: f, maximak"
        print k, f, maximak
        if len(maxima[k]) > 1:
            popt[k], pcov[k] = curve_fit(f, maximak[0], maximak[1])
            fitrange = np.arange(min(maximak[0]), max(maximak[0]), .05)
            pl.plot(fitrange, f(fitrange, *popt[k]), '--', label='fit '+k)
        else:
            print "maximak empty:", maximak
    return popt, pcov


def gauss(x, a=1., x0=0., sig=1., c=0.):
    x2 = np.square(x-x0)
    s2 = sig*sig
    return c + a*np.exp(-x2/s2)


def fit_peak(xdata, ydata, x0, y0=1., w=helpy.S_slr, form='gauss'):
    l = np.searchsorted(xdata, x0-w/2)
    r = np.searchsorted(xdata, x0+w/2)
    x = xdata[l:r+1]
    y = ydata[l:r+1]
    form = form.lower()
    if form.startswith('p'):
        c = poly.polyfit(x, y, 2)
        loc = -0.5*c[1]/c[2]
        height = c[0] - 0.25 * c[1]**2 / c[2]
    elif form.startswith('g'):
        c, _ = curve_fit(gauss, x, y, p0=[y0, x0, w, 0])
        loc = c[1]
        height = c[0] + c[3]
    return loc, height, x, y, c


def exp_decay(t, sig=1., a=1., c=0):
    """ exp_decay(t, sig, a, c)
        exponential decay function for fitting

        Args:
            t,  independent variable
        Params:
          sig,  decay constant
            a,  prefactor
            c,  constant offset

        Returns:
            value at t
    """
    return c + a*np.exp(-t/sig)


def log_decay(t, a=1, l=1., c=0.):
    return c - a*np.log(t/l)


def powerlaw(t, b=1., a=1., c=0):
    """ powerlaw(t, b, a, c)
        power law function for fitting
                                      -b
        powerlaw(t, b, a, c) = c + a t

        Args:
            t,  independent variable
        Params:
            b,  exponent (power)
            a,  prefactor
            c,  constant offset
        Returns:
            power law value at t
    """
    return c + a * np.power(t, -b)


decays = {'exp': exp_decay, 'pow': powerlaw}


def chained_power(t, d1, d2, b1=1, b2=1, c1=0, c2=0, ret_crossover=False):
    p1 = powerlaw(t, b1, d1, c1)
    p2 = powerlaw(t, b2, d2, c2)
    cp = np.maximum(p1, p2)
    if ret_crossover:
        ct = t[np.abs(p1-p2).argmin()]
        print ct
        ct = np.power(d1/d2, -np.reciprocal(b2-b1))
        print ct
        return cp, ct
    else:
        return cp


def shift_power(t, tc=0, b=1, a=1, c=0):
    return powerlaw(tc-t, b, a, c)


def critical_power(t, f, tc=0, b=None, a=None, c=None):
    """ Find critical point with powerlaw divergence
    """
    lp = sum(i is not None for i in [tc, b, a, c])
    p0 = [tc, b, a, c][:lp] + [0, 1, 1, 0][lp:]
    popt, pcov = curve_fit(shift_power, t, f, p0=p0[:lp], sigma=np.log(f))
    return popt

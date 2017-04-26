#!/usr/bin/env python
# encoding: utf-8
"""Various statistical correlation functions for use in analyzing granular
particle dynamics and collective structure.

Copyright (c) 2012--2017 Lee Walsh, Department of Physics, University of
Massachusetts; all rights reserved.
"""

from __future__ import division

from math import sqrt, pi
from cmath import phase
from itertools import combinations

import numpy as np
from scipy import ndimage, signal, stats
from scipy.spatial import distance, Voronoi, Delaunay, cKDTree as KDTree
from scipy.optimize import curve_fit
from skimage.morphology import binary_dilation, disk as skdisk

import helpy

ss = helpy.S_slr  # side length of square in pixels
rr = helpy.R_slr  # radius of disk in pixels

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
        max_sep = 0
        if len(positions) < 1e4:
            max_sep = distance.cdist(positions, positions).max()/2
        radius = max(max_sep, d.max()) + ss/2
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
    assert n >= 2, "Cannot have pairs inside list of 1"
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
    # faster than squareform(distance.pdist(positions)) wtf
    r = distance.cdist(positions, positions)
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
    """determine and rotate reference frame by the primary angles"""
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
    """calculate the 2d pair distribution function g(x, y)"""
    if margin < ss:
        margin *= ss
    center = 0.5*(positions.max(0) + positions.min(0))
    d = helpy.dist(positions, center)   # distances to center
    dmask = d < d.max() - margin
    r = distance.cdist(positions, positions[dmask])  # .ravel()
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
    f = binary_dilation(f, skdisk(ss/2))
    return fft2(f, overwrite_x=True)


def orient_op(orientations, m=4, positions=None, margin=0,
              ret_complex=True, do_err=False, globl=False, locl=False):
    """orient_op(orientations, m=4, positions=None, margin=0,
                 ret_complex=True, do_err=False, globl=False, locl=False)

       calculate the global m-fold particle orientational order parameter

                1   N    i m theta
        Phi  = --- SUM e          j
           m    N  j=1
    """
    if not (globl or locl):
        globl = True
        locl = orientations.ndim == 2
    np.mod(orientations, tau/m, orientations)
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
    """ Find the smallest m-fold difference between angles (in radians)

        parameters
        ----------
        i, j : one or two arrays of angles. If a single array, take difference
            along last axis. If two arrays, difference is between the two.
        m : degree of rotational symmetry. That is, the branch cut will be taken
            at 2*pi/m. Default is 1. If m = 0, simply returns the difference.
        sign : whether to keep negative sign when i < j, or take absolute value

        returns
        -------
        diffs : array of m-fold differences. If two arrays are given, shape is
            unchanged, otherwise shape is reduced by one in last dimension.
    """
    if j is None and i.shape[-1] == 2:
        i, j = i.T
    diff = np.diff(i, axis=1) if j is None else j - i

    if m == 0:
        return diff
    m = tau/m
    diff = (diff + m/2) % m - m/2
    return diff if sign else np.abs(diff)


def bin_sum(r, f, bins=10):
    """Binned sum of function f(r)

    Parameters:
        r:      independent variable to be binned over
        f:      function to be summed
        bins:   (default 10): number of bins or bin edges `len(nbins)+1`

    Returns:
        total:  the total value per bin
        count:  number of values summed per bin (histogram)
        bins:   bin edges
    """
    multi = isinstance(f, tuple)
    if bins is 1:
        if r.dtype.kind not in 'iu':
            assert np.allclose(r, np.around(r)), 'need integer array for bins=1'
            print 'converting to int array'
            r = r.astype(int)
        count = np.bincount(r)
        if multi:
            total = [np.bincount(r, weights=fi) for fi in f]
        else:
            total = np.bincount(r, weights=f)
        bins = np.arange(len(count)+1)
    count, bins = np.histogramdd(r, bins)
    if multi:
        total = [np.histogramdd(r, bins, weights=fi)[0] for fi in f]
    else:
        total = np.histogramdd(r, bins, weights=f)[0]
    if len(bins) == 1:
        bins = bins[0]
    return total, count.astype(int), bins


def bin_average(r, f, bins=10):
    """Binned average of function f(r)

    Parameters:
        r:      independent variable to be binned over
        f:      function to be averaged
        bins:   (default 10): number of bins or bin edges `len(nbins)+1`

    Returns:
        avgs:   the mean value per bin
        bins:   bin edges
    """
    multi = isinstance(f, tuple)
    total, count, bins = bin_sum(r, f, bins)
    average = [t/count for t in total] if multi else total/count
    return average, bins


def autocorr(f, side='right', cumulant=True, norm=1, mode='same',
             verbose=False, reverse=False, ret_dx=False):
    """ autocorrelate f with itself

        The auto-correlation of function f returns
            <f(x) f(x - dx)>
        averaged over x, as a function of dx

        See also `crosscorr(f, g, ...)`

        f:      1d array, as function of x
        side:   'right' returns only dx > 0, (x' < x)
                'left'  returns only dx < 0, (x < x')
                'center' or 'both'  returns entire correlation
        cumulant: 'initial' or 'mean' to subtract initial value or mean
        norm:   normalize by the correlation at no shift, i.e. <f(x) g(x) >
        mode:   passed to scipy.signal.correlate, has little effect here, but
                returns shorter correlation array
    """
    return crosscorr(f, f, side=side, cumulant=cumulant, norm=norm, mode=mode,
                     verbose=verbose, reverse=reverse, ret_dx=ret_dx)


def crosscorr(f, g, side='both', cumulant=False, norm=False, mode='same',
              verbose=False, reverse=False, ret_dx=False):
    """ cross correlate functions f and g

        The cross-correlation of f and g returns
            <f(x) g(x - dx)>
        averaged over x, as function of dx

        parameters
        ----------
        f, g:       1d arrays, as function of x, with same lengths
        side:       'right' returns only dx > 0, (x' < x)
                    'left'  returns only dx < 0, (x < x')
                    'center' or 'both' returns entire correlation
        cumulant:   'initial' to subtract initial value (of f) or
                    'mean' to subtract mean (of both, same as one or the other)
        norm:       normalize by the correlation at no shift, i.e. <f(x) g(x) >
        mode:       passed to scipy.signal.correlate, has little effect here.
        ret_dx:     if True, return the dx shift between f and g, that is,
                    if we are looking at <f(x) g(x')> then dx = x - x'
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

    correlator = signal.convolve if reverse else signal.correlate
    if f.ndim == g.ndim == 2:
        # apply the correlator function to each pair of columns in f, g
        c = np.stack([correlator(*fgi, mode=mode) for fgi in zip(f.T, g.T)])
    elif f.ndim == g.ndim == 1:
        c = correlator(f, g, mode=mode)
    else:
        raise ValueError("arrays must have same dimensionality of 1 or 2")
    if verbose and (f is g):
        maxi = c.argmax()
        assert maxi == m, ("autocorrelation not peaked at 0: "
                           "max ({}) not at m ({})").format(maxi, m)

    # divide by overlap
    n = np.concatenate([np.arange(l - m, l), np.arange(l, m - (L - l), -1)])
    if verbose:
        overlap = correlator(np.ones(l), np.ones(l), mode=mode).astype(int)
        if verbose > 1:
            print n
            print '      n: {}\noverlap: {}'.format(n, overlap)
        msg = "overlap miscalculated:\n\t{}\n\t{}"
        assert np.allclose(n, overlap), msg.format(n, overlap)
        assert n[m] == l, "overlap normalizer not l at m"
    c /= n
    c = c.T
    if verbose:
        msg = ("normalization calculations don't all match: "
               "c[m]: {}, np.dot(f, g): {}, c.max(): {}")
        fgs = c[m], np.dot(f, g)/len(f), c[m]  # c.max()
        if verbose > 1 and norm in (0, 1):
            print ("subtracting", "normalizing by")[norm], "scaler:", fgs[0]
        assert np.allclose(fgs[0], fgs), msg.format(*fgs)

    if side == 'both':
        side = 'center'
    if isinstance(cumulant, bool):
        cumulant = 'mean'*cumulant  # cumulant=True --> 'mean'

    if cumulant.startswith('init'):
        c -= limited_mean(f*g, 'init', side)
    elif cumulant.startswith('mean'):
        c -= limited_mean(f, 'final', side) * limited_mean(g, 'init', side)

    if norm:
        c /= c[m]
    elif verbose > 1:
        print 'central value:', c[m]

    if ret_dx:
        if side == 'center':
            return np.arange(-m, L-m), c
        elif side == 'left':
            # return np.arange(0, -m-1, -1), c[m::-1]
            return np.arange(-m, 1), c[:m+1]
        elif side == 'right':
            return np.arange(0, L-m), c[m:]

    if side == 'center':
        return c
    elif side == 'left':
        return c[m::-1]
    elif side == 'right':
        return c[m:]


def limited_mean(f, end, side='centered'):
    """ Mean at single point of some correlation function

    Say we want to calculate the correlation of a difference in f vs initial g:

        C(x) = <[f(x₀ + x) - f(x₀)] g(x₀)>
             = <f(x₀ + x) g(x₀)> - <f(x₀) g(x₀)>

    The first term is a simple cross-correlation and can be calculated by the
    function `crosscorr`. While the second term appears to be independent of x,
    and looks just like the mean of the product of the arrays, or the first term
    evaluated at x=0, it is in fact distinct. It is the mean only of a limited
    part of the array, constrained by the limits of the first term. Consider the
    following sum:
        C(x) = Σx₀ {f(x₀ + x) g(x₀) - f(x₀) g(x₀)}
    As finite arrays, f and g are only defined on the domain [0, L), so the
    evaluation points of f and g over the domain of the sum over x₀ must obey
        0 ≤ x₀ + x < L
        0 ≤ x₀ < L
    so for the sum, x₀ must range over
        [ 0, L - x) for x ≥ 0
        [-x, L)     for x < 0


    parameters
    ----------
    f : single array to average over first axis
    end : a string, which end of limit to average over
        'initial' average over first n elements, <f(x₀)>
        'final' average over last n elements, <f(x₀ + x)>
        'both' do both, stacked along axis 1, [[<f(x₀)>, <f(x₀ + x)>]]
        'sum' do both and add them, <f(x₀)> + <f(x₀ + x)>
    side : a string, direction of correlation (see `crosscorr`)
        'right' positive side (0 ≤ x < L)
        'left' negative side (-L < x ≤ 0)
        'center' center at 0 (-L/2 ≤ x < L/2)
    """
    f = np.asarray(f)
    L = len(f)
    r = slice(None, None, -1)
    if end.startswith('s'):
        f = f + f[r]
    elif side.startswith('c') or end.startswith('b'):
        f = np.stack([f[r], f] if side.startswith('l') else [f, f[r]], 1)
    elif side.startswith('r') and end.startswith('f'):
        f = f[r]
    elif side.startswith('l') and end.startswith('i'):
        f = f[r]
    n = np.arange(1, L + 1).reshape(L, *[-1]*(f.ndim-1))
    m = np.cumsum(f, 0) / n
    if side.startswith('r'):
        return m[r]
    elif side.startswith('l'):
        return m
    elif side.startswith('c'):
        s = (slice(-1 - L//2, -1), slice(None, -L//2 - 1, -1))
        if end.startswith('init'):
            s = zip(s, [1, 0])
        elif end.startswith('both'):
            s = zip(s, [r, slice(None)])
        elif end.startswith('fin'):
            s = zip(s, [0, 1])
        return np.concatenate([m[s[0]], m[s[1]]])


def msd(xs, ret_taus=False, ret_vector=False):
    """ calculate the mean squared displacement

        msd = < [x(t₀ + ) - x(t₀)]**2 >
            = < x(t₀ + tau)**2 > + < x(t₀)**2 > - 2 * < x(t₀ + tau) x(t₀) >
            = <xx> + <x₀x₀> - 2*<xx₀>

        The first two terms are, respectively, the average over all initial and
        final values of x² for valid t₀ given tau. That is, terms at t = t₀ and
        t = t₀ + tau are valid for t₀ ∈ [0, T - tau). The mean over these limits
        are given by `limited_mean`

        Note:
        * only accepts the positions in 1 or 2d array (no data structure)
        * time must be axis 0, but any number of dimensions is allowed (axis 1)
        * can only do dt0 = dtau = 1
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
    xx0 = autocorr(xs, side='right', cumulant=False, norm=False, mode='full',
                   verbose=False, reverse=False, ret_dx=False)

    # First terms are the initial mean and final mean of x²
    x2s = limited_mean(xs*xs, end='sum', side='right')

    out = x2s - 2*xx0
    if not ret_vector or ret_vector.startswith('disp'):
        out = out.sum(1)  # straight sum over dimensions (x² + y² + ...)

    return np.column_stack([np.arange(T), out]) if ret_taus else out


def msd_correlate(x, y, n, corr_args):
    """calculate the various terms in the msd correlation"""
    xy = x * y
    x_yn = crosscorr(x, y*n, **corr_args)
    xy_n = crosscorr(xy, n, **corr_args)
    xyn_ = limited_mean(xy*n, end='init', side=corr_args['side'])
    return xy_n - 2*x_yn + xyn_


def msd_body(xs, os, ret_taus=False):
    """calculate mean squared displacement in body frame"""
    xs = np.asarray(xs)
    d = xs.ndim
    if d == 1:
        return msd(xs, ret_taus)
    elif d == 2:
        T, d = xs.shape
        assert d == 2
    else:
        msg = "can't handle xs.ndims > 2. xs.shape is {}"
        raise ValueError(msg.format(xs.shape))

    corr_args = {'side': 'right', 'cumulant': False, 'mode': 'full'}
    ns = np.column_stack([np.cos(os), np.sin(os)])
    ps = ns[:, ::-1]
    ys = xs[:, ::-1]
    progress = msd_correlate(xs, xs, ns*ns, corr_args)
    diversion = msd_correlate(xs, xs, ps*ps, corr_args)
    crossterms = msd_correlate(xs, ys, ns*ps, corr_args)
    progress += crossterms
    diversion -= crossterms
    taus = [np.arange(T)] if ret_taus else []
    return np.column_stack(taus + [progress.sum(1), diversion.sum(1)])


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
    r = distance.pdist(positions[loc_mask])
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
        distances = distance.cdist(positions, positions)[ix, neighbors]
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
    """calculate area of polygon"""
    area = 0.0
    n = len(corners)
    for i in xrange(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    return abs(area) / 2.0


def density(positions, method, vor=None, **neigh):
    """calculste density by various methods: ('vor', 'dist', 'inv')"""
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
    """calculate density by gaussian kernel"""
    dens = np.zeros(extent, float)
    indices = np.around(positions).astype('u4').T
    dens[indices] = unit_length*unit_length
    if scale is None:
        scale = 2*unit_length
    ndimage.gaussian_filter(dens, scale, mode='constant')


def voronoi_density(pos_or_vor):
    """calculate density by voronoi area"""
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


def conjmul(a, b):
    """conjugate multiplication a* times b
    """
    return np.conj(a) * b


def pair_angle_corr(positions, psims, rbins=10):
    """calculate pair-angle correlation"""
    return radial_correlation(positions, psims, rbins, correland=conjmul)


def radial_correlation(positions, values, bins=10, correland='*', do_avg=True):
    """radial_correlation(positions, values, bins=10, correland='*')

    Correlate between all pairs of values, binned by radial separation distance.

    Parameters
    positions:  position of each value (N, d)
    values:     values to correlate (N, ...)
    bins:       ultimately passed to np.histogram
    correland:  function of two values to give the integrand of the correlation

    Returns
    correlation:    the correlated values per bin
    bins:       as returned by np.histogram
    """
    n = len(positions)
    assert n >= 2, "must have at least two items"
    multi = isinstance(values, tuple)
    assert n == len(values[0] if multi else values), "lengths do not match"

    ij = pair_indices(n, True).T
    rij = distance.pdist(positions)
    if not np.isscalar(bins) and rij.min() > np.max(bins):
        return 0, 0, bins
    if correland == '*':
        correland = np.multiply
    if multi:
        vij = tuple([correland(*v[ij]) for v in values])
    else:
        vij = correland(*values[ij])
    bin_func = bin_average if do_avg else bin_sum
    return bin_func(rij, vij, bins)


def site_mean(positions, values, bins=10, coord='xy'):
    """site_mean(positions, values, bins=10, coord='xy')

    Bin and average the values as a function of their locations.

    Parameters
    positions:  position of each value (N, d)
    values:     values to average (N, ...)
    bins:       ultimately passed to np.histogram
    coord:      choice of 'xy' == 'cartesian', 'polar', 'radial'
    """
    if coord.startswith(('p', 'r')):
        if np.all(positions.min(0) > 0):
            positions = positions - positions.mean(0)
        x, y = positions.T
        positions = np.hypot(y, x)  # radial coordinate r
        if coord.startswith('p'):
            positions = np.stack([positions, np.arctan2(y, x)], 1)
    elif not coord.startswith(('x', 'c')):
        raise ValueError("Unknown coordinate system `coord={}`".format(coord))
    return bin_average(positions, values, bins)


class vonmises_m(stats.rv_continuous):
    """generate von Mises distribution for any m"""

    def __init__(self, m):
        self.shapes = ''
        for i in range(m):
            self.shapes += 'k%d,l%d' % (i, i)
        self.shapes += ',scale'
        stats.rv_continuous.__init__(self, a=-np.inf, b=np.inf, shapes=self.shapes)
        self.numargs = 2*m

    def _pdf(self, x, *lks):
        """probability distribution function"""
        print 'lks', lks
        locs, kappas = lks[:len(lks)/2], lks[len(lks)/2:]
        print 'x', x
        print 'locs', locs
        print 'kapps', kappas
        # return np.sum([stats.vonmises.pdf(x, l, k)
        #                for l, k in zip(locs, kappas)], 0)
        ret = np.zeros_like(x)
        for l, k in zip(locs, kappas):
            ret += stats.vonmises.pdf(x, l, k)
        return ret / len(locs)


class vonmises_4(stats.rv_continuous):
    """generate von Mises distribution for m = 4"""

    def __init__(self):
        stats.rv_continuous.__init__(self, a=-np.inf, b=np.inf)

    def _pdf(self, x,
             l1, l2, l3, l4,
             k1, k2, k3, k4,
             a1, a2, a3, a4, c):
        """probability distribution function"""
        return a1*stats.vonmises.pdf(x, k1, l1) + \
               a2*stats.vonmises.pdf(x, k2, l2) + \
               a3*stats.vonmises.pdf(x, k3, l3) + \
               a4*stats.vonmises.pdf(x, k4, l4) + c


def vm4_pdf(x,
            l1, l2, l3, l4,
            k1, k2, k3, k4,
            a1, a2, a3, a4, c):
    """calculate the probability distribution function for m = 4 von Mises"""
    return a1*stats.vonmises.pdf(x, k1, l1) + \
           a2*stats.vonmises.pdf(x, k2, l2) + \
           a3*stats.vonmises.pdf(x, k3, l3) + \
           a4*stats.vonmises.pdf(x, k4, l4) + c


def primary_angles(angles, m=4, bins=720, ret_hist=False):
    """estimate the m primary orientation angles from all angles"""
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
    """load a saved g(r) array"""
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
        a_smoothed = ndimage.gaussian_filter(a, sig, mode='reflect')
    else:
        a_smoothed = a.mean()
    h = signal.hilbert(a - a_smoothed)
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

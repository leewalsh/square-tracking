#!/usr/bin/env python
# encoding: utf-8
"""Various functions for functions, such as common mathematical functions,
curve-fitting, and functionals.

Copyright (c) 2012--2017 Lee Walsh, Department of Physics, University of
Massachusetts; all rights reserved.
"""

from __future__ import division

import itertools as it
from collections import namedtuple

import numpy as np
from numpy.polynomial import polynomial
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

import helpy

pi = np.pi
twopi = 2*pi
rt2 = np.sqrt(2)


def poly_exp(x, gamma, amp, *coeffs):
    """ exponential decay with a polynomial decay scale

                     gamma
                 - x
          -----------------------
          c0 + c1 x + c2 x² + ...
    a * e
    """
    d = polynomial.polyval(x, coeffs or (1,))
    return amp * np.exp(-x**gamma/d)


def vary_gauss(arr, sig=1, verbose=False):
    """gaussian filter with variable width

    parameters
    ----------
    arr : array to be smoothed
    sig : gaussian width, which may be any of the following types:
        - scalar:     width = sig * x
        - tuple:      width = sig[0] + sig[1]*x + ...
        - callable:   width = sig(x)
        - arraylike:  width = sig


    """
    n = len(arr)
    out = np.empty_like(arr)

    if np.isscalar(sig):
        sig *= np.arange(n)
    elif isinstance(sig, tuple):
        sig = polynomial.polyval(np.arange(n), sig)
    elif callable(sig):
        sig = sig(np.arange(n))
    elif hasattr(sig, '__getitem__'):
        assert len(arr) == len(sig)
    else:
        raise TypeError('`sig` is neither callable nor arraylike')
    for i, s in enumerate(sig):
        # build the kernel:
        w = round(2*s)  # kernel half-width, must be integer
        if s == 0:
            s = 1
        k = gauss(np.arange(-w, w + 1, dtype=float), sig=s)

        # slice the array (min/max prevent going past ends)
        al = max(i - w, 0)
        ar = min(i + w + 1, n)
        ao = arr[al:ar]

        # and the kernel
        kl = max(w - i, 0)
        kr = min(w - i + n, 2*w + 1)
        ko = k[kl:kr]
        out[i] = np.dot(ao, ko)/ko.sum()

    return out


def fit_peak(xdata, ydata, x0, y0=1., w=helpy.S_slr, form='gauss'):
    """fit a peak (gaussian or parabola) to a high point in a curve"""
    l, r = np.searchsorted(xdata, [x0-w/2, x0+w/2])
    x = xdata[l:r+1]
    y = ydata[l:r+1]
    form = form.lower()
    if form.startswith('p'):
        c = polynomial.polyfit(x, y, 2)
        loc = -0.5*c[1]/c[2]
        height = c[0] - 0.25 * c[1]**2 / c[2]
    elif form.startswith('g'):
        c, _ = curve_fit(gauss, x, y, p0=[y0, x0, w, 0])
        loc = c[1]
        height = c[0] + c[3]
    return loc, height, x, y, c


def exp_decay(t, sig=1., a=1., c=0):
    """exponential decay function
                                           - t
                                          -----
                                           sig
       exp_decay(t, sig, a, c) = c + a * e
    """
    return c + a*np.exp(-t/sig)


def log_decay(t, a=1, l=1., c=0.):
    """logarithmic decay function

                                            t
       log_decay(t, a, l, c) = c - a * log ---
                                            l
    """
    return c - a*np.log(t/l)


def powerlaw(t, b=1., a=1., c=0):
    """power law decay function
                                     -b
       powerlaw(t, b, a, c) = c + a t

    """
    return c + a * np.power(t, -b)


decays = {'exp': exp_decay, 'pow': powerlaw}


def chained_power(t, d1, d2, b1=1, b2=1, c1=0, c2=0, ret_crossover=False):
    """double power law decay, constant slows to smaller value at crossover time
    """
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


def shift_power(t, tc=0, a=1, b=1, c=0, dt=0):
    """power law decay function with (protected to keep positive) timeshift"""
    tshift = np.sqrt((tc-t)**2 + dt**2) if dt else tc - t
    return powerlaw(tshift, b, a, c)


def critical_power(t, f, tc=0, a=None, b=None, c=None,
                   dt=None, df=None, abs_df=False):
    """ Find critical point with powerlaw divergence
    """
    p0 = [i for i in [tc, a, b, c] if i is not None]
    if dt:
        func = lambda *args: shift_power(*args, **dict(dt=dt))
    else:
        func = shift_power
    return curve_fit(func, t, f, p0=p0, sigma=df, absolute_sigma=abs_df)


def gauss(x, a=1., x0=0., sig=1., c=0.):
    """gaussian function (e.g., the pdf of a normal distribution)

                                        - (x - x0)²
                                        -----------
                                            sig²
    gauss(x, a, x0, sig, c) = c + a * e
    """
    x2 = np.square(x-x0)
    s2 = sig*sig
    return c + a*np.exp(-x2/s2)


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


def interp_nans(f, x=None, max_gap=10, inplace=False, verbose=False):
    """ Replace nans in function f(x) with their linear interpolation

        parameters
        ----------
        f : 1d or 2d array with some nans
        x : x-values for array f (in case non-uniform)
        max_gap : upper limit for number of consecutive nans to interpolate.
            nans in a consecutive run of length over max_gap will remain.
        inplace : whether to overwite nans in f, or to return a copy.
        verbose : whether to print information about gaps interpolated.

        returns
        -------
        interpolated : f (itself if inplace otherwise a copy) with nans replaced
            by the interpolated values (may still have nans).
    """
    n = len(f)
    if n < 3:
        return f
    if f.ndim == 1:
        nans = np.isnan(f)
    elif f.ndim == 2:
        nans = np.isnan(f[:, 0])
    else:
        raise ValueError("Only 1d or 2d")
    if np.count_nonzero(nans) in (0, n):
        return f
    ifin = (~nans).nonzero()[0]
    nf = len(ifin)
    if nf < 2:
        return f
    if not inplace:
        f = f.copy()
    # to detect nans at either endpoint, pad before and after
    bef, aft = int(nans[0]), int(nans[-1])
    if bef or aft:
        bfin = np.empty(nf+bef+aft, int)
        if bef:
            bfin[0] = -1
        if aft:
            bfin[-1] = len(f)
        bfin[bef:-aft or None] = ifin
    else:
        bfin = ifin
    gaps = np.diff(bfin) - 1
    if verbose:
        print '\t      interp {:7} {:8} {:10}'.format(
            '{}@{}'.format(gaps.max(), gaps.argmax()),
            np.count_nonzero(gaps), gaps.sum())
    inan = ((gaps > 0) & (gaps <= max_gap)).nonzero()[0]
    if len(inan) < 1:
        return f
    gaps = gaps[inan]
    inan = np.repeat(inan, gaps)
    inan = np.concatenate(map(range, gaps)) + bfin[inan] + 1
    xnan, xfin = (inan, ifin) if x is None else (x[inan], x[ifin])
    if not inplace:
        f = f.copy()
    for c in f.T if f.ndim > 1 else [f]:
        c[inan] = np.interp(xnan, xfin, c[ifin])
    return f


def fill_gaps(f, x, max_gap=10, ret_gaps=False, verbose=False):
    """ fill gaps in a function f(x)

            f = [9, 2, 5, 0]  --->  [9, 2, *, *, 5, *, *, 0]
            x = [0, 1, 4, 7]  --->  [0, 1, 2, 3, 4, 5, 6, 7]

        where * is the representation of np.nan in f.dtype

        parameters
        ----------
        f : values at gapped x.
        x : array with missing values, must be linear.
        max_gap : upper limit for size of gap to fill. if largest gap exceeds
            this, return (None, x[, gaps])
        ret_gaps : whether to return gap sizes found
        verbose : whether to print information about gaps filled.

        returns
        -------
        filled_f : expanded f with gaps filled by np.nan (or equivalent for f)
        filled_x : expanded x with all gaps interpolated
        [gaps] : (if ret_gaps) array of the sizes of each gap.
    """
    gaps = np.diff(x) - 1
    ret_gaps = (gaps,) if ret_gaps else ()
    mx = gaps.max()
    if not mx:
        if verbose > 1:
            print 'no gaps'
        return (f, x) + ret_gaps
    elif mx > max_gap:
        if verbose:
            print 'too large'
        if ret_gaps:
            return (None, x) + ret_gaps
    if verbose:
        print 'filled'
    gapi = gaps.nonzero()[0]
    gaps = gaps[gapi]
    gapi = np.repeat(gapi, gaps)
    filler = np.full(1, np.nan, f.dtype)
    missing = np.concatenate(map(range, gaps)) + x[gapi] + 1
    f = np.insert(f, gapi+1, filler)
    x = np.insert(x, gapi+1, missing)
    return (f, x) + ret_gaps


def cumtrapz(y, x=None, dx=None, axis=-1):
    if x is None:
        x = np.arange(len(y)) * dx
    elif dx is None:
        dx = x[1:] - x[:-1]
    out = np.cumsum(dx * (y[1:] + y[:-1])/2)
    return np.concatenate([[0], out])


def der_test(f, dx=None, x=None, fprime=None, **kwargs):
    """run der(f, **kwargs) with some different options and plot"""

    np.random.seed(5829)
    if dx is None and x is None:
        dx = 1
        x = np.arange(10 if isinstance(f, basestring) else len(f))
    elif x is None:
        x = dx * np.arange(10/dx if isinstance(f, basestring) else len(f))
    elif dx is None:
        dx = x[1] - x[0]

    fig, ax = plt.subplots(figsize=(12, 9))
    if f == 'step':
        f = np.ones_like(x)
        f[:len(f)//2] = 0
        fprime = np.zeros_like(f)
        fprime[len(f)//2] = 1/dx
    elif f == 'lin':
        f, fprime = x, np.ones_like(x)
    elif f == 'kicks':
        N = 10
        tkick = np.arange(N + 1)
        t = np.arange(0, N, dx)
        vkick = np.random.normal(loc=0, scale=1, size=N)
        xkick = np.concatenate([[0], np.cumsum(vkick)])
        x = np.interp(t, tkick, xkick)
        x, f = t, x
        ax.plot(tkick, xkick, '-',
                lw=4, c='gray', label='f')
        ax.step(tkick, np.append(vkick, [None]), where='post',
                marker='*', ls='--', c='gray', ms=16, label="f'")

    widths = (1, 2) + tuple(np.arange(.5, 1.3, .1)[::-1])

    ax.plot(x, f - f[0], '-', lw=4, c='k', label='f')
    if fprime is not None:
        if fprime.ndim == 2:
            xprime, fprime = fprime
        else:
            xprime = x
        ax.plot(xprime, fprime, '*--', c='k', ms=16, label="f'")

    colors = map('C{}'.format, xrange(len(widths)))
    for width, color in zip(widths, colors):
        print 'width:', width
        df = der(f, x=x, iwidth=width)
        idf = cumtrapz(df, x, dx)
        label = '({})'.format(width)
        ax.plot(x, df, ('o' if isinstance(width, float) else ' x+'[width])+':',
                c=color, label='der' + label)
        ax.plot(x, idf, '--', c=color, label='np.trapz' + label)
    ax.legend(loc='best')
    fig.tight_layout()


def der(f, dx=None, x=None, xwidth=None, iwidth=None, order=1, min_scale=1):
    """ Take a finite derivative of f(x) using convolution with gaussian

    A function convolved with the derivative of a gaussian kernel gives the
    derivative of the function convolved with the integral of the kernel of a
    gaussian kernel. For any convolution:
        (f * g)' = f * g' = g * f'
    so we start with f and g', and return g and f', a smoothed derivative.

    Optionally can not smooth by giving width 0.

    parameters
    ----------
    f : an array to differentiate
    xwidth or iwidth : smoothing width (sigma) for gaussian.
        use iwidth for index units, (simple array index width)
        use xwidth for the physical units of x (x array is required)
        use 0 for no smoothing.
    x or dx : required for normalization
        if x is provided, dx = np.diff(x)
        otherwise, a scalar dx is presumed
        if dx=1, use a simple finite difference with np.diff
        if dx>1, convolves with the derivative of a gaussian, sigma=dx
    order : how many derivatives to take
    min_scale : the smallest physical scale involved in index units. e.g., fps.

    returns
    -------
    df_dx : the `order`th derivative of f with respect to x
    """
    if dx is None and x is None:
        dx = 1
    elif dx is None:
        dx = x.copy()
        dx[:-1] = dx[1:] - dx[:-1]
        assert dx[:-1].min() > 1e-6, ("Non-increasing independent variable "
                                      "(min step {})".format(dx[:-1].min()))
        dx[-1] = dx[-2]
        if np.allclose(dx, dx[0]):
            dx = dx[0]

    if xwidth is None and iwidth is None:
        if x is None:
            iwidth = 1
        else:
            xwidth = 1
    if iwidth is None:
        iwidth = xwidth / dx

    if iwidth == 0 or iwidth is 1:
        if order == 1:
            df = f.copy()
            df[:-1] = df[1:] - df[:-1]
            df[-1] = df[-2]
        else:
            df = np.diff(f, n=order)
            beg, end = order//2, (order+1)//2
            df = np.concatenate([[df[0]]*beg, df, [df[-1]]*end])
    elif iwidth is 2 and order == 1:
        return np.gradient(f, dx)
    else:
        from scipy.ndimage import correlate1d
        min_iwidth = 0.25
        if iwidth < min_iwidth:
            msg = "Width of {} too small for reliable results using {}"
            raise UserWarning(msg.format(iwidth, min_iwidth))
        # kernel truncated at truncate*iwidth; it is 4 by default
        truncate = np.clip(4, min_scale/iwidth, 100/iwidth)
        kern = gaussian_kernel(iwidth, order=order, truncate=truncate)
        df = correlate1d(f, kern, mode='nearest')

    return df/dx**order


def gaussian_kernel(sigma, order=0, truncate=4.0):
    """ mostly copied from scipy.ndimage.gaussian_filter1d """
    if order not in range(4):
        raise ValueError('Order outside 0..3 not implemented')
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sd = sd * sd
    # calculate the kernel:
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * float(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
    # implement first, second and third order derivatives:
    if order:
        weights[lw] *= -1.0 / sd if order == 2 else 0.0
        for ii in range(1, lw + 1):
            x = float(ii)
            if order == 1:  # first derivative
                tmp = -x / sd * weights[lw + ii]
            elif order == 2:  # second derivative
                tmp = (x * x / sd - 1.0) * weights[lw + ii] / sd
            elif order == 3:  # third derivative
                tmp = (3.0 - x * x / sd) * x * weights[lw + ii] / (sd * sd)
            weights[lw + ii] = tmp * (1.0 if order == 2 else -1.0)
            weights[lw - ii] = tmp
        s = np.arange(-lw, lw+1)**order / np.prod(np.arange(1, order + 1))
        s = np.dot(weights, s)
    else:
        s = np.sum(weights)
    return np.array(weights) / s


def flip(f, x=None):
    """reverse a function f(x) about x = 0, giving f(-x)

    returns
    -------
    f_pos : the function f(x) in the overlapping domain
    f_neg : the reversed function f(-x)
    x : argument x to function f
    i : slice to the overlapping region
    """
    if x is None:
        l = len(f)/2
        x = np.arange(0.5-l, l)
        g = f[::-1]
        pos = slice(None)
    else:
        neg = np.searchsorted(x, -x)
        pos = slice(neg[-1], neg[0] + 1)
        x = x[pos]
        g = f[neg[pos]]
        f = f[pos]
    return f, g, x, pos


def symmetric(f, g=None, x=None, parity=None):
    """Separate a function into its symmetric and anti-symmetric parts.

    parameters
    ----------
    f:      values of the function
    x:      points at which function values are given
            if None, assume uniform and centered at 0
    parity: which part to return, +1 for symmetric, -1 for anti-symmetric,
            None or 0 for both

    returns
    -------
    part:   the (anti-)symmetric part(s) of f, given by (1/2) * (f(x) +/- f(-x))
    x:      x, or view on x,
    """
    if g is None:
        f, g, x, i = flip(f, x)

    parity = parity or np.array([[1], [-1]])
    part = (f + parity*g)/2
    return part, x


def symmetry(f, x=None, parity=None, integrate=False):
    """Calculate degree of even or odd symmetry of a function.

    parameters
    ----------
    f:      values of the function
    x:      points at which function values are given
            if None, assume uniform and centered at 0
    parity: type of symmetry, use +1 if even, -1 if odd, None for both
    integrate:  if True, return full array otherwise integrate it

    returns
    -------
    x:      as given, or centered range
    part:   (anti-)symmetric part, given by
                part = (f(x) + parity*f(-x))/2
    normed: if not integrate, part normalized to [0, 1], given by
                abs(part) / (abs(sym) + abs(antisym))
    total:  if integrate, the normalized sum given by mean(normed)
    """
    f, g, x, i = flip(f, x)
    x0 = np.searchsorted(x, 0)

    parts, x = symmetric(f, g, x)
    mags = np.abs(parts)
    normed = mags/mags.sum(0)
    if parity:
        p = {1: 0, -1: 1}[parity]
        normed = normed[p]
    if integrate:
        total = np.nanmean(normed[..., x0:], -1)
    sym = namedtuple('sym', 'x i symmetric antisymmetric symmetry'.split())
    return sym(x, i, *parts, symmetry=(total if integrate else normed))


def propagate(func, uncert, size=1000, domain=1, plot=False, verbose=False):
    """testing function for propagating uncertainties"""
    if size >= 10:
        size = np.log10(size)
    size = int(round(size))
    print '1e{}'.format(size),
    size = 10**size
    if np.isscalar(uncert):
        uncert = [uncert]*2
    domain = np.atleast_1d(domain)
    domains = []
    for dom in domain:
        if np.isscalar(dom):
            domains.append((0, dom))
        elif len(dom) == 1:
            domains.append((0, dom[0]))
        else:
            domains.append(dom)
    x_true = np.row_stack([np.random.rand(size)*(dom[1]-dom[0]) + dom[0]
                           for dom in domains])
    x_err = np.row_stack([np.random.normal(scale=u, size=size) if u > 0 else
                          np.zeros(size) for u in uncert])
    x_meas = x_true + x_err
    if verbose:
        print
        for k, v in dict(x_true=x_true, x_meas=x_meas, x_err=x_err).iteritems():
            print k + ':', v.shape, 'min', v.min(1), 'max', v.max(1)
    xfmt = 'x: [{d[1][0]:5.2g}, {d[1][1]:5.2g}) +/- {dx:<5.4g} '
    thetafmt = 'theta: [{d[0][0]:.2g}, {d[0][1]:.3g}) +/- {dtheta:<5.4g} '
    if func == 'nn':
        dtheta, _ = uncert
        print thetafmt.format(dtheta=dtheta, d=domains)+'->',
        f = lambda x: np.cos(x[0])*np.cos(x[1])
        f_uncert = dtheta/rt2
    elif func == 'rn':
        dtheta, dx = uncert
        print (thetafmt+xfmt+'->').format(dtheta=dtheta, dx=dx, d=domains),
        f = lambda x: np.cos(x[0])*x[1]
        f_uncert = np.sqrt(dx**2 + (x_true[1]*dtheta)**2).mean()/rt2
    elif func == 'rr':
        dx, _ = uncert
        print xfmt.format(dx=dx, d=domains)+'->',
        f = lambda x: x[0]*x[1]  # (x[0]-x[0].mean())*(x[1]-x[1].mean())
        f_uncert = rt2*dx*np.sqrt((x_true[0]**2).mean())
    else:
        f_uncert = None
    f_true = f(x_true)
    f_meas = f(x_meas)
    f_err = f_meas - f_true
    if False and 'r' in func:
        f_err /= np.sqrt(f_meas**2 + f_true**2)/2
        print 'quad',
    if plot:
        fig = plt.gcf()
        fig.clear()
        ax = plt.gca()
        if size <= 10000:
            ax.scatter(f_true, f_err, marker='.', c='k', label='f_err v f_true')
        else:
            ax.hexbin(f_true, f_err)
    nbins = 25 if plot else 7
    f_bins = np.linspace(f_true.min(), f_true.max()*(1+1e-8), num=1+nbins)
    f_bini = np.digitize(f_true, f_bins)
    ubini = np.unique(f_bini)
    f_stds = [f_err[f_bini == i].std() for i in ubini]
    if plot:
        ax.plot((f_bins[1:]+f_bins[:-1])/2, f_stds, 'or')
    if verbose:
        print
        print '[', ', '.join(map('{:.3g}'.format, f_bins)), ']'
        print np.row_stack([ubini, np.bincount(f_bini)[ubini]])
        print '[', ', '.join(map('{:.3g}'.format, f_stds)), ']'
    f_err_std = f_err.std()
    ratio = f_uncert/f_err_std
    missed = ratio - 1
    print '{:< 9.4f}/{:< 9.4f} = {:<.3f} ({: >+7.2%})'.format(
        f_uncert, f_err_std, ratio, missed),
    print '='*int(-np.log10(np.abs(missed)))
    if verbose:
        print
    return f_err_std


def sigprint(sigma):
    """print some info about uncertainty sigma"""
    sigfmt = ('{:7.4g}, '*5)[:-2].format
    mn, mx = sigma.min(), sigma.max()
    return sigfmt(mn, sigma.mean(), mx, sigma.std(ddof=1), mx/mn)


def sigma_for_fit(arr, std_err, std_dev=None, added=None, x=None, plot=False,
                  relative=None, const=None, xnorm=None, ignore=None,
                  verbose=False):
    """calculate the uncertainty for fitting a function"""
    if x is None:
        x = np.arange(len(arr))
    if ignore is not None:
        ignore.sort()
        xignore = list(np.searchsorted(x, ignore))
        try:
            x0 = xignore.pop(ignore.index(0))
            ignore_inds = [x0-1, x0, x0+1][x0 < 1:]
        except ValueError:
            ignore_inds = []
        if len(xignore):
            ignore_inds.extend(range(xignore.pop(-1)+1, len(arr)))
        if len(xignore):
            ignore_inds.extend(range(xignore[0]))
    if plot:
        ax = plot if isinstance(plot, plt.Axes) else plt.gca()
        plot = True
        plotted = []
        colors = it.cycle('rgbcmyk')
    try:
        mods = it.product(const, relative, xnorm)
    except TypeError:
        mods = [(const, relative, xnorm)]
    for const, relative, xnorm in mods:
        signame = 'std_err'
        sigma = std_err.copy()
        sigma[ignore_inds] = np.inf
        if plot:
            c = colors.next()
            if signame not in plotted:
                ax.plot(x, std_err, '.'+c, label=signame)
                plotted.append(signame)
        if relative:
            sigma /= arr
            signame += '/arr'
            if plot and signame not in plotted:
                ax.plot(x, sigma, ':'+c, label=signame)
                plotted.append(signame)
        if const is not None:
            isconst = np.isscalar(const)
            offsetname = '({:.3g})'.format(const) if isconst else 'const'
            sigma = np.hypot(sigma, const)
            signame = 'sqrt({}^2 + {}^2)'.format(signame, offsetname)
            if verbose:
                print 'adding const',
                print 'sqrt(sigma^2 + {}^2)'.format(offsetname)
            if plot and signame not in plotted:
                ax.plot(x, sigma, '-'+c, label=signame)
                if isconst:
                    ax.axhline(const, ls='--', c=c, label='const')
                else:
                    ax.plot(x, const, '^'+c, label='const')
        if xnorm:
            if xnorm == 'log':
                label = 'log(1 + x)'
                xnorm = np.log1p(x)
            elif xnorm == 1:
                label = 'x'
                xnorm = x
            else:
                label = 'x^{}'.format(xnorm)
                xnorm = x**xnorm
            signame += '*' + label
            sigma *= xnorm
            if plot and label not in plotted:
                ax.plot(x, xnorm, '--'+c, label=label)
                plotted.append(label)
            if plot and signame not in plotted:
                ax.plot(x, sigma, '-.'+c, label=signame)
                plotted.append(signame)
        if verbose:
            print 'sigma =', signame
            print 'nan_info',
            helpy.nan_info(sigma, True)
            print 'sigprint', sigprint(sigma)
    if plot:
        ax.legend(loc='upper left', fontsize='x-small')
    return sigma

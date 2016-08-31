#!/usr/bin/env python
# encoding: utf-8

from __future__ import division

import itertools as it

import numpy as np
import matplotlib.pyplot as plt

import helpy

pi = np.pi
twopi = 2*pi
rt2 = np.sqrt(2)


def interp_nans(f, x=None, max_gap=10, inplace=False, verbose=False):
    """ Replace nans in function f(x) with their linear interpolation"""
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
        fmt = '\t      interp {:7} {:8} {:10}'.format
        print fmt('{}@{}'.format(gaps.max(), gaps.argmax()),
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


def der(f, dx=None, x=None, xwidth=None, iwidth=None, order=1, min_scale=1):
    """ Take a finite derivative of f(x) using convolution with gaussian

    A function convolved with the derivative of a gaussian kernel gives the
    derivative of the function convolved with the integral of the kernel of a
    gaussian kernel.  For any convolution:
        (f * g)' = f * g' = g * f'
    so we start with f and g', and return g and f', a smoothed derivative.

    Optionally can not smooth by giving width 0.

    parameters
    ----------
    f : an array to differentiate
    xwidth or iwidth : smoothing width (sigma) for gaussian.
        use iwidth for index units, (simple array index width)
        use xwidth for the physical units of x (x array is required)
        use 0 for no smoothing. Gives an array shorter by 1.
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

    if xwidth is None and iwidth is None:
        if x is None:
            iwidth = 1
        else:
            xwidth = 1
    if iwidth is None:
        iwidth = xwidth / dx

    if iwidth == 0:
        if order == 1:
            df = f.copy()
            df[:-1] = df[1:] - df[:-1]
            df[-1] = df[-2]
        else:
            df = np.diff(f, n=order)
            beg, end = order//2, (order+1)//2
            df = np.concatenate([[df[0]]*beg, df, [df[-1]]*end])
    else:
        min_iwidth = 0.5
        if iwidth < min_iwidth:
            msg = "Width of {} too small for reliable results using {}"
            raise UserWarning(msg.format(iwidth, min_iwidth))
            iwidth = min_iwidth
        from scipy.ndimage import gaussian_filter1d
        # kernel truncated at truncate*iwidth; it is 4 by default
        truncate = np.clip(4, min_scale/iwidth, 100/iwidth)
        df = gaussian_filter1d(f, iwidth, order=order, truncate=truncate)

    return df/dx**order


def print_stats(**kwargs):
    for k, v in kwargs.iteritems():
        print k + ':', v.shape, 'min', v.min(1), 'max', v.max(1)


def propagate(func, uncert, size=1000, domain=1, plot=False, verbose=False):
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
        print_stats(x_true=x_true, x_meas=x_meas, x_err=x_err)
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
    sigfmt = ('{:7.4g}, '*5)[:-2].format
    mn, mx = sigma.min(), sigma.max()
    return sigfmt(mn, sigma.mean(), mx, sigma.std(ddof=1), mx/mn)


def sigma_for_fit(arr, std_err, std_dev=None, added=None, x=None, plot=False,
                  relative=None, const=None, xnorm=None, ignore=None,
                  verbose=False):
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

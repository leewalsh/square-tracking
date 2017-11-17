#!/usr/bin/env python
# encoding: utf-8
"""Helpful python plotting functions and classes for use in various analysis.

Copyright (c) 2012--2017 Lee Walsh, Department of Physics, University of
Massachusetts; all rights reserved.
"""

from __future__ import division

import os
import sys
import itertools as it
from math import sqrt
from collections import OrderedDict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import helpy
import correlation as corr


def circle_click(im):
    """saves points as they are clicked, then find the circle that they define

    To use:
    when image is shown, click three non-co-linear points along the perimeter.
    neither should be vertically nor horizontally aligned (gives divide by zero)
    when three points have been clicked, a circle should appear.
    Then close the figure to allow the script to continue.
    """
    if matplotlib.is_interactive():
        raise RuntimeError("Cannot do circle_click in interactive/pylab mode")

    print ("Please click three points on circumference of the boundary, "
           "then close the figure")
    clicks = []
    center = []
    if isinstance(im, basestring):
        im = plt.imread(im)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im)

    def circle_three_points(*xs):
        """ With three points, calculate circle
            e.g., see paulbourke.net/geometry/circlesphere
            returns center, radius as (xo, yo), r
        """
        xs = np.squeeze(xs)
        if xs.shape == (3, 2):
            xs = xs.T
        (x1, x2, x3), (y1, y2, y3) = xs

        ma = (y2-y1)/(x2-x1)
        mb = (y3-y2)/(x3-x2)
        xo = ma*mb*(y1-y3) + mb*(x1+x2) - ma*(x2+x3)
        xo /= 2*(mb-ma)
        yo = (y1+y2)/2 - (xo - (x1+x2)/2)/ma
        r = ((xo - x1)**2 + (yo - y1)**2)**0.5

        return xo, yo, r

    def circle_click_connector(click):
        """receive and save clicks. when there are three, calculate the circle

         * swap x, y to convert image coordinates to cartesian
         * the return value cannot be saved, so modify a mutable variable
           (center) from an outer scope
        """
        clicks.append([click.ydata, click.xdata])
        print 'click {}: x: {:.2f}, y: {:.2f}'.format(len(clicks), *clicks[-1])
        if len(clicks) == 3:
            center.extend(circle_three_points(clicks))
            print 'center {:.2f}, {:.2f}, radius {:.2f}'.format(*center)
            cpatch = matplotlib.patches.Circle(
                center[1::-1], center[2], linewidth=3, color='g', fill=False)
            ax.add_patch(cpatch)
            fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', circle_click_connector)
    plt.show()
    return center


def axline(ax, orient, x, start=0, stop=1, coords='ax', **kwargs):
    """plot straight lines; wrapper of pyplot axvline, axhline, vlines, hlines

    parameters
    ----------
    ax:     an axes object
    orient: orientation of line, 'h' or 'v'
    x:      where to place the line, x-value for 'v' or y-value for 'h'
    start, stop:    beginning and end of line
    coords: 'ax' or 'data', coordinates for `start` and `stop`.
    kwargs: style args
    """
    f = ax.__getattribute__(
        {'ax': 'ax{}line', 'data': '{}lines'}[coords].format(orient)
    )
    if coords == 'data':
        ks = {'c': 'colors', 'color': 'colors',
              'ls': 'linestyles', 'linestyle': 'linestyles'}
        for k, v in ks.items():
            if k in kwargs:
                kwargs[v] = kwargs.pop(k)
    return f(x, start, stop, **kwargs)


def mark_value(ax, x, label='', method='vline', annotate=None, line=None):
    if not ax.get_xlim()[0] < x < ax.get_xlim()[1]:
        return
    if method == 'vline':
        line_default = dict(color='gray', linestyle='--', linewidth=0.5,
                            zorder=0.1, start=0, stop=0.68)
        annotate_default = dict(ha='left', va='center', annotation_clip=True,
                                xytext=(4, 0), textcoords='offset points',
                                xy=(x, 0.1), xycoords=('data', 'axes fraction'))
        line = dict(line_default, **(line or {}))
        lines = axline(ax, 'v', x, **line)
    elif method == 'corner':
        x, y = x
        line = dict(dict(color='k', linestyle=':', linewidth=1, zorder=0.1),
                    **(line or {}))
        annotate_default = dict(xytext=(11, 11), textcoords='offset points',
                                xy=(x, y), xycoords='data',
                                arrowprops=dict(arrowstyle='->', lw=0.5))
        lines = [axline(ax, 'v', x, ax.get_ylim()[0], y, coords='data', **line),
                 axline(ax, 'h', y, ax.get_xlim()[0], x, coords='data', **line)]
    elif method == 'axis':
        annotate_default = dict(xy=(x, 0), xycoords=('data', 'axes fraction'),
                                xytext=(0, 9), textcoords='offset points',
                                ha='center', va='baseline', annotation_clip=True,
                                arrowprops=dict(arrowstyle='->', lw=0.5))
    else:
        raise ValueError("Unknown method " + method)

    annotate = dict(annotate_default, **(annotate or {}))
    annotation = ax.annotate(label, **annotate)

    return lines, annotation


def draw_circles(centers, rs, ax=None, fig=None, **kwargs):
    """draw circles on an axis

    parameters:
        centers:    one or a list of (x, y) pairs
        rs:         one or a list of radii (in data units)
        ax or fig:  axis or figure on which to draw
        kwargs:     arguments passed to the patch (e.g.: color, fill, zorder)

    returns:
        patches:    a list of the patch objects
    """
    cs = np.atleast_2d(centers)
    rs = np.atleast_1d(np.abs(rs))
    n = max(map(len, (cs, rs)))
    b = np.broadcast_to(cs, (n, 2)), np.broadcast_to(rs, (n,))
    patches = [matplotlib.patches.Circle(*cr, **kwargs) for cr in it.izip(*b)]
    if ax is None:
        if fig is None:
            ax = plt.gca()
        else:
            ax = fig.gca()
    map(ax.add_patch, patches)
    ax.figure.canvas.draw()
    return patches


def rc(*keys):
    params = {
        'warbler': {
            'figure.maxsize': np.array([25.6, 13.56]),
            'figure.maxwidth': 25.6,
            'figure.maxheight': 13.56,
        },
        'what': {
            'figure.maxsize': np.array([12.8, 7.16]),
            'figure.maxwidth': 12.8,
            'figure.maxheight': 7.16,
        },
    }[helpy.gethost()]
    if len(keys) > 1:
        return {key: params[key] for key in keys}
    elif keys:
        return params[keys[0]]
    else:
        return params


def rcParam_diff(rcname='CURR', rcs=None, rcParams=None, quiet=False,
                 always=('text.usetex',), never=('backend',)):
    if rcs is None:
        rcs = OrderedDict([('DEFAULT', plt.rcParamsDefault.copy())])
    elif 'DEFAULT' not in rcs:
        rcs['DEFAULT'] = plt.rcParamsDefault.copy()

    if rcParams is None:
        rcParams = plt.rcParams
    rcs[rcname] = rcParams.copy()

    fmt = "{:>21}: " + ' => '.join(["{:^16}"]*len(rcs))
    ps = [['KEY'] + rcs.keys()]
    for key in sorted(rcs['DEFAULT']):
        if key in never:
            continue
        vals = [rc[key] for rc in rcs.values()]
        changed = any(vals[i] != vals[i-1] for i in xrange(1, len(vals)))
        if changed or key in always:
            for i in xrange(len(vals)-1, 0, -1):
                if vals[i] == vals[i-1]:
                    vals[i] = '...'
            ps.append([key] + vals)
    if not quiet:
        print '\n'.join(it.starmap(fmt.format, ps))
    return rcs


def check_neighbors(prefix, frame, data=None, im=None, **neighbor_args):
    """interactively display neighbors defined by corr.neighborhoods to check"""
    from correlation import neighborhoods
    fig, ax = plt.subplots()

    if im is None:
        im = helpy.find_tiffs(prefix=prefix, frames=frame,
                              single=True, load=True)[1]
    ax.imshow(im, cmap='gray', origin='lower')

    if data is None:
        data = helpy.load_data(prefix)
        data = data[data['t'] >= 0]
    fdata = helpy.splitter(data, 'f')
    frame = fdata[frame]
    positions = frame['xy']
    neighs, mask, dists = neighborhoods(positions, **neighbor_args)
    fmt = 'particle {} track {}: {} neighbors at dist {} - {} ({})'.format
    for pt, ns, m, ds, d in zip(positions, neighs, mask, dists, frame):
        ns, ds = ns[~m], ds[~m]
        if len(ns) == 0:
            continue
        print fmt(d['id'], d['t'], len(ns), ds.min(), ds.max(), ds.mean())
        # print '\tneighbors:', (len(ns)*'{:5d} ').format(*ns)
        # print '\tdistances:', (len(ns)*'{:5.2f} ').format(*ds)
        ax.scatter(*positions.T[::-1], c='k', marker='o')
        ax.scatter(*positions[ns].T[::-1], c='w', marker='o')
        ax.scatter(pt[1], pt[0], c='r', marker='o')
        plt.waitforbuttonpress()


def plt_text(*args, **kwargs):
    return [plt.text(*arg, **kwargs) for arg in zip(*args)]


def set_axes_size_inches(ax, axsize, clear=None, tight=None):
    """Resize figure to control size of axes in inches"""
    if clear is not None:
        properties = {'ticks': {'xticks': [], 'yticks': []},
                      'labels': {'xlabel': '', 'ylabel': ''},
                      'title': {'title': ''}}
        ax.set(**{k: v for c in clear for (k, v) in properties[c].iteritems()})
    if isinstance(tight, dict):
        ax.figure.tight_layout(**tight)
    elif tight is True:
        ax.figure.tight_layout()
    elif tight is None or tight is False:
        pass
    elif np.isscalar(tight):
        ax.figure.tight_layout(pad=tight)
    else:
        raise ValueError('unexpected value `{}` for tight'.format(tight))

    portion_of_fig = np.diff(ax.get_position(), axis=0)[0]
    ax.figure.set_size_inches(axsize / portion_of_fig, forward=True)

    return ax.figure.get_size_inches()


def animate(imstack, fsets, fcsets, fosets=None, fisets=None,
            meta={}, f_nums=None, verbose=False, clean=0, plottracks=False):
    if verbose:
        print "Animating tracks!"

    def handle_event(event):
        animator_next = advance(event.key, **animator)
        animator.update(animator_next)

    def advance(key, f_idx, f_num, xlim=None, ylim=None):
        if verbose:
            print '\tpressed {}'.format(key),
        if key in ('left', 'up'):
            # going back resets to original limits, so fix them here
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            if f_idx >= 1:
                f_idx -= 1
        elif key in ('right', 'down'):
            # save new limits in case we go back and need to fix them
            xlim, ylim = ax.get_xlim(), ax.get_ylim()
            if f_idx < f_max - 1:
                f_idx += 1
        elif key == 'b':
            f_idx = 0
        elif key == 'e':
            f_idx = f_max - 1
        elif key == 's':
            savename = raw_input('save pdf to '+os.getcwd()+'/')
            if savename:
                ax.set(title='', xticks=[], yticks=[])
                if not savename.endswith('.pdf'):
                    savename += '.pdf'
                fig.savefig(savename)  # , bbox_inches='tight', pad_inches=0)
                print "saved to", savename
        elif key == 'i':
            print title.format(f_num, nos, nts, ncs)
        else:
            plt.close()
            f_idx = -1
        f_num = f_nums[f_idx] if 0 <= f_idx < f_max else -1
        if verbose:
            if f_idx >= 0:
                print 'next up {} ({})'.format(f_idx, f_num),
            else:
                print 'will exit'
            sys.stdout.flush()
        return {'f_idx': f_idx, 'f_num': f_num, 'xlim': xlim, 'ylim': ylim}

    # Access dataset parameters
    side = meta.get('sidelength', 17)
    rc = meta.get('orient_rcorner')
    drc = meta.get('orient_drcorner') or sqrt(rc)
    txtoff = min(rc, side/2)/2

    if meta.get('boundary'):
        fig, ax, (p, bnds) = plot_background(
            imstack[0], ppi=meta['boundary'][-1]/4, boundary=meta['boundary'],
            cut_margin=(meta.get('track_cut') and meta.get('track_cut_margin')))
    else:
        fig, ax, p = plot_background(imstack[0])

    title = "frame {:5d}\n{:3d} oriented, {:3d} tracked, {:3d} detected"
    ax.set_title(title.format(-1, 0, 0, 0))
    need_legend = not clean

    # Calcuate which frames to loop through
    lengths = map(len, [imstack, fsets, fcsets])
    f_max = min(lengths)
    assert f_max, 'Lengths imstack: {}, fsets: {}, fcsets: {}'.format(*lengths)
    if f_nums is None:
        f_nums = range(f_max)
    elif isinstance(f_nums, tuple):
        f_nums = range(*f_nums)
    elif len(f_nums) > f_max:
        f_nums = f_nums[:f_max]
    elif len(f_nums) < f_max:
        f_nums = range(f_max)
    else:
        raise ValueError('expected `f_nums` to be None, tuple, or list')
    animator = {'f_idx': 0,
                'f_num': f_nums[0]}

    repeat = f_old = animator['f_idx']

    while 0 <= animator['f_idx'] < f_max:
        # Check frame number
        if repeat > 5:
            if verbose:
                print 'stuck on frame {f_idx} ({f_num})'.format(**animator),
            break
        if animator['f_idx'] == f_old:
            repeat += 1
        else:
            repeat = 0
            f_old = animator['f_idx']
        assert animator['f_num'] == f_nums[animator['f_idx']], "f != fs[i]"
        if verbose:
            print 'showing frame {f_idx} ({f_num})'.format(**animator),

        # Load the data for this frame
        xyo = helpy.consecutive_fields_view(fsets[animator['f_num']], 'xyo')
        xyc = helpy.consecutive_fields_view(fcsets[animator['f_num']], 'xy')
        x, y, o = xyo.T
        omask = np.isfinite(o)

        ts = helpy.quick_field_view(fsets[animator['f_num']], 't')
        tracked = ts >= 0

        # Change background image
        p.set_data(imstack[animator['f_idx']])
        remove = []

        # plot the detected and tracked center dots
        dot_color = 'white'
        ps = ax.scatter(y[tracked], x[tracked], s=64, c=dot_color,
                        marker='o', edgecolors='black')
        remove.append(ps)

        # plot the tracks as smaller center dot that remains
        if plottracks:
            cmap = plt.get_cmap('Set3')
            dot_color = cmap(ts[tracked] % cmap.N)**3  # cube to darken
            ax.scatter(y[tracked], x[tracked], s=3, c=dot_color,
                       zorder=0.1*animator['f_idx']/f_max,  # later tracks above
                       )
        if not clean:
            # plot untracked center dots
            us = ax.scatter(y[~tracked], x[~tracked], c='c', zorder=.8)
            # plot all corner dots
            cs = ax.scatter(xyc[:, 1], xyc[:, 0], c='g', s=10, zorder=.6)
            remove.extend([us, cs])
            # plot center and corner dots as circles
            for dot, xy in zip(('center', 'corner'), (xyo, xyc)):
                ph = draw_circles(xy[:, 1::-1], meta[dot+'_kern'], ax=ax,
                                  lw=.5, color='k', fill=False, zorder=.6)
                remove.extend(ph)
            # plot valid corner distance circles
            for dr in (-drc, 0, drc):
                pc = draw_circles(xyo[:, 1::-1], rc+dr, ax=ax,
                                  lw=.5, color='g', fill=False, zorder=.5)
                remove.extend(pc)

        q = plot_orientations(xyo, ts, omask, clean=clean, side=side, ax=ax)
        remove.extend(q)

        # interpolated framesets
        if fisets is not None and animator['f_num'] > 0:
            # interpolated points have id = 0, so nonzero gives non-interpolated
            fiset = np.sort(fisets[animator['f_num']], order='id')
            ini = np.nonzero(helpy.quick_field_view(fiset, 'id'))[0]
            if len(ini) < len(fiset):
                fipset = np.delete(fiset, ini)
                xi, yi = helpy.quick_field_view(fipset, 'xy').T
                tsi = helpy.quick_field_view(fipset, 't')
                # plot interpolated center dots
                ips = ax.scatter(yi, xi, c='r' if clean else 'pink', zorder=.7)
                remove.append(ips)
                if not clean:
                    # label interpolated track number
                    itxt = plt_text(yi, xi + txtoff, tsi.astype('S'),
                                    color='pink', zorder=.9, ha='center')
                    remove.extend(itxt)

            # plot interpolated n-hat orientation arrows
            # orient may be interpolated, whether or not point is
            ioi = ini[omask[tracked]]
            if len(ioi) < len(fiset):
                fioset = np.delete(fiset, ioi)
                xyoi = helpy.consecutive_fields_view(fioset, 'xyo')
                iq = plot_orientations(xyoi, clean=clean, side=side, ax=ax,
                                       zorder=0.3,
                                       facecolor='black' if clean else 'gray')
                remove.extend(iq)

        # extended orientation data
        if fosets is not None:
            # load corners shape (n_particles, n_corners_per_particle, n_dim)
            oc = helpy.quick_field_view(fosets[animator['f_num']], 'corner')
            oca = oc.reshape(-1, 2)  # flatten

            # plot all corners
            ocs = ax.plot(oca[:, 1], oca[:, 0], linestyle='', marker='o',
                          color='white' if clean else 'black',
                          markeredgecolor='black', markersize=4, zorder=1)
            remove.extend(ocs)

            if not clean:
                # print corner separation angle
                cdisp = oc[omask] - xyo[omask, None, :2]
                cang = corr.dtheta(np.arctan2(cdisp[..., 1], cdisp[..., 0]))
                cang_s = np.nan_to_num(np.degrees(cang)).astype(int).astype('S')
                cx, cy = oc[omask].mean(1).T  # mean per particle
                ctxt = plt_text(cy, cx, cang_s, color='orange', zorder=.9,
                                horizontalalignment='center',
                                verticalalignment='center')
                remove.extend(ctxt)

        if not clean:
            # label track number
            txt = plt_text(y[tracked], x[tracked]+txtoff,
                           ts[tracked].astype('S'),
                           color='r', zorder=.9, horizontalalignment='center')
            remove.extend(txt)

        # count statistics for frame, print in title
        nts = np.count_nonzero(tracked)
        nos = np.count_nonzero(omask)
        ncs = len(o)
        ax.set_title(title.format(animator['f_num'], nos, nts, ncs))

        # generate legend (only once)
        if need_legend:
            need_legend = False
            if rc > 0:
                pc[0].set_label('r to corner')      # patch corner
            q[0].set_label('orientation')           # quiver
            ps.set_label('centers')                 # point scatter
            if fosets is None:
                cs.set_label('corners')             # corner scatter
            else:
                cs.set_label('unused corner')       # corner scatter
                ocs[0].set_label('used corners')    # oriented corner scatter
            if len(txt):
                txt[0].set_label('track id')
            ax.legend(fontsize='small')

        # update the figure and wait for instructions
        fig.canvas.draw()
        fig.canvas.mpl_connect('key_press_event', handle_event)
        plt.show(block=False)
        plt.waitforbuttonpress()

        # clean up this frame before moving to next
        for rem in remove:
            rem.remove()
        if verbose:
            print '\tdone with frame {} ({})'.format(f_old, f_nums[f_old])
            sys.stdout.flush()
    if verbose:
        print 'loop broken'


def plot_boundary(boundary, margin=0, ax=None,
                  coords='xy', set_lim=False, **kwargs):
    """draw a circle for the boundary, and a second for cutting margin
    """
    bndc = boundary[:2] if coords == 'xy' else boundary[1::-1]
    bndr = boundary[2]

    kwargs = dict(dict(color='tab:red', fill=False, zorder=1), **kwargs)
    bnds = draw_circles(bndc, bndr, ax=ax, **kwargs)
    if margin:
        bnds = draw_circles(bndc, bndr - margin, ax=ax,
                            label='margin', linestyle='--', **kwargs)

    ax.set_aspect('equal')
    if set_lim:
        w0, w = bndc[0] - bndr + margin, bndc[0] + bndr - margin
        h0, h = bndc[1] - bndr + margin, bndc[1] + bndr - margin
        ax.set_xlim(w0, w)
        ax.set_ylim(h0, h)

    return bnds


def plot_background(bgimage, ppi=109, boundary=None, cut_margin=None,
                    ax=None, verbose=False):
    """plot the background image and size appropriately"""
    if isinstance(bgimage, basestring):
        bgimage = plt.imread(bgimage)

    if boundary is None:
        h, w = bgimage.shape
        h0, w0 = 0, 0
    else:
        bndc = boundary[1::-1]
        bndr = boundary[2]
        w0, w = bndc[0] - bndr, bndc[0] + bndr
        h0, h = bndc[1] - bndr, bndc[1] + bndr

    figsize = np.array([w-w0, h-h0]) / ppi

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    p = ax.imshow(bgimage, cmap='gray', origin='lower', zorder=0)

    if boundary is not None:
        if cut_margin is not None:
            bndr = [bndr, bndr - cut_margin]
        bnds = draw_circles(bndc, bndr, ax=ax,
                            color='tab:red', fill=False, zorder=1)
        if cut_margin is not None:
            bnds[1].set_label('cut margin')
        p = (p, bnds)

    if verbose:
        print 'plotting actual size {:.2f}x{:.2f} in'.format(*figsize)
        print '{:d}x{:d} pix {:.2f} ppi'.format(int(w), int(h), ppi)

    fig.set_size_inches(figsize*1.02)
    ax.set_xlim(w0, w)
    ax.set_ylim(h0, h)
    set_axes_size_inches(ax, figsize, clear=['title', 'ticks'], tight=0)

    return fig, ax, p


def plot_orientations(xyo, ts=None, omask=None, clean=False, side=1,
                      ax=None, **kwargs):
    """plot orientation normals as arrows, some can be labeled with nhat"""
    if ax is None:
        fig, ax = plt.subplots()

    # mask the orientation data
    if omask is None:
        omask = np.where(np.isfinite(xyo[:, -1]))
    xyo = xyo[omask]
    xo, yo, oo = xyo.T
    so, co = np.sin(oo), np.cos(oo)

    # plot n-hat orientation arrows
    quiver_args = dict(angles='xy', units='xy', scale_units='xy',
                       width=side/8, scale=1/side, linewidth=0.75,
                       facecolor='black', edgecolor='white', zorder=0.4)
    quiver_args.update(kwargs)
    q = ax.quiver(yo, xo, so, co, **quiver_args)

    if ts is None:
        return [q]
    else:
        out = [q]

    # label n-hat
    t = 3
    to = ts[omask]
    nhat_offsets = side * np.stack([(so - co*2/3), (co + so*2/3)], axis=1)

    for i in np.where(to == t)[0] if clean else xrange(len(to)):
        a = ax.annotate(r'$\hat n$', xy=(yo[i], xo[i]),
                        xytext=nhat_offsets[i], textcoords='offset points',
                        ha='center', va='center', fontsize='large')
        out.append(a)

    return out


def plot_tracks(data, bgimage=None, style='t', slice=None, cmap='Set3',
                save=False, ax=None, verbose=False, **kwargs):
    """ Plots the tracks of particles in 2D

    parameters
    ----------
    data : tracksets or framesets to plot trackwise or framewise
    bgimage : a background image to plot on top of (the first frame tif, e.g.)
    save : where to save the figure
    ax : provide an axes to plot in
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    k = data.keys()[0]
    assert np.allclose(k, data[k][style])

    if bgimage is not None:
        plot_background(bgimage)

    cmap = plt.get_cmap(cmap)
    slice = helpy.parse_slice(slice)

    for k, d in data.iteritems():
        x, y = d['xy'][slice].T
        if style == 'f':
            p = ax.scatter(y, x, c=cmap(d['t'] % cmap.N), marker='o', **kwargs)
        elif style == 't':
            p = ax.plot(y, x, ls='-', c=cmap(k % cmap.N), **kwargs)

    ax.set_aspect('equal')
    fig.tight_layout()

    if save:
        save = save + '_tracks.png'
        print "saving tracks image to",
        print save if verbose else os.path.basename(save)
        fig.savefig(save, frameon=False, dpi=300)

    return p

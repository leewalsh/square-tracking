#!/usr/bin/env python
# encoding: utf-8
"""Helpful python plotting functions and classes for use in various analysis.

Copyright (c) 2012--2017 Lee Walsh, Department of Physics, University of
Massachusetts; all rights reserved.
"""

from __future__ import division

import os
import sys

from math import sqrt

import numpy as np
import matplotlib.pyplot as plt

import helpy
import correlation as corr


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

    if meta.get('track_cut'):
        fig, ax, (p, bnds) = plot_background(
            imstack[0], ppi=meta['boundary'][-1]/4,
            boundary=meta['boundary'], cut_margin=meta.get('track_cut_margin'))
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
                ph = helpy.draw_circles(xy[:, 1::-1], meta[dot+'_kern'], ax=ax,
                                        lw=.5, color='k', fill=False, zorder=.6)
                remove.extend(ph)
            # plot valid corner distance circles
            for dr in (-drc, 0, drc):
                pc = helpy.draw_circles(xyo[:, 1::-1], rc+dr, ax=ax,
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
        plt.waitforbuttonpress()

        # clean up this frame before moving to next
        for rem in remove:
            rem.remove()
        if verbose:
            print '\tdone with frame {} ({})'.format(f_old, f_nums[f_old])
            sys.stdout.flush()
    if verbose:
        print 'loop broken'


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
        bnds = helpy.draw_circles(bndc, bndr, ax=ax,
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
    if omask is not None:
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

#!/usr/bin/env python
from __future__ import division

import sys
import math
import itertools as it
import glob
from pprint import pprint

import numpy as np
from scipy.spatial import Voronoi, Delaunay, cKDTree as KDTree
from scipy.ndimage import gaussian_filter1d
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

import helpy
import helplt
import correlation as corr


def initialize_mdata(data):
    """melting data array to hold calculated statistics
    """

    # 'id', 'f', 't' hold the same data copied from original tracks data
    keep_fields = ['id', 'f', 't']
    melt_dtype = [(name, data.dtype[name]) for name in keep_fields]

    # new fields to hold:
    # shell, radius (from initial c.o.m.), local density, local psi, local phi
    melt_dtype.extend(zip(['sh', 'rad', 'dens', 'psi', 'phi'],
                          ['i4', 'f4', 'f4', 'f4', 'f4']))

    mdata = np.empty(data.shape, melt_dtype)
    for keep in keep_fields:
        mdata[keep][:] = data[keep]
    return mdata


def merge_melting(prefix_pattern):
    merged_prefix = helpy.replace_all(prefix_pattern, '*?', '') + 'MRG'
    helpy.save_log_entry(merged_prefix, 'argv')
    prefixes = [p[:-9] for p in glob.iglob(
        helpy.with_suffix(prefix_pattern, '_MELT.npz'))]

    metas, datas, mdatas = [], [], []
    for prefix in prefixes:
        meta = helpy.load_meta(prefix)
        print prefix
        data, mdata = helpy.load_data(prefix, 't m')
        print '\tinterpolating data'
        # to get the benefits of tracksets (interpolation, stub filtering):
        tsets = helpy.load_tracksets(data, run_repair='interp',
                                     run_track_orient=True)
        data = np.concatenate(tsets.values())
        data.sort(order=['f', 't'])
        assert np.all(data['id'] == mdata['id'])

        frames, mframes = helpy.load_framesets((data, mdata), ret_dict=False)
        f0 = int(meta['start_frame'])
        data['f'] -= f0
        mdata['f'] -= f0
        frames = frames[f0:]
        mframes = mframes[f0:]

        meta['start_frame'] = 0
        meta['end_frame'] = meta.get('end_frame') and meta['end_frame'] - f0
        metas.append(meta)
        datas.append(frames)
        mdatas.append(mframes)
    print 'concatenating all'
    data = [f for fs in it.izip_longest(*datas, fillvalue=[])
            for f in fs if len(f)]
    mdata = [f for fs in it.izip_longest(*mdatas, fillvalue=[])
             for f in fs if len(f)]
    data = np.concatenate(data)
    mdata = np.concatenate(mdata)
    meta = helpy.merge_meta(metas, excl_start=('center', 'corner'))
    meta.update(merged=prefixes, end_frame=np.max(meta['end_frame']))
    print '\n\t'.join(['merged sets:'] + prefixes)
    pprint(meta)
    return merged_prefix, meta, data, mdata


def melting_stats(frame, dens_method, neigh_args):
    """Calculate structural order parameters for a frame

    parameters
    ----------
    frame :     slice of melting data array for single frame
    dens_method: method for calculating density, passed to corr.density
    neigh_args: options for selecting neighborhood, passed to corr.neighborhoods

    returns
    -------
    all returned statistics are defined per particle, and returned as 1d arrays
    with the same shape as `frame`
    rad :   radial distance from cluster center of mass
    dens :  local density, calculated by corr.density
    psi :   local bond-orientational order parameter, from corr.pair_angle_op
    phi :   local molecular orientational order parameter, from corr.orient_op
    """
    xy = frame['xy']
    orient = frame['o']

    vor = Voronoi(xy)
    tess = Delaunay(xy)
    tree = KDTree(xy)
    neighborhoods = corr.neighborhoods(xy, tess=tess, tree=tree, **neigh_args)

    # Radial distance:
    com = xy.mean(0)
    rad = helpy.dist(xy, com)

    # Density:
    dens = corr.density(xy, dens_method, vor=vor, tess=tess,
                        neighbors=neighborhoods)

    # Order parameters:
    neigh, nmask = neighborhoods[:2]
    fewest = 2  # particles with only 1 neighbor trivially have |psi| = 1
    nmask[(~nmask).sum(1) < fewest] = True

    # Pair-angle op psi
    bond_angles, _ = corr.pair_angles(xy, neigh, nmask)
    psi = corr.pair_angle_op(bond_angles, m=M, locl=True)

    # molecular-angle op phi
    particle_angles, _ = corr.pair_angles(orient, neigh, nmask)
    phi = corr.orient_op(particle_angles, m=M, locl=True)

    return rad, dens, psi, phi


def find_start_frame(data, estimate=None, bounds=None, plot=False):
    """Determine the time of the onset of motion

    parameters
    ---------
    data :      the tracked data
    estimate :  an estimate, as frame index number, for start
    bounds :    a scalar indicating lower bound for start,
                or a two-tuple of (lower, upper) bounds for the start
    plot :      whether or not to plot the motion vs time

    returns
    -------
    start :     frame index for onset of motion
    ax :        an axes object, if plot was requested
    """
    estimate = estimate or 10
    if bounds is None:
        first = estimate // 2
        last = estimate * 100
    elif np.isscalar(bounds):
        first = bounds
        last = None
    else:
        first, last = bounds
    if last is None:
        last = first + (data['f'][-1] - first) // 3
    print "seeking start frame between {} and {}.".format(first, last)

    positions = helpy.load_trackstack(data, length=last)['xy']
    displacements = helpy.dist(positions, positions[0])
    distances = helpy.dist(np.gradient(positions, axis=0))
    ds = displacements.mean(1) * distances.mean(1)
    dm = np.minimum.accumulate(ds[::-1])[::-1] == np.maximum.accumulate(ds)
    dmi = np.nonzero(dm[first:last])[0] + first
    start = dmi[0] - 1
    print "\tstart frame: {}".format(start)

    if plot:
        fig, ax = plt.subplots()
        f = np.arange(len(ds))/args.fps
        ax.plot(f, ds, '-')
        ax.plot(f[dmi], ds[dmi], '*')

    return start


def find_ref_coords(positions):
    """Calculate the 4-fold orientational basis vectors

    returns
    -------
    ang :   primary phase of basis
    basis : basis angles as list of vectors
    """
    neighs, mask, dists = corr.neighborhoods(positions, size=2)
    pair_angles = corr.pair_angles(positions, neighs, mask, 'absolute')
    psi, ang = corr.pair_angle_op(*pair_angles, m=4, globl=True)
    print 'psi first frame:', psi
    if psi < 0.85:
        raise RuntimeError('ref_coords based on weak psi = {:.3f}'.format(psi))
    cos, sin = np.cos(ang), np.sin(ang)
    ref_basis = np.array([[cos, sin], [-sin, cos]])
    ref_origin = positions.mean(0)
    return ref_origin, ref_basis


def square_size(num):
    """given number of particles, return the perfect square and its width
    """
    width = int(round(math.sqrt(num)))
    num = width*width
    return num, width


def rectify_lattice(positions, ref_coords=None):
    """rotate by dominant lattice phase and translate to center of mass"""
    if ref_coords is None:
        ref_coords = find_ref_coords(positions)
    ref_origin, ref_basis = ref_coords
    rectified = corr.rotate2d(positions - ref_origin, basis=ref_basis)
    return rectified


def assign_shell(positions, ids=None, N=None, maxt=None, is_rectified=False):
    """given (N, 2) positions array, assign shell number to each particle

    shell number is assigned as maximum coordinate, written in a basis aligned
    with the global phase from the global bond-angle order parameter, with its
    origin at the center of mass, with unit length the average nearest-neighbor

    if W = sqrt(N) is even, smallest value is 0.5; if even, smallest value is 0.
    largest value is (W - 1)/2
    """
    if not is_rectified:
        positions = rectify_lattice(positions)
    N, W = square_size(N or len(positions))
    spacing = (positions.max(0) - positions.min(0)) / (W - 1)
    positions /= spacing
    shells = (np.abs(positions).max(1) + (1 - W % 2)/2.).round().astype(int)
    if ids is not None:
        ni, mi = len(ids), max(ids.max(), maxt, N)
        if ni <= mi or np.any(ids != np.arange(ni)):
            shells_by_id = np.full(1+mi, -1, 'i4')
            shells_by_id[ids] = shells
            return shells_by_id
    return shells


def split_shells(mdata, zero_to=0, do_mean=True, maxshell=None):
    """Split melting data into dict of slices for each shell.

    parameters
    ----------
    mdata:      melting data with 'sh' field.
    zero_to:    shell with which to merge shell zero. e.g., `zero_to=1` will
                include center particle in first shell.
    do_mean:    include an additional `shell` which is a merging of all shells

    return
    ------
    shells:     dict from integers to mdata slices.
    """
    splindex = np.where(mdata['sh'], mdata['sh'], zero_to) if zero_to else 'sh'
    shells = helpy.splitter(mdata, splindex, noncontiguous=True, ret_dict=True)
    if do_mean:
        shells[(maxshell or max(shells)) + 1] = mdata[mdata['sh'] >= 0]
    return shells


def average_shells(shells, fields=None, by='f'):
    """average all particles within each shell"""
    if fields is None:
        fields = [f for f in shells.dtype.names if f not in 'id f t sh']
    averaged = {}
    for s, shell in shells.iteritems():
        averaged[s] = {}
        for field in fields:
            i = np.where(np.isfinite(shell[field]))
            vals, bins = corr.bin_average(shell[by][i], shell[field][i], 1)
            averaged[s][field] = vals
            averaged[s][by] = bins[:-1]
    return averaged


def make_plot_args(meta_or_args):
    if isinstance(meta_or_args, dict):
        meta = meta_or_args
    else:
        _, meta = helpy.sync_args_meta(
            meta_or_args, {},
            'width end start fps side',
            'crystal_width end_frame start_frame fps sidelength'
        )
    M = meta['crystal_width']//2
    fps, side = meta['fps'], meta['sidelength']
    if 'boundary' in meta:
        rad = meta['boundary'][-1]/2.0/side
    else:
        rad = None

    plot_args = {
        'line_props': helpy.transpose_dict_of_lists({
            'label': ['center', 'inner'] + range(2, M) + ['outer', 'all'],
            'c':    map(plt.get_cmap('Dark2'), xrange(M+1)) + ['black'],
            'lw':   [1]*(M+1) + [2],
        }),
        'xylabel': {
            'f':    r'$t \, f$',
            'rad':  r'radial distance $r - \langle r \rangle$',
            'dens': r'density $\langle r_{ij}\rangle^{-2}$',
            'psi':  r'bond angle order $\Psi$',
            'phi':  r'molecular angle order $\Phi$',
        },
        'xylim': {
            'f':    (-25,
                     meta.get('end_frame')
                     and (meta['end_frame'] - meta['start_frame'])/fps),
            'rad':  (0, rad),
            'dens': (0, 1.1),
            'phi':  (0, 1.1),
            'psi':  (0, 1.1),
        },
        'unit': {
            'f':    1/fps,
            'rad':  1/side,
            'dens': side**2,
            'phi':  1,
            'psi':  1,
            'sh':   1,
        },
        'norm': {
            'f':    None,
            'rad':  (0, rad or 400/side),
            'dens': (0, 1),
            'phi':  (0, 1),
            'psi':  (0, 1),
            'sh':   None,
        },
    }
    return plot_args


def plot_by_shell(shells, x, y, start=0, smooth=0, zoom=1, **plot_args):
    line_props = plot_args.get('line_props', [{}]*len(shells))
    unit = plot_args.get('unit', {x: 1, y: 1})
    ax = plot_args.get('ax')
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))

    for s, shell in shells.iteritems():
        if s < 0:
            continue
        if x == 'f':
            ys = shell[y]
            xs = shell[x] - start
            if smooth:
                ys = gaussian_filter1d(ys, smooth, mode='nearest', truncate=2)
            ax.plot(xs*unit[x], ys*unit[y], **line_props[s])
            ax.set_xlim(plot_args['xylim'][x])
            ax.set_ylim(plot_args['xylim'][y])
        elif x in ('dens', 'phi', 'psi'):
            xs, ys = shell[x], shell[y]
            ax.scatter(xs*unit[x], ys*unit[y], s=0.01, **line_props[s])
        else:
            raise ValueError("Unknown x-value `{!r}`.".format(x))

    ax.legend(fontsize='small')
    ax.set_xlabel(plot_args['xylabel'][x])
    ax.set_ylabel(plot_args['xylabel'][y])

    return ax


def plot_parametric_hist(mdata, x, y, ax=None, **plot_args):
    unit = plot_args.get('unit', {x: 1, y: 1})
    extent = plot_args['norm'][x] + plot_args['norm'][y]
    if ax is None:
        fig, ax = plt.subplots()
    xs, ys = mdata[x]*unit[x], mdata[y]*unit[y],
    hexbin_args = plot_args.get('hexbin', {})
    if hexbin_args.get('marginals'):
        bins = hexbin_args.get('gridsize', 100)
        counts, xbins, ybins = np.histogram2d(
            np.nan_to_num(xs),
            np.nan_to_num(ys),
            bins=bins,
            # normed=True,
            range=[[np.nanmin(xs), np.nanmax(xs)],
                   [np.nanmin(ys), np.nanmax(ys)]])
        xi = np.digitize(np.nan_to_num(xs), xbins, True)
        yi = np.digitize(np.nan_to_num(ys), ybins, True)
        zs = np.where(np.isnan(xi) | np.isnan(yi), np.nan, counts[xi-1, yi-1])
    else:
        zs = None
    hexbin = ax.hexbin(xs, ys, zs,
                       norm=matplotlib.colors.LogNorm(),
                       extent=extent,
                       **hexbin_args)

    ax.set_xlabel(plot_args['xylabel'][x])
    ax.set_ylabel(plot_args['xylabel'][y])

    return ax, hexbin


def plot_by_config(prefix_pattern, smooth=1, side=1, fps=1):
    configs = ['inward', 'aligned', 'random', 'outward']
    stats = ['phi', 'psi', 'dens']
    stats = {stat: {config: np.load(prefix_pattern.format(config, stat)+'.npy')
                    for config in configs}
             for stat in stats}

    xlims = {'dens': 300, 'phi': 200, 'psi': 150}
    statlabels = {
        'dens': r'$\mathrm{density}\ \langle r_{ij}\rangle^{-2}$',
        'psi':  r'$\mathrm{bond\ angle\ order}\ \Psi$',
        'phi':  r'$\mathrm{molecular\ angle\ order}\ \Phi$'}
    plt.rc('text', usetex=True)
    figs = {}
    for stat, v in stats.iteritems():
        colors = ['orange', 'brown', 'magenta', 'cyan']
        fig, ax = plt.subplots(figsize=(4, 3))
        for conf in configs:
            ax.plot(np.arange(len(v[conf]))/fps,
                    gaussian_filter1d(v[conf], smooth, mode='constant', cval=1),
                    lw=2, label=conf, c=colors.pop())
        ax.set_xscale('log')
        ax.set_xlabel(r'$t \, f$')
        ax.set_xlim(1, xlims[stat])
        ax.set_ylim(0, 1)
        ax.set_ylabel(statlabels[stat])
        ax.legend(loc='lower left', fontsize='small')
        fig.savefig(prefix_pattern.format('ALL', stat) + '.pdf')
        figs[stat] = fig
    return figs


def plot_spatial(frame, mframe, vor=None, tess=None, ax=None, **kw):
    """plot map of parameter for given frame"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))

    if vor.points.shape[1] != 2:
        raise ValueError("Voronoi diagram is not 2-D")

    if kw.get('show_points', True):
        ax.plot(vor.points[:, 0], vor.points[:, 1], '.')
    if kw.get('show_vertices', True):
        ax.plot(vor.vertices[:, 0], vor.vertices[:, 1], 'o')

    line_colors = kw.get('line_colors', 'k')
    line_width = kw.get('line_width', 1.0)
    line_alpha = kw.get('line_alpha', 1.0)

    center = vor.points.mean(axis=0)
    ptp_bound = vor.points.ptp(axis=0)

    finite_segments = []
    infinite_segments = []
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            finite_segments.append(vor.vertices[simplex])
        else:
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[i] + direction * ptp_bound.max()

            infinite_segments.append([vor.vertices[i], far_point])

    ax.add_collection(LineCollection(finite_segments,
                                     colors=line_colors,
                                     lw=line_width,
                                     alpha=line_alpha,
                                     linestyle='solid'))
    ax.add_collection(LineCollection(infinite_segments,
                                     colors=line_colors,
                                     lw=line_width,
                                     alpha=line_alpha,
                                     linestyle='dashed'))

    # adjust bounds:
    margin = 0.1 * vor.points.ptp(axis=0)
    xy_min = vor.points.min(axis=0) - margin
    xy_max = vor.points.max(axis=0) + margin
    ax.set_xlim(xy_min[0], xy_max[0])
    ax.set_ylim(xy_min[1], xy_max[1])

    return ax


def plot_regions(frame, mframe=None, vor=None, colors='sh', norm=None, ax=None, **plot_args):
    """plot some parameter in the voronoi cells"""
    xy = helpy.quick_field_view(frame, 'xy')
    if vor is None:
        vor = Voronoi(xy)
    xyo = helpy.consecutive_fields_view(frame, 'xyo')
    # convert to i, j (row, col) coordinates to match image version:
    # (x, y, o) = (y, x, pi/2 - o)
    yxo = xyo[:, [1, 0, 2]] * [[1, 1, -1]] + [[0, 0, np.pi/2]]

    unit = plot_args.get('unit')
    cmap = plt.get_cmap('Dark2' if colors == 'sh' else 'viridis')
    if norm is None:
        norm = matplotlib.colors.Normalize()
    if colors in frame.dtype.names:
        colors = cmap(norm(frame[colors]))
    elif mframe is not None and colors in mframe.dtype.names:
        if colors == 'sh':
            norm = lambda x: x
            bg = np.max(mframe[colors]*unit[colors])
        else:
            bg = norm(0)
        colors = mframe[colors]*unit[colors]
        colors = norm(colors)
        colors = cmap(colors)
    elif np.isscalar(colors):
        colors = it.repeat(cmap(colors))
    elif len(colors) == len(frame):
        colors = cmap(norm(colors))
    elif isinstance(colors, tuple):
        colors = it.repeat(colors)
    else:
        raise ValueError("Unknown color `{!r}`".format(colors))

    if ax is None:
        fig, ax = plt.subplots()

    ax.set_facecolor(cmap(bg))

    patch_args = dict(edgecolor='k', zorder=0.5)
    quiver_args = dict(edgecolor='none', side=1/unit['rad'], zorder=2)
    scatter_args = dict(c=colors, edgecolors='k', zorder=1)

    patches = [
        ax.fill(*vor.vertices[vor.regions[j]].T, fc=colors[i], **patch_args)
        for i, j in enumerate(vor.point_region) if -1 not in vor.regions[j]
    ]
    helplt.plot_orientations(yxo, ax=ax, **quiver_args)
    scatter = ax.scatter(*vor.points.T, **scatter_args)

    ax.axis('equal')
    vmn, vmx = vor.min_bound, vor.max_bound
    xlim, ylim = ([[-0.1], [0.1]] * (vmx - vmn) + [vmn, vmx]).T
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax, patches, scatter, norm


def melt_analysis(data):
    mdata = initialize_mdata(data)

    frames, mframes = helpy.load_framesets((data, mdata), ret_dict=False)
    shells = assign_shell(frames[0]['xy'], frames[0]['t'],
                          maxt=data['t'].max())
    mdata['sh'] = shells[mdata['t']]
    maxshell = shells.max()

    dens_method = 'dist'

    for frame, melt in it.izip(frames, mframes):
        nn = np.where(melt['sh'] == maxshell, 3, 4)
        neigh_args = {'size': (nn,)*2}

        rad, dens, psi, phi = melting_stats(frame, dens_method, neigh_args)
        melt['rad'] = rad
        melt['dens'] = dens
        melt['psi'] = psi
        melt['phi'] = phi
    return mdata, frames, mframes


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    arg = parser.add_argument
    arg('prefix', metavar='PRE', help="Filename prefix with full or relative "
        "path (<prefix>_POSITIONS.npz, <prefix>_CORNER_POSITIONS.npz, etc)")
    arg('-m', '--melt', action='store_true', help='Calculate melting stats')
    arg('-w', '--width', type=int, help='Crystal width')
    arg('-s', '--side', type=float,
        help='Particle size in pixels, for unit normalization')
    arg('--start', type=float, help='First frame')
    arg('--end', type=float, help='Last frame')
    arg('--smooth', type=float, default=0, help='frames to smooth over')
    arg('-f', '--fps', type=float,
        help="Number of frames per shake (or second) for unit normalization")
    arg('--noshow', action='store_false', dest='show',
        help="Don't show figures (just save them)")
    arg('--nosave', action='store_false', dest='save',
        help="Don't save outputs or figures")
    arg('--noplot', action='store_false', dest='plot',
        help="Don't generate (fewer, not none at this point) figures")
    arg('-z', '--zoom', metavar="ZOOM", type=float,
        help="Factor by which to zoom out (in if ZOOM < 1)")
    arg('-v', '--verbose', action='count', help='Be verbose, may repeat: -vv')

    args = parser.parse_args()
    if not args.verbose:
        from warnings import filterwarnings
        filterwarnings('ignore', category=RuntimeWarning,
                       module='numpy|scipy|matplot')

    if glob.has_magic(args.prefix):
        args.prefix, meta, data, mdata = merge_melting(args.prefix)
        if args.save:
            data, mdata = map(helpy.remove_self_views, [data, mdata])
            helpy.save_meta(args.prefix, meta)
            np.savez_compressed(args.prefix+'_TRACKS', data=data)
            np.savez_compressed(args.prefix+'_MELT', data=mdata)
            sys.exit()
    else:
        helpy.save_log_entry(args.prefix, 'argv')
        meta = helpy.load_meta(args.prefix)

    helpy.sync_args_meta(
        args, meta,
        ['side', 'fps', 'start', 'end', 'width', 'zoom'],
        ['sidelength', 'fps', 'start_frame', 'end_frame',
         'crystal_width', 'crystal_zoom'],
        [1, 1, 0, None, None, 1])

    M = 4  # number of neighbors

    W = args.width
    if W is None:
        data = helpy.load_data(args.prefix)
        N, W = square_size(helpy.mode(data['f'][data['t'] >= 0], count=True))
        meta['crystal_width'] = W
    N = W*W
    nshells = (W+1)//2
    maxshell = W//2
    print args.prefix
    print "Crystal size {W}x{W} = {N} ({s} shells)".format(W=W, N=N, s=nshells)

    if args.save:
        helpy.save_meta(args.prefix, meta)

    data = helpy.load_data(args.prefix)
    tsets = helpy.load_tracksets(data, run_repair='interp',
                                 run_track_orient=True)
    # to get the benefits of tracksets (interpolation, stub filtering):
    data = np.concatenate(tsets.values())
    data.sort(order=['f', 't'])

    if not args.start:
        args.start = find_start_frame(data, plot=args.plot)
    if args.melt:
        print 'calculating'
        mdata, frames, mframes = melt_analysis(data)
        if args.save:
            np.savez_compressed(args.prefix + '_MELT.npz', data=mdata)
            helpy.save_meta(args.prefix, meta, start_frame=args.start)
    else:
        mdata = np.load(args.prefix + '_MELT.npz')['data']
        frames, mframes = helpy.load_framesets((data, mdata), ret_dict=False)

    if args.plot:
        print "plotting"
        plot_args = make_plot_args(maxshell, args)
        stats = ['rad', 'dens', 'psi', 'phi']

        endex = args.end and np.searchsorted(mdata['f'], args.end)
        shells = split_shells(mdata[:endex], zero_to=1, maxshell=maxshell)
        shell_means = average_shells(shells, stats, 'f')

        if args.save:
            axes = it.repeat(None, len(stats))
            save_name = '{prefix}_{stat}.pdf'
        else:
            nrows = len(stats)
            figsize = list(plt.rcParams['figure.figsize'])
            figsize[1] = min(figsize[1]*nrows,
                             helplt.rc('figure.maxheight'))
            fig, axes = plt.subplots(figsize=figsize,
                                     nrows=nrows, sharex='col')
            save_name = ''
        for stat, ax in zip(stats, axes):
            save = save_name.format(prefix=args.prefix, stat=stat)
            ax = plot_by_shell(shell_means, 'f', stat, start=args.start,
                               zoom=args.zoom, smooth=args.smooth, save=save,
                               ax=ax, **plot_args)

        print " * shells vs density"
        shells.pop(nshells)
        stats.remove('dens')
        nrows = len(stats)
        figsize = list(plt.rcParams['figure.figsize'])
        figsize[1] = min(figsize[1]*nrows,
                         helplt.rc('figure.maxheight'))
        fig, axes = plt.subplots(figsize=figsize,
                                 nrows=nrows, sharex='col')
        for stat, ax in zip(stats, axes):
            plot_by_shell(shells, 'dens', stat, start=args.start,
                          ax=ax, **plot_args)

        print " * parametric histogram"
        stats = ['rad', 'dens', 'psi', 'phi']
        nrows = ncols = len(stats) - 1
        figsize = list(plt.rcParams['figure.figsize'])
        figsize[0] = min(figsize[0]*ncols,
                         helplt.rc('figure.maxwidth'))
        figsize[1] = min(figsize[1]*nrows,
                         helplt.rc('figure.maxheight'))
        f0 = args.start
        fs = np.array([0, 250, 500, 750, 1000, 1250])
        fbs = ['start'] + fs.tolist() + ['end']
        periods = np.split(mframes, fs + f0)
        for p, period in enumerate(periods):
            period = np.concatenate(period)
            fig, axes = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols,
                                     sharex='col', sharey='row')
            fig.suptitle(r'$t \, f = [{}, {})$'.format(fbs[p], fbs[p+1]))
            for j, x in enumerate(stats[:-1]):
                for i, y in enumerate(stats[j+1:]):
                    ax = axes[i+j, j]
                    plot_parametric_hist(period, x, y, ax=ax, **plot_args)

        print " * parameter maps"
        stats = ['sh', 'rad', 'dens', 'psi', 'phi']
        f0 = args.start
        fs = np.array([0, 250, 500, 750, 1000, 1250]) + f0
        nrows = len(stats)
        ncols = len(fs)
        figsize = list(plt.rcParams['figure.figsize'])
        figsize[0] = min(figsize[0]*ncols,
                         helplt.rc('figure.maxwidth'))
        figsize[1] = min(figsize[1]*nrows, 10)  # helplt.rc('figure.maxheight'))
        fig, axes = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols,
                                 sharex='all', sharey='all')
        norm_opts = plot_args.pop('norm')
        for i, stat in enumerate(stats):
            norm = norm_opts[stat]
            if norm is None:
                norm = (0, 1)
            if isinstance(norm, list):
                norm = tuple(np.percentile(mdata[stat]*plot_args['unit'][stat],
                                           norm))
            if isinstance(norm, tuple):
                norm = matplotlib.colors.Normalize(*norm)
            for j, f in enumerate(fs):
                ax = axes[i, j]
                plot_regions(frames[f], mframes[f], norm=norm,
                             ax=ax, colors=stat, **plot_args)
                ax.set_xticks([])
                ax.set_yticks([])
                if i == nrows-1:
                    ax.set_xlabel('time = {}'.format((f - f0)/args.fps))
                if j == 0:
                    ax.set_ylabel(stat)
                    xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        if args.show:
            plt.show()

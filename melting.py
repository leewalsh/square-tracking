#!/usr/bin/env python
from __future__ import division

import math
import itertools as it
import glob

import numpy as np
from scipy.spatial import Voronoi, Delaunay, cKDTree as KDTree
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

import helpy
import correlation as corr


def initialize_mdata(data):
    """melting data array to hold calculated statistics
    """

    # 'id', 'f', 't' hold the same data copied from original tracks data
    keep_fields = ['id', 'f', 't']
    melt_dtype = [(name, data.dtype[name]) for name in keep_fields]

    # new fields to hold:
    # shell, radius (from initial c.o.m.), local density, local psi, local phi
    melt_dtype.extend(zip(['sh', 'r', 'dens', 'psi', 'phi'],
                          ['i4', 'f4', 'f4', 'f4', 'f4']))

    mdata = np.empty(data.shape, melt_dtype)
    for keep in keep_fields:
        mdata[keep][:] = data[keep]
    return mdata


def melting_stats(frame, dens_method, neigh_args):
    xy = frame['xy']
    orient = frame['o']

    vor = Voronoi(xy)
    tess = Delaunay(xy)
    tree = KDTree(xy)
    neighborhoods = corr.neighborhoods(xy, tess=tess, tree=tree, **neigh_args)

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

    return dens, psi, phi


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

    positions = helpy.load_trackstack(data, length=last)['xy']
    displacements = helpy.dist(positions, positions[0])
    distances = helpy.dist(np.gradient(positions, axis=0))
    ds = displacements.mean(1) * distances.mean(1)
    dm = np.minimum.accumulate(ds[::-1])[::-1] == np.maximum.accumulate(ds)
    dmi = np.nonzero(dm[first:last])[0] + first
    start = dmi[0] - 1

    if plot:
        fig, ax = plt.subplots()
        f = np.arange(len(ds))/args.fps
        ax.plot(f, ds, '-')
        ax.plot(f[dmi], ds[dmi], '*')

    return start


def find_ref_basis(positions=None, psi=None):
    neighs, mask, dists = corr.neighborhoods(positions, size=2)
    pair_angles = corr.pair_angles(positions, neighs, mask, 'absolute')
    psi, ang = corr.pair_angle_op(*pair_angles, m=4, globl=True)
    print 'psi first frame:', psi
    if psi < 0.8:
        print 'RuntimeWarning: ref_basis based on weak psi =', psi
    cos, sin = np.cos(ang), np.sin(ang)
    basis = np.array([[cos, sin], [-sin, cos]])
    return ang, basis


def square_size(num):
    """given number of particles, return the perfect square and its width
    """
    width = int(round(math.sqrt(num)))
    num = width*width
    return num, width


def assign_shell(positions, ids=None, N=None, maxt=None, ref_basis=None):
    """given (N, 2) positions array, assign shell number to each particle

    shell number is assigned as maximum coordinate, written in a basis aligned
    with the global phase from the global bond-angle order parameter, with its
    origin at the center of mass, with unit length the average nearest-neighbor

    if W = sqrt(N) is even, smallest value is 0.5; if even, smallest value is 0.
    largest value is (W - 1)/2
    """
    N, W = square_size(N or len(positions))
    assert W % 2, "Michael's code requires integer shells"
    if ref_basis is None:
        _, ref_basis = find_ref_basis(positions)
    positions = corr.rotate2d(positions, basis=ref_basis)
    positions -= positions.mean(0)
    spacing = (positions.max(0) - positions.min(0)) / (W - 1)
    positions /= spacing
    shells = np.abs(positions).max(1).round().astype(int)
    if ids is not None:
        ni, mi = len(ids), max(ids.max(), maxt, N)
        if ni <= mi or np.any(ids != np.arange(ni)):
            shells_by_id = np.full(1+mi, -1, 'i4')
            shells_by_id[ids] = shells
            return shells_by_id
    return shells


def split_shells(mdata, zero_to=0, do_mean=True):
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
        shells[nshells] = mdata[mdata['sh'] >= 0]
    return shells


def average_shells(shells, fields=None, by='f'):
    """average all particles within each shell"""
    if fields is None:
        fields = [f for f in shells.dtype.names if f not in 'id f t sh']
    averaged = {}
    for s, shell in shells.iteritems():
        vals = tuple(shell[field] for field in fields)
        vals, bins = corr.bin_average(shell[by], vals, 1)
        averaged[s] = dict(zip(fields, vals))
        averaged[s][by] = bins[:-1]
    return averaged


def make_plot_args(nshells, args):
    line_props = helpy.transpose_dict_of_lists({
        'label': ['center', 'inner'] + range(2, nshells-1) + ['outer', 'all'],
        'c':    map(plt.get_cmap('Dark2'), xrange(nshells)) + ['black'],
        'lw':   [1]*nshells + [2],
    })
    xylabel = {
        'f':    r'$t \, f$',
        'dens': r'density $\langle r_{ij}\rangle^{-2}$',
        'psi':  r'bond angle order $\Psi$',
        'phi':  r'molecular angle order $\Phi$',
    }
    unit = {
        'f':    1/args.fps,
        'dens': args.side**2,
        'phi':  1,
        'psi':  1,
    }
    return dict(line_props=line_props, xylabel=xylabel, unit=unit)


def plot_by_shell(shells, x, y, start=0, smooth=0, zoom=1, **plot_args):
    line_props = plot_args.get('line_props', [{}]*len(shells))
    unit = plot_args.get('unit', {x: 1, y: 1})
    save = plot_args.get('save')
    ax = plot_args.get('ax')
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3) if save else None)
    else:
        fig = ax.get_figure()

    for s, shell in shells.iteritems():
        if s < 0:
            continue
        if x == 'f':
            ys = shell[y]
            xs = shell[x] - start
            if smooth:
                ys = gaussian_filter1d(ys, smooth, mode='nearest', truncate=2)
            ax.plot(xs*unit[x], ys*unit[y], **line_props[s])
            xlim = ax.get_xlim()
            ax.set_xlim(-25, xlim[1]*zoom)
            ax.set_ylim(0, 1.1)
        elif x in ('dens', 'phi', 'psi'):
            xs, ys = shell[x], shell[y]
            ax.scatter(xs*unit[x], ys*unit[y], s=0.01, **line_props[s])
        else:
            raise ValueError("Unknown x-value `{!r}`.".format(x))

    ax.legend(fontsize='small')
    ax.set_xlabel(plot_args['xylabel'][x])
    ax.set_ylabel(plot_args['xylabel'][y])

    fig.tight_layout()
    if save:
        fig.savefig(save)

    return fig, ax


def plot_by_config(prefix_pattern, smooth=1, side=1, fps=1):
    configs = ['inward', 'aligned', 'random', 'outward']
    stats = ['phi', 'psi', 'dens']
    stats = {stat: {config: np.load(prefix_pattern.format(config, stat)+'.npy')
                    for config in configs}
             for stat in stats}
    xlims = {'dens': 300, 'phi': 200, 'psi': 150}
    plt.rc('text', usetex=True)
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
        statlabels = {
            'dens': r'$\mathrm{density}\ \langle r_{ij}\rangle^{-2}$',
            'psi': r'$\mathrm{bond\ angle\ order}\ \Psi$',
            'phi': r'$\mathrm{molecular\ angle\ order}\ \Phi$'}
        ax.set_ylabel(statlabels[stat])
        ax.legend(loc='lower left', fontsize='small')
        fig.savefig(prefix_pattern.format('ALL', stat) + '.pdf')


def melt_analysis(data):
    mdata = initialize_mdata(data)

    frames, mframes = helpy.splitter((data, mdata), 'f')
    shells = assign_shell(frames[0]['xy'], frames[0]['t'],
                          maxt=data['t'].max())
    mdata['sh'] = shells[mdata['t']]

    dens_method = 'dist'
    # Calculate radial speed (not MSD!) (maybe?)
    for frame, melt in it.izip(frames, mframes):
        nn = np.where(melt['sh'] == nshells-1, 3, 4)
        neigh_args = {'size': (nn,)*2}

        dens, psi, phi = melting_stats(frame, dens_method, neigh_args)
        melt['dens'] = dens
        melt['psi'] = psi
        melt['phi'] = phi
    return mdata


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

    if '*' in args.prefix or '?' in args.prefix:
        prefix_pattern = args.prefix
        args.prefix = helpy.replace_all(args.prefix, '*?', '') + '_MRG'
        helpy.save_log_entry(args.prefix, 'argv')
        prefixes = [p[:-9] for p in glob.iglob(
            helpy.with_suffix(prefix_pattern, '_MELT.npz'))]
        metas, mdatas = zip(*[(helpy.load_meta(prefix),
                               helpy.load_data(prefix, 'm'))
                              for prefix in prefixes])
        for meta, mdata in zip(metas, mdatas):
            mdata['f'] = mdata['f'].astype(int) - int(meta['start_frame'])
        mdata = np.concatenate(mdatas)
        meta = helpy.merge_meta(metas, excl={'start_frame'},
                                excl_start=('center', 'corner'))
        if args.save:
            np.savez_compressed(args.prefix+'_MELT', data=mdata)
            helpy.save_meta(args.prefix, meta, merged=prefixes)
            print 'merged sets', prefixes, 'saved to', args.prefix
    else:
        helpy.save_log_entry(args.prefix, 'argv')
        meta = helpy.load_meta(args.prefix)

    helpy.sync_args_meta(
        args, meta,
        ['side', 'fps', 'start', 'width', 'zoom'],
        ['sidelength', 'fps', 'start_frame', 'crystal_width', 'crystal_zoom'],
        [1, 1, 0, None, 1])

    M = 4  # number of neighbors

    W = args.width
    if W is None:
        data = helpy.load_data(args.prefix)
        N, W = square_size(helpy.mode(data['f'][data['t'] >= 0], count=True))
        meta['crystal_width'] = W
    N = W*W
    nshells = (W+1)//2
    print args.prefix
    print "Crystal size {W}x{W} = {N} ({s} shells)".format(W=W, N=N, s=nshells)

    if args.save:
        helpy.save_meta(args.prefix, meta)

    if args.melt:
        print 'calculating'
        data = helpy.load_data(args.prefix)
        tsets = helpy.load_tracksets(data, run_repair='interp',
                                     run_track_orient=True)
        # to get the benefits of tracksets (interpolation, stub filtering):
        data = np.concatenate(tsets.values())
        data.sort(order=['f', 't'])
        if not args.start:
            args.start = find_start_frame(data, plot=args.plot)
        mdata = melt_analysis(data)
        if args.save:
            np.savez_compressed(args.prefix + '_MELT.npz', data=mdata)
            helpy.save_meta(args.prefix, meta, start_frame=args.start)
    else:
        mdata = np.load(args.prefix + '_MELT.npz')['data']

    if args.plot:
        plot_args = make_plot_args(nshells, args)
        stats = ['dens', 'psi', 'phi']

        shells = split_shells(mdata, zero_to=1)
        shells = average_shells(shells, stats, 'f')

        if args.save:
            axes = it.repeat(None, len(stats))
            save_name = '{prefix}_{stat}.pdf'
        else:
            nrows = len(stats)
            fig, axes = plt.subplots(figsize=(6.4, 4.8*nrows),
                                     nrows=nrows, sharex='col')
            save_name = ''
        for stat, ax in zip(stats, axes):
            save = save_name.format(prefix=args.prefix, stat=stat)
            plot_by_shell(shells, 'f', stat, start=args.start, zoom=args.zoom,
                          smooth=args.smooth, save=save, ax=ax, **plot_args)

        shells.pop(nshells)
        stats = ['psi', 'phi']
        nrows = len(stats)
        fig, axes = plt.subplots(figsize=(6.4, 4.8*nrows),
                                 nrows=nrows, sharex='col')
        for stat, ax in zip(stats, axes):
            plot_by_shell(shells, 'dens', stat, start=args.start,
                          ax=ax, **plot_args)
        if args.show:
            plt.show()

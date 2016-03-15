#!/usr/bin/env python
from __future__ import division

import sys
import math
import itertools as it

import numpy as np
from scipy.spatial import Voronoi, Delaunay, cKDTree as KDTree
import matplotlib.pyplot as plt

import helpy
import correlation as corr

'''
This program inputs a positions file (probably produced by
tracks_to_pos.py) and creates a .npz file containing analysis
of the particle positions with the following quantities:
* Psi
* Densities
* "Radial" psi (organized by shell)
* Radial densities
* Radial r (r being the distance from the crystal's center of mass)
* Radial speed (per frame)
* Radial MSD (with a fixed tau=5 frames).

Example usage:
analysis.py 12x12_random
'''


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


def plot_against_shell(mdata, field, zero_to=0, ax=None, side=1, fps=1):
    units = side*side if field == 'dens' else 1
    if ax is None:
        fig, ax = plt.subplots()
    smax = mdata['sh'].max()
    splindex = np.where(mdata['sh'], mdata['sh'], zero_to)
    for s, shell in helpy.splitter(mdata, splindex, noncontiguous=True):
        col = plt.cm.jet(s/smax)
        mean_by_frame = corr.bin_average(shell['f'], shell[field]*units, 1)
        ax.plot(np.arange(len(mean_by_frame))/fps, mean_by_frame,
                label='Shell {}'.format(s), c=col)
    ax.legend()
    ax.set_title(field)
    ax.set_xlim(0, 500)


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
    if len(sys.argv) < 2:
        print "Please specify a filename."
        sys.exit(2)
    fname = sys.argv[1]
    try:
        sys.argv.remove('-v')
        verbose = True
    except ValueError:
        from warnings import filterwarnings
        filterwarnings('ignore', category=RuntimeWarning,
                       module='numpy|scipy|matplot')
        verbose = False

    helpy.save_log_entry(fname, 'argv')
    meta = helpy.load_meta(fname)
    fps = meta.get('fps', 1)
    side = meta.get('sidelength', 1)

    M = 4  # number of neighbors
    data = helpy.load_data(fname)

    if len(sys.argv) > 2:
        W = int(sys.argv[2])
    else:
        W = meta.get('crystal_size')
    if W is None:
        N, W = square_size(helpy.mode(data['f'][data['t'] >= 0], count=True))
    N = W*W
    meta['crystal_size'] = N
    nshells = (W+1)//2
    print "Crystal size {W}x{W} = {N} ({s} shells)".format(W=W, N=N, s=nshells)

    tsets = helpy.load_tracksets(data, min_length=-N, run_fill_gaps='interp',
                                 run_track_orient=True)
    # to get the benefits of tracksets (interpolation, stub filtering):
    data = np.concatenate(tsets.values())
    data.sort(order=['f', 't'])

    mdata = melt_analysis(data)

    helpy.save_meta(fname, meta)
    np.savez_compressed(fname + '_MELT.npz', data=mdata)

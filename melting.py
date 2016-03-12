#!/usr/bin/env python
from __future__ import division

import sys
import math
import itertools as it

import numpy as np
from scipy.spatial import Voronoi, Delaunay, ConvexHull, cKDTree as KDTree

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
                          ['u4', 'f4', 'f4', 'c8', 'c8']))

    mdata = np.empty(data.shape, melt_dtype)
    for keep in keep_fields:
        mdata[keep][:] = data[keep]
    return mdata


def melting_stats(frames):
    xy = frame['xy']
    orient = frame['o']

    vor = Voronoi(xy)
    tess = Delaunay(xy)
    tree = KDTree(xy)
    neigh_def = {'size': (1, 4), 'voronoi': True, 'reach': 'min*1.42'}
    neighborhoods = corr.neighborhoods(xy, tess=tess, tree=tree, **neigh_def)

    # Density:
    dens = corr.density(xy, 'vor', vor=vor, tess=tess, neighbors=neighborhoods)

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


def assign_shell(positions, ids=None, N=None, ref_basis=None):
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
        ni, mi = len(ids), ids.max() + 1
        if ni < mi or np.any(ids != np.arange(ni)):
            shells_by_id = np.full(mi, -1, 'u4')
            shells_by_id[ids] = shells
            return shells_by_id
    return shells


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "Please specify a filename."
        sys.exit(2)
    fname = sys.argv[1]
    try:
        sys.argv.remove('-v')
        verbose = True
    except ValueError:
        verbose = False

    helpy.save_log_entry(fname, 'argv')
    meta = helpy.load_meta(fname)

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

    mdata = initialize_mdata(data)

    frames, mframes = helpy.splitter((data, mdata), 'f')
    shells = assign_shell(frames[0]['xy'], frames[0]['t'])
    mdata['sh'] = shells[mdata['t']]

    # Calculate radial speed (not MSD!) (maybe?)
    for frame, melt in it.izip(frames, mframes):
        dens, psi, phi = melting_stats(frame)
        melt['dens'] = dens
        melt['psi'] = psi
        melt['phi'] = phi


    helpy.save_meta(fname)
    np.savez_compressed(fname + '_MELT.npz', data=mdata)

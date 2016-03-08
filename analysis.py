#!/usr/bin/env python
from __future__ import division

import sys
import numpy as np
from scipy.spatial import Voronoi, cKDTree as KDTree
import math
import cmath

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


def get_angle(p, q):
    # angle of (q - p) from the horizontal
    r = (q[0] - p[0], q[1] - p[1])
    return math.atan2(r[1], r[0])


def poly_area(corners):
    # calculate area of polygon
    area = 0.0
    n = len(corners)
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    return abs(area) / 2.0


def calc_MSD(tau, squared):
    ret = []
    for t, frame in enumerate(frames[:-tau]):
        disps = [[] for i in xrange(nshells)]
        other_frame = frames[t + tau]
        # range over all particles in frames t, t+tau
        for i, p in enumerate(frames[t]):
            for j, q in enumerate(other_frame):
                if frame_IDs[t + tau][j] == frame_IDs[t][i]:
                    break
            else:
                if verbose:
                    print("ID {0} not found in frame {1}!".format(
                          frame_IDs[t][i], t+tau))
                continue
            # found corresponding particle q in other frame
            shell = shells[frame_IDs[t][i]]
            x = (p[0]-q[0])**2 + (p[1]-q[1])**2
            disps[shell].append(x if squared else math.sqrt(x))
        ret.append(disps)
    return ret


def take_avg(stat):
    return [[np.mean(s) if len(s) else -1 for s in f] for f in stat]


def find_ref_basis(positions=None, psi=None):
    side = 17.61
    maxdist = side*math.sqrt(2)
    pair_angles = corr.pair_angles(positions, 4, 'absolute', dub=maxdist)
    psi, ang, psims = corr.pair_angle_op(*pair_angles, m=4)
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
    if len(sys.argv) > 2:
        W = sys.argv[2]
        N = W*W
    else:
        W = N = None
    verbose = '-v' in sys.argv
    helpy.save_log_entry(fname, 'argv')

    M = 4  # number of neighbors

    data = helpy.load_data(fname)
    if N is None:
        N, W = square_size(helpy.mode(data['f'][data['t'] >= 0], count=True))
    nshells = (W+1)//2

    tracks = helpy.load_tracksets(data, min_length=-N, run_fill_gaps='interp',
                                  run_track_orient=True)
    data = data[np.in1d(data['t'], tracks.keys())]
    frames = helpy.splitter(data[['x', 'y']].view(('f4', (2,))), data['f'])
    frame_IDs = helpy.splitter(data['t'], data['f'])

    shells = assign_shell(frames[0], frame_IDs[0])
    initial_pos = dict(zip(frame_IDs[0], frames[0]))

    psi_data = []           # (frame, particle)
    frame_densities = []    # (frame, particle)
    radial_psi = []         # (frame, shell, particle)
    radial_densities = []   # (frame, shell, particle)
    radial_r = []           # (frame, shell, particle)

    for j, frame in enumerate(frames):
        fshells = shells[frame_IDs[j]]
        shell_ind = [np.where(fshells == s) for s in xrange(nshells)]

        vor = Voronoi(frame)
        areas = []                                      # (particle,)
        r_densities = [[] for i in xrange(nshells)]     # (shell, particle)
        r_psi = [[] for i in xrange(nshells)]           # (shell, particle)
        r_r = helpy.dist(frame, frame.mean(0))          # (particle,)
        radial_r.append([r_r[si] for si in shell_ind])  # (shell, particle)

        for i in xrange(len(vor.points)):
            region = vor.regions[vor.point_region[i]]
            if region and -1 not in region:
                # finite Voronoi cell
                area = poly_area(vor.vertices[region])
                if area > 0:
                    areas.append(area)
                    r_densities[fshells[i]].append(1/area)

        frame_densities.append(1/np.array(areas))
        radial_densities.append(r_densities)


        psi_loop = True
        if psi_loop:
            tree = KDTree(frame)
            distances, neighbors = tree.query(frame, k=M+1)
            distances = distances[:, 1:]  # nearest particle is self
            neighbors = neighbors[:, 1:]
            psi_frame = []
            for i, p in enumerate(frame):
                ns = neighbors[i]
                # if p is an edge or corner, remove diagonal neighbors
                thresh = 0.9*math.sqrt(2)*distances[i].min()
                ns = ns[distances[i] < thresh]
                if len(ns) > 1:
                    # if 1 neighbor, |psi| will trivially be 1
                    psi = np.mean([cmath.exp(M*get_angle(p, frame[n])*1j)
                                   for n in ns])
                    psi = abs(psi)
                    psi_frame.append(psi)
                    r_psi[fshells[i]].append(psi)

            psi_data.append(psi_frame)
        else:
            angles, nmask = corr.pair_angles(frame, M)
            psi, _, psis = corr.pair_angle_op(angles, nmask, M)
            psi_data.append(psis)
            r_psi = [psis[si] for si in shell_ind]
        radial_psi.append(r_psi)


    # Calculate radial speed (not MSD!)
    radial_speed = calc_MSD(tau=1, squared=False)   # (frame, shell, particle)
    radial_MSD = calc_MSD(tau=5, squared=True)      # (frame, shell, particle)

    # psi_data          (frame, particle)
    # frame_densities   (frame, particle)
    # radial_psi        (frame, shell, particle)
    # radial_densities  (frame, shell, particle)
    # radial_r          (frame, shell, particle)

    max_density = max([max(densities) for densities in frame_densities])
    frame_densities = [densities / (max_density*6.5**2)
                       for densities in frame_densities]

    # take averages
    radial_psi = take_avg(radial_psi)
    square_area = 6.5**2
    radial_densities = [[y*square_area for y in x]
                        for x in take_avg(radial_densities)]
    radial_r = take_avg(radial_r)
    radial_speed = take_avg(radial_speed)
    radial_MSD = take_avg(radial_MSD)

    m = max([max(x) for x in radial_densities])
    radial_densities /= m
    np.savez(fname + "_DATA.npz", psi=psi_data, densities=frame_densities,
             radial_psi=radial_psi, radial_densities=radial_densities,
             radial_r=radial_r, radial_speed=radial_speed,
             radial_msd=radial_MSD)

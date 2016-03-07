#!/usr/bin/env python
from __future__ import division

import sys
import numpy as np
from scipy.spatial import KDTree, Voronoi
import math, cmath

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
        disps = []
        for v in range(nshells + 1): # contains all s=k
            disps.append([])
        other_frame = frames[t + tau]
        # range over all particles in frames t, t+tau
        for i, p in enumerate(frames[t]):
            for j, q in enumerate(other_frame):
                if frame_IDs[t + tau][j] == frame_IDs[t][i]:
                    break
            else:
                print("ID {0} not found in frame {1}!".format(
                    frame_IDs[t][i],t+tau))
                continue
            # found corresponding particle q in other frame
            shell = shells[frame_IDs[t][i]]
            x = (p[0]-q[0])**2 + (p[1]-q[1])**2
            disps[shell].append(x if squared else math.sqrt(x))
        ret.append(disps)
    return ret

def take_avg(stat, ignore_first):
    if ignore_first:
        for frame in stat:
            frame[2].extend(frame[1])
            frame[1] = []
    ret = [[sum(l)/len(l) if len(l)>0 else -1.
            for l in frame[1:]] for frame in stat]
    return [x[1:] for x in ret] if ignore_first else ret


def find_ref_basis(positions=None, psi=None):
    side = 17.61
    pair_angles = corr.pair_angles(positions, 4, 'absolute', dub=side*math.sqrt(2))
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


def assign_shell(positions, N=None, ref_basis=None):
    """given (N, 2) positions array, assign shell number to each particle

    shell number is assigned as maximum coordinate, written in a basis aligned
    with the global phase from the global bond-angle order parameter, with its
    origin at the center of mass, with unit length the average nearest-neighbor

    if W = sqrt(N) is even, smallest value is 0.5; if even, smallest value is 0.
    largest value is (W - 1)/2
    """
    N, W = square_size(N or len(positions))
    if ref_basis is None:
        _, ref_basis = find_ref_basis(positions)
    positions = corr.rotate2d(positions, basis=ref_basis)
    positions -= positions.mean(0)
    spacing = (positions.max(0) - positions.min(0)) / (W - 1)
    positions /= spacing
    shells = np.abs(2*positions).max(1).round()/2
    return shells


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print "Please specify a filename."
        sys.exit(2)
    fname = sys.argv[1]
    helpy.save_log_entry(fname, 'argv')

    M = 4  # number of neighbors

    data = helpy.load_data(fname)
    frames = helpy.splitter(data[['x', 'y']].view(('f4', (2,))), data['f'])
    frame_IDs = helpy.splitter(data['t'], data['f'])

    shells = assign_shell(frames[0])
    nshells = shells.max()

    psi_data = []
    frame_densities = []
    MSDs = []
    radial_psi = []
    radial_densities = []
    radial_r = []

    # Find initial positions for each ID
    initial_pos = dict(zip(frame_IDs[0], frames[0]))

    for j, frame in enumerate(frames):
        vor = Voronoi(frame)
        areas = []
        r_densities = []
        r_psi = []
        r_r = []

        for v in range(nshells + 1):
            # each list contains all s=k
            r_densities.append([])
            r_psi.append([])
            r_r.append([])

        for i, p in enumerate(vor.points):
            region = vor.regions[vor.point_region[i]]
            if -1 in region:
                # infinite Voronoi cell
                continue
            areas.append(poly_area([vor.vertices[q] for q in region]))
            if areas[-1] > 0.:
                r_densities[shells[frame_IDs[j][i]]].append(1. / areas[-1])

        areas = np.asarray(areas)
        densities = 1. / areas[areas > 0.]
        frame_densities.append(densities)
        radial_densities.append(r_densities)

        tree = KDTree(frame)
        COM = frame.mean(0)
        psi_frame = []
        for i, p in enumerate(frame):
            query_ret = tree.query([p], k=M+1)

            # remove p from the neighbors list by slicing
            neighbors = [tree.data[x] for x in query_ret[1][0]][1:]
            # if p is an edge or corner, remove extra neighbors
            min_dist = min((n[0]-p[0])**2 + (n[1]-p[1])**2 for n in neighbors)
            thresh = min_dist * 2 * .9 # slightly less than a diagonal
            neighbors = [n for n in neighbors
                         if (n[0]-p[0])**2 + (n[1]-p[1])**2 < thresh]
            N = len(neighbors)
            psi = sum(cmath.exp(M*get_angle(p, n) * 1j) for n in neighbors) / N
            psi = abs(psi)
            shell = shells[frame_IDs[j][i]]
            if N > 1: # if N=1, |psi| will trivially be 1
                psi_frame.append(psi)
                r_psi[shell].append(psi)
            r = math.sqrt((COM[0]-p[0])**2 + (COM[1]-p[1])**2)
            r_r[shell].append(r)
            p_0 = initial_pos[frame_IDs[j][i]]
            squared_disp = (p[0]-p_0[0])**2 + (p[1]-p_0[1])**2

        psi_data.append(psi_frame)
        radial_psi.append(r_psi)
        radial_r.append(r_r)


    # Calculate radial speed (not MSD!)
    radial_speed = calc_MSD(1, False)
    radial_MSD = calc_MSD(5, True)

    max_density = max([max(densities) for densities in frame_densities])
    frame_densities = [densities / (max_density*6.5**2)
                       for densities in frame_densities]

    #take averages
    radial_psi = take_avg(radial_psi, True)
    radial_densities = [[y*6.5**2 for y in x]
                        for x in take_avg(radial_densities, True)]
    radial_r = take_avg(radial_r, True)
    radial_speed = take_avg(radial_speed, True)
    radial_MSD = take_avg(radial_MSD, True)

    m = max([max(x) for x in radial_densities])
    radial_densities /= m
    np.savez(fname + "_DATA.npz", psi=psi_data, densities=frame_densities,
             radial_psi=radial_psi, radial_densities=radial_densities,
             radial_r=radial_r, radial_speed=radial_speed,
             radial_msd=radial_MSD)

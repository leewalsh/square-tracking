
import numpy as np
from numpy.linalg import norm


def count_in_ring(positions,center,r,dr=1):
    """ count_in_ring(positions,center,r,dr)
        return number of particles in a ring of
            centered at center,
            radius r,
            thickness dr (defaults to 1.0),
        normalized by the area of the ring
    """

    count = 0
    for position in positions:
        if r - dr/2. < norm(position-center) < r + dr/2.
            count += 1
        else continue

    ring_area = 2 * np.pi * r * dr
    return count / ring_area



def pair_corr(positions, dr=1, rmax=200):
    """ pair_corr(positions)

        the pair correlation function g(r)

        takes a list of positions of particles in one 2-d frame
        dr is step size in r for function g(r)
            (units are those implied in coords of positions)
    """

    rs = np.arange(dr,rmax,dr)
    g = np.zeros(shape(rs))
    dg = np.zeros(shape(rs))
    for r in rs:
        gr = [count_in_ring(positions,position,r) for position in positions]
        #gr = map( lambda x,y=r,p=positions: count_in_ring(p,x,y), positions )
        #gr = []
        #for position in positions:
        #    gr.append(count_in_ring(positions,position,r))
        g[r] = np.mean(gr)
        dg[r] = np.std(gr)


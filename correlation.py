#!/usr/bin/env python

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
        if r - dr < norm(np.array(position)-np.array(center)) < r + dr :
            count += 1
        else: continue

    ring_area = 2 * np.pi * r * dr
    return count / ring_area

def pair_corr(positions, dr=22, rmax=220):
    """ pair_corr(positions)

        the pair correlation function g(r)

        takes a list of positions of particles in one 2-d frame
        dr is step size in r for function g(r)
            (units are those implied in coords of positions)
    """
    rs = np.arange(dr,rmax,dr)
    g = np.zeros(np.shape(rs))
    dg = np.zeros(np.shape(rs))
    rg = np.zeros(np.shape(rs))
    for ir,r in enumerate(rs):
        # Three ways to do same thing: list comp, map/lambda, loop
        #gr = [count_in_ring(positions,position,r) for position in positions]
        #gr = map( lambda x,y=r,p=positions: count_in_ring(p,x,y), positions )
        gr = []
        for position in positions:
            if norm(np.array(position)-np.array((300,300))) < 300-rmax:
                gr.append(count_in_ring(positions,position,r))
        if np.array(gr).any():
            g[ir]  = np.mean(gr)
            dg[ir] = np.std(gr)
            rg[ir] = r
        else: print "none for r =",r
    return g,dg,rg

def pair_corr_hist(positions, dr=11,rmax=220):
    """ pair_corr_hist(positions):
        the pair correlation function g(r)
        calculated using a histogram of distances between particle pairs
    """
    distances = []
    for pos1 in positions:
        if norm(np.array(pos1)-np.array((300,300))) < 300-rmax:
            distances.append(
                    [norm(np.array(pos2) - np.array(pos1)) for pos2 in positions]
                    )
    distances = np.array(distances)
    distances = distances[np.nonzero(distances)]
    return np.histogram(distances
            , bins = 10*rmax/dr
            , weights = 1/(np.pi*np.array(distances)*dr)
            )

def get_positions(data,frame):
    return zip(data['x'][data['f']==frame],data['y'][data['f']==frame])

def avg_hists(gs,rgs):
    rg = rgs[0]
    g_avg = [ np.mean(gs[:,r]) for r in rg ]
    dg_avg = [ np.std(gs[:,r]) for r in rg ]
    return g_avg, dg_avg, rg

def build_gs(data,prefix,framestep=100):
    frames = np.arange(min(data['f']),max(data['f']),framestep)
    gs = [ np.zeros(int(prefix[1:])) for frame in frames ]
    gs = np.array(gs)
    rgs = np.array(gs)
    print "gs initiated with shape",np.shape(gs)
    print "frames type",type(frames)
    for nf,frame in enumerate(frames):
        print "\t appending for frame",frame
        positions = get_positions(data,frame)
        #g,dg,rg = pair_corr(positions)
        g,rg = pair_corr_hist(positions)
        rg = rg[1:]

        gs[nf,:len(g)]  = g
        rgs[nf,:len(g)] = rg

    return gs,rgs

if __name__ == '__main__':
    import matplotlib.pyplot as pl

    locdir = '/Users/leewalsh/Physics/Squares/spatial_diffusion/'  #rock
    prefix = 'n416'
    datapath = locdir+prefix+'_results.txt'
    
    print "loading data from",datapath
    data = np.genfromtxt(datapath,
            skip_header = 1,
            usecols = [0,2,3,5],
            names   = "id,x,y,f",
            dtype   = [int,float,float,int])
    data['id'] -= 1 # data from imagej is 1-indexed
    print "\t...loaded"
    print "loading positions"
    gs,rgs = build_gs(data,prefix)
    print "\t...gs,rgs built"
    print "averaging over all frames..."
    g,dg,rg = avg_hists(gs,rgs)
    print "\t...averaged"
    

    #pl.figure()
    pl.plot(np.array(rg)/22.,g,'.-',label=prefix)
    pl.title("g[r],%s,dr%d"%(prefix,22))
    pl.show()
